"""A vectorised RL environment that steps in Newton and reuses mjlab's managers.

Everything except the physics comes from mjlab: the observation terms, the action term, and the
reward / termination / event managers all construct against `NewtonEnv` unchanged (37 reward terms,
9 termination terms). That is deliberate -- the point of the port is to change the simulator, not to
re-derive an MDP, and a reward that differs silently is indistinguishable from a policy that learns
badly.

Physics is Newton's: `ModelBuilder.replicate` for parallel worlds, `SolverMuJoCo.step` at dt=0.005,
with the two spec repairs from `newton_simple_fix` (free-joint damping, compressed mass-matrix
layout) applied to Newton's own model.

The step order follows mjlab's `ManagerBasedRlEnv.step` exactly, including applying the action on
every physics substep rather than once per control step -- `Sonic53Action` re-applies the startup
hold, the table pose and the reference object tracking on each call, so calling it once per control
step runs all of that at a quarter rate.
"""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
import torch
import warp as wp


class NewtonVecEnv:
  """Duck-types enough of mjlab's ManagerBasedRlEnv for `RslRlVecEnvWrapper`."""

  def __init__(self, cfg, xml: str, num_envs: int, device: str = "cuda:0",
               njmax: int = 2048, nconmax: int = 256, object_entity: str = "apple",
               sdf_object_stl: str | None = None, sdf_resolution: int = 128,
               sdf_hydroelastic: bool = False, viser_port: int | None = None,
               render_every: int = 4, solver_kwargs: dict | None = None,
               cuda_graph: bool = False) -> None:
    import mujoco
    import newton
    from newton.solvers import SolverMuJoCo
    from newton_simple_fix import capture_spec, restore_simple_bodies, restore_freejoint_damping
    from newton_sensors import transplant_sensors
    from newton_bridge import NewtonEnv

    self.cfg = cfg
    self.device = device
    self.num_envs = int(num_envs)
    self.render_mode = None
    self.extras: dict[str, Any] = {}

    self.physics_dt = float(cfg.sim.mujoco.timestep) if hasattr(cfg.sim, "mujoco") else 0.005
    self.decimation = int(cfg.decimation)
    self.step_dt = self.physics_dt * self.decimation
    self.max_episode_length = int(math.ceil(float(cfg.episode_length_s) / self.step_dt))

    # --- Newton model: one authored scene, replicated into parallel worlds -----
    scene = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(scene)
    scene.default_shape_cfg.gap = 0.0          # Newton's default of 0.1 needs 10cm of penetration
    scene.add_mjcf(xml, collapse_fixed_joints=False, parse_mujoco_options=True)

    # Swap the object's collider before replicating, so every world gets it. mjlab authors the object
    # as a 4 cm sphere and its own mesh path uses the *_cir160 convex hull instead of the real shape.
    self._ref_mj_pre = mujoco.MjModel.from_xml_path(xml)
    if sdf_object_stl:
      from grab_objects import swap_collider_to_sdf
      swap_collider_to_sdf(scene, self._ref_mj_pre, f"{object_entity}/{object_entity}",
                           sdf_object_stl, resolution=sdf_resolution,
                           hydroelastic=sdf_hydroelastic)

    world = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(world)
    world.default_shape_cfg.gap = 0.0
    world.replicate(scene, world_count=self.num_envs)
    self.nmodel = world.finalize()

    # Newton drops <sensor> entirely, and 136 of this scene's sensors are the contact sensors the
    # grasp rewards gate on -- multi_tip_surface (weight 5.0), contact_duration (1.0) and
    # object_hard_lift (2.0) read exactly zero without them, for every step of every run. They are
    # added to Newton's own spec just before it compiles, so MuJoCo evaluates its own sensors.
    import mujoco as _mj
    _orig_compile = _mj.MjSpec.compile
    _xml_for_sensors = xml

    def _compile_with_sensors(spec_self, *a, **k):
      try:
        transplant_sensors(spec_self, _xml_for_sensors, verbose=True)
      except Exception as e:
        print(f"[sensors] transplant failed ({type(e).__name__}: {e}); "
              "contact-gated rewards will read zero")
      return _orig_compile(spec_self, *a, **k)

    _mj.MjSpec.compile = _compile_with_sensors
    try:
      with capture_spec() as cap:
      # update_data_interval=0 keeps mjw_data authoritative, so the direct joint/root writes the
      # action term performs during the startup hold are not overwritten from Newton's State.
        # SDF collision has its own solver parameters (sdf_iterations / sdf_initpoints) that
        # default to None; they are passed through here so they can be set for mesh colliders.
        _kw = dict(enable_multiccd=True, update_data_interval=0, njmax=njmax, nconmax=nconmax)
        _kw.update(solver_kwargs or {})
        self.solver = SolverMuJoCo(self.nmodel, **_kw)
    finally:
      _mj.MjSpec.compile = _orig_compile

    # Reported every run because its absence is silent: with nsensor=0 the contact-gated reward
    # terms return exactly 0.0 forever instead of erroring, which cost 3576 wasted iterations once.
    _ns = int(self.solver.mjw_model.nsensor)
    _nc = int((wp.to_torch(self.solver.mjw_model.sensor_type) == 42).sum()) if _ns else 0
    print(f"[newton-env] sensors: nsensor={_ns} contact={_nc} "
          f"nsensordata={int(self.solver.mjw_model.nsensordata)}"
          + ("" if _nc else "  <-- contact-gated rewards will read zero"))
    self._ref_mj = mujoco.MjModel.from_xml_path(xml)
    restore_freejoint_damping(cap.spec, xml, verbose=False)

    # The simple-body fix recompiles Newton's spec and installs put_model/put_data of the result.
    # That is right for an analytic-primitive object, and destructive for a mesh one: Newton attaches
    # the SDF volumes to the warp model when it builds the solver, so replacing that model wholesale
    # leaves the mesh collider with no distance field behind it. Measured: with the fix the first
    # solver step produced 150 NaNs in qpos; without it the same scene steps cleanly.
    #
    # It is also unnecessary here. The fix recovers MuJoCo's compressed mass-matrix layout for a body
    # that qualifies as "simple" -- a free body with isotropic, centred inertia. A real mesh is
    # neither, so nC=1102 is the correct answer for it rather than a defect to repair.
    if sdf_object_stl:
      print(f"[newton-env] simple-body fix skipped: the object is a mesh collider, whose SDF lives "
            f"on the warp model the fix would replace (nC={int(self.solver.mjw_model.nC)})")
    else:
      restore_simple_bodies(self.solver, cap.spec, nworld=self.num_envs,
                            nconmax=nconmax, njmax=njmax, verbose=False)
      nC, nC_ref = int(self.solver.mjw_model.nC), int(self._ref_mj.nC)
      if nC != nC_ref:
        raise RuntimeError(f"nC {nC} != {nC_ref} with a primitive object; the spec repair did not take")

    self.control = self.nmodel.control()
    self.state_in, self.state_out = self.nmodel.state(), self.nmodel.state()

    self._env = NewtonEnv(self.solver.mj_model, self.solver.mjw_data, self.num_envs, device,
                          control=self.control, rename_from=self._ref_mj,
                          physics_dt=self.physics_dt, decimation=self.decimation,
                          solver=self.solver, object_entity=object_entity)
    self._env.forward()
    # Termination terms read these off the env the managers were built with, not off this wrapper.
    self._env.max_episode_length = self.max_episode_length
    self._env.max_episode_length_s = float(cfg.episode_length_s)
    self._env.cfg = cfg
    self._env.common_step_counter = 0

    # --- mjlab's own managers, against Newton state ---------------------------
    from mjlab.tasks.apple_eat import mdp as amdp
    from mjlab.managers.reward_manager import RewardManager
    from mjlab.managers.termination_manager import TerminationManager
    from mjlab.managers.event_manager import EventManager

    sonic_cfg = (cfg.actions.get("sonic_action") if isinstance(cfg.actions, dict)
                 else getattr(cfg.actions, "sonic_action"))
    self.action_term = amdp.Sonic53Action(sonic_cfg, self._env)
    self.action_manager = self._env.bind_action_manager(
      self.action_term.action_dim, {"sonic_action": self.action_term})
    self.action_manager.total_action_dim = self.action_term.action_dim

    # Prefer mjlab's real ObservationManager: it owns history buffers, group concatenation and the
    # nan policy. Hand-assembling the groups reproduces none of that, and a history buffer that is
    # not maintained is the exact defect that cost this project a working grasp once already.
    self.observation_manager = None
    try:
      from mjlab.managers.observation_manager import ObservationManager
      self.observation_manager = ObservationManager(cfg.observations, self._env)
      print(f"[newton-env] using mjlab ObservationManager "
            f"({len(self.observation_manager.active_terms)} groups)")
    except Exception as e:
      # Fatal rather than a fallback. The per-group builders below are early-port scaffolding and
      # are not the code mjlab runs; silently switching to them would mean training on a different
      # observation than the one whose parity with mjlab was measured at 8.3e-07.
      raise RuntimeError(
        f"mjlab's ObservationManager failed to construct ({type(e).__name__}: {e}). "
        "Refusing to fall back to the per-group builders -- fix the cause instead.") from e

    obs_cfg = cfg.observations if isinstance(cfg.observations, dict) else vars(cfg.observations)
    self._obs_builders = {}
    if self.observation_manager is None:
      for gname, gcfg in obs_cfg.items():
        terms = getattr(gcfg, "terms", None) or (gcfg if isinstance(gcfg, dict) else None)
        if not terms:
          continue
        t = terms.get("policy") or next(iter(terms.values()))
        try:
          self._obs_builders[gname] = t.func(t, self._env)
        except Exception as e:
          print(f"[newton-env] observation builder {gname!r} unavailable "
                f"({type(e).__name__}: {e})")

    # mjlab's MetricsManager is what produces the task-level numbers -- lift_success, contact,
    # object tracking error. Without it the only things logged are PPO internals, and a run whose
    # physics has failed looks identical to one that is merely early: 3576 iterations were spent
    # that way before the NaN was found by hand.
    self.metrics_manager = None
    try:
      from mjlab.managers.metrics_manager import MetricsManager
      self.metrics_manager = MetricsManager(cfg.metrics, self._env)
      print(f"[newton-env] MetricsManager: "
            f"{len(getattr(self.metrics_manager, 'active_terms', []) or [])} terms")
    except Exception as e:
      # Fatal: the MetricsManager is what produces lift_success, sequence_success and
      # object_mpjpe_mm -- the numbers the whole port is judged on. A run without it looks healthy
      # and reports nothing true.
      raise RuntimeError(
        f"mjlab's MetricsManager failed to construct ({type(e).__name__}: {e})") from e

    # scale_by_dt must come from the config, not the constructor default. mjlab passes
    # cfg.scale_rewards_by_dt here (manager_based_rl_env.py:330) and this task sets it
    # False; taking the default of True multiplies every reward term by dt=0.02, making
    # the whole reward signal 50x smaller than mjlab's while every individual term still
    # measures identical -- which is why this survived a term-by-term parity check.
    self.reward_manager = RewardManager(cfg.rewards, self._env,
                                        scale_by_dt=cfg.scale_rewards_by_dt)
    self.termination_manager = TerminationManager(cfg.terminations, self._env)
    self.event_manager = EventManager(cfg.events, self._env)

    # CUDA graph capture of the physics substep. mujoco_warp issues hundreds of short kernels per
    # step, so at these env counts the GPU spends most of its time idle between launches -- 15-20%
    # utilisation at 2048 envs. A captured graph replays the whole launch sequence as one command.
    #
    # Two graphs, not one: the substep swaps state_in/state_out, and a captured graph bakes in the
    # buffer pointers it was recorded with. Decimation is even, so alternating A->B and B->A
    # returns the buffers to their original orientation at the end of each control step.
    self._use_cuda_graph = bool(cuda_graph)
    self._graphs: list = []
    self._graph_warmup = 4      # uncaptured substeps per parity before capturing
    self.episode_length_buf = self._env.episode_length_buf
    self._all = torch.arange(self.num_envs, dtype=torch.long, device=device)

    # --- optional live view ---------------------------------------------------
    self.viewer = None
    self._render_every = max(1, int(render_every))
    self._render_tick = 0
    if viser_port:
      import newton.viewer as _nv
      self.viewer = _nv.ViewerViser(port=int(viser_port),
                                    label=f"Newton training ({self.num_envs} envs)")
      # Newton draws visual geometry and hides colliders. The object here has only a collider -- the
      # SDF mesh -- so without this the robot appears to grasp nothing at all.
      _flags = wp.to_torch(self.nmodel.shape_flags)
      _sbody = wp.to_torch(self.nmodel.shape_body)
      _VIS = int(newton.ShapeFlags.VISIBLE)
      for _bid in torch.unique(_sbody):
        _idx = (_sbody == _bid).nonzero(as_tuple=True)[0]
        if int((_flags[_idx] & _VIS).sum()) == 0:
          _flags[_idx] |= _VIS
      self.viewer.set_model(self.nmodel)
      print(f"[newton-env] viser on http://localhost:{viser_port} "
            f"(rendering every {self._render_every} control steps)")
    # mjlab's event manager needs the global step count for reset-mode terms that fire on a schedule
    # (min_step_count_between_reset). It is a counter over env-steps, not control steps.
    self.common_step_counter = 0

  # ---------------------------------------------------------------- interface
  @property
  def unwrapped(self) -> "NewtonVecEnv":
    return self

  @property
  def scene(self):
    return self._env.scene

  @property
  def observation_space(self):
    return None

  @property
  def action_space(self):
    return None

  def get_observations(self):
    if self.observation_manager is not None:
      return self.observation_manager.compute()
    from tensordict import TensorDict
    return TensorDict({g: b(self._env) for g, b in self._obs_builders.items()},
                      batch_size=[self.num_envs])

  def reset(self, env_ids: torch.Tensor | None = None):
    ids = self._all if env_ids is None else env_ids
    self._reset_idx(ids)
    return self.get_observations(), self.extras

  def _reset_idx(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    self.event_manager.apply(mode="reset", env_ids=env_ids,
                             global_env_step_count=self.common_step_counter * self.num_envs)

    # Every manager's reset() has to run, in mjlab's order (it marks the sequence order-sensitive).
    # Skipping them is not merely a logging gap: reward_manager.reset zeroes the per-episode sums,
    # metrics_manager.reset clears episodic accumulators like lift_success, and action_manager.reset
    # drops the previous action the smoothness terms difference against -- so without these, state
    # leaks across episode boundaries for the whole run. The task has no curriculum or command
    # manager (both cfg dicts are empty), so mjlab's calls to those have no counterpart here.
    log = self.extras.setdefault("log", {})
    for mgr in (self.observation_manager, self.action_manager, self.reward_manager,
                self.metrics_manager, self.event_manager, self.termination_manager):
      fn = getattr(mgr, "reset", None)
      if fn is None:      # the action manager here is a bridge view, not mjlab's ActionManager
        continue
      info = fn(env_ids)
      if info:
        log.update(info)

    self._env.episode_length_buf[env_ids] = 0
    # qpos was written; xpos/xquat are stale until forward kinematics runs, and the observation
    # terms read the derived fields rather than qpos.
    self._env.forward()


  def _physics_substep_graphed(self, parity: int) -> None:
    """Run one solver substep, replaying a captured CUDA graph instead of relaunching kernels.

    `parity` selects which of the two buffer orientations this substep uses. Capture happens
    lazily on the first substep of each parity, after the uncaptured warm-up steps have forced
    every lazy kernel compilation and allocation to happen -- capturing an allocation would bake a
    stale pointer into the graph.
    """
    import warp as wp

    while len(self._graphs) <= parity:
      self._graphs.append(None)

    if self._graphs[parity] is None:
      if self._graph_warmup > 0:
        self._graph_warmup -= 1
        self.solver.step(self.state_in, self.state_out, self.control, None, self.physics_dt)
        return
      try:
        with wp.ScopedCapture() as cap:
          self.solver.step(self.state_in, self.state_out, self.control, None, self.physics_dt)
        self._graphs[parity] = cap.graph
        print(f"[newton-env] captured CUDA graph for substep parity {parity}")
        # Capture records the launches without running them, so this substep's physics has not
        # happened yet. Replaying the fresh graph performs it, keeping the captured step on the
        # trajectory instead of silently skipping it.
        wp.capture_launch(cap.graph)
        return
      except Exception as e:
        print(f"[newton-env] CUDA graph capture failed ({type(e).__name__}: {e}); "
              "continuing without it")
        self._use_cuda_graph = False
        self.solver.step(self.state_in, self.state_out, self.control, None, self.physics_dt)
        return

    wp.capture_launch(self._graphs[parity])

  def step(self, action: torch.Tensor):
    self.extras["log"] = dict()
    self.action_manager.advance(action)
    self.action_term.process_actions(action)

    for i in range(self.decimation):
      self.action_term.apply_actions()
      if self._use_cuda_graph:
        self._physics_substep_graphed(i % 2)
      else:
        self.solver.step(self.state_in, self.state_out, self.control, None, self.physics_dt)
      self.state_in, self.state_out = self.state_out, self.state_in

    self._env.episode_length_buf += 1
    self.common_step_counter += 1
    self._env.common_step_counter = self.common_step_counter

    if self.viewer is not None:
      self._render_tick += 1
      if self._render_tick % self._render_every == 0:
        self._render()

    # Order follows mjlab's step (manager_based_rl_env.py:451): terminations, then the reward,
    # then the metrics -- and the reward has to be published on the env as `reward_buf` before the
    # metrics run, because reward_total_metric reads it.
    terminated = self.termination_manager.compute()
    time_out = getattr(self.termination_manager, "time_outs",
                       torch.zeros_like(terminated, dtype=torch.bool))

    reward = self.reward_manager.compute(dt=self.step_dt)
    self._env.reward_buf = reward

    if self.metrics_manager is not None:
      # compute() takes no dt. The earlier `compute(dt=...)` wrapped in `except Exception: pass`
      # raised TypeError on every step of every run and was swallowed, so the manager never
      # produced a metric -- the Episode_Metrics/* values that reached tensorboard were
      # zero-initialised accumulators being flushed at reset.
      self.metrics_manager.compute()

    done_ids = (terminated | time_out).nonzero(as_tuple=False).squeeze(-1)
    if len(done_ids) > 0:
      self._reset_idx(done_ids)

    obs = self.get_observations()
    # surface the task metrics so the runner logs them alongside the PPO curves
    self.extras.setdefault("log", {}).update(self._env.extras.get("log", {}))
    self.extras["time_outs"] = time_out
    # mjlab's wrapper unpacks five values: terminated and truncated are separate, because a timeout
    # must bootstrap the value function while a real termination must not.
    return obs, reward, terminated, time_out, self.extras

  def _render(self) -> None:
    """Draw from mjw_data rather than Newton's State.

    Two reasons. The table is a mocap body: its pose is written to mocap_pos and never enters State,
    so it would draw at the origin with the object apparently floating. And with N replicated worlds
    Newton's bodies are laid out world-major (world w, local body b -> w*B + b) against mujoco's
    per-world (w, b+1), so the two have to be related explicitly rather than copied straight across.
    """
    bq = wp.to_torch(self.state_in.body_q)
    xp = wp.to_torch(self.solver.mjw_data.xpos)
    xq = wp.to_torch(self.solver.mjw_data.xquat)
    nworld, nb_mj = xp.shape[0], xp.shape[1]
    nb = nb_mj - 1                                   # mujoco body 0 is the world
    if bq.shape[0] == nworld * nb:
      bq[:, 0:3] = xp[:, 1:1 + nb].reshape(-1, 3)
      bq[:, 3:7] = xq[:, 1:1 + nb].reshape(-1, 4)[:, [1, 2, 3, 0]]   # wxyz -> xyzw
    self.viewer.begin_frame(self.common_step_counter * self.step_dt)
    self.viewer.log_state(self.state_in)
    self.viewer.end_frame()

  def close(self) -> None:
    if self.viewer is not None:
      try:
        self.viewer.close()
      except Exception:
        pass
    return None
