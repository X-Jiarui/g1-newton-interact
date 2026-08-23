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
               cuda_graph: bool = False, effortless_action: bool = False,
               table_under_object: bool = False, object_solref: str | None = None,
               dump_qpos: str | None = None, dump_steps: int = 600,
               newton_video: str | None = None, video_size: str = "960x720",
               video_steps: int = 500, video_cam: str | None = None) -> None:
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

    # Contact stiffness for the object's collider. Measured with the shared defaults
    # (solref 0.02 1.0): mjlab's analytic sphere settles 0.37mm into the table, the stapler's
    # convex-mesh collider settles 1.93mm and transiently 11.6mm -- deep enough to see. The knob
    # is scoped to the object geom so the robot's contacts keep mjlab's parameters exactly.
    if object_solref and sdf_object_stl:
      import mujoco as _mj
      _vals = [float(x) for x in object_solref.replace(" ", "").split(",")]
      _mm = self.solver.mj_model
      _hit = 0
      for _g in range(_mm.ngeom):
        _n = _mj.mj_id2name(_mm, _mj.mjtObj.mjOBJ_GEOM, _g) or ""
        if "apple" not in _n:
          continue
        _mm.geom_solref[_g][:len(_vals)] = _vals
        _hit += 1
      if _hit == 0:
        raise RuntimeError("--object-solref matched no object geom")
      import warp as _wp
      _wp.to_torch(self.solver.mjw_model.geom_solref)[:] = _wp.to_torch(
        _wp.array(_mm.geom_solref, dtype=float))
      print(f"[newton-env] object solref -> {_vals} on {_hit} geom(s) "
            f"(resting penetration 0.04mm stapler / 0.28mm mug, against 1.88mm "
            f"at the shared default and 0.37mm for mjlab's sphere)")

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
    # The torque law inside Sonic53Action.apply_actions is discarded on this backend --
    # set_joint_effort_target is a verified no-op here -- so it can be cut out along with the host
    # synchronisation it performs on every substep. See src/newton_action.py.
    _action_cls = amdp.Sonic53Action
    if effortless_action:
      from newton_action import make_effortless
      _action_cls = make_effortless(amdp.Sonic53Action)
      print("[newton-env] action term: torque law removed (it reaches no actuator here)")
    # Must be installed before the action term first writes a table pose. See src/newton_table.py:
    # the scene was authored around a 4cm sphere, so a real collider does not rest where the
    # reference puts the object unless the table moves to meet it.
    if table_under_object:
      if not sdf_object_stl:
        raise RuntimeError("--table-under-object needs the object mesh (--sdf-object) to know "
                           "where the real collider bottom is")
      from newton_table import install as _install_table
      _ref_pkl = os.environ.get("APPLE_EAT_PKL")
      if not _ref_pkl:
        raise RuntimeError("--table-under-object needs APPLE_EAT_PKL: the object's resting height "
                           "is read from the reference clip, not guessed")
      _install_table(self.solver.mj_model, sdf_object_stl, _ref_pkl,
                     z_offset=float(os.environ.get("APPLE_SCENE_Z_OFFSET", 0.0)))

    self.action_term = _action_cls(sonic_cfg, self._env)
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
    # Contact sensors have to reach the MDP through the scene, not just the model. The transplant
    # above puts them in Newton's compiled model, but mjlab reads them as scene entities
    # (`env.scene["hand_apple_contact"].data.found`) and falls back to zeros on a KeyError -- which
    # is why contact_duration (weight 1.0) and object_hard_lift (weight 2.0) never paid out, and
    # why every contact metric read 0.0 while the policy was visibly grasping.
    #
    # A one-environment mjlab env is built purely to obtain its constructed sensor objects; they are
    # then rebound onto Newton's buffers. Constructing them by hand would mean reimplementing
    # mjlab's pattern expansion, which is the kind of second implementation that has drifted before.
    import copy as _copy
    from mjlab.envs import ManagerBasedRlEnv as _MjlabEnv
    from newton_bridge import bind_contact_sensors

    _sensor_cfg = _copy.deepcopy(cfg)
    _sensor_cfg.scene.num_envs = 1
    _probe = _MjlabEnv(cfg=_sensor_cfg, device=device, render_mode=None)
    _bound = bind_contact_sensors(self._env.scene, _probe.scene, self.solver.mj_model,
                                  self.solver.mjw_model, self.solver.mjw_data, device)
    self._contact_sensors = [self._env.scene[n] for n in _bound]
    print(f"[newton-env] contact sensors on the scene: {len(_bound)} "
          f"({', '.join(n for n in _bound if 'contact' in n and n.count('_') < 4)})")

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
    # Optional recording of env 0 for video. The table is a mocap body, so its pose lives outside
    # qpos; without it the renderer draws the table at the origin and the object appears to float.
    self._video_path = newton_video
    self._video_steps = int(video_steps)
    self._video_size = video_size
    self._video_cam = video_cam
    self._viewer_gl = None
    self._video_frames: list = []
    self._cam_eye = self._cam_target = None

    self._dump_qpos_path = dump_qpos
    self._dump_steps = int(dump_steps)
    self._qpos_log: list = []
    self._mocap_log: list = []

    self._use_cuda_graph = bool(cuda_graph)
    self._graphs: list = []
    self._graph_warmup = 4      # uncaptured substeps per parity before capturing
    # The bridge reads the runner's `_residual_*` bookkeeping through this back-reference; the
    # runner writes it on this object, not on the bridge the MDP terms see.
    self._env._owner = self

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


  def _record_frame(self) -> None:
    """Append env 0's qpos and mocap pose, and write the npz once enough frames are collected."""
    import warp as wp
    import numpy as np

    d = self.solver.mjw_data
    self._qpos_log.append(wp.to_torch(d.qpos)[0].detach().cpu().numpy().copy())
    self._mocap_log.append((wp.to_torch(d.mocap_pos)[0].detach().cpu().numpy().copy(),
                            wp.to_torch(d.mocap_quat)[0].detach().cpu().numpy().copy()))
    if len(self._qpos_log) < self._dump_steps:
      return
    # Mocap body names travel with the poses. The renderer replays into a different model whose
    # mocap bodies need not be in the same order -- index 2 here is the table, and writing it into
    # index 0 there put the table under the robot's feet in the first video made this way.
    import mujoco as _mj
    m = self.solver.mj_model
    names = [_mj.mj_id2name(m, _mj.mjtObj.mjOBJ_BODY, b) or ""
             for b in range(m.nbody) if m.body_mocapid[b] >= 0]
    order = [b for b in range(m.nbody) if m.body_mocapid[b] >= 0]
    names = [x for _, x in sorted(zip([int(m.body_mocapid[b]) for b in order], names))]
    np.savez_compressed(self._dump_qpos_path,
                        qpos=np.stack(self._qpos_log),
                        mocap_pos=np.stack([q[0] for q in self._mocap_log]),
                        mocap_quat=np.stack([q[1] for q in self._mocap_log]),
                        mocap_names=np.array(names))
    print(f"[newton-env] wrote {self._dump_qpos_path} "
          f"({len(self._qpos_log)} frames of env 0 for rendering)")


  def _setup_newton_viewer(self) -> None:
    """Newton's own offscreen renderer, drawing the scene the physics actually holds.

    Replaying qpos through mjlab's visual model instead -- which is what the first videos did --
    draws the object as the 4cm placeholder sphere the scene authors, because the real mesh only
    ever reached the collider. It also has to re-apply the mocap table by hand, and getting that
    mapping wrong put the table under the robot's feet. Rendering Newton's own model has neither
    failure mode: what is drawn is what was simulated.
    """
    import numpy as np
    import newton
    import torch
    import warp as wp

    # pyglet creates a shadow window even when the viewer is asked for headless, which fails on a
    # box with no display. Its headless path uses EGL and must be selected before pyglet is imported.
    os.environ.setdefault("PYGLET_HEADLESS", "1")
    import pyglet as _pyglet
    _pyglet.options["headless"] = True
    import newton.viewer as _nv

    w, h = (int(x) for x in self._video_size.lower().split("x"))
    self._viewer_gl = _nv.ViewerGL(width=w, height=h, headless=True)

    # Newton draws visual shapes, not colliders. mjlab's robot carries visual meshes, but the object
    # and the table are each a single geom that is both visual and collidable, and arrives here
    # marked collide-but-not-visible -- a video of a robot grasping thin air over an invisible
    # table. Reveal colliders only on bodies that have no visible shape at all, so the robot is not
    # drawn twice.
    flags = wp.to_torch(self.nmodel.shape_flags)
    sbody = wp.to_torch(self.nmodel.shape_body)
    VIS = int(newton.ShapeFlags.VISIBLE)
    revealed = 0
    for bid in torch.unique(sbody):
      idx = (sbody == bid).nonzero(as_tuple=True)[0]
      if int((flags[idx] & VIS).sum()) == 0:
        flags[idx] |= VIS
        revealed += len(idx)

    self._viewer_gl.set_model(self.nmodel)

    # The viewer spreads parallel worlds apart for display -- _auto_compute_world_offsets runs on
    # set_model -- so with four envs the robot is drawn far from where the physics puts it, and a
    # camera aimed at the true scene sees bare floor with a speck on the horizon. Collapse the
    # offsets so the picture matches the simulation, and draw only world 0 so the video shows one
    # robot at full size instead of four small ones.
    try:
      self._viewer_gl.set_world_offsets((0.0, 0.0, 0.0))
    except Exception as e:
      print(f"[newton-env] could not clear world offsets ({type(e).__name__}: {e})")
    if self.num_envs > 1:
      try:
        self._viewer_gl.set_visible_worlds([0])
      except Exception as e:
        print(f"[newton-env] could not restrict rendering to world 0 "
              f"({type(e).__name__}: {e}); all {self.num_envs} will be drawn")
    cam = getattr(self._viewer_gl, "camera", None)
    if cam is not None:
      # Framed on the hands and the object rather than the whole scene: the earlier eye sat 1.69m
      # back, which fits the robot and the table in but leaves the grasp itself a few pixels wide.
      # This sits 1.07m out in three-quarter view: the whole robot reads large, and the object on the
      # table stays in frame with both hands. Override with --video-cam "ex,ey,ez,tx,ty,tz".
      eye, target = [1.55, -0.55, 1.05], [0.60, -0.10, 0.84]
      if self._video_cam:
        nums = [float(x) for x in self._video_cam.replace(" ", "").split(",")]
        if len(nums) != 6:
          raise ValueError(f"--video-cam wants 6 numbers (eye xyz, target xyz), got {len(nums)}")
        eye, target = nums[:3], nums[3:]

      # Anchor on environment 0. Replicating worlds spreads them across the ground plane, so world
      # coordinates that frame the robot at one env count point at bare floor at another -- checked
      # with one env, the framing was right; the four-env video it was used for showed the robot as
      # a speck on the horizon.
      origin = self._env.scene.env_origins[0].detach().cpu().numpy()
      eye = [eye[i] + float(origin[i]) for i in range(3)]
      target = [target[i] + float(origin[i]) for i in range(3)]
      self._cam_eye, self._cam_target = eye, target
      print(f"[newton-env] env 0 origin {np.round(origin, 3).tolist()}")
      print(f"[newton-env] camera at {eye} looking at {target} "
            f"({np.linalg.norm(np.array(eye) - np.array(target)):.2f} m out)")
    else:
      self._cam_eye = self._cam_target = None

    print(f"[newton-env] Newton ViewerGL {w}x{h} headless, "
          f"{revealed} collider shape(s) revealed, writing {self._video_path}")


  def _apply_camera(self) -> None:
    """Point the viewer's camera at the target, through the API the viewer actually reads.

    ViewerGL has no look-at: `set_camera(pos, pitch, yaw)` is the entry point, and assigning to
    `viewer.camera.pos` does nothing, because set_model replaces the Camera object. For a Z-up
    scene the viewer derives its forward vector as pitch = asin(dz), yaw = atan2(dy, dx).
    """
    import numpy as np
    import warp as wp

    if self._cam_eye is None or self._viewer_gl is None:
      return
    eye = np.asarray(self._cam_eye, dtype=np.float64)
    d = np.asarray(self._cam_target, dtype=np.float64) - eye
    n = np.linalg.norm(d)
    if n < 1e-9:
      return
    d /= n
    pitch = float(np.rad2deg(np.arcsin(np.clip(d[2], -1.0, 1.0))))
    yaw = float(np.rad2deg(np.arctan2(d[1], d[0])))
    try:
      self._viewer_gl.set_camera(wp.vec3(*(float(x) for x in eye)), pitch, yaw)
    except Exception as e:
      if not getattr(self, "_cam_warned", False):
        print(f"[newton-env] camera placement failed ({type(e).__name__}: {e})")
        self._cam_warned = True

  def _capture_video_frame(self) -> None:
    """Render one frame from Newton's own state, after syncing it from the authoritative mjw_data."""
    import numpy as np
    import warp as wp

    if self._viewer_gl is None:
      self._setup_newton_viewer()

    # mjw_data is authoritative here (update_data_interval=0), and the table is a mocap body whose
    # pose never enters State -- without this sync it renders at the origin. Newton body i is
    # MuJoCo body i+1 (MuJoCo body 0 is the world), and Newton stores quaternions xyzw against
    # MuJoCo's wxyz; both conventions were measured, not assumed.
    bq = wp.to_torch(self.state_in.body_q)
    xp = wp.to_torch(self.solver.mjw_data.xpos)
    xq = wp.to_torch(self.solver.mjw_data.xquat)
    nworld = xp.shape[0]
    per_world = bq.shape[0] // nworld
    bqv = bq.view(nworld, per_world, 7)
    bqv[:, :, 0:3] = xp[:, 1:1 + per_world]
    bqv[:, :, 3:7] = xq[:, 1:1 + per_world][:, :, [1, 2, 3, 0]]

    # Re-applied every frame. set_model rebuilds Camera, and the viewer refits it to the scene on
    # its first frame; either one silently discards a placement made at setup. That is why a
    # one-env check looked right -- the refit lands near the robot when the scene is small -- while
    # the four-env video it stood in for rendered the robot as a speck on the horizon.
    self._apply_camera()

    t = len(self._video_frames) * self.step_dt
    self._viewer_gl.begin_frame(t)
    self._viewer_gl.log_state(self.state_in)
    self._viewer_gl.end_frame()
    img = self._viewer_gl.get_frame()
    self._video_frames.append(
      np.asarray(img.numpy() if hasattr(img, "numpy") else img).copy())

    if len(self._video_frames) < self._video_steps:
      return
    import imageio.v2 as imageio
    wr = imageio.get_writer(self._video_path, fps=int(round(1.0 / self.step_dt)),
                            codec="libx264", macro_block_size=None, quality=8)
    for f in self._video_frames:
      wr.append_data(f)
    wr.close()
    self._viewer_gl.close()
    self._viewer_gl = None
    print(f"[newton-env] wrote {self._video_path} "
          f"({len(self._video_frames)} frames from Newton's own renderer)")
    self._video_path = None

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
    for _sensor in getattr(self, "_contact_sensors", ()):
      _sensor.update(self.step_dt)

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
    if self._dump_qpos_path is not None and len(self._qpos_log) < self._dump_steps:
      self._record_frame()

    if self._video_path is not None:
      self._capture_video_frame()

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
