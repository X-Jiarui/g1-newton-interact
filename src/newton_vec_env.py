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
               sdf_hydroelastic: bool = False) -> None:
    import mujoco
    import newton
    from newton.solvers import SolverMuJoCo
    from newton_simple_fix import capture_spec, restore_simple_bodies, restore_freejoint_damping
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

    with capture_spec() as cap:
      # update_data_interval=0 keeps mjw_data authoritative, so the direct joint/root writes the
      # action term performs during the startup hold are not overwritten from Newton's State.
      self.solver = SolverMuJoCo(self.nmodel, enable_multiccd=True, update_data_interval=0,
                                 njmax=njmax, nconmax=nconmax)
    self._ref_mj = mujoco.MjModel.from_xml_path(xml)
    restore_freejoint_damping(cap.spec, xml, verbose=False)
    restore_simple_bodies(self.solver, cap.spec, nworld=self.num_envs,
                          nconmax=nconmax, njmax=njmax, verbose=False)
    nC, nC_ref = int(self.solver.mjw_model.nC), int(self._ref_mj.nC)
    if nC != nC_ref:
      # With the analytic sphere this would be a defect: the object is a free body with isotropic
      # diagonal inertia, so MuJoCo's compressed mass-matrix layout applies and a mismatch means the
      # spec repair did not take. With a real mesh collider it is the correct answer -- the inertia
      # is neither isotropic nor centred, so the body genuinely is not "simple" and the compressed
      # layout does not apply to it.
      note = "expected for a mesh collider" if sdf_object_stl else "UNEXPECTED for a sphere object"
      print(f"[newton-env] nC {nC} vs reference {nC_ref} ({note})")
      if not sdf_object_stl:
        raise RuntimeError(f"nC {nC} != {nC_ref} with a sphere object; the spec repair did not take")

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
      print(f"[newton-env] ObservationManager unavailable ({type(e).__name__}: {e}); "
            "falling back to per-group builders")

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
        except Exception:
          pass

    self.reward_manager = RewardManager(cfg.rewards, self._env)
    self.termination_manager = TerminationManager(cfg.terminations, self._env)
    self.event_manager = EventManager(cfg.events, self._env)

    self.episode_length_buf = self._env.episode_length_buf
    self._all = torch.arange(self.num_envs, dtype=torch.long, device=device)
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
    self._env.episode_length_buf[env_ids] = 0
    # qpos was written; xpos/xquat are stale until forward kinematics runs, and the observation
    # terms read the derived fields rather than qpos.
    self._env.forward()

  def step(self, action: torch.Tensor):
    self.extras["log"] = dict()
    self.action_manager.advance(action)
    self.action_term.process_actions(action)

    for _ in range(self.decimation):
      self.action_term.apply_actions()
      self.solver.step(self.state_in, self.state_out, self.control, None, self.physics_dt)
      self.state_in, self.state_out = self.state_out, self.state_in

    self._env.episode_length_buf += 1
    self.common_step_counter += 1
    self._env.common_step_counter = self.common_step_counter

    reward = self.reward_manager.compute(dt=self.step_dt)
    terminated = self.termination_manager.compute()
    time_out = getattr(self.termination_manager, "time_outs",
                       torch.zeros_like(terminated, dtype=torch.bool))

    done_ids = (terminated | time_out).nonzero(as_tuple=False).squeeze(-1)
    if len(done_ids) > 0:
      self._reset_idx(done_ids)

    obs = self.get_observations()
    self.extras["time_outs"] = time_out
    # mjlab's wrapper unpacks five values: terminated and truncated are separate, because a timeout
    # must bootstrap the value function while a real termination must not.
    return obs, reward, terminated, time_out, self.extras

  def close(self) -> None:
    return None
