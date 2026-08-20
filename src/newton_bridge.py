"""Run mjlab's own observation and action code against Newton's mujoco_warp state.

Why an adapter instead of a reimplementation: the residual policy sees 1328 dims assembled by
`ResidualFeatureGroupObs`, and `Sonic53Action` does considerably more than emit ctrl -- during the
first 36 steps it replays the reference pose, during the first 30 it writes root and joint state
directly, and it also drives the table pose and the reference object tracking. Re-deriving any of
that is an opportunity for a silent mismatch, and a mismatch does not raise: the policy still runs,
just against a different world than it was trained in.

What makes the adapter thin is that Newton's SolverMuJoCo runs mujoco_warp underneath -- the same
engine mjlab uses -- and the two compiled models are verified identical in joint, body and actuator
ordering. So `mjw_data` has mjlab's exact layout (qpos 83, qvel 81, ctrl 138, xpos/xquat 92) and the
index maps are shared.

Reading `mjw_data` rather than Newton's `State` also sidesteps Newton's convention differences
wholesale. Newton stores quaternions xyzw, reports linear velocity at the CoM and angular velocity in
the world frame; MuJoCo uses wxyz, the body-frame origin, and body-frame angular velocity. Those
differences live in Newton's Model/State API. `mjw_data` is MuJoCo's own structure and still carries
MuJoCo's conventions, so nothing has to be converted -- and nothing can be forgotten.

Warp arrays are exposed through `wp.to_torch`, which shares memory rather than copying, so reads are
views on live simulator state exactly as they are in mjlab.
"""

from __future__ import annotations

from typing import Any, Sequence

import mujoco
import numpy as np
import torch
import warp as wp


def _t(arr) -> torch.Tensor:
  """Zero-copy view of a warp array as torch."""
  return wp.to_torch(arr)


class _Indexing:
  """The ctrl ids mjlab's entity exposes, resolved by actuator name."""

  def __init__(self, ctrl_ids: torch.Tensor) -> None:
    self.ctrl_ids = ctrl_ids


class _EntityData:
  """mjlab's Entity.data field names, backed by mujoco_warp Data."""

  def __init__(self, owner: "NewtonEntity") -> None:
    self._o = owner

  # --- joints (entity-local order, which equals mjlab's) --------------------
  @property
  def joint_pos(self) -> torch.Tensor:
    return _t(self._o._d.qpos)[:, self._o._qadr]

  @property
  def joint_vel(self) -> torch.Tensor:
    return _t(self._o._d.qvel)[:, self._o._vadr]

  # --- root ------------------------------------------------------------------
  @property
  def root_link_pos_w(self) -> torch.Tensor:
    return _t(self._o._d.xpos)[:, self._o._root_body]

  @property
  def root_link_quat_w(self) -> torch.Tensor:
    return _t(self._o._d.xquat)[:, self._o._root_body]

  @property
  def root_link_pose_w(self) -> torch.Tensor:
    return torch.cat((self.root_link_pos_w, self.root_link_quat_w), dim=-1)

  @property
  def root_link_vel_w(self) -> torch.Tensor:
    # mujoco cvel is (angular, linear) in the body frame; mjlab's root_link_vel_w is (lin, ang).
    cv = _t(self._o._d.cvel)[:, self._o._root_body]
    return torch.cat((cv[..., 3:6], cv[..., 0:3]), dim=-1)

  @property
  def root_link_lin_vel_w(self) -> torch.Tensor:
    return self.root_link_vel_w[..., 0:3]

  @property
  def root_link_ang_vel_w(self) -> torch.Tensor:
    return self.root_link_vel_w[..., 3:6]

  # --- bodies ----------------------------------------------------------------
  @property
  def body_link_pos_w(self) -> torch.Tensor:
    return _t(self._o._d.xpos)[:, self._o._body_ids]

  @property
  def body_link_quat_w(self) -> torch.Tensor:
    return _t(self._o._d.xquat)[:, self._o._body_ids]

  @property
  def default_joint_pos(self) -> torch.Tensor:
    return self._o._default_joint_pos

  def __getattr__(self, name: str) -> Any:
    raise AttributeError(
      f"NewtonEntity.data has no mapping for {name!r}. Add it here rather than letting the "
      "feature builder read something else -- a wrong number here is silent."
    )


class NewtonEntity:
  """One mjlab entity (robot / object / table) viewed onto a mujoco_warp model+data."""

  def __init__(self, model, data, prefix: str, device: str = "cuda:0") -> None:
    self._m, self._d, self._prefix, self.device = model, data, prefix, device
    O = mujoco.mjtObj

    def names(objtype, count):
      return [mujoco.mj_id2name(model, objtype, i) or "" for i in range(count)]

    all_bodies = names(O.mjOBJ_BODY, model.nbody)
    all_joints = names(O.mjOBJ_JOINT, model.njnt)
    all_acts = names(O.mjOBJ_ACTUATOR, model.nu)

    self._body_ids = torch.tensor(
      [i for i, n in enumerate(all_bodies) if n.startswith(prefix)], dtype=torch.long, device=device)
    self.body_names = tuple(n[len(prefix):] for n in all_bodies if n.startswith(prefix))
    if len(self._body_ids) == 0:
      raise ValueError(f"no bodies with prefix {prefix!r}")
    self._root_body = int(self._body_ids[0])

    # Hinge joints only, in model order -- mjlab's robot.joint_names excludes the free joint.
    jids = [i for i, n in enumerate(all_joints)
            if n.startswith(prefix) and int(model.jnt_type[i]) == mujoco.mjtJoint.mjJNT_HINGE]
    self.joint_names = tuple(all_joints[i][len(prefix):] for i in jids)
    self._qadr = torch.tensor([int(model.jnt_qposadr[i]) for i in jids], dtype=torch.long, device=device)
    self._vadr = torch.tensor([int(model.jnt_dofadr[i]) for i in jids], dtype=torch.long, device=device)

    # The free joint, if this entity has one: needed to write root state.
    free = [i for i, n in enumerate(all_joints)
            if n.startswith(prefix) and int(model.jnt_type[i]) == mujoco.mjtJoint.mjJNT_FREE]
    self._free_qadr = int(model.jnt_qposadr[free[0]]) if free else None
    self._free_vadr = int(model.jnt_dofadr[free[0]]) if free else None

    # ctrl ids: only the real servos. The `xml_motor_unused_*` actuators were measured to hold ctrl
    # at exactly 0.0 for the whole rollout, so writing to them would be writing to nothing -- but
    # including them would also silently shift every ctrl index.
    act_ids = [i for i, n in enumerate(all_acts)
               if n.startswith(prefix) and "xml_motor_unused" not in n]
    self.actuator_names = tuple(all_acts[i] for i in act_ids)
    self.indexing = _Indexing(torch.tensor(act_ids, dtype=torch.long, device=device))
    # ctrl slot for each hinge joint, in this entity's joint order
    trn = {int(model.actuator_trnid[i][0]): i for i in act_ids}
    self._ctrl_for_joint = torch.tensor([trn[j] for j in jids], dtype=torch.long, device=device)

    self._default_joint_pos = torch.zeros((data.qpos.shape[0], len(jids)), device=device)
    self.data = _EntityData(self)

  # --- lookups mjlab uses ----------------------------------------------------
  def find_bodies(self, name_keys, preserve_order: bool = False):
    keys = [name_keys] if isinstance(name_keys, str) else list(name_keys)
    idx, names = [], []
    for k in keys:
      for i, n in enumerate(self.body_names):
        if n == k or n.endswith("/" + k) or k in n:
          idx.append(i); names.append(n); break
    return idx, names

  def find_joints(self, name_keys, preserve_order: bool = False):
    keys = [name_keys] if isinstance(name_keys, str) else list(name_keys)
    idx, names = [], []
    for k in keys:
      for i, n in enumerate(self.joint_names):
        if n == k or n.endswith("/" + k):
          idx.append(i); names.append(n); break
    return idx, names

  # --- writes ----------------------------------------------------------------
  def set_joint_position_target(self, target: torch.Tensor, joint_ids=None) -> None:
    ctrl = _t(self._d.ctrl)
    slots = self._ctrl_for_joint if joint_ids is None else self._ctrl_for_joint[joint_ids]
    ctrl[:, slots] = target.to(ctrl.dtype)

  def set_joint_effort_target(self, effort: torch.Tensor, joint_ids=None) -> None:
    # Deliberately a no-op. Measured in mjlab with the trained policy: ctrl on every
    # `xml_motor_unused_*` actuator stays exactly 0.0 and never varies, so this torque law reaches no
    # actuator that can produce force. Implementing it here would ADD a force mjlab does not apply.
    return

  def write_joint_state_to_sim(self, q, qd, joint_ids=None, env_ids=None) -> None:
    qpos, qvel = _t(self._d.qpos), _t(self._d.qvel)
    qa = self._qadr if joint_ids is None else self._qadr[joint_ids]
    va = self._vadr if joint_ids is None else self._vadr[joint_ids]
    e = slice(None) if env_ids is None else env_ids
    qpos[e][:, qa] = q.to(qpos.dtype) if env_ids is None else q.to(qpos.dtype)
    if env_ids is None:
      qpos[:, qa] = q.to(qpos.dtype); qvel[:, va] = qd.to(qvel.dtype)
    else:
      qpos[env_ids[:, None], qa[None, :]] = q.to(qpos.dtype)
      qvel[env_ids[:, None], va[None, :]] = qd.to(qvel.dtype)

  def write_root_state_to_sim(self, root_state: torch.Tensor, env_ids=None) -> None:
    """root_state is mjlab's 13: pos(3), quat(4, wxyz), lin vel(3), ang vel(3)."""
    if self._free_qadr is None:
      raise ValueError(f"entity {self._prefix!r} has no free joint; cannot write root state")
    qpos, qvel = _t(self._d.qpos), _t(self._d.qvel)
    qa, va = self._free_qadr, self._free_vadr
    e = slice(None) if env_ids is None else env_ids
    qpos[e, qa:qa + 7] = root_state[:, 0:7].to(qpos.dtype)
    qvel[e, va:va + 6] = root_state[:, 7:13].to(qvel.dtype)

  def write_root_pose_to_sim(self, pose: torch.Tensor, env_ids=None) -> None:
    if self._free_qadr is None:
      return
    qpos = _t(self._d.qpos)
    e = slice(None) if env_ids is None else env_ids
    qpos[e, self._free_qadr:self._free_qadr + 7] = pose.to(qpos.dtype)


class SceneView(dict):
  def __getitem__(self, key):
    try:
      return super().__getitem__(key)
    except KeyError:
      raise KeyError(f"scene has no entity {key!r}; present: {sorted(self.keys())}") from None


class NewtonEnv:
  """The `env` mjlab's builders and action term expect, backed by Newton."""

  def __init__(self, model, data, num_envs: int, device: str = "cuda:0",
               object_entity: str = "apple") -> None:
    self.num_envs = int(num_envs)
    self.device = device
    self.mj_model, self.mjw_data = model, data
    scene = SceneView()
    scene["robot"] = NewtonEntity(model, data, "robot/", device)
    scene[object_entity] = NewtonEntity(model, data, f"{object_entity}/", device)
    try:
      scene["table"] = NewtonEntity(model, data, "table/", device)
    except ValueError:
      pass          # some scenes have no table entity; mjlab guards for this too
    self.scene = scene
    self.scene.env_origins = torch.zeros((self.num_envs, 3), device=device)
    self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=device)
    self.extras: dict[str, Any] = {}
