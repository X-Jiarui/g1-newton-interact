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

# mjlab's own maths, reused rather than re-derived. `root_link_vel_w` in particular is NOT a slice of
# cvel: MuJoCo reports cvel about the subtree COM, and mjlab corrects for the offset between that and
# the body frame origin. Re-deriving it by slicing produced a plausible-looking velocity that was
# simply wrong, which is precisely the silent-error class this bridge exists to avoid.
from mjlab.entity.data import compute_velocity_from_cvel
from mjlab.utils.lab_api.math import quat_apply_inverse


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
    o = self._o
    pos = _t(o._d.xpos)[:, o._root_body]
    com = _t(o._d.subtree_com)[:, o._root_body]
    cvel = _t(o._d.cvel)[:, o._root_body]
    return compute_velocity_from_cvel(pos, com, cvel)

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

  # --- body-frame derivations, defined exactly as mjlab defines them ------------
  @property
  def root_link_ang_vel_b(self) -> torch.Tensor:
    return quat_apply_inverse(self.root_link_quat_w, self.root_link_ang_vel_w)

  @property
  def root_link_lin_vel_b(self) -> torch.Tensor:
    return quat_apply_inverse(self.root_link_quat_w, self.root_link_lin_vel_w)

  @property
  def gravity_vec_w(self) -> torch.Tensor:
    g = torch.zeros_like(self.root_link_pos_w)
    g[:, 2] = -1.0
    return g

  @property
  def projected_gravity_b(self) -> torch.Tensor:
    return quat_apply_inverse(self.root_link_quat_w, self.gravity_vec_w)

  @property
  def body_link_lin_vel_w(self) -> torch.Tensor:
    o = self._o
    pos = _t(o._d.xpos)[:, o._body_ids]
    com = _t(o._d.subtree_com)[:, o._body_ids]
    cvel = _t(o._d.cvel)[:, o._body_ids]
    return compute_velocity_from_cvel(pos, com, cvel)[..., 0:3]

  @property
  def body_link_ang_vel_w(self) -> torch.Tensor:
    return _t(self._o._d.cvel)[:, self._o._body_ids, 0:3]

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

  def __init__(self, model, data, prefix: str, device: str = "cuda:0", control=None,
               rename_from=None) -> None:
    self._m, self._d, self._prefix, self.device = model, data, prefix, device
    # Control targets must go through Newton's Control object, not mjw_data.ctrl: SolverMuJoCo calls
    # _apply_mjc_control on every step, so a direct ctrl write is overwritten before it is ever used
    # (measured: 0.777 -> 0.0 after one step). control.mujoco.ctrl is the actuator-indexed array these
    # general/MJCF actuators read, and its indices were measured to be identity with the model's
    # actuator order -- which is itself identical to mjlab's.
    self._control = control
    O = mujoco.mjtObj

    def names(objtype, count):
      return [mujoco.mj_id2name(model, objtype, i) or "" for i in range(count)]

    all_bodies = names(O.mjOBJ_BODY, model.nbody)
    all_joints = names(O.mjOBJ_JOINT, model.njnt)
    all_acts = names(O.mjOBJ_ACTUATOR, model.nu)

    # Newton's MJCF export renames everything to a flattened path -- `robot/left_hip_pitch_joint`
    # comes back as `mjlab scene_worldbody_robot_pelvis_..._robot_left_hip_pitch_joint` -- and drops
    # actuator names entirely. mjlab addresses entities by its own names, so they are mapped back by
    # longest flattened-suffix match. Longest wins because `..._left_hip_roll_joint` also ends with
    # `_roll_joint`; a first or shortest match would mis-assign silently. This is the same mapping
    # verified at 92/92 bodies and 71/71 joints in docs/newton_live_canon.json.
    if rename_from is not None:
      all_bodies = _rename(all_bodies, [mujoco.mj_id2name(rename_from, O.mjOBJ_BODY, i) or ""
                                        for i in range(rename_from.nbody)], "bodies")
      all_joints = _rename(all_joints, [mujoco.mj_id2name(rename_from, O.mjOBJ_JOINT, i) or ""
                                        for i in range(rename_from.njnt)], "joints")
      # Actuators are unnamed on the Newton side; name each by the joint it drives, which is the
      # identifying property anyway and is what mjlab's ctrl resolution keys on.
      named_acts = []
      for i in range(model.nu):
        jid = int(model.actuator_trnid[i][0])
        named_acts.append(all_joints[jid] if 0 <= jid < len(all_joints) else "")
      all_acts = named_acts

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
    # With actuators named by their joint, the "unused" ones cannot be told apart by name -- but two
    # actuators share each joint and only one is real. The real servo is the one MuJoCo will actually
    # drive: forcerange is non-degenerate and the bias is affine (a position servo). The measured
    # mjlab model has the leftovers at forcerange [0,0] for the body and a 100x smaller gain on the
    # hand, so the larger-forcerange actuator per joint is the real one.
    by_joint: dict[str, list[int]] = {}
    for i, n in enumerate(all_acts):
      if n.startswith(prefix):
        by_joint.setdefault(n, []).append(i)
    act_ids = []
    for n, cands in by_joint.items():
      if len(cands) == 1:
        act_ids.append(cands[0]); continue
      best = max(cands, key=lambda i: float(model.actuator_forcerange[i][1]
                                            - model.actuator_forcerange[i][0]))
      act_ids.append(best)
    act_ids.sort()
    self.actuator_names = tuple(all_acts[i] for i in act_ids)
    self.indexing = _Indexing(torch.tensor(act_ids, dtype=torch.long, device=device))
    # ctrl slot for each hinge joint, in this entity's joint order
    trn = {int(model.actuator_trnid[i][0]): i for i in act_ids}
    self._ctrl_for_joint = torch.tensor([trn[j] for j in jids], dtype=torch.long, device=device)

    # A mocap body is driven by writing mocap_pos/mocap_quat rather than qpos: mjlab's table is
    # exactly this (body 'table/mocap_base', mocapid 0), and _write_table_pose branches on it.
    mocaps = [int(model.body_mocapid[int(i)]) for i in self._body_ids
              if int(model.body_mocapid[int(i)]) >= 0]
    self._mocap_id = mocaps[0] if mocaps else None

    self._default_joint_pos = torch.zeros((data.qpos.shape[0], len(jids)), device=device)
    self.data = _EntityData(self)

  @property
  def is_mocap(self) -> bool:
    return self._mocap_id is not None

  def write_mocap_pose_to_sim(self, pose: torch.Tensor, env_ids=None) -> None:
    if self._mocap_id is None:
      raise ValueError(f"entity {self._prefix!r} has no mocap body")
    mp, mq = _t(self._d.mocap_pos), _t(self._d.mocap_quat)
    e = slice(None) if env_ids is None else env_ids
    mp[e, self._mocap_id] = pose[:, 0:3].to(mp.dtype)
    mq[e, self._mocap_id] = pose[:, 3:7].to(mq.dtype)

  def write_external_wrench_to_sim(self, forces: torch.Tensor, torques: torch.Tensor,
                                   env_ids=None, body_ids=None) -> None:
    """Set xfrc_applied, exactly as mjlab does -- world frame, persists until overwritten.

    With the pelvis start-assist gain at 0.0 (what every candidate trained with) this writes zeros.
    It is still implemented rather than stubbed: mjlab's wrench persists between steps, so silently
    skipping the write would leave whatever was there before, and a no-op would diverge the moment
    anyone ran with the assist enabled.
    """
    x = _t(self._d.xfrc_applied)
    e = slice(None) if env_ids is None else env_ids
    ids = (self._body_ids if body_ids is None
           else self._body_ids[torch.as_tensor(body_ids, dtype=torch.long, device=self._body_ids.device)])
    x[e][:, ids, 0:3] = forces.to(x.dtype)
    x[e][:, ids, 3:6] = torques.to(x.dtype)
    if env_ids is None:
      x[:, ids, 0:3] = forces.to(x.dtype)
      x[:, ids, 3:6] = torques.to(x.dtype)

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
    if self._control is None:
      raise RuntimeError("no Newton Control bound; position targets would be silently discarded")
    # Reshape by world count, not to a single row: with N parallel worlds control.mujoco.ctrl holds
    # N x nu entries and a hardcoded (1, -1) silently addresses only the first world.
    nworld = _t(self._d.qpos).shape[0]
    ctrl = _t(self._control.mujoco.ctrl).view(nworld, -1)
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
    if env_ids is None:
      qpos[:, qa] = q.to(qpos.dtype)
      qvel[:, va] = qd.to(qvel.dtype)
    else:
      e = env_ids.view(-1, 1)
      qpos[e, qa.view(1, -1)] = q.to(qpos.dtype)
      qvel[e, va.view(1, -1)] = qd.to(qvel.dtype)

  def write_root_state_to_sim(self, root_state: torch.Tensor, env_ids=None) -> None:
    """root_state is mjlab's 13: pos(3), quat(4, wxyz), lin vel(3), ang vel(3) -- ALL world frame.

    The angular velocity has to be rotated into the body frame before it goes into qvel: MuJoCo's
    free joint stores linear velocity in the world frame but angular velocity in the body frame, and
    mjlab does this conversion in Entity.write_root_velocity. Writing the world-frame value straight
    through leaves the pose perfectly correct and the angular velocity wrong -- measured at 0.34 rad/s
    of error, which lands directly on the ASTRA tracker's gyro input (astra_obs slots 0-2) and on
    nothing else, so every other observation looks fine while the tracker is fed a lie.
    """
    if self._free_qadr is None:
      raise ValueError(f"entity {self._prefix!r} has no free joint; cannot write root state")
    qpos, qvel = _t(self._d.qpos), _t(self._d.qvel)
    qa, va = self._free_qadr, self._free_vadr
    e = slice(None) if env_ids is None else env_ids
    pose = root_state[:, 0:7].to(qpos.dtype)
    qpos[e, qa:qa + 7] = pose
    quat_w = pose[:, 3:7]
    ang_vel_b = quat_apply_inverse(quat_w, root_state[:, 10:13].to(qpos.dtype))
    qvel[e, va:va + 3] = root_state[:, 7:10].to(qvel.dtype)
    qvel[e, va + 3:va + 6] = ang_vel_b.to(qvel.dtype)

  def write_root_pose_to_sim(self, pose: torch.Tensor, env_ids=None) -> None:
    if self._free_qadr is None:
      return
    qpos = _t(self._d.qpos)
    e = slice(None) if env_ids is None else env_ids
    qpos[e, self._free_qadr:self._free_qadr + 7] = pose.to(qpos.dtype)


def _rename(newton_names: Sequence[str], ref_names: Sequence[str], what: str) -> list[str]:
  """Map Newton's flattened names back onto mjlab's, by longest flattened-suffix match."""
  flat = sorted(((r.replace("/", "_"), r) for r in ref_names if r), key=lambda t: -len(t[0]))
  out, unmapped = [], 0
  for n in newton_names:
    hit = ""
    for f, r in flat:
      if n == r or n.endswith(f):
        hit = r
        break
    if not hit:
      unmapped += 1
    out.append(hit or n)
  if unmapped:
    print(f"[bridge] WARNING: {unmapped}/{len(newton_names)} {what} could not be mapped back to "
          "mjlab names; entities addressing them will not resolve")
  return out


class SceneView(dict):
  def __getitem__(self, key):
    try:
      return super().__getitem__(key)
    except KeyError:
      raise KeyError(f"scene has no entity {key!r}; present: {sorted(self.keys())}") from None


class _ActionManagerView:
  """mjlab's ActionManager surface, limited to what the feature builders read.

  `_last_final_action` and `_last_residual` are fed from here, and those two observation groups are
  exactly the ones that silently went stale in the mjlab eval harness and cost a working checkpoint
  its entire grasp. They are kept explicitly rather than defaulted to zero so that a caller which
  forgets to advance them fails visibly.
  """

  def __init__(self, num_envs: int, dim: int, device, terms: dict | None = None) -> None:
    self.action = torch.zeros(num_envs, dim, device=device)
    self.prev_action = torch.zeros(num_envs, dim, device=device)
    self._terms = terms or {}

  def reset(self, env_ids=None) -> dict:
    """Match mjlab's ActionManager.reset: clear the action history for the envs being reset.

    Without this the previous action survives an episode boundary, and the two observation groups
    built from it start the new episode describing the end of the old one.
    """
    if env_ids is None:
      self.action.zero_()
      self.prev_action.zero_()
    else:
      self.action[env_ids] = 0.0
      self.prev_action[env_ids] = 0.0
    return {}

  def get_term(self, name: str):
    try:
      return self._terms[name]
    except KeyError:
      raise KeyError(f"no action term {name!r}; present: {sorted(self._terms)}") from None

  def advance(self, new_action: torch.Tensor) -> None:
    self.prev_action.copy_(self.action)
    self.action.copy_(new_action.to(self.action.dtype))


class NewtonEnv:
  """The `env` mjlab's builders and action term expect, backed by Newton."""

  # The residual runner publishes its per-step bookkeeping with `setattr(self.env.unwrapped, ...)`,
  # which in this port is NewtonVecEnv -- while every observation, reward and metric term is handed
  # this bridge instead. mjlab has one env object and the two coincide; here they do not, so 27
  # `_residual_*` attributes were written to one object and read from the other. The readers fall
  # back to zeros rather than raising, so the policy trained blind to its own previous action and
  # every residual metric reported 0.0000 while mjlab's climbed.
  #
  # Delegation is limited to that documented prefix: any other unknown attribute still raises,
  # which is what caught this in the first place.
  def __getattr__(self, name: str):
    if name.startswith("_residual_"):
      owner = self.__dict__.get("_owner")
      if owner is not None:
        try:
          return getattr(owner, name)
        except AttributeError:
          pass
    raise AttributeError(
      f"{type(self).__name__} has no attribute {name!r}"
      + ("" if name.startswith("_residual_") else
         "; if a manager needs it, map it explicitly rather than defaulting it to zero"))


  def __init__(self, model, data, num_envs: int, device: str = "cuda:0",
               object_entity: str = "apple", control=None, rename_from=None,
               physics_dt: float = 0.005, decimation: int = 4, solver=None) -> None:
    self.num_envs = int(num_envs)
    self.device = device
    # mjlab's contact-duration accounting integrates against step_dt, so this has to be the CONTROL
    # period (physics dt x decimation), not the physics dt.
    self.physics_dt = float(physics_dt)
    self.step_dt = float(physics_dt) * int(decimation)
    self.cfg = None
    self._solver = solver
    self.mj_model, self.mjw_data, self.control = model, data, control
    scene = SceneView()
    mk = lambda p: NewtonEntity(model, data, p, device, control=control, rename_from=rename_from)
    scene["robot"] = mk("robot/")
    scene[object_entity] = mk(f"{object_entity}/")
    try:
      scene["table"] = mk("table/")
    except ValueError:
      pass          # some scenes have no table entity; mjlab guards for this too
    self.scene = scene
    self.scene.env_origins = torch.zeros((self.num_envs, 3), device=device)
    self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=device)
    self.extras: dict[str, Any] = {}
    self.action_manager: _ActionManagerView | None = None   # set once the action term exists

  def forward(self) -> None:
    """Recompute derived state (xpos, xquat, cvel, subtree_com) from qpos/qvel.

    Writing qpos does not move anything by itself: xpos and xquat are outputs of forward kinematics.
    After a reset -- or any direct joint/root write -- the derived fields still describe the PREVIOUS
    pose, and observations read the derived fields. Measured before this existed: joint_pos matched
    mjlab exactly while root_link_pos_w was off by 1.7 cm and the object's orientation was flatly
    wrong, so the policy opened on a world that did not exist.

    Inside the rollout loop a solver.step() follows every write, which does this implicitly; the reset
    is the case that needs it explicitly.
    """
    import mujoco_warp as _mjw
    if self._solver is None:
      raise RuntimeError("no solver bound; cannot run forward kinematics")
    import warp as _wp
    with _wp.ScopedDevice(self._solver.model.device):
      _mjw.fwd_position(self._solver.mjw_model, self.mjw_data, factorize=False)
      _mjw.fwd_velocity(self._solver.mjw_model, self.mjw_data)

  def bind_action_manager(self, dim: int, terms: dict) -> "_ActionManagerView":
    self.action_manager = _ActionManagerView(self.num_envs, dim, self.device, terms)
    return self.action_manager
