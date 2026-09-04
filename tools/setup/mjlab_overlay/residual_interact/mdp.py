"""MDP terms for SONIC-backed residual interaction RL."""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.apple_eat import mdp as apple_mdp
from mjlab.tasks.apple_eat import object_pool
from mjlab.utils.lab_api.math import quat_apply, quat_conjugate, quat_mul

NUM_BODY = apple_mdp.NUM_BODY
NUM_HAND = apple_mdp.NUM_HAND
ACTION_DIM = NUM_BODY + NUM_HAND
OBS_DIM_NO_TEACHER = apple_mdp.OBS_DIM_NO_TEACHER
ENC_INPUT_DIM = apple_mdp.ENC_INPUT_DIM
BODY_DEC_HIST_DIM = apple_mdp.BODY_DEC_HIST_DIM
HAND_DEC_HIST_DIM = apple_mdp.HAND_DEC_HIST_DIM
SONIC_ENCODER_OBS_DIM = 1762
ASTRA_OBS_DIM = 136
ASTRA_POLICY_ACTION_SCALE = 0.25
RAW_CONTACT_THRESHOLD = 0.03
LIVE_CONTACT_THRESHOLD = 0.06

BODY_29_DOF_NAMES = apple_mdp.BODY_29_DOF_NAMES
HAND_24_DOF_NAMES = apple_mdp.HAND_24_DOF_NAMES
Sonic53ActionCfg = apple_mdp.Sonic53ActionCfg
Sonic53Action = apple_mdp.Sonic53Action
IsaacStyleObs2964 = apple_mdp.IsaacStyleObs2964
ASTRA_DEFAULT_BODY_PKL = np.array(
  [
    -0.10000000149011612,
    0.0,
    0.0,
    0.30000001192092896,
    -0.20000000298023224,
    0.0,
    -0.10000000149011612,
    0.0,
    0.0,
    0.30000001192092896,
    -0.20000000298023224,
    0.0,
    0.0,
    0.0,
    0.0,
    0.20000000298023224,
    0.30000001192092896,
    0.0,
    1.2799999713897705,
    0.0,
    0.0,
    0.0,
    0.20000000298023224,
    -0.30000001192092896,
    0.0,
    1.2799999713897705,
    0.0,
    0.0,
    0.0,
  ],
  dtype=np.float32,
)
ASTRA_ACTION_SCALE_PKL = np.array(
  [
    2.190186023712158,
    1.4026458263397217,
    2.190186023712158,
    1.4026458263397217,
    1.7543092966079712,
    1.7543092966079712,
    2.190186023712158,
    1.4026458263397217,
    2.190186023712158,
    1.4026458263397217,
    1.7543092966079712,
    1.7543092966079712,
    2.190186023712158,
    1.7543092966079712,
    1.7543092966079712,
    1.7543092966079712,
    1.7543092966079712,
    1.7543092966079712,
    1.7543092966079712,
    1.7543092966079712,
    0.2980034649372101,
    0.2980034649372101,
    1.7543092966079712,
    1.7543092966079712,
    1.7543092966079712,
    1.7543092966079712,
    1.7543092966079712,
    0.2980034649372101,
    0.2980034649372101,
  ],
  dtype=np.float32,
)
ASTRA_EFFORT_BODY_PKL = np.array(
  [
    88.0,
    139.0,
    88.0,
    139.0,
    50.0,
    50.0,
    88.0,
    139.0,
    88.0,
    139.0,
    50.0,
    50.0,
    88.0,
    50.0,
    50.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    5.0,
    5.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    5.0,
    5.0,
  ],
  dtype=np.float32,
)
ASTRA_KP_BODY_PKL = ASTRA_EFFORT_BODY_PKL / ASTRA_ACTION_SCALE_PKL
ASTRA_KD_BODY_PKL = np.array(
  [
    2.5578897,
    6.30880165,
    2.5578897,
    6.30880165,
    1.81444573,
    1.81444573,
    2.5578897,
    6.30880165,
    2.5578897,
    6.30880165,
    1.81444573,
    1.81444573,
    2.5578897,
    1.81444573,
    1.81444573,
    0.90722287,
    0.90722287,
    0.90722287,
    0.90722287,
    0.90722287,
    1.06814146,
    1.06814146,
    0.90722287,
    0.90722287,
    0.90722287,
    0.90722287,
    0.90722287,
    1.06814146,
    1.06814146,
  ],
  dtype=np.float32,
)

APPLE_RADIUS = apple_mdp.APPLE_RADIUS
APPLE_MASS = apple_mdp.APPLE_MASS
OBJECT_STL = apple_mdp.OBJECT_STL
USE_OBJECT_MESH = apple_mdp.USE_OBJECT_MESH
APPLE_LINEAR_DAMPING = apple_mdp.APPLE_LINEAR_DAMPING
APPLE_ANGULAR_DAMPING = apple_mdp.APPLE_ANGULAR_DAMPING
TABLE_XY_SIZE = apple_mdp.TABLE_XY_SIZE
TABLE_THICKNESS = apple_mdp.TABLE_THICKNESS
TABLE_OBJECT_GAP = apple_mdp.TABLE_OBJECT_GAP
OBJECT_SPAWN_CLEARANCE = apple_mdp.OBJECT_SPAWN_CLEARANCE
OBJECT_INIT_LIFT = apple_mdp.OBJECT_INIT_LIFT
_GMR_ROOT = Path(os.environ.get("GMR_ROOT", "/home/jiarui/jiarui/GMR"))
_HAND_KIND = os.environ.get("APPLE_HAND_KIND", "xhand").strip().lower()
# The Wuji model is built by tools/grasp_lab/build_g1_wuji.py, which grafts the Wuji gen-1 hand onto
# the same xhand mount frames; both hands share the palm-frame convention (verified: palm axes are
# identity in the mount frame for both), so no corrective rotation is needed.
_ROBOT_XML = (
  _GMR_ROOT / "assets" / "g1_xhand" / "g1_mocap_29dof_with_hands.xml"
  if _HAND_KIND == "xhand"
  else _GMR_ROOT / "assets" / "g1_wuji" / "g1_mocap_29dof_with_wuji_hands.xml"
)

RESIDUAL_FEATURE_GROUPS: tuple[str, ...] = (
  "proprio_history",
  "sonic_obs_or_latent",
  "tracker_action",
  "reference_phase",
  "reference_preview",
  "object_state",
  "hand_object_geometry",
  "object_surface_geometry",
  "object_bps_geometry",
  "omnigrasp_object_context",
  "contact_features",
  "object_future",
  "object_history",
  "object_event_anchor",
  "placement_goal",
  "tracking_error",
  "last_residual",
  "last_final_action",
)
OBSERVATION_GROUPS: tuple[str, ...] = (
  "sonic_encoder_obs",
  "astra_obs",
  *RESIDUAL_FEATURE_GROUPS,
)

# Object-future horizons, in control steps at 50 Hz. Defaults chosen from the measured
# information content of this clip (see _object_future). Override with
# APPLE_OBJ_FUTURE_NEAR / APPLE_OBJ_FUTURE_FAR.
_OBJ_FUTURE_NEAR: tuple[int, ...] = tuple(
  int(v)
  for v in os.environ.get("APPLE_OBJ_FUTURE_NEAR", "1,2,4,8").split(",")
  if v.strip()
)
_OBJ_FUTURE_FAR: tuple[int, ...] = tuple(
  int(v)
  for v in os.environ.get("APPLE_OBJ_FUTURE_FAR", "16,32,64,128,160").split(",")
  if v.strip()
)
_OBJ_HISTORY_LAGS: tuple[int, ...] = tuple(
  sorted(
    {
      0,
      *(
        int(v)
        for v in os.environ.get(
          "APPLE_OBJ_HISTORY_LAGS",
          "0,1,2,4,8,16,32,64,128,256",
        ).split(",")
        if v.strip()
      ),
    }
  )
)
if any(lag < 0 for lag in _OBJ_HISTORY_LAGS):
  raise ValueError(f"APPLE_OBJ_HISTORY_LAGS must be non-negative: {_OBJ_HISTORY_LAGS}")


_TIP_BODY_NAMES_XHAND = (
  "left_hand_thumb_rota_tip",
  "left_hand_index_rota_tip",
  "left_hand_mid_tip",
  "left_hand_ring_tip",
  "left_hand_pinky_tip",
  "right_hand_thumb_rota_tip",
  "right_hand_index_rota_tip",
  "right_hand_mid_tip",
  "right_hand_ring_tip",
  "right_hand_pinky_tip",
)

# Wuji has no fingertip bodies of its own -- each fingerN_tip_link is a geom on fingerN_link4 -- so
# the model builder adds one body per finger at that geom's pose. Order matches the xhand list:
# left thumb..pinky then right thumb..pinky, which the RIGHT = slice(5, 10) convention relies on.
_TIP_BODY_NAMES_WUJI = (
  "left_finger1_tip",
  "left_finger2_tip",
  "left_finger3_tip",
  "left_finger4_tip",
  "left_finger5_tip",
  "right_finger1_tip",
  "right_finger2_tip",
  "right_finger3_tip",
  "right_finger4_tip",
  "right_finger5_tip",
)

# Omnigrasp's pregrasp reward supervises `hand_bodies`: per hand the wrist plus THREE joints of
# every finger (`R_Index1/2/3` ...), 16 bodies a side. The Wuji hand maps onto that one for one:
# the palm link stands in for the wrist, and `link2/3/4` are the three moving joints -- `link1` has
# zero travel (measured, see the finger-IK notes) and `_tip` is a geomless derived frame, so
# neither carries information the other links do not.
_HAND_BODY_NAMES_WUJI = tuple(
  f"{side}_{part}"
  for side in ("left", "right")
  for part in ("palm_link", *(f"finger{i}_link{j}" for i in range(1, 6) for j in (2, 3, 4)))
)
# The xhand set was never enumerated; fall back to its fingertips so a run with that hand keeps
# working rather than raising on a name the model does not have.
_HAND_BODY_NAMES_XHAND = None

_TIP_BODY_NAMES = (
  _TIP_BODY_NAMES_XHAND if _HAND_KIND == "xhand" else _TIP_BODY_NAMES_WUJI
)

# Bodies used for CONTACT sensing, which is not the same set as the ones used for tip positions.
# A contact sensor can only fire on a body that owns collidable geoms: the xhand tip bodies do, but
# the Wuji tip bodies are geomless frames, so keying the sensor on them reports zero contact
# forever. Wuji's distal link fingerN_link4 carries the tip_link collision mesh those frames were
# derived from, so it is the correct contact body -- same count and same left1..5 then right1..5
# order, which the RIGHT = slice(5, 10) convention relies on.
_TIP_CONTACT_BODY_NAMES_WUJI = tuple(
  f"{side}_finger{f}_link4" for side in ("left", "right") for f in range(1, 6)
)
_TIP_CONTACT_BODY_NAMES = (
  _TIP_BODY_NAMES_XHAND if _HAND_KIND == "xhand" else _TIP_CONTACT_BODY_NAMES_WUJI
)

# Broad hand-vs-object sensor. xhand names every hand body *_hand_*, which matches 62 bodies; on
# Wuji that same pattern matches only the 2 mount stubs and misses every finger, so match Wuji's
# palm and finger links explicitly.
_HAND_BODY_EXPR = (
  ".*hand.*" if _HAND_KIND == "xhand" else ".*(palm_link|finger[0-9]+_link[0-9]+)"
)

_ACTION_GROUP_INDICES: dict[str, tuple[int, ...]] = {
  "body": tuple(range(NUM_BODY)),
  "leg": tuple(range(0, 12)),
  "arm": tuple(range(15, NUM_BODY)),
  "hand": tuple(range(NUM_BODY, ACTION_DIM)),
}
_BODY_LINK_GROUPS: dict[str, tuple[int, ...]] = {
  "all": tuple(range(NUM_BODY)),
  "body": tuple(range(NUM_BODY)),
  "lower": tuple(range(0, 15)),
  "lower_body": tuple(range(0, 15)),
  "waist_down": tuple(range(0, 15)),
  "upper": tuple(range(15, NUM_BODY)),
  "upper_body": tuple(range(15, NUM_BODY)),
  "arms": tuple(range(15, NUM_BODY)),
  "ankle": (4, 5, 10, 11),
  "ankles": (4, 5, 10, 11),
  "ankle_wrist": (4, 5, 10, 11, 19, 20, 21, 26, 27, 28),
  "ankles_wrists": (4, 5, 10, 11, 19, 20, 21, 26, 27, 28),
  "lower_wrist": tuple(range(0, 15))
  + tuple(
    idx
    for idx, name in enumerate(BODY_29_DOF_NAMES)
    if name.startswith(("left_wrist_", "right_wrist_"))
  ),
  "lower_plus_wrist": tuple(range(0, 15))
  + tuple(
    idx
    for idx, name in enumerate(BODY_29_DOF_NAMES)
    if name.startswith(("left_wrist_", "right_wrist_"))
  ),
  "left_wrist": tuple(
    idx for idx, name in enumerate(BODY_29_DOF_NAMES) if name.startswith("left_wrist_")
  ),
  "right_wrist": tuple(
    idx for idx, name in enumerate(BODY_29_DOF_NAMES) if name.startswith("right_wrist_")
  ),
}
_ROBOT_ENTITY_CFG = SceneEntityCfg("robot")
_APPLE_ENTITY_CFG = SceneEntityCfg("apple")


def _ref(device: str):
  return apple_mdp._ref(device)


def _tracking_frame(env, n_frames: int, startup_steps: int = 36) -> torch.Tensor:
  return apple_mdp._tracking_frame(env, n_frames, startup_steps=startup_steps)


def _active_after_startup(env, startup_steps: int = 36) -> torch.Tensor:
  return apple_mdp._active_after_startup(env, startup_steps=startup_steps)


def _heading_local_vec(
  root_quat_wxyz: torch.Tensor, vec_w: torch.Tensor
) -> torch.Tensor:
  return apple_mdp._heading_local_vec(root_quat_wxyz, vec_w)


def _heading_local_rot6d(
  root_quat_wxyz: torch.Tensor, quat_wxyz: torch.Tensor
) -> torch.Tensor:
  return apple_mdp._heading_local_rot6d(root_quat_wxyz, quat_wxyz)


def _quat_to_rot6d(q: torch.Tensor) -> torch.Tensor:
  return apple_mdp._quat_to_rot6d(q)


def _quat_local_vec(quat_wxyz: torch.Tensor, vec_w: torch.Tensor) -> torch.Tensor:
  quat = quat_wxyz / quat_wxyz.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
  inv = quat_conjugate(quat)
  if vec_w.ndim == 3:
    batch, count, _ = vec_w.shape
    inv_b = inv.unsqueeze(1).expand(batch, count, 4)
    return quat_apply(inv_b.reshape(-1, 4), vec_w.reshape(-1, 3)).reshape(
      batch, count, 3
    )
  return quat_apply(inv, vec_w)


def _initial_cuboid_scene_poses(ref: dict, env_origins: torch.Tensor, env=None):
  return apple_mdp._initial_cuboid_scene_poses(ref, env_origins, env=env)


def _reference_frame_tensor(env, ref: dict, frame: int | torch.Tensor) -> torch.Tensor:
  n = max(int(ref["n_frames"]), 1)
  lo, hi = apple_mdp._clip_bounds(env, n)
  if isinstance(frame, torch.Tensor):
    frame_t = frame.to(device=env.device, dtype=torch.long)
    if frame_t.ndim == 0:
      # a 0-d tensor is a scalar frame: local to each env's clip
      frame_t = frame_t.expand(env.num_envs) + lo
    # a per-env tensor already carries global rows
  else:
    frame_t = (
      torch.full((env.num_envs,), int(frame), device=env.device, dtype=torch.long) + lo
    )
  return torch.maximum(torch.minimum(frame_t, hi), lo)


def _reference_object_pose_w(env, ref: dict, frame: int | torch.Tensor) -> torch.Tensor:
  frame_t = _reference_frame_tensor(env, ref, frame)
  pos = ref["obj_pos"][frame_t]
  quat = ref["obj_quat"][frame_t]
  pos, quat = _still_until_cf(env, ref, frame_t, pos, quat)
  return torch.cat([pos + env.scene.env_origins, quat], dim=-1)


def _still_until_cf(env, ref: dict, frame_t, pos, quat):
  """Hold the object's reference where it rests until this env reaches its own contact frame.

  Omnigrasp's object target does not move until the table is removed --
  `TrajGenerator3D(..., starting_still_dt=self.table_remove_frame * self.dt)` -- so the window in
  which "the reference lifted the object and the policy did not" simply does not exist there. Ours
  does: measured on R28, the reference starts lifting 20-40 frames BEFORE cf, the real object stays
  on the table, and `object_reference_window` then ends 70-100% of all episodes before the grasp is
  even due. This closes that window.

  After cf the reference is REBASED, `ref'(f) = ref(0) + (ref(f) - ref(cf))`, so the carry starts
  from where the object actually is instead of jumping to wherever the human had already lifted it.
  Rebasing is what makes the seam continuous; freezing alone would put a several-centimetre step at
  cf and trip `og_object_far` on the spot.

  Off unless OBJ_REF_STILL_UNTIL_CF is set.
  """
  if os.environ.get("OBJ_REF_STILL_UNTIL_CF", "").strip() not in ("1", "on", "true"):
    return pos, quat
  from mjlab.tasks.residual_interact import staged_mdp as _smdp

  n = max(int(ref["n_frames"]), 1)
  lo, hi = apple_mdp._clip_bounds(env, n)
  cf_l = _smdp.cf_local(env)
  known = cf_l >= 0
  cf_row = torch.minimum(lo + cf_l.clamp_min(0), hi)
  rest_pos = ref["obj_pos"][lo]
  rest_quat = ref["obj_quat"][lo]
  cf_pos = ref["obj_pos"][cf_row]
  before = known & (frame_t < cf_row)
  after = known & ~before
  pos = torch.where(before.unsqueeze(-1), rest_pos, pos)
  pos = torch.where(after.unsqueeze(-1), rest_pos + (pos - cf_pos), pos)
  quat = torch.where(before.unsqueeze(-1), rest_quat, quat)
  return pos, quat


def _reference_object_pos_w(env, ref: dict, frame: int | torch.Tensor) -> torch.Tensor:
  return _reference_object_pose_w(env, ref, frame)[:, :3]


def _reference_object_pos_local(
  env, ref: dict, frame: int | torch.Tensor
) -> torch.Tensor:
  frame_t = _reference_frame_tensor(env, ref, frame)
  return ref["obj_pos"][frame_t]


def _write_table_pose(table: Entity, table_pose: torch.Tensor, env_ids=None) -> None:
  apple_mdp._write_table_pose(table, table_pose, env_ids=env_ids)


def _clear_residual_action_cache(env, env_ids: torch.Tensor) -> None:
  for attr in (
    "_residual_last_base_action",
    "_residual_last_astra_action_pkl",
    "_residual_last_residual_action",
    "_residual_last_final_action",
    "_residual_body_sample_delta_pre_clip",
    "_residual_body_sample_delta_post_clip",
    "_residual_body_sample_clip_frac",
    "_residual_hand_control_gate",
    "_residual_hand_primitive_close",
    "_residual_hand_primitive_delta",
    "_residual_policy_contact_duration",
    "_residual_policy_hard_lift_duration",
    "_omnigrasp_grab_duration",
    "_omnigrasp_style_prev_tip_err",
    "_omnigrasp_contact_latch",
    "_omnigrasp_leash_contact_latch",
    "_omnigrasp_leash_contact_steps",
  ):
    value = getattr(env, attr, None)
    if isinstance(value, torch.Tensor) and value.shape[:1] == (env.num_envs,):
      with torch.inference_mode():
        value[env_ids] = 0.0


def reset_to_residual_interact_frame0(env, env_ids: torch.Tensor | None):
  env_ids = apple_mdp.resolve_env_ids(env, env_ids)
  apple_mdp.reset_to_apple_eat_frame0(env, env_ids)
  _clear_residual_action_cache(env, env_ids)


def reset_to_residual_interact_curriculum(
  env,
  env_ids: torch.Tensor | None,
  contact_prob: float = 0.5,
  contact_frame_start: int = 420,
  contact_frame_end: int = 470,
  contact_frame_anchor: str = "clip_contact",
  contact_offset_start: int = -30,
  contact_offset_end: int = 30,
  uniform_precontact_prob: float = 0.0,
  precontact_margin_frames: int = 25,
):
  env_ids = apple_mdp.resolve_env_ids(env, env_ids)
  ref = _ref(env.device)
  forced = getattr(env, "_force_reference_start_frame", None)
  if forced is not None:
    apple_mdp.reset_to_apple_eat_frame(env, env_ids, frame=forced)
    _clear_residual_action_cache(env, env_ids)
    return

  from mjlab.tasks.residual_interact import omnigrasp_style_mdp

  n_local = int(ref["n_frames"])
  clip_len = (
    apple_mdp._clip_bounds(env, n_local)[1]
    - apple_mdp._clip_bounds(env, n_local)[0]
    + 1
  )
  clip_len_e = clip_len[env_ids]

  # RSI_ANCHOR_CF=1 moves the start distribution to Omnigrasp's: it samples uniformly over the
  # motion and then clamps to `contact_time - 0.5 s` (humanoid_omnigrasp.py:515), which for a GRAB
  # clip whose contact is 1.4-2.1 s into a 7-24 s recording collapses 85-95% of environments onto
  # that one instant. The grasp is then the only thing left to learn; the walk up to the table is
  # not in the training distribution at all. Ours starts 22-92 frames out and mostly dies on the way.
  if os.environ.get("RSI_ANCHOR_CF", "").strip() in ("1", "on", "true"):
    contact_frame_anchor = "clip_contact"
    contact_offset_start = int(os.environ.get("RSI_CF_OFFSET_START", "-20"))
    contact_offset_end = int(os.environ.get("RSI_CF_OFFSET_END", "-10"))

  frame = torch.zeros(env_ids.numel(), dtype=torch.long, device=env.device)
  draw = torch.rand(env_ids.numel(), device=env.device)
  contact_mask = draw < float(contact_prob)
  if contact_mask.any():
    if str(contact_frame_anchor) == "clip_contact":
      # Per-clip window: each clip makes contact at its own frame, so a shared absolute
      # window lands in a different phase of every clip. `contact_frame` is per clip and
      # LOCAL, computed over that clip's own span.
      cf_per_clip = omnigrasp_style_mdp._contact_reference(env.device)["contact_frame"]
      cf_per_clip = torch.as_tensor(
        cf_per_clip, device=env.device, dtype=torch.long
      ).flatten()
      cf = cf_per_clip[apple_mdp._clip_id(env)[env_ids]]
      lo_e = (cf + int(contact_offset_start)).clamp(min=0)
      hi_e = cf + int(contact_offset_end)
    else:
      lo_e = torch.full_like(clip_len_e, max(0, int(contact_frame_start)))
      hi_e = torch.full_like(clip_len_e, max(0, int(contact_frame_end)))
    # never past this clip's own last frame
    hi_e = torch.minimum(hi_e, clip_len_e - 1)
    lo_e = torch.minimum(lo_e, hi_e)
    span = (hi_e - lo_e + 1).clamp(min=1)
    draw_f = (torch.rand(env_ids.numel(), device=env.device) * span.float()).long()
    picked = (lo_e + draw_f).clamp(max=clip_len_e - 1)
    frame = torch.where(contact_mask, picked, frame)
  uniform_mask = (~contact_mask) & (
    draw < float(contact_prob) + float(uniform_precontact_prob)
  )
  if uniform_mask.any():
    # Omnigrasp RSI: uniform start at least `precontact_margin_frames` before
    # the reference contact frame, so episodes begin approaching the object.
    from mjlab.tasks.residual_interact import omnigrasp_style_mdp

    # per clip: int() here collapsed a per-clip tensor onto clip 0's contact frame
    cfu = torch.as_tensor(
      omnigrasp_style_mdp._contact_reference(env.device)["contact_frame"],
      device=env.device,
      dtype=torch.long,
    ).flatten()[apple_mdp._clip_id(env)[env_ids]]
    hi_u = (cfu - int(precontact_margin_frames)).clamp(min=0)
    hi_u = torch.minimum(hi_u, clip_len_e - 1)
    picked_u = (
      torch.rand(env_ids.numel(), device=env.device) * (hi_u + 1).float()
    ).long()
    picked_u = torch.minimum(picked_u, hi_u)
    frame = torch.where(uniform_mask, picked_u, frame)
  apple_mdp.reset_to_apple_eat_frame(env, env_ids, frame=frame)
  _clear_residual_action_cache(env, env_ids)


def not_fallen(env, asset_cfg: SceneEntityCfg = _ROBOT_ENTITY_CFG) -> torch.Tensor:
  return apple_mdp.not_fallen(env, asset_cfg=asset_cfg)


def object_drift_termination(
  env,
  object_cfg: SceneEntityCfg = _APPLE_ENTITY_CFG,
  threshold: float = 0.75,
) -> torch.Tensor:
  if object_cfg.name != "apple":
    return apple_mdp.object_drift_termination(
      env, object_cfg=object_cfg, threshold=threshold
    )
  active = _active_after_startup(env).bool()
  return (_object_drift(env) > float(threshold)) & active


def object_reference_window_termination(
  env,
  threshold: float = 0.05,
  window: int = 20,
  activate_after_frame: int = 0,
  grace_steps: int = 36,
) -> torch.Tensor:
  """Terminate when the object is nowhere near its reference path.

  Passes if the live object is within `threshold` of the reference object position
  at ANY frame in [f - window, f + window], where f is the current tracking frame.
  Tolerating a +-window offset scores path-following rather than exact timing, so a
  policy that grasps slightly early or late is not punished, while one that leaves
  the apple on the table is.
  """
  # OBJ_REF_WINDOW_ENABLE=0 turns this off without touching the termination list, so the run stays
  # index-comparable with one that keeps it. Measured on R28: at 0.05 m this term takes 70-100% of
  # all resets, against 0.1-13% for og_object_far at Omnigrasp's own 0.12 m, and it fires BEFORE the
  # grasp is due because the reference lifts the object first.
  if os.environ.get("OBJ_REF_WINDOW_ENABLE", "1").strip() in ("0", "off", "false"):
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  ref = _ref(env.device)
  n = int(ref["n_frames"])
  obj: Entity = object_pool.active(env)
  live = obj.data.root_link_pos_w

  frame = _tracking_frame(env, n)
  frame_i = frame.round().long() if frame.dtype.is_floating_point else frame.long()

  # Clamp inside THIS env's clip band. `n - 1` is clip 0's last row, so under MIX every
  # clip-1 env was compared against clip 0's final frame -- the dominant terminator was
  # scoring the stapler against the apple's last pose.
  _lo, _hi = apple_mdp._clip_bounds(env, n)
  w = max(int(window), 0)
  best = None
  for off in range(-w, w + 1):
    cand = torch.minimum(torch.maximum(frame_i + off, _lo), _hi)
    d = (live - _reference_object_pos_w(env, ref, cand)).norm(dim=-1)
    best = d if best is None else torch.minimum(best, d)

  active = (env.episode_length_buf >= int(grace_steps)) & (
    (frame_i - _lo) >= int(activate_after_frame)
  )
  value = (best > float(threshold)) & active

  _safe_log(env, "Metric/obj_ref_window_dist", best)
  _safe_log(env, "Metric/obj_ref_window_active", active.float())
  _safe_log(env, "Metric/obj_ref_window_fail", value.float())
  return value


def _wrist_dist_at_frame(env, frame: torch.Tensor) -> torch.Tensor:
  """max(left, right) wrist distance to the reference wrist AT a given frame.

  `_body_link_dist_mean_for_group` only ever evaluates the current tracking frame,
  so the windowed check needs this frame-parameterised form.
  """
  robot: Entity = env.scene["robot"]
  body_pos_ref_all, _ = _reference_body_xyz_cache(env.device)
  body_ids, ref_ids = _body_xyz_tracking_ids(env)
  live = robot.data.body_link_pos_w[:, body_ids]
  target = body_pos_ref_all[frame][:, ref_ids] + env.scene.env_origins[:, None, :]
  dist = (live - target).norm(dim=-1)
  out = []
  for group in ("left_wrist", "right_wrist"):
    mask = _body_link_group_mask(env, group)
    out.append(dist[:, mask].mean(dim=-1))
  return torch.maximum(out[0], out[1])


def wrist_target_far_termination(
  env,
  active_after_steps: int = 100,
  threshold: float = 0.20,
  window: int = 0,
  pre_cf_only: bool = True,
) -> torch.Tensor:
  """Kill an episode whose wrist has fallen more than `threshold` from its per-frame reference.

  Restricted to the APPROACH by default. Before cf the whole task is getting the hand to the
  grasp, and a wrist that has lost its reference by 20 cm is not going to arrive; after cf the
  task is the OBJECT, and the wrist is expected to leave the reference trajectory in whatever way
  keeps the object held. Killing on wrist error there would punish a carry that is succeeding on
  the only terms that matter.

  Two clocks meet in this gate and both are needed. `active_after_steps` is on the CONTROL-STEP
  clock and counts the 36 startup steps the reference is held for, so it expresses "how much
  reference motion has elapsed" -- start_frame cancels out of it. The cf test is on the
  REFERENCE-FRAME clock. `active_after_steps` alone cannot express a phase relative to cf,
  because RSI draws start_frame uniformly from 0-50: at the old default of 100 this opened 31
  frames BEFORE cf for an env starting at frame 0 and 19 frames AFTER cf for one starting at 50.
  """
  if int(window) > 0:
    ref = _ref(env.device)
    n = int(ref["n_frames"])
    f = _tracking_frame(env, n)
    f_i = f.round().long() if f.dtype.is_floating_point else f.long()
    w = int(window)
    dist = None
    for off in range(-w, w + 1):
      d = _wrist_dist_at_frame(env, (f_i + off).clamp(0, n - 1))
      dist = d if dist is None else torch.minimum(dist, d)
    left = right = dist
  else:
    left = _body_link_dist_mean_for_group(env, "left_wrist", use_tracking_weights=False)
    right = _body_link_dist_mean_for_group(
      env, "right_wrist", use_tracking_weights=False
    )
  dist = torch.maximum(left, right)
  active = env.episode_length_buf >= int(active_after_steps)
  if bool(pre_cf_only):
    # Imported here, not at module scope: staged_mdp imports this module.
    from mjlab.tasks.residual_interact import staged_mdp as _smdp

    active = active & _smdp.before_cf(env)
  value = (dist > float(threshold)) & active
  active_f = active.float()
  threshold_f = torch.full_like(dist, float(threshold))
  _safe_log(env, "Metric/wrist_target_far_left_dist", left)
  _safe_log(env, "Metric/wrist_target_far_right_dist", right)
  _safe_log(env, "Metric/wrist_target_far_dist", dist)
  _safe_log(env, "Metric/wrist_target_far_threshold", threshold_f)
  _safe_log(env, "Metric/wrist_target_far_active", active_f)
  _safe_log(env, "Metric/wrist_target_far_candidate", value.float())
  return value


class NoProgressTermination(apple_mdp.NoProgressTermination):
  pass


def _zeros(env, dim: int) -> torch.Tensor:
  return torch.zeros(env.num_envs, dim, device=env.device)


def _last_base_action(env) -> torch.Tensor:
  value = getattr(env, "_residual_last_base_action", None)
  if value is None:
    return _zeros(env, ACTION_DIM)
  return value.to(env.device)


def _last_residual_action(env) -> torch.Tensor:
  value = getattr(env, "_residual_last_residual_action", None)
  if value is None:
    return _zeros(env, ACTION_DIM)
  return value.to(env.device)


def _last_raw_residual_action(env) -> torch.Tensor:
  value = getattr(env, "_residual_last_raw_residual_action", None)
  if value is None:
    return _last_residual_action(env)
  return value.to(env.device)


def _previous_residual_delta(env) -> torch.Tensor:
  value = getattr(env, "_residual_previous_residual_delta", None)
  if value is None:
    return _zeros(env, ACTION_DIM)
  return value.to(env.device)


def _last_final_action(env) -> torch.Tensor:
  value = getattr(env, "_residual_last_final_action", None)
  if value is None:
    return env.action_manager.prev_action.clone()
  return value.to(env.device)


def _last_action_mean(env) -> torch.Tensor:
  value = getattr(env, "_residual_last_action_mean", None)
  if value is None:
    return _last_final_action(env)
  return value.to(env.device)


def _hand_sample_delta_pre_clip(env) -> torch.Tensor:
  value = getattr(env, "_residual_hand_sample_delta_pre_clip", None)
  if value is None:
    return _zeros(env, NUM_HAND)
  return value.to(env.device)


def _hand_sample_delta_post_clip(env) -> torch.Tensor:
  value = getattr(env, "_residual_hand_sample_delta_post_clip", None)
  if value is None:
    return _zeros(env, NUM_HAND)
  return value.to(env.device)


def _hand_sample_clip_frac(env) -> torch.Tensor:
  value = getattr(env, "_residual_hand_sample_clip_frac", None)
  if value is None:
    return torch.zeros(env.num_envs, device=env.device)
  return value.to(env.device)


def _body_sample_delta_pre_clip(env) -> torch.Tensor:
  value = getattr(env, "_residual_body_sample_delta_pre_clip", None)
  if value is None:
    return _zeros(env, NUM_BODY)
  return value.to(env.device)


def _body_sample_delta_post_clip(env) -> torch.Tensor:
  value = getattr(env, "_residual_body_sample_delta_post_clip", None)
  if value is None:
    return _zeros(env, NUM_BODY)
  return value.to(env.device)


def _body_sample_clip_frac(env) -> torch.Tensor:
  value = getattr(env, "_residual_body_sample_clip_frac", None)
  if value is None:
    return torch.zeros(env.num_envs, device=env.device)
  return value.to(env.device)


def _hand_action_std_mean(env) -> torch.Tensor:
  value = getattr(env, "_residual_hand_action_std_mean", None)
  if value is None:
    return torch.zeros(env.num_envs, device=env.device)
  value = value.to(env.device)
  if value.numel() == 1:
    return value.expand(env.num_envs)
  return value.reshape(-1)


def _hand_primitive_close(env) -> torch.Tensor:
  value = getattr(env, "_residual_hand_primitive_close", None)
  if value is None:
    return torch.zeros(env.num_envs, 2, device=env.device)
  value = value.to(env.device)
  if value.ndim != 2 or value.shape[-1] != 2:
    return torch.zeros(env.num_envs, 2, device=env.device)
  return value


def _hand_primitive_delta(env) -> torch.Tensor:
  value = getattr(env, "_residual_hand_primitive_delta", None)
  if value is None:
    return _zeros(env, NUM_HAND)
  value = value.to(env.device)
  if value.ndim != 2 or value.shape[-1] != NUM_HAND:
    return _zeros(env, NUM_HAND)
  return value


def _last_token_delta(env) -> torch.Tensor:
  value = getattr(env, "_residual_last_token_delta", None)
  if value is None:
    return _zeros(env, 64)
  return value.to(env.device)


def _previous_token_delta(env) -> torch.Tensor:
  value = getattr(env, "_residual_previous_token_delta", None)
  if value is None:
    return _zeros(env, 64)
  return value.to(env.device)


def _last_decoder_body_delta(env) -> torch.Tensor:
  value = getattr(env, "_residual_last_decoder_body_delta", None)
  if value is None:
    return _zeros(env, NUM_BODY)
  return value.to(env.device)


def _hand_control_gate(env) -> torch.Tensor:
  value = getattr(env, "_residual_hand_control_gate", None)
  if value is None:
    return torch.ones(env.num_envs, device=env.device)
  value = value.to(env.device)
  if value.ndim > 1:
    value = value.squeeze(-1)
  return value


def _action_group_ids(env, group: str) -> torch.Tensor:
  attr = f"_residual_{group}_action_ids"
  value = getattr(env, attr, None)
  if value is None:
    value = torch.tensor(
      _ACTION_GROUP_INDICES[group],
      device=env.device,
      dtype=torch.long,
    )
    setattr(env, attr, value)
  return value.to(env.device)


def _action_group(action: torch.Tensor, env, group: str) -> torch.Tensor:
  return action.index_select(1, _action_group_ids(env, group))


def _action_group_norm(action: torch.Tensor, env, group: str) -> torch.Tensor:
  return _action_group(action, env, group).norm(dim=-1)


def _residual_action_mask(env) -> torch.Tensor:
  value = getattr(env, "_residual_action_mask", None)
  if isinstance(value, torch.Tensor) and value.numel() == ACTION_DIM:
    return value.to(device=env.device)
  return torch.ones(ACTION_DIM, device=env.device)


def _clip_fraction(
  action: torch.Tensor,
  threshold: float | None,
  mask: torch.Tensor | None = None,
) -> torch.Tensor:
  if threshold is None or float(threshold) <= 0.0:
    return torch.zeros(action.shape[0], device=action.device)
  values = action
  if mask is not None:
    active = mask.to(device=action.device, dtype=torch.bool)
    if not torch.any(active):
      return torch.zeros(action.shape[0], device=action.device)
    values = values[:, active]
  eps = max(float(threshold) * 1e-4, 1e-6)
  return (values.abs() >= float(threshold) - eps).float().mean(dim=-1)


# Which metrics are worth splitting per clip in a mix run. Deliberately not everything: _safe_log
# runs ~50 times a step, and a masked mean per clip per key would add hundreds of small reductions
# to a step that is already 70% env-count-independent overhead. These are the ones that answer
# "which half is stuck" -- the outcome metrics plus the object terms that drive them.
_PER_CLIP_LOG_KEYS = frozenset(
  {
    "PhaseA/lift_success",
    "PhaseA/lift_duration_s",
    "PhaseA/live_contact_006",
    "PhaseA/object_mpjpe_mm",
    "PhaseA/sequence_success",
    "Stage/physical_contact",
    "Metric/hand_to_obj_dist",
    "Metric/hand_to_obj_under_005_frac",
    "OmniGraspReward/object_tracking_gated",
    "OmniGraspReward/contact_bonus",
    "OmniGraspReward/pregrasp",
  }
)


def _per_clip_log_ids(env):
  """Round-robin clip assignment, or None when the run has one clip and a split says nothing.

  Cached on the env: _clip_id caches its own tensor, but reading `ids.max()` to learn the clip
  count is a device sync, and paying that on every _safe_log call would cost more than the split.
  """
  cached = getattr(env, "_per_clip_log_cache", None)
  if cached is None:
    try:
      ids = apple_mdp._clip_id(env)
      n_clips = int(ids.max().item()) + 1
    except Exception:
      cached = (None, 1)
    else:
      cached = (ids, n_clips)
    env._per_clip_log_cache = cached
  ids, n_clips = cached
  return ids if n_clips > 1 else None


def _safe_log(env, key: str, value: torch.Tensor) -> None:
  log = env.extras.setdefault("log", {})
  if getattr(env, "_eval_detailed_log", False):
    log[key] = torch.nan_to_num(value.detach(), nan=0.0)
    return
  # Original order: mean FIRST, then nan_to_num. Kept exactly, so this metric stays bit-comparable
  # with every curve logged before per-clip logging existed. The per-clip split below uses the
  # per-element cleaned tensor instead, which is the only form a per-clip mask can use.
  log[key] = torch.nan_to_num(value.detach().mean(), nan=0.0)
  if key not in _PER_CLIP_LOG_KEYS:
    return
  clean = torch.nan_to_num(value.detach(), nan=0.0)
  clip_id = _per_clip_log_ids(env)
  if clip_id is None or clean.ndim == 0 or clean.shape[0] != clip_id.shape[0]:
    # Some callers pass an already-reduced scalar. Masking that would log the same number under
    # both clips and read as "the two halves agree", which is the opposite of what this is for.
    return
  ids = clip_id.to(clean.device)
  n_clips = int(getattr(env, "_per_clip_log_cache", (None, 1))[1])
  for c in range(n_clips):
    mask = ids == c
    if bool(mask.any()):
      log[f"{key}/clip{c}"] = clean[mask].mean()


def _quat_yaw(q: torch.Tensor) -> torch.Tensor:
  q = q / q.norm(dim=-1, keepdim=True).clamp_min(1.0e-9)
  w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
  return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _quat_to_rotvec(q: torch.Tensor) -> torch.Tensor:
  q = q / q.norm(dim=-1, keepdim=True).clamp_min(1.0e-9)
  q = torch.where(q[:, 0:1] < 0.0, -q, q)
  xyz = q[:, 1:4]
  sin_half = xyz.norm(dim=-1, keepdim=True)
  half_angle = torch.atan2(sin_half, q[:, 0:1])
  return 2.0 * half_angle * xyz / sin_half.clamp_min(1.0e-9)


def _wrap_pi(x: torch.Tensor) -> torch.Tensor:
  return torch.atan2(torch.sin(x), torch.cos(x))


def _astra_last_action_pkl(env) -> torch.Tensor:
  value = getattr(env, "_residual_last_astra_action_pkl", None)
  if isinstance(value, torch.Tensor) and value.shape == (env.num_envs, NUM_BODY):
    return value.to(env.device)
  return torch.zeros(env.num_envs, NUM_BODY, device=env.device)


def _astra_ref_root_cvel(env, ref: dict) -> torch.Tensor:
  value = getattr(env, "_astra_ref_root_cvel", None)
  expected_shape = (int(ref["n_frames"]), 6)
  if isinstance(value, torch.Tensor) and tuple(value.shape) == expected_shape:
    return value.to(env.device)
  fps = 50.0
  lin_w = torch.zeros_like(ref["root_pos"])
  ang_w = torch.zeros_like(ref["root_pos"])
  if int(ref["n_frames"]) > 1:
    lin_w[1:] = (ref["root_pos"][1:] - ref["root_pos"][:-1]) * fps
    lin_w[0] = lin_w[1]
    q_delta = quat_mul(ref["root_rot"][1:], quat_conjugate(ref["root_rot"][:-1]))
    ang_w[1:] = _quat_to_rotvec(q_delta) * fps
    ang_w[0] = ang_w[1]
  quat = ref["root_rot"]
  cvel = torch.cat(
    [
      _heading_local_vec(quat, ang_w),
      _heading_local_vec(quat, lin_w),
    ],
    dim=-1,
  )
  env._astra_ref_root_cvel = cvel
  return cvel


class AstraObs136:
  """Humanoid-GPT/ASTRA non-privileged 136D body-tracker observation."""

  def __init__(self, cfg, env):
    del cfg
    self.device = env.device
    self.num_envs = env.num_envs
    self.body_joint_ids: torch.Tensor | None = None
    self.astra_default = torch.tensor(
      ASTRA_DEFAULT_BODY_PKL, device=self.device, dtype=torch.float32
    )
    self.gravity_w = torch.tensor(
      [0.0, 0.0, -1.0], device=self.device, dtype=torch.float32
    ).view(1, 3)

  def _resolve(self, env) -> None:
    if self.body_joint_ids is not None:
      return
    robot: Entity = env.scene["robot"]
    ids, _ = robot.find_joints(BODY_29_DOF_NAMES, preserve_order=True)
    self.body_joint_ids = torch.tensor(ids, device=env.device, dtype=torch.long)

  def reset(self, env_ids=None):
    del env_ids

  def __call__(self, env, **kwargs) -> torch.Tensor:
    del kwargs
    self._resolve(env)
    assert self.body_joint_ids is not None
    robot: Entity = env.scene["robot"]
    ref = _ref(env.device)
    frame_curr = _tracking_frame(env, ref["n_frames"])
    frame_next = torch.minimum(
      frame_curr + 1, apple_mdp._clip_bounds(env, int(ref["n_frames"]))[1]
    )

    root_pose = robot.data.root_link_pose_w
    root_quat = root_pose[:, 3:7]
    ref_root_pos_curr_w = ref["root_pos"][frame_curr] + env.scene.env_origins
    ref_root_quat_curr = ref["root_rot"][frame_curr]
    ref_root_pos_next_w = ref["root_pos"][frame_next] + env.scene.env_origins
    ref_root_quat_next = ref["root_rot"][frame_next]

    q = robot.data.joint_pos[:, self.body_joint_ids]
    qd = robot.data.joint_vel[:, self.body_joint_ids]
    gyro = robot.data.root_link_ang_vel_b
    gravity = quat_apply(
      quat_conjugate(root_quat),
      self.gravity_w.expand(env.num_envs, 3),
    )
    ref_gravity = quat_apply(
      quat_conjugate(ref_root_quat_next),
      self.gravity_w.expand(env.num_envs, 3),
    )

    ref_cvel = _astra_ref_root_cvel(env, ref)[frame_next]

    yaw_d = _wrap_pi(_quat_yaw(ref_root_quat_curr) - _quat_yaw(root_quat))
    yaw_cmd = torch.stack([torch.cos(yaw_d), torch.sin(yaw_d)], dim=-1)
    xy_cmd = _heading_local_vec(
      root_quat,
      torch.cat(
        [
          ref_root_pos_curr_w[:, :2] - root_pose[:, :2],
          torch.zeros(env.num_envs, 1, device=env.device),
        ],
        dim=-1,
      ),
    )[:, :2]

    obs = torch.cat(
      [
        gyro,
        gravity,
        q - self.astra_default.unsqueeze(0),
        qd,
        _astra_last_action_pkl(env),
        ref["dof_pos"][frame_next, :NUM_BODY] - self.astra_default.unsqueeze(0),
        ref_root_pos_next_w[:, 2:3] - env.scene.env_origins[:, 2:3],
        ref_gravity,
        ref_cvel,
        yaw_cmd,
        xy_cmd,
      ],
      dim=-1,
    )
    if obs.shape[-1] != ASTRA_OBS_DIM:
      raise RuntimeError(f"AstraObs136 produced shape {tuple(obs.shape)}.")
    return torch.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)


def _cached_ids(env, attr: str, resolver) -> torch.Tensor:
  value = getattr(env, attr, None)
  if value is None:
    value = torch.tensor(resolver(), device=env.device, dtype=torch.long)
    setattr(env, attr, value)
  return value.to(env.device)


def _body_joint_ids(env) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  return _cached_ids(
    env,
    "_residual_body_joint_ids",
    lambda: robot.find_joints(BODY_29_DOF_NAMES, preserve_order=True)[0],
  )


def _hand_joint_ids(env) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  return _cached_ids(
    env,
    "_residual_hand_joint_ids",
    lambda: robot.find_joints(HAND_24_DOF_NAMES, preserve_order=True)[0],
  )


def _tip_body_ids(env) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  return _cached_ids(
    env,
    "_residual_tip_body_ids",
    lambda: robot.find_bodies(_TIP_BODY_NAMES, preserve_order=True)[0],
  )


def _hand_body_names() -> tuple[str, ...]:
  names = _HAND_BODY_NAMES_XHAND if _HAND_KIND == "xhand" else _HAND_BODY_NAMES_WUJI
  return names or _TIP_BODY_NAMES


def _hand_body_ids(env) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  return _cached_ids(
    env,
    "_residual_hand_body_ids",
    lambda: robot.find_bodies(_hand_body_names(), preserve_order=True)[0],
  )


def _tip_distances(env) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  obj: Entity = object_pool.active(env)
  tip_pos = robot.data.body_link_pos_w[:, _tip_body_ids(env)]
  return (tip_pos - obj.data.root_link_pos_w.unsqueeze(1)).norm(dim=-1)


def _object_drift(env) -> torch.Tensor:
  ref = _ref(env.device)
  obj: Entity = object_pool.active(env)
  frame = _tracking_frame(env, ref["n_frames"])
  ref_pos = _reference_object_pos_w(env, ref, frame)
  return (obj.data.root_link_pos_w - ref_pos).norm(dim=-1)


def _drift_reward_gate(
  env,
  target: float = 0.0,
  margin: float = 0.10,
  power: float = 1.0,
) -> torch.Tensor:
  if float(target) <= 0.0:
    return torch.ones(env.num_envs, device=env.device)
  drift = _object_drift(env)
  if float(margin) <= 0.0:
    gate = (drift <= float(target)).float()
  else:
    width = max(float(margin), 1.0e-6)
    gate = ((float(target) + width - drift) / width).clamp(min=0.0, max=1.0)
  exponent = max(float(power), 1.0e-6)
  if abs(exponent - 1.0) > 1.0e-6:
    gate = gate.pow(exponent)
  _safe_log(env, "ResidualMetric/drift_gate", gate)
  return gate


def _physical_contact(
  env,
  force_threshold: float = 1.0,
  history_force_epsilon: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Return contact, grasp-like contact, and close-force flags from the apple sensor.

  Falls back to zeros if a run directory was produced before the sensor existed.
  """
  try:
    sensor = object_pool.active_sensor(env, "hand_apple_contact")
    data = sensor.data
    if data.found is None:
      raise RuntimeError("hand_apple_contact has no found data")
    found = data.found > 0
    contact = found.any(dim=-1)
    grasp_contact = found.sum(dim=-1) >= 2
    force_mag = None
    if data.force is not None:
      force_mag = data.force.norm(dim=-1)
    if data.force_history is not None:
      history_force_mag = data.force_history.norm(dim=-1).amax(dim=-1)
      force_mag = (
        history_force_mag
        if force_mag is None
        else torch.maximum(force_mag, history_force_mag)
      )
    if force_mag is None:
      force_close = contact
    else:
      recent_contact = force_mag > float(history_force_epsilon)
      contact = contact | recent_contact.any(dim=-1)
      grasp_contact = grasp_contact | (recent_contact.sum(dim=-1) >= 2)
      force_close = force_mag.amax(dim=-1) > float(force_threshold)
    return contact.float(), grasp_contact.float(), force_close.float()
  except Exception:
    zeros = torch.zeros(env.num_envs, device=env.device)
    return zeros, zeros, zeros


def _primary_contact_from_history(
  env,
  history_force_epsilon: float = 1.0e-6,
) -> torch.Tensor:
  """Return per-fingertip contact flags, including force history in a policy step."""
  try:
    sensor = object_pool.active_sensor(env, "hand_apple_contact")
    data = sensor.data
    n_primary = len(_TIP_BODY_NAMES)
    contact = torch.zeros(env.num_envs, n_primary, dtype=torch.bool, device=env.device)

    def reduce_slots(value: torch.Tensor) -> torch.Tensor:
      value = value.to(env.device)
      if value.shape[-1] == n_primary:
        return value.bool()
      num_slots = max(int(getattr(sensor.cfg, "num_slots", 1)), 1)
      if value.shape[-1] == n_primary * num_slots:
        return value.view(env.num_envs, n_primary, num_slots).any(dim=-1)
      if value.shape[-1] > n_primary:
        return value[:, :n_primary].bool()
      return torch.nn.functional.pad(value.bool(), (0, n_primary - value.shape[-1]))

    if data.found is not None:
      contact |= reduce_slots(data.found > 0)
    if data.force is not None:
      contact |= reduce_slots(data.force.norm(dim=-1) > float(history_force_epsilon))
    if data.force_history is not None:
      history_contact = data.force_history.norm(dim=-1).amax(dim=-1)
      contact |= reduce_slots(history_contact > float(history_force_epsilon))
    return contact
  except Exception:
    return torch.zeros(
      env.num_envs,
      len(_TIP_BODY_NAMES),
      dtype=torch.bool,
      device=env.device,
    )


def _contact_vector_to_primary(
  env,
  sensor: ContactSensor,
  value: torch.Tensor,
  n_primary: int,
) -> torch.Tensor:
  """Reduce per-slot contact vectors to one max-magnitude vector per primary."""
  value = value.to(env.device)
  if value.shape[-2] == n_primary:
    return value
  num_slots = max(int(getattr(sensor.cfg, "num_slots", 1)), 1)
  if value.shape[-2] == n_primary * num_slots:
    slots = value.view(env.num_envs, n_primary, num_slots, 3)
    idx = slots.norm(dim=-1).argmax(dim=-1)
    gather_idx = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)
    return slots.gather(2, gather_idx).squeeze(2)
  if value.shape[-2] > n_primary:
    return value[:, :n_primary]
  pad = torch.zeros(
    env.num_envs,
    n_primary - value.shape[-2],
    3,
    device=env.device,
    dtype=value.dtype,
  )
  return torch.cat([value, pad], dim=1)


def _contact_history_vector_to_primary(
  env,
  sensor: ContactSensor,
  value: torch.Tensor,
  n_primary: int,
) -> torch.Tensor:
  """Reduce force history to one max-magnitude vector per primary."""
  value = value.to(env.device)
  if value.ndim != 4:
    return torch.zeros(env.num_envs, n_primary, 3, device=env.device)
  h = value.shape[-2]
  if value.shape[1] == n_primary:
    candidates = value
  else:
    num_slots = max(int(getattr(sensor.cfg, "num_slots", 1)), 1)
    if value.shape[1] == n_primary * num_slots:
      candidates = value.view(env.num_envs, n_primary, num_slots * h, 3)
    elif value.shape[1] > n_primary:
      candidates = value[:, :n_primary]
    else:
      pad = torch.zeros(
        env.num_envs,
        n_primary - value.shape[1],
        h,
        3,
        device=env.device,
        dtype=value.dtype,
      )
      candidates = torch.cat([value, pad], dim=1)
  idx = candidates.norm(dim=-1).argmax(dim=-1)
  gather_idx = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)
  return candidates.gather(2, gather_idx).squeeze(2)


def _primary_contact_force_features(env) -> tuple[torch.Tensor, torch.Tensor]:
  """Return per-fingertip contact flags and log-scaled local force vectors."""
  n_primary = len(_TIP_BODY_NAMES)
  flags = _primary_contact_from_history(env).float()
  force = torch.zeros(env.num_envs, n_primary, 3, device=env.device)
  try:
    sensor = object_pool.active_sensor(env, "hand_apple_contact")
    data = sensor.data
    if data.force is not None:
      force = _contact_vector_to_primary(env, sensor, data.force, n_primary)
    if data.force_history is not None:
      hist_force = _contact_history_vector_to_primary(
        env, sensor, data.force_history, n_primary
      )
      use_hist = hist_force.norm(dim=-1) > force.norm(dim=-1)
      force = torch.where(use_hist.unsqueeze(-1), hist_force, force)
  except Exception:
    pass

  force = torch.where(flags.bool().unsqueeze(-1), force, torch.zeros_like(force))
  force_norm = force.norm(dim=-1, keepdim=True)
  force_scaled = torch.where(
    force_norm > 1.0e-6,
    force / force_norm.clamp_min(1.0e-6) * torch.log1p(force_norm),
    torch.zeros_like(force),
  )
  robot: Entity = env.scene["robot"]
  root_quat = robot.data.root_link_pose_w[:, 3:7]
  force_local = _heading_local_vec(root_quat, force_scaled)
  return flags, force_local


def _contact_flags_from_sensor(
  env,
  sensor_name: str,
  history_force_epsilon: float = 1.0e-6,
) -> torch.Tensor:
  """Return per-primary contact flags for a named contact sensor."""
  try:
    sensor = object_pool.active_sensor(env, sensor_name)
    data = sensor.data
    n_primary = len(sensor.primary_names)
    contact = torch.zeros(env.num_envs, n_primary, dtype=torch.bool, device=env.device)

    def reduce_slots(value: torch.Tensor) -> torch.Tensor:
      value = value.to(env.device)
      if value.shape[-1] == n_primary:
        return value.bool()
      num_slots = max(int(getattr(sensor.cfg, "num_slots", 1)), 1)
      if value.shape[-1] == n_primary * num_slots:
        return value.view(env.num_envs, n_primary, num_slots).any(dim=-1)
      if value.shape[-1] > n_primary:
        return value[:, :n_primary].bool()
      return torch.nn.functional.pad(value.bool(), (0, n_primary - value.shape[-1]))

    if data.found is not None:
      contact |= reduce_slots(data.found > 0)
    if data.force is not None:
      contact |= reduce_slots(data.force.norm(dim=-1) > float(history_force_epsilon))
    if data.force_history is not None:
      history_contact = data.force_history.norm(dim=-1).amax(dim=-1)
      contact |= reduce_slots(history_contact > float(history_force_epsilon))
    return contact
  except Exception:
    return torch.zeros(env.num_envs, 0, dtype=torch.bool, device=env.device)


def _hand_body_contact_from_history(env) -> torch.Tensor:
  return _contact_flags_from_sensor(env, "hand_body_apple_contact")


def _hand_body_contact(env) -> torch.Tensor:
  contact = _hand_body_contact_from_history(env)
  if contact.numel() == 0:
    return torch.zeros(env.num_envs, device=env.device)
  return contact.any(dim=-1).float()


def _non_tip_hand_body_contact(env) -> torch.Tensor:
  contact = _hand_body_contact_from_history(env)
  if contact.numel() == 0:
    return torch.zeros(env.num_envs, device=env.device)
  try:
    sensor = object_pool.active_sensor(env, "hand_body_apple_contact")
    non_tip = torch.tensor(
      [name not in _TIP_BODY_NAMES for name in sensor.primary_names],
      dtype=torch.bool,
      device=env.device,
    )
    if not torch.any(non_tip):
      return torch.zeros(env.num_envs, device=env.device)
    return contact[:, non_tip].any(dim=-1).float()
  except Exception:
    return torch.zeros(env.num_envs, device=env.device)


def _contact_duration(env) -> torch.Tensor:
  duration = getattr(env, "_residual_policy_contact_duration", None)
  contact = _primary_contact_from_history(env)
  if not isinstance(duration, torch.Tensor) or duration.shape != contact.shape:
    duration = torch.zeros(contact.shape, dtype=torch.float32, device=env.device)
    env._residual_policy_contact_duration = duration

  step_id = int(getattr(env, "common_step_counter", 0))
  last_step_id = getattr(env, "_residual_policy_contact_duration_step", None)
  if last_step_id != step_id:
    duration.copy_(
      torch.where(
        contact,
        duration + float(env.step_dt),
        torch.zeros_like(duration),
      )
    )
    env._residual_policy_contact_duration_step = step_id

  try:
    sensor = object_pool.active_sensor(env, "hand_apple_contact")
    current = sensor.data.current_contact_time
    if current is not None and current.shape == duration.shape:
      return torch.maximum(duration, current.to(env.device))
  except Exception:
    pass
  return duration


def _body_xyz_names() -> tuple[str, ...]:
  # Use the child links actuated by the 29 body joints as the body-xyz tracking set.
  return tuple(name.replace("_joint", "_link") for name in BODY_29_DOF_NAMES)


def _reference_body_xyz_cache(device: str) -> tuple[torch.Tensor, tuple[str, ...]]:
  ref = _ref(device)
  cached = ref.get("body_link_pos")
  cached_names = ref.get("body_link_names")
  if (
    isinstance(cached, torch.Tensor)
    and cached.device.type == torch.device(device).type
    and cached_names
  ):
    return cached, tuple(cached_names)

  model = mujoco.MjModel.from_xml_path(str(_ROBOT_XML))
  data = mujoco.MjData(model)
  body_names = _body_xyz_names()
  body_ids: list[int] = []
  kept_names: list[str] = []
  for name in body_names:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id >= 0:
      body_ids.append(int(body_id))
      kept_names.append(name)
  if not body_ids:
    raise RuntimeError(f"No body xyz tracking links found in {_ROBOT_XML}")

  free_joints = [
    idx for idx, typ in enumerate(model.jnt_type) if typ == mujoco.mjtJoint.mjJNT_FREE
  ]
  if not free_joints:
    raise RuntimeError(f"No free joint found in {_ROBOT_XML}")
  free_qadr = int(model.jnt_qposadr[free_joints[0]])
  joint_qadr: list[int] = []
  for name in (*BODY_29_DOF_NAMES, *HAND_24_DOF_NAMES):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
      raise RuntimeError(f"Reference FK joint not found in {_ROBOT_XML}: {name}")
    joint_qadr.append(int(model.jnt_qposadr[joint_id]))

  root_pos = ref["root_pos"].detach().cpu().numpy()
  root_rot = ref["root_rot"].detach().cpu().numpy()
  q = ref["dof_pos"].detach().cpu().numpy()
  out: list = []
  out_q: list = []
  # the arrays span every clip when APPLE_EAT_PKL_MIX is used; n_frames is per clip
  for frame in range(int(q.shape[0])):
    data.qpos[:] = model.qpos0
    data.qpos[free_qadr : free_qadr + 3] = root_pos[frame]
    data.qpos[free_qadr + 3 : free_qadr + 7] = root_rot[frame]
    for qadr, value in zip(joint_qadr, q[frame], strict=True):
      data.qpos[qadr] = value
    mujoco.mj_forward(model, data)
    out.append(data.xpos[body_ids].copy())
  body_pos = torch.from_numpy(np.stack(out).astype(np.float32)).to(device=device)
  ref["body_link_pos"] = body_pos
  ref["body_link_names"] = tuple(kept_names)
  return body_pos, tuple(kept_names)


def _reference_tip_pos_cache(device: str) -> tuple[torch.Tensor, tuple[str, ...]]:
  pos, _, names = _reference_body_pose_cache(device, _TIP_BODY_NAMES, "tip_body")
  return pos, names


def _reference_body_pose_cache(
  device: str, names: tuple[str, ...], key: str
) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]:
  """Reference world POSE (position and orientation) of `names`, one entry per reference frame.

  The forward kinematics were already being run over every frame here; only `data.xpos` was kept
  and `data.xquat` -- the same call's other output -- was thrown away. Orientation costs nothing
  extra to cache and is half of what a hand-shape target needs: a fingertip can sit in the right
  place with the finger curled the wrong way.
  """
  ref = _ref(device)
  cached = ref.get(f"{key}_pos")
  cached_quat = ref.get(f"{key}_quat")
  cached_names = ref.get(f"{key}_names")
  if (
    isinstance(cached, torch.Tensor)
    and isinstance(cached_quat, torch.Tensor)
    and cached.device.type == torch.device(device).type
    and cached_names
  ):
    return cached, cached_quat, tuple(cached_names)

  model = mujoco.MjModel.from_xml_path(str(_ROBOT_XML))
  data = mujoco.MjData(model)
  body_ids: list[int] = []
  kept_names: list[str] = []
  for name in names:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id >= 0:
      body_ids.append(int(body_id))
      kept_names.append(name)
  if not body_ids:
    raise RuntimeError(f"None of {key} bodies found in {_ROBOT_XML}: {names}")

  free_joints = [
    idx for idx, typ in enumerate(model.jnt_type) if typ == mujoco.mjtJoint.mjJNT_FREE
  ]
  if not free_joints:
    raise RuntimeError(f"No free joint found in {_ROBOT_XML}")
  free_qadr = int(model.jnt_qposadr[free_joints[0]])
  joint_qadr: list[int] = []
  for name in (*BODY_29_DOF_NAMES, *HAND_24_DOF_NAMES):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
      raise RuntimeError(f"Reference FK joint not found in {_ROBOT_XML}: {name}")
    joint_qadr.append(int(model.jnt_qposadr[joint_id]))

  root_pos = ref["root_pos"].detach().cpu().numpy()
  root_rot = ref["root_rot"].detach().cpu().numpy()
  q = ref["dof_pos"].detach().cpu().numpy()
  out = []
  # the arrays span every clip when APPLE_EAT_PKL_MIX is used; n_frames is per clip
  for frame in range(int(q.shape[0])):
    data.qpos[:] = model.qpos0
    data.qpos[free_qadr : free_qadr + 3] = root_pos[frame]
    data.qpos[free_qadr + 3 : free_qadr + 7] = root_rot[frame]
    for qadr, value in zip(joint_qadr, q[frame], strict=True):
      data.qpos[qadr] = value
    mujoco.mj_forward(model, data)
    out.append(data.xpos[body_ids].copy())
    out_q.append(data.xquat[body_ids].copy())
  pos = torch.from_numpy(np.stack(out).astype(np.float32)).to(device=device)
  quat = torch.from_numpy(np.stack(out_q).astype(np.float32)).to(device=device)
  ref[f"{key}_pos"] = pos
  ref[f"{key}_quat"] = quat
  ref[f"{key}_names"] = tuple(kept_names)
  return pos, quat, tuple(kept_names)


def _reference_release_frame(device: str, lift_threshold: float = 0.03) -> torch.Tensor:
  """Last reference frame at which the apple is still clearly off the table.

  After this the reference has put it down, so it is the moment the policy must have completed the
  placement. Measured on apple_eat_1: 416 with a 3 cm threshold (410 at 5 cm, 420 at 2 cm), against
  contact at 73 and the lift peak at 215 -- so the choice of threshold moves it by only ~10 frames.
  """
  ref = _ref(device)
  key = f"reference_release_frame_{float(lift_threshold):.4f}"
  cached = ref.get(key)
  if isinstance(cached, int):
    return cached
  # per clip: each has its own rest height and its own last lifted frame
  z = ref["obj_pos"][:, 2]
  out = []
  for start, length in apple_mdp.clip_spans(ref):
    zc = z[start : start + length]
    lifted = torch.nonzero(zc > zc[0] + float(lift_threshold), as_tuple=False).flatten()
    out.append(int(lifted[-1].item()) if lifted.numel() > 0 else length - 1)
  value = torch.tensor(out, dtype=torch.long, device=z.device)
  ref[key] = value
  return value


def _raw_tip_object_contact_frame(
  device: str,
  contact_threshold: float = RAW_CONTACT_THRESHOLD,
) -> torch.Tensor:
  """Return the first raw frame whose closest fingertip is within threshold."""
  ref = _ref(device)
  key = f"raw_tip_object_contact_frame_{float(contact_threshold):.4f}"
  cached = ref.get(key)
  if torch.is_tensor(cached):
    return cached

  tip_pos, _ = _reference_tip_pos_cache(device)
  obj_pos = ref["obj_pos"].to(device)
  min_dist = (tip_pos - obj_pos[:, None, :]).norm(dim=-1).min(dim=-1).values
  out = []
  for start, length in apple_mdp.clip_spans(ref):
    md = min_dist[start : start + length]
    hits = (md <= float(contact_threshold)).nonzero(as_tuple=False).flatten()
    out.append(int(hits[0].item()) if hits.numel() > 0 else int(md.argmin().item()))
  frame = torch.tensor(out, dtype=torch.long, device=min_dist.device)
  ref[key] = frame
  return frame


def _body_xyz_tracking_ids(env) -> tuple[torch.Tensor, torch.Tensor]:
  cached = getattr(env, "_residual_body_xyz_tracking_ids", None)
  if cached is not None:
    return cached
  robot: Entity = env.scene["robot"]
  _, names = _reference_body_xyz_cache(env.device)
  body_ids, found_names = robot.find_bodies(names, preserve_order=True)
  name_to_ref_idx = {name: idx for idx, name in enumerate(names)}
  ref_idx = [name_to_ref_idx[name] for name in found_names]
  if not body_ids:
    raise RuntimeError("No runtime robot bodies found for body xyz tracking.")
  result = (
    torch.tensor(body_ids, dtype=torch.long, device=env.device),
    torch.tensor(ref_idx, dtype=torch.long, device=env.device),
  )
  env._residual_body_xyz_tracking_ids = result
  return result


# Tracking weights for residual_tracking_reward. Named so they can be found and
# changed in one place rather than buried as literals in the loop below.
_UPPER_TRACKING_KEYS = ("shoulder", "elbow", "wrist")
_UPPER_TRACKING_WEIGHT = 4.0
_ANKLE_TRACKING_WEIGHT = 2.0


def _body_xyz_tracking_weights(env) -> torch.Tensor:
  cached = getattr(env, "_residual_body_xyz_tracking_weights", None)
  if cached is not None:
    return cached
  _, names = _reference_body_xyz_cache(env.device)
  weights = torch.ones(len(names), dtype=torch.float32, device=env.device)
  for idx, name in enumerate(names):
    # The whole arm gets the wrist weight. Weighting only the wrist asked for an
    # accurate hand position without asking for the arm configuration that reaches
    # it, and the shoulder/elbow are what actually carry the wrist to the object.
    # shoulder/elbow/wrist == _BODY_LINK_GROUPS["upper"] == range(15, NUM_BODY).
    if any(k in name for k in _UPPER_TRACKING_KEYS):
      weights[idx] = _UPPER_TRACKING_WEIGHT
    elif "ankle" in name:
      weights[idx] = _ANKLE_TRACKING_WEIGHT
  _, ref_ids = _body_xyz_tracking_ids(env)
  result = weights[ref_ids]
  env._residual_body_xyz_tracking_weights = result
  return result


def _body_link_group_mask(env, link_group: str) -> torch.Tensor:
  group = str(link_group).strip().lower().replace("-", "_")
  if group not in _BODY_LINK_GROUPS:
    raise ValueError(
      f"Unknown body link group {link_group!r}. "
      f"Expected one of {sorted(_BODY_LINK_GROUPS)}."
    )
  cache_key = f"_residual_body_link_group_mask_{group}"
  cached = getattr(env, cache_key, None)
  if cached is not None:
    return cached

  _, names = _reference_body_xyz_cache(env.device)
  _, ref_ids = _body_xyz_tracking_ids(env)
  selected = {
    BODY_29_DOF_NAMES[idx].replace("_joint", "_link")
    for idx in _BODY_LINK_GROUPS[group]
  }
  values = [names[int(ref_idx.item())] in selected for ref_idx in ref_ids]
  mask = torch.tensor(values, dtype=torch.bool, device=env.device)
  if not bool(mask.any().item()):
    raise RuntimeError(f"No body xyz tracking links matched group {link_group!r}.")
  setattr(env, cache_key, mask)
  return mask


def _body_link_group_ref_entries(
  env, link_group: str
) -> tuple[tuple[str, ...], torch.Tensor]:
  group = str(link_group).strip().lower().replace("-", "_")
  if group not in _BODY_LINK_GROUPS:
    raise ValueError(
      f"Unknown body link group {link_group!r}. "
      f"Expected one of {sorted(_BODY_LINK_GROUPS)}."
    )
  cache_key = f"_residual_body_link_group_ref_entries_{group}"
  cached = getattr(env, cache_key, None)
  if cached is not None:
    return cached

  _, names = _reference_body_xyz_cache(env.device)
  _, ref_ids = _body_xyz_tracking_ids(env)
  ref_positions = {
    names[int(ref_idx)]: pos
    for pos, ref_idx in enumerate(ref_ids.detach().cpu().tolist())
  }
  selected = [
    BODY_29_DOF_NAMES[idx].replace("_joint", "_link")
    for idx in _BODY_LINK_GROUPS[group]
  ]
  matched_names: list[str] = []
  matched_positions: list[int] = []
  for name in selected:
    pos = ref_positions.get(name)
    if pos is None:
      continue
    matched_names.append(name)
    matched_positions.append(pos)
  if not matched_positions:
    raise RuntimeError(f"No body xyz tracking links matched group {link_group!r}.")

  result = (
    tuple(matched_names),
    torch.tensor(matched_positions, dtype=torch.long, device=env.device),
  )
  setattr(env, cache_key, result)
  return result


def _body_link_weights_for_group(
  env,
  link_group: str,
  *,
  use_tracking_weights: bool = True,
) -> torch.Tensor:
  weights = (
    _body_xyz_tracking_weights(env)
    if use_tracking_weights
    else torch.ones_like(_body_xyz_tracking_weights(env))
  )
  mask = _body_link_group_mask(env, link_group)
  return weights * mask.to(dtype=weights.dtype)


def _body_link_distances_for_group(
  env, link_group: str
) -> tuple[tuple[str, ...], torch.Tensor]:
  names, positions = _body_link_group_ref_entries(env, link_group)
  return names, _body_link_distances(env).index_select(dim=-1, index=positions)


def _body_link_weights_for_group_entries(
  env, link_group: str, *, use_tracking_weights: bool = True
) -> torch.Tensor:
  _, positions = _body_link_group_ref_entries(env, link_group)
  weights = (
    _body_xyz_tracking_weights(env)
    if use_tracking_weights
    else torch.ones_like(_body_xyz_tracking_weights(env))
  )
  return weights.index_select(dim=0, index=positions)


def _body_link_dist_mean_for_group(
  env, link_group: str, *, use_tracking_weights: bool = True
) -> torch.Tensor:
  dist = _body_link_distances(env)
  weights = _body_link_weights_for_group(
    env, link_group, use_tracking_weights=use_tracking_weights
  )
  return (dist * weights.unsqueeze(0)).sum(dim=-1) / weights.sum().clamp_min(1e-6)


def _body_link_dist_mse_for_group(
  env, link_group: str, *, use_tracking_weights: bool = True
) -> torch.Tensor:
  dist = _body_link_distances(env)
  weights = _body_link_weights_for_group(
    env, link_group, use_tracking_weights=use_tracking_weights
  )
  return (dist.pow(2) * weights.unsqueeze(0)).sum(dim=-1) / weights.sum().clamp_min(
    1e-6
  )


def _body_xyz_error_for_group(
  env, link_group: str, *, use_tracking_weights: bool = True
) -> torch.Tensor:
  per_link = _body_link_error_vectors(env).pow(2).mean(dim=-1)
  weights = _body_link_weights_for_group(
    env, link_group, use_tracking_weights=use_tracking_weights
  )
  return (per_link * weights.unsqueeze(0)).sum(dim=-1) / weights.sum().clamp_min(1e-6)


def _body_link_axis_abs_mean_for_group(
  env, link_group: str, axis: str, *, use_tracking_weights: bool = True
) -> torch.Tensor:
  axis_idx_by_name = {"x": 0, "y": 1, "z": 2}
  axis_name = str(axis).strip().lower()
  if axis_name not in axis_idx_by_name:
    raise ValueError(f"Unknown axis {axis!r}. Expected one of x, y, z.")
  per_link = _body_link_error_vectors(env)[..., axis_idx_by_name[axis_name]].abs()
  weights = _body_link_weights_for_group(
    env, link_group, use_tracking_weights=use_tracking_weights
  )
  return (per_link * weights.unsqueeze(0)).sum(dim=-1) / weights.sum().clamp_min(1e-6)


def _body_hand_err_mse(env) -> tuple[torch.Tensor, torch.Tensor]:
  robot: Entity = env.scene["robot"]
  ref = _ref(env.device)
  frame = _tracking_frame(env, ref["n_frames"])
  body_q = robot.data.joint_pos[:, _body_joint_ids(env)]
  hand_q = robot.data.joint_pos[:, _hand_joint_ids(env)]
  body_err = (body_q - ref["dof_pos"][frame, :NUM_BODY]).pow(2).mean(dim=-1)
  hand_err = (hand_q - ref["dof_pos"][frame, NUM_BODY:ACTION_DIM]).pow(2).mean(dim=-1)
  active = _active_after_startup(env)
  return body_err * active, hand_err * active


def _contact_window_weight(
  frame: torch.Tensor,
  start_frame: int,
  end_frame: int,
  margin_frames: int,
) -> torch.Tensor:
  start = max(int(start_frame), 0)
  end = max(start, int(end_frame))
  margin = max(int(margin_frames), 0)
  if margin == 0:
    return ((frame >= start) & (frame <= end)).float()
  frame_f = frame.float()
  ramp_in = ((frame_f - float(start - margin)) / float(margin)).clamp(0.0, 1.0)
  ramp_out = ((float(end + margin) - frame_f) / float(margin)).clamp(0.0, 1.0)
  return torch.minimum(ramp_in, ramp_out)


def _raw_contact_pose_errors(
  env,
  start_frame: int = 380,
  end_frame: int = 440,
  margin_frames: int = 20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  robot: Entity = env.scene["robot"]
  obj: Entity = object_pool.active(env)
  ref = _ref(env.device)
  frame = _tracking_frame(env, ref["n_frames"])
  body_q = robot.data.joint_pos[:, _body_joint_ids(env)]
  hand_q = robot.data.joint_pos[:, _hand_joint_ids(env)]
  ref_body_q = ref["dof_pos"][frame, :NUM_BODY]
  ref_hand_q = ref["dof_pos"][frame, NUM_BODY:ACTION_DIM]
  ref_obj_pos = _reference_object_pos_w(env, ref, frame)
  arm_ids = _action_group_ids(env, "arm")
  arm_err = (
    (body_q.index_select(1, arm_ids) - ref_body_q.index_select(1, arm_ids))
    .pow(2)
    .mean(dim=-1)
  )
  hand_err = (hand_q - ref_hand_q).pow(2).mean(dim=-1)
  object_pos_err = (obj.data.root_link_pos_w - ref_obj_pos).norm(dim=-1)
  window = _contact_window_weight(frame, start_frame, end_frame, margin_frames)
  window = window * _active_after_startup(env)
  return arm_err, hand_err, object_pos_err, window


def _raw_tip_object_errors(
  env,
  start_frame: int = 380,
  end_frame: int = 440,
  margin_frames: int = 20,
  top_k: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  robot: Entity = env.scene["robot"]
  obj: Entity = object_pool.active(env)
  ref = _ref(env.device)
  frame = _tracking_frame(env, ref["n_frames"])
  ref_tip_pos, _ = _reference_tip_pos_cache(env.device)
  ref_obj_pos = _reference_object_pos_local(env, ref, frame)
  ref_rel = ref_tip_pos[frame] - ref_obj_pos.unsqueeze(1)
  live_tip_pos = robot.data.body_link_pos_w[:, _tip_body_ids(env)]
  live_rel = live_tip_pos - obj.data.root_link_pos_w.unsqueeze(1)
  ref_dist = ref_rel.norm(dim=-1)
  k = min(max(int(top_k), 1), ref_rel.shape[1])
  ref_contact_ids = ref_dist.topk(k=k, dim=-1, largest=False).indices
  dist_err = (live_rel - ref_rel).pow(2).sum(dim=-1).sqrt()
  contact_dist_err = dist_err.gather(1, ref_contact_ids).mean(dim=-1)
  contact_ref_dist = ref_dist.gather(1, ref_contact_ids).mean(dim=-1)
  window = _contact_window_weight(frame, start_frame, end_frame, margin_frames)
  window = window * _active_after_startup(env)
  return contact_dist_err, contact_ref_dist, window


def _raw_tip_object_radial_errors(
  env,
  start_frame: int = 380,
  end_frame: int = 440,
  margin_frames: int = 20,
  top_k: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  robot: Entity = env.scene["robot"]
  obj: Entity = object_pool.active(env)
  ref = _ref(env.device)
  frame = _tracking_frame(env, ref["n_frames"])
  ref_tip_pos, _ = _reference_tip_pos_cache(env.device)
  ref_obj_pos = _reference_object_pos_local(env, ref, frame)
  ref_rel = ref_tip_pos[frame] - ref_obj_pos.unsqueeze(1)
  ref_dist = ref_rel.norm(dim=-1)
  k = min(max(int(top_k), 1), ref_rel.shape[1])
  ref_contact_ids = ref_dist.topk(k=k, dim=-1, largest=False).indices
  live_tip_pos = robot.data.body_link_pos_w[:, _tip_body_ids(env)]
  live_dist = (live_tip_pos - obj.data.root_link_pos_w.unsqueeze(1)).norm(dim=-1)
  contact_live_dist = live_dist.gather(1, ref_contact_ids).mean(dim=-1)
  contact_ref_dist = ref_dist.gather(1, ref_contact_ids).mean(dim=-1)
  radial_err = (contact_live_dist - contact_ref_dist).abs()
  window = _contact_window_weight(frame, start_frame, end_frame, margin_frames)
  window = window * _active_after_startup(env)
  return radial_err, contact_live_dist, contact_ref_dist, window


def _window_conditional_as_env(
  value: torch.Tensor,
  window: torch.Tensor,
) -> torch.Tensor:
  value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
  window = torch.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
  denom = window.sum().clamp_min(1.0e-6)
  scalar = (value * window).sum() / denom
  return scalar.expand_as(window)


class ResidualFeatureGroupObs:
  """Configurable residual feature observation term.

  Each instance returns one named feature group.  The task exposes all supported
  groups as top-level observation groups so the runner can select/ablate them
  without editing code.
  """

  def __init__(self, cfg, env):
    group = str(cfg.params["group"])
    if group not in RESIDUAL_FEATURE_GROUPS:
      raise ValueError(f"Unknown residual feature group '{group}'.")
    self.group = group
    self.ref_preview_steps = tuple(
      int(v) for v in cfg.params.get("ref_preview_steps", (1, 5, 10))
    )
    self.device = env.device
    self.num_envs = env.num_envs
    self.body_joint_ids: torch.Tensor | None = None
    self.hand_joint_ids: torch.Tensor | None = None
    self.tip_body_ids: torch.Tensor | None = None
    self.hist_len = 4
    # body_jp + body_jv + hand_jp + hand_jv + root_vel_local(3) + gravity(3).
    # 112 with the 24-DOF xhand, 144 with the 40-DOF Wuji hand -- derive it, do not hardcode.
    self.proprio_dim = 2 * (NUM_BODY + NUM_HAND) + 6
    self.proprio_hist = torch.zeros(
      self.num_envs, self.hist_len, self.proprio_dim, device=self.device
    )
    # Set on reset; consumed by _proprio_history to fill the whole history with the
    # true post-reset state rather than shifting zeros through it.
    self.proprio_hist_needs_seed = torch.ones(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.contact_duration = torch.zeros(self.num_envs, 1, device=self.device)
    self.object_hist_lags = _OBJ_HISTORY_LAGS
    self.object_hist_capacity = max(self.object_hist_lags) + 1
    self.object_hist_raw_dim = 13
    self.object_hist: torch.Tensor | None = None
    self.object_hist_valid_len: torch.Tensor | None = None
    self.object_hist_needs_seed: torch.Tensor | None = None
    self.object_hist_last_step: torch.Tensor | None = None
    self.object_hist_lag_tensor: torch.Tensor | None = None
    if self.group == "object_history":
      self.object_hist = torch.zeros(
        self.num_envs,
        self.object_hist_capacity,
        self.object_hist_raw_dim,
        device=self.device,
      )
      self.object_hist_valid_len = torch.zeros(
        self.num_envs, dtype=torch.long, device=self.device
      )
      self.object_hist_needs_seed = torch.ones(
        self.num_envs, dtype=torch.bool, device=self.device
      )
      self.object_hist_last_step = torch.full(
        (self.num_envs,), -1, dtype=torch.long, device=self.device
      )
      self.object_hist_lag_tensor = torch.tensor(
        self.object_hist_lags, dtype=torch.long, device=self.device
      )

  def _resolve(self, env) -> None:
    robot: Entity = env.scene["robot"]
    if self.body_joint_ids is None:
      ids, _ = robot.find_joints(BODY_29_DOF_NAMES, preserve_order=True)
      self.body_joint_ids = torch.tensor(ids, device=env.device, dtype=torch.long)
    if self.hand_joint_ids is None:
      ids, _ = robot.find_joints(HAND_24_DOF_NAMES, preserve_order=True)
      self.hand_joint_ids = torch.tensor(ids, device=env.device, dtype=torch.long)
    if self.tip_body_ids is None:
      ids, _ = robot.find_bodies(_TIP_BODY_NAMES, preserve_order=True)
      self.tip_body_ids = torch.tensor(ids, device=env.device, dtype=torch.long)

  def reset(self, env_ids=None):
    if env_ids is None:
      env_ids = slice(None)
    # Do not leave a zeroed history behind: zero means "at default pose, stationary,
    # zero gravity", which contradicts the reference pose reset just wrote.
    self.proprio_hist[env_ids] = 0.0
    self.proprio_hist_needs_seed[env_ids] = True
    self.contact_duration[env_ids] = 0.0
    if self.object_hist is not None:
      assert self.object_hist_valid_len is not None
      assert self.object_hist_needs_seed is not None
      assert self.object_hist_last_step is not None
      self.object_hist[env_ids] = 0.0
      self.object_hist_valid_len[env_ids] = 0
      self.object_hist_needs_seed[env_ids] = True
      self.object_hist_last_step[env_ids] = -1

  def __call__(self, env, **kwargs) -> torch.Tensor:
    del kwargs
    self._resolve(env)
    if self.group == "sonic_obs_or_latent":
      raise RuntimeError(
        "'sonic_obs_or_latent' is provided by IsaacStyleObs2964, "
        "not ResidualFeatureGroupObs."
      )
    builders = {
      "proprio_history": self._proprio_history,
      "tracker_action": self._tracker_action,
      "reference_phase": self._reference_phase,
      "reference_preview": self._reference_preview,
      "object_state": self._object_state,
      "hand_object_geometry": self._hand_object_geometry,
      "object_surface_geometry": self._object_surface_geometry,
      "object_bps_geometry": self._object_bps_geometry,
      "omnigrasp_object_context": self._omnigrasp_object_context,
      "contact_features": self._contact_features,
      "object_future": self._object_future,
      "object_history": self._object_history,
      "object_event_anchor": self._object_event_anchor,
      "placement_goal": self._placement_goal,
      "tracking_error": self._tracking_error,
      "last_residual": self._last_residual,
      "last_final_action": self._last_final_action,
    }
    value = builders[self.group](env)
    return torch.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)

  def _robot_object(self, env) -> tuple[Entity, Entity, dict, torch.Tensor]:
    robot: Entity = env.scene["robot"]
    obj: Entity = object_pool.active(env)
    ref = _ref(env.device)
    frame = _tracking_frame(env, ref["n_frames"])
    return robot, obj, ref, frame

  def _current_joint_state(
    self, env
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    robot: Entity = env.scene["robot"]
    assert self.body_joint_ids is not None and self.hand_joint_ids is not None
    body_jp = robot.data.joint_pos[:, self.body_joint_ids]
    body_jv = robot.data.joint_vel[:, self.body_joint_ids]
    hand_jp = robot.data.joint_pos[:, self.hand_joint_ids]
    hand_jv = robot.data.joint_vel[:, self.hand_joint_ids]
    return body_jp, body_jv, hand_jp, hand_jv

  def _proprio_history(self, env) -> torch.Tensor:
    robot, _, ref, _ = self._robot_object(env)
    body_jp, body_jv, hand_jp, hand_jv = self._current_joint_state(env)
    root_pose = robot.data.root_link_pose_w
    root_quat = root_pose[:, 3:7]
    root_vel_local = _heading_local_vec(root_quat, robot.data.root_link_vel_w[:, :3])
    gravity = quat_apply(
      quat_conjugate(root_quat),
      torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, 3),
    )
    cur = torch.cat(
      [
        body_jp - ref["default_body"].unsqueeze(0),
        body_jv,
        hand_jp - ref["default_hand"].unsqueeze(0),
        hand_jv,
        root_vel_local,
        gravity,
      ],
      dim=-1,
    )
    seed = self.proprio_hist_needs_seed
    if bool(seed.any()):
      # First observation after a reset: the whole history is this state, so the
      # policy is never fed a fabricated past that disagrees with where it is.
      self.proprio_hist[seed] = cur[seed].unsqueeze(1).expand(-1, self.hist_len, -1)
      self.proprio_hist_needs_seed[seed] = False
    keep = ~seed
    if bool(keep.any()):
      self.proprio_hist[keep, :-1] = self.proprio_hist[keep, 1:].clone()
      self.proprio_hist[keep, -1] = cur[keep]
    return torch.cat(
      [
        self.proprio_hist.reshape(env.num_envs, -1),
        root_pose[:, 2:3],
        _last_final_action(env),
      ],
      dim=-1,
    )

  def _tracker_action(self, env) -> torch.Tensor:
    base = _last_base_action(env)
    if torch.count_nonzero(base).item() == 0:
      base = apple_mdp.teacher_action(env)
    return base

  def _reference_phase(self, env) -> torch.Tensor:
    ref = _ref(env.device)
    frame = _tracking_frame(env, ref["n_frames"])
    # phase is a within-clip quantity: the raw frame is a global row into the concatenated
    # reference, so it must be brought back to its own clip before dividing
    lo, _ = apple_mdp._clip_bounds(env, int(ref["n_frames"]))
    f_local = (frame - lo).float()
    denom = max(float(ref["n_frames"] - 1), 1.0)
    phase = f_local / denom
    contact_row = apple_mdp.per_clip_rows(
      env,
      _raw_tip_object_contact_frame(
        env.device, contact_threshold=RAW_CONTACT_THRESHOLD
      ),
    )
    c_local = (contact_row - lo).float()
    time_to_contact = (c_local - f_local) / denom
    contact_phase = (f_local / c_local.clamp_min(1.0)).clamp(min=0.0, max=1.0)
    post_contact = (frame >= contact_row).float()
    active = _active_after_startup(env)
    return torch.stack(
      [
        phase,
        torch.sin(2.0 * torch.pi * phase),
        torch.cos(2.0 * torch.pi * phase),
        active,
        time_to_contact,
        contact_phase,
        post_contact,
      ],
      dim=-1,
    )

  def _reference_preview(self, env) -> torch.Tensor:
    robot, obj, ref, frame = self._robot_object(env)
    root_pose = robot.data.root_link_pose_w
    root_quat = root_pose[:, 3:7]
    body_jp, _, hand_jp, _ = self._current_joint_state(env)
    q = torch.cat([body_jp, hand_jp], dim=-1)
    chunks = []
    for step in self.ref_preview_steps:
      _hi = apple_mdp._clip_bounds(env, int(ref["n_frames"]))[1]
      fut = torch.minimum(frame + int(step), _hi)
      root_delta = _heading_local_vec(
        root_quat, ref["root_pos"][fut] + env.scene.env_origins - root_pose[:, :3]
      )
      obj_delta = _heading_local_vec(
        root_quat,
        _reference_object_pos_w(env, ref, fut) - obj.data.root_link_pos_w,
      )
      q_delta = ref["dof_pos"][fut, :ACTION_DIM] - q
      chunks.extend([root_delta, obj_delta, q_delta])
    return torch.cat(chunks, dim=-1)

  def _object_state(self, env) -> torch.Tensor:
    robot, obj, ref, frame = self._robot_object(env)
    root_pose = robot.data.root_link_pose_w
    root_quat = root_pose[:, 3:7]
    obj_pos = obj.data.root_link_pos_w
    obj_quat = obj.data.root_link_pose_w[:, 3:7]
    obj_vel = obj.data.root_link_vel_w[:, :3]
    ref_obj = _reference_object_pos_w(env, ref, frame)
    final_obj = _reference_object_pos_w(env, ref, int(ref["n_frames"]) - 1)
    return torch.cat(
      [
        _heading_local_vec(root_quat, obj_pos - root_pose[:, :3]),
        _heading_local_rot6d(root_quat, obj_quat),
        _heading_local_vec(root_quat, obj_vel),
        _heading_local_vec(root_quat, ref_obj - obj_pos),
        _heading_local_vec(root_quat, final_obj - obj_pos),
        obj_pos[:, 2:3],
        obj_vel.norm(dim=-1, keepdim=True),
      ],
      dim=-1,
    )

  def _object_history_raw_state(self, env) -> torch.Tensor:
    obj: Entity = object_pool.active(env)
    return torch.cat(
      [
        obj.data.root_link_pos_w,
        obj.data.root_link_pose_w[:, 3:7],
        obj.data.root_link_vel_w[:, :3],
        obj.data.root_link_vel_w[:, 3:6],
      ],
      dim=-1,
    )

  def _update_object_history(self, env) -> None:
    assert self.object_hist is not None
    assert self.object_hist_valid_len is not None
    assert self.object_hist_needs_seed is not None
    assert self.object_hist_last_step is not None

    cur = self._object_history_raw_state(env)
    step_id = int(getattr(env, "common_step_counter", 0))
    seed = self.object_hist_needs_seed
    if bool(seed.any()):
      self.object_hist[seed] = (
        cur[seed].unsqueeze(1).expand(-1, self.object_hist_capacity, -1)
      )
      self.object_hist_valid_len[seed] = 1
      self.object_hist_needs_seed[seed] = False
      self.object_hist_last_step[seed] = step_id

    advance = self.object_hist_last_step != step_id
    if bool(advance.any()):
      self.object_hist[advance, :-1] = self.object_hist[advance, 1:].clone()
      self.object_hist[advance, -1] = cur[advance]
      self.object_hist_valid_len[advance] = torch.clamp(
        self.object_hist_valid_len[advance] + 1,
        max=self.object_hist_capacity,
      )
      self.object_hist_last_step[advance] = step_id

  def _object_history(self, env) -> torch.Tensor:
    """Sparse live object trajectory memory in the current robot heading frame."""
    robot, obj, _, _ = self._robot_object(env)
    assert self.object_hist is not None
    assert self.object_hist_valid_len is not None
    assert self.object_hist_lag_tensor is not None

    self._update_object_history(env)

    lags = self.object_hist_lag_tensor
    gather = self.object_hist_capacity - 1 - lags
    hist = self.object_hist[:, gather]
    n_lags = int(lags.numel())

    root_quat = robot.data.root_link_pose_w[:, 3:7]
    cur_pos = obj.data.root_link_pos_w
    cur_quat = obj.data.root_link_pose_w[:, 3:7]
    hist_pos = hist[:, :, 0:3]
    hist_quat = hist[:, :, 3:7]
    hist_lin_vel = hist[:, :, 7:10]
    hist_ang_vel = hist[:, :, 10:13]

    root_quat_lag = root_quat.unsqueeze(1).expand(-1, n_lags, -1).reshape(-1, 4)
    cur_quat_lag = cur_quat.unsqueeze(1).expand(-1, n_lags, -1).reshape(-1, 4)
    dpos = _heading_local_vec(
      root_quat_lag,
      (hist_pos - cur_pos.unsqueeze(1)).reshape(-1, 3),
    ).reshape(env.num_envs, n_lags, 3)
    lin_vel = _heading_local_vec(root_quat_lag, hist_lin_vel.reshape(-1, 3)).reshape(
      env.num_envs, n_lags, 3
    )
    ang_vel = _heading_local_vec(root_quat_lag, hist_ang_vel.reshape(-1, 3)).reshape(
      env.num_envs, n_lags, 3
    )
    quat_err = quat_mul(hist_quat.reshape(-1, 4), quat_conjugate(cur_quat_lag))
    rot6d = _quat_to_rot6d(quat_err).reshape(env.num_envs, n_lags, 6)

    valid = (self.object_hist_valid_len.unsqueeze(1) > lags.unsqueeze(0)).float()
    max_lag = max(float(self.object_hist_capacity - 1), 1.0)
    age = (lags.float() / max_lag).unsqueeze(0).expand(env.num_envs, -1)
    lag_features = torch.cat(
      [
        dpos,
        lin_vel,
        ang_vel,
        rot6d,
        age.unsqueeze(-1) * valid.unsqueeze(-1),
        valid.unsqueeze(-1),
      ],
      dim=-1,
    )
    length = (
      self.object_hist_valid_len.float().clamp(max=self.object_hist_capacity)
      / float(self.object_hist_capacity)
    ).unsqueeze(-1)
    return torch.cat([lag_features.reshape(env.num_envs, -1), length], dim=-1)

  def _tip_geometry(self, env) -> tuple[torch.Tensor, torch.Tensor]:
    robot: Entity = env.scene["robot"]
    obj: Entity = object_pool.active(env)
    assert self.tip_body_ids is not None
    tip_pos = robot.data.body_link_pos_w[:, self.tip_body_ids]
    vec = tip_pos - obj.data.root_link_pos_w.unsqueeze(1)
    dist = vec.norm(dim=-1)
    return vec, dist

  def _hand_object_geometry(self, env) -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    root_quat = robot.data.root_link_pose_w[:, 3:7]
    vec, dist = self._tip_geometry(env)
    vec_local = _heading_local_vec(root_quat, vec).reshape(env.num_envs, -1)
    min_dist, min_idx = dist.min(dim=-1)
    min_idx_norm = min_idx.float().unsqueeze(-1) / max(float(dist.shape[-1] - 1), 1.0)
    return torch.cat([vec_local, dist, min_dist.unsqueeze(-1), min_idx_norm], dim=-1)

  def _object_surface_geometry(self, env) -> torch.Tensor:
    robot, obj, ref, frame = self._robot_object(env)
    assert self.tip_body_ids is not None

    root_pose = robot.data.root_link_pose_w
    root_quat = root_pose[:, 3:7]
    obj_pos = obj.data.root_link_pos_w
    obj_quat = obj.data.root_link_pose_w[:, 3:7]
    ref_obj_pos = _reference_object_pos_w(env, ref, frame)
    ref_obj_quat = ref["obj_quat"][frame]

    dirs = torch.tensor(
      [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
      ],
      device=env.device,
      dtype=obj_pos.dtype,
    )
    n_points = int(dirs.shape[0])

    def points_from_pose(pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
      dirs_b = dirs.unsqueeze(0).expand(env.num_envs, n_points, 3)
      quat_b = quat.unsqueeze(1).expand(env.num_envs, n_points, 4)
      rotated = quat_apply(quat_b.reshape(-1, 4), dirs_b.reshape(-1, 3)).reshape(
        env.num_envs, n_points, 3
      )
      return pos.unsqueeze(1) + float(APPLE_RADIUS) * rotated

    points_w = points_from_pose(obj_pos, obj_quat)
    ref_points_w = points_from_pose(ref_obj_pos, ref_obj_quat)
    root_quat_b = root_quat.unsqueeze(1).expand(env.num_envs, n_points, 4)
    root_pos_b = root_pose[:, :3].unsqueeze(1)
    live_points_local = _heading_local_vec(
      root_quat_b.reshape(-1, 4),
      (points_w - root_pos_b).reshape(-1, 3),
    ).reshape(env.num_envs, -1)
    ref_delta_local = _heading_local_vec(
      root_quat_b.reshape(-1, 4),
      (ref_points_w - points_w).reshape(-1, 3),
    ).reshape(env.num_envs, -1)

    tip_pos = robot.data.body_link_pos_w[:, self.tip_body_ids]
    tip_surface_dist = (tip_pos.unsqueeze(2) - points_w.unsqueeze(1)).norm(dim=-1)
    min_dist_per_tip = tip_surface_dist.min(dim=-1).values
    min_dist_per_point = tip_surface_dist.min(dim=1).values
    ref0_z = _reference_object_pos_w(env, ref, 0)[:, 2:3]
    height_above_ref0 = obj_pos[:, 2:3] - ref0_z
    radius = torch.full(
      (env.num_envs, 1), float(APPLE_RADIUS), device=env.device, dtype=obj_pos.dtype
    )

    return torch.cat(
      [
        live_points_local,
        ref_delta_local,
        tip_surface_dist.reshape(env.num_envs, -1),
        min_dist_per_tip,
        min_dist_per_point,
        radius,
        height_above_ref0,
      ],
      dim=-1,
    )

  def _object_bps_geometry(self, env) -> torch.Tensor:
    robot, obj, ref, frame = self._robot_object(env)
    assert self.tip_body_ids is not None

    root_pose = robot.data.root_link_pose_w
    root_pos = root_pose[:, :3]
    root_quat = root_pose[:, 3:7]
    obj_pos = obj.data.root_link_pos_w
    obj_quat = obj.data.root_link_pose_w[:, 3:7]
    ref_obj_pos = _reference_object_pos_w(env, ref, frame)
    ref_obj_quat = ref["obj_quat"][frame]

    n_points = 32
    idx = torch.arange(n_points, device=env.device, dtype=obj_pos.dtype)
    z = 1.0 - 2.0 * (idx + 0.5) / float(n_points)
    radius_xy = (1.0 - z.square()).clamp_min(0.0).sqrt()
    theta = idx * (torch.pi * (3.0 - 5.0**0.5))
    dirs = torch.stack(
      [radius_xy * torch.cos(theta), radius_xy * torch.sin(theta), z],
      dim=-1,
    )
    dirs_b = dirs.unsqueeze(0).expand(env.num_envs, n_points, 3)

    def surface_points(pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
      quat_b = quat.unsqueeze(1).expand(env.num_envs, n_points, 4)
      offset = quat_apply(quat_b.reshape(-1, 4), dirs_b.reshape(-1, 3)).reshape(
        env.num_envs, n_points, 3
      )
      return pos.unsqueeze(1) + float(APPLE_RADIUS) * offset

    surface_w = surface_points(obj_pos, obj_quat)
    ref_surface_w = surface_points(ref_obj_pos, ref_obj_quat)
    root_quat_p = root_quat.unsqueeze(1).expand(env.num_envs, n_points, 4)

    surface_heading = _heading_local_vec(
      root_quat_p.reshape(-1, 4),
      (surface_w - root_pos.unsqueeze(1)).reshape(-1, 3),
    ).reshape(env.num_envs, -1)
    surface_delta_heading = _heading_local_vec(
      root_quat_p.reshape(-1, 4),
      (ref_surface_w - surface_w).reshape(-1, 3),
    ).reshape(env.num_envs, -1)

    tip_pos = robot.data.body_link_pos_w[:, self.tip_body_ids]
    tip_vec = tip_pos - obj_pos.unsqueeze(1)
    tip_vec_object = _quat_local_vec(obj_quat, tip_vec).reshape(env.num_envs, -1)
    tip_dist = tip_vec.norm(dim=-1)
    tip_signed_surface_dist = tip_dist - float(APPLE_RADIUS)
    tip_surface_dist = (tip_pos.unsqueeze(2) - surface_w.unsqueeze(1)).norm(dim=-1)
    min_surface_per_tip = tip_surface_dist.min(dim=-1).values
    min_tip_per_surface = tip_surface_dist.min(dim=1).values

    future_steps = torch.tensor((0, 10, 20, 40), device=env.device, dtype=torch.long)
    future = (frame.unsqueeze(1) + future_steps.unsqueeze(0)).clamp(
      max=int(ref["n_frames"]) - 1
    )
    ref_future = ref["obj_pos"][future.reshape(-1)].reshape(env.num_envs, -1, 3)
    ref_future = ref_future + env.scene.env_origins.unsqueeze(1)
    future_delta_object = _quat_local_vec(
      obj_quat.unsqueeze(1).expand(env.num_envs, ref_future.shape[1], 4).reshape(-1, 4),
      (ref_future - obj_pos.unsqueeze(1)).reshape(-1, 3),
    ).reshape(env.num_envs, -1)

    ref0_z = _reference_object_pos_w(env, ref, 0)[:, 2:3]
    height_above_ref0 = obj_pos[:, 2:3] - ref0_z
    physical_contact, grasp_contact, force_close = _physical_contact(env)
    contact_summary = torch.stack(
      [physical_contact, grasp_contact, force_close], dim=-1
    )

    return torch.cat(
      [
        surface_heading,
        surface_delta_heading,
        tip_vec_object,
        tip_dist,
        tip_signed_surface_dist,
        tip_surface_dist.reshape(env.num_envs, -1),
        min_surface_per_tip,
        min_tip_per_surface,
        future_delta_object,
        height_above_ref0,
        contact_summary,
      ],
      dim=-1,
    )

  def _omnigrasp_object_context(self, env) -> torch.Tensor:
    robot, obj, ref, frame = self._robot_object(env)
    assert self.tip_body_ids is not None

    root_pose = robot.data.root_link_pose_w
    root_pos = root_pose[:, :3]
    root_quat = root_pose[:, 3:7]
    obj_pos = obj.data.root_link_pos_w
    obj_quat = obj.data.root_link_pose_w[:, 3:7]
    obj_vel = obj.data.root_link_vel_w[:, :3]
    obj_ang_vel = obj.data.root_link_ang_vel_w

    future_steps = torch.tensor(
      (0, 5, 10, 20, 40, 80, 120), device=env.device, dtype=torch.long
    )
    future = (frame.unsqueeze(1) + future_steps.unsqueeze(0)).clamp(
      max=int(ref["n_frames"]) - 1
    )
    n_future = int(future.shape[1])
    future_flat = future.reshape(-1)
    ref_pos = ref["obj_pos"][future_flat].reshape(env.num_envs, n_future, 3)
    ref_pos = ref_pos + env.scene.env_origins.unsqueeze(1)
    ref_quat = ref["obj_quat"][future_flat].reshape(env.num_envs, n_future, 4)
    ref_vel = ref["obj_vel"][future_flat].reshape(env.num_envs, n_future, 3)

    root_quat_f = root_quat.unsqueeze(1).expand(env.num_envs, n_future, 4)
    obj_quat_f = obj_quat.unsqueeze(1).expand(env.num_envs, n_future, 4)
    obj_pos_f = obj_pos.unsqueeze(1)
    obj_vel_f = obj_vel.unsqueeze(1)

    obj_ref_delta = ref_pos - obj_pos_f
    future_pos_heading = _heading_local_vec(
      root_quat_f.reshape(-1, 4), obj_ref_delta.reshape(-1, 3)
    ).reshape(env.num_envs, -1)
    future_pos_object = _quat_local_vec(obj_quat, obj_ref_delta).reshape(
      env.num_envs, -1
    )
    future_quat_err = quat_mul(
      ref_quat.reshape(-1, 4),
      quat_conjugate(obj_quat_f.reshape(-1, 4)),
    )
    future_rot_err = _quat_to_rot6d(future_quat_err).reshape(env.num_envs, -1)
    future_vel_delta = ref_vel - obj_vel_f
    future_vel_heading = _heading_local_vec(
      root_quat_f.reshape(-1, 4), future_vel_delta.reshape(-1, 3)
    ).reshape(env.num_envs, -1)
    future_vel_object = _quat_local_vec(obj_quat, future_vel_delta).reshape(
      env.num_envs, -1
    )

    ref0_z = _reference_object_pos_w(env, ref, 0)[:, 2:3]
    height_above_ref0 = obj_pos[:, 2:3] - ref0_z
    current_object = torch.cat(
      [
        _heading_local_vec(root_quat, obj_pos - root_pos),
        _heading_local_rot6d(root_quat, obj_quat),
        _heading_local_vec(root_quat, obj_vel),
        _heading_local_vec(root_quat, obj_ang_vel),
        _quat_local_vec(obj_quat, root_pos - obj_pos),
        _quat_to_rot6d(quat_mul(quat_conjugate(obj_quat), root_quat)),
        height_above_ref0,
        obj_vel.norm(dim=-1, keepdim=True),
      ],
      dim=-1,
    )

    dirs = torch.tensor(
      [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, -1.0, -1.0],
      ],
      device=env.device,
      dtype=obj_pos.dtype,
    )
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1.0)
    n_points = int(dirs.shape[0])
    dirs_b = dirs.unsqueeze(0).expand(env.num_envs, n_points, 3)

    def surface_points(pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
      quat_b = quat.unsqueeze(1).expand(env.num_envs, n_points, 4)
      offset = quat_apply(quat_b.reshape(-1, 4), dirs_b.reshape(-1, 3)).reshape(
        env.num_envs, n_points, 3
      )
      return pos.unsqueeze(1) + float(APPLE_RADIUS) * offset

    ref_pos_now = ref["obj_pos"][frame] + env.scene.env_origins
    ref_quat_now = ref["obj_quat"][frame]
    surface_w = surface_points(obj_pos, obj_quat)
    ref_surface_w = surface_points(ref_pos_now, ref_quat_now)
    root_quat_p = root_quat.unsqueeze(1).expand(env.num_envs, n_points, 4)
    surface_heading = _heading_local_vec(
      root_quat_p.reshape(-1, 4),
      (surface_w - root_pos.unsqueeze(1)).reshape(-1, 3),
    ).reshape(env.num_envs, -1)
    surface_ref_delta = _heading_local_vec(
      root_quat_p.reshape(-1, 4),
      (ref_surface_w - surface_w).reshape(-1, 3),
    ).reshape(env.num_envs, -1)

    tip_pos = robot.data.body_link_pos_w[:, self.tip_body_ids]
    tip_vec = tip_pos - obj_pos.unsqueeze(1)
    tip_vec_heading = _heading_local_vec(
      root_quat.unsqueeze(1).expand(env.num_envs, tip_vec.shape[1], 4).reshape(-1, 4),
      tip_vec.reshape(-1, 3),
    ).reshape(env.num_envs, -1)
    tip_vec_object = _quat_local_vec(obj_quat, tip_vec).reshape(env.num_envs, -1)
    tip_dist = tip_vec.norm(dim=-1)
    tip_surface_dist = (tip_pos.unsqueeze(2) - surface_w.unsqueeze(1)).norm(dim=-1)
    min_surface_per_tip = tip_surface_dist.min(dim=-1).values
    min_tip_per_surface = tip_surface_dist.min(dim=1).values

    contact_flags, contact_force_local = _primary_contact_force_features(env)
    physical_contact, grasp_contact, force_close = _physical_contact(env)
    contact_summary = torch.cat(
      [
        physical_contact.unsqueeze(-1),
        grasp_contact.unsqueeze(-1),
        force_close.unsqueeze(-1),
        contact_flags,
        contact_force_local.reshape(env.num_envs, -1),
      ],
      dim=-1,
    )

    table: Entity = env.scene["table"]
    table_pos = table.data.root_link_pos_w
    table_quat = table.data.root_link_pose_w[:, 3:7]
    table_top = table_pos[:, 2:3] + 0.5 * float(TABLE_THICKNESS)
    table_features = torch.cat(
      [
        _heading_local_vec(root_quat, table_pos - root_pos),
        _heading_local_rot6d(root_quat, table_quat),
        _quat_local_vec(obj_quat, table_pos - obj_pos),
        obj_pos[:, 2:3] - float(APPLE_RADIUS) - table_top,
      ],
      dim=-1,
    )

    return torch.cat(
      [
        future_pos_heading,
        future_pos_object,
        future_rot_err,
        future_vel_heading,
        future_vel_object,
        current_object,
        surface_heading,
        surface_ref_delta,
        tip_vec_heading,
        tip_vec_object,
        tip_dist,
        tip_surface_dist.reshape(env.num_envs, -1),
        min_surface_per_tip,
        min_tip_per_surface,
        contact_summary,
        table_features,
      ],
      dim=-1,
    )

  def _contact_features(self, env) -> torch.Tensor:
    obj: Entity = object_pool.active(env)
    _, dist = self._tip_geometry(env)
    min_dist = dist.min(dim=-1, keepdim=True).values
    touch = (min_dist <= RAW_CONTACT_THRESHOLD).float()
    next_duration = torch.where(
      touch.bool(),
      self.contact_duration + env.step_dt,
      torch.zeros_like(self.contact_duration),
    )
    self.contact_duration.copy_(next_duration)
    obj_speed = obj.data.root_link_vel_w[:, :3].norm(dim=-1, keepdim=True)
    ref = _ref(env.device)
    ref_obj0 = _reference_object_pos_w(env, ref, 0)
    lifted = (obj.data.root_link_pos_w[:, 2:3] > (ref_obj0[:, 2:3] + 0.04)).float()
    force_close_proxy = torch.exp(-20.0 * min_dist) * touch
    contact_flags, contact_force_local = _primary_contact_force_features(env)
    return torch.cat(
      [
        touch,
        min_dist,
        obj_speed,
        lifted,
        self.contact_duration,
        force_close_proxy,
        contact_flags,
        contact_force_local.reshape(env.num_envs, -1),
      ],
      dim=-1,
    )

  def _object_future(self, env) -> torch.Tensor:
    """The object's reference path ahead, decoupled from the joint preview.

    Near field carries velocity as well as position, because the carry phase needs to know whether
    the reference is accelerating (it sets how hard to pre-load the grip) and no existing group
    provides future object velocity. Far field is position only and stops at 160 steps, where the
    predictable motion saturates.
    """
    robot, obj, ref, frame = self._robot_object(env)
    root_quat = robot.data.root_link_pose_w[:, 3:7]
    obj_pos = obj.data.root_link_pos_w
    obj_quat = obj.data.root_link_pose_w[:, 3:7]
    n = int(ref["n_frames"])
    chunks = []

    for k in _OBJ_FUTURE_NEAR:
      fut = torch.clamp(frame + int(k), max=n - 1)
      dpos = _reference_object_pos_w(env, ref, fut) - obj_pos
      dvel = ref["obj_vel"][fut] - obj.data.root_link_vel_w[:, :3]
      chunks.append(_heading_local_vec(root_quat, dpos))
      chunks.append(_heading_local_vec(root_quat, dvel))

    for k in _OBJ_FUTURE_FAR:
      fut = torch.clamp(frame + int(k), max=n - 1)
      dpos = _reference_object_pos_w(env, ref, fut) - obj_pos
      chunks.append(_heading_local_vec(root_quat, dpos))
      # Same displacement in the object's own frame: invariant to the robot's pose, which is the
      # useful description while carrying.
      chunks.append(quat_apply(quat_conjugate(obj_quat), dpos))

    return torch.cat(chunks, dim=-1)

  def _object_event_anchor(self, env) -> torch.Tensor:
    """Release timing plus the object's reference pose at release, and near-future object motion."""
    robot, obj, ref, frame = self._robot_object(env)
    root_quat = robot.data.root_link_pose_w[:, 3:7]
    n = int(ref["n_frames"])
    release = _reference_release_frame(env.device)  # per clip, LOCAL frames
    release_row = apple_mdp.per_clip_rows(env, release)  # per env, GLOBAL rows
    lo, _ = apple_mdp._clip_bounds(env, n)
    denom = max(float(n - 1), 1.0)

    f = frame.float()
    f_local = (frame - lo).float()  # phase is measured within the clip
    rel_local = (release_row - lo).float().clamp_min(1.0)
    frames_to_release = (release_row.float() - f) / denom
    release_phase = (f_local / rel_local).clamp(0.0, 1.0)
    post_release = (frame >= release_row).float()

    obj_pos = obj.data.root_link_pos_w
    rel_target = _reference_object_pos_w(env, ref, release_row)
    to_release = _heading_local_vec(root_quat, rel_target - obj_pos)

    rel_quat = ref["obj_quat"][release_row]  # already per env; no broadcast needed
    quat_err = quat_mul(rel_quat, quat_conjugate(obj.data.root_link_pose_w[:, 3:7]))

    # Reference object velocity just ahead: tells the policy whether the target is accelerating.
    fut1 = torch.clamp(frame + 1, max=n - 1)
    fut5 = torch.clamp(frame + 5, max=n - 1)
    ref_vel1 = _heading_local_vec(root_quat, ref["obj_vel"][fut1])
    ref_vel5 = _heading_local_vec(root_quat, ref["obj_vel"][fut5])
    # Live angular velocity: the rolling-in-hand slip mode, absent from object_state.
    obj_angvel = _heading_local_vec(root_quat, obj.data.root_link_vel_w[:, 3:6])

    active = _active_after_startup(env)
    return torch.cat(
      [
        frames_to_release.unsqueeze(-1),
        release_phase.unsqueeze(-1),
        post_release.unsqueeze(-1),
        to_release,
        to_release.norm(dim=-1, keepdim=True),
        _quat_to_rot6d(quat_err),
        ref_vel1,
        ref_vel5,
        obj_angvel,
        active.unsqueeze(-1),
      ],
      dim=-1,
    )

  def _placement_goal(self, env) -> torch.Tensor:
    robot, obj, ref, _ = self._robot_object(env)
    root_quat = robot.data.root_link_pose_w[:, 3:7]
    target_pos = _reference_object_pos_w(env, ref, int(ref["n_frames"]) - 1)
    target_quat = ref["obj_quat"][-1].unsqueeze(0).expand(env.num_envs, -1)
    pos_err = target_pos - obj.data.root_link_pos_w
    quat_err = quat_mul(target_quat, quat_conjugate(obj.data.root_link_pose_w[:, 3:7]))
    return torch.cat(
      [
        _heading_local_vec(root_quat, pos_err),
        _quat_to_rot6d(quat_err),
        pos_err.norm(dim=-1, keepdim=True),
        quat_err[:, 0:1],
      ],
      dim=-1,
    )

  def _tracking_error(self, env) -> torch.Tensor:
    robot, obj, ref, frame = self._robot_object(env)
    root_pose = robot.data.root_link_pose_w
    root_quat = root_pose[:, 3:7]
    body_jp, _, hand_jp, _ = self._current_joint_state(env)
    q = torch.cat([body_jp, hand_jp], dim=-1)
    ref_root = ref["root_pos"][frame] + env.scene.env_origins
    root_pos_err = _heading_local_vec(root_quat, ref_root - root_pose[:, :3])
    root_quat_err = quat_mul(ref["root_rot"][frame], quat_conjugate(root_quat))
    q_err = ref["dof_pos"][frame, :ACTION_DIM] - q
    obj_err = _heading_local_vec(
      root_quat,
      _reference_object_pos_w(env, ref, frame) - obj.data.root_link_pos_w,
    )
    return torch.cat(
      [root_pos_err, _quat_to_rot6d(root_quat_err), q_err, obj_err], dim=-1
    )

  def _last_residual(self, env) -> torch.Tensor:
    residual = _last_residual_action(env)
    if torch.count_nonzero(residual).item() == 0:
      residual = env.action_manager.prev_action - apple_mdp.teacher_action(env)
    return residual

  def _last_final_action(self, env) -> torch.Tensor:
    return _last_final_action(env)


class SonicEncoderObs1762:
  """Official SONIC encoder observation for the frozen ONNX base tracker."""

  def __init__(self, cfg, env):
    del env
    self.root_anchor_mode = str(cfg.params.get("root_anchor_mode", "identity")).strip()
    if self.root_anchor_mode not in {"identity", "relative"}:
      raise ValueError(
        "root_anchor_mode must be 'identity' or 'relative', "
        f"got {self.root_anchor_mode!r}."
      )

  def reset(self, env_ids=None):
    del env_ids

  def __call__(self, env, **kwargs) -> torch.Tensor:
    del kwargs
    ref = _ref(env.device)
    frame = _tracking_frame(env, ref["n_frames"])
    n_frames = int(ref["n_frames"])
    _plo, _phi = apple_mdp._clip_bounds(env, n_frames)
    enc_obs = torch.zeros(env.num_envs, SONIC_ENCODER_OBS_DIM, device=env.device)
    enc_obs[:, 1] = 1.0
    pkl_for_il = torch.tensor(apple_mdp.PKL_FOR_IL, device=env.device, dtype=torch.long)

    for fi in range(10):
      ff = torch.maximum(torch.minimum(frame + fi * 5, _phi), _plo)
      ref_body_il = ref["dof_pos"][ff, :NUM_BODY][:, pkl_for_il]
      enc_obs[:, 4 + fi * NUM_BODY : 4 + (fi + 1) * NUM_BODY] = ref_body_il

      nf = torch.maximum(torch.minimum(frame + fi * 5 + 1, _phi), _plo)
      pf = torch.maximum(torch.minimum(frame + fi * 5 - 1, _phi), _plo)
      body_next = ref["dof_pos"][nf, :NUM_BODY][:, pkl_for_il]
      body_prev = ref["dof_pos"][pf, :NUM_BODY][:, pkl_for_il]
      enc_obs[:, 294 + fi * NUM_BODY : 294 + (fi + 1) * NUM_BODY] = (
        body_next - body_prev
      ) * 25.0

    if self.root_anchor_mode == "identity":
      identity_anchor = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], device=env.device)
      for fi in range(10):
        enc_obs[:, 601 + fi * 6 : 601 + (fi + 1) * 6] = identity_anchor.unsqueeze(0)
      return torch.nan_to_num(enc_obs, nan=0.0, posinf=1e6, neginf=-1e6).clamp(
        -1e6, 1e6
      )

    robot: Entity = env.scene["robot"]
    actual_heading = apple_mdp._calc_heading_quat(robot.data.root_link_quat_w)
    ref_rot_0 = ref["root_rot"][torch.zeros_like(frame)]
    ref_heading_0 = apple_mdp.quat_mul(
      actual_heading,
      apple_mdp.quat_conjugate(apple_mdp._calc_heading_quat(ref_rot_0)),
    )
    for fi in range(10):
      ff = torch.maximum(torch.minimum(frame + fi * 5, _phi), _plo)
      transformed_ref = apple_mdp.quat_mul(ref_heading_0, ref["root_rot"][ff])
      rel = apple_mdp.quat_mul(
        apple_mdp.quat_conjugate(actual_heading), transformed_ref
      )
      enc_obs[:, 601 + fi * 6 : 601 + (fi + 1) * 6] = apple_mdp._quat_to_rot6d(rel)
    return torch.nan_to_num(enc_obs, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)


def _joint_tracking_terms(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  robot: Entity = env.scene["robot"]
  ref = _ref(env.device)
  frame = _tracking_frame(env, ref["n_frames"])
  joint_names = list(BODY_29_DOF_NAMES) + list(HAND_24_DOF_NAMES)
  joint_ids, _ = robot.find_joints(joint_names, preserve_order=True)
  joint_ids_t = torch.tensor(joint_ids, device=env.device, dtype=torch.long)
  q = robot.data.joint_pos[:, joint_ids_t]
  q_err = (q - ref["dof_pos"][frame, :ACTION_DIM]).pow(2).mean(dim=-1)
  root_err = (
    robot.data.root_link_pos_w[:, :3] - (ref["root_pos"][frame] + env.scene.env_origins)
  ).norm(dim=-1)
  active = _active_after_startup(env)
  return q_err, root_err, active


def _body_xyz_error(env) -> torch.Tensor:
  per_link = _body_link_error_vectors(env).pow(2).mean(dim=-1)
  weights = _body_xyz_tracking_weights(env)
  return (per_link * weights.unsqueeze(0)).sum(dim=-1) / weights.sum().clamp_min(1e-6)


def _body_link_error_vectors(env) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  ref = _ref(env.device)
  frame = _tracking_frame(env, ref["n_frames"])
  body_pos_ref_all, _ = _reference_body_xyz_cache(env.device)
  body_ids, ref_ids = _body_xyz_tracking_ids(env)
  live = robot.data.body_link_pos_w[:, body_ids]
  target = body_pos_ref_all[frame][:, ref_ids] + env.scene.env_origins[:, None, :]
  return live - target


def _body_link_distances(env) -> torch.Tensor:
  return _body_link_error_vectors(env).norm(dim=-1)


def _body_link_dist_mean(env) -> torch.Tensor:
  dist = _body_link_distances(env)
  weights = _body_xyz_tracking_weights(env)
  return (dist * weights.unsqueeze(0)).sum(dim=-1) / weights.sum().clamp_min(1e-6)


def _body_link_dist_mse(env) -> torch.Tensor:
  dist = _body_link_distances(env)
  weights = _body_xyz_tracking_weights(env)
  return (dist.pow(2) * weights.unsqueeze(0)).sum(dim=-1) / weights.sum().clamp_min(
    1e-6
  )


def residual_tracking_reward(
  env,
  distance_std: float = 0.03,
  xyz_weight: float | None = None,
  link_group: str = "all",
  use_tracking_weights: bool = True,
  per_link_reward: bool = False,
  log_per_link: bool = False,
  log_prefix: str = "tracking",
  close_bonus_weight: float = 0.0,
  close_bonus_threshold: float = 0.03,
  close_bonus_margin: float = 0.01,
  miss_penalty_weight: float = 0.0,
  miss_penalty_margin: float = 0.01,
  post_frame: int = -1,
  post_frame_scale: float = 1.0,
) -> torch.Tensor:
  del xyz_weight
  dist = _body_link_dist_mean_for_group(
    env, link_group, use_tracking_weights=use_tracking_weights
  )
  std = max(float(distance_std), 1.0e-6)
  active = _active_after_startup(env)
  close_margin = max(float(close_bonus_margin), 1.0e-6)
  miss_margin = max(float(miss_penalty_margin), 1.0e-6)
  prefix = str(log_prefix).strip().replace("/", "_") or "tracking"

  if per_link_reward or log_per_link:
    link_names, link_dist = _body_link_distances_for_group(env, link_group)
    link_weights = _body_link_weights_for_group_entries(
      env, link_group, use_tracking_weights=use_tracking_weights
    )
    link_base = torch.exp(-0.5 * (link_dist / std).pow(2))
    link_close_miss = torch.relu(link_dist - float(close_bonus_threshold))
    link_close_progress = 1.0 - (link_close_miss / close_margin).clamp(max=1.0)
    link_close_bonus = float(close_bonus_weight) * link_close_progress
    link_miss_penalty = float(miss_penalty_weight) * (
      link_close_miss / miss_margin
    ).clamp(max=1.0)
    link_value = link_base + link_close_bonus - link_miss_penalty
    weighted = link_weights.unsqueeze(0)
    weight_sum = link_weights.sum().clamp_min(1e-6)
    if per_link_reward:
      base = (link_base * weighted).sum(dim=-1) / weight_sum
      close_bonus = (link_close_bonus * weighted).sum(dim=-1) / weight_sum
      miss_penalty = (link_miss_penalty * weighted).sum(dim=-1) / weight_sum
    else:
      base = torch.exp(-0.5 * (dist / std).pow(2))
      close_miss = torch.relu(dist - float(close_bonus_threshold))
      close_progress = 1.0 - (close_miss / close_margin).clamp(max=1.0)
      close_bonus = float(close_bonus_weight) * close_progress
      miss_penalty = float(miss_penalty_weight) * (close_miss / miss_margin).clamp(
        max=1.0
      )
    if log_per_link:
      _safe_log(env, f"Metric/{prefix}/link_dist_mean", dist * active)
      _safe_log(env, f"Metric/{prefix}/link_dist_max", link_dist.max(dim=-1).values)
      _safe_log(env, f"Metric/{prefix}/link_dist_min", link_dist.min(dim=-1).values)
      _safe_log(
        env,
        f"Metric/{prefix}/link_dist_std",
        link_dist.std(dim=-1, unbiased=False),
      )
      for idx, name in enumerate(link_names):
        _safe_log(env, f"Metric/{prefix}/{name}_dist", link_dist[:, idx] * active)
        _safe_log(env, f"ResidualReward/{prefix}/{name}", link_value[:, idx] * active)
        _safe_log(
          env, f"ResidualReward/{prefix}/{name}_base", link_base[:, idx] * active
        )
        _safe_log(
          env,
          f"ResidualReward/{prefix}/{name}_close_bonus",
          link_close_bonus[:, idx] * active,
        )
  else:
    base = torch.exp(-0.5 * (dist / std).pow(2))
    close_miss = torch.relu(dist - float(close_bonus_threshold))
    close_progress = 1.0 - (close_miss / close_margin).clamp(max=1.0)
    close_bonus = float(close_bonus_weight) * close_progress
    miss_penalty = float(miss_penalty_weight) * (close_miss / miss_margin).clamp(
      max=1.0
    )
  phase_scale = torch.ones_like(active)
  if int(post_frame) >= 0 and abs(float(post_frame_scale) - 1.0) > 1.0e-6:
    ref = _ref(env.device)
    # CLIP-LOCAL, not the global row. Under MIX clip c frame f lives at row c * n_frames + f, so
    # comparing the global row against a frame number makes every clip after the first satisfy the
    # gate from its very first step -- clip 1's smallest row is already n_frames.
    frame = apple_mdp.local_tracking_frame(env, int(ref["n_frames"]))
    phase_scale = torch.where(
      frame >= int(post_frame),
      torch.full_like(active, float(post_frame_scale)),
      phase_scale,
    )
  value = (base + close_bonus - miss_penalty) * active * phase_scale
  _safe_log(env, "ResidualReward/tracking", value)
  _safe_log(env, "ResidualReward/tracking_base", base * active)
  _safe_log(env, "ResidualReward/tracking_close_bonus", close_bonus * active)
  _safe_log(env, "ResidualReward/tracking_miss_penalty", -miss_penalty * active)
  _safe_log(env, f"ResidualReward/{prefix}_phase_scale", phase_scale * active)
  if prefix != "tracking":
    _safe_log(env, f"ResidualReward/{prefix}", value)
    _safe_log(env, f"ResidualReward/{prefix}_base", base * active)
    _safe_log(env, f"ResidualReward/{prefix}_close_bonus", close_bonus * active)
    _safe_log(env, f"ResidualReward/{prefix}_miss_penalty", -miss_penalty * active)
  _safe_log(env, "Metric/tracking_region_link_dist_mean", dist)
  _safe_log(
    env,
    "Metric/tracking_region_xyz_err_mse",
    _body_xyz_error_for_group(
      env, link_group, use_tracking_weights=use_tracking_weights
    ),
  )
  _safe_log(env, "Metric/body_link_dist_mean", _body_link_dist_mean(env))
  _safe_log(env, "Metric/body_link_dist_mse", _body_link_dist_mse(env))
  _safe_log(env, "Metric/body_xyz_err_mse", _body_xyz_error(env))
  return value


def residual_axis_tracking_reward(
  env,
  axis: str = "z",
  axis_std: float = 0.015,
  link_group: str = "left_wrist",
  use_tracking_weights: bool = True,
  log_prefix: str = "axis_tracking",
) -> torch.Tensor:
  err = _body_link_axis_abs_mean_for_group(
    env, link_group, axis, use_tracking_weights=use_tracking_weights
  )
  std = max(float(axis_std), 1.0e-6)
  active = _active_after_startup(env)
  value = torch.exp(-0.5 * (err / std).pow(2)) * active
  prefix = str(log_prefix).strip().replace("/", "_") or f"{link_group}_{axis}_tracking"
  _safe_log(env, f"ResidualReward/{prefix}", value)
  _safe_log(env, f"Metric/{prefix}_abs_err", err * active)
  return value


def residual_hand_action_tracking_reward(
  env,
  k: float = 10.0,
  gate_mode: str = "none",
  near_threshold: float = 0.15,
) -> torch.Tensor:
  teacher = apple_mdp.teacher_action(env)[:, NUM_BODY:ACTION_DIM]
  action = env.action_manager.action[:, NUM_BODY:ACTION_DIM]
  err = (action - teacher).pow(2).mean(dim=-1)
  mode = str(gate_mode).strip().lower()
  if mode in {"none", "always"}:
    gate = torch.ones_like(err)
  elif mode in {"near", "near_contact"}:
    near = _tip_distances(env).min(dim=-1).values <= float(near_threshold)
    physical, _, _ = _physical_contact(env)
    gate = torch.maximum(near.float(), physical)
  elif mode in {"contact", "contact_latch"}:
    latch = getattr(env, "_omnigrasp_contact_latch", None)
    gate = latch if isinstance(latch, torch.Tensor) else torch.zeros_like(err)
  else:
    raise ValueError(
      "hand_action_tracking gate_mode must be none, near_contact, or contact_latch; "
      f"got {gate_mode!r}."
    )
  active = _active_after_startup(env)
  gate = gate.to(device=err.device, dtype=err.dtype) * active
  value = torch.exp(-float(k) * err) * gate
  _safe_log(env, "ResidualReward/hand_action_tracking", value)
  _safe_log(env, "Metric/hand_action_mse", err)
  _safe_log(env, "Metric/hand_action_tracking_gate", gate)
  return value


def residual_raw_contact_tracking_reward(
  env,
  k: float = 8.0,
  arm_weight: float = 1.0,
  hand_weight: float = 1.5,
  object_weight: float = 0.0,
  object_pos_std: float = 0.06,
  start_frame: int = 380,
  end_frame: int = 440,
  margin_frames: int = 20,
) -> torch.Tensor:
  arm_err, hand_err, object_pos_err, window = _raw_contact_pose_errors(
    env,
    start_frame=start_frame,
    end_frame=end_frame,
    margin_frames=margin_frames,
  )
  err = float(arm_weight) * arm_err + float(hand_weight) * hand_err
  pose_score = torch.exp(-float(k) * err)
  object_score = torch.exp(
    -0.5 * (object_pos_err / max(float(object_pos_std), 1.0e-6)).pow(2)
  )
  object_mix = 1.0 - float(object_weight) + float(object_weight) * object_score
  value = pose_score * object_mix.clamp(min=0.0, max=1.0) * window
  _safe_log(env, "ResidualReward/raw_contact_tracking", value)
  _safe_log(env, "Metric/raw_contact_arm_err_mse", arm_err * window)
  _safe_log(env, "Metric/raw_contact_hand_err_mse", hand_err * window)
  _safe_log(env, "Metric/raw_contact_object_pos_err", object_pos_err * window)
  _safe_log(env, "Metric/raw_contact_window", window)
  return value


def residual_raw_tip_object_tracking_reward(
  env,
  std: float = 0.05,
  top_k: int = 2,
  start_frame: int = 380,
  end_frame: int = 440,
  margin_frames: int = 20,
) -> torch.Tensor:
  dist_err, ref_dist, window = _raw_tip_object_errors(
    env,
    start_frame=start_frame,
    end_frame=end_frame,
    margin_frames=margin_frames,
    top_k=top_k,
  )
  value = torch.exp(-0.5 * (dist_err / max(float(std), 1.0e-6)).pow(2)) * window
  _safe_log(env, "ResidualReward/raw_tip_object_tracking", value)
  _safe_log(env, "Metric/raw_tip_object_dist_err", dist_err * window)
  _safe_log(env, "Metric/raw_tip_object_ref_dist", ref_dist * window)
  return value


def residual_raw_tip_radial_tracking_reward(
  env,
  std: float = 0.08,
  reach_std: float = 0.20,
  reach_weight: float = 0.5,
  top_k: int = 1,
  start_frame: int = 380,
  end_frame: int = 440,
  margin_frames: int = 20,
) -> torch.Tensor:
  radial_err, live_dist, ref_dist, window = _raw_tip_object_radial_errors(
    env,
    start_frame=start_frame,
    end_frame=end_frame,
    margin_frames=margin_frames,
    top_k=top_k,
  )
  radial_score = torch.exp(-0.5 * (radial_err / max(float(std), 1.0e-6)).pow(2))
  reach_err = torch.relu(live_dist - ref_dist)
  reach_score = torch.exp(-reach_err / max(float(reach_std), 1.0e-6))
  mix = radial_score + float(reach_weight) * reach_score
  value = mix / (1.0 + max(float(reach_weight), 0.0)) * window
  _safe_log(env, "ResidualReward/raw_tip_radial_tracking", value)
  _safe_log(env, "Metric/raw_tip_radial_err", radial_err * window)
  _safe_log(env, "Metric/raw_tip_radial_live_dist", live_dist * window)
  _safe_log(env, "Metric/raw_tip_radial_ref_dist", ref_dist * window)
  return value


def residual_task_reward(env, dist_scale: float = 6.0) -> torch.Tensor:
  min_tip_dist = apple_mdp.fingertip_min_dist(env)
  value = torch.exp(-float(dist_scale) * min_tip_dist) * _active_after_startup(env)
  _safe_log(env, "ResidualReward/task", value)
  return value


def residual_grasp_reward(
  env,
  contact_dist: float = 0.08,
  close_std: float = 0.06,
  lift_height: float = 0.04,
  object_speed_threshold: float = 0.03,
  close_weight: float = 0.25,
  near_weight: float = 1.0,
  contact_weight: float = 4.0,
  force_weight: float = 2.0,
  object_motion_weight: float = 1.0,
  lift_weight: float = 2.0,
  drift_target: float = 0.0,
  drift_margin: float = 0.10,
  drift_power: float = 1.0,
) -> torch.Tensor:
  min_tip_dist = apple_mdp.fingertip_min_dist(env)
  ref = _ref(env.device)
  obj: Entity = object_pool.active(env)
  near = (min_tip_dist < float(contact_dist)).float()
  close = torch.exp(-min_tip_dist / max(float(close_std), 1e-6))
  physical_contact, grasp_contact, force_close = _physical_contact(env)
  object_moving = (
    obj.data.root_link_vel_w[:, :3].norm(dim=-1) > float(object_speed_threshold)
  ).float()
  lifted = (
    obj.data.root_link_pos_w[:, 2]
    > (_reference_object_pos_w(env, ref, 0)[:, 2] + float(lift_height))
  ).float()
  near_or_contact = torch.clamp(near + physical_contact, max=1.0)
  drift_gate = _drift_reward_gate(env, drift_target, drift_margin, drift_power)
  gated_interaction = (
    float(contact_weight) * physical_contact
    + float(contact_weight) * grasp_contact
    + float(force_weight) * force_close
    + float(object_motion_weight) * object_moving * near_or_contact
    + float(lift_weight) * lifted
  )
  value = (
    float(close_weight) * close
    + float(near_weight) * near
    + drift_gate * gated_interaction
  ) * _active_after_startup(env)
  _safe_log(env, "ResidualReward/grasp", value)
  return value


def residual_surface_contact_reward(
  env,
  surface_dist: float = APPLE_RADIUS,
  std: float = 0.02,
  near_dist: float = 0.10,
  contact_weight: float = 1.0,
  force_weight: float = 0.5,
  drift_target: float = 0.0,
  drift_margin: float = 0.10,
  drift_power: float = 1.0,
  gate_shaping: bool = True,
) -> torch.Tensor:
  min_tip_dist = apple_mdp.fingertip_min_dist(env)
  target = float(surface_dist)
  scale = max(float(std), 1.0e-6)
  surface = torch.exp(-0.5 * ((min_tip_dist - target) / scale).pow(2))
  near_gate = (min_tip_dist < float(near_dist)).float()
  physical_contact, _, force_close = _physical_contact(env)
  drift_gate = _drift_reward_gate(env, drift_target, drift_margin, drift_power)
  shaping = surface * near_gate
  interaction = (
    float(contact_weight) * physical_contact + float(force_weight) * force_close
  )
  if bool(gate_shaping):
    value = (shaping + interaction) * drift_gate
  else:
    value = shaping + drift_gate * interaction
  value = value * _active_after_startup(env)
  _safe_log(env, "ResidualReward/surface_contact", value)
  return value


def residual_multi_tip_surface_reward(
  env,
  surface_dist: float = APPLE_RADIUS,
  std: float = 0.025,
  near_dist: float = 0.10,
  top_k: int = 3,
  multi_tip_count: int = 2,
  multi_tip_weight: float = 0.5,
  contact_weight: float = 1.0,
  grasp_weight: float = 1.0,
  force_weight: float = 0.5,
  drift_target: float = 0.0,
  drift_margin: float = 0.10,
  drift_power: float = 1.0,
  gate_shaping: bool = True,
  opposition_gate_floor: float = 1.0,
  opposition_gate_hi: float = 1.0,
  opposition_gate_lo: float = 0.0,
  opposition_gate_hand: str = "right",
  opposition_gate_pair: tuple[int, int] = (0, 2),
) -> torch.Tensor:
  dist = _tip_distances(env)
  target = float(surface_dist)
  scale = max(float(std), 1.0e-6)
  surface = torch.exp(-0.5 * ((dist - target) / scale).pow(2))
  k = min(max(int(top_k), 1), surface.shape[-1])
  top_surface = surface.topk(k=k, dim=-1).values.mean(dim=-1)
  near_count = (dist < float(near_dist)).float().sum(dim=-1)
  multi_tip_score = (near_count / max(float(multi_tip_count), 1.0)).clamp(max=1.0)
  physical_contact, grasp_contact, force_close = _physical_contact(env)
  drift_gate = _drift_reward_gate(env, drift_target, drift_margin, drift_power)
  # Only the shaping is gated.  Genuine physical contact keeps its full value --
  # if the hand ever really closes on the apple we want that paid in full.
  shaping = top_surface + float(multi_tip_weight) * multi_tip_score
  shaping = shaping * _opposition_gate(
    env,
    opposition_gate_floor,
    opposition_gate_hi,
    opposition_gate_lo,
    opposition_gate_hand,
    opposition_gate_pair,
  )
  interaction = (
    float(contact_weight) * physical_contact
    + float(grasp_weight) * grasp_contact
    + float(force_weight) * force_close
  )
  if bool(gate_shaping):
    value = (shaping + interaction) * drift_gate
  else:
    value = shaping + drift_gate * interaction
  value = value * _active_after_startup(env)
  _safe_log(env, "ResidualReward/multi_tip_surface", value)
  _safe_log(
    env,
    "Metric/grasp_opposition_cos",
    _grasp_opposition_cos(env, opposition_gate_hand, opposition_gate_pair),
  )
  return value


def residual_object_drift_limit_reward(
  env,
  target: float = 0.25,
) -> torch.Tensor:
  value = -torch.relu(_object_drift(env) - float(target)).pow(2)
  value = value * _active_after_startup(env)
  _safe_log(env, "ResidualReward/object_drift_limit", value)
  return value


def residual_contact_duration_reward(
  env,
  target_duration: float = 0.08,
  top_k: int = 2,
  contact_weight: float = 0.5,
  grasp_weight: float = 1.0,
  drift_target: float = 0.0,
  drift_margin: float = 0.10,
  drift_power: float = 1.0,
) -> torch.Tensor:
  duration = _contact_duration(env)
  k = min(max(int(top_k), 1), duration.shape[-1])
  top_duration = duration.topk(k=k, dim=-1).values.mean(dim=-1)
  duration_score = (top_duration / max(float(target_duration), 1.0e-6)).clamp(max=1.0)
  physical_contact, grasp_contact, _ = _physical_contact(env)
  drift_gate = _drift_reward_gate(env, drift_target, drift_margin, drift_power)
  value = (
    (
      duration_score
      + float(contact_weight) * physical_contact
      + float(grasp_weight) * grasp_contact
    )
    * drift_gate
    * _active_after_startup(env)
  )
  _safe_log(env, "ResidualReward/contact_duration", value)
  return value


def residual_placement_reward(env, std: float = 0.25) -> torch.Tensor:
  ref = _ref(env.device)
  obj: Entity = object_pool.active(env)
  target_pos = _reference_object_pos_w(env, ref, int(ref["n_frames"]) - 1)
  dist = (obj.data.root_link_pos_w - target_pos).norm(dim=-1)
  value = torch.exp(-dist / max(float(std), 1e-6)) * _active_after_startup(env)
  _safe_log(env, "ResidualReward/placement", value)
  return value


def residual_stability_reward(
  env,
  torso_height_min: float = 0.55,
  action_jerk_weight: float = 0.05,
  object_speed_weight: float = 0.04,
) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  obj: Entity = object_pool.active(env)
  active = _active_after_startup(env)
  low_root = torch.relu(float(torso_height_min) - robot.data.root_link_pos_w[:, 2])
  jerk = (
    (env.action_manager.action - env.action_manager.prev_action).pow(2).mean(dim=-1)
  )
  obj_speed = obj.data.root_link_vel_w[:, :3].norm(dim=-1)
  penalty = (
    low_root + float(action_jerk_weight) * jerk + float(object_speed_weight) * obj_speed
  )
  value = -penalty * active
  _safe_log(env, "ResidualReward/stability", value)
  return value


def residual_l2_reward(env) -> torch.Tensor:
  residual = _last_residual_action(env)
  value = -residual.pow(2).mean(dim=-1) * _active_after_startup(env)
  _safe_log(env, "ResidualReward/residual_l2", value)
  return value


def residual_smooth_reward(env) -> torch.Tensor:
  residual = _last_residual_action(env)
  prev = _previous_residual_delta(env)
  value = -(residual - prev).pow(2).mean(dim=-1) * _active_after_startup(env)
  _safe_log(env, "ResidualReward/residual_smooth", value)
  return value


def residual_token_l2_reward(env) -> torch.Tensor:
  token = _last_token_delta(env)
  value = -token.pow(2).mean(dim=-1) * _active_after_startup(env)
  _safe_log(env, "ResidualReward/token_l2", value)
  return value


def residual_token_smooth_reward(env) -> torch.Tensor:
  token = _last_token_delta(env)
  prev = _previous_token_delta(env)
  value = -(token - prev).pow(2).mean(dim=-1) * _active_after_startup(env)
  _safe_log(env, "ResidualReward/token_smooth", value)
  return value


def residual_decoder_body_delta_l2_reward(env) -> torch.Tensor:
  delta = _last_decoder_body_delta(env)
  value = -delta.pow(2).mean(dim=-1) * _active_after_startup(env)
  _safe_log(env, "ResidualReward/decoder_body_delta_l2", value)
  return value


def residual_decoder_body_delta_norm_limit_reward(
  env,
  target: float = 0.60,
) -> torch.Tensor:
  delta_norm = _last_decoder_body_delta(env).norm(dim=-1)
  value = -torch.relu(delta_norm - float(target)).pow(2) * _active_after_startup(env)
  _safe_log(env, "ResidualReward/decoder_body_delta_norm_limit", value)
  return value


def residual_decoder_body_delta_ratio_limit_reward(
  env,
  target: float = 0.08,
) -> torch.Tensor:
  base = _action_group_norm(_last_base_action(env), env, "body").clamp_min(1e-6)
  ratio = _last_decoder_body_delta(env).norm(dim=-1) / base
  value = -torch.relu(ratio - float(target)).pow(2) * _active_after_startup(env)
  _safe_log(env, "ResidualReward/decoder_body_delta_ratio_limit", value)
  return value


def residual_decoder_body_delta_joint_limit_reward(
  env,
  target: float = 0.25,
) -> torch.Tensor:
  delta = _last_decoder_body_delta(env).abs()
  value = -torch.relu(delta - float(target)).pow(2).mean(dim=-1)
  value = value * _active_after_startup(env)
  _safe_log(env, "ResidualReward/decoder_body_delta_joint_limit", value)
  return value


def residual_hand_ratio_limit_reward(
  env,
  target: float = 1.30,
) -> torch.Tensor:
  ratio = _residual_group_ratio(env, "hand")
  value = -torch.relu(ratio - float(target)).pow(2) * _active_after_startup(env)
  _safe_log(env, "ResidualReward/hand_ratio_limit", value)
  return value


def residual_norm_metric(env) -> torch.Tensor:
  value = _last_residual_action(env).norm(dim=-1)
  _safe_log(env, "ResidualMetric/residual_norm", value)
  return value


def hand_control_gate_metric(env) -> torch.Tensor:
  value = _hand_control_gate(env)
  _safe_log(env, "ResidualMetric/hand_control_gate", value)
  return value


def hand_mean_delta_norm_metric(env) -> torch.Tensor:
  value = _action_group_norm(
    _last_action_mean(env) - _last_base_action(env), env, "hand"
  )
  _safe_log(env, "ResidualMetric/hand_mean_delta_norm", value)
  return value


def hand_sample_delta_pre_clip_norm_metric(env) -> torch.Tensor:
  value = _hand_sample_delta_pre_clip(env).norm(dim=-1)
  _safe_log(env, "ResidualMetric/hand_sample_delta_pre_clip_norm", value)
  return value


def hand_sample_delta_post_clip_norm_metric(env) -> torch.Tensor:
  value = _hand_sample_delta_post_clip(env).norm(dim=-1)
  _safe_log(env, "ResidualMetric/hand_sample_delta_post_clip_norm", value)
  return value


def hand_sample_clip_frac_metric(env) -> torch.Tensor:
  value = _hand_sample_clip_frac(env)
  _safe_log(env, "ResidualMetric/hand_sample_clip_frac", value)
  return value


def body_sample_delta_pre_clip_norm_metric(env) -> torch.Tensor:
  value = _body_sample_delta_pre_clip(env).norm(dim=-1)
  _safe_log(env, "ResidualMetric/body_sample_delta_pre_clip_norm", value)
  return value


def body_sample_delta_post_clip_norm_metric(env) -> torch.Tensor:
  value = _body_sample_delta_post_clip(env).norm(dim=-1)
  _safe_log(env, "ResidualMetric/body_sample_delta_post_clip_norm", value)
  return value


def body_sample_clip_frac_metric(env) -> torch.Tensor:
  value = _body_sample_clip_frac(env)
  _safe_log(env, "ResidualMetric/body_sample_clip_frac", value)
  return value


def hand_action_std_mean_metric(env) -> torch.Tensor:
  value = _hand_action_std_mean(env)
  _safe_log(env, "ResidualMetric/hand_action_std_mean", value)
  return value


def hand_close_left_metric(env) -> torch.Tensor:
  value = _hand_primitive_close(env)[:, 0]
  _safe_log(env, "ResidualMetric/hand_close_left", value)
  return value


def hand_close_right_metric(env) -> torch.Tensor:
  value = _hand_primitive_close(env)[:, 1]
  _safe_log(env, "ResidualMetric/hand_close_right", value)
  return value


def hand_close_mean_metric(env) -> torch.Tensor:
  value = _hand_primitive_close(env).mean(dim=-1)
  _safe_log(env, "ResidualMetric/hand_close_mean", value)
  return value


def hand_primitive_delta_norm_metric(env) -> torch.Tensor:
  value = _hand_primitive_delta(env).norm(dim=-1)
  _safe_log(env, "ResidualMetric/hand_primitive_delta_norm", value)
  return value


def base_action_norm_metric(env) -> torch.Tensor:
  value = _last_base_action(env).norm(dim=-1)
  _safe_log(env, "ResidualMetric/base_action_norm", value)
  return value


def final_action_norm_metric(env) -> torch.Tensor:
  value = _last_final_action(env).norm(dim=-1)
  _safe_log(env, "ResidualMetric/final_action_norm", value)
  return value


def token_residual_norm_metric(env) -> torch.Tensor:
  value = _last_token_delta(env).norm(dim=-1)
  _safe_log(env, "ResidualMetric/token_residual_norm", value)
  return value


def token_residual_smooth_norm_metric(env) -> torch.Tensor:
  value = (_last_token_delta(env) - _previous_token_delta(env)).norm(dim=-1)
  _safe_log(env, "ResidualMetric/token_residual_smooth_norm", value)
  return value


def token_residual_clip_frac_metric(env) -> torch.Tensor:
  value = _clip_fraction(
    _last_token_delta(env),
    getattr(env, "_residual_token_clip", None),
  )
  _safe_log(env, "ResidualMetric/token_residual_clip_frac", value)
  return value


def decoder_body_delta_norm_metric(env) -> torch.Tensor:
  value = _last_decoder_body_delta(env).norm(dim=-1)
  _safe_log(env, "ResidualMetric/decoder_body_delta_norm", value)
  return value


def decoder_body_delta_ratio_metric(env) -> torch.Tensor:
  base = _action_group_norm(_last_base_action(env), env, "body").clamp_min(1e-6)
  value = _last_decoder_body_delta(env).norm(dim=-1) / base
  _safe_log(env, "ResidualMetric/decoder_body_delta_ratio", value)
  return value


def decoder_body_delta_joint_rms_metric(env) -> torch.Tensor:
  value = _last_decoder_body_delta(env).pow(2).mean(dim=-1).sqrt()
  _safe_log(env, "ResidualMetric/decoder_body_delta_joint_rms", value)
  return value


def decoder_body_delta_joint_abs_max_metric(env) -> torch.Tensor:
  value = _last_decoder_body_delta(env).abs().amax(dim=-1)
  _safe_log(env, "ResidualMetric/decoder_body_delta_joint_abs_max", value)
  return value


def contact_frac_metric(env, threshold: float = LIVE_CONTACT_THRESHOLD) -> torch.Tensor:
  value = (apple_mdp.fingertip_min_dist(env) <= float(threshold)).float()
  _safe_log(env, "ResidualMetric/contact_frac", value)
  return value


def hand_body_contact_frac_metric(env) -> torch.Tensor:
  value = _hand_body_contact(env)
  _safe_log(env, "ResidualMetric/hand_body_contact_frac", value)
  return value


def non_tip_hand_body_contact_frac_metric(env) -> torch.Tensor:
  value = _non_tip_hand_body_contact(env)
  _safe_log(env, "ResidualMetric/non_tip_hand_body_contact_frac", value)
  return value


def surface_contact_frac_metric(
  env,
  surface_dist: float = APPLE_RADIUS,
  tolerance: float = 0.02,
) -> torch.Tensor:
  value = (
    (apple_mdp.fingertip_min_dist(env) - float(surface_dist)).abs() < float(tolerance)
  ).float()
  _safe_log(env, "ResidualMetric/surface_contact_frac", value)
  return value


def multi_tip_near_frac_metric(
  env,
  near_dist: float = 0.10,
  multi_tip_count: int = 2,
) -> torch.Tensor:
  value = (
    (_tip_distances(env) < float(near_dist)).sum(dim=-1) >= int(multi_tip_count)
  ).float()
  _safe_log(env, "ResidualMetric/multi_tip_near_frac", value)
  return value


def contact_duration_max_metric(env) -> torch.Tensor:
  value = _contact_duration(env).amax(dim=-1)
  _safe_log(env, "ResidualMetric/contact_duration_max", value)
  return value


def contact_duration_frac_metric(env) -> torch.Tensor:
  duration = _contact_duration(env).amax(dim=-1)
  value = (duration > 0.0).float()
  active_count = value.sum().clamp(min=1.0)
  active_mean = (duration * value).sum() / active_count
  _safe_log(env, "ResidualMetric/contact_duration_frac", value)
  _safe_log(env, "ResidualMetric/contact_duration_active_mean", active_mean)
  return value


def object_motion_frac_metric(env, speed_threshold: float = 0.05) -> torch.Tensor:
  obj: Entity = object_pool.active(env)
  value = (
    obj.data.root_link_vel_w[:, :3].norm(dim=-1) > float(speed_threshold)
  ).float()
  _safe_log(env, "ResidualMetric/object_motion_frac", value)
  return value


def placement_error_metric(env) -> torch.Tensor:
  ref = _ref(env.device)
  obj: Entity = object_pool.active(env)
  target_pos = _reference_object_pos_w(env, ref, int(ref["n_frames"]) - 1)
  value = (obj.data.root_link_pos_w - target_pos).norm(dim=-1)
  _safe_log(env, "ResidualMetric/placement_error", value)
  return value


def reward_total_metric(env) -> torch.Tensor:
  value = torch.nan_to_num(env.reward_buf, nan=0.0, posinf=0.0, neginf=0.0)
  _safe_log(env, "Reward/total", value)
  return value


def residual_base_ratio_metric(env) -> torch.Tensor:
  value = _last_residual_action(env).norm(dim=-1) / _last_base_action(env).norm(
    dim=-1
  ).clamp_min(1e-6)
  _safe_log(env, "ResidualMetric/residual_base_ratio", value)
  return value


def _residual_group_norm_metric(env, group: str, log_key: str) -> torch.Tensor:
  value = _action_group_norm(_last_residual_action(env), env, group)
  _safe_log(env, log_key, value)
  return value


def _residual_group_ratio(env, group: str) -> torch.Tensor:
  residual = _action_group_norm(_last_residual_action(env), env, group)
  base = _action_group_norm(_last_base_action(env), env, group).clamp_min(1e-6)
  return residual / base


def _residual_group_ratio_metric(env, group: str, log_key: str) -> torch.Tensor:
  value = _residual_group_ratio(env, group)
  _safe_log(env, log_key, value)
  return value


def body_residual_norm_metric(env) -> torch.Tensor:
  return _residual_group_norm_metric(
    env,
    "body",
    "ResidualMetric/body_residual_norm",
  )


def hand_residual_norm_metric(env) -> torch.Tensor:
  return _residual_group_norm_metric(
    env,
    "hand",
    "ResidualMetric/hand_residual_norm",
  )


def leg_residual_norm_metric(env) -> torch.Tensor:
  return _residual_group_norm_metric(
    env,
    "leg",
    "ResidualMetric/leg_residual_norm",
  )


def arm_residual_norm_metric(env) -> torch.Tensor:
  return _residual_group_norm_metric(
    env,
    "arm",
    "ResidualMetric/arm_residual_norm",
  )


def body_residual_ratio_metric(env) -> torch.Tensor:
  return _residual_group_ratio_metric(
    env,
    "body",
    "ResidualMetric/body_residual_ratio",
  )


def astra_body_delta_norm_metric(env) -> torch.Tensor:
  return _residual_group_norm_metric(
    env,
    "body",
    "ResidualMetric/astra_body_delta_norm",
  )


def astra_body_delta_ratio_metric(env) -> torch.Tensor:
  return _residual_group_ratio_metric(
    env,
    "body",
    "ResidualMetric/astra_body_delta_ratio",
  )


def astra_body_delta_joint_rms_metric(env) -> torch.Tensor:
  body_delta = _action_group(_last_residual_action(env), env, "body")
  value = body_delta.pow(2).mean(dim=-1).sqrt()
  _safe_log(env, "ResidualMetric/astra_body_delta_joint_rms", value)
  return value


def hand_residual_ratio_metric(env) -> torch.Tensor:
  return _residual_group_ratio_metric(
    env,
    "hand",
    "ResidualMetric/hand_residual_ratio",
  )


def leg_residual_ratio_metric(env) -> torch.Tensor:
  return _residual_group_ratio_metric(
    env,
    "leg",
    "ResidualMetric/leg_residual_ratio",
  )


def arm_residual_ratio_metric(env) -> torch.Tensor:
  return _residual_group_ratio_metric(
    env,
    "arm",
    "ResidualMetric/arm_residual_ratio",
  )


def residual_clip_frac_metric(env) -> torch.Tensor:
  value = _clip_fraction(
    _last_raw_residual_action(env),
    getattr(env, "_residual_action_clip", None),
    mask=_residual_action_mask(env),
  )
  _safe_log(env, "ResidualMetric/residual_clip_frac", value)
  return value


def final_action_clip_frac_metric(env) -> torch.Tensor:
  value = _clip_fraction(
    _last_final_action(env),
    getattr(env, "_residual_final_action_clip", None),
  )
  _safe_log(env, "ResidualMetric/final_action_clip_frac", value)
  return value


def body_final_delta_norm_metric(env) -> torch.Tensor:
  value = _action_group_norm(
    _last_final_action(env) - _last_base_action(env),
    env,
    "body",
  )
  _safe_log(env, "ResidualMetric/body_final_delta_norm", value)
  return value


def hand_final_delta_norm_metric(env) -> torch.Tensor:
  value = _action_group_norm(
    _last_final_action(env) - _last_base_action(env),
    env,
    "hand",
  )
  _safe_log(env, "ResidualMetric/hand_final_delta_norm", value)
  return value


def grasp_contact_frac_metric(env) -> torch.Tensor:
  _, value, _ = _physical_contact(env)
  _safe_log(env, "ResidualMetric/grasp_contact_frac", value)
  return value


def body_err_mse_metric(env) -> torch.Tensor:
  body_err, _ = _body_hand_err_mse(env)
  _safe_log(env, "Metric/body_err_mse", body_err)
  return body_err


def body_xyz_err_mse_metric(env) -> torch.Tensor:
  value = _body_xyz_error(env) * _active_after_startup(env)
  _safe_log(env, "Metric/body_xyz_err_mse", value)
  return value


def body_link_dist_mean_metric(env) -> torch.Tensor:
  value = _body_link_dist_mean(env) * _active_after_startup(env)
  _safe_log(env, "Metric/body_link_dist_mean", value)
  return value


def body_link_dist_mse_metric(env) -> torch.Tensor:
  value = _body_link_dist_mse(env) * _active_after_startup(env)
  _safe_log(env, "Metric/body_link_dist_mse", value)
  return value


def lower_body_xyz_err_mse_metric(env) -> torch.Tensor:
  value = _body_xyz_error_for_group(env, "lower") * _active_after_startup(env)
  _safe_log(env, "Metric/lower_body_xyz_err_mse", value)
  return value


def upper_body_xyz_err_mse_metric(env) -> torch.Tensor:
  value = _body_xyz_error_for_group(env, "upper") * _active_after_startup(env)
  _safe_log(env, "Metric/upper_body_xyz_err_mse", value)
  return value


def lower_body_link_dist_mean_metric(env) -> torch.Tensor:
  value = _body_link_dist_mean_for_group(env, "lower") * _active_after_startup(env)
  _safe_log(env, "Metric/lower_body_link_dist_mean", value)
  return value


def upper_body_link_dist_mean_metric(env) -> torch.Tensor:
  value = _body_link_dist_mean_for_group(env, "upper") * _active_after_startup(env)
  _safe_log(env, "Metric/upper_body_link_dist_mean", value)
  return value


def lower_wrist_link_dist_mean_metric(env) -> torch.Tensor:
  value = _body_link_dist_mean_for_group(env, "lower_wrist") * _active_after_startup(
    env
  )
  _safe_log(env, "Metric/lower_wrist_link_dist_mean", value)
  return value


def ankle_link_dist_mean_metric(
  env, use_tracking_weights: bool = False
) -> torch.Tensor:
  value = _body_link_dist_mean_for_group(
    env, "ankles", use_tracking_weights=use_tracking_weights
  ) * _active_after_startup(env)
  _safe_log(env, "Metric/ankle_link_dist_mean", value)
  return value


def ankle_wrist_link_dist_mean_metric(
  env, use_tracking_weights: bool = False
) -> torch.Tensor:
  value = _body_link_dist_mean_for_group(
    env, "ankle_wrist", use_tracking_weights=use_tracking_weights
  ) * _active_after_startup(env)
  _safe_log(env, "Metric/ankle_wrist_link_dist_mean", value)
  return value


def left_wrist_link_dist_mean_metric(env) -> torch.Tensor:
  value = _body_link_dist_mean_for_group(env, "left_wrist") * _active_after_startup(env)
  _safe_log(env, "Metric/left_wrist_link_dist_mean", value)
  return value


def right_wrist_link_dist_mean_metric(env) -> torch.Tensor:
  value = _body_link_dist_mean_for_group(env, "right_wrist") * _active_after_startup(
    env
  )
  _safe_log(env, "Metric/right_wrist_link_dist_mean", value)
  return value


def body_link_axis_abs_metric(
  env,
  link_group: str,
  axis: str,
  log_key: str,
) -> torch.Tensor:
  value = _body_link_axis_abs_mean_for_group(
    env, link_group, axis
  ) * _active_after_startup(env)
  _safe_log(env, f"Metric/{log_key}", value)
  return value


def hand_action_mse_metric(env) -> torch.Tensor:
  teacher = apple_mdp.teacher_action(env)[:, NUM_BODY:ACTION_DIM]
  action = env.action_manager.action[:, NUM_BODY:ACTION_DIM]
  value = (action - teacher).pow(2).mean(dim=-1) * _active_after_startup(env)
  _safe_log(env, "Metric/hand_action_mse", value)
  return value


def hand_err_mse_metric(env) -> torch.Tensor:
  _, hand_err = _body_hand_err_mse(env)
  _safe_log(env, "Metric/hand_err_mse", hand_err)
  return hand_err


def raw_contact_arm_err_mse_metric(env) -> torch.Tensor:
  arm_err, _, _, window = _raw_contact_pose_errors(env)
  value = arm_err * window
  _safe_log(env, "Metric/raw_contact_arm_err_mse", value)
  return value


def raw_contact_hand_err_mse_metric(env) -> torch.Tensor:
  _, hand_err, _, window = _raw_contact_pose_errors(env)
  value = hand_err * window
  _safe_log(env, "Metric/raw_contact_hand_err_mse", value)
  return value


def raw_contact_object_pos_err_metric(env) -> torch.Tensor:
  _, _, object_pos_err, window = _raw_contact_pose_errors(env)
  value = object_pos_err * window
  _safe_log(env, "Metric/raw_contact_object_pos_err", value)
  return value


def raw_contact_window_metric(env) -> torch.Tensor:
  _, _, _, window = _raw_contact_pose_errors(env)
  _safe_log(env, "Metric/raw_contact_window", window)
  return window


def raw_tip_object_dist_err_metric(
  env,
  top_k: int = 2,
  start_frame: int = 380,
  end_frame: int = 440,
  margin_frames: int = 20,
  log_key: str = "Metric/raw_tip_object_dist_err",
) -> torch.Tensor:
  dist_err, _, window = _raw_tip_object_errors(
    env,
    start_frame=start_frame,
    end_frame=end_frame,
    margin_frames=margin_frames,
    top_k=top_k,
  )
  value = dist_err * window
  _safe_log(env, log_key, value)
  return value


def raw_tip_object_ref_dist_metric(
  env,
  top_k: int = 2,
  start_frame: int = 380,
  end_frame: int = 440,
  margin_frames: int = 20,
  log_key: str = "Metric/raw_tip_object_ref_dist",
) -> torch.Tensor:
  _, ref_dist, window = _raw_tip_object_errors(
    env,
    start_frame=start_frame,
    end_frame=end_frame,
    margin_frames=margin_frames,
    top_k=top_k,
  )
  value = ref_dist * window
  _safe_log(env, log_key, value)
  return value


def raw_tip_object_dist_err_cond_metric(
  env,
  top_k: int = 2,
  start_frame: int = 380,
  end_frame: int = 440,
  margin_frames: int = 20,
  log_key: str = "Metric/raw_tip_object_dist_err_cond",
) -> torch.Tensor:
  dist_err, _, window = _raw_tip_object_errors(
    env,
    start_frame=start_frame,
    end_frame=end_frame,
    margin_frames=margin_frames,
    top_k=top_k,
  )
  value = _window_conditional_as_env(dist_err, window)
  _safe_log(env, log_key, value)
  return value


def raw_tip_object_ref_dist_cond_metric(
  env,
  top_k: int = 2,
  start_frame: int = 380,
  end_frame: int = 440,
  margin_frames: int = 20,
  log_key: str = "Metric/raw_tip_object_ref_dist_cond",
) -> torch.Tensor:
  _, ref_dist, window = _raw_tip_object_errors(
    env,
    start_frame=start_frame,
    end_frame=end_frame,
    margin_frames=margin_frames,
    top_k=top_k,
  )
  value = _window_conditional_as_env(ref_dist, window)
  _safe_log(env, log_key, value)
  return value


def raw_tip_object_dist_err_top1_metric(env) -> torch.Tensor:
  return raw_tip_object_dist_err_metric(
    env, top_k=1, log_key="Metric/raw_tip_object_dist_err_top1"
  )


def raw_tip_object_ref_dist_top1_metric(env) -> torch.Tensor:
  return raw_tip_object_ref_dist_metric(
    env, top_k=1, log_key="Metric/raw_tip_object_ref_dist_top1"
  )


def raw_tip_object_dist_err_top2_metric(env) -> torch.Tensor:
  return raw_tip_object_dist_err_metric(
    env, top_k=2, log_key="Metric/raw_tip_object_dist_err_top2"
  )


def raw_tip_object_ref_dist_top2_metric(env) -> torch.Tensor:
  return raw_tip_object_ref_dist_metric(
    env, top_k=2, log_key="Metric/raw_tip_object_ref_dist_top2"
  )


def raw_tip_object_dist_err_top4_metric(env) -> torch.Tensor:
  return raw_tip_object_dist_err_metric(
    env, top_k=4, log_key="Metric/raw_tip_object_dist_err_top4"
  )


def raw_tip_object_ref_dist_top4_metric(env) -> torch.Tensor:
  return raw_tip_object_ref_dist_metric(
    env, top_k=4, log_key="Metric/raw_tip_object_ref_dist_top4"
  )


def raw_tip_object_dist_err_cond_top1_metric(env) -> torch.Tensor:
  return raw_tip_object_dist_err_cond_metric(
    env, top_k=1, log_key="Metric/raw_tip_object_dist_err_cond_top1"
  )


def raw_tip_object_ref_dist_cond_top1_metric(env) -> torch.Tensor:
  return raw_tip_object_ref_dist_cond_metric(
    env, top_k=1, log_key="Metric/raw_tip_object_ref_dist_cond_top1"
  )


def raw_tip_object_dist_err_cond_top2_metric(env) -> torch.Tensor:
  return raw_tip_object_dist_err_cond_metric(
    env, top_k=2, log_key="Metric/raw_tip_object_dist_err_cond_top2"
  )


def raw_tip_object_ref_dist_cond_top2_metric(env) -> torch.Tensor:
  return raw_tip_object_ref_dist_cond_metric(
    env, top_k=2, log_key="Metric/raw_tip_object_ref_dist_cond_top2"
  )


def raw_tip_object_dist_err_cond_top4_metric(env) -> torch.Tensor:
  return raw_tip_object_dist_err_cond_metric(
    env, top_k=4, log_key="Metric/raw_tip_object_dist_err_cond_top4"
  )


def raw_tip_object_ref_dist_cond_top4_metric(env) -> torch.Tensor:
  return raw_tip_object_ref_dist_cond_metric(
    env, top_k=4, log_key="Metric/raw_tip_object_ref_dist_cond_top4"
  )


def raw_tip_radial_err_metric(
  env,
  top_k: int = 1,
  start_frame: int = 380,
  end_frame: int = 440,
  margin_frames: int = 20,
  log_key: str = "Metric/raw_tip_radial_err",
) -> torch.Tensor:
  radial_err, _, _, window = _raw_tip_object_radial_errors(
    env,
    start_frame=start_frame,
    end_frame=end_frame,
    margin_frames=margin_frames,
    top_k=top_k,
  )
  value = radial_err * window
  _safe_log(env, log_key, value)
  return value


def raw_tip_radial_err_cond_metric(
  env,
  top_k: int = 1,
  start_frame: int = 380,
  end_frame: int = 440,
  margin_frames: int = 20,
  log_key: str = "Metric/raw_tip_radial_err_cond",
) -> torch.Tensor:
  radial_err, _, _, window = _raw_tip_object_radial_errors(
    env,
    start_frame=start_frame,
    end_frame=end_frame,
    margin_frames=margin_frames,
    top_k=top_k,
  )
  value = _window_conditional_as_env(radial_err, window)
  _safe_log(env, log_key, value)
  return value


def raw_tip_radial_live_dist_cond_metric(
  env,
  top_k: int = 1,
  start_frame: int = 380,
  end_frame: int = 440,
  margin_frames: int = 20,
  log_key: str = "Metric/raw_tip_radial_live_dist_cond",
) -> torch.Tensor:
  _, live_dist, _, window = _raw_tip_object_radial_errors(
    env,
    start_frame=start_frame,
    end_frame=end_frame,
    margin_frames=margin_frames,
    top_k=top_k,
  )
  value = _window_conditional_as_env(live_dist, window)
  _safe_log(env, log_key, value)
  return value


def raw_tip_radial_ref_dist_cond_metric(
  env,
  top_k: int = 1,
  start_frame: int = 380,
  end_frame: int = 440,
  margin_frames: int = 20,
  log_key: str = "Metric/raw_tip_radial_ref_dist_cond",
) -> torch.Tensor:
  _, _, ref_dist, window = _raw_tip_object_radial_errors(
    env,
    start_frame=start_frame,
    end_frame=end_frame,
    margin_frames=margin_frames,
    top_k=top_k,
  )
  value = _window_conditional_as_env(ref_dist, window)
  _safe_log(env, log_key, value)
  return value


def hand_to_obj_dist_metric(env) -> torch.Tensor:
  value = _tip_distances(env).min(dim=-1).values
  _safe_log(env, "Metric/hand_to_obj_dist", value)
  return value


class EpisodeAnyTipUnderMetric:
  """Sticky per-episode success for ever reaching a hand-object distance."""

  def __init__(self, cfg, env):
    self.threshold_m = float(cfg.params["threshold_m"])
    self.log_key = str(cfg.params["log_key"])
    self.device = env.device
    self.num_envs = env.num_envs
    self._state = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

  def reset(self, env_ids=None):
    if env_ids is None:
      env_ids = slice(None)
    self._state[env_ids] = 0.0

  def __call__(self, env, **kwargs) -> torch.Tensor:
    del kwargs
    if self._state.shape[0] != env.num_envs or str(self._state.device) != env.device:
      self.device = env.device
      self.num_envs = env.num_envs
      self._state = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    current = (
      _tip_distances(env).min(dim=-1).values <= self.threshold_m
    ).float() * _active_after_startup(env)
    self._state = torch.maximum(self._state, current)
    _safe_log(env, self.log_key, self._state)
    return self._state


def _stable_not_fallen_value(env) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  active = _active_after_startup(env)
  root_z_ok = (robot.data.root_link_pos_w[:, 2] >= 0.45).float()
  up_local = torch.zeros(env.num_envs, 3, device=env.device)
  up_local[:, 2] = 1.0
  up_w = quat_apply(robot.data.root_link_quat_w, up_local)
  upright_ok = (up_w[:, 2] >= 0.55).float()
  return root_z_ok * upright_ok * active


class EpisodeStableAnyTipUnderMetric(EpisodeAnyTipUnderMetric):
  """Sticky reach success gated by not-fallen/upright body state."""

  def __call__(self, env, **kwargs) -> torch.Tensor:
    del kwargs
    if self._state.shape[0] != env.num_envs or str(self._state.device) != env.device:
      self.device = env.device
      self.num_envs = env.num_envs
      self._state = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    current = (
      _tip_distances(env).min(dim=-1).values <= self.threshold_m
    ).float() * _stable_not_fallen_value(env)
    self._state = torch.maximum(self._state, current)
    _safe_log(env, self.log_key, self._state)
    return self._state


class PhaseALiftSuccessMetric:
  """Sticky success for lifting 3 cm with strict contact held for 0.5 seconds."""

  def __init__(self, cfg, env):
    self.lift_height_m = float(cfg.params.get("lift_height_m", 0.03))
    self.hold_duration_s = float(cfg.params.get("hold_duration_s", 0.5))
    self.min_contact_tips = int(cfg.params.get("min_contact_tips", 2))
    self.device = env.device
    self.num_envs = env.num_envs
    self._duration = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
    self._success = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

  def reset(self, env_ids=None):
    if env_ids is None:
      env_ids = slice(None)
    self._duration[env_ids] = 0.0
    self._success[env_ids] = 0.0

  def __call__(self, env, **kwargs) -> torch.Tensor:
    del kwargs
    if (
      self._success.shape[0] != env.num_envs or str(self._success.device) != env.device
    ):
      self.device = env.device
      self.num_envs = env.num_envs
      self._duration = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
      self._success = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    obj: Entity = object_pool.active(env)
    ref = _ref(env.device)
    ref_z0 = _reference_object_pos_w(env, ref, 0)[:, 2]
    lifted = obj.data.root_link_pos_w[:, 2] - ref_z0 >= self.lift_height_m
    physical_contact, grasp_contact, force_close = _physical_contact(env)
    contact_count = _primary_contact_from_history(env).sum(dim=-1)
    multitip = contact_count >= max(self.min_contact_tips, 1)
    strict_contact = (grasp_contact > 0.5) | (force_close > 0.5) | multitip
    active = _active_after_startup(env) > 0.5
    holding = lifted & strict_contact & active
    self._duration = torch.where(
      holding,
      self._duration + float(env.step_dt),
      torch.zeros_like(self._duration),
    )
    current = (self._duration >= self.hold_duration_s).float()
    self._success = torch.maximum(self._success, current)
    _safe_log(env, "PhaseA/lift_duration_s", self._duration)
    _safe_log(env, "PhaseA/lift_success", self._success)
    return self._success


def obj_drift_metric(env) -> torch.Tensor:
  value = _object_drift(env)
  _safe_log(env, "Metric/obj_drift", value)
  return value


def live_contact_006_metric(
  env, threshold: float = LIVE_CONTACT_THRESHOLD
) -> torch.Tensor:
  geometric = (_tip_distances(env).min(dim=-1).values <= float(threshold)).float()
  physical, _, _ = _physical_contact(env)
  value = torch.maximum(geometric, physical) * _active_after_startup(env)
  _safe_log(env, "PhaseA/live_contact_006", value)
  return value


def ttr_at_012_metric(env, threshold: float = 0.12) -> torch.Tensor:
  value = (_object_drift(env) <= float(threshold)).float()
  _safe_log(env, "PhaseA/ttr_at_012", value)
  return value


def object_mpjpe_mm_metric(env) -> torch.Tensor:
  value = 1000.0 * _object_drift(env)
  _safe_log(env, "PhaseA/object_mpjpe_mm", value)
  return value


def sequence_success_metric(env, threshold: float = 0.12) -> torch.Tensor:
  ref = _ref(env.device)
  frame = _tracking_frame(env, ref["n_frames"])
  # the frame is a global row; the end of a clip is that clip's own last row
  _, _hi = apple_mdp._clip_bounds(env, int(ref["n_frames"]))
  at_end = frame >= _hi
  value = (at_end & (_object_drift(env) <= float(threshold))).float()
  _safe_log(env, "PhaseA/sequence_success", value)
  return value


def obj_speed_metric(env) -> torch.Tensor:
  obj: Entity = object_pool.active(env)
  value = obj.data.root_link_vel_w[:, :3].norm(dim=-1)
  _safe_log(env, "Metric/obj_speed", value)
  return value


def ep_len_metric(env) -> torch.Tensor:
  value = env.episode_length_buf.float()
  _safe_log(env, "Metric/ep_len", value)
  return value


def reference_start_frame_metric(env) -> torch.Tensor:
  ref = _ref(env.device)
  value = apple_mdp._reference_start_frame(env, ref["n_frames"]).float()
  _safe_log(env, "Metric/reference_start_frame", value)
  return value


def tracking_frame_metric(env) -> torch.Tensor:
  ref = _ref(env.device)
  n = int(ref["n_frames"])
  # Report progress INSIDE this env's clip. The global row is not a frame number: it
  # carries clip_id * n_frames, so averaging it across clips yields a value that
  # matches no frame of either clip.
  value = apple_mdp.local_tracking_frame(env, n).float()
  _safe_log(env, "Metric/tracking_frame", value)
  _safe_log(env, "Metric/tracking_frame_global_row", _tracking_frame(env, n).float())
  return value


def near_contact_flag(env) -> torch.Tensor:
  value = (
    _tip_distances(env).min(dim=-1).values <= LIVE_CONTACT_THRESHOLD
  ).float() * _active_after_startup(env)
  _safe_log(env, "Stage/near_contact", value)
  return value


def physical_contact_flag(env) -> torch.Tensor:
  value, _, _ = _physical_contact(env)
  value = value * _active_after_startup(env)
  _safe_log(env, "Stage/physical_contact", value)
  return value


def hand_body_contact_flag(env) -> torch.Tensor:
  value = _hand_body_contact(env) * _active_after_startup(env)
  _safe_log(env, "Stage/hand_body_contact", value)
  return value


def non_tip_hand_body_contact_flag(env) -> torch.Tensor:
  value = _non_tip_hand_body_contact(env) * _active_after_startup(env)
  _safe_log(env, "Stage/non_tip_hand_body_contact", value)
  return value


def force_close_flag(env) -> torch.Tensor:
  _, _, value = _physical_contact(env)
  value = value * _active_after_startup(env)
  _safe_log(env, "Stage/force_close", value)
  return value


def object_moving_flag(env, speed_threshold: float = 0.05) -> torch.Tensor:
  value = (
    obj_speed_metric(env) > float(speed_threshold)
  ).float() * _active_after_startup(env)
  _safe_log(env, "Stage/object_moving", value)
  return value


def stable_not_fallen_flag(env) -> torch.Tensor:
  value = _stable_not_fallen_value(env)
  _safe_log(env, "Stage/stable_not_fallen", value)
  return value


# --- Stage B: human-derived enveloping grasp layout -------------------------
# Target fingertip-to-object-centre distances and the thumb/middle opposition
# angle were measured on GRAB s1/apple_lift (subject's right hand) at its best
# enveloping frame and rescaled to the sim apple radius.  See
# GRASP_LAYOUT_DESIGN.md.  Order is (thumb, index, middle, ring, pinky).
GRASP_LAYOUT_TIP_TARGETS_M = (0.0418, 0.0372, 0.0408, 0.0406, 0.0513)
GRASP_LAYOUT_OPPOSITION_COS = -0.871  # thumb vs middle, 150.6 degrees


def _grasp_opposition_cos(
  env,
  hand: str = "right",
  oppose_pair: tuple[int, int] = (0, 2),
) -> torch.Tensor:
  """Cosine between the thumb and finger directions seen from the object centre.

  +1 means both are on the SAME side (the hand is poking at the object), -1 means
  they are opposed around it (what an enveloping grasp needs; the reference's own
  grasp sits at -0.871).
  """
  robot: Entity = env.scene["robot"]
  obj: Entity = object_pool.active(env)
  ids = _tip_body_ids(env)
  sl = slice(5, 10) if str(hand).strip().lower().startswith("r") else slice(0, 5)
  rel = robot.data.body_link_pos_w[:, ids[sl]] - obj.data.root_link_pos_w.unsqueeze(1)
  unit = rel / rel.norm(dim=-1, keepdim=True).clamp_min(1.0e-9)
  a, b = int(oppose_pair[0]), int(oppose_pair[1])
  return (unit[:, a] * unit[:, b]).sum(dim=-1)


def _opposition_gate(
  env,
  floor: float,
  hi: float,
  lo: float,
  hand: str,
  oppose_pair: tuple[int, int],
) -> torch.Tensor:
  """Scale in [floor, 1] that opens as the hand rotates into opposition.

  Measured across three unrelated policies (hands-only, hands-only with a full
  approach, and arms+hands with reference editing), the fingertip reaches 0.28 cm
  from the apple surface while the opposition cosine never drops below +0.62 --
  i.e. poking with one fingertip already saturates the tip-surface reward, so
  there is no reason to ever rotate the palm.  This gate removes most of that
  payout unless the palm is actually around the object.

  The floor is deliberately non-zero and the ramp deliberately spans the whole
  cosine range: a hard gate would be exactly 0 at the current +0.92 and kill the
  gradient, which is the same mistake as putting a narrow Gaussian on a quantity
  that starts 6 sigma away.
  """
  f = min(max(float(floor), 0.0), 1.0)
  if f >= 1.0:
    return torch.ones(env.num_envs, device=env.device)
  span = max(float(hi) - float(lo), 1.0e-6)
  ramp = ((float(hi) - _grasp_opposition_cos(env, hand, oppose_pair)) / span).clamp(
    0.0, 1.0
  )
  return f + (1.0 - f) * ramp


def residual_grasp_layout_reward(
  env,
  tip_target_dists: tuple[float, ...] = GRASP_LAYOUT_TIP_TARGETS_M,
  radial_std: float = 0.02,
  radial_coarse_std: float = 0.12,
  radial_coarse_weight: float = 0.5,
  opposition_target: float = GRASP_LAYOUT_OPPOSITION_COS,
  opposition_std: float = 0.30,
  opposition_mode: str = "monotonic",
  radial_weight: float = 0.6,
  opposition_weight: float = 0.4,
  oppose_pair: tuple[int, int] = (0, 2),
  hand: str = "right",
) -> torch.Tensor:
  """Reward an enveloping grasp layout around the object.

  Deliberately rotation invariant: it constrains each fingertip's distance to
  the object centre and the thumb/finger opposition angle, but never the wrist
  pose.  The retargeted reference wrist pose does not afford a grasp (its thumb
  sits 9-14 cm from the apple centre), so rewarding it is counter-productive;
  this term instead lets the policy pick any approach that wraps the object.
  """
  robot: Entity = env.scene["robot"]
  obj: Entity = object_pool.active(env)
  ids = _tip_body_ids(env)
  sl = slice(5, 10) if str(hand).strip().lower().startswith("r") else slice(0, 5)
  tip_pos = robot.data.body_link_pos_w[:, ids[sl]]
  rel = tip_pos - obj.data.root_link_pos_w.unsqueeze(1)
  dist = rel.norm(dim=-1)

  target = torch.tensor(
    tuple(float(v) for v in tip_target_dists),
    device=dist.device,
    dtype=dist.dtype,
  )
  err = dist - target.unsqueeze(0)
  # Two scales on purpose: the fine Gaussian is what a real grasp must satisfy,
  # but on its own it is identically zero once the hand is more than ~6 cm off,
  # which is exactly where an untrained policy starts (~17 cm).  The coarse
  # Gaussian keeps a usable gradient over that whole approach range.
  std_fine = max(float(radial_std), 1.0e-6)
  std_coarse = max(float(radial_coarse_std), 1.0e-6)
  radial_fine = torch.exp(-0.5 * (err / std_fine) ** 2).mean(dim=-1)
  radial_coarse = torch.exp(-0.5 * (err / std_coarse) ** 2).mean(dim=-1)
  w_c = min(max(float(radial_coarse_weight), 0.0), 1.0)
  radial = (1.0 - w_c) * radial_fine + w_c * radial_coarse

  unit = rel / rel.norm(dim=-1, keepdim=True).clamp_min(1.0e-9)
  a, b = int(oppose_pair[0]), int(oppose_pair[1])
  oppose_cos = (unit[:, a] * unit[:, b]).sum(dim=-1)
  # A Gaussian on the cosine is a trap: the untrained policy sits at cos ~ +0.96
  # while the target is -0.871, i.e. 6 sigma away at std=0.30, so the term is
  # ~7e-9 and the thumb never receives any pressure to rotate into opposition.
  # The monotonic form has useful gradient over the whole [-1, 1] range.
  if str(opposition_mode).strip().lower() == "gaussian":
    std_o = max(float(opposition_std), 1.0e-6)
    opposition = torch.exp(
      -0.5 * ((oppose_cos - float(opposition_target)) / std_o) ** 2
    )
  else:
    opposition = ((1.0 - oppose_cos) * 0.5).clamp(0.0, 1.0)

  w_r, w_o = float(radial_weight), float(opposition_weight)
  value = (w_r * radial + w_o * opposition) / max(w_r + w_o, 1.0e-6)
  value = value * _active_after_startup(env)

  _safe_log(env, "GraspLayout/radial", radial)
  _safe_log(env, "GraspLayout/radial_fine", radial_fine)
  _safe_log(env, "GraspLayout/radial_coarse", radial_coarse)
  _safe_log(env, "GraspLayout/opposition", opposition)
  _safe_log(env, "GraspLayout/oppose_cos", oppose_cos)
  _safe_log(
    env,
    "GraspLayout/tip_dist_abs_err",
    (dist - target.unsqueeze(0)).abs().mean(dim=-1),
  )
  _safe_log(env, "ResidualReward/grasp_layout", value)
  return value


def fell_over_early(
  env,
  min_root_z: float = 0.65,
  min_upright: float = 0.80,
  asset_cfg: SceneEntityCfg = _ROBOT_ENTITY_CFG,
) -> torch.Tensor:
  """Terminate as soon as a fall is unrecoverable instead of after full collapse.

  The stock `fell_over` fires at root_z < 0.45 while the pelvis starts near
  0.79 m, so every doomed episode keeps simulating through the entire topple --
  pure wasted rollout.  It also ignores tilt, so a robot that is already past
  recovery but still tall keeps running.

  Safe to make aggressive here: over the whole apple_lift reference the root
  height stays in 0.784-0.817 m and the torso uprightness (world z of the body
  z-axis) stays in 0.9745-1.0, so these thresholds cannot clip valid reaching.

  Set both thresholds <= 0 to disable.
  """
  if float(min_root_z) <= 0.0 and float(min_upright) <= 0.0:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  robot: Entity = env.scene[asset_cfg.name]
  active = _active_after_startup(env).bool()
  root_z = robot.data.root_link_pos_w[:, 2]
  up_local = torch.zeros(env.num_envs, 3, device=env.device)
  up_local[:, 2] = 1.0
  up_w = quat_apply(robot.data.root_link_quat_w, up_local)
  too_low = root_z < float(min_root_z)
  too_tilted = up_w[:, 2] < float(min_upright)
  value = (too_low | too_tilted) & active
  _safe_log(env, "Stage/fell_over_early", value.float())
  _safe_log(env, "Stage/upright_z", up_w[:, 2])
  return value


# The retargeted reference never opposes the thumb: its
# right_hand_thumb_rota_joint2 peaks at 0.309 rad over the whole clip while the
# IK-solved envelope grasp of the 4 cm apple needs ~0.639 rad.  Tracking the
# reference hand therefore can never produce a grasp, so this rewards the one
# degree of freedom that is missing, directly.
# The thumb is finger1 on the Wuji hand; the xhand name does not exist in that model, so
# hard-coding it made every reward reading this joint silently unusable under wuji.
THUMB_OPPOSITION_JOINT = (
  "right_hand_thumb_rota_joint2" if _HAND_KIND == "xhand" else "right_finger1_joint2"
)
THUMB_OPPOSITION_TARGET_RAD = 0.6394


def residual_thumb_opposition_reward(
  env,
  joint_name: str = THUMB_OPPOSITION_JOINT,
  target_rad: float = THUMB_OPPOSITION_TARGET_RAD,
  std: float = 0.40,
  near_threshold: float = 0.25,
) -> torch.Tensor:
  """Reward the thumb rotating into opposition while the hand is near the object.

  Gated on proximity so it only shapes the pregrasp and does not fight reference
  tracking during the rest of the clip.  std is deliberately wide: the joint
  starts near 0.11 rad, so the error is ~0.53 rad and a narrow std would be dead.
  """
  robot: Entity = env.scene["robot"]
  ids, _ = robot.find_joints([str(joint_name)], preserve_order=True)
  q = robot.data.joint_pos[:, ids[0]]
  s = max(float(std), 1.0e-6)
  score = torch.exp(-0.5 * ((q - float(target_rad)) / s) ** 2)
  near = (_tip_distances(env).min(dim=-1).values <= float(near_threshold)).float()
  value = score * near * _active_after_startup(env)
  _safe_log(env, "GraspLayout/thumb_q", q)
  _safe_log(env, "GraspLayout/thumb_score", score)
  _safe_log(env, "ResidualReward/thumb_opposition", value)
  return value
