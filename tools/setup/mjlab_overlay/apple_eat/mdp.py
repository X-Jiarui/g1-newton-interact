"""MDP terms for the MJLab apple-eat PPO prototype."""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_conjugate,
  quat_mul,
)

NUM_BODY = 29

# Which dexterous hand the robot wears. "xhand" is the original 12-DOF-per-hand hand; "wuji" is the
# Wuji generation-1 hand (repo path hand/, NOT hand2/) with 5 fingers x 4 independently actuated
# joints = 20 per hand. Everything downstream reads NUM_HAND and HAND_DOF_NAMES, so this switch is
# the only place the choice is made.
HAND_KIND = os.environ.get("APPLE_HAND_KIND", "xhand").strip().lower()
if HAND_KIND not in ("xhand", "wuji"):
  raise ValueError(f"APPLE_HAND_KIND must be 'xhand' or 'wuji', got {HAND_KIND!r}")

_WUJI_HAND_DOF_NAMES = ('left_finger1_joint1', 'left_finger1_joint2', 'left_finger1_joint3', 'left_finger1_joint4', 'left_finger2_joint1', 'left_finger2_joint2', 'left_finger2_joint3', 'left_finger2_joint4', 'left_finger3_joint1', 'left_finger3_joint2', 'left_finger3_joint3', 'left_finger3_joint4', 'left_finger4_joint1', 'left_finger4_joint2', 'left_finger4_joint3', 'left_finger4_joint4', 'left_finger5_joint1', 'left_finger5_joint2', 'left_finger5_joint3', 'left_finger5_joint4', 'right_finger1_joint1', 'right_finger1_joint2', 'right_finger1_joint3', 'right_finger1_joint4', 'right_finger2_joint1', 'right_finger2_joint2', 'right_finger2_joint3', 'right_finger2_joint4', 'right_finger3_joint1', 'right_finger3_joint2', 'right_finger3_joint3', 'right_finger3_joint4', 'right_finger4_joint1', 'right_finger4_joint2', 'right_finger4_joint3', 'right_finger4_joint4', 'right_finger5_joint1', 'right_finger5_joint2', 'right_finger5_joint3', 'right_finger5_joint4')

NUM_HAND = 24 if HAND_KIND == "xhand" else len(_WUJI_HAND_DOF_NAMES)

# Fingertip bodies, in the order callers slice: left thumb..pinky then right thumb..pinky, so
# RIGHT = slice(5, 10) keeps working. Wuji finger1 is the thumb.
_XHAND_TIP_BODY_NAMES = ('left_hand_thumb_rota_tip', 'left_hand_index_rota_tip', 'left_hand_mid_tip', 'left_hand_ring_tip', 'left_hand_pinky_tip', 'right_hand_thumb_rota_tip', 'right_hand_index_rota_tip', 'right_hand_mid_tip', 'right_hand_ring_tip', 'right_hand_pinky_tip')
_WUJI_TIP_BODY_NAMES = ('left_finger1_tip', 'left_finger2_tip', 'left_finger3_tip', 'left_finger4_tip', 'left_finger5_tip', 'right_finger1_tip', 'right_finger2_tip', 'right_finger3_tip', 'right_finger4_tip', 'right_finger5_tip')
TIP_BODY_NAMES = (
  _XHAND_TIP_BODY_NAMES if HAND_KIND == "xhand" else _WUJI_TIP_BODY_NAMES
)
# Column layout of the retargeted reference (qpos[7:] of the robot MJCF). The hands are grafted
# onto the wrist bodies, so finger joints sit INSIDE the arm chain and the right arm's start index
# depends on how many joints the left hand contributes: xhand 12/side -> right arm at 34, wuji
# 20/side -> right arm at 42. The names keep the _53 suffix because ~6 call sites use them.
if HAND_KIND == "xhand":
  _RIGHT_ARM_START = 34
  _PER_HAND = 12
else:
  _RIGHT_ARM_START = 42
  _PER_HAND = 20
REF_DOF_DIM = 22 + 2 * _PER_HAND + 7
BODY_IDX_IN_53 = np.array(
  list(range(22)) + list(range(_RIGHT_ARM_START, _RIGHT_ARM_START + 7)), dtype=np.int64
)
HAND_IDX_IN_53 = np.array(
  list(range(22, 22 + _PER_HAND))
  + list(range(_RIGHT_ARM_START + 7, _RIGHT_ARM_START + 7 + _PER_HAND)),
  dtype=np.int64,
)
OBS_DIM_NO_TEACHER = 2964
ENC_INPUT_DIM = 1254
BODY_DEC_HIST_DIM = 930
HAND_DEC_HIST_DIM = 780

_ROBOT_ENTITY_CFG = SceneEntityCfg("robot")
_APPLE_ENTITY_CFG = SceneEntityCfg("apple")

BODY_29_DOF_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)
ISAACLAB_BODY_ORDER = (
  "left_hip_pitch_joint",
  "right_hip_pitch_joint",
  "waist_yaw_joint",
  "left_hip_roll_joint",
  "right_hip_roll_joint",
  "waist_roll_joint",
  "left_hip_yaw_joint",
  "right_hip_yaw_joint",
  "waist_pitch_joint",
  "left_knee_joint",
  "right_knee_joint",
  "left_shoulder_pitch_joint",
  "right_shoulder_pitch_joint",
  "left_ankle_pitch_joint",
  "right_ankle_pitch_joint",
  "left_shoulder_roll_joint",
  "right_shoulder_roll_joint",
  "left_ankle_roll_joint",
  "right_ankle_roll_joint",
  "left_shoulder_yaw_joint",
  "right_shoulder_yaw_joint",
  "left_elbow_joint",
  "right_elbow_joint",
  "left_wrist_roll_joint",
  "right_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "right_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_wrist_yaw_joint",
)

_XHAND_24_DOF_NAMES = (
  "left_hand_thumb_bend_joint",
  "left_hand_thumb_rota_joint1",
  "left_hand_thumb_rota_joint2",
  "left_hand_index_bend_joint",
  "left_hand_index_joint1",
  "left_hand_index_joint2",
  "left_hand_mid_joint1",
  "left_hand_mid_joint2",
  "left_hand_ring_joint1",
  "left_hand_ring_joint2",
  "left_hand_pinky_joint1",
  "left_hand_pinky_joint2",
  "right_hand_thumb_bend_joint",
  "right_hand_thumb_rota_joint1",
  "right_hand_thumb_rota_joint2",
  "right_hand_index_bend_joint",
  "right_hand_index_joint1",
  "right_hand_index_joint2",
  "right_hand_mid_joint1",
  "right_hand_mid_joint2",
  "right_hand_ring_joint1",
  "right_hand_ring_joint2",
  "right_hand_pinky_joint1",
  "right_hand_pinky_joint2",
)

# The xhand list above is kept verbatim; HAND_DOF_NAMES is what the code should read,
# and HAND_24_DOF_NAMES stays as an alias so the ~25 importing modules need no edit.
HAND_DOF_NAMES = (
  _XHAND_24_DOF_NAMES if HAND_KIND == "xhand" else _WUJI_HAND_DOF_NAMES
)
HAND_24_DOF_NAMES = HAND_DOF_NAMES

PKL_FOR_IL = np.array(
  [
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    22,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
  ],
  dtype=np.int64,
)
IL_FOR_PKL = np.array(
  [
    0,
    3,
    6,
    9,
    13,
    17,
    1,
    4,
    7,
    10,
    14,
    18,
    2,
    5,
    8,
    11,
    15,
    19,
    21,
    23,
    25,
    27,
    12,
    16,
    20,
    22,
    24,
    26,
    28,
  ],
  dtype=np.int64,
)

SONIC_DEFAULT_ANGLES_PKL = np.array(
  [
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    0.0,
    0.0,
    0.0,
    0.2,
    0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
    0.2,
    -0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
  ],
  dtype=np.float32,
)
_ARMATURE_5020 = 0.003609725
_ARMATURE_7520_14 = 0.010177520
_ARMATURE_7520_22 = 0.025101925
_ARMATURE_4010 = 0.00425
_OMEGA = 10.0 * 2.0 * np.pi
_S5020 = _ARMATURE_5020 * _OMEGA**2
_S7514 = _ARMATURE_7520_14 * _OMEGA**2
_S7522 = _ARMATURE_7520_22 * _OMEGA**2
_S4010 = _ARMATURE_4010 * _OMEGA**2
_D5020 = 2.0 * 2.0 * _ARMATURE_5020 * _OMEGA
_D7514 = 2.0 * 2.0 * _ARMATURE_7520_14 * _OMEGA
_D7522 = 2.0 * 2.0 * _ARMATURE_7520_22 * _OMEGA
_D4010 = 2.0 * 2.0 * _ARMATURE_4010 * _OMEGA
_E5020, _E7514, _E7522, _E4010 = 25.0, 88.0, 139.0, 5.0
KP_BODY_PKL = np.array(
  [
    _S7522,
    _S7522,
    _S7514,
    _S7522,
    2 * _S5020,
    2 * _S5020,
    _S7522,
    _S7522,
    _S7514,
    _S7522,
    2 * _S5020,
    2 * _S5020,
    _S7514,
    2 * _S5020,
    2 * _S5020,
    _S5020,
    _S5020,
    _S5020,
    _S5020,
    _S5020,
    _S4010,
    _S4010,
    _S5020,
    _S5020,
    _S5020,
    _S5020,
    _S5020,
    _S4010,
    _S4010,
  ],
  dtype=np.float32,
)
KD_BODY_PKL = np.array(
  [
    _D7522,
    _D7522,
    _D7514,
    _D7522,
    2 * _D5020,
    2 * _D5020,
    _D7522,
    _D7522,
    _D7514,
    _D7522,
    2 * _D5020,
    2 * _D5020,
    _D7514,
    2 * _D5020,
    2 * _D5020,
    _D5020,
    _D5020,
    _D5020,
    _D5020,
    _D5020,
    _D4010,
    _D4010,
    _D5020,
    _D5020,
    _D5020,
    _D5020,
    _D5020,
    _D4010,
    _D4010,
  ],
  dtype=np.float32,
)
EFFORT_BODY_PKL = np.array(
  [
    _E7522,
    _E7522,
    _E7514,
    _E7522,
    _E5020,
    _E5020,
    _E7522,
    _E7522,
    _E7514,
    _E7522,
    _E5020,
    _E5020,
    _E7514,
    _E5020,
    _E5020,
    _E5020,
    _E5020,
    _E5020,
    _E5020,
    _E5020,
    _E4010,
    _E4010,
    _E5020,
    _E5020,
    _E5020,
    _E5020,
    _E5020,
    _E4010,
    _E4010,
  ],
  dtype=np.float32,
)
KP_53 = np.concatenate([KP_BODY_PKL, np.full(NUM_HAND, 300.0, dtype=np.float32)])
KD_53 = np.concatenate([KD_BODY_PKL, np.full(NUM_HAND, 8.0, dtype=np.float32)])
EFFORT_53 = np.concatenate([EFFORT_BODY_PKL, np.full(NUM_HAND, 30.0, dtype=np.float32)])
SONIC_ACTION_SCALE_PKL = np.array(
  [
    0.25 * _E7522 / _S7522,
    0.25 * _E7522 / _S7522,
    0.25 * _E7514 / _S7514,
    0.25 * _E7522 / _S7522,
    0.25 * _E5020 / _S5020,
    0.25 * _E5020 / _S5020,
    0.25 * _E7522 / _S7522,
    0.25 * _E7522 / _S7522,
    0.25 * _E7514 / _S7514,
    0.25 * _E7522 / _S7522,
    0.25 * _E5020 / _S5020,
    0.25 * _E5020 / _S5020,
    0.25 * _E7514 / _S7514,
    0.25 * _E5020 / _S5020,
    0.25 * _E5020 / _S5020,
    0.25 * _E5020 / _S5020,
    0.25 * _E5020 / _S5020,
    0.25 * _E5020 / _S5020,
    0.25 * _E5020 / _S5020,
    0.25 * _E5020 / _S5020,
    0.25 * _E4010 / _S4010,
    0.25 * _E4010 / _S4010,
    0.25 * _E5020 / _S5020,
    0.25 * _E5020 / _S5020,
    0.25 * _E5020 / _S5020,
    0.25 * _E5020 / _S5020,
    0.25 * _E5020 / _S5020,
    0.25 * _E4010 / _S4010,
    0.25 * _E4010 / _S4010,
  ],
  dtype=np.float32,
)
SONIC_DEFAULT_ANGLES_IL = SONIC_DEFAULT_ANGLES_PKL[PKL_FOR_IL]

_GROOT = Path(
  os.environ.get(
    "GROOT_ROOT",
    "/home/jiarui/projects/GR00T-WholeBodyControl",
  )
)
def _rank_index() -> int:
  """This worker's rank; 0 when not launched distributed."""
  return int(os.environ.get("RANK", "0"))


def _rank_world() -> int:
  return int(os.environ.get("WORLD_SIZE", "1"))


def rank_pick(name: str, default: str = "") -> str:
  """`name` from the environment. `name`_LIST is rejected, not honoured.

  A per-rank variant of this function was tried and does not work: torchrunx has not set RANK,
  LOCAL_RANK or WORLD_SIZE by the time this module is imported -- verified in the workers, not only
  the launcher -- so every rank resolved entry 0 and a two-clip run trained one clip twice while
  reporting nothing. Use APPLE_EAT_PKL_MIX for several clips in one batch; per-rank selection needs
  lazy resolution and is not implemented yet.
  """
  if os.environ.get(name + "_LIST", "").strip():
    raise RuntimeError(
      f"{name}_LIST is set, but per-rank selection at import time does not work: RANK/WORLD_SIZE "
      f"are unset when this module is imported, so every rank would take the first entry. "
      f"Use {name}_MIX for multi-clip training in one batch (see APPLE_EAT_PKL_MIX)."
    )
  return os.environ.get(name, default)


_APPLE_PKL = Path(
  rank_pick(
    "APPLE_EAT_PKL",
    str(
      (
        _GROOT / "interaction" / "scaled_grab_dataset_vfoot" / "s1" / "apple_eat_1.pkl"
      )
      if HAND_KIND == "xhand"
      # The wuji hand needs its own retarget: the reference stores per-joint finger angles, and
      # xhand's 12-per-hand columns have no correspondence to wuji's 5x4 joints. Derived from _GROOT
      # so the same code works on the office PC and inside the vast containers.
      else _GROOT / "interaction" / "scaled_grab_dataset_wuji" / "s1" / "apple_eat_1.pkl"
    ),
  )
)
_TARGET_FPS = 50.0
APPLE_RADIUS = 0.04
# Vertical shift applied to the apple AND the table (the table height is derived
# from the apple, so one offset moves both). Applied to the reference itself, so
# placement, rewards and object_reference_window all stay consistent.
SCENE_Z_OFFSET = float(os.environ.get("APPLE_SCENE_Z_OFFSET", "0.0"))
# Derive the table height from the apple's REST pose instead of its per-frame pose.
# With the per-frame derivation an RSI reset mid-carry moved the table up to +48 cm to sit
# under the lifted apple, which confined RSI to frames [0,94] and [421,572] and left the
# whole carry/put-back phase untrained. The table is static furniture; it should not follow
# the apple. Set APPLE_TABLE_FIXED_HEIGHT=0 to restore the old behaviour.
TABLE_FIXED_HEIGHT = os.environ.get("APPLE_TABLE_FIXED_HEIGHT", "1") not in ("0", "false", "False")
# Place the apple at the RSI frame's reference position instead of always at frame 0.
# Needed to seed the carry / put-back half of the clip, which RSI otherwise never reaches.
# Only meaningful together with TABLE_FIXED_HEIGHT, and only useful if the retargeted hand
# pose actually grips the apple -- otherwise it just drops.
RSI_SCENE_FOLLOWS_ROBOT = os.environ.get(
  "APPLE_RSI_SCENE_FOLLOWS_ROBOT", "0"
) not in ("0", "false", "False")
APPLE_MASS = 0.3
# Path to the object's collision mesh, as recorded by the retarget pipeline. The simulation has
# always used a sphere and nothing read this; it is exposed so the object can optionally be its real
# shape. Empty string when the reference does not carry one.
USE_OBJECT_MESH = rank_pick("APPLE_OBJECT_MESH", "").strip().lower() in (
  "1", "true", "yes", "on"
)


def _object_stl_path(pkl: "Path | None" = None) -> str:
  """Absolute path to the object mesh for the active reference, or "" when meshes are disabled.

  Raises if meshes ARE enabled but nothing can be found, rather than quietly reverting to the sphere.
  """
  if not USE_OBJECT_MESH:
    return ""
  pkl = pkl if pkl is not None else _APPLE_PKL
  try:
    with pkl.open("rb") as f:
      obj = pickle.load(f).get("object", {})
  except Exception as exc:
    raise RuntimeError(f"cannot read {pkl} to locate the object mesh: {exc}") from exc

  recorded = str(obj.get("stl_path", ""))
  name = Path(recorded).name if recorded else ""
  if not name:
    raise RuntimeError(f"{pkl} has no object stl_path; cannot use a mesh object")

  # Search dataset-relative first: the recorded path is absolute on whichever machine retargeted the
  # clip, so it is not portable. <pkl>/../meshes is where process_dataset.py puts them.
  candidates = [
    pkl.parent.parent / "meshes" / name,
    pkl.parent / "meshes" / name,
    Path(recorded),
  ]
  base = Path(name).stem
  budget = os.environ.get("APPLE_OBJECT_HULL", "").strip()
  for cand in candidates:
    if not cand.exists():
      continue
    if budget == "raw":
      return str(cand)
    if budget:
      hull = cand.with_name(f"{base}_col{budget}.stl")
      if not hull.exists():
        raise FileNotFoundError(
          f"APPLE_OBJECT_HULL={budget} requested but {hull} is missing"
        )
      return str(hull)
    reduced = cand.with_name(f"{base}_collision.stl")
    return str(reduced if reduced.exists() else cand)

  raise FileNotFoundError(
    "APPLE_OBJECT_MESH is set but the object mesh could not be found. Looked for "
    + ", ".join(str(c) for c in candidates)
    + ". Copy the dataset's meshes/ directory to this machine, or unset APPLE_OBJECT_MESH."
  )


OBJECT_STL = _object_stl_path()


def _object_bottom_offset() -> float:
  """Distance from the object's frame origin down to its lowest point at reference frame 0.

  For the sphere this is exactly the radius, so returning APPLE_RADIUS when meshes are off keeps the
  existing behaviour bit-identical.
  """
  if not (USE_OBJECT_MESH and OBJECT_STL):
    return float(APPLE_RADIUS)
  import trimesh

  with _APPLE_PKL.open("rb") as f:
    obj = pickle.load(f).get("object", {})
  quat = np.asarray(obj.get("quat_wxyz_mj"), dtype=np.float64)[0]
  w, x, y, z = quat
  rot = np.array([
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ])
  verts = np.asarray(trimesh.load(OBJECT_STL, process=False).vertices, dtype=np.float64)
  low = float((verts @ rot.T)[:, 2].min())
  print(f"[apple_eat] object bottom offset {(-low) * 100:.3f} cm from "
        f"{Path(OBJECT_STL).name} (sphere assumption was {APPLE_RADIUS * 100:.3f} cm)")
  return float(-low)


OBJECT_BOTTOM_OFFSET = _object_bottom_offset()


def _object_bottom_offsets_per_clip() -> list[float]:
  """Bottom offset for each clip's own object. One entry unless a mix is active."""
  paths = _mix_paths()
  if len(paths) == 1:
    return [OBJECT_BOTTOM_OFFSET]
  import trimesh

  # A clip whose collider is forced back to the analytic sphere must keep the sphere's bottom offset
  # too. Reading it off the mesh instead sits the apple 0.906 cm from the height it was validated at
  # -- the same table-height shift that silently broke a run described as a byte-for-byte
  # replication. APPLE_OBJECT_PERCLIP is read here rather than imported from residual_interact to
  # avoid an import cycle; it is an environment variable, so both modules see the same value.
  kinds = [
    x.strip().lower()
    for x in rank_pick("APPLE_OBJECT_PERCLIP", "").split(",")
    if x.strip()
  ]
  if kinds and len(kinds) != len(paths):
    raise ValueError(
      f"APPLE_OBJECT_PERCLIP has {len(kinds)} entries for {len(paths)} clips in APPLE_EAT_PKL_MIX"
    )

  out = []
  for i, p in enumerate(paths):
    stl = _object_stl_path(p)
    if not stl or (kinds and kinds[i] == "sphere"):
      print(
        f"[apple_eat] MIX bottom offset {APPLE_RADIUS * 100:.3f} cm (analytic sphere) "
        f"for {p.stem}",
        flush=True,
      )
      out.append(float(APPLE_RADIUS))
      continue
    with p.open("rb") as f:
      obj = pickle.load(f).get("object", {})
    w, x, y, z = np.asarray(obj.get("quat_wxyz_mj"), dtype=np.float64)[0]
    rot = np.array([
      [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    verts = np.asarray(trimesh.load(stl, process=False).vertices, dtype=np.float64)
    off = float(-(verts @ rot.T)[:, 2].min())
    print(f"[apple_eat] MIX bottom offset {off * 100:.3f} cm from {Path(stl).name}", flush=True)
    out.append(off)
  return out


_OBJECT_BOTTOM_OFFSETS_CACHE: list[float] | None = None


def object_bottom_offsets() -> list[float]:
  """Per-clip bottom offsets, resolved on first use (``_mix_paths`` is defined further down)."""
  global _OBJECT_BOTTOM_OFFSETS_CACHE
  if _OBJECT_BOTTOM_OFFSETS_CACHE is None:
    _OBJECT_BOTTOM_OFFSETS_CACHE = _object_bottom_offsets_per_clip()
  return _OBJECT_BOTTOM_OFFSETS_CACHE
APPLE_LINEAR_DAMPING = 0.05
APPLE_ANGULAR_DAMPING = 0.05
TABLE_XY_SIZE = 0.30 * 0.7
TABLE_OBJECT_GAP = 0.0
OBJECT_SPAWN_CLEARANCE = 0.0
OBJECT_INIT_LIFT = 0.0


def _estimate_cuboid_table_fill_thickness(
  min_thickness: float = 0.04,
  floor_margin: float = 0.08,
) -> float:
  with _APPLE_PKL.open("rb") as f:
    data = pickle.load(f)
  obj = data.get("object", {})
  pos = np.asarray(obj.get("pos_mj", obj.get("pos", [])), dtype=np.float32)
  if pos.size == 0:
    return float(min_thickness)
  max_obj_z = float(np.nanmax(pos[:, 2]))
  if not np.isfinite(max_obj_z):
    return float(min_thickness)
  return float(max(max_obj_z + floor_margin, min_thickness))


TABLE_THICKNESS = 0.04

from mjlab.tasks.apple_eat import object_pool  # noqa: E402


def _clip_id(env) -> torch.Tensor:
  """Which clip each environment trains on. Round-robin, so the split is exact and reproducible."""
  value = getattr(env, "_reference_clip_id", None)
  if value is None or value.shape[0] != env.num_envs:
    ref = _ref(str(env.episode_length_buf.device))
    n_clips = int(ref.get("n_clips", 1))
    value = (
      torch.arange(env.num_envs, device=env.episode_length_buf.device, dtype=torch.long) % n_clips
    )
    env._reference_clip_id = value
    if n_clips > 1:
      counts = [int((value == c).sum()) for c in range(n_clips)]
      print(f"[apple_eat] MIX: env split across clips {ref['clip_names']} -> {counts}", flush=True)
  return value


def mix_object_stls() -> list[str]:
  """Each clip's own object mesh, in clip order. Empty string means "use the analytic sphere"."""
  return [_object_stl_path(p) for p in _mix_paths()]


def mix_clip_count() -> int:
  return len(_mix_paths())


def object_entity_count() -> int:
  '''How many object entities the scene config should author.

  One per clip for a host that clones a single world template and parks the unused objects; ONE
  when the host authors a separate world per object and hands each environment only its own
  (`APPLE_OBJECT_PER_WORLD=1`). Do not use this for anything but scene authoring -- the number of
  CLIPS is still mix_clip_count(), and the reference, the gates and the terminations all key off
  that.
  '''
  import os as _os

  if _os.environ.get("APPLE_OBJECT_PER_WORLD", "").strip() in ("1", "true", "yes", "on"):
    return 1
  return mix_clip_count()


def clip_spans(ref) -> list[tuple[int, int]]:
  """(first_row, length) for each clip in the concatenated reference."""
  n = int(ref["n_frames"])
  lengths = ref.get("clip_lengths")
  if lengths is None:
    return [(0, n)]
  return [(c * n, int(lengths[c])) for c in range(int(ref["n_clips"]))]


def per_clip_rows(env, per_clip_local) -> torch.Tensor:
  """Per-clip LOCAL frame indices -> per-env GLOBAL rows into the concatenated reference."""
  ref = _ref(str(env.episode_length_buf.device))
  lo, _ = _clip_bounds(env, int(ref["n_frames"]))
  t = per_clip_local
  if not torch.is_tensor(t):
    t = torch.tensor([int(t)], device=lo.device, dtype=torch.long)
  t = t.to(device=lo.device, dtype=torch.long).reshape(-1)
  if t.numel() == 1:
    return lo + t[0]
  return lo + t[_clip_id(env)]


def per_clip_gather(env, per_clip: "torch.Tensor") -> "torch.Tensor":
  """Select each environment's row from a (n_clips, ...) per-clip tensor."""
  if per_clip.shape[0] == 1:
    return per_clip[0]
  return per_clip.to(env.episode_length_buf.device)[_clip_id(env)]


def _clip_bounds(env, n_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
  """First and last valid GLOBAL row for each environment's clip."""
  ref = _ref(str(env.episode_length_buf.device))
  clip = _clip_id(env)
  lo = clip * int(n_frames)
  lengths = ref.get("clip_lengths")
  if lengths is None:
    hi = lo + int(n_frames) - 1
  else:
    hi = lo + lengths.to(lo.device)[clip] - 1
  return lo, hi


def _reference_start_frame(env, n_frames: int) -> torch.Tensor:
  value = getattr(env, "_reference_start_frame", None)
  if (
    value is None
    or value.shape[0] != env.num_envs
    or value.device != env.episode_length_buf.device
  ):
    # frame 0 of this env's own clip, in global rows
    value = _clip_id(env) * int(n_frames)
    env._reference_start_frame = value
  lo, hi = _clip_bounds(env, n_frames)
  torch.clamp_(value, min=0)
  torch.minimum(value, hi, out=value)
  torch.maximum(value, lo, out=value)
  return value


def _set_reference_start_frame(
  env, env_ids: torch.Tensor, frame: int | torch.Tensor, n_frames: int
) -> torch.Tensor:
  start_frame = _reference_start_frame(env, n_frames)
  if isinstance(frame, torch.Tensor):
    frame_t = frame.to(device=env.episode_length_buf.device, dtype=torch.long)
    if frame_t.ndim == 0:
      frame_t = frame_t.expand(env_ids.numel())
    elif frame_t.shape[0] == env.num_envs:
      frame_t = frame_t[env_ids]
  else:
    frame_t = torch.full(
      (env_ids.numel(),),
      int(frame),
      dtype=torch.long,
      device=env.episode_length_buf.device,
    )
  # callers pass a frame within the clip; the buffer holds global rows
  lo, hi = _clip_bounds(env, n_frames)
  frame_t = frame_t.clamp(0, max(int(n_frames) - 1, 0)) + lo[env_ids]
  frame_t = torch.minimum(frame_t, hi[env_ids])
  start_frame[env_ids] = frame_t
  return frame_t


def _cuboid_scene_poses_at_frame(
  ref: dict,
  env_origins: torch.Tensor,
  frame: int | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  if isinstance(frame, torch.Tensor):
    frame_t = frame.to(device=env_origins.device, dtype=torch.long)
    if frame_t.ndim == 0:
      frame_t = frame_t.expand(env_origins.shape[0])
  else:
    frame_t = torch.full(
      (env_origins.shape[0],), int(frame), dtype=torch.long, device=env_origins.device
    )
  # frame_t holds GLOBAL rows into the concatenated reference; clamping it to one clip's length
  # would send every clip-1 environment to clip 0's poses
  n_per_clip = max(int(ref["n_frames"]), 1)
  clip_t = torch.div(frame_t, n_per_clip, rounding_mode="floor")
  lengths = ref.get("clip_lengths")
  band_lo = clip_t * n_per_clip
  band_hi = band_lo + (lengths.to(frame_t.device)[clip_t] - 1 if lengths is not None
                       else n_per_clip - 1)
  frame_t = torch.maximum(torch.minimum(frame_t, band_hi), band_lo)

  obj_pos_w = ref["obj_pos"][frame_t] + env_origins
  obj_quat_w = ref["obj_quat"][frame_t]

  _offsets = object_bottom_offsets()
  bottom = (
    torch.tensor(_offsets, dtype=obj_pos_w.dtype, device=obj_pos_w.device)[clip_t]
    if len(_offsets) > 1
    else torch.full_like(obj_pos_w[:, 2], float(OBJECT_BOTTOM_OFFSET))
  )

  obj_bottom_nominal = obj_pos_w[:, 2] - bottom
  if TABLE_FIXED_HEIGHT:
    # Rest height of each environment's OWN clip (its frame 0), so the table is identical at every
    # RSI frame and correct for the object that world actually carries.
    rest_z = ref["obj_pos"][band_lo, 2].to(obj_pos_w.device)
    table_top = (rest_z - bottom - TABLE_OBJECT_GAP).clone()
  else:
    table_top = obj_bottom_nominal - TABLE_OBJECT_GAP

  table_pos_w = obj_pos_w.clone()
  table_pos_w[:, 2] = table_top - 0.5 * TABLE_THICKNESS
  table_quat_w = torch.zeros(env_origins.shape[0], 4, device=env_origins.device)
  table_quat_w[:, 0] = 1.0

  obj_pos_w = obj_pos_w.clone()
  spawn_bottom_gap = TABLE_OBJECT_GAP + OBJECT_SPAWN_CLEARANCE
  if TABLE_FIXED_HEIGHT:
    # Keep the apple where the reference has it; only stop it sinking into the table.
    # Applying the re-seat here would drag a carried apple back down to rest height.
    min_z = table_top + spawn_bottom_gap + bottom
    obj_pos_w[:, 2] = torch.maximum(obj_pos_w[:, 2], min_z)
  else:
    obj_pos_w[:, 2] += (table_top + spawn_bottom_gap) - obj_bottom_nominal
  obj_pos_w[:, 2] += OBJECT_INIT_LIFT

  obj_pose = torch.cat([obj_pos_w, obj_quat_w], dim=-1)
  table_pose = torch.cat([table_pos_w, table_quat_w], dim=-1)
  return obj_pose, table_pose


def _table_reference_frame(env, default_frame: int | torch.Tensor, n_frames: int):
  forced = getattr(env, "_force_table_reference_frame", None)
  if forced is None:
    return default_frame
  if isinstance(forced, torch.Tensor):
    forced_t = forced.to(device=env.episode_length_buf.device, dtype=torch.long)
    if forced_t.ndim == 0:
      return forced_t.clamp(0, max(int(n_frames) - 1, 0))
    return forced_t.clamp(0, max(int(n_frames) - 1, 0))
  return max(0, min(int(forced), int(n_frames) - 1))


def clip_frame0_rows(ref: dict, num_envs: int, device, env=None) -> torch.Tensor:
  """Global row of frame 0 of each environment's own clip; zeros without a mix.

  The clip assignment MUST come from _clip_id, not be recomputed here. This used to hardcode
  `arange % n_clips` on the assumption that _clip_id always returns that. A host that injects its
  own `env._reference_clip_id` -- the Newton port groups environments by object so identical
  worlds can be replicated in one block -- breaks the assumption silently: the reference, the
  object reset and the terminations all follow the injected map while the every-step table write
  followed the round-robin one, so 7 of every 8 environments got another clip's table height and
  their object fell through empty space. Pass `env` wherever one is available.
  """
  n_clips = int(ref.get("n_clips", 1))
  if n_clips <= 1:
    return torch.zeros(num_envs, device=device, dtype=torch.long)
  if env is not None:
    cid = _clip_id(env)
    if cid.shape[0] != num_envs:
      raise RuntimeError(
        f"_clip_id gave {cid.shape[0]} entries for {num_envs} environments; the scene anchor and "
        "the reference would disagree")
    return cid.to(device=device, dtype=torch.long) * int(ref["n_frames"])
  idx = torch.arange(num_envs, device=device, dtype=torch.long)
  return (idx % n_clips) * int(ref["n_frames"])


def _initial_cuboid_scene_poses(
  ref: dict,
  env_origins: torch.Tensor,
  env=None,
) -> tuple[torch.Tensor, torch.Tensor]:
  return _cuboid_scene_poses_at_frame(
    ref, env_origins,
    clip_frame0_rows(ref, env_origins.shape[0], env_origins.device, env=env),
  )


# Omnigrasp removes the table at the reference contact time -- `humanoid_omnigrasp.py:762`,
# `self._table_states[env_ids, 2] = 100`, with the removal frame set per environment from that
# clip's own contact time. Off unless TABLE_REMOVE_AFTER_CF is set, so nothing changes for a run
# that does not ask for it.
_TABLE_REMOVE_AFTER_CF = os.environ.get("TABLE_REMOVE_AFTER_CF", "").strip()
_TABLE_REMOVE_DROP = float(os.environ.get("TABLE_REMOVE_DROP", "0.30") or 0.30)


def _table_removed_mask(env) -> torch.Tensor | None:
  """Which environments have passed their own contact frame by TABLE_REMOVE_AFTER_CF frames."""
  if not _TABLE_REMOVE_AFTER_CF:
    return None
  # Imported here, not at module scope: staged_mdp imports residual_interact.mdp, which imports
  # this module.
  from mjlab.tasks.residual_interact import staged_mdp as _smdp

  return _smdp.after_cf(env, int(_TABLE_REMOVE_AFTER_CF))


def _drop_table_after_cf(env, table_pose: torch.Tensor) -> torch.Tensor:
  """Sink the table once the grasp should exist, so an unheld object falls instead of resting.

  This does NOT hand the policy a lift. The object is a full rigid body under gravity either way;
  what goes away is the support a not-quite-grasp could lean on, which is also what turns "failed
  to grasp" from an unlabelled non-event into an immediate `og_object_far` termination.

  Down by a fixed amount rather than Omnigrasp's +100 m: `table_top` feeds an observation channel
  (`_object_table_obs`), and a 100 m reading is a number no normaliser has ever seen.
  """
  gone = _table_removed_mask(env)
  if gone is None:
    return table_pose
  # Visible in the training row: a removal that never fires and one that fires at reset look the
  # same in every other metric.
  from mjlab.tasks.residual_interact import mdp as _rmdp

  _rmdp._safe_log(env, "Metric/table_removed", gone.float())
  table_pose = table_pose.clone()
  table_pose[:, 2] = table_pose[:, 2] - gone.to(table_pose.dtype) * abs(_TABLE_REMOVE_DROP)
  return table_pose


def _write_table_pose(table: Entity, table_pose: torch.Tensor, env_ids=None) -> None:
  if table.is_mocap:
    table.write_mocap_pose_to_sim(table_pose, env_ids=env_ids)
    return
  table_state = torch.zeros(table_pose.shape[0], 13, device=table_pose.device)
  table_state[:, :7] = table_pose
  table.write_root_state_to_sim(table_state, env_ids=env_ids)


def _resample_arrays(
  arrays: list[np.ndarray],
  src_fps: float,
  target_fps: float = _TARGET_FPS,
) -> list[np.ndarray]:
  if abs(float(src_fps) - float(target_fps)) <= 0.5:
    return [np.asarray(arr, dtype=np.float32) for arr in arrays]
  src_n = len(arrays[0])
  duration = (src_n - 1) / max(float(src_fps), 1e-6)
  target_n = max(int(duration * float(target_fps)) + 1, 2)
  src_t = np.arange(src_n, dtype=np.float32) / float(src_fps)
  target_t = np.arange(target_n, dtype=np.float32) / float(target_fps)
  out_arrays: list[np.ndarray] = []
  for arr in arrays:
    arr = np.asarray(arr, dtype=np.float32)
    out = np.zeros((target_n, arr.shape[1]), dtype=np.float32)
    for i in range(arr.shape[1]):
      out[:, i] = np.interp(target_t, src_t, arr[:, i])
    if arr.shape[1] == 4:
      out /= np.linalg.norm(out, axis=1, keepdims=True).clip(min=1e-9)
    out_arrays.append(out)
  return out_arrays


def _env_flag(name: str, default: bool = False) -> bool:
  value = os.environ.get(name)
  if value is None:
    return default
  return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_quat_np(q: np.ndarray) -> np.ndarray:
  return q / np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1.0e-9)


def _slerp_wxyz_np(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
  q0 = _normalize_quat_np(np.asarray(q0, dtype=np.float32))
  q1 = _normalize_quat_np(np.asarray(q1, dtype=np.float32))
  dot = np.sum(q0 * q1, axis=-1, keepdims=True)
  q1 = np.where(dot < 0.0, -q1, q1)
  dot = np.abs(dot).clip(0.0, 1.0)
  alpha = np.asarray(alpha, dtype=np.float32).reshape(-1, 1)
  linear = dot > 0.9995
  theta_0 = np.arccos(dot)
  sin_theta_0 = np.sin(theta_0).clip(min=1.0e-9)
  theta = theta_0 * alpha
  s0 = np.sin(theta_0 - theta) / sin_theta_0
  s1 = np.sin(theta) / sin_theta_0
  out = s0 * q0 + s1 * q1
  out = np.where(linear, (1.0 - alpha) * q0 + alpha * q1, out)
  return _normalize_quat_np(out.astype(np.float32))


def _resample_qpos_wxyz_np(
  qpos: np.ndarray,
  src_fps: float,
  target_fps: float = _TARGET_FPS,
) -> np.ndarray:
  qpos = np.asarray(qpos, dtype=np.float32)
  if abs(float(src_fps) - float(target_fps)) <= 0.5:
    return qpos.copy()
  n_src = int(qpos.shape[0])
  duration = (n_src - 1) / max(float(src_fps), 1.0e-6)
  n_tgt = int(np.floor(duration * float(target_fps))) + 1
  src_t = np.linspace(0.0, duration, n_src, endpoint=True, dtype=np.float32)
  target_t = np.linspace(0.0, duration, n_tgt, endpoint=True, dtype=np.float32)
  out = np.zeros((n_tgt, qpos.shape[1]), dtype=np.float32)
  for i in range(3):
    out[:, i] = np.interp(target_t, src_t, qpos[:, i])
  for i in range(7, qpos.shape[1]):
    out[:, i] = np.interp(target_t, src_t, qpos[:, i])
  right = np.searchsorted(src_t, target_t, side="right").clip(1, n_src - 1)
  left = right - 1
  span = (src_t[right] - src_t[left]).clip(min=1.0e-9)
  alpha = (target_t - src_t[left]) / span
  out[:, 3:7] = _slerp_wxyz_np(qpos[left, 3:7], qpos[right, 3:7], alpha)
  out[0, 3:7] = qpos[0, 3:7]
  out[-1, 3:7] = qpos[-1, 3:7]
  return out


def _apply_ema_qpos_np(qpos: np.ndarray, alpha: float = 0.8) -> np.ndarray:
  qpos = np.asarray(qpos, dtype=np.float32)
  if qpos.shape[0] <= 1 or alpha <= 0.0:
    return qpos.copy()
  smoothed = np.empty_like(qpos)
  smoothed[0] = qpos[0]
  for t in range(1, qpos.shape[0]):
    q_prev = smoothed[t - 1, 3:7]
    q_curr = qpos[t, 3:7]
    if float(np.dot(q_prev, q_curr)) < 0.0:
      q_curr = -q_curr
    smoothed[t, :3] = smoothed[t - 1, :3] * alpha + qpos[t, :3] * (1.0 - alpha)
    smoothed[t, 7:] = smoothed[t - 1, 7:] * alpha + qpos[t, 7:] * (1.0 - alpha)
    q_blend = q_prev * alpha + q_curr * (1.0 - alpha)
    smoothed[t, 3:7] = q_blend / max(float(np.linalg.norm(q_blend)), 1.0e-8)
  return smoothed


def _mix_paths() -> list[Path]:
  """The clips this batch trains on. One entry unless APPLE_EAT_PKL_MIX is set."""
  mix = os.environ.get("APPLE_EAT_PKL_MIX", "").strip()
  if not mix:
    return [_APPLE_PKL]
  paths = [Path(x.strip()) for x in mix.split(",") if x.strip()]
  if len(paths) < 2:
    raise ValueError("APPLE_EAT_PKL_MIX needs at least two clips; use APPLE_EAT_PKL for one")
  missing = [str(p) for p in paths if not p.exists()]
  if missing:
    raise FileNotFoundError(f"APPLE_EAT_PKL_MIX clips not found: {missing}")
  return paths


def _load_ref(device: str):
  paths = _mix_paths()
  if len(paths) == 1:
    return _load_ref_single(device, paths[0])
  return _load_ref_mix(device, paths)


def _gt_contact_frame(data: dict, n_resampled: int) -> int:
  """First frame with object contact, from GRAB's labels, in RESAMPLED frames. -1 if absent.

  GRAB ships per-vertex contact at threshold 2e-05 and process_dataset.py carries it into the pkl
  already downsampled to the pkl's own frame count, so the only conversion needed is pkl -> resampled.

  This is ground truth and replaces a distance heuristic that is object-shape dependent: thresholding
  fingertip-to-object-CENTRE distance works on a 4 cm sphere and is 3.2 s late on a 12.3 cm stapler.
  """
  contact = data.get("contact")
  if not isinstance(contact, dict) or "object" not in contact:
    return -1
  labels = np.asarray(contact["object"])
  if labels.ndim != 2 or labels.shape[0] < 2:
    return -1
  touched = (labels > 0).any(axis=1)
  if not touched.any():
    return -1
  f_pkl = int(np.argmax(touched))
  n_pkl = int(labels.shape[0])
  return int(round(f_pkl * (n_resampled - 1) / max(n_pkl - 1, 1)))


def _load_ref_mix(device: str, paths: list[Path]):
  """Pad every clip to the longest and concatenate along time.

  Clip c frame f becomes global row c * max_frames + f, so the per-env frame index carries the clip
  identity and every downstream `ref[key][frame]` keeps working unchanged.
  """
  refs = [_load_ref_single(device, p) for p in paths]
  lengths = [int(r["n_frames"]) for r in refs]
  max_frames = max(lengths)
  out: dict = {}
  for key, value in refs[0].items():
    if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] in lengths:
      padded = []
      for r, n in zip(refs, lengths):
        v = r[key]
        if v.shape[0] != n:
          raise ValueError(f"clip arrays disagree on length for {key!r}: {v.shape[0]} vs {n}")
        if n < max_frames:
          # repeat the last frame; the per-clip bound stops the policy before it is ever read
          v = torch.cat([v, v[-1:].expand(max_frames - n, *v.shape[1:])], dim=0)
        padded.append(v)
      out[key] = torch.cat(padded, dim=0).contiguous()
    else:
      out[key] = value
  out["n_frames"] = max_frames
  out["n_clips"] = len(paths)
  out["clip_lengths"] = torch.tensor(lengths, dtype=torch.long, device=device)
  out["clip_names"] = [p.name for p in paths]
  # Non-frame-shaped values above were copied from refs[0]; this one is per clip and must not be.
  out["gt_contact_frames"] = [int(r["gt_contact_frames"][0]) for r in refs]
  print(f"[apple_eat] MIX: {len(paths)} clips padded to {max_frames} frames each -> "
        f"{max_frames * len(paths)} rows", flush=True)
  for p, n in zip(paths, lengths):
    print(f"[apple_eat]   {p.name}: {n} frames", flush=True)
  return out


def _load_ref_single(device: str, pkl: Path | None = None):
  pkl = pkl if pkl is not None else _APPLE_PKL
  with pkl.open("rb") as f:
    data = pickle.load(f)
  r53 = data["robot_53dof"]
  obj = data["object"]
  table = data["table"]
  fps = float(data.get("fps", _TARGET_FPS))
  use_astra_official_ref = _env_flag("APPLE_EAT_ASTRA_OFFICIAL_REF")
  if use_astra_official_ref and "robot_29dof" in data:
    r29 = data["robot_29dof"]
    root_rot_wxyz = np.asarray(r29["root_rot"], dtype=np.float32)[:, [3, 0, 1, 2]]
    qpos29 = np.concatenate(
      [
        np.asarray(r29["root_pos"], dtype=np.float32),
        root_rot_wxyz,
        np.asarray(r29["dof_pos"], dtype=np.float32),
      ],
      axis=-1,
    )
    ema_alpha = float(os.environ.get("APPLE_EAT_ASTRA_EMA_ALPHA", "0.8"))
    qpos29 = _apply_ema_qpos_np(qpos29, alpha=ema_alpha)
    qpos29 = _resample_qpos_wxyz_np(qpos29, fps, _TARGET_FPS)
    root_pos_np = qpos29[:, :3]
    root_rot_xyzw_np = qpos29[:, [4, 5, 6, 3]]
    q_body_np = qpos29[:, 7:]
    (
      q53_np,
      obj_pos_np,
      obj_quat_np,
      table_pos_np,
      table_quat_np,
    ) = _resample_arrays(
      [
        np.asarray(r53["dof_pos"], dtype=np.float32),
        np.asarray(obj["pos_mj"], dtype=np.float32),
        np.asarray(obj["quat_wxyz_mj"], dtype=np.float32),
        np.asarray(table["pos_mj"], dtype=np.float32),
        np.asarray(table["quat_wxyz_mj"], dtype=np.float32),
      ],
      fps,
      _TARGET_FPS,
    )
  else:
    (
      root_pos_np,
      root_rot_xyzw_np,
      q53_np,
      obj_pos_np,
      obj_quat_np,
      table_pos_np,
      table_quat_np,
    ) = _resample_arrays(
      [
        np.asarray(r53["root_pos"], dtype=np.float32),
        np.asarray(r53["root_rot"], dtype=np.float32),
        np.asarray(r53["dof_pos"], dtype=np.float32),
        np.asarray(obj["pos_mj"], dtype=np.float32),
        np.asarray(obj["quat_wxyz_mj"], dtype=np.float32),
        np.asarray(table["pos_mj"], dtype=np.float32),
        np.asarray(table["quat_wxyz_mj"], dtype=np.float32),
      ],
      fps,
      _TARGET_FPS,
    )
    q_body_np = None
  fps = _TARGET_FPS
  root_pos = torch.tensor(root_pos_np, dtype=torch.float32, device=device)
  root_rot_xyzw = torch.tensor(root_rot_xyzw_np, dtype=torch.float32, device=device)
  root_rot = root_rot_xyzw[:, [3, 0, 1, 2]]
  q53 = torch.tensor(q53_np, dtype=torch.float32, device=device)
  body_idx = torch.tensor(BODY_IDX_IN_53, dtype=torch.long, device=device)
  hand_idx = torch.tensor(HAND_IDX_IN_53, dtype=torch.long, device=device)
  q_body = (
    torch.tensor(q_body_np, dtype=torch.float32, device=device)
    if q_body_np is not None
    else q53[:, body_idx]
  )
  if q53.shape[1] != REF_DOF_DIM:
    raise ValueError(
      f"reference {pkl} has {q53.shape[1]} dof columns but hand kind "
      f"{HAND_KIND!r} expects {REF_DOF_DIM}. Retarget this clip for the selected hand "
      f"(interaction/scale_retarget/process_dataset.py --gmr-robot) instead of reusing "
      f"another hand's reference."
    )
  q_hand = q53[:, hand_idx]
  q = torch.cat([q_body, q_hand], dim=-1)
  obj_pos = torch.tensor(obj_pos_np, dtype=torch.float32, device=device)
  obj_quat = torch.tensor(obj_quat_np, dtype=torch.float32, device=device)
  table_pos = torch.tensor(table_pos_np, dtype=torch.float32, device=device)
  if SCENE_Z_OFFSET != 0.0:
    obj_pos[:, 2] += SCENE_Z_OFFSET
    table_pos[:, 2] += SCENE_Z_OFFSET
    print(f"[apple_eat] SCENE_Z_OFFSET={SCENE_Z_OFFSET:+.4f} m applied to apple+table; "
          f"apple rest height now {float(obj_pos[0, 2]):.4f} m")
  table_quat = torch.tensor(table_quat_np, dtype=torch.float32, device=device)
  qvel = torch.zeros_like(q)
  if q.shape[0] > 1:
    qvel[1:] = (q[1:] - q[:-1]) * fps
    qvel[0] = qvel[1]
  obj_vel = torch.zeros_like(obj_pos)
  if obj_pos.shape[0] > 1:
    obj_vel[1:] = (obj_pos[1:] - obj_pos[:-1]) * fps
    obj_vel[0] = obj_vel[1]
  root0 = root_pos[0].clone()
  heading0 = _calc_heading_quat(root_rot[0:1])[0]
  inv_heading0 = quat_conjugate(heading0.unsqueeze(0))[0]
  q_rep = inv_heading0.unsqueeze(0).expand(root_pos.shape[0], -1)

  def canonicalize_pos(pos: torch.Tensor) -> torch.Tensor:
    raw = pos.clone()
    rel = raw.clone()
    rel[:, :2] -= root0[:2]
    rel[:, :3] = quat_apply(q_rep, rel[:, :3])
    rel[:, 2] = raw[:, 2]
    return rel

  def canonicalize_quat(quat: torch.Tensor) -> torch.Tensor:
    out = quat_mul(q_rep, quat)
    return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-9)

  if not _env_flag("APPLE_EAT_SKIP_REF_CANONICALIZE"):
    root_pos = canonicalize_pos(root_pos)
    root_rot = canonicalize_quat(root_rot)
    obj_pos = canonicalize_pos(obj_pos)
    obj_quat = canonicalize_quat(obj_quat)
    table_pos = canonicalize_pos(table_pos)
    table_quat = canonicalize_quat(table_quat)
    obj_vel = quat_apply(q_rep, obj_vel)
  root_lin_vel = torch.zeros_like(root_pos)
  root_ang_vel = torch.zeros_like(root_pos)
  if root_pos.shape[0] > 1:
    root_lin_vel[1:] = (root_pos[1:] - root_pos[:-1]) * fps
    root_lin_vel[0] = root_lin_vel[1]
    root_delta = quat_mul(root_rot[1:], quat_conjugate(root_rot[:-1]))
    root_ang_vel[1:] = _quat_to_rotvec(root_delta) * fps
    root_ang_vel[0] = root_ang_vel[1]
  return {
    "root_pos": root_pos,
    "root_rot": root_rot,
    "root_lin_vel": root_lin_vel,
    "root_ang_vel": root_ang_vel,
    "dof_pos": q,
    "dof_vel": qvel,
    "obj_pos": obj_pos,
    "obj_quat": obj_quat,
    "obj_vel": obj_vel,
    "table_pos": table_pos,
    "table_quat": table_quat,
    "default_body": q[:, :NUM_BODY].mean(dim=0),
    "default_hand": q[:, NUM_BODY : NUM_BODY + NUM_HAND].mean(dim=0),
    "n_frames": q.shape[0],
    # First frame the DATASET says the object is touched, in resampled frames. A list of one so the
    # mix loader can replace it with one entry per clip; see _load_ref_mix.
    "gt_contact_frames": [_gt_contact_frame(data, int(q.shape[0]))],
  }


_REF_CACHE: dict[str, dict] = {}




def _ref(device: str):
  if device not in _REF_CACHE:
    _REF_CACHE[device] = _load_ref(device)
  return _REF_CACHE[device]


def _calc_heading_quat(q: torch.Tensor) -> torch.Tensor:
  w, z = q[:, 0], q[:, 3]
  n_sq = w * w + z * z
  n = n_sq.clamp(min=1e-12).sqrt()
  out = torch.zeros_like(q)
  out[:, 0] = 1.0
  valid = n_sq > 1e-12
  out[valid, 0] = w[valid] / n[valid]
  out[valid, 3] = z[valid] / n[valid]
  return out


def _quat_to_rot6d(q: torch.Tensor) -> torch.Tensor:
  q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)
  w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
  r00 = 1.0 - 2.0 * (y * y + z * z)
  r01 = 2.0 * (x * y - z * w)
  r10 = 2.0 * (x * y + z * w)
  r11 = 1.0 - 2.0 * (x * x + z * z)
  r20 = 2.0 * (x * z - y * w)
  r21 = 2.0 * (y * z + x * w)
  return torch.stack([r00, r01, r10, r11, r20, r21], dim=-1)


def _quat_to_rotvec(q: torch.Tensor) -> torch.Tensor:
  q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)
  xyz = q[:, 1:4]
  sin_half = xyz.norm(dim=-1, keepdim=True)
  half_angle = torch.atan2(sin_half, q[:, 0:1])
  return 2.0 * half_angle * xyz / sin_half.clamp_min(1e-9)


def _active_after_startup(env, startup_steps: int = 36) -> torch.Tensor:
  return (env.episode_length_buf >= int(startup_steps)).float()


def _active_for_bootstrap_loss(env, startup_steps: int = 36) -> torch.Tensor:
  return (env.episode_length_buf >= int(startup_steps)).float()


def _tracking_frame(env, n_frames: int, startup_steps: int = 36) -> torch.Tensor:
  frame = _reference_start_frame(env, n_frames) + (
    env.episode_length_buf - int(startup_steps)
  ).clamp(min=0)
  # the bound is this env's own clip end, not a global scalar: clips differ in length and each one
  # occupies its own band of rows
  _, hi = _clip_bounds(env, n_frames)
  return torch.minimum(frame, hi)


def local_tracking_frame(env, n_frames: int, startup_steps: int = 36) -> torch.Tensor:
  """Frame index inside this env's OWN clip, counted from 0.

  `_tracking_frame` returns a GLOBAL row into the concatenated reference, which is what
  `ref[key][frame]` needs. That row is meaningless as a progress number: under MIX the
  clips are laid end to end, so clip 1 starts at row n_frames and a mean over clips
  mixes two different origins. Use this for metrics, logging, and any comparison
  against a scalar frame number.
  """
  lo, _ = _clip_bounds(env, n_frames)
  return _tracking_frame(env, n_frames, startup_steps=startup_steps) - lo


def effective_sonic_action(env) -> torch.Tensor:
  try:
    term = env.action_manager.get_term("sonic_action")
    return term.raw_action.clone()
  except Exception:
    return env.action_manager.action.clone()


@dataclass(kw_only=True)
class Sonic53ActionCfg(ActionTermCfg):
  """SONIC action adapter: 29 body DOF plus the mounted hand.

  53 wide with the 24-DOF xhand, 69 with the 40-DOF Wuji gen-1 hand. The base tracker
  only ever emits the 29 body actions; the hand comes from the residual
  (base_hand_mode=zero), so widening the hand does not touch the frozen tracker.
  """

  body_action_scale: float = 1.0
  hand_action_scale: float = 1.0
  body_action_clip: float = 5.0
  hand_action_clip: float = 5.0
  tracking_start_assist_gain: float = 1.5
  tracking_start_assist_steps: int = 120
  tracking_start_assist_hold_xy: bool = False
  tracking_start_assist_follow_ref_xy: bool = False
  table_reference_frame: int | None = None
  reference_object_tracking: bool = False
  reference_object_tracking_clearance: float = 0.0
  reference_object_tracking_start_frame: int = 0
  reference_object_tracking_end_frame: int = -1

  def build(self, env):
    return Sonic53Action(self, env)


class Sonic53Action(ActionTerm):
  cfg: Sonic53ActionCfg

  def __init__(self, cfg: Sonic53ActionCfg, env):
    super().__init__(cfg, env)
    self._raw_actions = torch.zeros(
      self.num_envs, NUM_BODY + NUM_HAND, device=self.device
    )
    self._target_joint_names = list(BODY_29_DOF_NAMES) + list(HAND_24_DOF_NAMES)
    target_ids, _ = self._entity.find_joints(
      tuple(self._target_joint_names), preserve_order=True
    )
    self._target_ids = torch.tensor(target_ids, device=self.device, dtype=torch.long)
    self._il_to_pkl = torch.tensor(IL_FOR_PKL, device=self.device, dtype=torch.long)
    self._body_default_pkl = torch.tensor(
      SONIC_DEFAULT_ANGLES_PKL, device=self.device, dtype=torch.float32
    )
    self._body_scale_pkl = torch.tensor(
      SONIC_ACTION_SCALE_PKL, device=self.device, dtype=torch.float32
    )
    self._kp = torch.tensor(KP_53, device=self.device, dtype=torch.float32)
    self._kd = torch.tensor(KD_53, device=self.device, dtype=torch.float32)
    self._effort_limit = torch.tensor(
      EFFORT_53, device=self.device, dtype=torch.float32
    )
    pelvis_ids, _ = self._entity.find_bodies(("pelvis",), preserve_order=True)
    self._pelvis_body_id = int(pelvis_ids[0]) if pelvis_ids else 0
    self._band_anchor = torch.zeros(self.num_envs, 3, device=self.device)
    self._band_anchor[:, :2] = self._env.scene.env_origins[:, :2]
    self._band_anchor[:, 2] = self._env.scene.env_origins[:, 2] + 0.80
    self._band_target_quat = torch.zeros(self.num_envs, 4, device=self.device)
    self._band_target_quat[:, 0] = 1.0
    if self.cfg.table_reference_frame is not None:
      self._env._force_table_reference_frame = int(self.cfg.table_reference_frame)

  @property
  def action_dim(self) -> int:
    # 29 body + hand DOF of whichever hand is mounted: 53 with xhand, 69 with the Wuji hand.
    # The class name keeps "53" only because other modules import it by name.
    return NUM_BODY + NUM_HAND

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    clipped = torch.nan_to_num(
      actions.to(self.device), nan=0.0, posinf=0.0, neginf=0.0
    ).clone()
    body_clip = float(self.cfg.body_action_clip)
    hand_clip = float(self.cfg.hand_action_clip)
    if body_clip > 0.0:
      clipped[:, :NUM_BODY] = clipped[:, :NUM_BODY].clamp(-body_clip, body_clip)
    if hand_clip > 0.0:
      clipped[:, NUM_BODY : NUM_BODY + NUM_HAND] = clipped[
        :, NUM_BODY : NUM_BODY + NUM_HAND
      ].clamp(-hand_clip, hand_clip)
    self._raw_actions[:] = clipped

  def apply_actions(self) -> None:
    ref = _ref(self.device)
    start_frame = _reference_start_frame(self._env, ref["n_frames"])
    body_action_pkl = self._raw_actions[:, :NUM_BODY][:, self._il_to_pkl]
    body_target = (
      self._body_default_pkl.unsqueeze(0)
      + body_action_pkl
      * self._body_scale_pkl.unsqueeze(0)
      * float(self.cfg.body_action_scale)
    ).clamp(-3.14, 3.14)
    hand_target = ref["default_hand"].unsqueeze(0) + self._raw_actions[
      :, NUM_BODY : NUM_BODY + NUM_HAND
    ] * float(self.cfg.hand_action_scale)
    target = torch.cat([body_target, hand_target], dim=-1)
    startup_mask = self._env.episode_length_buf <= 36
    hard_hold_mask = self._env.episode_length_buf <= 30
    if startup_mask.any():
      startup_env_ids = startup_mask.nonzero(as_tuple=False).squeeze(-1)
      target[startup_env_ids] = ref["dof_pos"][
        start_frame[startup_env_ids], : NUM_BODY + NUM_HAND
      ]
    if hard_hold_mask.any():
      root_state = torch.zeros(int(hard_hold_mask.sum().item()), 13, device=self.device)
      env_ids = hard_hold_mask.nonzero(as_tuple=False).squeeze(-1)
      hard_frame = start_frame[env_ids]
      root_state[:, :3] = (
        ref["root_pos"][hard_frame] + self._env.scene.env_origins[env_ids]
      )
      root_state[:, 3:7] = ref["root_rot"][hard_frame]
      self._entity.write_root_state_to_sim(root_state, env_ids=env_ids)
      q = ref["dof_pos"][hard_frame, : NUM_BODY + NUM_HAND]
      qd = torch.zeros_like(q)
      self._entity.write_joint_state_to_sim(
        q, qd, joint_ids=self._target_ids, env_ids=env_ids
      )
    anchor_update = self._env.episode_length_buf == 31
    if anchor_update.any():
      env_ids = anchor_update.nonzero(as_tuple=False).squeeze(-1)
      self._band_anchor[env_ids] = self._entity.data.body_link_pos_w[
        env_ids, self._pelvis_body_id
      ]
      self._band_target_quat[env_ids] = self._entity.data.body_link_quat_w[
        env_ids, self._pelvis_body_id
      ]
    try:
      table = self._env.scene["table"]
    except KeyError:
      table = None
    if table is not None:
      # frame 0 of each env's OWN clip: a scalar 0 is global row 0, which is clip 0's table, and
      # this write covers every environment
      table_frame = _table_reference_frame(
        self._env,
        clip_frame0_rows(ref, self._env.num_envs, self._env.scene.env_origins.device,
                         env=self._env),
        int(ref["n_frames"]),
      )
      _, table_pose = _cuboid_scene_poses_at_frame(
        ref, self._env.scene.env_origins, table_frame
      )
      table_pose = _drop_table_after_cf(self._env, table_pose)
      _write_table_pose(table, table_pose)
    self._apply_reference_object_tracking(ref)
    self._entity.set_joint_position_target(target, joint_ids=self._target_ids)
    q = self._entity.data.joint_pos[:, self._target_ids]
    qd = self._entity.data.joint_vel[:, self._target_ids]
    if bool(getattr(self, "_teacher_velocity_feedforward", False)):
      frame = _tracking_frame(self._env, ref["n_frames"])
      target_qd = ref["dof_vel"][frame, : NUM_BODY + NUM_HAND]
      effort = self._kp.unsqueeze(0) * (target - q) + self._kd.unsqueeze(0) * (
        target_qd - qd
      )
    else:
      effort = self._kp.unsqueeze(0) * (target - q) - self._kd.unsqueeze(0) * qd
    effort = effort.clamp(
      -self._effort_limit.unsqueeze(0), self._effort_limit.unsqueeze(0)
    )
    if hard_hold_mask.any():
      effort[hard_hold_mask] = 0.0
    self._entity.set_joint_effort_target(effort, joint_ids=self._target_ids)
    self._apply_startup_pelvis_band()

  def reset(self, env_ids=None) -> None:
    self._raw_actions[env_ids] = 0.0
    ids = (
      torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
    )
    self._band_anchor[ids, :2] = self._env.scene.env_origins[ids, :2]
    self._band_anchor[ids, 2] = self._env.scene.env_origins[ids, 2] + 0.80
    self._band_target_quat[ids] = 0.0
    self._band_target_quat[ids, 0] = 1.0

  def _apply_startup_pelvis_band(self) -> None:
    ep_len = self._env.episode_length_buf
    gain = torch.zeros(self.num_envs, device=self.device)
    gain = torch.where((ep_len >= 31) & (ep_len <= 33), torch.ones_like(gain), gain)
    release = (ep_len >= 34) & (ep_len <= 36)
    release_alpha = ((ep_len.float() - 34.0) / 3.0).clamp(0.0, 1.0)
    gain = torch.where(release, 1.0 - release_alpha, gain)
    assist = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    if (
      self.cfg.tracking_start_assist_gain > 0.0
      and self.cfg.tracking_start_assist_steps > 0
    ):
      assist = (ep_len > 36) & (ep_len < 36 + int(self.cfg.tracking_start_assist_steps))
      assist_progress = (
        (ep_len.float() - 36.0) / float(self.cfg.tracking_start_assist_steps)
      ).clamp(0.0, 1.0)
      assist_gain = float(self.cfg.tracking_start_assist_gain) * (1.0 - assist_progress)
      gain = torch.where(assist, torch.maximum(gain, assist_gain), gain)
    body_pos = self._entity.data.body_link_pos_w[:, self._pelvis_body_id]
    body_quat = self._entity.data.body_link_quat_w[:, self._pelvis_body_id]
    lin_vel = self._entity.data.body_link_lin_vel_w[:, self._pelvis_body_id]
    ang_vel = self._entity.data.body_link_ang_vel_w[:, self._pelvis_body_id]
    target_anchor = self._band_anchor
    if self.cfg.tracking_start_assist_follow_ref_xy and assist.any():
      ref = _ref(self.device)
      frame = _tracking_frame(self._env, ref["n_frames"])
      target_anchor = self._band_anchor.clone()
      target_anchor[assist, :2] = (
        ref["root_pos"][frame[assist], :2] + self._env.scene.env_origins[assist, :2]
      )
    s = gain.unsqueeze(-1)
    force = (s * 800.0 * (target_anchor - body_pos) - s * 200.0 * lin_vel).clamp(
      -1500.0, 1500.0
    )
    if (
      not self.cfg.tracking_start_assist_hold_xy
      and not self.cfg.tracking_start_assist_follow_ref_xy
      and assist.any()
    ):
      force[assist, :2] = 0.0
    quat_err = quat_mul(self._band_target_quat, quat_conjugate(body_quat))
    rotvec = _quat_to_rotvec(quat_err)
    torque = (s * 500.0 * rotvec - s * 10.0 * ang_vel).clamp(-250.0, 250.0)
    active = (gain > 0.0).unsqueeze(-1)
    force = force * active
    torque = torque * active
    self._entity.write_external_wrench_to_sim(
      force.unsqueeze(1),
      torque.unsqueeze(1),
      body_ids=[self._pelvis_body_id],
    )

  def _apply_reference_object_tracking(self, ref: dict) -> None:
    if not bool(self.cfg.reference_object_tracking):
      return
    try:
      apple = object_pool.active(self._env)
    except KeyError:
      return
    n_frames = int(ref["n_frames"])
    frame = _tracking_frame(self._env, n_frames)
    start = max(0, int(self.cfg.reference_object_tracking_start_frame))
    end = int(self.cfg.reference_object_tracking_end_frame)
    if end < 0:
      end = n_frames - 1
    end = max(start, min(end, n_frames - 1))
    mask = (frame >= start) & (frame <= end)
    if not mask.any():
      return
    env_ids = mask.nonzero(as_tuple=False).squeeze(-1)
    frame_ids = frame[env_ids]
    obj_state = torch.zeros(int(env_ids.numel()), 13, device=self.device)
    obj_state[:, :3] = ref["obj_pos"][frame_ids] + self._env.scene.env_origins[env_ids]
    obj_state[:, 2] += float(self.cfg.reference_object_tracking_clearance)
    obj_state[:, 3:7] = ref["obj_quat"][frame_ids]
    obj_state[:, 7:10] = ref["obj_vel"][frame_ids]
    object_pool.write_root_state(self._env, obj_state, env_ids)


def _heading_local_vec(
  root_quat_wxyz: torch.Tensor, vec_w: torch.Tensor
) -> torch.Tensor:
  heading_inv = quat_conjugate(_calc_heading_quat(root_quat_wxyz))
  if vec_w.dim() == 2:
    return quat_apply(heading_inv, vec_w)
  orig_shape = vec_w.shape
  heading_inv = heading_inv.view(orig_shape[0], *([1] * (vec_w.dim() - 2)), 4)
  heading_inv = heading_inv.expand(*orig_shape[:-1], 4).reshape(-1, 4)
  return quat_apply(heading_inv, vec_w.reshape(-1, 3)).reshape(orig_shape)


def _heading_local_rot6d(
  root_quat_wxyz: torch.Tensor, quat_wxyz: torch.Tensor
) -> torch.Tensor:
  heading_inv = quat_conjugate(_calc_heading_quat(root_quat_wxyz))
  rel = quat_mul(
    heading_inv, quat_wxyz / quat_wxyz.norm(dim=-1, keepdim=True).clamp_min(1e-9)
  )
  return _quat_to_rot6d(rel)


class IsaacStyleObs2964:
  def __init__(self, cfg, env):
    del cfg
    self.env = env
    self.device = env.device
    self.num_envs = env.num_envs
    self.body_joint_ids = None
    self.hand_joint_ids = None
    self.tip_body_ids = None
    self.hist_body_jp = torch.zeros(self.num_envs, 10, NUM_BODY, device=self.device)
    self.hist_body_jv = torch.zeros_like(self.hist_body_jp)
    self.hist_hand_jp = torch.zeros(self.num_envs, 10, NUM_HAND, device=self.device)
    self.hist_hand_jv = torch.zeros_like(self.hist_hand_jp)
    self.hist_ang_vel = torch.zeros(self.num_envs, 10, 3, device=self.device)
    self.hist_anchor = torch.zeros(self.num_envs, 10, 6, device=self.device)
    self.hist_body_jp_il = torch.zeros_like(self.hist_body_jp)
    self.hist_body_jv_il = torch.zeros_like(self.hist_body_jp)
    self.hist_body_act_il = torch.zeros_like(self.hist_body_jp)
    self.hist_gravity = torch.zeros(self.num_envs, 10, 3, device=self.device)
    self.hist_hand_act = torch.zeros_like(self.hist_hand_jp)

  def _resolve(self, env):
    robot: Entity = env.scene["robot"]
    if self.body_joint_ids is None:
      ids, _ = robot.find_joints(BODY_29_DOF_NAMES, preserve_order=True)
      self.body_joint_ids = torch.tensor(ids, device=self.device, dtype=torch.long)
      ids, _ = robot.find_joints(HAND_24_DOF_NAMES, preserve_order=True)
      self.hand_joint_ids = torch.tensor(ids, device=self.device, dtype=torch.long)
      body_names = TIP_BODY_NAMES
      ids, _ = robot.find_bodies(body_names, preserve_order=True)
      self.tip_body_ids = torch.tensor(ids, device=self.device, dtype=torch.long)

  def reset(self, env_ids=None):
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
    elif isinstance(env_ids, slice):
      env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
    env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
    ref = _ref(self.device)
    n = env_ids.numel()
    frame = _reference_start_frame(self.env, ref["n_frames"])[env_ids]
    body_pkl = ref["dof_pos"][frame, :NUM_BODY]
    hand = ref["dof_pos"][frame, NUM_BODY : NUM_BODY + NUM_HAND]
    body_jp_norm = body_pkl - ref["default_body"].unsqueeze(0)
    hand_jp_norm = hand - ref["default_hand"].unsqueeze(0)
    pkl_for_il = torch.tensor(PKL_FOR_IL, device=self.device, dtype=torch.long)
    body_jp_il = body_pkl[:, pkl_for_il] - torch.tensor(
      SONIC_DEFAULT_ANGLES_IL, device=self.device
    )
    body_scale = torch.tensor(
      SONIC_ACTION_SCALE_PKL, device=self.device, dtype=torch.float32
    )
    teacher_body_il = (
      (body_pkl - torch.tensor(SONIC_DEFAULT_ANGLES_PKL, device=self.device))
      / body_scale
    )[:, pkl_for_il]
    gravity = quat_apply(
      quat_conjugate(ref["root_rot"][frame]),
      torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(n, 3),
    )
    anchor = _quat_to_rot6d(ref["root_rot"][frame])
    for i in range(10):
      self.hist_body_jp[env_ids, i] = body_jp_norm
      self.hist_body_jv[env_ids, i] = 0.0
      self.hist_hand_jp[env_ids, i] = hand_jp_norm
      self.hist_hand_jv[env_ids, i] = 0.0
      self.hist_ang_vel[env_ids, i] = 0.0
      self.hist_anchor[env_ids, i] = anchor
      self.hist_body_jp_il[env_ids, i] = body_jp_il
      self.hist_body_jv_il[env_ids, i] = 0.0
      self.hist_body_act_il[env_ids, i] = teacher_body_il
      self.hist_hand_act[env_ids, i] = 0.0
      self.hist_gravity[env_ids, i] = gravity

  @staticmethod
  def _append_history(buf: torch.Tensor, value: torch.Tensor) -> None:
    buf[:, :-1].copy_(buf[:, 1:].clone())
    buf[:, -1].copy_(value)

  def __call__(
    self, env, asset_cfg: SceneEntityCfg = _ROBOT_ENTITY_CFG
  ) -> torch.Tensor:
    del asset_cfg
    self._resolve(env)
    robot: Entity = env.scene["robot"]
    apple: Entity = object_pool.active(env)
    ref = _ref(env.device)
    frame = _tracking_frame(env, ref["n_frames"])

    body_jp = robot.data.joint_pos[:, self.body_joint_ids]
    body_jv = robot.data.joint_vel[:, self.body_joint_ids]
    hand_jp = robot.data.joint_pos[:, self.hand_joint_ids]
    hand_jv = robot.data.joint_vel[:, self.hand_joint_ids]
    root_pose = robot.data.root_link_pose_w
    root_quat = root_pose[:, 3:7]
    root_h = root_pose[:, 2:3] - env.scene.env_origins[:, 2:3]
    root_vel = robot.data.root_link_vel_w
    ang_vel = robot.data.root_link_ang_vel_b
    grav_world = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(
      env.num_envs, 3
    )
    gravity = quat_apply(quat_conjugate(root_quat), grav_world)
    obj_pos = apple.data.root_link_pos_w
    obj_quat = apple.data.root_link_pose_w[:, 3:7]

    anchor = _quat_to_rot6d(root_quat)

    self._append_history(self.hist_body_jp, body_jp - ref["default_body"].unsqueeze(0))
    self._append_history(self.hist_body_jv, body_jv)
    self._append_history(self.hist_hand_jp, hand_jp - ref["default_hand"].unsqueeze(0))
    self._append_history(self.hist_hand_jv, hand_jv)
    self._append_history(self.hist_ang_vel, ang_vel)
    self._append_history(self.hist_anchor, anchor)
    pkl_for_il = torch.tensor(PKL_FOR_IL, device=env.device, dtype=torch.long)
    self._append_history(
      self.hist_body_jp_il,
      body_jp[:, pkl_for_il] - torch.tensor(SONIC_DEFAULT_ANGLES_IL, device=env.device),
    )
    self._append_history(self.hist_body_jv_il, body_jv[:, pkl_for_il])
    self._append_history(self.hist_gravity, gravity)
    effective_action = effective_sonic_action(env)
    startup_teacher_mask = (env.episode_length_buf >= 31) & (
      env.episode_length_buf <= 36
    )
    if startup_teacher_mask.any():
      effective_action[startup_teacher_mask] = teacher_action(env)[startup_teacher_mask]
    self._append_history(self.hist_body_act_il, effective_action[:, :NUM_BODY])
    self._append_history(self.hist_hand_act, effective_action[:, NUM_BODY:])

    root_vel_local = _heading_local_vec(root_quat, root_vel[:, :3])
    obj_pos_root_local = _heading_local_vec(root_quat, obj_pos - root_pose[:, :3])
    obj_rot_6d = _heading_local_rot6d(root_quat, obj_quat)
    remaining = (int(ref["n_frames"]) - frame).clamp(min=1)
    step = (remaining // 10).clamp(min=1)
    waypoint = torch.arange(1, 11, device=env.device, dtype=torch.long).unsqueeze(0)
    future_frames = (frame.unsqueeze(1) + waypoint * step.unsqueeze(1)).clamp(
      max=int(ref["n_frames"]) - 1
    )
    fut = ref["obj_pos"][future_frames] + env.scene.env_origins.unsqueeze(1)
    fut_obj_delta_local = _heading_local_vec(
      root_quat, fut - obj_pos.unsqueeze(1)
    ).reshape(env.num_envs, 30)
    tip_pos = robot.data.body_link_pos_w[:, self.tip_body_ids]
    fingertip_to_obj_local = _heading_local_vec(
      root_quat, tip_pos - obj_pos.unsqueeze(1)
    ).reshape(env.num_envs, tip_pos.shape[1], 3)
    fingertip_to_obj_local = fingertip_to_obj_local[:, :10].reshape(env.num_envs, 30)
    contact_force_w = torch.zeros(env.num_envs, 10, 3, device=env.device)
    try:
      sensor = env.scene["hand_apple_contact"]
      force = sensor.data.force
      if force is not None:
        n_tip = min(10, force.shape[1])
        contact_force_w[:, :n_tip] = force[:, :n_tip]
    except Exception:
      pass
    contact_force_local = _heading_local_vec(root_quat, contact_force_w)
    contact_force_mag = contact_force_local.norm(dim=-1, keepdim=True)
    contact_force_local_log = (
      contact_force_local / contact_force_mag.clamp(min=1e-6)
    ) * torch.log1p(contact_force_mag)
    contact_force_local_log = contact_force_local_log.reshape(env.num_envs, 30)
    t_norm = frame.float() / max(float(ref["n_frames"] - 1), 1.0)

    enc_input = torch.cat(
      [
        self.hist_body_jp.reshape(env.num_envs, -1),
        self.hist_body_jv.reshape(env.num_envs, -1),
        self.hist_hand_jp.reshape(env.num_envs, -1),
        self.hist_hand_jv.reshape(env.num_envs, -1),
        self.hist_ang_vel.reshape(env.num_envs, -1),
        self.hist_anchor.reshape(env.num_envs, -1),
        root_h,
        root_vel_local,
        obj_pos_root_local,
        obj_rot_6d,
        fut_obj_delta_local,
        fingertip_to_obj_local,
        contact_force_local_log,
        t_norm.unsqueeze(-1),
      ],
      dim=-1,
    )
    body_dec_hist = torch.cat(
      [
        self.hist_ang_vel.reshape(env.num_envs, -1),
        self.hist_body_jp_il.reshape(env.num_envs, -1),
        self.hist_body_jv_il.reshape(env.num_envs, -1),
        self.hist_body_act_il.reshape(env.num_envs, -1),
        self.hist_gravity.reshape(env.num_envs, -1),
      ],
      dim=-1,
    )
    hand_dec_hist = torch.cat(
      [
        self.hist_ang_vel.reshape(env.num_envs, -1),
        self.hist_hand_jp.reshape(env.num_envs, -1),
        self.hist_hand_jv.reshape(env.num_envs, -1),
        self.hist_hand_act.reshape(env.num_envs, -1),
        self.hist_gravity.reshape(env.num_envs, -1),
      ],
      dim=-1,
    )
    obs = torch.cat([enc_input, body_dec_hist, hand_dec_hist], dim=-1)
    return torch.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)


def zero_reward(env) -> torch.Tensor:
  return torch.zeros(env.num_envs, device=env.device)


def fingertip_min_dist(
  env,
  robot_cfg: SceneEntityCfg = _ROBOT_ENTITY_CFG,
  object_cfg: SceneEntityCfg = _APPLE_ENTITY_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[robot_cfg.name]
  obj: Entity = env.scene[object_cfg.name]
  body_ids, _ = robot.find_bodies(
    TIP_BODY_NAMES,
    preserve_order=True,
  )
  body_ids_t = torch.tensor(body_ids, device=env.device, dtype=torch.long)
  tip_pos = robot.data.body_link_pos_w[:, body_ids_t]
  return (
    (tip_pos - obj.data.root_link_pos_w.unsqueeze(1)).norm(dim=-1).min(dim=-1).values
  )


def isaac_action_mimic_reward(
  env,
  pre_weight: float = 0.40,
  post_weight: float = 0.10,
  k: float = 2.0,
  body_weight: float = 1.0,
  hand_weight: float = 0.5,
  apply_active_mask: bool = True,
) -> torch.Tensor:
  ref = _ref(env.device)
  frame = _tracking_frame(env, ref["n_frames"])
  actions = env.action_manager.action
  body_scale = torch.tensor(
    SONIC_ACTION_SCALE_PKL, device=env.device, dtype=torch.float32
  )
  body_default = torch.tensor(
    SONIC_DEFAULT_ANGLES_PKL, device=env.device, dtype=torch.float32
  )
  pkl_for_il = torch.tensor(PKL_FOR_IL, device=env.device, dtype=torch.long)
  teacher_body_pkl = (
    ref["dof_pos"][frame, :NUM_BODY] - body_default.unsqueeze(0)
  ) / body_scale.unsqueeze(0)
  teacher_body_il = teacher_body_pkl[:, pkl_for_il]
  teacher_hand = ref["dof_pos"][frame, NUM_BODY : NUM_BODY + NUM_HAND] - ref[
    "default_hand"
  ].unsqueeze(0)
  body_err = (actions[:, :NUM_BODY] - teacher_body_il).pow(2).mean(dim=-1)
  hand_err = (
    (actions[:, NUM_BODY : NUM_BODY + NUM_HAND] - teacher_hand).pow(2).mean(dim=-1)
  )
  raw = torch.exp(
    -float(k) * (float(body_weight) * body_err + float(hand_weight) * hand_err)
  )
  grasp_start = int(ref["n_frames"]) // 2
  weight = torch.where(
    frame > grasp_start,
    torch.full_like(raw, float(post_weight)),
    torch.full_like(raw, float(pre_weight)),
  )
  value = weight * raw
  if apply_active_mask:
    value = value * _active_after_startup(env)
  return value


def teacher_action(env) -> torch.Tensor:
  ref = _ref(env.device)
  frame = _tracking_frame(env, ref["n_frames"])
  body_scale = torch.tensor(
    SONIC_ACTION_SCALE_PKL, device=env.device, dtype=torch.float32
  )
  body_default = torch.tensor(
    SONIC_DEFAULT_ANGLES_PKL, device=env.device, dtype=torch.float32
  )
  pkl_for_il = torch.tensor(PKL_FOR_IL, device=env.device, dtype=torch.long)
  body_pkl = (
    ref["dof_pos"][frame, :NUM_BODY] - body_default.unsqueeze(0)
  ) / body_scale.unsqueeze(0)
  body_il = body_pkl[:, pkl_for_il]
  hand = ref["dof_pos"][frame, NUM_BODY : NUM_BODY + NUM_HAND] - ref[
    "default_hand"
  ].unsqueeze(0)
  return torch.cat([body_il, hand], dim=-1).detach()


def aux_bc_lambdas(
  env,
  bc_lambda_pre: float = 2.0,
  bc_lambda_mid: float = 0.5,
  bc_lambda_contact: float = 0.05,
  bc_contact_dist: float = 0.12,
) -> torch.Tensor:
  ref = _ref(env.device)
  frame = _tracking_frame(env, ref["n_frames"])
  min_tip_dist = fingertip_min_dist(env)
  obj = object_pool.active(env)
  post_phase = (frame - _clip_bounds(env, int(ref["n_frames"]))[0]) > (
      int(ref["n_frames"]) // 2
    )
  lifted = obj.data.root_link_pos_w[:, 2] > (ref["obj_pos"][0, 2] + 0.05)
  touch_like = min_tip_dist < float(bc_contact_dist)
  contact_or_lift = touch_like | lifted
  lambdas = torch.where(
    ~post_phase,
    torch.full_like(min_tip_dist, float(bc_lambda_pre)),
    torch.where(
      contact_or_lift,
      torch.full_like(min_tip_dist, float(bc_lambda_contact)),
      torch.full_like(min_tip_dist, float(bc_lambda_mid)),
    ),
  )
  lambdas = torch.where(
    _active_for_bootstrap_loss(env).bool(), lambdas, torch.zeros_like(lambdas)
  )
  return lambdas.unsqueeze(-1).detach()


def dagger_prior_lambdas(
  env,
  pre_lambda: float = 20.0,
  ready_lambda: float = 2.0,
  contact_lambda: float = 0.2,
  ready_dist: float = 0.30,
  contact_dist: float = 0.12,
) -> torch.Tensor:
  ref = _ref(env.device)
  frame = _tracking_frame(env, ref["n_frames"])
  min_tip_dist = fingertip_min_dist(env)
  obj = object_pool.active(env)
  near_pregrasp = (frame - _clip_bounds(env, int(ref["n_frames"]))[0]) >= max(
      (int(ref["n_frames"]) // 2) - 20, 0
    )
  lifted = obj.data.root_link_pos_w[:, 2] > (ref["obj_pos"][0, 2] + 0.05)
  contact_or_lift = (min_tip_dist < float(contact_dist)) | lifted
  ready = near_pregrasp & (min_tip_dist < float(ready_dist))
  lambdas = torch.full_like(min_tip_dist, float(pre_lambda))
  lambdas = torch.where(ready, torch.full_like(lambdas, float(ready_lambda)), lambdas)
  lambdas = torch.where(
    contact_or_lift, torch.full_like(lambdas, float(contact_lambda)), lambdas
  )
  lambdas = torch.where(
    _active_for_bootstrap_loss(env).bool(), lambdas, torch.zeros_like(lambdas)
  )
  return lambdas.unsqueeze(-1).detach()


def isaac_bootstrap_reward(env) -> torch.Tensor:
  active = _active_after_startup(env)
  action_mimic = isaac_action_mimic_reward(env)
  action_mimic_log = isaac_action_mimic_reward(env, apply_active_mask=False)
  min_tip_dist = fingertip_min_dist(env)
  stage = 8.2 * 0.9 * torch.exp(-5.0 * min_tip_dist) * active

  robot: Entity = env.scene["robot"]
  power = (robot.data.qfrc_actuator * robot.data.joint_vel).abs().sum(dim=-1)
  power_penalty = (-0.0005 * power).clamp(-1.0, 1.0) * active
  power_penalty = torch.where(
    _tracking_frame(env, _ref(env.device)["n_frames"]) <= 3,
    torch.zeros_like(power_penalty),
    power_penalty,
  )

  foot_penalty = torch.zeros(env.num_envs, device=env.device)
  try:
    foot_ids, _ = robot.find_bodies(
      ("left_ankle_roll_link", "right_ankle_roll_link"), preserve_order=True
    )
    foot_ids_t = torch.tensor(foot_ids, device=env.device, dtype=torch.long)
    foot_speed = robot.data.body_link_lin_vel_w[:, foot_ids_t].norm(dim=-1)
    foot_penalty = foot_speed.sum(dim=-1).mul(0.85).clamp(0.0, 1.0) * active
  except Exception:
    foot_penalty = torch.zeros(env.num_envs, device=env.device)

  total = stage + action_mimic + power_penalty - foot_penalty
  total = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)
  log = env.extras.setdefault("log", {})
  log["Reward/total"] = total.mean()
  log["Reward/stage"] = stage.mean()
  log["Reward/pregrasp_contrib"] = stage.mean()
  log["Reward/grab_contrib"] = torch.zeros((), device=env.device)
  log["Interact/action_mimic"] = action_mimic_log.mean()
  log["Reward/power_penalty"] = power_penalty.mean()
  log["Reward/slippage_penalty"] = (-foot_penalty).mean()
  log["Metric/multitip_dist"] = min_tip_dist.mean()
  log["Metric/pass_contact_time_frac"] = torch.zeros((), device=env.device)
  log["Metric/start_frame_mean"] = (
    _tracking_frame(env, _ref(env.device)["n_frames"]).float().mean()
  )
  return total


def hand_to_object_reward(
  env,
  robot_cfg: SceneEntityCfg = _ROBOT_ENTITY_CFG,
  object_cfg: SceneEntityCfg = _APPLE_ENTITY_CFG,
  std: float = 0.25,
) -> torch.Tensor:
  robot: Entity = env.scene[robot_cfg.name]
  obj: Entity = env.scene[object_cfg.name]
  hand_body_names = TIP_BODY_NAMES
  body_ids, _ = robot.find_bodies(hand_body_names, preserve_order=True)
  body_ids_t = torch.tensor(body_ids, device=env.device, dtype=torch.long)
  hand_pos = robot.data.body_link_pos_w[:, body_ids_t]
  obj_pos = obj.data.root_link_pos_w
  min_dist = (hand_pos - obj_pos.unsqueeze(1)).norm(dim=-1).min(dim=-1).values
  return torch.exp(-min_dist / max(float(std), 1e-6)) * _active_after_startup(env)


def object_reference_reward(
  env,
  object_cfg: SceneEntityCfg = _APPLE_ENTITY_CFG,
  std: float = 0.25,
) -> torch.Tensor:
  ref = _ref(env.device)
  obj: Entity = env.scene[object_cfg.name]
  frame = _tracking_frame(env, ref["n_frames"])
  ref_pos = ref["obj_pos"][frame] + env.scene.env_origins
  dist = (obj.data.root_link_pos_w - ref_pos).norm(dim=-1)
  return torch.exp(-dist / max(float(std), 1e-6)) * _active_after_startup(env)


def contact_bonus(
  env,
  robot_cfg: SceneEntityCfg = _ROBOT_ENTITY_CFG,
  object_cfg: SceneEntityCfg = _APPLE_ENTITY_CFG,
  threshold: float = 0.08,
) -> torch.Tensor:
  robot: Entity = env.scene[robot_cfg.name]
  obj: Entity = env.scene[object_cfg.name]
  hand_body_names = TIP_BODY_NAMES
  body_ids, _ = robot.find_bodies(hand_body_names, preserve_order=True)
  body_ids_t = torch.tensor(body_ids, device=env.device, dtype=torch.long)
  hand_pos = robot.data.body_link_pos_w[:, body_ids_t]
  min_dist = (
    (hand_pos - obj.data.root_link_pos_w.unsqueeze(1)).norm(dim=-1).min(dim=-1).values
  )
  return (min_dist < float(threshold)).float() * _active_after_startup(env)


def lift_reward(
  env,
  object_cfg: SceneEntityCfg = _APPLE_ENTITY_CFG,
  table_top_z: float = 0.78,
  lift_height: float = 0.03,
) -> torch.Tensor:
  obj: Entity = env.scene[object_cfg.name]
  return (
    obj.data.root_link_pos_w[:, 2] > float(table_top_z + lift_height)
  ).float() * _active_after_startup(env)


def object_motion_reward(
  env,
  object_cfg: SceneEntityCfg = _APPLE_ENTITY_CFG,
  speed_std: float = 0.25,
) -> torch.Tensor:
  obj: Entity = env.scene[object_cfg.name]
  speed = obj.data.root_link_vel_w[:, :3].norm(dim=-1)
  return (
    1.0 - torch.exp(-speed / max(float(speed_std), 1e-6))
  ) * _active_after_startup(env)


def reset_to_apple_eat_frame(
  env, env_ids: torch.Tensor | None, frame: int | torch.Tensor = 0
):
  env_ids = resolve_env_ids(env, env_ids)
  ref = _ref(env.device)
  frame_t = _set_reference_start_frame(env, env_ids, frame, ref["n_frames"])
  robot: Entity = env.scene["robot"]
  apple: Entity = object_pool.active(env)
  table: Entity = env.scene["table"]
  joint_names = list(BODY_29_DOF_NAMES) + list(HAND_24_DOF_NAMES)
  joint_ids, _ = robot.find_joints(joint_names, preserve_order=True)
  joint_ids_t = torch.tensor(joint_ids, device=env.device, dtype=torch.long)
  n = len(env_ids)
  root_state = torch.zeros(n, 13, device=env.device)
  root_state[:, :3] = ref["root_pos"][frame_t] + env.scene.env_origins[env_ids]
  root_state[:, 3:7] = ref["root_rot"][frame_t]
  root_state[:, 7:10] = ref["root_lin_vel"][frame_t]
  root_state[:, 10:13] = ref["root_ang_vel"][frame_t]
  robot.write_root_state_to_sim(root_state, env_ids=env_ids)
  q = ref["dof_pos"][frame_t]
  qd = ref["dof_vel"][frame_t]
  robot.write_joint_state_to_sim(q, qd, joint_ids=joint_ids_t, env_ids=env_ids)

  # The task scene is anchored at frame 0. RSI only advances the robot along
  # the pre-contact approach; the apple and table must remain at their initial
  # poses until the policy physically moves the apple.
  # frame 0 of each environment's own clip: global row 0 belongs to clip 0 alone
  _n_per = max(int(ref["n_frames"]), 1)
  _band_lo = torch.div(frame_t, _n_per, rounding_mode="floor") * _n_per
  scene_frame = frame_t.clone() if RSI_SCENE_FOLLOWS_ROBOT else _band_lo
  obj_pose, table_pose = _cuboid_scene_poses_at_frame(
    ref,
    env.scene.env_origins[env_ids],
    scene_frame,
  )
  table_frame = _table_reference_frame(env, scene_frame, int(ref["n_frames"]))
  if isinstance(table_frame, torch.Tensor) and table_frame.shape[0] == env.num_envs:
    table_frame = table_frame[env_ids]
  _, table_pose = _cuboid_scene_poses_at_frame(
    ref,
    env.scene.env_origins[env_ids],
    table_frame,
  )

  obj_state = torch.zeros(n, 13, device=env.device)
  obj_state[:, :7] = obj_pose
  object_pool.write_root_state(env, obj_state, env_ids)

  _write_table_pose(table, table_pose, env_ids=env_ids)


def reset_to_apple_eat_frame0(env, env_ids: torch.Tensor | None):
  reset_to_apple_eat_frame(env, env_ids, frame=0)


def not_fallen(env, asset_cfg: SceneEntityCfg = _ROBOT_ENTITY_CFG) -> torch.Tensor:
  active = _active_after_startup(env).bool()
  robot: Entity = env.scene[asset_cfg.name]
  root_z = robot.data.root_link_pos_w[:, 2]
  return (root_z < 0.45) & active


def object_drift_termination(
  env,
  object_cfg: SceneEntityCfg = _APPLE_ENTITY_CFG,
  threshold: float = 0.75,
) -> torch.Tensor:
  ref = _ref(env.device)
  obj: Entity = env.scene[object_cfg.name]
  active = _active_after_startup(env).bool()
  frame = _tracking_frame(env, ref["n_frames"])
  ref_pos = ref["obj_pos"][frame] + env.scene.env_origins
  dist = (obj.data.root_link_pos_w - ref_pos).norm(dim=-1)
  return (dist > float(threshold)) & active


class NoProgressTermination:
  def __init__(self, cfg, env):
    del cfg
    self.best = torch.full((env.num_envs,), float("inf"), device=env.device)
    self.count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

  def reset(self, env_ids=None):
    if env_ids is None:
      env_ids = slice(None)
    self.best[env_ids] = float("inf")
    self.count[env_ids] = 0

  def __call__(
    self,
    env,
    min_delta: float = 0.01,
    warmup_steps: int = 80,
    patience_steps: int = 80,
  ) -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    obj: Entity = object_pool.active(env)
    body_ids, _ = robot.find_bodies(
      TIP_BODY_NAMES,
      preserve_order=True,
    )
    body_ids_t = torch.tensor(body_ids, device=env.device, dtype=torch.long)
    hand_pos = robot.data.body_link_pos_w[:, body_ids_t]
    min_dist = (
      (hand_pos - obj.data.root_link_pos_w.unsqueeze(1)).norm(dim=-1).min(dim=-1).values
    )
    improved = min_dist < (self.best - float(min_delta))
    self.best.copy_(torch.minimum(self.best, min_dist))
    active = env.episode_length_buf > int(warmup_steps)
    self.count.copy_(
      torch.where(active & ~improved, self.count + 1, torch.zeros_like(self.count))
    )
    return self.count > int(patience_steps)
