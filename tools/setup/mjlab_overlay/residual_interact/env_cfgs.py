"""Environment config for SONIC-backed residual interaction RL."""

from __future__ import annotations

import functools
import os
from pathlib import Path

import mujoco
import torch

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.tasks.apple_eat import object_pool
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import time_out
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.residual_interact import (
  mdp,
  omnigrasp_faithful_mdp,
  omnigrasp_style_mdp,
  staged_mdp,
)
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

_GMR_ROOT = Path(os.environ.get("GMR_ROOT", "/home/jiarui/jiarui/GMR"))
_HAND_KIND = os.environ.get("APPLE_HAND_KIND", "xhand").strip().lower()
# Real object mesh instead of the 4 cm sphere. Off unless APPLE_OBJECT_MESH is set, so every
# existing run keeps the geometry it trained with.
_OBJECT_MESH = mdp.OBJECT_STL if getattr(mdp, "USE_OBJECT_MESH", False) else ""
# Cap on convex-hull vertices for the object collision mesh. The raw GRAB meshes hull
# to ~31k vertices, which crashes the solver; 64 is plenty for these shapes.
_OBJECT_MAXHULLVERT = int(mdp.apple_mdp.rank_pick("APPLE_OBJECT_MAXHULLVERT", "64"))


def _num_list(name: str, default: str) -> tuple[float, ...]:
  # rank_pick so a hybrid run can give one rank a different contact setup if it ever needs to;
  # with no *_LIST set this is exactly os.environ.get.
  return tuple(float(x) for x in mdp.apple_mdp.rank_pick(name, default).split(",") if x.strip())


# Contact solver parameters for the object geom. The defaults below ARE MuJoCo's defaults, so an unset
# environment leaves behaviour identical. Raising priority above the table's makes these win instead
# of being averaged with it.
_OBJECT_SOLREF = _num_list("APPLE_OBJECT_SOLREF", "0.02,1.0")
_OBJECT_SOLIMP = _num_list("APPLE_OBJECT_SOLIMP", "0.9,0.95,0.001,0.5,2.0")
_OBJECT_PRIORITY = int(mdp.apple_mdp.rank_pick("APPLE_OBJECT_PRIORITY", "0"))
# "box" collides an axis-aligned box matched to the collision mesh's bounding box, through MuJoCo's
# analytic box-box collider instead of GJK/EPA on a mesh. Both cap at 4 contact points in this
# mujoco_warp build and typically report 2; the gain is not the count but the consistency. Measured on
# a gravity-only settle against the table, worst-case penetration over 400 steps:
#   mesh hull, default solimp                 -15.87 mm
#   mesh hull, stiff solimp + priority 3       -7.07 mm
#   mesh hull, maxhullvert 8 or 160             -5.4 mm
#   box primitive, stiff solimp + priority 3    -0.76 mm
# GJK/EPA occasionally converges to a much deeper contact; the analytic collider cannot.
# The drawn shape is unaffected: the real mesh is added as a contype=0 visual, the same split the
# apple task uses with its 4 cm sphere.
_OBJECT_PRIMITIVE = mdp.apple_mdp.rank_pick("APPLE_OBJECT_PRIMITIVE", "").strip().lower()
# Geom group for the object's visual mesh. 0 keeps it visible in every renderer, because MuJoCo's
# default geomgroup mask is {1,1,1,0,0,0}. Set to 4 to give it its own toggleable group -- the Viser
# launcher does this so the Object shape dropdown can hide it. The collider is always group 5.
# Radius of the placeholder sphere in the `parts` branch. On the GRAB stapler the largest part's
# centroid has 8.75 mm of clearance to that part's own surface, and the part is convex and contained
# in the object, so 1 mm is buried with an 8x margin. Raise only after re-measuring on the object
# with the thinnest largest-part.
_PARTS_PLACEHOLDER_RADIUS = 0.001
_OBJECT_VISUAL_GROUP = int(os.environ.get("APPLE_OBJECT_VISUAL_GROUP", "0"))
if _OBJECT_VISUAL_GROUP not in (0, 4):
  raise ValueError(f"APPLE_OBJECT_VISUAL_GROUP must be 0 or 4, got {_OBJECT_VISUAL_GROUP}")
# Geom group for the COLLIDER. 5 by default, which MuJoCo's default mask {1,1,1,0,0,0} hides -- so
# what a video shows is the visual mesh, never the boxes the fingers actually meet. Set this to 2 and
# the visual group to 4 to record the collider itself, which is the only way to see a grasp failing
# against a shape that is not the shape being drawn.
_OBJECT_BOX_GROUP = int(os.environ.get("APPLE_OBJECT_BOX_GROUP", "5"))
if not 0 <= _OBJECT_BOX_GROUP <= 5:
  raise ValueError(f"APPLE_OBJECT_BOX_GROUP must be 0..5, got {_OBJECT_BOX_GROUP}")
if _OBJECT_BOX_GROUP == _OBJECT_VISUAL_GROUP:
  raise ValueError(
    f"APPLE_OBJECT_BOX_GROUP and APPLE_OBJECT_VISUAL_GROUP are both {_OBJECT_BOX_GROUP}; the "
    "collider and the mesh would toggle together, which defeats the point of separating them")
if _OBJECT_PRIMITIVE not in ("", "sphere", "box", "boxstack", "parts"):
  raise ValueError(
    f"APPLE_OBJECT_PRIMITIVE must be '', 'sphere', 'box', 'boxstack' or 'parts', "
    f"got {_OBJECT_PRIMITIVE!r}")
# Per-clip object representation, one name per clip in APPLE_EAT_PKL_MIX order, e.g. "sphere,parts".
# Needed because APPLE_OBJECT_MESH is global: with a mesh available for every clip, the apple would
# stop using the analytic sphere it was validated with. This only chooses which branch of
# _apple_spec each variant takes; the per-world scattering is VariantEntityCfg's job.
_OBJECT_PERCLIP = tuple(
  x.strip().lower()
  for x in mdp.apple_mdp.rank_pick("APPLE_OBJECT_PERCLIP", "").split(",")
  if x.strip()
)
for _k in _OBJECT_PERCLIP:
  if _k not in ("sphere", "box", "boxstack", "parts"):
    raise ValueError(
      f"APPLE_OBJECT_PERCLIP entries must be 'sphere', 'box', 'boxstack' or 'parts', got {_k!r}")
# boxstack cell counts along the long axis and the height, "n_x,n_z". 2,3 is the measured optimum
# for the stapler: it keeps the waist on both halves and leaves the paper throat open.
_OBJECT_BOXSTACK = tuple(
  int(x) for x in mdp.apple_mdp.rank_pick("APPLE_OBJECT_BOXSTACK", "2,3").split(",")
)
if len(_OBJECT_BOXSTACK) != 2 or min(_OBJECT_BOXSTACK) < 1:
  raise ValueError(f"APPLE_OBJECT_BOXSTACK must be 'n_x,n_z' with both >= 1, "
                   f"got {_OBJECT_BOXSTACK!r}")


def _object_part_files(mesh_path: str) -> tuple[list[str], tuple[float, float, float]]:
  """Pre-baked convex-decomposition parts for a mesh, plus a point provably inside the object.

  Reads `<base>_parts.json`, written by tools/model/decompose_object.py, next to the mesh. Raises if
  it is missing rather than silently falling back to a hull -- a collider that is not the one the
  config asked for is exactly the failure mode that cost a whole replication attempt.

  The second return value is the centroid of the largest part. Each part is convex and contained in
  the object, so a small sphere at that centroid cannot poke out; it is where the placeholder sphere
  primitive goes for objects that are not round.
  """
  import json

  base = Path(mesh_path)
  manifest = base.with_name(f"{base.stem}_parts.json")
  if not manifest.exists():
    raise FileNotFoundError(
      f"{manifest} is missing; build it with "
      f"`python tools/model/decompose_object.py --mesh {mesh_path} --parts N --max-hull-vert 64`"
    )
  meta = json.loads(manifest.read_text())
  files = [str(base.with_name(name)) for name in meta["files"]]
  missing = [f for f in files if not Path(f).exists()]
  if missing:
    raise FileNotFoundError(f"{manifest} lists parts that do not exist: {missing}")
  biggest = max(meta["parts"], key=lambda p: p["volume_cm3"])
  centre = tuple(float(x) for x in biggest["centroid_m"])
  return files, centre


def _boxstack_cells(mesh_path: str, n_x: int, n_z: int, nbins: int = 40):
  """Axis-aligned box decomposition of a mesh: [(centre, half_extents)] in metres.

  z cut points are the n_z-1 largest steps in the half-width profile, spaced at least 15 % of the
  height apart so two cuts cannot land on the same step. Cells with almost no vertices are dropped,
  which is what leaves a real void open instead of bridging it.
  """
  import numpy as np, trimesh

  v = np.asarray(trimesh.load(mesh_path, process=False).vertices, dtype=np.float64)
  lo, hi = v.min(0), v.max(0)

  zs, ws = [], []
  edges = np.linspace(lo[2], hi[2], nbins + 1)
  for i in range(nbins):
    m = (v[:, 2] >= edges[i]) & (v[:, 2] <= edges[i + 1])
    if m.sum() < 5:
      continue
    zs.append(0.5 * (edges[i] + edges[i + 1]))
    ws.append(float(np.abs(v[m][:, 1]).max()))
  zs, ws = np.array(zs), np.array(ws)

  cuts = []
  if n_z > 1 and len(ws) > 2:
    span = hi[2] - lo[2]
    for i in np.argsort(np.abs(np.diff(ws)))[::-1]:
      zc = 0.5 * (zs[i] + zs[i + 1])
      # Keep clear of the object's OWN top and bottom, not just of the other cuts. The cuts are
      # placed at the largest steps in the half-width profile, and on anything with a flat face the
      # largest step is the one at that face -- so without this the first cut lands a millimetre
      # inside the boundary and the cell behind it is a slab, which is a bad contact primitive no
      # matter how good the volume numbers look. Measured before this line existed: 17 of 23 sweep
      # objects had a box under 2 mm half-extent, a cube had two 1.5 mm slivers at spill 0.00, and
      # the stapler -- the object the decomposition was tuned on -- had none, which is why it was
      # never noticed.
      if zc - lo[2] <= 0.15 * span or hi[2] - zc <= 0.15 * span:
        continue
      if all(abs(zc - c) > 0.15 * span for c in cuts):
        cuts.append(float(zc))
      if len(cuts) == n_z - 1:
        break
  zcuts = [float(lo[2])] + sorted(cuts) + [float(hi[2])]
  xcuts = np.linspace(lo[0], hi[0], n_x + 1)

  cells = []
  for i in range(len(xcuts) - 1):
    for j in range(len(zcuts) - 1):
      m = ((v[:, 0] >= xcuts[i]) & (v[:, 0] <= xcuts[i + 1])
           & (v[:, 2] >= zcuts[j]) & (v[:, 2] <= zcuts[j + 1]))
      if m.sum() < 20:
        continue
      t = v[m]
      centre = (0.5 * (xcuts[i] + xcuts[i + 1]),
                0.5 * (t[:, 1].min() + t[:, 1].max()),
                0.5 * (zcuts[j] + zcuts[j + 1]))
      half = (0.5 * (xcuts[i + 1] - xcuts[i]),
              0.5 * (t[:, 1].max() - t[:, 1].min()),
              0.5 * (zcuts[j + 1] - zcuts[j]))
      if min(half) <= 1e-5:
        continue
      cells.append((tuple(float(c) for c in centre), tuple(float(h) for h in half)))
  if not cells:
    raise ValueError(f"boxstack produced no cells for {mesh_path}")
  return cells

# Collision mesh for the object, separate from the drawn mesh. Defaults to the inscribed 160-face hull
# next to the visual mesh when one exists; set APPLE_OBJECT_COLLISION="" to collide the raw mesh.
def _collision_mesh_path(mesh: str | None = None) -> str:
  mesh = mesh if mesh is None else mesh
  if not mesh:
    return ""
  override = os.environ.get("APPLE_OBJECT_COLLISION")
  raw = Path(mesh)
  base = raw.name.split("_col")[0].split("_ins")[0].replace(".stl", "")
  if override is not None:
    if override == "":
      return mesh
    cand = raw.with_name(f"{base}_{override}.stl")
  else:
    cand = raw.with_name(f"{base}_cir160.stl")
  if not cand.exists():
    raise FileNotFoundError(f"object collision mesh {cand} is missing; build it with "
                            f"build_inscribed_hull.py")
  return str(cand)


_OBJECT_COLLISION_MESH = _collision_mesh_path()
# Joint-name pattern for the mounted hand. xhand joints are named right_hand_index_joint1; Wuji's are
# right_finger1_joint1, so a single ".*_hand_.*" cannot cover both.
_HAND_JOINT_EXPR = (".*_hand_.*",) if _HAND_KIND == "xhand" else (".*_finger[0-9]+_joint[0-9]+",)
# The Wuji model is built by tools/grasp_lab/build_g1_wuji.py, which grafts the Wuji gen-1 hand onto
# the same xhand mount frames; both hands share the palm-frame convention (verified: palm axes are
# identity in the mount frame for both), so no corrective rotation is needed.
_ROBOT_XML = (
  _GMR_ROOT / "assets" / "g1_xhand" / "g1_mocap_29dof_with_hands.xml"
  if _HAND_KIND == "xhand"
  else _GMR_ROOT / "assets" / "g1_wuji" / "g1_mocap_29dof_with_wuji_hands.xml"
)
TABLE_BASE_XY_SIZE = 0.30
DEFAULT_TABLE_XY_SCALE = 0.7


def _robot_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(_ROBOT_XML))
  _rename_xml_actuators_for_builtin_position_control(spec)
  _apply_official_joint_dynamics(spec)
  return spec


def _astra_robot_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(_ROBOT_XML))
  _rename_xml_actuators_for_builtin_position_control(spec)
  _apply_astra_body_dynamics(spec)
  return spec


def _rename_xml_actuators_for_builtin_position_control(spec: mujoco.MjSpec) -> None:
  for actuator in spec.actuators:
    if not str(actuator.name).startswith("xml_motor_unused_"):
      actuator.name = f"xml_motor_unused_{actuator.name}"


def _apply_official_joint_dynamics(spec: mujoco.MjSpec) -> None:
  body_armature = {
    **{
      name: mdp.apple_mdp._ARMATURE_7520_22
      for name in (
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_knee_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_knee_joint",
      )
    },
    **{
      name: mdp.apple_mdp._ARMATURE_7520_14
      for name in (
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "waist_yaw_joint",
      )
    },
    **{
      name: 2.0 * mdp.apple_mdp._ARMATURE_5020
      for name in (
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
      )
    },
    **{
      name: mdp.apple_mdp._ARMATURE_5020
      for name in (
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
      )
    },
    **{
      name: mdp.apple_mdp._ARMATURE_4010
      for name in (
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
      )
    },
  }
  body_and_hand = set(mdp.BODY_29_DOF_NAMES) | set(mdp.HAND_24_DOF_NAMES)
  for joint in spec.joints:
    if joint.name in body_armature:
      joint.armature = float(body_armature[joint.name])
    if joint.name in body_and_hand:
      joint.frictionloss = 0.0


def _apply_astra_body_dynamics(spec: mujoco.MjSpec) -> None:
  body_armature = {
    **{
      name: 0.01017752004
      for name in (
        "left_hip_pitch_joint",
        "left_hip_yaw_joint",
        "right_hip_pitch_joint",
        "right_hip_yaw_joint",
        "waist_yaw_joint",
      )
    },
    **{
      name: 0.025101925
      for name in (
        "left_hip_roll_joint",
        "left_knee_joint",
        "right_hip_roll_joint",
        "right_knee_joint",
      )
    },
    **{
      name: 0.00721945
      for name in (
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
      )
    },
    **{
      name: 0.003609725
      for name in (
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
      )
    },
    **{
      name: 0.00425
      for name in (
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
      )
    },
  }
  body_and_hand = set(mdp.BODY_29_DOF_NAMES) | set(mdp.HAND_24_DOF_NAMES)
  for joint in spec.joints:
    if joint.name in body_armature:
      joint.armature = float(body_armature[joint.name])
    if joint.name in mdp.BODY_29_DOF_NAMES:
      joint.frictionloss = 0.1
    elif joint.name in body_and_hand:
      joint.frictionloss = 0.0


def object_variant_half_extents() -> list:
  """Box half-extents for each clip's object, in clip order."""
  import trimesh

  out = []
  for mesh in mdp.apple_mdp.mix_object_stls():
    if not mesh:
      out.append(None)
      continue
    coll = _collision_mesh_path(mesh) or mesh
    b = trimesh.load(coll, process=False).bounds
    out.append(tuple(float(x) for x in (b[1] - b[0]) * 0.5))
  return out


def install_object_variant_sizes(env) -> None:
  """Give every world the box collider of the clip it is actually tracking.

  No-op unless APPLE_EAT_PKL_MIX names several clips and the collider is a box primitive. Must be
  called after the simulation is built, alongside install_astra_body_pd.
  """
  import mujoco as _mj
  import torch as _torch

  if _OBJECT_PRIMITIVE != "box" or mdp.apple_mdp.object_entity_count() <= 1:
    return
  halves = object_variant_half_extents()
  if any(h is None for h in halves):
    raise RuntimeError(
      "APPLE_OBJECT_PRIMITIVE=box with a mix, but a clip records no object mesh, so its box size "
      "cannot be derived. Give every clip a mesh or drop the box primitive."
    )
  u = env.unwrapped if hasattr(env, "unwrapped") else env
  m = u.sim.mj_model
  gid = next(
    g for g in range(m.ngeom)
    if (_mj.mj_id2name(m, _mj.mjtObj.mjOBJ_GEOM, g) or "").endswith("apple_geom")
  )
  clip = mdp.apple_mdp._clip_id(u)
  dev = clip.device
  half_t = _torch.tensor(halves, dtype=_torch.float32, device=dev)   # (n_clips, 3)
  per_world = half_t[clip]                                          # (nworld, 3)

  for field, value in (
    ("geom_size", per_world),
    ("geom_rbound", per_world.norm(dim=-1)),
  ):
    arr = getattr(u.sim.model, field, None)
    if arr is None:
      raise RuntimeError(f"model has no {field}; cannot size per-world colliders")
    t = arr._tensor if hasattr(arr, "_tensor") else arr
    if t.shape[0] != u.num_envs:
      raise RuntimeError(
        f"{field} has leading dim {t.shape[0]} for {u.num_envs} worlds; it was not expanded per "
        f"world, so a per-world write would apply to all of them"
      )
    if t.stride(0) == 0:
      raise RuntimeError(f"{field} is a zero-stride expand; per-world writes would alias")
    t[:, gid] = value if value.dim() > 1 else value

  aabb = getattr(u.sim.model, "geom_aabb", None)
  if aabb is not None:
    t = aabb._tensor if hasattr(aabb, "_tensor") else aabb
    if t.stride(0) != 0 and t.shape[0] == u.num_envs:
      t[:, gid, 0] = 0.0          # a box primitive is centred on its geom frame
      t[:, gid, 1] = per_world

  names = [n if n else "sphere" for n in
           (Path(x).name if x else "" for x in mdp.apple_mdp.mix_object_stls())]
  print(f"[residual_interact] per-world box colliders installed: "
        + ", ".join(f"{names[c]}={tuple(round(h, 4) for h in halves[c])}"
                    for c in range(len(halves))), flush=True)


def _object_entities() -> dict:
  """One scene entity per object, keyed by object_pool.entity_names().

  This replaces VariantEntityCfg for the mixed case. A merged variant body shares every geom index
  between objects and only scatters SOME model fields per world -- and its scatter loop skips
  non-mesh primitives entirely, which put a 4 cm sphere inside the stapler in every stapler world.
  Separate entities have separate geoms, so nothing has to mean two things at once, and the pose is
  qpos: state, per-world by construction.

  Single clip returns exactly one entity under the historical name, so nothing changes for runs that
  do not set APPLE_EAT_PKL_MIX.
  """
  n_clips = mdp.apple_mdp.object_entity_count()
  names = object_pool.entity_names(max(n_clips, 1))
  if n_clips <= 1:
    return {names[0]: EntityCfg(spec_fn=_apple_spec)}

  meshes = mdp.apple_mdp.mix_object_stls()
  if _OBJECT_PERCLIP and len(_OBJECT_PERCLIP) != n_clips:
    raise ValueError(
      f"APPLE_OBJECT_PERCLIP has {len(_OBJECT_PERCLIP)} entries for {n_clips} clips in "
      f"APPLE_EAT_PKL_MIX"
    )
  kinds = _OBJECT_PERCLIP or (None,) * n_clips
  out = {}
  for i, (name, m, k) in enumerate(zip(names, meshes, kinds)):
    out[name] = EntityCfg(spec_fn=functools.partial(_apple_spec, object_mesh=m, primitive=k))
  print("[residual_interact] object entities: "
        + ", ".join(f"{n}={Path(m).name if m else 'sphere'}[{k or _OBJECT_PRIMITIVE or 'hull'}]"
                    for n, m, k in zip(names, meshes, kinds)), flush=True)
  return out


def _object_sensors(sensor_cls, match_cls) -> tuple:
  """The hand/object contact sensors, one pair per object entity."""
  names = object_pool.entity_names(max(mdp.apple_mdp.object_entity_count(), 1))
  out = []
  for i, ent in enumerate(names):
    suffix = "" if i == 0 else f"_c{i}"
    out.append(sensor_cls(
      name=f"hand_apple_contact{suffix}",
      primary=match_cls(mode="body", pattern=mdp._TIP_CONTACT_BODY_NAMES, entity="robot"),
      secondary=match_cls(mode="body", pattern="apple", entity=ent),
      fields=("found", "force", "dist", "normal", "tangent"),
      reduce="maxforce",
      num_slots=1,
      global_frame=True,
      track_air_time=True,
      history_length=4,
    ))
    out.append(sensor_cls(
      name=f"hand_body_apple_contact{suffix}",
      primary=match_cls(mode="body", pattern=mdp._HAND_BODY_EXPR, entity="robot"),
      secondary=match_cls(mode="body", pattern="apple", entity=ent),
      fields=("found", "force"),
      reduce="maxforce",
      num_slots=1,
      track_air_time=False,
      history_length=4,
    ))
    # Omnigrasp reads table contact directly (humanoid_omnigrasp.py:1152) to decide object_lifted:
    # "has contact AND is not touching the table". Without this sensor that clause has to be
    # proxied by height above rest, which is a different statement.
    out.append(sensor_cls(
      name=f"object_table_contact{suffix}",
      primary=match_cls(mode="body", pattern="apple", entity=ent),
      secondary=match_cls(mode="body", pattern="table", entity="table"),
      fields=("found", "force"),
      reduce="maxforce",
      num_slots=1,
      track_air_time=False,
      history_length=4,
    ))
  return tuple(out)


def _object_entity_cfg_UNUSED():
  """One entity, or one variant per clip when APPLE_EAT_PKL_MIX names several."""
  n_clips = mdp.apple_mdp.object_entity_count()
  if n_clips <= 1:
    return EntityCfg(spec_fn=_apple_spec)
  from mjlab.entity import VariantEntityCfg

  meshes = mdp.apple_mdp.mix_object_stls()
  if _OBJECT_PERCLIP and len(_OBJECT_PERCLIP) != n_clips:
    raise ValueError(
      f"APPLE_OBJECT_PERCLIP has {len(_OBJECT_PERCLIP)} entries for {n_clips} clips in "
      f"APPLE_EAT_PKL_MIX"
    )
  kinds = _OBJECT_PERCLIP or (None,) * n_clips
  variants = {
    f"clip{i}": functools.partial(_apple_spec, object_mesh=m, primitive=k)
    for i, (m, k) in enumerate(zip(meshes, kinds))
  }
  # must match apple_eat.mdp._clip_id, which is arange(num_envs) % n_clips
  def assign(num_envs: int) -> list[int]:
    return [i % n_clips for i in range(num_envs)]

  print(f"[residual_interact] object variants: "
        + ", ".join(f"clip{i}={Path(m).name if m else 'sphere'}"
                    f"[{k or _OBJECT_PRIMITIVE or 'hull'}]"
                    for i, (m, k) in enumerate(zip(meshes, kinds))), flush=True)
  return VariantEntityCfg(variants=variants, assignment=assign,
                          collisions=_variant_slot_collision_cfgs(kinds))


# Padded mesh slots are SYNTHESIZED by build_merged_variant_spec, not copied: it sets type, name,
# contype/conaffinity, a placeholder meshname, mass=0 and density=0, and leaves everything else at
# MjSpec defaults. So the per-variant `friction`, `solimp`, `solref` and `priority` set in
# _apple_spec are DISCARDED for every slot the template variant does not itself fill, and the
# stapler's parts would collide with default softness -- docs/09 measured a mesh collider at default
# solimp sinking 15.87 mm into the table.
#
# Reordering the clips does not fix this. It only moves which geom is wrong: the template takes
# variant 0's values, so with the stapler first the parts are right and the shared `apple_geom`
# primitive gets stiffened instead -- and that geom is the apple's validated 4 cm collider in the
# apple's worlds. `geom_priority` and `geom_solref` are per-geom but NOT per-world, so one geom
# cannot hold both objects' contact settings.
#
# CollisionCfg is the supported way out: Entity applies it AFTER the merge, and it selects geoms by
# name, so the slots can be stiffened while `apple_geom` keeps MuJoCo's defaults.
def _variant_slot_collision_cfgs(kinds) -> tuple:
  if not any(k == "parts" for k in kinds):
    return ()
  from mjlab.utils.spec_config import CollisionCfg

  print(f"[residual_interact] restoring contact params on merged mesh slots: "
        f"priority={_OBJECT_PRIORITY} solref={tuple(_OBJECT_SOLREF)}", flush=True)
  return (
    CollisionCfg(
      # Both the renamed and the synthesized slots carry this reserved name, whatever the clip order.
      geom_names_expr=(r"mjlab/pad/.*/collision/.*",),
      # CollisionCfg.disable_other_geoms defaults to TRUE: it switches contype/conaffinity to 0 on
      # every geom the expression does not match. That silently took the apple sphere out of contact
      # entirely -- ngeom was unchanged and the only visible symptom would have been an apple falling
      # through the table. This config exists to restore params on the slots, not to re-scope which
      # geoms collide.
      disable_other_geoms=False,
      contype=1,
      conaffinity=1,
      condim=3,
      priority=_OBJECT_PRIORITY,
      friction=(1.0, 0.005, 0.0001),
      solref=tuple(_OBJECT_SOLREF),
      solimp=tuple(_OBJECT_SOLIMP),
    ),
  )


def _apple_spec(object_mesh: str | None = None,
                primitive: str | None = None) -> mujoco.MjSpec:
  """Build the object entity.

  `object_mesh` overrides the clip-0 mesh and `primitive` overrides APPLE_OBJECT_PRIMITIVE, both for
  per-world variants. Every branch emits exactly one non-mesh primitive geom named `apple_geom`,
  because variants.py requires primitives to match in count, name, type and order across variants --
  only mesh geom counts may differ.
  """
  _mesh = _OBJECT_MESH if object_mesh is None else object_mesh
  _coll = _collision_mesh_path(_mesh) if object_mesh is not None else _OBJECT_COLLISION_MESH
  _prim = _OBJECT_PRIMITIVE if primitive is None else primitive
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="apple", pos=(0.55, 0.0, 0.84))
  joint = body.add_freejoint(name="apple_joint")
  joint.damping[:] = (
    mdp.APPLE_LINEAR_DAMPING,
    mdp.APPLE_LINEAR_DAMPING,
    mdp.APPLE_ANGULAR_DAMPING,
  )
  if _mesh and _prim == "boxstack":
    coll = _coll or _mesh
    cells = _boxstack_cells(coll, *_OBJECT_BOXSTACK)
    vols = [8.0 * h[0] * h[1] * h[2] for _, h in cells]
    total = sum(vols)
    print(f"[object] boxstack collider, {len(cells)} boxes "
          f"({_OBJECT_BOXSTACK[0]}x{_OBJECT_BOXSTACK[1]} cells requested)", flush=True)
    spec.add_mesh(name="object_visual_mesh", file=_mesh)
    for k, ((cx, cy, cz), half) in enumerate(cells):
      print(f"[object]   box{k}: centre ({cx:+.4f},{cy:+.4f},{cz:+.4f}) "
            f"half ({half[0]:.4f},{half[1]:.4f},{half[2]:.4f}) m", flush=True)
      gk = body.add_geom(
        name="apple_geom" if k == 0 else f"apple_geom_{k}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=half,
        pos=(cx, cy, cz),
        # Mass by volume so the composite inertia is sensible; MuJoCo sums the geoms.
        mass=mdp.APPLE_MASS * vols[k] / total,
        friction=(1.0, 0.005, 0.0001),
        rgba=(0.15, 0.55, 0.95, 0.45),
      )
      gk.group = _OBJECT_BOX_GROUP
    _vis = body.add_geom(
      name="apple_visual",
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname="object_visual_mesh",
      mass=0.0,
      contype=0,
      conaffinity=0,
      rgba=(0.8, 0.05, 0.03, 1.0),
    )
    _vis.group = _OBJECT_VISUAL_GROUP
  elif _mesh and _prim == "box":
    # Half-extents from the collision mesh's bounding box, so the box matches what maxhullvert=8
    # would have produced -- but as a primitive, which uses the analytic box-box collider.
    import trimesh

    _cm = trimesh.load(_coll or _mesh, process=False)
    _half = (_cm.bounds[1] - _cm.bounds[0]) * 0.5
    print(f"[object] box primitive collider, half-extents {tuple(round(float(h), 4) for h in _half)} m")
    spec.add_mesh(name="object_visual_mesh", file=_mesh)
    body.add_geom(
      name="apple_geom",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=tuple(float(h) for h in _half),
      mass=mdp.APPLE_MASS,
      friction=(1.0, 0.005, 0.0001),
      # Alpha is non-zero so the collider is actually visible when its group is enabled; group 3
      # is hidden by default, so the scene looks unchanged until you ask for it.
      rgba=(0.15, 0.55, 0.95, 0.45),
    )
    body.geoms[-1].group = 5
    _vis = body.add_geom(
      name="apple_visual",
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname="object_visual_mesh",
      mass=0.0,
      contype=0,
      conaffinity=0,
      rgba=(0.8, 0.05, 0.03, 1.0),
    )
    _vis.group = _OBJECT_VISUAL_GROUP
  elif _mesh and _prim == "parts":
    # One mesh geom per pre-baked convex part. Mesh geom COUNT is the only thing variants.py lets
    # differ between variants, so this is the representation that lets one world hold a stapler and
    # another an apple. Measured on the GRAB stapler: 5 parts spill 10.3 % outside the object where
    # a single hull spills 21.8 % and the boxstack 34.8 %.
    # The RAW mesh, never _coll: the collision mesh is already a convex hull, and decomposing a
    # hull recovers nothing -- there is no concavity left in it to split on.
    files, inside = _object_part_files(_mesh)
    print(f"[object] parts collider, {len(files)} convex parts from "
          f"{Path(_mesh).stem}_parts.json", flush=True)
    spec.add_mesh(name="object_visual_mesh", file=_mesh)
    # The placeholder primitive. It exists only so this variant's primitive signature matches a
    # variant whose collider IS a sphere; buried at the largest part's centroid it touches nothing.
    # It carries no mass -- body_mass is variant-dependent, so the parts can hold all of it.
    _g0 = body.add_geom(
      name="apple_geom",
      type=mujoco.mjtGeom.mjGEOM_SPHERE,
      size=(_PARTS_PLACEHOLDER_RADIUS,),
      pos=inside,
      mass=0.0,
      friction=(1.0, 0.005, 0.0001),
      rgba=(0.95, 0.25, 0.25, 0.25),
    )
    _g0.group = 5
    for k, f in enumerate(files):
      spec.add_mesh(name=f"object_part{k}", file=f)
      gk = body.add_geom(
        name=f"apple_geom_part{k}",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=f"object_part{k}",
        # Split evenly: the parts overlap, so a volume-weighted split would double-count the
        # overlaps. MuJoCo sums geom masses into the body, and the total is what matters.
        mass=mdp.APPLE_MASS / len(files),
        friction=(1.0, 0.005, 0.0001),
        rgba=(0.15, 0.55, 0.95, 0.45),
      )
      gk.group = _OBJECT_BOX_GROUP
    _vis = body.add_geom(
      name="apple_visual",
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname="object_visual_mesh",
      mass=0.0,
      contype=0,
      conaffinity=0,
      rgba=(0.8, 0.05, 0.03, 1.0),
    )
    _vis.group = _OBJECT_VISUAL_GROUP
  elif _mesh and _prim == "sphere":
    # Force the analytic sphere even though a mesh is available. This is what keeps apple_eat_1 on
    # the collider it was validated with (lift_success 0.726) inside a mix that enables meshes for
    # the other clips.
    spec.add_mesh(name="object_visual_mesh", file=_mesh)
    _g0 = body.add_geom(
      name="apple_geom",
      type=mujoco.mjtGeom.mjGEOM_SPHERE,
      size=(mdp.APPLE_RADIUS,),
      mass=mdp.APPLE_MASS,
      friction=(1.0, 0.005, 0.0001),
      rgba=(0.15, 0.55, 0.95, 0.45),
    )
    _g0.group = 5
    _vis = body.add_geom(
      name="apple_visual",
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname="object_visual_mesh",
      mass=0.0,
      contype=0,
      conaffinity=0,
      rgba=(0.8, 0.05, 0.03, 1.0),
    )
    _vis.group = _OBJECT_VISUAL_GROUP
  elif _mesh:
    # Collision: a coarse inscribed hull, because mjwarp contact quality degrades with face count.
    # Visual: the real mesh, contype/conaffinity 0 so it never participates in contact.
    coll = _coll or _mesh
    obj_mesh = spec.add_mesh(name="object_mesh", file=coll)
    obj_mesh.maxhullvert = _OBJECT_MAXHULLVERT
    if _coll and _coll != _mesh:
      spec.add_mesh(name="object_visual_mesh", file=_mesh)
    body.add_geom(
      name="apple_geom",
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname="object_mesh",
      mass=mdp.APPLE_MASS,
      friction=(1.0, 0.005, 0.0001),
      rgba=(0.8, 0.05, 0.03, 1.0),
    )
  else:
    body.add_geom(
      name="apple_geom",
      type=mujoco.mjtGeom.mjGEOM_SPHERE,
      size=(mdp.APPLE_RADIUS,),
      mass=mdp.APPLE_MASS,
      friction=(1.0, 0.005, 0.0001),
      rgba=(0.8, 0.05, 0.03, 1.0),
    )
  # Every collider, not just the first: boxstack adds several, and stiffened contact on one of them
  # while the rest keep defaults would make the object behave differently depending on which box the
  # table happens to touch.
  _colliders = [x for x in body.geoms
                if str(x.name).startswith("apple_geom") and int(x.contype) != 0]
  assert _colliders, "contact params would land on a non-colliding geom"
  # An explicitly forced `sphere` keeps MuJoCo's defaults whatever APPLE_OBJECT_SOLIMP/SOLREF/
  # PRIORITY say. Its entire purpose is to be the collider the apple was validated with, and in that
  # run those variables were unset, so the loop below wrote the defaults back as a no-op. In a mix
  # they are set for the other object's parts, and without this gate they would silently stiffen the
  # apple too -- the same shape of accident as the five APPLE_OBJECT_* variables that broke a
  # replication. `_prim == ""` is untouched: that is the old single-object path.
  if _prim == "sphere":
    print("[object] forced sphere: contact params left at MuJoCo defaults", flush=True)
    return spec
  for g in _colliders:
    g.solref[: len(_OBJECT_SOLREF)] = list(_OBJECT_SOLREF)
    g.solimp[: len(_OBJECT_SOLIMP)] = list(_OBJECT_SOLIMP)
    g.priority = _OBJECT_PRIORITY
  g = _colliders[0]
  if (_mesh and not _prim and _coll
      and _coll != _mesh):
    # The hull encloses the mesh, so leaving it opaque hides the apple entirely. Group 3 + alpha 0
    # keeps it purely a collider.
    g.group = 5
    g.rgba[:] = (0.15, 0.55, 0.95, 0.45)
    vis = body.add_geom(
      name="apple_visual",
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname="object_visual_mesh",
      mass=0.0,
      contype=0,
      conaffinity=0,
      rgba=(0.8, 0.05, 0.03, 1.0),
    )
    vis.group = _OBJECT_VISUAL_GROUP
  return spec


def _table_spec(table_xy_scale: float = DEFAULT_TABLE_XY_SCALE) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="table", pos=(0.0, 0.0, 0.0))
  xy = float(table_xy_scale) * TABLE_BASE_XY_SIZE
  body.add_geom(
    name="table_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(0.5 * xy, 0.5 * xy, 0.5 * mdp.TABLE_THICKNESS),
    friction=(1.0, 0.005, 0.0001),
    rgba=(0.55, 0.40, 0.25, 1.0),
  )
  return spec


def set_table_xy_scale(cfg: ManagerBasedRlEnvCfg, table_xy_scale: float) -> None:
  scale = float(table_xy_scale)

  def _scaled_table_spec() -> mujoco.MjSpec:
    return _table_spec(table_xy_scale=scale)

  cfg.scene.entities["table"].spec_fn = _scaled_table_spec


def set_astra_body_dynamics(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.scene.entities["robot"].spec_fn = _astra_robot_spec
  cfg.scene.entities["robot"].articulation.actuators = _astra_body_actuator_cfgs()


def install_astra_body_pd(env) -> None:
  """Synchronize runtime action-term and simulator PD buffers with ASTRA release."""
  term = env.action_manager.get_term("sonic_action")
  kp = torch.as_tensor(
    mdp.ASTRA_KP_BODY_PKL, device=term._kp.device, dtype=term._kp.dtype
  )
  kd = torch.as_tensor(
    mdp.ASTRA_KD_BODY_PKL, device=term._kd.device, dtype=term._kd.dtype
  )
  effort = torch.as_tensor(
    mdp.ASTRA_EFFORT_BODY_PKL,
    device=term._effort_limit.device,
    dtype=term._effort_limit.dtype,
  )
  term._kp[: mdp.NUM_BODY] = kp
  term._kd[: mdp.NUM_BODY] = kd
  term._effort_limit[: mdp.NUM_BODY] = effort

  mj_model = env.sim.mj_model
  actuator_ids: list[int] = []
  for name in mdp.BODY_29_DOF_NAMES:
    for candidate in (name, f"robot/{name}"):
      try:
        actuator_ids.append(int(mj_model.actuator(candidate).id))
        break
      except KeyError:
        continue
    else:
      raise RuntimeError(f"Missing ASTRA body actuator {name!r}.")

  ids_cpu = torch.tensor(actuator_ids, dtype=torch.long)
  kp_cpu = kp.detach().cpu()
  kd_cpu = kd.detach().cpu()
  effort_cpu = effort.detach().cpu()
  mj_model.actuator_gainprm[ids_cpu.numpy(), 0] = kp_cpu.numpy()
  mj_model.actuator_biasprm[ids_cpu.numpy(), 1] = -kp_cpu.numpy()
  mj_model.actuator_biasprm[ids_cpu.numpy(), 2] = -kd_cpu.numpy()
  mj_model.actuator_forcerange[ids_cpu.numpy(), 0] = -effort_cpu.numpy()
  mj_model.actuator_forcerange[ids_cpu.numpy(), 1] = effort_cpu.numpy()

  model = env.sim.model
  ids = ids_cpu.to(env.device)
  if len(model.actuator_gainprm.shape) == 3:
    model.actuator_gainprm[:, ids, 0] = kp.view(1, -1)
    model.actuator_biasprm[:, ids, 1] = -kp.view(1, -1)
    model.actuator_biasprm[:, ids, 2] = -kd.view(1, -1)
    model.actuator_forcerange[:, ids, 0] = -effort.view(1, -1)
    model.actuator_forcerange[:, ids, 1] = effort.view(1, -1)
  else:
    model.actuator_gainprm[ids, 0] = kp
    model.actuator_biasprm[ids, 1] = -kp
    model.actuator_biasprm[ids, 2] = -kd
    model.actuator_forcerange[ids, 0] = -effort
    model.actuator_forcerange[ids, 1] = effort


def _astra_body_actuator_cfgs() -> tuple[BuiltinPositionActuatorCfg, ...]:
  def body_group(
    target_names_expr: tuple[str, ...],
    idx: int,
    armature: float,
  ) -> BuiltinPositionActuatorCfg:
    return BuiltinPositionActuatorCfg(
      target_names_expr=target_names_expr,
      stiffness=float(mdp.ASTRA_KP_BODY_PKL[idx]),
      damping=float(mdp.ASTRA_KD_BODY_PKL[idx]),
      effort_limit=float(mdp.ASTRA_EFFORT_BODY_PKL[idx]),
      armature=float(armature),
      frictionloss=0.1,
    )

  return (
    body_group(
      (
        ".*_hip_pitch_joint",
        ".*_hip_yaw_joint",
        "waist_yaw_joint",
      ),
      0,
      0.01017752004,
    ),
    body_group(
      (
        ".*_hip_roll_joint",
        ".*_knee_joint",
      ),
      1,
      0.025101925,
    ),
    body_group(
      (
        ".*_ankle_pitch_joint",
        ".*_ankle_roll_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
      ),
      4,
      0.00721945,
    ),
    body_group(
      (
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_elbow_joint",
        ".*_wrist_roll_joint",
      ),
      15,
      0.003609725,
    ),
    body_group((".*_wrist_pitch_joint", ".*_wrist_yaw_joint"), 20, 0.00425),
    BuiltinPositionActuatorCfg(
      target_names_expr=_HAND_JOINT_EXPR,
      stiffness=300.0,
      damping=8.0,
      effort_limit=30.0,
      frictionloss=0.0,
    ),
  )


# Reference-preview lookahead, in control steps at 50 Hz. The CLI override for the
# observation term's params dict does not take effect (it parses, but the term keeps its own
# default), so this is the knob that actually works. Default is unchanged so resumed runs keep
# their input dimension; set APPLE_REF_PREVIEW_STEPS="1,5,10,20,40,80" to widen it.
_DEFAULT_REF_PREVIEW_STEPS: tuple[int, ...] = tuple(
  int(v) for v in os.environ.get("APPLE_REF_PREVIEW_STEPS", "1,5,10").split(",") if v.strip()
)


def _feature_group_cfg(
  group: str, ref_preview_steps: tuple[int, ...]
) -> ObservationGroupCfg:
  if group == "sonic_obs_or_latent":
    return ObservationGroupCfg(
      {"policy": ObservationTermCfg(func=mdp.IsaacStyleObs2964)},
      enable_corruption=False,
      nan_policy="sanitize",
    )
  if group == "sonic_encoder_obs":
    return ObservationGroupCfg(
      {
        "policy": ObservationTermCfg(
          func=mdp.SonicEncoderObs1762,
          params={"root_anchor_mode": "identity"},
        )
      },
      enable_corruption=False,
      nan_policy="sanitize",
    )
  if group == "astra_obs":
    return ObservationGroupCfg(
      {"policy": ObservationTermCfg(func=mdp.AstraObs136)},
      enable_corruption=False,
      nan_policy="sanitize",
    )
  return ObservationGroupCfg(
    {
      "policy": ObservationTermCfg(
        func=mdp.ResidualFeatureGroupObs,
        params={"group": group, "ref_preview_steps": ref_preview_steps},
      )
    },
    enable_corruption=False,
    nan_policy="sanitize",
  )


def residual_interact_env_cfg(
  play: bool = False,
  ref_preview_steps: tuple[int, ...] = _DEFAULT_REF_PREVIEW_STEPS,
) -> ManagerBasedRlEnvCfg:
  observations = {
    group: _feature_group_cfg(group, ref_preview_steps)
    for group in mdp.OBSERVATION_GROUPS
  }

  actions: dict[str, ActionTermCfg] = {
    "sonic_action": mdp.Sonic53ActionCfg(
      entity_name="robot",
      body_action_scale=1.0,
      hand_action_scale=1.0,
      tracking_start_assist_steps=120,
      tracking_start_assist_follow_ref_xy=False,
    )
  }

  rewards = {
    "tracking": RewardTermCfg(
      func=mdp.residual_tracking_reward,
      weight=1.0,
      params={
        "distance_std": 0.03,
        "link_group": "all",
        "use_tracking_weights": True,
        "per_link_reward": False,
        "log_per_link": False,
        "log_prefix": "tracking",
        "close_bonus_weight": 0.0,
        "close_bonus_threshold": 0.03,
        "close_bonus_margin": 0.01,
        "miss_penalty_weight": 0.0,
        "miss_penalty_margin": 0.01,
        "post_frame": -1,
        "post_frame_scale": 1.0,
      },
    ),
    "left_wrist_tracking": RewardTermCfg(
      func=mdp.residual_tracking_reward,
      weight=0.0,
      params={
        "distance_std": 0.03,
        "link_group": "left_wrist",
        "use_tracking_weights": True,
        "per_link_reward": False,
        "log_per_link": False,
        "log_prefix": "left_wrist_tracking",
        "close_bonus_weight": 0.0,
        "close_bonus_threshold": 0.03,
        "close_bonus_margin": 0.01,
        "miss_penalty_weight": 0.0,
        "miss_penalty_margin": 0.01,
        "post_frame": -1,
        "post_frame_scale": 1.0,
      },
    ),
    "right_wrist_tracking": RewardTermCfg(
      func=mdp.residual_tracking_reward,
      weight=0.0,
      params={
        "distance_std": 0.03,
        "link_group": "right_wrist",
        "use_tracking_weights": True,
        "per_link_reward": False,
        "log_per_link": False,
        "log_prefix": "right_wrist_tracking",
        "close_bonus_weight": 0.0,
        "close_bonus_threshold": 0.03,
        "close_bonus_margin": 0.01,
        "miss_penalty_weight": 0.0,
        "miss_penalty_margin": 0.01,
        "post_frame": -1,
        "post_frame_scale": 1.0,
      },
    ),
    "left_wrist_z_tracking": RewardTermCfg(
      func=mdp.residual_axis_tracking_reward,
      weight=0.0,
      params={
        "axis": "z",
        "axis_std": 0.015,
        "link_group": "left_wrist",
        "use_tracking_weights": True,
        "log_prefix": "left_wrist_z_tracking",
      },
    ),
    "right_wrist_z_tracking": RewardTermCfg(
      func=mdp.residual_axis_tracking_reward,
      weight=0.0,
      params={
        "axis": "z",
        "axis_std": 0.015,
        "link_group": "right_wrist",
        "use_tracking_weights": True,
        "log_prefix": "right_wrist_z_tracking",
      },
    ),
    "hand_action_tracking": RewardTermCfg(
      func=mdp.residual_hand_action_tracking_reward,
      weight=0.0,
      params={
        "gate_mode": "near_contact",
        "near_threshold": 0.15,
      },
    ),
    "raw_contact_tracking": RewardTermCfg(
      func=mdp.residual_raw_contact_tracking_reward,
      weight=0.0,
      params={
        "k": 8.0,
        "arm_weight": 1.0,
        "hand_weight": 1.5,
        "object_weight": 0.0,
        "object_pos_std": 0.06,
        "start_frame": 380,
        "end_frame": 440,
        "margin_frames": 20,
      },
    ),
    "raw_tip_object_tracking": RewardTermCfg(
      func=mdp.residual_raw_tip_object_tracking_reward,
      weight=0.0,
      params={
        "std": 0.05,
        "top_k": 2,
        "start_frame": 380,
        "end_frame": 440,
        "margin_frames": 20,
      },
    ),
    "raw_tip_radial_tracking": RewardTermCfg(
      func=mdp.residual_raw_tip_radial_tracking_reward,
      weight=0.0,
      params={
        "std": 0.08,
        "reach_std": 0.20,
        "reach_weight": 0.5,
        "top_k": 1,
        "start_frame": 380,
        "end_frame": 440,
        "margin_frames": 20,
      },
    ),
    # --- staged on the dataset's own contact frame (staged_mdp.py) --------------------------
    # Before cf the re-solved reference is worth tracking: the hand arrives at the object the way
    # the human's did. After cf the task is where the OBJECT goes and the arm configuration is one
    # of many ways to hold a thing, so tracking it fights the object term. All default to weight 0.
    # --- the grasp the retarget now solves for, spent as reward -------------------------------
    # Step 5 solves the 20 finger joints against the human's own fingertips (SMPL-X mesh) and step 6
    # ramps them in over the 30 frames before cf. Tracking those joints is tracking the grasp.
    "staged_hand_pose": RewardTermCfg(
      func=staged_mdp.staged_hand_pose_tracking_reward,
      weight=0.0,
      params={
        "angle_std": 0.25,
        "pre_weight": 1.0,
        # small, not zero: after cf the hand must STAY closed. Letting this vanish invites the
        # policy to open the hand the moment the object term takes over.
        "post_weight": 0.2,
        "log_prefix": "staged_hand_pose",
      },
    ),
    # Omnigrasp's close_hand_flag: score only the tips the REFERENCE brings near the object, so the
    # fingers the human never used are not pushed into a cage around a pinched object.
    "staged_tip_object": RewardTermCfg(
      func=staged_mdp.staged_tip_object_reward,
      weight=0.0,
      params={
        "distance_std": 0.06,
        "near_threshold": 0.25,
        "pre_weight": 1.0,
        "post_weight": 0.3,
        "log_prefix": "staged_tip_object",
      },
    ),
    # --- R15: one approach target, and multi_tip_surface confined to the carry phase ---------
    # Omnigrasp's pregrasp term ported whole: the 16 hand bodies a side, position AND orientation,
    # against their pose at cf. Weight 0.0 here; a reward yaml turns it on INSTEAD of staged_tip_cf,
    # which does the same job with five points and no orientation -- running both would pay the
    # approach twice and change the total reward scale.
    "staged_hand_cf": RewardTermCfg(
      func=staged_mdp.staged_hand_cf_reward,
      weight=0.0,
      params={
        "k_pos": 100.0,
        "k_rot": 10.0,
        "w_pos": 0.9,
        "w_rot": 0.1,
        "close_distance": 0.20,
        "progress_cap": 0.10,
        "near_threshold": 0.20,
        "pre_weight": 1.0,
        "post_weight": 0.0,
        "log_prefix": "staged_hand_cf",
      },
    ),
    "staged_tip_cf": RewardTermCfg(
      func=staged_mdp.staged_tip_cf_reward,
      weight=0.0,
      params={
        # 0.50, raised from 0.20, and 0.20 was itself raised from 0.06 because at that width the
        # reward 0.52 m out was 6e-13 and a weight-5.0 term carried no gradient until the approach
        # was already solved by other means.
        #
        # 0.20 was still too narrow for what the approach actually is. Measured on the reference:
        # the object does NOT move before cf (0.0000-0.018 m) and neither does the table -- the
        # ROBOT walks 0.15 to 0.60 m to reach it. That whole walk happens beyond the 0.20 m switch,
        # so it was paid by the progress term alone, which is a pure derivative: standing still at
        # 1.2 m and standing still at 0.25 m both earn zero. There was no potential pulling the
        # robot across the distance it has to cover. Measured consequence: 81 % of airplane's
        # env-steps sat in that far field at 0.66 reward per step against 2.1-4.4 in the near
        # field, closing at 0.2 m/s when the clip needs 0.43 m/s.
        #
        # At 0.50 the level term has real slope over the whole walk -- 0.028 at 1.2 m, 0.139 at
        # 0.8, 0.243 at 0.6, 0.363 at 0.4, 0.462 at the switch -- while the endgame is untouched
        # (0.622 at 5 cm against 0.609 before, 0.973 at 1 cm against 0.972). The seam stays
        # continuous because the far field is scaled by level(switch), which moves with it.
        #
        # The cost, stated: hovering is worth more than it was. Standing at 0.6 m now pays ~1.1
        # per step where it paid 0. It is still strictly less than closing in, so the gradient
        # points the right way -- unlike before the seam fix, when closing was a 3.3x pay cut.
        "distance_std": 0.50,
        "near_threshold": 0.10,
        "pre_weight": 1.0,
        "post_weight": 0.0,
        "log_prefix": "staged_tip_cf",
      },
    ),
    "staged_multi_tip_surface": RewardTermCfg(
      func=staged_mdp.staged_multi_tip_surface_reward,
      weight=0.0,
      params={
        # identical to multi_tip_surface's own params -- only the phase mask is new
        "surface_dist": mdp.APPLE_RADIUS,
        "std": 0.025,
        "near_dist": 0.10,
        "top_k": 3,
        "multi_tip_count": 2,
        "multi_tip_weight": 0.5,
        "contact_weight": 1.0,
        "grasp_weight": 1.0,
        "force_weight": 0.5,
        "drift_target": 0.0,
        "drift_margin": 0.10,
        "drift_power": 1.0,
        "gate_shaping": True,
        "opposition_gate_floor": 1.0,
        "opposition_gate_hi": 1.0,
        "opposition_gate_lo": 0.0,
        "pre_weight": 0.0,
        "post_weight": 1.0,
        "log_prefix": "staged_mts",
      },
    ),
    "staged_upper_tracking": RewardTermCfg(
      func=staged_mdp.staged_link_tracking_reward,
      weight=0.0,
      params={
        "link_group": "upper",
        "distance_std": 0.05,
        "pre_weight": 1.0,
        # Not a hard zero. Omnigrasp can drop its body term to zero because its action space is a
        # pretrained motion latent that cannot produce a non-human pose; ours is joint residuals on
        # a frozen tracker, so nothing structurally stops the arm windmilling once this is released.
        "post_weight": 0.15,
        "log_prefix": "staged_upper",
      },
    ),
    "staged_lower_tracking": RewardTermCfg(
      func=staged_mdp.staged_link_tracking_reward,
      weight=0.0,
      params={
        "link_group": "lower",
        "distance_std": 0.05,
        # Tracked throughout: GMR's feet carry the highest IK weights, and the upper body needs
        # something to stand on once it is released.
        "pre_weight": 1.0,
        "post_weight": 1.0,
        "log_prefix": "staged_lower",
      },
    ),
    "staged_object_tracking": RewardTermCfg(
      func=staged_mdp.staged_object_tracking_reward,
      weight=0.0,
      params={
        "distance_std": 0.08,
        "pre_weight": 0.0,
        "post_weight": 1.0,
        # Omnigrasp's one load-bearing idea: without a contact gate the policy collects the object
        # reward for standing next to an object nobody moved.
        "require_contact": True,
        "contact_threshold": 0.02,
      },
    ),
    "object_trajectory_tracking": RewardTermCfg(
      func=omnigrasp_style_mdp.object_trajectory_tracking_reward,
      weight=0.0,
      params={
        # -1 = this clip's own contact frame. Was a hard-coded 65, which paid nothing
        # between the grasp and frame 65 on the (at least) half of clips whose cf is
        # earlier -- cf is 47 on the median clip and as early as 22.
        "start_frame": -1,
        "gate_mode": "soft_contact",
        "contact_threshold": 0.06,
        "soft_contact_std": 0.08,
        "pos_std": 0.06,
        "rot_std": 0.70,
        "vel_std": 0.40,
        "pos_weight": 0.55,
        "rot_weight": 0.25,
        "vel_weight": 0.20,
      },
    ),
    "task": RewardTermCfg(func=mdp.residual_task_reward, weight=0.0),
    "grasp": RewardTermCfg(
      func=mdp.residual_grasp_reward,
      weight=0.0,
      params={
        "contact_dist": 0.08,
        "close_std": 0.06,
        "lift_height": 0.04,
        "object_speed_threshold": 0.03,
        "close_weight": 0.25,
        "near_weight": 1.0,
        "contact_weight": 4.0,
        "force_weight": 2.0,
        "object_motion_weight": 1.0,
        "lift_weight": 2.0,
        "drift_target": 0.0,
        "drift_margin": 0.10,
        "drift_power": 1.0,
      },
    ),
    "surface_contact": RewardTermCfg(
      func=mdp.residual_surface_contact_reward,
      weight=0.0,
      params={
        "surface_dist": mdp.APPLE_RADIUS,
        "std": 0.02,
        "near_dist": 0.10,
        "contact_weight": 1.0,
        "force_weight": 0.5,
        "drift_target": 0.0,
        "drift_margin": 0.10,
        "drift_power": 1.0,
        "gate_shaping": True,
        # 1.0 disables the gate (previous behaviour).  Lower it to stop
        # paying for one fingertip poking from the wrong side.
        "opposition_gate_floor": 1.0,
        "opposition_gate_hi": 1.0,
        "opposition_gate_lo": 0.0,
      },
    ),
    "multi_tip_surface": RewardTermCfg(
      func=mdp.residual_multi_tip_surface_reward,
      weight=0.0,
      params={
        "surface_dist": mdp.APPLE_RADIUS,
        "std": 0.025,
        "near_dist": 0.10,
        "top_k": 3,
        "multi_tip_count": 2,
        "multi_tip_weight": 0.5,
        "contact_weight": 1.0,
        "grasp_weight": 1.0,
        "force_weight": 0.5,
        "drift_target": 0.0,
        "drift_margin": 0.10,
        "drift_power": 1.0,
        "gate_shaping": True,
        # 1.0 disables the gate (previous behaviour).  Lower it to stop
        # paying for one fingertip poking from the wrong side.
        "opposition_gate_floor": 1.0,
        "opposition_gate_hi": 1.0,
        "opposition_gate_lo": 0.0,
      },
    ),
    "object_drift_limit": RewardTermCfg(
      func=mdp.residual_object_drift_limit_reward,
      weight=0.0,
      params={"target": 0.25},
    ),
    "staged_contact_duration": RewardTermCfg(
      func=staged_mdp.staged_contact_duration_reward,
      weight=0.0,
      params={
        "target_duration": 0.08,
        "top_k": 2,
        "contact_weight": 0.5,
        "grasp_weight": 1.0,
        "drift_target": 0.0,
        "pre_weight": 0.0,
        "post_weight": 1.0,
      },
    ),
    "contact_duration": RewardTermCfg(
      func=mdp.residual_contact_duration_reward,
      weight=0.0,
      params={
        "target_duration": 0.08,
        "top_k": 2,
        "contact_weight": 0.5,
        "grasp_weight": 1.0,
        "drift_target": 0.0,
        "drift_margin": 0.10,
        "drift_power": 1.0,
      },
    ),
    "object_lift_hold": RewardTermCfg(
      func=omnigrasp_style_mdp.object_lift_hold_reward,
      weight=0.0,
      params={
        # -1 = this clip's own contact frame. Was a hard-coded 65, which paid nothing
        # between the grasp and frame 65 on the (at least) half of clips whose cf is
        # earlier -- cf is 47 on the median clip and as early as 22.
        "start_frame": -1,
        "lift_height": 0.04,
        "near_std": 0.08,
        "target_duration": 0.10,
        "top_k": 2,
        "lifted_weight": 1.0,
        "near_weight": 0.5,
        "contact_weight": 1.0,
        "grasp_weight": 1.0,
        "duration_weight": 1.0,
        "force_weight": 0.5,
        "drop_penalty_weight": 0.5,
        "drop_dist": 0.18,
      },
    ),
    "object_hard_lift": RewardTermCfg(
      func=omnigrasp_style_mdp.object_hard_lift_reward,
      weight=0.0,
      params={
        # -1 = this clip's own contact frame. Was a hard-coded 65, which paid nothing
        # between the grasp and frame 65 on the (at least) half of clips whose cf is
        # earlier -- cf is 47 on the median clip and as early as 22.
        "start_frame": -1,
        "lift_height": 0.03,
        "duration_target": 0.30,
        "min_contact_tips": 2,
        "height_weight": 1.0,
        "hard_lift_weight": 1.0,
        "duration_weight": 1.0,
        "force_weight": 0.25,
      },
    ),
    "object_omnigrasp_grab": RewardTermCfg(
      func=omnigrasp_style_mdp.object_omnigrasp_grab_reward,
      weight=0.0,
      params={
        # -1 = this clip's own contact frame. Was a hard-coded 65, which paid nothing
        # between the grasp and frame 65 on the (at least) half of clips whose cf is
        # earlier -- cf is 47 on the median clip and as early as 22.
        "start_frame": -1,
        "contact_threshold": 0.06,
        "lift_height": 0.03,
        "object_motion_speed": 0.02,
        "duration_target": 0.30,
        "min_contact_tips": 2,
        "pos_std": 0.04,
        "rot_std": 0.55,
        "vel_std": 0.35,
        "pos_weight": 0.65,
        "rot_weight": 0.20,
        "vel_weight": 0.15,
        "tracking_weight": 1.0,
        "contact_bonus_weight": 0.25,
        "lift_bonus_weight": 0.75,
        "duration_bonus_weight": 0.75,
        "force_bonus_weight": 0.25,
      },
    ),
    "placement": RewardTermCfg(func=mdp.residual_placement_reward, weight=0.0),
    "stability": RewardTermCfg(func=mdp.residual_stability_reward, weight=0.0),
    "residual_l2": RewardTermCfg(func=mdp.residual_l2_reward, weight=0.0),
    "residual_smooth": RewardTermCfg(func=mdp.residual_smooth_reward, weight=0.0),
    "token_l2": RewardTermCfg(func=mdp.residual_token_l2_reward, weight=0.0),
    "token_smooth": RewardTermCfg(func=mdp.residual_token_smooth_reward, weight=0.0),
    "decoder_body_delta_l2": RewardTermCfg(
      func=mdp.residual_decoder_body_delta_l2_reward,
      weight=0.0,
    ),
    "decoder_body_delta_norm_limit": RewardTermCfg(
      func=mdp.residual_decoder_body_delta_norm_limit_reward,
      weight=0.0,
      params={"target": 0.60},
    ),
    "decoder_body_delta_ratio_limit": RewardTermCfg(
      func=mdp.residual_decoder_body_delta_ratio_limit_reward,
      weight=0.0,
      params={"target": 0.08},
    ),
    "decoder_body_delta_joint_limit": RewardTermCfg(
      func=mdp.residual_decoder_body_delta_joint_limit_reward,
      weight=0.0,
      params={"target": 0.25},
    ),
    "hand_ratio_limit": RewardTermCfg(
      func=mdp.residual_hand_ratio_limit_reward,
      weight=0.0,
      params={"target": 1.30},
    ),
    "omnigrasp_style": RewardTermCfg(
      func=omnigrasp_style_mdp.omnigrasp_style_reward,
      weight=0.0,
      params={
        "contact_threshold": 0.03,
        "pregrasp_reference_distance": 0.12,
        "pregrasp_top_k": 4,
        "body_weight": 1.0,
        "body_xyz_weight": 12.0,
        "pregrasp_weight": 1.0,
        "pregrasp_tip_std": 0.08,
        "pregrasp_hand_q_std": 0.45,
        "pregrasp_progress_scale": 0.03,
        "pregrasp_approach_fallback_distance": 0.0,
        "pregrasp_tip_score_weight": 0.65,
        "pregrasp_hand_q_score_weight": 0.25,
        "pregrasp_progress_weight": 0.10,
        "phase_mode": "frame",
        "object_weight": 1.0,
        "object_pos_std": 0.06,
        "object_rot_std": 0.70,
        "object_vel_std": 0.40,
        "object_pos_weight": 0.55,
        "object_rot_weight": 0.25,
        "object_vel_weight": 0.20,
        "contact_bonus_weight": 0.10,
      },
    ),
    # Hand shape before first contact. Off by default: weight 0.0 changes nothing, so every
    # existing run and every archived config reproduces exactly. See docs/13.
    # --- Omnigrasp, ported line for line. See omnigrasp_faithful_mdp and docs/17.
    # ONE term: the two stages replace each other inside it, exactly as humanoid_omnigrasp.py:677
    # does. Expressing them as two frame-gated terms is only equivalent if the gates partition,
    # and the old ones did not -- they keyed on the clip frame, Omnigrasp keys on episode time.
    "og_faithful": RewardTermCfg(
      func=omnigrasp_faithful_mdp.omnigrasp_faithful_reward,
      weight=0.0,
      params={
        "grasp_start_seconds": 1.0,
        "close_distance_pregrasp": 0.2,
        "close_distance_contact": 0.1,
      },
    ),
    # env_x_grab_z.yaml has power_reward: True and penality_slippage: True. Both return a POSITIVE
    # magnitude, so both carry a negative weight here.
    "og_power": RewardTermCfg(
      func=omnigrasp_faithful_mdp.og_power_penalty,
      weight=0.0,
      params={"coefficient": 0.0005},
    ),
    "og_slippage": RewardTermCfg(
      func=omnigrasp_faithful_mdp.og_slippage_penalty,
      weight=0.0,
      params={"coefficient": 0.3},
    ),
    "hand_shape": RewardTermCfg(
      func=omnigrasp_style_mdp.hand_shape_reward,
      weight=0.0,
      params={
        "std": 0.01,
        "pre_contact_weight": 1.0,
        "post_contact_weight": 0.0,
        "contact_threshold": 0.03,
        "pregrasp_reference_distance": 0.12,
        "pregrasp_top_k": 4,
      },
    ),
    "thumb_opposition": RewardTermCfg(
      func=mdp.residual_thumb_opposition_reward,
      weight=0.0,
      params={
        "joint_name": mdp.THUMB_OPPOSITION_JOINT,
        "target_rad": mdp.THUMB_OPPOSITION_TARGET_RAD,
        "std": 0.40,
        "near_threshold": 0.25,
      },
    ),
    "grasp_layout": RewardTermCfg(
      func=mdp.residual_grasp_layout_reward,
      weight=0.0,
      params={
        "tip_target_dists": mdp.GRASP_LAYOUT_TIP_TARGETS_M,
        "radial_std": 0.02,
        "radial_coarse_std": 0.12,
        "radial_coarse_weight": 0.5,
        "opposition_target": mdp.GRASP_LAYOUT_OPPOSITION_COS,
        "opposition_std": 0.30,
        "opposition_mode": "monotonic",
        "radial_weight": 0.6,
        "opposition_weight": 0.4,
        "oppose_pair": (0, 2),
        "hand": "right",
      },
    ),
  }

  metrics = {
    "reward_total": MetricsTermCfg(func=mdp.reward_total_metric),
    "residual_norm": MetricsTermCfg(func=mdp.residual_norm_metric),
    "residual_base_ratio": MetricsTermCfg(func=mdp.residual_base_ratio_metric),
    "body_residual_norm": MetricsTermCfg(func=mdp.body_residual_norm_metric),
    "hand_residual_norm": MetricsTermCfg(func=mdp.hand_residual_norm_metric),
    "leg_residual_norm": MetricsTermCfg(func=mdp.leg_residual_norm_metric),
    "arm_residual_norm": MetricsTermCfg(func=mdp.arm_residual_norm_metric),
    "body_residual_ratio": MetricsTermCfg(func=mdp.body_residual_ratio_metric),
    "astra_body_delta_norm": MetricsTermCfg(func=mdp.astra_body_delta_norm_metric),
    "astra_body_delta_ratio": MetricsTermCfg(func=mdp.astra_body_delta_ratio_metric),
    "astra_body_delta_joint_rms": MetricsTermCfg(
      func=mdp.astra_body_delta_joint_rms_metric
    ),
    "hand_residual_ratio": MetricsTermCfg(func=mdp.hand_residual_ratio_metric),
    "leg_residual_ratio": MetricsTermCfg(func=mdp.leg_residual_ratio_metric),
    "arm_residual_ratio": MetricsTermCfg(func=mdp.arm_residual_ratio_metric),
    "residual_clip_frac": MetricsTermCfg(func=mdp.residual_clip_frac_metric),
    "final_action_clip_frac": MetricsTermCfg(func=mdp.final_action_clip_frac_metric),
    "body_final_delta_norm": MetricsTermCfg(func=mdp.body_final_delta_norm_metric),
    "hand_final_delta_norm": MetricsTermCfg(func=mdp.hand_final_delta_norm_metric),
    "base_action_norm": MetricsTermCfg(func=mdp.base_action_norm_metric),
    "final_action_norm": MetricsTermCfg(func=mdp.final_action_norm_metric),
    "hand_control_gate": MetricsTermCfg(func=mdp.hand_control_gate_metric),
    "hand_mean_delta_norm": MetricsTermCfg(func=mdp.hand_mean_delta_norm_metric),
    "hand_sample_delta_pre_clip_norm": MetricsTermCfg(
      func=mdp.hand_sample_delta_pre_clip_norm_metric
    ),
    "hand_sample_delta_post_clip_norm": MetricsTermCfg(
      func=mdp.hand_sample_delta_post_clip_norm_metric
    ),
    "hand_sample_clip_frac": MetricsTermCfg(func=mdp.hand_sample_clip_frac_metric),
    "body_sample_delta_pre_clip_norm": MetricsTermCfg(
      func=mdp.body_sample_delta_pre_clip_norm_metric
    ),
    "body_sample_delta_post_clip_norm": MetricsTermCfg(
      func=mdp.body_sample_delta_post_clip_norm_metric
    ),
    "body_sample_clip_frac": MetricsTermCfg(func=mdp.body_sample_clip_frac_metric),
    "hand_action_std_mean": MetricsTermCfg(func=mdp.hand_action_std_mean_metric),
    "hand_close_left": MetricsTermCfg(func=mdp.hand_close_left_metric),
    "hand_close_right": MetricsTermCfg(func=mdp.hand_close_right_metric),
    "hand_close_mean": MetricsTermCfg(func=mdp.hand_close_mean_metric),
    "hand_primitive_delta_norm": MetricsTermCfg(
      func=mdp.hand_primitive_delta_norm_metric
    ),
    "token_residual_norm": MetricsTermCfg(func=mdp.token_residual_norm_metric),
    "token_residual_smooth_norm": MetricsTermCfg(
      func=mdp.token_residual_smooth_norm_metric
    ),
    "token_residual_clip_frac": MetricsTermCfg(
      func=mdp.token_residual_clip_frac_metric
    ),
    "decoder_body_delta_norm": MetricsTermCfg(func=mdp.decoder_body_delta_norm_metric),
    "decoder_body_delta_ratio": MetricsTermCfg(
      func=mdp.decoder_body_delta_ratio_metric
    ),
    "decoder_body_delta_joint_rms": MetricsTermCfg(
      func=mdp.decoder_body_delta_joint_rms_metric
    ),
    "decoder_body_delta_joint_abs_max": MetricsTermCfg(
      func=mdp.decoder_body_delta_joint_abs_max_metric
    ),
    "contact_frac": MetricsTermCfg(
      func=mdp.contact_frac_metric,
      params={"threshold": 0.06},
    ),
    "live_contact_006_frac": MetricsTermCfg(
      func=mdp.live_contact_006_metric,
      params={"threshold": 0.06},
    ),
    # The staged terms are all gated on this; if it reads 0 or n_frames the gate is
    # degenerate and every staged term is running on a broken split. Check it first.
    "cf_phase": MetricsTermCfg(func=staged_mdp.cf_phase_metric),
    "lift_success": MetricsTermCfg(
      func=mdp.PhaseALiftSuccessMetric,
      params={
        "lift_height_m": 0.03,
        "hold_duration_s": 0.5,
        "min_contact_tips": 2,
      },
      reduce="last",
    ),
    "ttr_at_012": MetricsTermCfg(
      func=mdp.ttr_at_012_metric,
      params={"threshold": 0.12},
    ),
    "object_mpjpe_mm": MetricsTermCfg(func=mdp.object_mpjpe_mm_metric),
    "sequence_success": MetricsTermCfg(
      func=mdp.sequence_success_metric,
      params={"threshold": 0.12},
      reduce="last",
    ),
    "hand_body_contact_frac": MetricsTermCfg(func=mdp.hand_body_contact_frac_metric),
    "non_tip_hand_body_contact_frac": MetricsTermCfg(
      func=mdp.non_tip_hand_body_contact_frac_metric
    ),
    "surface_contact_frac": MetricsTermCfg(func=mdp.surface_contact_frac_metric),
    "multi_tip_near_frac": MetricsTermCfg(func=mdp.multi_tip_near_frac_metric),
    "contact_duration_max": MetricsTermCfg(func=mdp.contact_duration_max_metric),
    "contact_duration_frac": MetricsTermCfg(func=mdp.contact_duration_frac_metric),
    "grasp_contact_frac": MetricsTermCfg(func=mdp.grasp_contact_frac_metric),
    "object_motion_frac": MetricsTermCfg(func=mdp.object_motion_frac_metric),
    "placement_error": MetricsTermCfg(func=mdp.placement_error_metric),
    "body_err_mse": MetricsTermCfg(func=mdp.body_err_mse_metric),
    "body_xyz_err_mse": MetricsTermCfg(func=mdp.body_xyz_err_mse_metric),
    "body_link_dist_mean": MetricsTermCfg(func=mdp.body_link_dist_mean_metric),
    "body_link_dist_mse": MetricsTermCfg(func=mdp.body_link_dist_mse_metric),
    "lower_body_xyz_err_mse": MetricsTermCfg(func=mdp.lower_body_xyz_err_mse_metric),
    "upper_body_xyz_err_mse": MetricsTermCfg(func=mdp.upper_body_xyz_err_mse_metric),
    "lower_body_link_dist_mean": MetricsTermCfg(
      func=mdp.lower_body_link_dist_mean_metric
    ),
    "upper_body_link_dist_mean": MetricsTermCfg(
      func=mdp.upper_body_link_dist_mean_metric
    ),
    "lower_wrist_link_dist_mean": MetricsTermCfg(
      func=mdp.lower_wrist_link_dist_mean_metric
    ),
    "ankle_link_dist_mean": MetricsTermCfg(
      func=mdp.ankle_link_dist_mean_metric,
      params={"use_tracking_weights": False},
    ),
    "ankle_wrist_link_dist_mean": MetricsTermCfg(
      func=mdp.ankle_wrist_link_dist_mean_metric,
      params={"use_tracking_weights": False},
    ),
    "left_wrist_link_dist_mean": MetricsTermCfg(
      func=mdp.left_wrist_link_dist_mean_metric
    ),
    "right_wrist_link_dist_mean": MetricsTermCfg(
      func=mdp.right_wrist_link_dist_mean_metric
    ),
    "left_wrist_x_abs_err": MetricsTermCfg(
      func=mdp.body_link_axis_abs_metric,
      params={
        "link_group": "left_wrist",
        "axis": "x",
        "log_key": "left_wrist_x_abs_err",
      },
    ),
    "left_wrist_y_abs_err": MetricsTermCfg(
      func=mdp.body_link_axis_abs_metric,
      params={
        "link_group": "left_wrist",
        "axis": "y",
        "log_key": "left_wrist_y_abs_err",
      },
    ),
    "left_wrist_z_abs_err": MetricsTermCfg(
      func=mdp.body_link_axis_abs_metric,
      params={
        "link_group": "left_wrist",
        "axis": "z",
        "log_key": "left_wrist_z_abs_err",
      },
    ),
    "right_wrist_x_abs_err": MetricsTermCfg(
      func=mdp.body_link_axis_abs_metric,
      params={
        "link_group": "right_wrist",
        "axis": "x",
        "log_key": "right_wrist_x_abs_err",
      },
    ),
    "right_wrist_y_abs_err": MetricsTermCfg(
      func=mdp.body_link_axis_abs_metric,
      params={
        "link_group": "right_wrist",
        "axis": "y",
        "log_key": "right_wrist_y_abs_err",
      },
    ),
    "right_wrist_z_abs_err": MetricsTermCfg(
      func=mdp.body_link_axis_abs_metric,
      params={
        "link_group": "right_wrist",
        "axis": "z",
        "log_key": "right_wrist_z_abs_err",
      },
    ),
    "hand_err_mse": MetricsTermCfg(func=mdp.hand_err_mse_metric),
    "raw_contact_arm_err_mse": MetricsTermCfg(func=mdp.raw_contact_arm_err_mse_metric),
    "raw_contact_hand_err_mse": MetricsTermCfg(
      func=mdp.raw_contact_hand_err_mse_metric
    ),
    "raw_contact_object_pos_err": MetricsTermCfg(
      func=mdp.raw_contact_object_pos_err_metric
    ),
    "raw_contact_window": MetricsTermCfg(func=mdp.raw_contact_window_metric),
    "raw_tip_object_dist_err": MetricsTermCfg(
      func=mdp.raw_tip_object_dist_err_metric,
      params={"top_k": 2},
    ),
    "raw_tip_object_ref_dist": MetricsTermCfg(
      func=mdp.raw_tip_object_ref_dist_metric,
      params={"top_k": 2},
    ),
    "raw_tip_object_dist_err_top1": MetricsTermCfg(
      func=mdp.raw_tip_object_dist_err_top1_metric
    ),
    "raw_tip_object_ref_dist_top1": MetricsTermCfg(
      func=mdp.raw_tip_object_ref_dist_top1_metric
    ),
    "raw_tip_object_dist_err_top2": MetricsTermCfg(
      func=mdp.raw_tip_object_dist_err_top2_metric
    ),
    "raw_tip_object_ref_dist_top2": MetricsTermCfg(
      func=mdp.raw_tip_object_ref_dist_top2_metric
    ),
    "raw_tip_object_dist_err_top4": MetricsTermCfg(
      func=mdp.raw_tip_object_dist_err_top4_metric
    ),
    "raw_tip_object_ref_dist_top4": MetricsTermCfg(
      func=mdp.raw_tip_object_ref_dist_top4_metric
    ),
    "raw_tip_object_dist_err_cond_top1": MetricsTermCfg(
      func=mdp.raw_tip_object_dist_err_cond_top1_metric
    ),
    "raw_tip_object_ref_dist_cond_top1": MetricsTermCfg(
      func=mdp.raw_tip_object_ref_dist_cond_top1_metric
    ),
    "raw_tip_object_dist_err_cond_top2": MetricsTermCfg(
      func=mdp.raw_tip_object_dist_err_cond_top2_metric
    ),
    "raw_tip_object_ref_dist_cond_top2": MetricsTermCfg(
      func=mdp.raw_tip_object_ref_dist_cond_top2_metric
    ),
    "raw_tip_object_dist_err_cond_top4": MetricsTermCfg(
      func=mdp.raw_tip_object_dist_err_cond_top4_metric
    ),
    "raw_tip_object_ref_dist_cond_top4": MetricsTermCfg(
      func=mdp.raw_tip_object_ref_dist_cond_top4_metric
    ),
    "raw_tip_radial_err": MetricsTermCfg(func=mdp.raw_tip_radial_err_metric),
    "raw_tip_radial_err_cond": MetricsTermCfg(func=mdp.raw_tip_radial_err_cond_metric),
    "raw_tip_radial_live_dist_cond": MetricsTermCfg(
      func=mdp.raw_tip_radial_live_dist_cond_metric
    ),
    "raw_tip_radial_ref_dist_cond": MetricsTermCfg(
      func=mdp.raw_tip_radial_ref_dist_cond_metric
    ),
    "omnigrasp_raw_contact_frame": MetricsTermCfg(
      func=omnigrasp_style_mdp.omnigrasp_raw_contact_frame_metric,
      params={"contact_threshold": 0.06},
    ),
    "omnigrasp_pregrasp_tip_err": MetricsTermCfg(
      func=omnigrasp_style_mdp.omnigrasp_pregrasp_tip_err_metric,
      params={"contact_threshold": 0.06},
    ),
    "omnigrasp_live_contact_gate": MetricsTermCfg(
      func=omnigrasp_style_mdp.omnigrasp_live_contact_gate_metric,
      params={"contact_threshold": 0.06},
    ),
    "omnigrasp_object_pos_err": MetricsTermCfg(
      func=omnigrasp_style_mdp.omnigrasp_object_pos_err_metric
    ),
    "omnigrasp_post_contact_phase": MetricsTermCfg(
      func=omnigrasp_style_mdp.omnigrasp_post_contact_phase_metric
    ),
    "hand_action_mse": MetricsTermCfg(func=mdp.hand_action_mse_metric),
    "hand_to_obj_dist": MetricsTermCfg(func=mdp.hand_to_obj_dist_metric),
    "hand_to_obj_under_030_frac": MetricsTermCfg(
      func=mdp.EpisodeAnyTipUnderMetric,
      params={
        "threshold_m": 0.30,
        "log_key": "Metric/hand_to_obj_under_030_frac",
      },
      reduce="last",
    ),
    "hand_to_obj_under_015_frac": MetricsTermCfg(
      func=mdp.EpisodeAnyTipUnderMetric,
      params={
        "threshold_m": 0.15,
        "log_key": "Metric/hand_to_obj_under_015_frac",
      },
      reduce="last",
    ),
    "hand_to_obj_under_005_frac": MetricsTermCfg(
      func=mdp.EpisodeAnyTipUnderMetric,
      params={
        "threshold_m": 0.05,
        "log_key": "Metric/hand_to_obj_under_005_frac",
      },
      reduce="last",
    ),
    "obj_drift": MetricsTermCfg(func=mdp.obj_drift_metric),
    "obj_speed": MetricsTermCfg(func=mdp.obj_speed_metric),
    "ep_len": MetricsTermCfg(func=mdp.ep_len_metric),
    "reference_start_frame": MetricsTermCfg(func=mdp.reference_start_frame_metric),
    "tracking_frame": MetricsTermCfg(func=mdp.tracking_frame_metric),
    "reach_030": MetricsTermCfg(
      func=mdp.EpisodeAnyTipUnderMetric,
      params={"threshold_m": 0.30, "log_key": "Stage/reach_030"},
      reduce="last",
    ),
    "reach_015": MetricsTermCfg(
      func=mdp.EpisodeAnyTipUnderMetric,
      params={"threshold_m": 0.15, "log_key": "Stage/reach_015"},
      reduce="last",
    ),
    "reach_005": MetricsTermCfg(
      func=mdp.EpisodeAnyTipUnderMetric,
      params={"threshold_m": 0.05, "log_key": "Stage/reach_005"},
      reduce="last",
    ),
    "stable_reach_030": MetricsTermCfg(
      func=mdp.EpisodeStableAnyTipUnderMetric,
      params={"threshold_m": 0.30, "log_key": "Stage/stable_reach_030"},
      reduce="last",
    ),
    "stable_reach_015": MetricsTermCfg(
      func=mdp.EpisodeStableAnyTipUnderMetric,
      params={"threshold_m": 0.15, "log_key": "Stage/stable_reach_015"},
      reduce="last",
    ),
    "stable_reach_005": MetricsTermCfg(
      func=mdp.EpisodeStableAnyTipUnderMetric,
      params={"threshold_m": 0.05, "log_key": "Stage/stable_reach_005"},
      reduce="last",
    ),
    "near_contact": MetricsTermCfg(func=mdp.near_contact_flag),
    "physical_contact": MetricsTermCfg(func=mdp.physical_contact_flag),
    "hand_body_contact": MetricsTermCfg(func=mdp.hand_body_contact_flag),
    "non_tip_hand_body_contact": MetricsTermCfg(
      func=mdp.non_tip_hand_body_contact_flag
    ),
    "force_close": MetricsTermCfg(func=mdp.force_close_flag),
    "object_moving": MetricsTermCfg(func=mdp.object_moving_flag),
    "stable_not_fallen": MetricsTermCfg(func=mdp.stable_not_fallen_flag),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "fell_over": TerminationTermCfg(func=mdp.not_fallen),
    # Early, parameterised version: stops a doomed episode at the first sign of
    # a fall (height OR tilt) instead of waiting for root_z < 0.45.  Disabled by
    # default (both thresholds 0) so existing runs are unaffected.
    "fell_over_early": TerminationTermCfg(
      func=mdp.fell_over_early,
      params={"min_root_z": 0.0, "min_upright": 0.0},
    ),
    "object_drift": TerminationTermCfg(func=mdp.object_drift_termination),
    # Omnigrasp leash: disabled by default (threshold<=0); Phase-A launch
    # scripts enable it with threshold=0.12 after the contact frame.
    "object_leash": TerminationTermCfg(
      func=omnigrasp_style_mdp.object_leash_termination,
      params={
        "threshold": 0.0,
        "contact_threshold": 0.06,
        "grace_frames": 0,
        "activation_mode": "contact_latch",
      },
    ),
    # Requires the object to actually travel along its reference path. Nothing else
    # in this set penalises ignoring the apple, so a policy that never touches it
    # was previously free to run to the episode limit.
    # Omnigrasp's ONLY termination (humanoid_omnigrasp.py:1215): instantaneous object-to-reference
    # distance over grab_termination_distance=0.12, no +-window, no fall-over check. Off by
    # default; a faithful run turns this on and object_reference_window off.
    # Kill a failed approach directly, instead of waiting for og_object_far to notice that the
    # reference has carried the object 12 cm away from the one still on the table.
    "tip_cf_miss": TerminationTermCfg(
      func=staged_mdp.tip_cf_miss_termination,
      params={"grace_frames": 10, "threshold": 0.03, "near_threshold": 0.10},
    ),
    "og_object_far": TerminationTermCfg(
      func=omnigrasp_faithful_mdp.og_object_far_termination,
      params={"distance": 0.12, "grace_steps": 2},
    ),
    "object_reference_window": TerminationTermCfg(
      func=mdp.object_reference_window_termination,
      params={
        "threshold": 0.05,
        "window": 20,
        "activate_after_frame": 0,
        "grace_steps": 36,
      },
    ),
    "wrist_target_far": TerminationTermCfg(
      func=mdp.wrist_target_far_termination,
      # window=0 keeps the instantaneous check; >0 accepts the wrist being within
      # threshold of its reference at any frame in +-window, like the object term.
      # 56 = the 36 startup control steps the reference is held for, plus 20 frames of grace
      # once it starts moving. `active_after_steps` is on the CONTROL-STEP clock, and start_frame
      # cancels out of it, so this is a fixed amount of elapsed reference motion for every env --
      # but it is NOT a fixed phase relative to cf, because RSI draws start_frame from 0-50. At
      # the old 100 the check opened 31 frames BEFORE cf for an env that started at frame 0 and
      # 19 frames AFTER cf for one that started at 50, and with ep_len at 104-150 it had almost
      # no episode left to fire in: 4.3 terminations per step against og_object_far's 14.9.
      # pre_cf_only: the approach is the only phase this check belongs to. After cf the task
      # is the object, and the wrist is expected to leave the reference trajectory in
      # whatever way keeps it held; killing on wrist error there punishes a carry that is
      # succeeding on the only terms that matter.
      params={"active_after_steps": 56, "threshold": 0.20, "window": 0,
              "pre_cf_only": True},
    ),
    # DISABLED (warmup beyond any reachable episode length), like fell_over_early and
    # object_leash above.
    #
    # This term ends an episode when the closest fingertip-to-object distance stops
    # improving by min_delta. `best` is a ratchet, so once the hand is actually holding the
    # object that distance is at its physical floor -- the object's radius -- and can never
    # improve again. Success satisfies the trigger. Measured: as physical_contact rose
    # 0.022 -> 0.224, no_progress terminations rose 2.99 -> 4.75 and became the dominant
    # terminator, capping mean_episode_length just under its 271-step arming point, which is
    # exactly where the reference lift begins (frame 116 apple / 87 stapler).
    #
    # It also covers nothing uniquely: "ignores the object" and "approaches then stalls" are
    # both caught by object_reference_window and wrist_target_far. Every archived run that
    # reached lift_success > 0.9 had this termination at exactly 0.000.
    #
    # Re-enabling it needs the progress measure to switch from approach-distance to object
    # height once contact is made; a shorter warmup alone makes it worse, not better.
    "no_progress": TerminationTermCfg(
      func=mdp.NoProgressTermination,
      params={"warmup_steps": 1_000_000_000, "patience_steps": 120, "min_delta": 0.005},
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={
        "robot": EntityCfg(
          spec_fn=_robot_spec,
          articulation=EntityArticulationInfoCfg(
            actuators=(
              BuiltinPositionActuatorCfg(
                target_names_expr=(
                  ".*_hip_pitch_joint",
                  ".*_hip_roll_joint",
                  ".*_knee_joint",
                ),
                stiffness=mdp.apple_mdp._S7522,
                damping=mdp.apple_mdp._D7522,
                effort_limit=mdp.apple_mdp._E7522,
                armature=mdp.apple_mdp._ARMATURE_7520_22,
                frictionloss=0.0,
              ),
              BuiltinPositionActuatorCfg(
                target_names_expr=(".*_hip_yaw_joint", "waist_yaw_joint"),
                stiffness=mdp.apple_mdp._S7514,
                damping=mdp.apple_mdp._D7514,
                effort_limit=mdp.apple_mdp._E7514,
                armature=mdp.apple_mdp._ARMATURE_7520_14,
                frictionloss=0.0,
              ),
              BuiltinPositionActuatorCfg(
                target_names_expr=(
                  ".*_ankle_pitch_joint",
                  ".*_ankle_roll_joint",
                  "waist_roll_joint",
                  "waist_pitch_joint",
                ),
                stiffness=2.0 * mdp.apple_mdp._S5020,
                damping=2.0 * mdp.apple_mdp._D5020,
                effort_limit=50.0,
                armature=2.0 * mdp.apple_mdp._ARMATURE_5020,
                frictionloss=0.0,
              ),
              BuiltinPositionActuatorCfg(
                target_names_expr=(
                  ".*_shoulder_pitch_joint",
                  ".*_shoulder_roll_joint",
                  ".*_shoulder_yaw_joint",
                  ".*_elbow_joint",
                  ".*_wrist_roll_joint",
                ),
                stiffness=mdp.apple_mdp._S5020,
                damping=mdp.apple_mdp._D5020,
                effort_limit=mdp.apple_mdp._E5020,
                armature=mdp.apple_mdp._ARMATURE_5020,
                frictionloss=0.0,
              ),
              BuiltinPositionActuatorCfg(
                target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
                stiffness=mdp.apple_mdp._S4010,
                damping=mdp.apple_mdp._D4010,
                effort_limit=mdp.apple_mdp._E4010,
                armature=mdp.apple_mdp._ARMATURE_4010,
                frictionloss=0.0,
              ),
              BuiltinPositionActuatorCfg(
                target_names_expr=_HAND_JOINT_EXPR,
                stiffness=300.0,
                damping=8.0,
                effort_limit=30.0,
                frictionloss=0.0,
              ),
            ),
          ),
          sort_actuators=False,
        ),
        **_object_entities(),
        "table": EntityCfg(spec_fn=_table_spec),
      },
      sensors=_object_sensors(ContactSensorCfg, ContactMatch),
      num_envs=16 if not play else 1,
      env_spacing=6.0,
    ),
    observations=observations,
    actions=actions,
    events={
      "reset_to_residual_interact_curriculum": EventTermCfg(
        func=mdp.reset_to_residual_interact_curriculum,
        mode="reset",
        params={
          # Every clip starts in ITS OWN frames 0-50. The frame passed here is clip-local:
          # _set_reference_start_frame adds that clip's origin and clamps to that clip's own
          # length, so a single window stays correct across clips of different lengths.
          # Contact is at frame 80 (apple) and 48 (stapler), so an episode always begins
          # before the grasp and has to perform it.
          "contact_prob": 1.0,
          "contact_frame_start": 0,
          "contact_frame_end": 50,
          "contact_frame_anchor": "absolute",
          "uniform_precontact_prob": 0.0,
          "precontact_margin_frames": 25,
        },
      ),
    },
    rewards=rewards,
    terminations=terminations,
    metrics=metrics,
    sim=SimulationCfg(
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
      nconmax=256,
      njmax=2048,
    ),
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="torso_link",
      distance=4.0,
      elevation=-10.0,
      azimuth=140.0,
    ),
    decimation=4,
    episode_length_s=12.0 if not play else 1e9,
    scale_rewards_by_dt=False,
  )
