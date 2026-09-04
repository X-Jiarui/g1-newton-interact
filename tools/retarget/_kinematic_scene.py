"""Shared scene construction for the kinematic (no-physics) reference replays.

Both `replay_reference_kinematic.py` (writes mp4) and `viser_reference_kinematic.py` (serves a scrubable
Viser page) build the *same* scene from this module, so the two views cannot drift apart. Everything
here is pure MuJoCo — the simulator stack is deliberately not imported, because this tool exists to show
what the retarget produced independently of the environment.

Two details this reproduces on purpose:

* The environment does **not** use the pkl's table pose. `apple_eat/mdp.py` forces an identity
  orientation and centres the table under the object's XY. The raw GRAB table transform is rotated ~90°
  about x and renders the slab standing on edge.
* Table height uses the **object's own bottom offset**, not the apple's 4 cm radius. For the stapler
  those differ by 1.96 cm (0.0204 vs 0.0400), which puts the table visibly too low.
"""

import os
import pickle
import uuid
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

SCENE = """
<mujoco model="replay">
  <include file="{robot}"/>
  {object_asset}
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <headlight ambient="0.5 0.5 0.5" diffuse="0.5 0.5 0.5"/>
  </visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <body name="ref_table" mocap="true" pos="0 0 -5">
      <geom name="ref_table_geom" type="box" size="{tx} {tx} {tt}" rgba="0.55 0.42 0.30 1"
            contype="0" conaffinity="0"/>
    </body>
    <body name="ref_apple" mocap="true" pos="0 0 -5">
      {object_geom}
    </body>
  </worldbody>
</mujoco>
"""

APPLE_RADIUS = 0.04
TABLE_XY_SIZE = 0.21
TABLE_THICKNESS = 0.04


@dataclass
class KinematicScene:
  """A compiled replay scene plus the reference arrays needed to pose it."""

  model: mujoco.MjModel
  data: mujoco.MjData
  dof_pos: np.ndarray       # (n_frames, n_dof)
  root_pos: np.ndarray      # (n_frames, 3)
  root_quat_wxyz: np.ndarray  # (n_frames, 4)
  object_pos: np.ndarray    # (n_frames, 3)
  object_quat: np.ndarray   # (n_frames, 4) wxyz
  table_pos: np.ndarray     # (n_frames, 3)
  table_quat: np.ndarray    # (n_frames, 4) wxyz
  mocap_apple: int
  mocap_table: int
  fps: float
  object_mesh: str          # "" when a sphere stand-in was used

  @property
  def n_frames(self) -> int:
    return len(self.dof_pos)

  def set_frame(self, i: int) -> None:
    """Pose the model at reference frame `i` by forward kinematics only."""
    i = int(np.clip(i, 0, self.n_frames - 1))
    d = self.data
    d.qpos[:3] = self.root_pos[i]
    d.qpos[3:7] = self.root_quat_wxyz[i]
    d.qpos[7:7 + self.dof_pos.shape[1]] = self.dof_pos[i]
    d.mocap_pos[self.mocap_apple] = self.object_pos[i]
    d.mocap_quat[self.mocap_apple] = self.object_quat[i]
    d.mocap_pos[self.mocap_table] = self.table_pos[i]
    d.mocap_quat[self.mocap_table] = self.table_quat[i]
    mujoco.mj_forward(self.model, d)

  def object_rise_cm(self) -> np.ndarray:
    """Object height above its first frame, in cm — the quantity `lift_success` thresholds."""
    return (self.object_pos[:, 2] - self.object_pos[0, 2]) * 100.0


def _object_bottom_offset(stl: str, quat_wxyz: np.ndarray, fallback: float) -> float:
  """Distance from the object's origin down to its lowest vertex, in its first-frame orientation."""
  if not stl or not Path(stl).exists():
    return fallback
  try:
    import trimesh

    v = np.asarray(trimesh.load(stl, process=False).vertices, dtype=np.float64)
    w, x, y, z = quat_wxyz
    rot = np.array([
      [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    off = float(-(v @ rot.T)[:, 2].min())
    print(f"[table] object bottom offset {off * 100:.2f} cm "
          f"(sphere assumption was {fallback * 100:.2f} cm)")
    return off
  except Exception as e:  # a missing trimesh or an unreadable mesh must not be fatal
    print(f"[table] could not measure the mesh bottom ({e}); using the sphere radius")
    return fallback


def build(pkl: str, robot_xml: str, object_mesh: str = "auto",
          apple_radius: float = APPLE_RADIUS, table_xy_size: float = TABLE_XY_SIZE,
          table_thickness: float = TABLE_THICKNESS) -> KinematicScene:
  """Compile the replay scene for one reference clip.

  object_mesh: "auto" uses the stl_path recorded in the reference, "none" forces a sphere of
  apple_radius, anything else is taken as an explicit .stl path.
  """
  d = pickle.load(open(pkl, "rb"))
  r = d["robot_53dof"]                       # container name; holds 69 columns for Wuji
  q = np.asarray(r["dof_pos"], dtype=np.float64)
  rp = np.asarray(r["root_pos"], dtype=np.float64)
  rr = np.asarray(r["root_rot"], dtype=np.float64)          # xyzw
  obj_p = np.asarray(d["object"]["pos_mj"], dtype=np.float64)
  obj_q = np.asarray(d["object"]["quat_wxyz_mj"], dtype=np.float64)

  stl = str(d["object"].get("stl_path", "")) if object_mesh != "none" else ""
  if object_mesh not in ("none", "auto"):
    stl = object_mesh

  bottom = _object_bottom_offset(stl, obj_q[0], apple_radius)
  table_top = float(obj_p[0, 2]) - bottom
  tab_p = np.asarray(d["table"]["pos_mj"], dtype=np.float64).copy()
  tab_p[:, 0] = obj_p[0, 0]
  tab_p[:, 1] = obj_p[0, 1]
  tab_p[:, 2] = table_top - 0.5 * table_thickness
  tab_q = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(tab_p), 1))

  if stl and Path(stl).exists():
    object_asset = f'<asset><mesh name="ref_object_mesh" file="{Path(stl).resolve()}"/></asset>'
    object_geom = ('<geom name="ref_apple_geom" type="mesh" mesh="ref_object_mesh" '
                   'rgba="0.85 0.15 0.15 1" contype="0" conaffinity="0"/>')
    print(f"[object] mesh {Path(stl).name}")
    used_mesh = stl
  else:
    object_asset = ""
    object_geom = (f'<geom name="ref_apple_geom" type="sphere" size="{apple_radius}" '
                   f'rgba="0.85 0.15 0.15 1" contype="0" conaffinity="0"/>')
    print(f"[object] sphere r={apple_radius}"
          + ("" if not stl else f" (mesh {stl} not found)"))
    used_mesh = ""

  xml = SCENE.format(robot=robot_xml, tx=0.5 * table_xy_size, tt=0.5 * table_thickness,
                     object_asset=object_asset, object_geom=object_geom)
  # Write the scene next to the robot xml: from_xml_string resolves the model's relative meshdir
  # against the cwd rather than the including file, so the STLs would not be found.
  # The name has to be unique per process. It used to be a fixed "_replay_scene_tmp.xml", and two
  # tools running at once then overwrote each other's scene between write and compile -- once as a
  # visible "XML parse error", but the silent case is worse: the second process compiles the FIRST
  # one's object mesh and reports distances to the wrong shape. Two dataset evaluations in parallel
  # is the normal case here, so this was a live correctness bug, not a theoretical one.
  scene_path = Path(robot_xml).with_name(
      f"_replay_scene_tmp_{os.getpid()}_{uuid.uuid4().hex[:8]}.xml")
  scene_path.write_text(xml.replace(robot_xml, Path(robot_xml).name))
  try:
    model = mujoco.MjModel.from_xml_path(str(scene_path))
  finally:
    scene_path.unlink(missing_ok=True)

  def mocap_id(name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
      raise SystemExit(f"body {name!r} missing from the compiled scene")
    return int(model.body_mocapid[bid])

  expected = 7 + q.shape[1]
  if model.nq < expected:
    raise SystemExit(f"model has nq={model.nq}, need at least {expected} for a "
                     f"{q.shape[1]}-DOF reference; wrong robot xml for this dataset?")

  return KinematicScene(
    model=model, data=mujoco.MjData(model), dof_pos=q, root_pos=rp,
    root_quat_wxyz=rr[:, [3, 0, 1, 2]], object_pos=obj_p, object_quat=obj_q,
    table_pos=tab_p, table_quat=tab_q,
    mocap_apple=mocap_id("ref_apple"), mocap_table=mocap_id("ref_table"),
    fps=float(d.get("fps", 30.0)), object_mesh=used_mesh)
