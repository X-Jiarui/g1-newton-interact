"""Put the table under the object, instead of moving the object onto the table.

The scene was authored around a 4cm sphere placeholder: the object's reference height and the
mocap table's height were chosen together so that a sphere of radius APPLE_RADIUS rests exactly on
the table surface. Swap in an object's true collider and that agreement breaks by however much the
real mesh's lowest point differs from the sphere's -- measured here, the stapler reaches 20.3mm
below its origin against the sphere's 40mm, so it falls 19.7mm before it rests; the mug reaches
52.5mm, so it starts 12.5mm inside the table and is pushed up.

Either way the object ends up somewhere the reference trajectory does not describe, and
`object_trajectory_tracking` (weight 2.0) can never be satisfied. Measured as object_mpjpe_mm:
23.6 for the stapler and 13.6 for the mug, against 3.6 for mjlab's sphere.

The object's pose is the quantity the reward tracks. The table's pose is tracked by nothing. So the
table moves: its surface is placed `gap` below wherever the object's real collider bottom is, and
the object stays where the reference puts it.
"""

from __future__ import annotations

import struct

import numpy as np

# The object is dropped onto the table from this height. Small enough that the
# settling is not visible and does not show up in object_mpjpe_mm, large enough
# that the object does not start already interpenetrating the surface.
DEFAULT_GAP = 0.0005


def _mesh_vertices(stl_path: str):
  """Vertices of a binary STL, in the mesh's own frame."""
  with open(stl_path, "rb") as f:
    head = f.read(84)
    n = struct.unpack("<I", head[80:84])[0]
    data = np.frombuffer(f.read(n * 50), dtype=np.uint8).reshape(n, 50)
  tris = data[:, 12:48].copy().view("<f4").reshape(n, 3, 3)
  return tris.reshape(-1, 3).astype(np.float64)


def _quat_to_mat(q):
  w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
  return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                   [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                   [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def object_bottom_at_rest(stl_path: str, reference_pkl: str, z_offset: float = 0.0,
                          frame: int = 0) -> float:
  """World height of the object's lowest collider point at the reference's first frame.

  Uses the object's reference orientation, not the mesh's own frame: the stapler lies flat at rest,
  where rotation costs only 0.5mm, but it is rotated 45mm out of that pose while being carried.
  """
  import pickle
  ref = pickle.load(open(reference_pkl, "rb"))
  obj = ref["object"]
  pos = np.asarray(obj["pos_mj"])[frame]
  quat = np.asarray(obj["quat_wxyz_mj"])[frame]
  v = _mesh_vertices(stl_path) @ _quat_to_mat(quat).T
  return float(pos[2]) + float(z_offset) + float(v[:, 2].min())


def _table_half_thickness(mj_model, table_body: str = "table/table") -> float:
  """Half height of the table's colliding geom, from the compiled model."""
  import mujoco
  bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, table_body)
  if bid < 0:
    # Newton's importer rewrites body names (`table/table` becomes something like
    # `mjlab scene_worldbody_table_table`), so match on the flattened suffix.
    want = table_body.replace("/", "_")
    cands = [i for i in range(mj_model.nbody)
             if (mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i) or "")
             .replace("/", "_").endswith(want)]
    if len(cands) != 1:
      names = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
               for i in range(mj_model.nbody)]
      raise RuntimeError(f"cannot identify the table body: {table_body!r} matched {len(cands)} of "
                         f"{[n for n in names if n and 'table' in n.lower()]}")
    bid = cands[0]
  halves = [float(mj_model.geom_size[g][2]) for g in range(mj_model.ngeom)
            if mj_model.geom_bodyid[g] == bid and mj_model.geom_contype[g] != 0]
  if not halves:
    raise RuntimeError("the table body has no colliding geom, so it has no surface")
  return max(halves)


def install(mj_model, stl_path: str, reference_pkl: str, z_offset: float = 0.0,
            gap: float = DEFAULT_GAP, verbose: bool = True) -> None:
  """Move the mocap table so the object's true collider rests where the reference places it.

  The shift is measured on the first table write rather than computed in advance. The table is a
  mocap body whose runtime pose comes from the reference clip through a transform this module does
  not model; reading it from a fresh MjData gives the authored pose (z=0), which is off by the
  whole table height. The first pose actually written is the truth, so the shift is derived from
  it once and reused.

  Patches the function in mjlab's module rather than editing mjlab, so the unmodified package stays
  available as the control this port is measured against.
  """
  import mjlab.tasks.apple_eat.mdp as apple_mdp

  orig = apple_mdp._write_table_pose
  if getattr(orig, "_newton_table_shift", False):
    raise RuntimeError("the table shift is already installed; installing twice would stack shifts")

  half = _table_half_thickness(mj_model)
  desired_top = object_bottom_at_rest(stl_path, reference_pkl, z_offset) - float(gap)
  state = {"delta": None}

  def shifted(table, table_pose, env_ids=None):
    if state["delta"] is None:
      current_top = float(table_pose[0, 2].item()) + half
      state["delta"] = desired_top - current_top
      if verbose:
        print(f"[newton-env] table top {current_top:.4f} -> {desired_top:.4f} "
              f"({1000.0 * state['delta']:+.1f} mm) so the object's true collider rests where the "
              f"reference places it")
    pose = table_pose.clone()
    pose[:, 2] += state["delta"]
    return orig(table, pose, env_ids=env_ids)

  shifted._newton_table_shift = True
  apple_mdp._write_table_pose = shifted
