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
DEFAULT_GAP = 0.003


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


def table_top_world(mj_model, mj_data, table_body: str = "table/table") -> float:
  """World height of the table's colliding surface, from its transformed vertices.

  Half-thickness was previously read from geom_size, then from a mesh's local z-extent. Both assume
  an orientation: MuJoCo reorients mesh assets, so the box written with half-extents
  (0.105, 0.105, 0.02) reported a z half of 0.105 -- five times too thick -- and the table was
  placed 85mm low while the printed number looked correct. Transforming the vertices by the geom's
  own world frame assumes nothing.
  """
  import mujoco
  import sys as _sys, os as _os
  _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
  from newton_extents import body_collider_extreme_z

  bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, table_body)
  if bid < 0:
    want = table_body.replace("/", "_")
    cands = [k for k in range(mj_model.nbody)
             if (mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, k) or "")
             .replace("/", "_").endswith(want)]
    if len(cands) != 1:
      raise RuntimeError(f"cannot identify the table body: {table_body!r} matched {len(cands)}")
    bid = cands[0]
  return body_collider_extreme_z(mj_model, mj_data, bid, "max")


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

  import mujoco as _mj
  import sys as _sys, os as _os
  _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
  from newton_extents import body_collider_extreme_z

  # Measure the authored surface and the authored body height together, so the runtime surface can
  # be tracked as "authored surface + however far the mocap pose has moved the body". Deriving the
  # surface from a half-thickness is what put the table 85mm low: MuJoCo reorients mesh assets, so
  # a box written with half-extents (0.105, 0.105, 0.02) reports a z half of 0.105.
  _d = _mj.MjData(mj_model)
  _mj.mj_forward(mj_model, _d)
  _bid = _mj.mj_name2id(mj_model, _mj.mjtObj.mjOBJ_BODY, "table/table")
  if _bid < 0:
    _cands = [k for k in range(mj_model.nbody)
              if (_mj.mj_id2name(mj_model, _mj.mjtObj.mjOBJ_BODY, k) or "")
              .replace("/", "_").endswith("table_table")]
    if len(_cands) != 1:
      raise RuntimeError(f"cannot identify the table body; matched {len(_cands)}")
    _bid = _cands[0]
  authored_top = body_collider_extreme_z(mj_model, _d, _bid, "max")
  authored_body_z = float(_d.xpos[_bid][2])
  desired_top = object_bottom_at_rest(stl_path, reference_pkl, z_offset) - float(gap)
  # Temporary probe hook: reproduce an earlier table height exactly, to test whether contact
  # behaved differently there rather than arguing from recollection.
  import os as _os
  _extra = float(_os.environ.get("TABLE_EXTRA_SHIFT", "0.0"))
  if _extra:
    desired_top += _extra
    print(f"[newton-env] TABLE_EXTRA_SHIFT {_extra*1000:+.1f} mm applied for this probe")

  state = {"delta": None}

  def shifted(table, table_pose, env_ids=None):
    if state["delta"] is None:
      # the authored surface moves with the mocap pose, so track the delta from it
      current_top = authored_top + (float(table_pose[0, 2].item()) - authored_body_z)
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
