"""World-frame extreme height of a body's colliding geometry, from its transformed vertices.

Half-thickness was read from geom_size, then from a mesh's local z-extent. Both assume an
orientation, and MuJoCo reorients mesh assets: the table box written with half-extents
(0.105, 0.105, 0.02) reported a z half of 0.105, five times too thick, so the table was placed 85mm
low while the printed number looked correct. Transforming the vertices by the geom's own world
frame assumes nothing.
"""

from __future__ import annotations

import numpy as np


def body_collider_extreme_z(mj_model, mj_data, body_id: int, which: str = "max") -> float:
  """Highest (``which="max"``) or lowest (``"min"``) world-frame point of a body's colliders."""
  import mujoco

  vals: list[float] = []
  for g in range(mj_model.ngeom):
    if mj_model.geom_bodyid[g] != body_id or mj_model.geom_contype[g] == 0:
      continue
    pos = np.asarray(mj_data.geom_xpos[g], dtype=np.float64)
    rot = np.asarray(mj_data.geom_xmat[g], dtype=np.float64).reshape(3, 3)
    if int(mj_model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH):
      did = int(mj_model.geom_dataid[g])
      va, vn = int(mj_model.mesh_vertadr[did]), int(mj_model.mesh_vertnum[did])
      v = np.asarray(mj_model.mesh_vert[va:va + vn], dtype=np.float64).reshape(-1, 3)
      z = (v @ rot.T)[:, 2] + pos[2]
      vals.append(float(z.max() if which == "max" else z.min()))
    else:
      half = float(np.abs(rot @ np.asarray(mj_model.geom_size[g], dtype=np.float64))[2])
      vals.append(float(pos[2] + half if which == "max" else pos[2] - half))
  if not vals:
    raise RuntimeError(f"body {body_id} has no colliding geom, so it has no surface")
  return max(vals) if which == "max" else min(vals)
