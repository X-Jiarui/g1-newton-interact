"""Is the swapped-in mesh intersecting the table at the reset pose?

The scene places the object using the geometry mjlab authored: a 4 cm analytic sphere. The stapler
mesh is 12.3 x 3.0 x 4.1 cm and the mug 11.6 x 8.2 x 10.5 cm, so the rest height computed for a
sphere does not put either of them on the table -- it puts them through it. One step of a solver
resolving that penetration is enough to produce NaN, which is what the rollout shows.
"""
from __future__ import annotations
import os, sys
import numpy as np, trimesh, mujoco

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MESHES = os.path.expanduser("~/jiarui/scaled_grab_wuji_all_o70/meshes")

for name, scene in (("stapler", "scene_stapler"), ("mug", "scene_mug")):
  xml = os.path.join(HERE, "assets", scene, "scene.xml")
  if not os.path.exists(xml):
    print(f"{name}: no scene"); continue
  m = mujoco.MjModel.from_xml_path(xml)
  d = mujoco.MjData(m); mujoco.mj_forward(m, d)

  def bid(n):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
  ab, tb = bid("apple/apple"), bid("table/table")
  apple_z = float(d.xpos[ab][2]) if ab >= 0 else float("nan")
  # table top from its geom
  tgeom = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == tb]
  ttop = float(d.geom_xpos[tgeom[0]][2] + m.geom_size[tgeom[0]][2]) if tgeom else float("nan")
  sphere = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == ab]
  srad = float(m.geom_size[sphere[0]][0]) if sphere else float("nan")

  mesh = trimesh.load(os.path.join(MESHES, f"{name}.stl"), force="mesh")
  lo, hi = mesh.bounds
  half_below = float(-lo[2])          # how far the mesh extends below its own origin

  print(f"\n=== {name} ===")
  print(f"  scene places the object origin at z = {apple_z:.4f} m")
  print(f"  table top at z = {ttop:.4f} m")
  print(f"  authored collider: sphere r = {srad:.4f} m  -> bottom at {apple_z - srad:.4f}")
  print(f"  mesh extends {half_below:.4f} m below its origin -> bottom at {apple_z - half_below:.4f}")
  pen_sphere = ttop - (apple_z - srad)
  pen_mesh = ttop - (apple_z - half_below)
  print(f"  sphere penetration into table: {pen_sphere*100:+.2f} cm")
  print(f"  MESH   penetration into table: {pen_mesh*100:+.2f} cm"
        f"{'   <-- intersecting' if pen_mesh > 0.002 else ''}")
  print(f"  mesh extents: {np.round(hi - lo, 4).tolist()} m")
