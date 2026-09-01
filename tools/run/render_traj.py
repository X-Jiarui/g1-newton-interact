"""Render recorded simulator state to video, and stack two runs side by side.

The trajectories are replayed through mjlab's FULL model (226 geoms, visual meshes included) rather
than Newton's converted one, which keeps only the 81 colliding geoms and would draw a robot made of
bare collision primitives. The physics being shown is still whichever simulator produced the qpos --
this only borrows the appearance.

mocap is replayed alongside qpos because the table is a mocap-driven body: qpos alone leaves it at the
origin and the apple would appear to float.
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np
import mujoco

ap = argparse.ArgumentParser()
ap.add_argument("--xml", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets/mjlab_scene/scene.xml"))
ap.add_argument("--left", required=True, help="npz trace, drawn on the left")
ap.add_argument("--right", default=None, help="optional npz trace, drawn on the right")
ap.add_argument("--left-label", default="left")
ap.add_argument("--right-label", default="right")
ap.add_argument("--out", required=True)
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--height", type=int, default=480)
ap.add_argument("--fps", type=int, default=50)
ap.add_argument("--stride", type=int, default=1)
ap.add_argument("--object-mesh", default=None,
                help="draw the object as this STL instead of the scene's placeholder sphere; the mesh is what the physics actually collided against")
A = ap.parse_args()

def _model_with_object_mesh(xml: str, stl: str | None):
  """Compile the render scene, optionally drawing the object as its real mesh.

  The scene authors the object as a 4cm sphere placeholder; the true mesh only ever reached the
  collider, swapped in at runtime. Rendering the placeholder shows a red ball where the physics
  had a stapler, which is worse than no picture at all.
  """
  if not stl:
    return mujoco.MjModel.from_xml_path(xml)
  spec = mujoco.MjSpec.from_file(xml)
  target = None
  for body in spec.bodies:
    for g in body.geoms:
      if g.name.endswith("apple_geom"):
        target = g
        break
  if target is None:
    raise SystemExit("no object geom named *apple_geom in the render scene")
  mesh_name = "render_object_mesh"
  mesh = spec.add_mesh()
  mesh.name = mesh_name
  mesh.file = os.path.abspath(stl)
  target.type = mujoco.mjtGeom.mjGEOM_MESH
  target.meshname = mesh_name
  target.size[:] = (0.0, 0.0, 0.0)
  return spec.compile()


m = _model_with_object_mesh(A.xml, A.object_mesh)
d = mujoco.MjData(m)
renderer = mujoco.Renderer(m, height=A.height, width=A.width)

cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
cam.lookat[:] = [0.80, -0.10, 0.85]
cam.distance = 1.9
cam.elevation = -14.0
cam.azimuth = 118.0

def load(p):
  z = np.load(p, allow_pickle=True)
  if "mocap_pos" in z:
    return z["qpos"], (z["mocap_pos"], z["mocap_quat"],
                       [str(x) for x in z["mocap_names"]] if "mocap_names" in z else None)
  # older traces stored one flat mocap row per frame, in the recording model's own order
  return z["qpos"], ((z["mocap"], None, None) if "mocap" in z and z["mocap"].size else None)

def _mocap_slot(name: str) -> int:
  """Mocap index in the render model for a body recorded under `name`.

  Newton rewrites body names on import, so the match is on the flattened suffix, and it must be
  unique: silently guessing an index is how the table ended up under the robot.
  """
  want = name.replace("/", "_")
  hits = []
  for b in range(m.nbody):
    if m.body_mocapid[b] < 0:
      continue
    n = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").replace("/", "_")
    if n.endswith(want) or want.endswith(n):
      hits.append(int(m.body_mocapid[b]))
  if len(hits) > 1:
    raise SystemExit(f"mocap body {name!r} matched {len(hits)} bodies in the render model; "
                     "refusing to guess -- picking one is how the table ended up under the robot")
  if not hits:
    # Newton's scene carries mocap bodies the render scene does not (the terrain plane). Leaving
    # such a body unposed is fine; inventing a slot for it is not. The table must still match, and
    # it does -- it is the only recorded mocap body the render model actually has.
    return -1
  return hits[0]

def frames(qpos, mocap):
  out = []
  n_mocap = m.nmocap
  for i in range(0, len(qpos), A.stride):
    d.qpos[:] = qpos[i]
    if mocap is not None and n_mocap:
      mpos, mquat, mnames = mocap
      if mquat is None:                       # legacy flat layout, index order as recorded
        row = mpos[i]
        d.mocap_pos[:] = row[: n_mocap * 3].reshape(n_mocap, 3)
        d.mocap_quat[:] = row[n_mocap * 3: n_mocap * 7].reshape(n_mocap, 4)
      elif mnames is None:
        d.mocap_pos[:] = mpos[i]
        d.mocap_quat[:] = mquat[i]
      else:
        for src, name in enumerate(mnames):
          dst = _mocap_slot(name)
          if dst < 0:
            continue
          d.mocap_pos[dst] = mpos[i][src]
          d.mocap_quat[dst] = mquat[i][src]
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    renderer.update_scene(d, camera=cam)
    out.append(renderer.render().copy())
  return out

lq, lm = load(A.left)
left = frames(lq, lm)
print(f"left  {A.left_label}: {len(left)} frames")
if A.right:
  rq, rm = load(A.right)
  right = frames(rq, rm)
  print(f"right {A.right_label}: {len(right)} frames")
  n = min(len(left), len(right))
  left, right = left[:n], right[:n]
else:
  right = None

def label(img, text, sub=None):
  try:
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, im.width, 26], fill=(0, 0, 0))
    dr.text((8, 7), text, fill=(255, 255, 255))
    if sub:
      dr.rectangle([0, im.height - 22, im.width, im.height], fill=(0, 0, 0))
      dr.text((8, im.height - 17), sub, fill=(200, 200, 200))
    return np.asarray(im)
  except Exception:
    return img

import imageio_ffmpeg
w = imageio_ffmpeg.write_frames(A.out, (A.width * (2 if right else 1), A.height),
                                fps=A.fps, quality=7)
w.send(None)
for i in range(len(left)):
  li = label(left[i], A.left_label, f"step {i * A.stride}")
  if right is not None:
    ri = label(right[i], A.right_label, f"step {i * A.stride}")
    frame = np.concatenate([li, ri], axis=1)
  else:
    frame = li
  w.send(np.ascontiguousarray(frame))
w.close()
print(f"wrote {A.out}  ({Path(A.out).stat().st_size/1e6:.2f} MB)")
