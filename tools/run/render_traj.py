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
A = ap.parse_args()

m = mujoco.MjModel.from_xml_path(A.xml)
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
  return z["qpos"], (z["mocap"] if "mocap" in z and z["mocap"].size else None)

def frames(qpos, mocap):
  out = []
  n_mocap = m.nmocap
  for i in range(0, len(qpos), A.stride):
    d.qpos[:] = qpos[i]
    if mocap is not None and n_mocap:
      mp = mocap[i][: n_mocap * 3].reshape(n_mocap, 3)
      mq = mocap[i][n_mocap * 3: n_mocap * 3 + n_mocap * 4].reshape(n_mocap, 4)
      d.mocap_pos[:] = mp
      d.mocap_quat[:] = mq
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
