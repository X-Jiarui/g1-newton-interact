"""Render mjlab and Newton trajectories side by side into one mp4.

Both panels are drawn with mjlab's full visual model -- the same 226 geoms, materials and meshes --
so nothing about the appearance differs between them. Only the state differs: each panel is driven by
the qpos (and table mocap pose) its own simulator produced. Newton's converted model carries 81
collision geoms and no visual-only geometry, so rendering it directly would show a different-looking
robot and invite the difference to be blamed on the render rather than the physics.

The table is driven through mocap_pos/mocap_quat, which live outside qpos, so they are replayed too.
"""
from __future__ import annotations
import argparse, os
import numpy as np, mujoco, imageio.v2 as imageio

ap = argparse.ArgumentParser()
ap.add_argument("--xml", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets/mjlab_scene/scene.xml"))
ap.add_argument("--left", default="/tmp/mjlab_qpos.npz")
ap.add_argument("--left-label", default="mjlab (MuJoCo Warp)")
ap.add_argument("--right", default="/tmp/newton_qpos.npz")
ap.add_argument("--right-label", default="Newton 1.5")
ap.add_argument("--out", default=os.path.expanduser("~/newton_vs_mjlab.mp4"))
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--height", type=int, default=520)
ap.add_argument("--fps", type=int, default=50)
# Newton's conversion turned every world-attached fixed body into a mocap body, so it has 3 where the
# render model has 1, and the table is at index 2 rather than 0. Copying slot-for-slot would place the
# table wherever Newton's terrain sits -- the origin -- and the apple would appear to float.
ap.add_argument("--left-mocap-idx", type=int, default=0)
ap.add_argument("--right-mocap-idx", type=int, default=2)
ap.add_argument("--azimuth", type=float, default=110.0)
ap.add_argument("--elevation", type=float, default=-12.0)
ap.add_argument("--distance", type=float, default=1.7)
A = ap.parse_args()

model = mujoco.MjModel.from_xml_path(A.xml)
print(f"render model: nq={model.nq} ngeom={model.ngeom} nmocap={model.nmocap}")

def load(path):
  z = np.load(path, allow_pickle=True)
  return (z["qpos"], z.get("mocap_pos"), z.get("mocap_quat"))

L, R = load(A.left), load(A.right)
for nm, src, idx in (("left", L, A.left_mocap_idx), ("right", R, A.right_mocap_idx)):
  if src[1] is not None:
    print(f"  {nm}: {src[1].shape[1]} mocap bodies, using index {min(idx, src[1].shape[1]-1)} "
          f"at {np.round(src[1][0][min(idx, src[1].shape[1]-1)], 3).tolist()}")
n = min(len(L[0]), len(R[0]))
print(f"frames: left {len(L[0])} right {len(R[0])} -> {n}")

try:
  from PIL import Image, ImageDraw
  HAVE_PIL = True
except Exception:
  HAVE_PIL = False
  print("PIL unavailable; panels will be unlabelled")


def make_renderer():
  d = mujoco.MjData(model)
  r = mujoco.Renderer(model, height=A.height, width=A.width)
  c = mujoco.MjvCamera()
  c.type = mujoco.mjtCamera.mjCAMERA_FREE
  c.azimuth, c.elevation, c.distance = A.azimuth, A.elevation, A.distance
  return d, r, c


dL, rL, cam = make_renderer()
dR, rR, _ = make_renderer()


def frame(data, rend, src, i, mocap_idx=0):
  q, mp, mq = src
  data.qpos[:] = q[i]
  if mp is not None and model.nmocap:
    j = min(mocap_idx, mp.shape[1] - 1)
    data.mocap_pos[0] = mp[i][j]
    data.mocap_quat[0] = mq[i][j]
  data.qvel[:] = 0
  mujoco.mj_forward(model, data)
  # Track the hand/table region rather than the world origin, so the grasp stays in shot.
  cam.lookat[:] = (0.72, -0.06, 0.80)
  rend.update_scene(data, camera=cam)
  return rend.render()


def label(img, text):
  if not HAVE_PIL:
    return img
  im = Image.fromarray(img)
  d = ImageDraw.Draw(im)
  d.rectangle([0, 0, im.width, 26], fill=(20, 20, 24))
  d.text((10, 7), text, fill=(240, 240, 245))
  return np.asarray(im)


writer = imageio.get_writer(A.out, fps=A.fps, codec="libx264",
                            macro_block_size=None, quality=8)
for i in range(n):
  a = label(frame(dL, rL, L, i, A.left_mocap_idx), f"{A.left_label}   step {i}")
  b = label(frame(dR, rR, R, i, A.right_mocap_idx), f"{A.right_label}   step {i}")
  sep = np.full((a.shape[0], 4, 3), 60, dtype=np.uint8)
  writer.append_data(np.concatenate([a, sep, b], axis=1))
  if i % 100 == 0:
    print(f"  frame {i}/{n}")
writer.close()
print(f"wrote {A.out}  ({n} frames, {n / A.fps:.1f}s at {A.fps}fps)")
