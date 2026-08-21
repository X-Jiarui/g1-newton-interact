"""Serve GRAB objects as Newton SDF colliders in a viser viewer on localhost.

Same workflow as the mjlab viser setup: a port you tunnel to and open in a browser. What is on screen
is Newton's own scene graph -- the objects are the meshes the dataset ships, collided through their
signed distance fields, not through a convex stand-in.

They are dropped onto a plane so the contact is doing something visible rather than posing.
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--objects", default="stapler,mug")
ap.add_argument("--port", type=int, default=8090)
ap.add_argument("--sdf-resolution", type=int, default=128)
ap.add_argument("--spacing", type=float, default=0.25)
ap.add_argument("--drop-height", type=float, default=0.18)
ap.add_argument("--steps", type=int, default=100000)
ap.add_argument("--still", default=None, help="also render a single PNG with ViewerGL (headless)")
ap.add_argument("--settle-steps", type=int, default=600, help="physics steps before the still")
A = ap.parse_args()

sys.path.insert(0, os.path.expanduser("~/projects/g1-newton-interact/src"))
import mjw_compat; mjw_compat.apply()
import newton, warp as wp
import newton.viewer as nv
from newton.solvers import SolverMuJoCo
from grab_objects import add_grab_object

names = [n.strip() for n in A.objects.split(",") if n.strip()]
builder = newton.ModelBuilder()
SolverMuJoCo.register_custom_attributes(builder)
builder.default_shape_cfg.gap = 0.0

print(f"loading {len(names)} GRAB objects straight into Newton (no MJCF):")
for i, n in enumerate(names):
  x = (i - (len(names) - 1) / 2.0) * A.spacing
  add_grab_object(builder, n, pos=(x, 0.0, A.drop_height),
                  sdf_resolution=A.sdf_resolution)

builder.add_ground_plane()
model = builder.finalize()
solver = SolverMuJoCo(model, njmax=512, nconmax=512)

if A.still:
  os.environ.setdefault("PYGLET_HEADLESS", "1")
  import pyglet as _pyglet
  _pyglet.options["headless"] = True
  viewer = nv.ViewerGL(width=1100, height=640, headless=True)
else:
  viewer = nv.ViewerViser(port=A.port, label="GRAB objects as Newton SDF colliders")
viewer.set_model(model)

# Newton does not draw colliders by default -- it draws visual geometry, and these bodies have only
# a collider each. Reveal them, or the scene is empty.
flags = wp.to_torch(model.shape_flags)
VIS = int(newton.ShapeFlags.VISIBLE)
flags |= VIS
print(f"\nviser: http://localhost:{A.port}")
print(f"  tunnel with:  ssh -L {A.port}:localhost:{A.port} jiarui@100.83.215.34")

state_in, state_out = model.state(), model.state()
control = model.control()
dt = 1.0 / 200.0
t = 0.0
if A.still:
  cam = getattr(viewer, "camera", None)
  if cam is not None:
    cam.pos = np.array([0.0, -0.55, 0.30], dtype=np.float32)
    cam.look_at(np.array([0.0, 0.0, 0.04], dtype=np.float32))
  for _ in range(A.settle_steps):          # let them settle before the picture
    solver.step(state_in, state_out, control, None, dt)
    state_in, state_out = state_out, state_in
  viewer.begin_frame(0.0); viewer.log_state(state_in); viewer.end_frame()
  img = viewer.get_frame()
  arr = np.asarray(img.numpy() if hasattr(img, "numpy") else img)
  if arr.dtype != np.uint8:
    arr = np.clip(arr * (255.0 if arr.max() <= 1.01 else 1.0), 0, 255).astype(np.uint8)
  if arr.shape[-1] == 4:
    arr = arr[..., :3]
  import imageio.v2 as _im
  _im.imwrite(A.still, arr)
  print(f"wrote {A.still}")
  viewer.close()
  raise SystemExit(0)
for k in range(A.steps):
  solver.step(state_in, state_out, control, None, dt)
  state_in, state_out = state_out, state_in
  viewer.begin_frame(t)
  viewer.log_state(state_in)
  viewer.end_frame()
  t += dt
  if k % 400 == 0:
    q = wp.to_torch(state_in.body_q).cpu().numpy()
    print(f"  step {k:6d}  heights {np.round(q[:len(names), 2], 4).tolist()}")
