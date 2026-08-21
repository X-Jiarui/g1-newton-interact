"""Did the collider swap leave the object with a valid mass and inertia?

Every solver setting produces NaN on the first step, which rules out convergence and points at the
model. The swap disables the authored sphere's collision flags and adds a mesh shape to the same
body -- and Newton derives body mass and inertia from its shapes, so a disabled-but-present sphere,
a mesh with a density that does not match, or a degenerate inertia would all show up here. A zero or
negative mass divides by zero the moment the solver runs.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, mujoco, torch

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--xml", default=os.path.join(HERE, "assets/scene_stapler/scene.xml"))
ap.add_argument("--sdf-object", default=None)
A = ap.parse_args()

sys.path.insert(0, os.path.join(HERE, "src"))
import mjw_compat; mjw_compat.apply()
import newton, warp as wp
from newton.solvers import SolverMuJoCo
from newton_simple_fix import capture_spec, restore_simple_bodies, restore_freejoint_damping
from grab_objects import swap_collider_to_sdf

ref = mujoco.MjModel.from_xml_path(A.xml)
names = [mujoco.mj_id2name(ref, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(ref.nbody)]
ai = names.index("apple/apple")
print(f"authored (mjlab): apple mass={ref.body_mass[ai]:.6f} kg  "
      f"inertia={np.round(ref.body_inertia[ai], 8).tolist()}  ipos={np.round(ref.body_ipos[ai],5).tolist()}")

for label, stl in (("no swap (sphere)", None), ("after SDF swap", A.sdf_object)):
  if label.startswith("after") and not stl:
    continue
  b = newton.ModelBuilder(); SolverMuJoCo.register_custom_attributes(b)
  b.default_shape_cfg.gap = 0.0
  b.add_mjcf(A.xml, collapse_fixed_joints=False, parse_mujoco_options=True)
  if stl:
    swap_collider_to_sdf(b, ref, "apple/apple", stl, resolution=128, verbose=False)
  m = b.finalize()
  with capture_spec() as cap:
    sv = SolverMuJoCo(m, enable_multiccd=True, update_data_interval=0, njmax=2048, nconmax=256)
  restore_freejoint_damping(cap.spec, A.xml, verbose=False)
  restore_simple_bodies(sv, cap.spec, verbose=False)
  mm = sv.mj_model
  print(f"\n=== {label} ===")
  print(f"  apple mass    = {mm.body_mass[ai]:.6f} kg")
  print(f"  apple inertia = {np.round(mm.body_inertia[ai], 8).tolist()}")
  print(f"  apple ipos    = {np.round(mm.body_ipos[ai], 5).tolist()}")
  bad_m = np.flatnonzero((mm.body_mass <= 0) & (np.arange(mm.nbody) > 0))
  bad_i = np.flatnonzero((mm.body_inertia <= 0).any(axis=1) & (np.arange(mm.nbody) > 0))
  print(f"  bodies with mass<=0:    {len(bad_m)}  {[names[i] for i in bad_m[:5]]}")
  print(f"  bodies with inertia<=0: {len(bad_i)}  {[names[i] for i in bad_i[:5]]}")
  print(f"  any non-finite in mass/inertia: "
        f"{bool(~np.isfinite(mm.body_mass).all() or ~np.isfinite(mm.body_inertia).all())}")
  print(f"  qpos0 finite: {bool(np.isfinite(mm.qpos0).all())}   nq={mm.nq}")
