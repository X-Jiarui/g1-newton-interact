"""Can the simple-body metadata be restored locally, without replacing Newton's model?

Newton compiles the MjSpec correctly (MuJoCo's compiler derives body_simple itself), then
_sync_mjw_inertias_to_mjc_cpu() invalidates it whenever a body's inertia or inertial frame differs
between the CPU model and the warp model:

    changed = ~isclose(mj_model.body_inertia, mjw_model.body_inertia, rtol=1e-6, atol=1e-8)
    body_simple[changed] = 0
    dof_simplenum[:] = 0        <- global

The warp model is float32 and the CPU model float64, so the comparison can trip on round-trip error
alone; and one body anywhere disables the simple-dof path everywhere. This measures how many bodies
are actually flagged, how large the differences are, and whether restoring the flags and re-running
mj_setConst brings nC back.
"""
import os, sys, numpy as np, mujoco
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
import mjw_compat; mjw_compat.apply()
import newton, warp as wp, mujoco_warp as mjw
from newton.solvers import SolverMuJoCo

XML = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets/mjlab_scene/scene.xml")
ref = mujoco.MjModel.from_xml_path(XML)
b = newton.ModelBuilder(); SolverMuJoCo.register_custom_attributes(b)
b.default_shape_cfg.gap = 0.0
b.add_mjcf(XML, collapse_fixed_joints=False, parse_mujoco_options=True)
m = b.finalize()
sv = SolverMuJoCo(m, enable_multiccd=True, update_data_interval=0, njmax=2048, nconmax=256)
mm = sv.mj_model

print(f"after construction:  nC={mm.nC}  (compiled reference {ref.nC})")
print(f"  body_simple nonzero: {int((mm.body_simple != 0).sum())}/{mm.nbody} "
      f"(reference {int((ref.body_simple != 0).sum())})")
print(f"  dof_simplenum nonzero: {int((mm.dof_simplenum != 0).sum())}/{mm.nv} "
      f"(reference {int((ref.dof_simplenum != 0).sum())})")

wi = sv.mjw_model.body_inertia.numpy()[0]
wq = sv.mjw_model.body_iquat.numpy()[0]
inert_bad = ~np.isclose(mm.body_inertia, wi, rtol=1e-6, atol=1e-8).all(axis=1)
quat_bad = ~np.isclose(mm.body_iquat, wq, rtol=1e-6, atol=1e-8).all(axis=1)
changed = inert_bad | quat_bad
nm = lambda i: mujoco.mj_id2name(ref, mujoco.mjtObj.mjOBJ_BODY, i) or f"<{i}>"
print(f"\nbodies the sync would flag as changed: {int(changed.sum())}/{mm.nbody}"
      f"   (inertia {int(inert_bad.sum())}, iquat {int(quat_bad.sum())})")
for i in np.flatnonzero(changed)[:6]:
    d = np.abs(mm.body_inertia[i] - wi[i]).max()
    print(f"   {nm(i):26s} max|d inertia|={d:.3g}  cpu={np.round(mm.body_inertia[i],6).tolist()}")
    print(f"   {'':26s} {'':22s}  warp={np.round(wi[i],6).tolist()}")

# --- attempt the local repair -------------------------------------------------
print("\n--- restoring the compiled metadata and re-deriving constants ---")
mm.body_simple[:] = ref.body_simple
mm.dof_simplenum[:] = ref.dof_simplenum
mm.body_sameframe[:] = ref.body_sameframe
try:
    mujoco.mj_setConst(mm, mujoco.MjData(mm))
    print(f"  after mj_setConst: nC={mm.nC}  body_simple nonzero="
          f"{int((mm.body_simple != 0).sum())}  dof_simplenum nonzero="
          f"{int((mm.dof_simplenum != 0).sum())}")
except Exception as e:
    print(f"  mj_setConst failed: {type(e).__name__}: {e}")
wm = mjw.put_model(mm)
print(f"  put_model -> mjw_model.nC = {int(wm.nC)}   (target {ref.nC})")
print("LOCAL_FIX_WORKS" if int(wm.nC) == int(ref.nC) else "LOCAL_FIX_INSUFFICIENT")
