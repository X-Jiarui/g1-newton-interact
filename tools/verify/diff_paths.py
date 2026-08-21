"""What still differs between the native+fix model and the separately compiled one?

nC now matches, yet the native path dropped the object in 2 of 7 runs where the compiled path held
in 12 of 12. That gap is not explained by the mass-matrix layout any more, so every field the two
warp models expose is compared -- with attention to the ones contact actually depends on.
"""
import os, sys, numpy as np, mujoco
sys.path.insert(0, os.path.expanduser("~/projects/g1-newton-interact/src"))
import mjw_compat; mjw_compat.apply()
import newton, warp as wp, mujoco_warp as mjw
from newton.solvers import SolverMuJoCo
from newton_simple_fix import capture_spec, restore_simple_bodies

XML = os.path.expanduser("~/projects/g1-newton-interact/assets/mjlab_scene/scene.xml")
ref = mujoco.MjModel.from_xml_path(XML)

def build_native():
    with capture_spec() as cap:
        b = newton.ModelBuilder(); SolverMuJoCo.register_custom_attributes(b)
        b.default_shape_cfg.gap = 0.0
        b.add_mjcf(XML, collapse_fixed_joints=False, parse_mujoco_options=True)
        sv = SolverMuJoCo(b.finalize(), enable_multiccd=True, update_data_interval=0,
                          njmax=2048, nconmax=256)
    restore_simple_bodies(sv, cap.spec, verbose=False)
    for h in (sv.mj_model, sv.mjw_model):
        dd = getattr(h, "dof_damping", None)
        if dd is None: continue
        if hasattr(dd, "assign"):
            dd.assign(ref.dof_damping.astype(dd.numpy().dtype).reshape(dd.numpy().shape))
        else:
            dd[:] = ref.dof_damping
    return sv

sv = build_native()
wm_native = sv.mjw_model
wm_ref = mjw.put_model(ref)
print(f"native+fix nC={int(wm_native.nC)}  compiled nC={int(wm_ref.nC)}")

# geom ordering differs between the two models, so per-geom arrays are compared by SORTED value:
# a difference in the multiset means a real parameter difference, not just a permutation.
CONTACT = ["geom_solref", "geom_solimp", "geom_friction", "geom_margin", "geom_gap",
           "geom_condim", "geom_priority", "geom_solmix", "dof_damping", "dof_armature",
           "dof_frictionloss", "body_mass", "body_inertia", "opt"]
print(f"\n{'field':22s} {'native':>10} {'compiled':>10}   verdict")
print("-" * 60)
for f in CONTACT:
    if f == "opt":
        continue
    A = getattr(wm_native, f, None); B = getattr(wm_ref, f, None)
    if A is None or B is None:
        print(f"{f:22s} {'absent' if A is None else 'ok':>10} {'absent' if B is None else 'ok':>10}")
        continue
    a = np.asarray(A.numpy()).reshape(-1).astype(np.float64)
    b_ = np.asarray(B.numpy()).reshape(-1).astype(np.float64)
    if a.shape != b_.shape:
        print(f"{f:22s} {a.shape!s:>10} {b_.shape!s:>10}   SHAPE")
        continue
    d_direct = np.abs(a - b_).max()
    d_sorted = np.abs(np.sort(a) - np.sort(b_)).max()
    verdict = "same" if d_sorted < 1e-5 else "VALUES DIFFER"
    extra = "" if d_direct < 1e-5 else f"  (as-ordered {d_direct:.3g}, sorted {d_sorted:.3g})"
    print(f"{f:22s} {'':>10} {'':>10}   {verdict}{extra}")

print("\nopt fields:")
for n in sorted(set(dir(wm_native.opt)) & set(dir(wm_ref.opt))):
    if n.startswith("_"): continue
    try:
        x = getattr(wm_native.opt, n); y = getattr(wm_ref.opt, n)
        xv = np.asarray(x.numpy() if hasattr(x, "numpy") else x).reshape(-1)
        yv = np.asarray(y.numpy() if hasattr(y, "numpy") else y).reshape(-1)
        if xv.shape == yv.shape and not np.allclose(xv.astype(np.float64), yv.astype(np.float64)):
            print(f"   {n:24s} native={xv[:3].tolist()} compiled={yv[:3].tolist()}")
    except Exception:
        pass
