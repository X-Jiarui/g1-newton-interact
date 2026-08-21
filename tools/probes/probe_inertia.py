"""Is Newton's body inertia diagonal, or diagonal-plus-round-off?

body_simple requires the body frame and the inertial frame to coincide. Newton stores inertia as a
full 3x3 tensor and decomposes it into principal axes when it builds the MjSpec. If the tensor is
diagonal only up to float round-off, that decomposition returns an arbitrary small rotation, iquat
stops being identity, and MuJoCo's compiler declines to mark the body simple -- which is exactly the
60-body iquat mismatch measured earlier.
"""
import os, sys, numpy as np, mujoco
sys.path.insert(0, os.path.expanduser("~/projects/g1-newton-interact/src"))
import mjw_compat; mjw_compat.apply()
import newton, warp as wp
from newton.solvers import SolverMuJoCo

XML = os.path.expanduser("~/projects/g1-newton-interact/assets/mjlab_scene/scene.xml")
ref = mujoco.MjModel.from_xml_path(XML)
b = newton.ModelBuilder(); SolverMuJoCo.register_custom_attributes(b)
b.default_shape_cfg.gap = 0.0
b.add_mjcf(XML, collapse_fixed_joints=False, parse_mujoco_options=True)
m = b.finalize()

I = wp.to_torch(m.body_inertia).cpu().numpy()
print(f"newton body_inertia shape={I.shape} dtype={I.dtype}")
nm = lambda i: mujoco.mj_id2name(ref, mujoco.mjtObj.mjOBJ_BODY, i) or f"<{i}>"

off = np.abs(I - np.stack([np.diag(np.diag(x)) for x in I])).reshape(len(I), -1).max(axis=1)
diag = np.abs(np.stack([np.diag(x) for x in I])).max(axis=1)
rel = np.divide(off, np.maximum(diag, 1e-12))
print(f"\nbodies whose inertia tensor is exactly diagonal: {int((off == 0).sum())}/{len(I)}")
print(f"bodies with off-diagonal terms: {int((off > 0).sum())}")
print(f"  relative off-diagonal magnitude: max={rel.max():.3g} median={np.median(rel[rel>0]) if (rel>0).any() else 0:.3g}")

for label, idx in (("apple", 88), ("table", 90)):
    print(f"\n{label} (newton body {idx} = mj {nm(idx+1)}):")
    print(f"  inertia tensor:\n{np.array2string(I[idx], precision=9, suppress_small=False)}")
    print(f"  off-diagonal max = {off[idx]:.6g}   relative = {rel[idx]:.6g}")
    print(f"  reference iquat  = {np.round(ref.body_iquat[idx+1], 6).tolist()}")
    print(f"  reference simple = {int(ref.body_simple[idx+1])}")

worst = np.argsort(-rel)[:5]
print("\nlargest relative off-diagonal:")
for i in worst:
    print(f"   newton {i:3d} {nm(i+1):26s} rel={rel[i]:.3g} off={off[i]:.3g}")
