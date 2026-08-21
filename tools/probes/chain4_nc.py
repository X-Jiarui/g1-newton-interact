"""Locate the nC difference and test whether rebuilding Newton's warp model from a compiled
MjModel removes the constraint-force divergence.

Newton does not compile the MJCF into an MjModel; it reconstructs one from its own Model. That
reconstruction produced nC=1102 where a compiled model gives nC=1087 -- nC is the size of MuJoCo's
compressed mass-matrix layout, so the two are solving with different sparse structures. This first
finds the field responsible, then tries the fix: hand mujoco_warp a properly compiled model (already
verified structurally identical -- same tree, same nM, same body order) and step again.
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
nt = sv.mj_model

print(f"compiled XML   : nM={ref.nM} nC={ref.nC}")
print(f"newton rebuilt : nM={nt.nM} nC={nt.nC}")
for f in ("dof_simplenum", "body_simple", "dof_treeid", "body_treeid", "dof_parentid"):
    a = getattr(ref, f, None); c = getattr(nt, f, None)
    if a is None or c is None:
        print(f"  {f:16s} absent"); continue
    a = np.asarray(a).reshape(-1); c = np.asarray(c).reshape(-1)
    if a.shape != c.shape:
        print(f"  {f:16s} SHAPE {a.shape} vs {c.shape}"); continue
    n = int((a != c).sum())
    print(f"  {f:16s} differing entries: {n}/{len(a)}")
    if n and f in ("body_simple", "dof_simplenum"):
        for i in np.flatnonzero(a != c)[:8]:
            if f == "body_simple":
                nm_ = mujoco.mj_id2name(ref, mujoco.mjtObj.mjOBJ_BODY, int(i)) or f"<{i}>"
            else:
                nm_ = mujoco.mj_id2name(ref, mujoco.mjtObj.mjOBJ_BODY,
                                        int(ref.dof_bodyid[int(i)])) or f"<{i}>"
            print(f"        [{int(i):3d}] {nm_:34s} xml={a[i]} newton={c[i]}")

# ---- the fix: build the warp model from the compiled MjModel -----------------
print("\nrebuilding newton's warp model from the compiled MjModel")
old_nC = sv.mjw_model.nC
sv.mj_model = ref
sv.mjw_model = mjw.put_model(ref)
sv.mjw_data = mjw.put_data(ref, mujoco.MjData(ref), nworld=1, nconmax=256, njmax=2048)
print(f"  nC: {old_nC} -> {sv.mjw_model.nC}   data.M {sv.mjw_data.M.numpy().shape}")

# dof_damping is still the field Newton's own conversion got wrong; re-apply on the new model.
dd = sv.mjw_model.dof_damping
dd.assign(ref.dof_damping.astype(dd.numpy().dtype).reshape(dd.numpy().shape))

# one step from a known state on both a reference mjwarp model and the rebuilt one
ref_wm = mjw.put_model(ref)
ref_wd = mjw.put_data(ref, mujoco.MjData(ref), nworld=1, nconmax=256, njmax=2048)
dd2 = ref_wm.dof_damping
dd2.assign(ref.dof_damping.astype(dd2.numpy().dtype).reshape(dd2.numpy().shape))

rng = np.random.default_rng(0)
q0 = ref.qpos0.copy(); q0[2] += 0.05
for wd in (ref_wd, sv.mjw_data):
    import torch as _t
    wp.to_torch(wd.qpos)[0].copy_(_t.tensor(q0, dtype=_t.float32, device="cuda:0"))
    wp.to_torch(wd.qvel).zero_()
    wp.to_torch(wd.ctrl).zero_()
with wp.ScopedDevice("cuda:0"):
    for _ in range(20):
        mjw.step(ref_wm, ref_wd)
        mjw.step(sv.mjw_model, sv.mjw_data)
a = np.asarray(ref_wd.qpos.numpy()).reshape(-1)
c = np.asarray(sv.mjw_data.qpos.numpy()).reshape(-1)
print(f"\n20 steps, same initial state, reference-mjwarp vs rebuilt-newton-model:")
print(f"  |dqpos|max = {np.abs(a - c).max():.6g}")
print("  (identical means the rebuilt model reproduces stock mujoco_warp exactly)")
