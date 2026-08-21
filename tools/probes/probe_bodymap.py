"""Verify the Newton-body -> MuJoCo-body index offset and the quaternion convention.

body_key is empty, so the mapping cannot be read by name. Newton has 91 bodies and the MjModel 92
(the extra being world), which suggests newton[i] == mj[i+1] -- suggests, not proves. Both the offset
and the quaternion layout are checked against live state before anything is driven from them.
"""
import os, sys, numpy as np, torch, mujoco
sys.path.insert(0, os.path.expanduser("~/projects/g1-newton-interact/src"))
import mjw_compat; mjw_compat.apply()
import newton, warp as wp, mujoco_warp as mjw
from newton.solvers import SolverMuJoCo

XML = os.path.expanduser("~/projects/g1-newton-interact/assets/mjlab_scene/scene.xml")
ref = mujoco.MjModel.from_xml_path(XML)
b = newton.ModelBuilder(); SolverMuJoCo.register_custom_attributes(b)
b.default_shape_cfg.gap = 0.0
b.add_mjcf(XML, collapse_fixed_joints=False, parse_mujoco_options=True)
m = b.finalize()
sv = SolverMuJoCo(m, enable_multiccd=True, update_data_interval=0, njmax=2048, nconmax=256)
sv.mj_model = ref
sv.mjw_model = mjw.put_model(ref)
sv.mjw_data = mjw.put_data(ref, mujoco.MjData(ref), nworld=1, nconmax=256, njmax=2048)

s0, s1, c = m.state(), m.state(), m.control()
qp = wp.to_torch(sv.mjw_data.qpos); qp[0, 2] += 0.08
for _ in range(30):
    sv.step(s0, s1, c, None, 0.005)
    s0, s1 = s1, s0

bq = wp.to_torch(s0.body_q).cpu().numpy()             # (91, 7)
xpos = wp.to_torch(sv.mjw_data.xpos)[0].cpu().numpy()  # (92, 3)
xquat = wp.to_torch(sv.mjw_data.xquat)[0].cpu().numpy()# (92, 4) wxyz

for off in (0, 1):
    n = min(len(bq), len(xpos) - off)
    d = np.abs(bq[:n, 0:3] - xpos[off:off + n]).max()
    print(f"position, newton[i] vs mj[i+{off}]: max|diff| = {d:.6g}")

off = 1
n = min(len(bq), len(xpos) - off)
q_newton = bq[:n, 3:7]
q_mj = xquat[off:off + n]
for name, perm in (("xyzw (newton order)", [1, 2, 3, 0]), ("wxyz (same as mujoco)", [0, 1, 2, 3])):
    conv = q_mj[:, perm]
    d = min(np.abs(q_newton - conv).max(), np.abs(q_newton + conv).max())
    print(f"quaternion as {name}: max|diff| = {d:.6g}")

nm = lambda i: mujoco.mj_id2name(ref, mujoco.mjtObj.mjOBJ_BODY, i) or "?"
print(f"\nspot check with offset 1:")
for i in (0, 88, 89, 90):
    if i < len(bq):
        print(f"  newton[{i}] {np.round(bq[i,0:3],3).tolist()}  <->  mj[{i+1}] {nm(i+1)} "
              f"{np.round(xpos[i+1],3).tolist()}")
