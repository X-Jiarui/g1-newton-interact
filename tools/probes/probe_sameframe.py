"""Which bodies lose 'simple', and what exactly differs about their inertial frame?

MuJoCo marks a body simple when its inertial frame coincides with its body frame (sameframe) and the
joint structure allows the shortcut. Newton's spec compiles to body_simple=2 where the source MJCF
gives 4, before any post-processing -- so the difference is in what the spec says about those bodies'
inertial frames, not in anything done afterwards.
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
sv = SolverMuJoCo(b.finalize(), enable_multiccd=True, update_data_interval=0, njmax=2048, nconmax=256)
mm = sv.mj_model
nm = lambda i: mujoco.mj_id2name(ref, mujoco.mjtObj.mjOBJ_BODY, i) or f"<{i}>"

r_simple = np.flatnonzero(ref.body_simple != 0)
n_simple = np.flatnonzero(mm.body_simple != 0)
print(f"reference simple bodies: {[nm(i) for i in r_simple]}")
print(f"newton    simple bodies: {[nm(i) for i in n_simple]}")
lost = sorted(set(r_simple.tolist()) - set(n_simple.tolist()))
print(f"lost: {[nm(i) for i in lost]}\n")

for i in lost:
    print(f"--- {nm(i)} (body {i}) ---")
    for f in ("body_sameframe", "body_ipos", "body_iquat", "body_inertia", "body_mass",
              "body_pos", "body_quat"):
        a = np.asarray(getattr(ref, f)[i]).reshape(-1)
        c = np.asarray(getattr(mm, f)[i]).reshape(-1)
        same = np.allclose(a, c, rtol=0, atol=0)
        mark = "" if same else "   <-- DIFFERS"
        print(f"   {f:16s} ref={np.round(a,9).tolist()}  newton={np.round(c,9).tolist()}{mark}")
    jadr = int(ref.body_jntadr[i]); jnum = int(ref.body_jntnum[i])
    jadr2 = int(mm.body_jntadr[i]); jnum2 = int(mm.body_jntnum[i])
    print(f"   joints           ref: adr={jadr} num={jnum} type={[int(ref.jnt_type[jadr+k]) for k in range(jnum)]}")
    print(f"                 newton: adr={jadr2} num={jnum2} type={[int(mm.jnt_type[jadr2+k]) for k in range(jnum2)]}")
    print()
