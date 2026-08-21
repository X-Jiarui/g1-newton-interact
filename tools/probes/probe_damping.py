"""Where does free-joint damping go on the native path, and can it be set without a compiled model?

The native path currently repairs dof_damping by copying from a separately compiled MjModel, which is
the last real dependency on MuJoCo's compiler. If ModelBuilder.joint_damping carries the MJCF value
-- or can be set before finalize -- the dependency disappears.
"""
import os, sys, numpy as np, mujoco
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
import mjw_compat; mjw_compat.apply()
import newton, warp as wp
from newton.solvers import SolverMuJoCo

XML = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets/mjlab_scene/scene.xml")
ref = mujoco.MjModel.from_xml_path(XML)
print(f"mjlab dof_damping: nonzero {int((ref.dof_damping != 0).sum())}/{ref.nv}")
print(f"  robot free joint dofs [0:6]: {np.round(ref.dof_damping[:6], 6).tolist()}")
print(f"  a hinge dof [6:9]:           {np.round(ref.dof_damping[6:9], 6).tolist()}")
print(f"  apple free joint dofs [-6:]: {np.round(ref.dof_damping[-6:], 6).tolist()}")

b = newton.ModelBuilder(); SolverMuJoCo.register_custom_attributes(b)
b.default_shape_cfg.gap = 0.0
b.add_mjcf(XML, collapse_fixed_joints=False, parse_mujoco_options=True)
jd = np.asarray(b.joint_damping, dtype=np.float64)
print(f"\nbuilder.joint_damping: len={len(jd)} nonzero={int((jd != 0).sum())}")
print(f"  first 12: {np.round(jd[:12], 6).tolist()}")
print(f"  last 8:   {np.round(jd[-8:], 6).tolist()}")

# does what the builder holds survive finalize + solver construction?
m = b.finalize()
sv = SolverMuJoCo(m, enable_multiccd=True, update_data_interval=0, njmax=2048, nconmax=256)
got = np.asarray(sv.mj_model.dof_damping, dtype=np.float64)
print(f"\nafter solver: dof_damping nonzero={int((got != 0).sum())}/{len(got)}")
print(f"  matches mjlab: {np.allclose(got, ref.dof_damping, rtol=1e-5, atol=1e-8)}")
bad = np.flatnonzero(np.abs(got - ref.dof_damping) > 1e-8)
print(f"  differing dofs: {len(bad)}  first few {bad[:8].tolist()}")
for i in bad[:6]:
    j = int(ref.dof_jntid[i])
    jn = mujoco.mj_id2name(ref, mujoco.mjtObj.mjOBJ_JOINT, j) or "?"
    print(f"     dof {int(i):3d} joint={jn[:38]:38s} type={int(ref.jnt_type[j])} "
          f"mjlab={ref.dof_damping[i]:.5f} newton={got[i]:.5f}")

# can it simply be set on the builder before finalize?
b2 = newton.ModelBuilder(); SolverMuJoCo.register_custom_attributes(b2)
b2.default_shape_cfg.gap = 0.0
b2.add_mjcf(XML, collapse_fixed_joints=False, parse_mujoco_options=True)
jd2 = list(b2.joint_damping)
for i in range(min(6, len(jd2))):
    jd2[i] = 7.77                      # distinctive marker on the robot's free-joint dofs
b2.joint_damping = jd2
sv2 = SolverMuJoCo(b2.finalize(), enable_multiccd=True, update_data_interval=0,
                   njmax=2048, nconmax=256)
got2 = np.asarray(sv2.mj_model.dof_damping, dtype=np.float64)
print(f"\nmarker 7.77 written to builder.joint_damping[:6] -> present in mj_model: "
      f"{bool((np.abs(got2 - 7.77) < 1e-6).any())}  (values {np.round(got2[:6],3).tolist()})")
