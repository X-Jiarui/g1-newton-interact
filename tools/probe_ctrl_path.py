"""How do control and state have to be handed to Newton so they survive a step?

SolverMuJoCo.step() calls _apply_mjc_control every step and _update_mjc_data on an interval, so both
ctrl and qpos written straight into mjw_data get overwritten. The rollout therefore has to go through
Newton's own Control and State -- unless update_data_interval=0 pins the state. Both are checked here
rather than assumed, because guessing wrong means a policy driving nothing.
"""
import numpy as np, newton, mujoco, warp as wp
from newton.solvers import SolverMuJoCo

XML = "/home/jiarui/projects/g1-newton-interact/assets/mjlab_scene/scene.xml"

for ctrl_direct in (False, True):
    b = newton.ModelBuilder(); SolverMuJoCo.register_custom_attributes(b)
    b.default_shape_cfg.gap = 0.0
    b.add_mjcf(XML, collapse_fixed_joints=False, parse_mujoco_options=True, ctrl_direct=ctrl_direct)
    m = b.finalize()
    c = m.control()
    s = m.state()
    print(f"\n=== ctrl_direct={ctrl_direct} ===")
    print("  control attrs:", [a for a in dir(c) if not a.startswith("_") and "joint" in a or a == "ctrl"][:8])
    for name in ("ctrl", "joint_target_q", "joint_target_qd", "joint_f"):
        a = getattr(c, name, None)
        print(f"    control.{name}: {None if a is None else getattr(a,'shape',None)}")
    for name in ("joint_q", "joint_qd"):
        a = getattr(s, name, None)
        print(f"    state.{name}: {None if a is None else getattr(a,'shape',None)}")

# does update_data_interval=0 preserve a direct qpos write across a step?
b = newton.ModelBuilder(); SolverMuJoCo.register_custom_attributes(b)
b.default_shape_cfg.gap = 0.0
b.add_mjcf(XML, collapse_fixed_joints=False, parse_mujoco_options=True, ctrl_direct=True)
m = b.finalize()
sv = SolverMuJoCo(m, enable_multiccd=True, update_data_interval=0)
s0, s1, c = m.state(), m.state(), m.control()
contacts = None
q = wp.to_torch(sv.mjw_data.qpos)
q[0, 7] = 0.4242                      # a hinge coordinate, distinctive
before = float(q[0, 7])
sv.step(s0, s1, c, contacts, 0.005)
after = float(wp.to_torch(sv.mjw_data.qpos)[0, 7])
print(f"\nupdate_data_interval=0: qpos[7] {before} -> {after} after one step "
      f"({'PRESERVED (physics moved it)' if abs(after-before) < 0.2 else 'CLOBBERED by state sync'})")

ctrl = wp.to_torch(sv.mjw_data.ctrl)
ctrl[0, 5] = 0.777
cb = float(ctrl[0, 5])
sv.step(s1, s0, c, contacts, 0.005)
ca = float(wp.to_torch(sv.mjw_data.ctrl)[0, 5])
print(f"direct ctrl write: {cb} -> {ca} after step "
      f"({'PRESERVED' if abs(ca-cb) < 1e-6 else 'OVERWRITTEN by _apply_mjc_control'})")
