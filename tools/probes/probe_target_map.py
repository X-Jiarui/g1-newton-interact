"""Which MuJoCo actuator does Newton's control.joint_target_q[dof] actually drive?

The control path has to be joint_target_q -- control.ctrl does not exist and direct ctrl writes are
overwritten every step. But mjlab indexes its 69 position targets by ITS joint order, and Newton's
joint_target_q is DOF-indexed over 81 entries including two free joints. Getting that mapping wrong
routes every command to the wrong joint and raises nothing, which is the single failure mode this
port has to be built against.

So it is measured: write a unique value into every DOF slot, step once, and read back which actuator
slot in mjw_data.ctrl received it.
"""
import numpy as np, torch, newton, mujoco, warp as wp
from newton.solvers import SolverMuJoCo

XML = "/home/jiarui/projects/g1-newton-interact/assets/mjlab_scene/scene.xml"
b = newton.ModelBuilder(); SolverMuJoCo.register_custom_attributes(b)
b.default_shape_cfg.gap = 0.0
b.add_mjcf(XML, collapse_fixed_joints=False, parse_mujoco_options=True)
m = b.finalize()
sv = SolverMuJoCo(m, enable_multiccd=True, update_data_interval=0)
s0, s1, c = m.state(), m.state(), m.control()

nv = sv.mj_model.nv
mj_ns = getattr(c, "mujoco", None)
print("control.mujoco present:", mj_ns is not None,
      "attrs:", [a for a in dir(mj_ns) if not a.startswith("_")][:8] if mj_ns else None)
tq = wp.to_torch(mj_ns.ctrl)
print("control.mujoco.ctrl shape:", tuple(tq.shape))
# unique per DOF, small enough to be a harmless target
n = tq.view(-1).shape[0]
probe = torch.arange(1, n + 1, dtype=tq.dtype, device=tq.device) * 0.001
tq.view(-1)[:] = probe
sv.step(s0, s1, c, None, 0.005)
ctrl = wp.to_torch(sv.mjw_data.ctrl).cpu().numpy().reshape(-1)

M = sv.mj_model
name = lambda t, i: mujoco.mj_id2name(M, t, i) or f"<{i}>"
hits, misses = [], []
for a in range(M.nu):
    v = ctrl[a]
    if abs(v) < 1e-9:
        misses.append(a); continue
    dof = int(round(v / 0.001)) - 1   # index into control.mujoco.ctrl
    jid = int(M.actuator_trnid[a][0])
    hits.append((a, dof, name(mujoco.mjtObj.mjOBJ_ACTUATOR, a), name(mujoco.mjtObj.mjOBJ_JOINT, jid),
                 a))

print(f"actuators receiving a target: {len(hits)} of {M.nu}   silent: {len(misses)}")
ok = sum(1 for _, dof, _, _, slot in hits if dof == slot)
print(f"of those, ctrl slot index == control.mujoco.ctrl index: {ok}/{len(hits)}  "
      f"(identity mapping means mjlab's ctrl indices carry over unchanged)")
for a, dof, an, jn, slot in hits[:6]:
    flag = "OK" if dof == slot else "MISMATCH"
    print(f"  act[{a:3d}] {an:40s} <- ctrl[{dof:3d}]  joint={jn}  {flag}")
if misses:
    print(f"  first silent actuators: {[name(mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in misses[:4]]}")
