"""Is the table still a mocap body after Newton's conversion, and is it where mjlab puts it?

The apple starts correctly at z=0.707 and then falls to the floor, which is what happens when nothing
supports it. mjlab drives the table by writing mocap_pos/mocap_quat every step. Newton's docs say
FIXED joints attached to the world become "nested bodies or mocap bodies" -- if the table came across
as something else, the mocap write lands nowhere and the table never moves into place.
"""
import numpy as np, torch, newton, mujoco, warp as wp
from newton.solvers import SolverMuJoCo

XML = "/home/jiarui/projects/g1-newton-interact/assets/mjlab_scene/scene.xml"
ref = mujoco.MjModel.from_xml_path(XML)
print(f"reference: nmocap={ref.nmocap}")
for i in range(ref.nbody):
    n = mujoco.mj_id2name(ref, mujoco.mjtObj.mjOBJ_BODY, i) or ""
    if "table" in n:
        print(f"   {n!r}: mocapid={int(ref.body_mocapid[i])} pos={ref.body_pos[i]}")

b = newton.ModelBuilder(); SolverMuJoCo.register_custom_attributes(b)
b.default_shape_cfg.gap = 0.0
b.add_mjcf(XML, collapse_fixed_joints=False, parse_mujoco_options=True)
m = b.finalize()
sv = SolverMuJoCo(m, enable_multiccd=True, update_data_interval=0)
M = sv.mj_model
print(f"\nnewton converted: nmocap={M.nmocap}")
tb = []
for i in range(M.nbody):
    n = mujoco.mj_id2name(M, mujoco.mjtObj.mjOBJ_BODY, i) or ""
    if "table" in n:
        tb.append((i, n, int(M.body_mocapid[i])))
for i, n, mid in tb:
    print(f"   body {i} mocapid={mid}  {n[:70]}")

# where do the table geoms sit right now, and do they collide?
print("\ntable geoms in the converted model:")
for g in range(M.ngeom):
    bid = int(M.geom_bodyid[g])
    n = mujoco.mj_id2name(M, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
    if "table" in n:
        print(f"   geom {g} type={int(M.geom_type[g])} contype={int(M.geom_contype[g])} "
              f"conaffinity={int(M.geom_conaffinity[g])} size={M.geom_size[g]} bodypos={M.body_pos[bid]}")
xpos = wp.to_torch(sv.mjw_data.xpos)[0].cpu().numpy()
for i, n, mid in tb:
    print(f"   live xpos[{i}] = {xpos[i]}")
