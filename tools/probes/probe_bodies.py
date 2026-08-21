"""Map Newton Model bodies onto the compiled MjModel's, so the render can be driven from mjw_data.

Newton's viewer draws State.body_q. Mocap-driven bodies (the table) never enter State, so they render
at the origin; and any body whose pose the solver does not pull back would be equally wrong. Syncing
every body transform from mjw_data.xpos/xquat before log_state makes the picture match the simulation
by construction -- but that needs an index map, and Newton renames bodies to a flattened path.
"""
import os, sys, numpy as np, mujoco
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
import mjw_compat; mjw_compat.apply()
import newton, warp as wp
from newton.solvers import SolverMuJoCo

XML = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets/mjlab_scene/scene.xml")
ref = mujoco.MjModel.from_xml_path(XML)
b = newton.ModelBuilder(); SolverMuJoCo.register_custom_attributes(b)
b.default_shape_cfg.gap = 0.0
b.add_mjcf(XML, collapse_fixed_joints=False, parse_mujoco_options=True)
m = b.finalize()

keys = list(getattr(m, "body_key", []) or [])
print(f"newton body_count={m.body_count}  keys={len(keys)}")
print(f"mjmodel nbody={ref.nbody}")
print("first 3 newton keys:", keys[:3])
print("last 3 newton keys:", keys[-3:])

mj_names = [mujoco.mj_id2name(ref, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(ref.nbody)]
flat = sorted([(n.replace("/", "_"), n) for n in mj_names if n], key=lambda t: -len(t[0]))
def canon(n):
    for f, r in flat:
        if n == r or n.endswith(f):
            return r
    return None

mapped, unmapped = [], []
for i, k in enumerate(keys):
    c = canon(k)
    (mapped if c else unmapped).append((i, k, c))
print(f"\nmapped {len(mapped)}/{len(keys)}   unmapped {len(unmapped)}")
for i, k, c in unmapped[:5]:
    print(f"   unmapped newton body {i}: {k[:70]}")
tbl = [(i, c) for i, k, c in mapped if c and "table" in c]
app = [(i, c) for i, k, c in mapped if c and "apple" in c]
print(f"table bodies: {tbl}")
print(f"apple bodies: {app}")
st = m.state()
print(f"state.body_q shape: {tuple(wp.to_torch(st.body_q).shape)}")
