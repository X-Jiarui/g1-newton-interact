"""Why do the apple and table not appear in Newton's render?

The robot draws fine, so the viewer is working and shapes do reach it. The apple (body 88) and table
(bodies 89/90) are in the model -- their body transforms were verified against mjw_data -- so the
question is whether their SHAPES are marked visible.

mjlab puts object colliders in a MuJoCo geom group that its own viewer hides (group 5 by default,
with the visual mesh in group 0). If Newton carries that group across as a visibility flag, the only
shapes the apple has are hidden ones, and the object renders as nothing at all.
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

print(f"shape_count={m.shape_count} body_count={m.body_count}")
cand = [a for a in dir(m) if a.startswith("shape_") and not a.startswith("shape_collision")]
print("shape_* arrays:", cand[:16])

sb = np.asarray(wp.to_torch(m.shape_body).cpu().numpy()).reshape(-1)
flags = None
for fname in ("shape_flags", "shape_visible", "shape_is_visible"):
    a = getattr(m, fname, None)
    if a is not None:
        flags = (fname, np.asarray(wp.to_torch(a).cpu().numpy()).reshape(-1))
        break
print("visibility array:", flags[0] if flags else "none found")

# mjlab geom groups, for comparison
gname = lambda i: mujoco.mj_id2name(ref, mujoco.mjtObj.mjOBJ_GEOM, i) or "?"
bname = lambda i: mujoco.mj_id2name(ref, mujoco.mjtObj.mjOBJ_BODY, i) or "?"
print("\nmjlab collidable geoms on apple/table, with their group:")
for g in range(ref.ngeom):
    bn = bname(int(ref.geom_bodyid[g]))
    if "apple" in bn or "table" in bn:
        coll = int(ref.geom_contype[g]) != 0 or int(ref.geom_conaffinity[g]) != 0
        print(f"   geom {g:3d} {gname(g):28s} body={bn:18s} group={int(ref.geom_group[g])} "
              f"collidable={coll}")

print("\nnewton shapes on those bodies (newton body = mj body - 1):")
for nb in (87, 88, 89, 90):
    idx = np.flatnonzero(sb == nb)
    fl = "" if flags is None else f" {flags[0]}={[int(flags[1][i]) for i in idx]}"
    print(f"   newton body {nb} (mj {nb+1} {bname(nb+1)}): {len(idx)} shape(s){fl}")
