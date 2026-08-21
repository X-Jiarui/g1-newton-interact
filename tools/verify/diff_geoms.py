"""Do Newton's 81 collision geoms carry the same contact parameters as mjlab's?

The earlier comparison printed SHAPE and moved on: the native model has 81 geoms, the compiled one
226 (the extra 145 being visual-only). So the contact parameters of the geoms that actually collide
were never compared -- and those are exactly what decides whether a grasp holds.

Geoms are matched by (owning body, type, size) since Newton renames everything and drops the visual
ones, then every contact-relevant field is compared per geom.
"""
import os, sys, numpy as np, mujoco
sys.path.insert(0, os.path.expanduser("~/projects/g1-newton-interact/src"))
import mjw_compat; mjw_compat.apply()
import newton, warp as wp
from newton.solvers import SolverMuJoCo
from newton_simple_fix import capture_spec, restore_simple_bodies

XML = os.path.expanduser("~/projects/g1-newton-interact/assets/mjlab_scene/scene.xml")
ref = mujoco.MjModel.from_xml_path(XML)

with capture_spec() as cap:
    b = newton.ModelBuilder(); SolverMuJoCo.register_custom_attributes(b)
    b.default_shape_cfg.gap = 0.0
    b.add_mjcf(XML, collapse_fixed_joints=False, parse_mujoco_options=True)
    sv = SolverMuJoCo(b.finalize(), enable_multiccd=True, update_data_interval=0,
                      njmax=2048, nconmax=256)
restore_simple_bodies(sv, cap.spec, verbose=False)
nt = sv.mj_model

bn_r = [mujoco.mj_id2name(ref, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(ref.nbody)]
bn_n = [mujoco.mj_id2name(nt, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(nt.nbody)]
flat = sorted([(r.replace("/", "_"), r) for r in bn_r if r], key=lambda t: -len(t[0]))
def canon(n):
    for f, r in flat:
        if n == r or n.endswith(f):
            return r
    return n

def collidable(M):
    return [i for i in range(M.ngeom)
            if int(M.geom_contype[i]) or int(M.geom_conaffinity[i])]

cr, cn = collidable(ref), collidable(nt)
print(f"collidable geoms: mjlab={len(cr)}  newton={len(cn)}")

# key each geom by (body name, type, rounded size) -- stable across renaming and reordering
def key(M, g, canonical):
    bnm = canonical(mujoco.mj_id2name(M, mujoco.mjtObj.mjOBJ_BODY, int(M.geom_bodyid[g])) or "")
    return (bnm, int(M.geom_type[g]), tuple(np.round(M.geom_size[g], 6)))

kr = {}
for g in cr:
    kr.setdefault(key(ref, g, lambda x: x), []).append(g)
matched, unmatched = [], []
for g in cn:
    k = key(nt, g, canon)
    if k in kr and kr[k]:
        matched.append((kr[k].pop(0), g))
    else:
        unmatched.append(g)
print(f"matched {len(matched)}/{len(cn)}   unmatched {len(unmatched)}")
if unmatched:
    # An unmatched geom is not necessarily a difference -- several identical geoms on one body match
    # in any order -- but it is unverified, and unverified contact parameters are the thing this
    # whole comparison exists to rule out.
    from collections import Counter
    cc = Counter()
    for g in unmatched:
        bnm = canon(mujoco.mj_id2name(nt, mujoco.mjtObj.mjOBJ_BODY, int(nt.geom_bodyid[g])) or "?")
        cc[(bnm, int(nt.geom_type[g]))] += 1
    print("  unmatched newton geoms by (body, type):")
    for (bnm, t), n in cc.most_common(10):
        print(f"     {bnm:34s} type={t}  x{n}")
    leftover = [g for k, v in kr.items() for g in v]
    cc2 = Counter()
    for g in leftover:
        bnm = mujoco.mj_id2name(ref, mujoco.mjtObj.mjOBJ_BODY, int(ref.geom_bodyid[g])) or "?"
        cc2[(bnm, int(ref.geom_type[g]))] += 1
    print("  leftover mjlab geoms by (body, type):")
    for (bnm, t), n in cc2.most_common(10):
        print(f"     {bnm:34s} type={t}  x{n}")

# The unmatched geoms are the same bodies on both sides, so they are the same geoms with different
# SIZES -- and one of them is the apple, the object being grasped. Compare their sizes directly.
print("\nsizes of the geoms that failed to match, by body:")
for bnm_want in ("apple/apple", "robot/left_ankle_roll_link", "robot/left_shoulder_pitch_link", "world"):
    rs = [np.round(ref.geom_size[g], 6).tolist() for g in cr
          if (mujoco.mj_id2name(ref, mujoco.mjtObj.mjOBJ_BODY, int(ref.geom_bodyid[g])) or "") == bnm_want]
    ns = [np.round(nt.geom_size[g], 6).tolist() for g in cn
          if canon(mujoco.mj_id2name(nt, mujoco.mjtObj.mjOBJ_BODY, int(nt.geom_bodyid[g])) or "") == bnm_want]
    print(f"   {bnm_want:32s}")
    print(f"      mjlab : {rs}")
    print(f"      newton: {ns}")

FIELDS = ["geom_solref", "geom_solimp", "geom_friction", "geom_margin", "geom_gap",
          "geom_condim", "geom_priority", "geom_solmix", "geom_contype", "geom_conaffinity"]
print(f"\n{'field':20s} {'geoms differing':>16}   worst")
print("-" * 62)
for f in FIELDS:
    A, B = getattr(ref, f), getattr(nt, f)
    n_bad, worst, example = 0, 0.0, None
    for gi, gj in matched:
        a = np.atleast_1d(A[gi]).astype(np.float64)
        c = np.atleast_1d(B[gj]).astype(np.float64)
        d = np.abs(a - c).max()
        if d > 1e-6:
            n_bad += 1
            if d > worst:
                worst, example = d, (gi, gj, a, c)
    line = f"{f:20s} {n_bad:>10}/{len(matched)}"
    if example:
        gi, gj, a, c = example
        bnm = mujoco.mj_id2name(ref, mujoco.mjtObj.mjOBJ_BODY, int(ref.geom_bodyid[gi])) or "?"
        line += f"   {worst:.4g}  e.g. {bnm}: mjlab={np.round(a,5).tolist()} newton={np.round(c,5).tolist()}"
    print(line)
