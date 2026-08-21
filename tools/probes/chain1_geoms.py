"""Chain step 0: does Newton have every collidable geom mjlab has?

mjlab compiles 226 geoms, Newton's conversion keeps 81. That gap has been assumed to be visual-only
geometry, never checked. If even one geom with contype/conaffinity set was dropped, Newton is missing
a collider -- and a missing fingertip collider produces exactly the observed symptom: the hand
arrives at the object and no contact ever forms.

Collidability is what matters, not the count: a geom with contype=0 AND conaffinity=0 can never
generate a contact, so dropping it changes nothing.
"""
import os, sys, numpy as np, mujoco
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
import mjw_compat; mjw_compat.apply()
import newton
from newton.solvers import SolverMuJoCo

XML = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets/mjlab_scene/scene.xml")
ref = mujoco.MjModel.from_xml_path(XML)

b = newton.ModelBuilder(); SolverMuJoCo.register_custom_attributes(b)
b.default_shape_cfg.gap = 0.0
b.add_mjcf(XML, collapse_fixed_joints=False, parse_mujoco_options=True)
m = b.finalize()
sv = SolverMuJoCo(m, enable_multiccd=True, update_data_interval=0, njmax=2048, nconmax=256)
nt = sv.mj_model

def collidable(M):
    return [i for i in range(M.ngeom)
            if int(M.geom_contype[i]) != 0 or int(M.geom_conaffinity[i]) != 0]

cr, cn = collidable(ref), collidable(nt)
print(f"mjlab : ngeom={ref.ngeom}  collidable={len(cr)}")
print(f"newton: ngeom={nt.ngeom}  collidable={len(cn)}")

def bodyname(M, g):
    return mujoco.mj_id2name(M, mujoco.mjtObj.mjOBJ_BODY, int(M.geom_bodyid[g])) or "?"

# Group by owning body, mapping Newton's flattened names back by longest suffix.
flat = sorted(((mujoco.mj_id2name(ref, mujoco.mjtObj.mjOBJ_BODY, i) or "").replace("/", "_"),
               mujoco.mj_id2name(ref, mujoco.mjtObj.mjOBJ_BODY, i) or "")
              for i in range(ref.nbody))
flat = sorted([f for f in flat if f[0]], key=lambda t: -len(t[0]))
def canon(n):
    for f, r in flat:
        if n == r or n.endswith(f):
            return r
    return n

from collections import Counter
cr_by = Counter(bodyname(ref, g) for g in cr)
cn_by = Counter(canon(bodyname(nt, g)) for g in cn)

missing = {k: cr_by[k] - cn_by.get(k, 0) for k in cr_by if cr_by[k] > cn_by.get(k, 0)}
extra = {k: cn_by[k] - cr_by.get(k, 0) for k in cn_by if cn_by[k] > cr_by.get(k, 0)}
print(f"\nbodies where newton has FEWER collidable geoms: {len(missing)}")
for k, v in sorted(missing.items(), key=lambda t: -t[1])[:15]:
    print(f"   {k:40s} mjlab={cr_by[k]} newton={cn_by.get(k,0)}  (short {v})")
print(f"bodies where newton has MORE: {len(extra)}")
for k, v in sorted(extra.items(), key=lambda t: -t[1])[:6]:
    print(f"   {k:40s} mjlab={cr_by.get(k,0)} newton={cn_by[k]}")

# fingertips specifically -- the geoms a grasp depends on
tips_r = [g for g in cr if "finger" in bodyname(ref, g) or "tip" in bodyname(ref, g)]
tips_n = [g for g in cn if "finger" in canon(bodyname(nt, g)) or "tip" in canon(bodyname(nt, g))]
print(f"\nfinger/tip collidable geoms: mjlab={len(tips_r)}  newton={len(tips_n)}")
print(f"non-collidable dropped (harmless): mjlab has {ref.ngeom - len(cr)} non-collidable geoms")
