"""Why does the sparse mass matrix have a different number of non-zeros in the two models?

nv is 81 on both sides, yet M holds 1087 non-zeros in mjlab and 1102 in Newton (and qLD 1087 vs
1138). nM is a pure function of the kinematic tree: each dof contributes one entry per ancestor dof
plus itself. Different nM with identical nv therefore means the two models do not agree on who is
whose ancestor -- a structural difference, not a numerical one, and one that changes how inertia
couples across the robot and so what the constraint solver computes.

Body counts and per-body fields were already verified equal, so this looks specifically at topology:
parent links, dof ordering, and the tree/body maps that nM is derived from.
"""
import os, sys, numpy as np, mujoco
sys.path.insert(0, os.path.expanduser("~/projects/g1-newton-interact/src"))
import mjw_compat; mjw_compat.apply()
import newton
from newton.solvers import SolverMuJoCo

XML = os.path.expanduser("~/projects/g1-newton-interact/assets/mjlab_scene/scene.xml")
ref = mujoco.MjModel.from_xml_path(XML)
b = newton.ModelBuilder(); SolverMuJoCo.register_custom_attributes(b)
b.default_shape_cfg.gap = 0.0
b.add_mjcf(XML, collapse_fixed_joints=False, parse_mujoco_options=True)
m = b.finalize()
sv = SolverMuJoCo(m, enable_multiccd=True, update_data_interval=0, njmax=2048, nconmax=256)
nt = sv.mj_model

print(f"nv   : mjlab={ref.nv} newton={nt.nv}")
print(f"nbody: mjlab={ref.nbody} newton={nt.nbody}")
print(f"njnt : mjlab={ref.njnt} newton={nt.njnt}")
print(f"nM   : mjlab={ref.nM} newton={nt.nM}")

bn_r = [mujoco.mj_id2name(ref, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(ref.nbody)]
bn_n = [mujoco.mj_id2name(nt, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(nt.nbody)]
flat = sorted([(r.replace("/", "_"), r) for r in bn_r if r], key=lambda t: -len(t[0]))
def canon(n):
    for f, r in flat:
        if n == r or n.endswith(f):
            return r
    return n
bn_n_c = [canon(x) for x in bn_n]

print(f"\nbody name sets identical: {sorted(x for x in bn_r if x) == sorted(x for x in bn_n_c if x)}")
print(f"body ORDER identical    : {bn_r == bn_n_c}")

# Parent links, compared by NAME so a different ordering cannot hide a different topology.
par_r = {bn_r[i]: bn_r[int(ref.body_parentid[i])] for i in range(ref.nbody) if bn_r[i]}
par_n = {bn_n_c[i]: bn_n_c[int(nt.body_parentid[i])] for i in range(nt.nbody) if bn_n_c[i]}
diff = {k: (par_r[k], par_n.get(k, "<absent>")) for k in par_r if par_n.get(k) != par_r[k]}
print(f"\nbodies whose PARENT differs: {len(diff)}")
for k, (a, c) in list(diff.items())[:12]:
    print(f"   {k:34s} mjlab_parent={a[:28]:30s} newton_parent={c[:28]}")

# dof depth = number of ancestor dofs; this is exactly what sets nM.
def depths(M, names):
    out = {}
    for d in range(M.nv):
        bid = int(M.dof_bodyid[d])
        n = 0
        p = int(M.dof_parentid[d])
        while p >= 0:
            n += 1
            p = int(M.dof_parentid[p])
        out.setdefault(names[bid], []).append(n)
    return out
dr, dn = depths(ref, bn_r), depths(nt, bn_n_c)
print(f"\nsum of (dof depth+1): mjlab={sum(sum(v)+len(v) for v in dr.values())} "
      f"newton={sum(sum(v)+len(v) for v in dn.values())}")
bad = [(k, dr[k], dn.get(k)) for k in dr if dn.get(k) != dr[k]]
print(f"bodies whose dof depths differ: {len(bad)}")
for k, a, c in bad[:12]:
    print(f"   {k:34s} mjlab={a} newton={c}")
