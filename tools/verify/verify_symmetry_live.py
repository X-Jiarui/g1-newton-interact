"""Compute the actual per-link tracking weights on a live env and compare left against right.

The weight function keys on body-part names ('shoulder', 'elbow', 'wrist', 'ankle'), which do not
mention a side -- so symmetry is structural. Reading that off the source is an argument; this runs
the function and diffs the numbers.
"""
from __future__ import annotations
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
import mjw_compat; mjw_compat.apply()
import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.residual_interact import mdp as rmdp

cfg = load_env_cfg("Mjlab-ResidualInteract-G1", play=True); cfg.scene.num_envs = 1
from mjlab.tasks.residual_interact.env_cfgs import set_astra_body_dynamics
set_astra_body_dynamics(cfg)
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
u = env.unwrapped

w = rmdp._body_xyz_tracking_weights(u).detach().cpu().numpy()
_, names_all = rmdp._reference_body_xyz_cache(u.device)
_, ref_ids = rmdp._body_xyz_tracking_ids(u)
names = [names_all[int(i)] for i in ref_ids.detach().cpu().numpy()]
print(f"tracked links: {len(names)}   weights: {sorted(set(np.round(w,4).tolist()))}")

def mate(n):
    if "left" in n:  return n.replace("left", "right")
    if "right" in n: return n.replace("right", "left")
    return None

idx = {n: i for i, n in enumerate(names)}
pairs, unpaired, asym = [], [], []
for n in names:
    m = mate(n)
    if m is None: continue
    if m not in idx:
        unpaired.append(n); continue
    if "left" in n:
        pairs.append((n, m, float(w[idx[n]]), float(w[idx[m]])))
for a, b, va, vb in pairs:
    if abs(va - vb) > 1e-9:
        asym.append((a, b, va, vb))

print(f"left/right link pairs: {len(pairs)}   sided links with no mirror: {len(unpaired)}")
if unpaired: print(f"  UNPAIRED: {unpaired}")
print(f"asymmetric pairs: {len(asym)}")
for a, b, va, vb in asym[:10]:
    print(f"   {a:30s} {va:.4f}  vs  {b:30s} {vb:.4f}")

byw = {}
for n, ww in zip(names, w):
    byw.setdefault(round(float(ww), 4), []).append(n)
for ww, ns in sorted(byw.items(), reverse=True):
    l = sum(1 for x in ns if "left" in x); r = sum(1 for x in ns if "right" in x)
    print(f"  weight {ww:>5}: {len(ns):2d} links  (left {l}, right {r}, unsided {len(ns)-l-r})")

print("\nSYMMETRIC" if not asym and not unpaired else "\nNOT SYMMETRIC")
env.close()
