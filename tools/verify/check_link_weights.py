"""Are the per-link tracking weights left/right symmetric?

`tracking` is the only reward with nonzero weight, and it uses use_tracking_weights=True over
link_group="all". So whatever symmetry the policy sees comes from those per-link weights, not from
the sided reward terms (which all sit at weight 0). If a left link is weighted differently from its
right counterpart, the policy is trained to favour one hand and nothing in the reward total shows it.
"""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/projects/g1-newton-interact/src"))
import mjw_compat; mjw_compat.apply()
import mjlab.tasks  # noqa: F401
from mjlab.tasks.residual_interact import mdp as rmdp

cands = [a for a in dir(rmdp) if "WEIGHT" in a.upper() and "TRACK" in a.upper()]
cands += [a for a in dir(rmdp) if a.upper().endswith("_WEIGHTS")]
print("weight tables found in mdp:", sorted(set(cands))[:10])

table = None
for name in sorted(set(cands)):
    obj = getattr(rmdp, name)
    if isinstance(obj, dict) and obj:
        table = (name, obj)
        break
if table is None:
    for name in sorted(set(cands)):
        obj = getattr(rmdp, name)
        print(f"  {name}: {type(obj).__name__} len={len(obj) if hasattr(obj,'__len__') else '?'}")
    raise SystemExit("no dict-shaped weight table found")

name, w = table
print(f"\nusing {name}: {len(w)} entries")

def mate(k):
    if "left" in k:  return k.replace("left", "right")
    if "right" in k: return k.replace("right", "left")
    return None

pairs, unpaired, asym = [], [], []
for k, v in w.items():
    m = mate(k)
    if m is None:
        continue
    if m not in w:
        unpaired.append(k); continue
    if "left" in k:
        pairs.append((k, m, float(v), float(w[m])))

print(f"left/right link pairs: {len(pairs)}   sided links with no mirror: {len(unpaired)}")
if unpaired:
    print(f"  UNPAIRED: {unpaired[:10]}")
for a, b, va, vb in sorted(pairs):
    if abs(va - vb) > 1e-9:
        asym.append((a, b, va, vb))
print(f"asymmetric pairs: {len(asym)}/{len(pairs)}")
for a, b, va, vb in asym[:12]:
    print(f"   {a:34s} {va:8.4f}   vs   {b:34s} {vb:8.4f}")

unsided = {k: float(v) for k, v in w.items() if mate(k) is None}
print(f"\nun-sided links: {len(unsided)}  e.g. {dict(list(unsided.items())[:6])}")
vals = np.array([float(v) for v in w.values()])
print(f"weight range: min={vals.min():.4f} max={vals.max():.4f} nonzero={int((vals!=0).sum())}/{len(vals)}")
print("SYMMETRIC" if not asym and not unpaired else "NOT SYMMETRIC")
