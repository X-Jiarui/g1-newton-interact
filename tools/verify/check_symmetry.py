"""Are the left/right reward terms symmetric?

The rewards are mjlab's, reused unchanged by the Newton env, so this checks mjlab's configuration.
A pair is symmetric when the two terms share a function family, a weight, and parameters that differ
only in the side they name. An asymmetric weight trains a policy that favours one hand, and the
reward total shows nothing.

Read from the live env config rather than the saved yaml: the yaml carries Python objects that do not
resolve outside a built env, and the live config is what training actually uses.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.expanduser("~/projects/g1-newton-interact/src"))
import mjw_compat; mjw_compat.apply()
import mjlab.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg

cfg = load_env_cfg("Mjlab-ResidualInteract-G1", play=False)
rewards = cfg.rewards if isinstance(cfg.rewards, dict) else vars(cfg.rewards)
rewards = {k: v for k, v in rewards.items() if hasattr(v, "weight")}
print(f"reward terms: {len(rewards)}")

def side_of(n):
    if n.startswith("left_") or "_left_" in n: return "left"
    if n.startswith("right_") or "_right_" in n: return "right"
    return None

pairs, unpaired = [], []
for n in rewards:
    if side_of(n) == "left":
        m = n.replace("left", "right", 1)
        (pairs if m in rewards else unpaired).append((n, m) if m in rewards else n)
for n in rewards:
    if side_of(n) == "right" and n.replace("right", "left", 1) not in rewards:
        unpaired.append(n)

print(f"left/right pairs: {len(pairs)}   unpaired sided terms: {len(unpaired)}")
if unpaired:
    print(f"  UNPAIRED (no mirror term exists): {unpaired}")

def norm(d, side):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, str): v = v.replace(side, "SIDE")
        elif isinstance(v, (list, tuple)):
            v = [x.replace(side, "SIDE") if isinstance(x, str) else x for x in v]
        out[k] = v
    return out

print(f"\n{'pair (side masked)':38s} {'w(L)':>9} {'w(R)':>9}  params differing")
print("-" * 92)
bad = []
for a, b in sorted(pairs):
    ca, cb = rewards[a], rewards[b]
    wa, wb = float(ca.weight), float(cb.weight)
    na, nb = norm(getattr(ca, "params", {}), "left"), norm(getattr(cb, "params", {}), "right")
    diffs = [k for k in sorted(set(na) | set(nb)) if na.get(k, "<->") != nb.get(k, "<->")]
    fa = getattr(ca.func, "__name__", str(ca.func)).replace("left", "SIDE")
    fb = getattr(cb.func, "__name__", str(cb.func)).replace("right", "SIDE")
    asym = abs(wa - wb) > 1e-12 or bool(diffs) or fa != fb
    if asym: bad.append(a)
    print(f"{a.replace('left','*'):38s} {wa:>9.4g} {wb:>9.4g}  "
          f"{diffs if diffs else 'none'}{'   <-- ASYMMETRIC' if asym else ''}")
print("-" * 92)
print(f"asymmetric pairs: {len(bad)}/{len(pairs)}" + (f"  -> {bad}" if bad else ""))

nz = {k: float(v.weight) for k, v in rewards.items() if abs(float(v.weight)) > 1e-12}
print(f"\nterms with nonzero weight: {len(nz)}/{len(rewards)} -> {nz}")
print(f"of those, sided: { {k: v for k, v in nz.items() if side_of(k)} or 'none' }")
