"""Diff two compiled-model fact-sheets and say exactly where they disagree.

Used at two points and in the same form both times: first to prove mjlab's exported MJCF still
compiles to the model it came from (a control on the export path itself), then to compare Newton's
`save_to_mjcf` output against that baseline. Keeping one comparator for both means a difference
cannot be explained away by a difference in how it was measured.

Names, not indices, are the keys throughout, and ordering is checked separately from content: a pure
permutation and a changed gain are different failures with different fixes, and a comparator that
reports "69 joints differ" for a reordering is useless.
"""
from __future__ import annotations
import argparse, json, sys

ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--tol", type=float, default=1e-6)
ap.add_argument("--max-show", type=int, default=8)
A_ = ap.parse_args()

a = json.load(open(A_.a)); b = json.load(open(A_.b))
tol = A_.tol
problems = 0


def close(x, y):
  if isinstance(x, list) and isinstance(y, list):
    return len(x) == len(y) and all(close(p, q) for p, q in zip(x, y))
  if isinstance(x, (int, float)) and isinstance(y, (int, float)):
    return abs(float(x) - float(y)) <= tol + tol * abs(float(y))
  return x == y


def hdr(t):
  print(f"\n=== {t} ===")


hdr("sizes")
for k in sorted(set(a["sizes"]) | set(b["sizes"])):
  x, y = a["sizes"].get(k), b["sizes"].get(k)
  flag = "" if x == y else "   <-- DIFFERS"
  if x != y:
    problems += 1
  print(f"  {k:8s} {x!s:>8} {y!s:>8}{flag}")

hdr("options")
for k in sorted(set(a["options"]) | set(b["options"])):
  x, y = a["options"].get(k), b["options"].get(k)
  if not close(x, y):
    problems += 1
    print(f"  {k:16s} {x!s:>28} {y!s:>28}   <-- DIFFERS")
print("  (only differences shown)")

for section, order_key in (("joints", "joint_order"), ("actuators", "actuator_order")):
  hdr(f"{section}: ordering")
  oa, ob = a.get(order_key), b.get(order_key)
  if oa == ob:
    print(f"  identical order ({len(oa)} entries)")
  else:
    problems += 1
    sa, sb = set(oa), set(ob)
    if sa == sb:
      pos = {n: i for i, n in enumerate(oa)}
      perm = [pos[n] for n in ob]
      d = [i for i, v in enumerate(perm) if v != i]
      print(f"  SAME SET, DIFFERENT ORDER: {len(d)} of {len(perm)} positions moved")
      for i in d[:A_.max_show]:
        print(f"    slot {i}: A={oa[i]}  B={ob[i]}")
    else:
      print(f"  SET DIFFERS: only in A ({len(sa - sb)}): {sorted(sa - sb)[:A_.max_show]}")
      print(f"               only in B ({len(sb - sa)}): {sorted(sb - sa)[:A_.max_show]}")

for section in ("joints", "actuators", "geoms", "bodies"):
  da, db = a.get(section, {}), b.get(section, {})
  hdr(f"{section}: content ({len(da)} vs {len(db)})")
  onlya, onlyb = sorted(set(da) - set(db)), sorted(set(db) - set(da))
  if onlya or onlyb:
    problems += 1
    print(f"  only in A ({len(onlya)}): {onlya[:A_.max_show]}")
    print(f"  only in B ({len(onlyb)}): {onlyb[:A_.max_show]}")
  shared = sorted(set(da) & set(db))
  # 'index' is expected to move whenever anything is reordered; reporting it would bury the
  # differences that actually change physics.
  fieldbad = {}
  for n in shared:
    for k, v in da[n].items():
      if k == "index":
        continue
      w = db[n].get(k)
      if not close(v, w):
        fieldbad.setdefault(k, []).append((n, v, w))
  if not fieldbad:
    print(f"  all {len(shared)} shared entries agree on every field")
  else:
    for k, items in sorted(fieldbad.items(), key=lambda kv: -len(kv[1])):
      problems += 1
      print(f"  field {k!r}: {len(items)} of {len(shared)} differ")
      for n, v, w in items[:A_.max_show]:
        print(f"      {n:44s} A={v!s:<24} B={w!s}")

print(f"\n{'=' * 60}")
print("FACTS_MATCH" if problems == 0 else f"FACTS_DIFFER ({problems} differing categories)")
sys.exit(0 if problems == 0 else 1)
