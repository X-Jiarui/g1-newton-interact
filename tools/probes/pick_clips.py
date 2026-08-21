"""Rank stapler and mug clips by how far the object actually moves.

A sequence where the object barely leaves the table cannot teach a grasp: the reward's lift terms
never fire and the clip trains approach only. Displacement is the cheapest filter that separates
"picks it up" from "touches it", and it was the filter that mattered in the previous sweep.
"""
import glob, os, pickle
import numpy as np

D = os.path.expanduser("~/jiarui/grab_g1_wuji_aligned")

def object_positions(d):
  # The pkl stores MuJoCo-frame poses under object.pos_mj / object.quat_wxyz_mj.
  return np.asarray(d["object"]["pos_mj"])

rows = []
for name in ("stapler", "mug"):
  for f in sorted(glob.glob(f"{D}/*/{name}*.pkl")):
    try:
      d = pickle.load(open(f, "rb"))
      p = object_positions(d)
      rise = float(p[:, 2].max() - p[:, 2].min()) * 100.0
      span = float(np.linalg.norm(p.max(axis=0) - p.min(axis=0))) * 100.0
      stl = os.path.basename(d["object"].get("stl_path", "?"))
      cf = d.get("object_alignment", {}).get("cf", -1)
      side = d.get("object_alignment", {}).get("side", "?")
      rows.append((name, rise, span, len(p), f"{f.split('/')[-2]}/{os.path.basename(f)[:-4]}", stl, cf, side))
    except Exception as e:
      print(f"  skip {f}: {type(e).__name__}: {e}")

for name in ("stapler", "mug"):
  sub = sorted([r for r in rows if r[0] == name], key=lambda r: -r[1])
  print(f"\n=== {name}: {len(sub)} clips, by vertical rise ===")
  print(f"{'rise_cm':>8} {'span_cm':>8} {'frames':>7} {'cf':>5} {'side':>6}  clip")
  for _, rise, span, n, clip, stl, cf, side in sub[:8]:
    print(f"{rise:8.1f} {span:8.1f} {n:7d} {cf:>5} {side:>6}  {clip}   mesh={stl}")
