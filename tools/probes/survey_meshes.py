"""Which objects in the dataset are actually non-convex, and by how much?

The stapler mesh in this dataset turns out to be already convex -- 'cir160' is a simplified variant,
so its groove is long gone. A representation demo needs an object whose concavity is real, and the
size of that concavity is what decides whether the demo shows anything.
"""
import os, glob, numpy as np, trimesh

D = os.path.expanduser("~/jiarui/scaled_grab_wuji_all_o70/meshes")
rows = []
for f in sorted(glob.glob(os.path.join(D, "*.stl"))):
    try:
        m = trimesh.load(f, force="mesh")
        if m.volume <= 0:
            continue
        h = m.convex_hull.volume
        rows.append((os.path.basename(f), len(m.vertices), m.volume * 1e6, h * 1e6,
                     100.0 * (h / m.volume - 1.0)))
    except Exception:
        continue

rows.sort(key=lambda r: -r[4])
print(f"{'mesh':30s} {'verts':>7} {'vol cm3':>9} {'hull cm3':>9} {'hull excess':>12}")
print("-" * 72)
for name, nv, v, h, ex in rows[:18]:
    print(f"{name:30s} {nv:>7} {v:>9.2f} {h:>9.2f} {ex:>11.1f}%")
print("...")
n_convex = sum(1 for r in rows if r[4] < 1.0)
print(f"\n{n_convex} of {len(rows)} meshes are effectively convex already (hull excess < 1%)")
