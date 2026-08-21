"""How much of the stapler survives each collision representation?

mjlab could only give the solver convex pieces: a convex hull, or a hand-fitted stack of <=20 boxes.
Both fill in concavities, and the stapler's side groove is exactly the feature a finger would hook
into. Newton can build an SDF from the mesh directly.

Volume error alone is a weak measure -- a hull can have a plausible volume and still have swallowed
the groove. So the test samples points that are OUTSIDE the true mesh and asks each representation
whether it thinks they are inside. Those are the points where a finger would be blocked by geometry
that does not exist.
"""
from __future__ import annotations
import os, sys, numpy as np

MESH = os.path.expanduser("~/jiarui/scaled_grab_wuji_all_o70/meshes/stapler_cir160.stl")

import trimesh
m = trimesh.load(MESH, force="mesh")
print(f"stapler mesh: {len(m.vertices)} verts, {len(m.faces)} faces, watertight={m.is_watertight}")
ext = m.bounds[1] - m.bounds[0]
print(f"  extents: {np.round(ext, 4).tolist()} m   volume={m.volume * 1e6:.2f} cm^3")

hull = m.convex_hull
print(f"  convex hull volume={hull.volume * 1e6:.2f} cm^3  "
      f"(+{100 * (hull.volume / m.volume - 1):.1f}% over the true shape)")

# --- Newton SDF ---
import newton
nm = newton.Mesh(m.vertices.astype(np.float32), m.faces.reshape(-1).astype(np.int32))
for res in (32, 64, 128):
    nm2 = newton.Mesh(m.vertices.astype(np.float32), m.faces.reshape(-1).astype(np.int32))
    try:
        nm2.build_sdf(max_resolution=res)
        sdf = nm2.sdf
        ok = sdf is not None
        print(f"  build_sdf(max_resolution={res}): {'ok' if ok else 'returned None'}")
        if ok and hasattr(sdf, "extract_isomesh"):
            iso = sdf.extract_isomesh()
            print(f"     isomesh: {type(iso).__name__}")
    except Exception as e:
        print(f"  build_sdf(max_resolution={res}) failed: {type(e).__name__}: {e}")

# --- the groove test -------------------------------------------------------
# Sample a grid over the bounding box; keep points OUTSIDE the true mesh. Any representation that
# calls such a point "inside" has invented material where the real object has a void.
rng = np.random.default_rng(0)
pts = rng.uniform(m.bounds[0], m.bounds[1], size=(60000, 3))
inside_true = m.contains(pts)
outside = pts[~inside_true]
print(f"\nsampled {len(pts)} points in the bounding box; {len(outside)} lie outside the true mesh")

inside_hull = hull.contains(outside)
n_bad_hull = int(inside_hull.sum())
print(f"  convex hull calls {n_bad_hull} of them solid "
      f"({100 * n_bad_hull / len(outside):.1f}% of the empty space it touches)")
vol_void = m.convex_hull.volume - m.volume
print(f"  that phantom material is {vol_void * 1e6:.2f} cm^3 -- "
      f"{100 * vol_void / m.volume:.0f}% of the object's own volume")
