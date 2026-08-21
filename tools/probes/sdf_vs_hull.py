"""Does Newton's SDF actually preserve concavity, and at what resolution?

Volume is the wrong headline number on its own -- what matters for manipulation is whether the empty
space stays empty, because that is where a finger goes. So for each representation we count the
points that are OUTSIDE the true mesh but which the representation calls solid: phantom material
that would block a finger that should fit.

Resolution is swept because an SDF is a sampled field: too coarse and the handle hole closes up.
"""
from __future__ import annotations
import os, numpy as np, trimesh, newton

D = os.path.expanduser("~/jiarui/scaled_grab_wuji_all_o70/meshes")
rng = np.random.default_rng(0)

def phantom_fraction(true_mesh, test_contains, pts_outside):
    return float(np.mean(test_contains(pts_outside)))

for name in ("mug.stl", "eyeglasses.stl"):
    p = os.path.join(D, name)
    m = trimesh.load(p, force="mesh")
    pts = rng.uniform(m.bounds[0], m.bounds[1], size=(12000, 3))
    outside = pts[~m.contains(pts)]
    hull_phantom = phantom_fraction(m, m.convex_hull.contains, outside)
    print(f"\n=== {name}  ({len(m.vertices)} verts, {m.volume*1e6:.1f} cm3, "
          f"bbox {np.round(m.bounds[1]-m.bounds[0],3).tolist()}) ===")
    print(f"  convex hull : {100*hull_phantom:5.1f}% of the empty space filled in with phantom solid")

    for res in (32, 64, 128):
        nm = newton.Mesh(m.vertices.astype(np.float32), m.faces.reshape(-1).astype(np.int32))
        try:
            nm.build_sdf(max_resolution=res)
        except Exception as e:
            print(f"  sdf res={res:4d}: build failed ({type(e).__name__})"); continue
        sdf = nm.sdf
        iso = sdf.extract_isomesh()
        del nm
        v = np.asarray(iso.vertices); f = np.asarray(iso.indices).reshape(-1, 3)
        tm = trimesh.Trimesh(vertices=v, faces=f, process=False)
        try:
            vol = abs(tm.volume) * 1e6
            ph = phantom_fraction(m, tm.contains, outside)
            print(f"  sdf res={res:4d}: {100*ph:5.1f}% phantom   volume={vol:7.1f} cm3 "
                  f"({100*(vol/(m.volume*1e6)-1):+6.1f}%)   isomesh {len(v)} verts")
        except Exception as e:
            print(f"  sdf res={res:4d}: isomesh {len(v)} verts, measure failed ({type(e).__name__})")
