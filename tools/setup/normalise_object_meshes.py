"""Collider for every object, sized by shape instead of GRAB's uniform tessellation.

Meshes MUST be loaded with process=True. An STL stores three corner vertices per triangle, so
unprocessed it is a polygon soup with no connectivity: quadric decimation then collapses across
disconnected pieces and a 40 mm cube comes back 18 mm wrong. Merged, the same cube reaches
0.0000 mm at 200 triangles.

Budget-by-max-deviation was tried first and abandoned: max deviation lives on a handful of sharp
corners, so it rejected 20 of 31 concave meshes at any reduction while mean error stayed ~0.01 mm.
Fixed targets with the error MEASURED and printed is the honest version -- nothing is trusted.
"""
import os, sys, glob, struct
import numpy as np, trimesh, fast_simplification
from scipy.spatial import ConvexHull

TARGET      = int(os.environ.get("TARGET", "8000"))   # non-convex triangle target
SAMPLES     = 8000
BOX_TOL_MM  = 0.01
CONVEX_FILL = 0.995
REJECT_MM   = float(os.environ.get("REJECT_MM", "5.0"))   # refuse to ship a collider worse than this

def write_stl(path, T):
    out = bytearray(b"\0"*80) + struct.pack("<I", len(T))
    for p, q, r in T:
        z = np.cross(q-p, r-p); n = np.linalg.norm(z)
        z = z/n if n > 0 else np.array([0.,0.,1.])
        out += struct.pack("<12fH", *z, *p, *q, *r, 0)
    open(path, "wb").write(bytes(out))

def box_tris(lo, hi):
    c = np.array([[lo[0],lo[1],lo[2]],[hi[0],lo[1],lo[2]],[hi[0],hi[1],lo[2]],[lo[0],hi[1],lo[2]],
                  [lo[0],lo[1],hi[2]],[hi[0],lo[1],hi[2]],[hi[0],hi[1],hi[2]],[lo[0],hi[1],hi[2]]])
    F = [(0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),(1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)]
    return np.array([[c[a],c[b],c[d]] for a,b,d in F])

def dev(orig, new):
    pts, _ = trimesh.sample.sample_surface(orig, SAMPLES)
    d = np.abs(trimesh.proximity.closest_point(new, pts)[1])
    return 1000*float(d.max()), 1000*float(d.mean())

D = sys.argv[1]
skip = ("_col","_cir","_ins","_hull","_box","_truehull")
files = sorted(f for f in glob.glob(D+"/*.stl") if not any(k in os.path.basename(f) for k in skip))
print("%-16s %9s %9s %8s %9s %9s  %s" % ("mesh","tris_in","tris_out","ratio","max_dev","mean_dev","class"))
print("-"*92)
tin = tout = 0; rejected = []
for p in files:
    name = os.path.basename(p)[:-4]
    m = trimesh.load(p, process=True)
    if not isinstance(m, trimesh.Trimesh): continue
    V = np.asarray(m.vertices, float); lo, hi = V.min(0), V.max(0)
    c, h = (lo+hi)/2, (hi-lo)/2
    box_dev = 1000*float(np.abs(np.abs(V-c)-h).min(1).max())
    try:
        H = ConvexHull(np.unique(np.round(V,9), axis=0)); fill = float(m.volume)/H.volume
    except Exception:
        H, fill = None, 0.0
    if box_dev < BOX_TOL_MM:
        T = box_tris(lo, hi); kind = "BOX"
        new = trimesh.Trimesh(vertices=T.reshape(-1,3), faces=np.arange(len(T)*3).reshape(-1,3), process=True)
    elif fill > CONVEX_FILL and H is not None:
        H2 = ConvexHull(H.points[H.vertices])
        new = trimesh.Trimesh(vertices=H2.points, faces=H2.simplices, process=True); kind = "CONVEX"
        if len(new.faces) > TARGET:
            v,f = fast_simplification.simplify(np.asarray(new.vertices,np.float32),
                                               np.asarray(new.faces,np.int32), 1.0-TARGET/len(new.faces))
            new = trimesh.Trimesh(vertices=v, faces=f, process=True); kind = "CONVEX+dec"
        T = np.asarray(new.vertices)[np.asarray(new.faces)]
    else:
        if len(m.faces) <= TARGET:
            print("%-16s %9d %9d %7s %9s %9s  already small" % (name[:16], len(m.faces), len(m.faces), "1x","-","-")); continue
        v,f = fast_simplification.simplify(np.asarray(m.vertices,np.float32),
                                           np.asarray(m.faces,np.int32), 1.0-TARGET/len(m.faces))
        new = trimesh.Trimesh(vertices=v, faces=f, process=True); kind = "NON-CONVEX"
        T = np.asarray(new.vertices)[np.asarray(new.faces)]
    mx, mn = dev(m, new)
    if mx > REJECT_MM:
        rejected.append((name, mx)); print("%-16s %9d %9d %7s %9.3f %9.4f  REJECTED (> %.1f mm)" %
              (name[:16], len(m.faces), len(T), "-", mx, mn, REJECT_MM)); continue
    write_stl(os.path.join(D, name+"_col.stl"), T)
    tin += len(m.faces); tout += len(T)
    print("%-16s %9d %9d %6.0fx %9.3f %9.4f  %s" % (name[:16], len(m.faces), len(T),
          len(m.faces)/max(len(T),1), mx, mn, kind))
print("-"*92)
print("shipped: %d -> %d triangles (%.0fx fewer)" % (tin, tout, tin/max(tout,1)))
print("rejected (shape does not survive, left alone):", rejected or "none")
