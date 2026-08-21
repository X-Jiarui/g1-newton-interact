"""Cross-sections through each collision representation, so the lost geometry is visible.

A volume number says the convex hull is too big; a slice says *where*. The mug's handle encloses a
hole a finger goes through, and the hull turns that hole into solid material. The section is taken
through the handle so the difference is the thing you would actually reach into.
"""
from __future__ import annotations
import os, numpy as np, trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import newton

D = os.path.expanduser("~/jiarui/scaled_grab_wuji_all_o70/meshes")
OUT = os.path.expanduser("~/projects/g1-newton-interact/media/object_representations.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

def section_polys(mesh, origin, normal):
    try:
        sec = mesh.section(plane_origin=origin, plane_normal=normal)
        if sec is None:
            return []
        planar, _ = sec.to_2D()
        return [np.asarray(p.exterior.coords) for p in planar.polygons_full]
    except Exception:
        return []

def sdf_mesh(m, res):
    nm = newton.Mesh(m.vertices.astype(np.float32), m.faces.reshape(-1).astype(np.int32))
    nm.build_sdf(max_resolution=res)
    iso = nm.sdf.extract_isomesh()
    return trimesh.Trimesh(vertices=np.asarray(iso.vertices),
                           faces=np.asarray(iso.indices).reshape(-1, 3), process=False)

OBJECTS = [("mug.stl", 2), ("eyeglasses.stl", 2)]
COLS = [("true mesh", None), ("convex hull (mjlab)", "hull"), ("Newton SDF 64", 64),
        ("Newton SDF 128", 128)]

fig, axes = plt.subplots(len(OBJECTS), len(COLS), figsize=(4.0 * len(COLS), 3.6 * len(OBJECTS)))
axes = np.atleast_2d(axes)

for r, (name, axis) in enumerate(OBJECTS):
    m = trimesh.load(os.path.join(D, name), force="mesh")
    centre = m.bounds.mean(axis=0)
    normal = np.zeros(3); normal[axis] = 1.0
    variants = {}
    variants["true mesh"] = m
    variants["convex hull (mjlab)"] = m.convex_hull
    for res in (64, 128):
        variants[f"Newton SDF {res}"] = sdf_mesh(m, res)

    for c, (label, _) in enumerate(COLS):
        ax = axes[r, c]
        polys = section_polys(variants[label], centre, normal)
        for p in polys:
            ax.fill(p[:, 0], p[:, 1], facecolor=("#c94f4f" if "hull" in label else "#3f7fb5"),
                    edgecolor="black", linewidth=0.8, alpha=0.85)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        vol = abs(variants[label].volume) * 1e6
        base = abs(m.volume) * 1e6
        pct = 100.0 * (vol / base - 1.0)
        ax.set_title(f"{label}\n{vol:.0f} cm³  ({pct:+.0f}%)", fontsize=10)
        if c == 0:
            ax.set_ylabel(name.replace(".stl", ""), fontsize=12)

fig.suptitle("Collision geometry through a cross-section: what each representation keeps\n"
             "red = convex hull, blue = true shape and SDF reconstructions", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT, dpi=130)
print(f"wrote {OUT}")
