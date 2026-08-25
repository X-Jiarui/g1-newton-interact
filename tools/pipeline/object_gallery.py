#!/usr/bin/env python3
"""Viser gallery of every object's COLLISION geometry resting on the table.

What is drawn is the collider, not the authored visual: for the native path the object's collider
IS the real STL (that is the point of the SDF pipeline), so the STL is what the physics sees. A
toggle overlays the convex hull, which is what the MuJoCo-contact path collides instead -- for the
stapler the hull fills in the cavity under the handle, and that difference is the leading
explanation for the training gap between the two paths.

Placement is analytic: the object's lowest vertex is set on the table top. The drop-test stage of
the SDF pipeline measured every one of these settling within 0.5 mm of that pose (median
0.0096 mm), so this is the resting pose to within half a millimetre, not a guess.
"""
import argparse, json, os, time
import numpy as np
import trimesh
import viser

TABLE_TOP_Z = 0.10
TABLE_HALF = 0.30
TABLE_THICK = 0.05

ap = argparse.ArgumentParser()
ap.add_argument("--mesh-dir", default="/home/jiarui/jiarui/scaled_grab_dataset_wuji/meshes")
ap.add_argument("--manifest", default=os.path.expanduser("~/friction_probe/sdf_manifest.json"))
ap.add_argument("--port", type=int, default=8100)
A = ap.parse_args()

import sys
sys.path.insert(0, os.path.expanduser("~/friction_probe/tools/pipeline"))
import build_object_sdfs as P

paths = P.discover_meshes(A.mesh_dir, include_variants=False)
names = [os.path.basename(p)[:-4] for p in paths]
print(f"[gallery] {len(names)} objects from {A.mesh_dir}")

stats = {}
if os.path.exists(A.manifest):
    for r in json.load(open(A.manifest)).get("objects", []):
        stats[r["mesh"][:-4]] = r
    print(f"[gallery] manifest: {len(stats)} entries")

server = viser.ViserServer(port=A.port, label="GRAB object colliders")
server.scene.set_up_direction("+z")

# table
tb = trimesh.creation.box(extents=(2 * TABLE_HALF, 2 * TABLE_HALF, TABLE_THICK))
tb.apply_translation((0.0, 0.0, TABLE_TOP_Z - TABLE_THICK / 2))
server.scene.add_mesh_simple("/table", tb.vertices, tb.faces, color=(60, 200, 190), opacity=1.0)

handles, hull_handles, info = {}, {}, {}
for name, path in zip(names, paths):
    m = trimesh.load(path, force="mesh")
    v = np.asarray(m.vertices, dtype=np.float64)
    v = v - np.array([v[:, 0].mean(), v[:, 1].mean(), v[:, 2].min()])   # centre xy, base at 0
    v[:, 2] += TABLE_TOP_Z
    f = np.asarray(m.faces)
    handles[name] = server.scene.add_mesh_simple(
        f"/obj/{name}", v, f, color=(240, 190, 40), visible=False)
    try:
        hull = trimesh.Trimesh(vertices=v, faces=f).convex_hull
        hull_handles[name] = server.scene.add_mesh_simple(
            f"/hull/{name}", np.asarray(hull.vertices), np.asarray(hull.faces),
            color=(220, 90, 90), opacity=0.35, visible=False)
    except Exception:
        hull_handles[name] = None
    ex = v.max(0) - v.min(0)
    s = stats.get(name, {})
    info[name] = (f"{name}   verts {len(v)}   extent {np.round(ex, 3).tolist()} m"
                  + (f"   sdf_res {s.get('resolution')}   voxel {1000*s.get('voxel_m', 0):.2f} mm"
                     f"   rest_pen {s.get('penetration_mm', float('nan')):+.4f} mm"
                     f"   contacts {s.get('contacts')}" if s else ""))

dd = server.gui.add_dropdown("object", tuple(names), initial_value=names[0])
hull_cb = server.gui.add_checkbox("overlay convex hull (what the MuJoCo path collides)", False)
txt = server.gui.add_text("info", initial_value=info[names[0]])
counter = server.gui.add_text("gallery", initial_value=f"{len(names)} objects, colliders as trained")

def apply() -> None:
    for n in names:
        handles[n].visible = (n == dd.value)
        if hull_handles[n] is not None:
            hull_handles[n].visible = (n == dd.value) and hull_cb.value
    txt.value = info[dd.value]

dd.on_update(lambda _: apply())
hull_cb.on_update(lambda _: apply())
apply()
print(f"[gallery] serving on http://localhost:{A.port}")
while True:
    time.sleep(1.0)
