"""Is the reported hand-object overlap real geometry, or an artifact of how SDF contacts report it?

`contact.dist` under mjWarp with an SDF collider reads -5 to -9.5 mm during training. A controlled
press test says the contact law yields 0.0075 mm per newton, so 9 mm would need ~1200 N against
measured forces of ~10 N -- two orders of magnitude apart. Either the forces are far larger than the
probe suggests, or the number is not the geometric interpenetration it looks like.

This settles it without any physics at all: forward-kinematic a recorded frame, then measure how far
each hand collider's surface actually lies inside the object's surface. Pure geometry, no solver, no
contact model. If the surfaces really do overlap by millimetres the meshes will say so; if they
barely touch, the solver's number is measuring something else.

Method: sample every hand collider's mesh vertices in the object's frame and evaluate them against
the object mesh. A vertex is inside when it is on the inner side of every face of the (convex) cube,
and its depth is the smallest distance to any face. Reported as the deepest such vertex per body.
The cube is convex so this is exact; for a non-convex object it would be a lower bound.

    python tools/probes/overlap_geometry.py traces/R30_CUBE_freerun.npz 285
"""
from __future__ import annotations

import struct
import sys

import mujoco
import numpy as np

XML = "assets/scene_stapler/scene.xml"
OBJ_STL = "traces/objects/cubesmall.stl"
OBJ_QADR = 76


def load_stl(path):
    with open(path, "rb") as fh:
        fh.read(80)
        n = struct.unpack("<I", fh.read(4))[0]
        raw = np.frombuffer(fh.read(n * 50), dtype=np.uint8).reshape(n, 50)
    normals = raw[:, 0:12].copy().view("<f4").astype(np.float64)
    tris = raw[:, 12:48].copy().view("<f4").reshape(n, 3, 3).astype(np.float64)
    return normals, tris


def main():
    npz = np.load(sys.argv[1], allow_pickle=True)
    frame = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    qpos = npz["qpos"][frame]

    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)

    obj_pos = qpos[OBJ_QADR:OBJ_QADR + 3]
    obj_quat = qpos[OBJ_QADR + 3:OBJ_QADR + 7]
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, obj_quat)
    R = R.reshape(3, 3)

    normals, tris = load_stl(OBJ_STL)
    # One plane per distinct face normal: the cube is convex, so "inside" is inside every plane.
    key = np.round(normals, 4)
    _, idx = np.unique(key, axis=0, return_index=True)
    planes = [(normals[i], float(np.dot(normals[i], tris[i, 0]))) for i in sorted(idx)]

    B, G = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_GEOM
    rows = []
    for g in range(m.ngeom):
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        bn = mujoco.mj_id2name(m, B, int(m.geom_bodyid[g])) or ""
        if "right_finger" not in bn and "right_palm" not in bn:
            continue
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = int(m.geom_dataid[g])
        va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        v = m.mesh_vert[va:va + vn].astype(np.float64)
        # geom frame -> world -> object frame
        gp, gm = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
        w = v @ gm.T + gp
        loc = (w - obj_pos) @ R

        depth = np.full(len(loc), np.inf)
        for nrm, off in planes:
            depth = np.minimum(depth, off - loc @ nrm)      # >0 means inside this plane
        inside = depth > 0
        if inside.any():
            rows.append((bn.replace("robot/", ""), float(depth[inside].max()) * 1000.0,
                         int(inside.sum()), len(loc)))

    print(f"frame {frame} of {len(npz['qpos'])}")
    print()
    if not rows:
        print("NO hand vertex lies inside the object mesh at this frame.")
        print("The solver's negative contact.dist is then NOT a geometric interpenetration.")
        return
    rows.sort(key=lambda r: -r[1])
    print("%-26s%14s%12s%10s" % ("hand body", "deepest mm", "verts in", "verts"))
    print("-" * 64)
    for bn, dep, ni, nv in rows:
        print("%-26s%14.3f%12d%10d" % (bn, dep, ni, nv))
    print()
    print("deepest hand vertex inside the object: %.3f mm" % max(r[1] for r in rows))


main()
