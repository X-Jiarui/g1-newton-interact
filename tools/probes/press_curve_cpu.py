"""Penetration-vs-force curve for the hand-object contact pair. CPU MuJoCo, no policy, no robot.

Why this exists: every comparison of two ROLLOUTS under different contact settings is confounded.
The physics changes the trajectory, so a setting can score a shallower overlap merely by keeping the
hand further from the object or by squeezing less -- measured on one checkpoint, the median squeeze
force across five settings spanned 39 to 273000, three orders of magnitude. Conditioning on hand
pose does not fix it either, because the object moves too, so "same hand pose" is not "same
geometry".

Here nothing is left to a policy. One fingertip collider is bolted to the world, the object is
placed clear of it along the press axis, and a known force in NEWTONS pushes the object in. The
settled overlap is read off. Geometry and load are identical across settings by construction, so the
only thing that can move the number is the contact law.

Three failures this design is built to avoid, each of which produced a garbage curve first:

  * A CLOSED GRASP cages the object between fingers with no escape direction, and the measurement
    degenerates into "how much of the initial overlap failed to resolve" -- flat and non-monotonic
    in force. One fingertip against one face has a well-posed equilibrium.
  * INITIALISING THE OBJECT AT ITS RECORDED POSE starts it already ~9 mm inside the hand, which is
    the same degeneracy arriving by a different route. It starts clear here and is pressed in.
  * PINNING A JOINTED ROBOT by writing qpos every step fights the integrator: on the full scene that
    produced `Nan, Inf or huge value in QACC` within 80 ms. There are no joints here at all.

What it does NOT reproduce, and what that costs: training collides the object as a Newton SDF under
mjWarp, this uses the mesh under stock MuJoCo. That changes WHERE contact points land, not the
compliance law -- solref/solimp are solver parameters with the same meaning on both paths. So the
SHAPE and the ORDERING of these curves transfer; the absolute millimetres do not, and must be
re-measured on a GPU box before being quoted as the training penetration.

    python tools/probes/press_curve_cpu.py
"""
from __future__ import annotations

import argparse
import os
import tempfile

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ap = argparse.ArgumentParser()
ap.add_argument("--xml", default=os.path.join(HERE, "assets/scene_stapler/scene.xml"))
ap.add_argument("--object-mesh", default=os.path.join(HERE, "traces/objects/cubesmall.stl"))
ap.add_argument("--finger-geom", default="right_finger2_link4",
                help="substring of the body whose collider is used as the pressing surface")
ap.add_argument("--forces", default="0.5,1,2,5,10,20,50,100")
ap.add_argument("--backoff", type=float, default=0.002, help="metres of clearance at t=0")
ap.add_argument("--timestep", type=float, default=0.002, help="match the training box (0.002 s)")
ap.add_argument("--settle", type=int, default=2500)
ap.add_argument("--object-mass", type=float, default=0.364, help="as reported by the trained scene")
A = ap.parse_args()

S95 = (0.9, 0.95, 0.001, 0.5, 2.0)
S99 = (0.9, 0.99, 0.001, 0.5, 2.0)
#  name,          object solref, object solimp, finger solref, finger solimp, object priority
SETTINGS = [
    ("base", (0.004, 1.0), S95, (0.02, 1.0), S95, 0),
    ("hand_only", (0.004, 1.0), S95, (0.004, 1.0), S99, 0),
    ("prio_only", (0.004, 1.0), S95, (0.02, 1.0), S95, 1),
    ("prio_solimp", (0.004, 1.0), S99, (0.02, 1.0), S95, 1),
    ("both_stiff", (0.004, 1.0), S99, (0.004, 1.0), S99, 1),
]


def dump_obj(path: str, verts: np.ndarray, faces: np.ndarray) -> None:
    with open(path, "w") as fh:
        for v in verts:
            fh.write("v %.6f %.6f %.6f\n" % tuple(v))
        for f in faces:
            fh.write("f %d %d %d\n" % (f[0] + 1, f[1] + 1, f[2] + 1))


def extract_finger(xml: str, want: str, out_obj: str):
    """Pull the real fingertip collider out of the scene, plus the contact params it ships with."""
    m = mujoco.MjModel.from_xml_path(xml)
    B, M = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_MESH
    for g in range(m.ngeom):
        bn = mujoco.mj_id2name(m, B, int(m.geom_bodyid[g])) or ""
        if want not in bn or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = int(m.geom_dataid[g])
        va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        dump_obj(out_obj, m.mesh_vert[va:va + vn], m.mesh_face[fa:fa + fn])
        return dict(name=mujoco.mj_id2name(m, M, mid),
                    friction=m.geom_friction[g].copy(), condim=int(m.geom_condim[g]),
                    timestep=float(m.opt.timestep))
    raise SystemExit(f"no mesh collider on a body matching {want!r}")


RIG = """
<mujoco model="press">
  <option timestep="{dt}" gravity="0 0 0" integrator="implicitfast"/>
  <asset>
    <mesh name="tip" file="{tip}"/>
    <mesh name="obj" file="{obj}"/>
  </asset>
  <worldbody>
    <body name="tipbody" pos="0 0 0">
      <geom name="tipgeom" type="mesh" mesh="tip" condim="{condim}"
            friction="{f0} {f1} {f2}" rgba="0.7 0.7 0.75 1"/>
    </body>
    <body name="objbody" pos="{ox} 0 0">
      <!-- A SLIDE joint, not a freejoint. Pressed against a convex fingertip with nothing to
           stop it sideways, a free cube slips off the tip and is gone -- 6000 steps later it has
           left the scene and every cell reads zero contacts. Constraining it to the press axis is
           what makes the equilibrium well-posed, and the axis is the only direction the
           measurement is about. -->
      <joint name="objjoint" type="slide" axis="1 0 0"/>
      <geom name="objgeom" type="mesh" mesh="obj" condim="{condim}"
            friction="{f0} {f1} {f2}" mass="{omass}" rgba="0.8 0.3 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def build(tip_obj, obj_obj, meta, start_x):
    return mujoco.MjModel.from_xml_string(RIG.format(
        dt=A.timestep, tip=tip_obj, obj=obj_obj, condim=meta["condim"],
        f0=meta["friction"][0], f1=meta["friction"][1], f2=meta["friction"][2],
        omass=A.object_mass, ox=start_x))


def sweep(model, forces, start_x, settle):
    d = mujoco.MjData(model)
    og = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "objgeom")
    tg = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tipgeom")
    ob = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "objbody")
    out = []
    for F in forces:
        mujoco.mj_resetData(model, d)
        d.qpos[0] = 0.0   # the slide joint is measured from the body's own start pos
        d.qvel[:] = 0.0
        d.xfrc_applied[:] = 0.0
        d.xfrc_applied[ob, 0] = -F          # push it along -x, into the fingertip at the origin
        for _ in range(settle):
            mujoco.mj_step(model, d)
        best, n = 0.0, 0
        for i in range(d.ncon):
            c = d.contact[i]
            if {int(c.geom1), int(c.geom2)} == {og, tg}:
                best = min(best, float(c.dist))
                n += 1
        out.append((F, -best * 1000.0, n))
    return out


def main():
    tmp = tempfile.mkdtemp(prefix="press_")
    tip_obj = os.path.join(tmp, "tip.obj")
    meta = extract_finger(A.xml, A.finger_geom, tip_obj)

    # The object mesh has to be an OBJ too so both come from the same loader.
    import struct
    with open(A.object_mesh, "rb") as fh:
        fh.read(80)
        n = struct.unpack("<I", fh.read(4))[0]
        raw = np.frombuffer(fh.read(n * 50), dtype=np.uint8).reshape(n, 50)
    tri = raw[:, 12:48].copy().view("<f4").reshape(n * 3, 3).astype(np.float64)
    verts, inv = np.unique(tri, axis=0, return_inverse=True)
    obj_obj = os.path.join(tmp, "obj.obj")
    dump_obj(obj_obj, verts, inv.reshape(n, 3))

    half = float(np.abs(verts).max())
    start_x = 0.02 + half + A.backoff
    forces = [float(x) for x in A.forces.split(",")]
    print(f"finger collider {meta['name']}   object half-extent {1000*half:.1f} mm   "
          f"start x {1000*start_x:.1f} mm   dt {A.timestep} s   settle {A.settle} steps")
    print(f"scene timestep was {meta['timestep']} s; friction {np.round(meta['friction'], 4).tolist()}"
          f"  condim {meta['condim']}")
    print()

    rows = {}
    for name, sro, sio, srf, sif, prio in SETTINGS:
        m = build(tip_obj, obj_obj, meta, start_x)
        og = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "objgeom")
        tg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "tipgeom")
        m.geom_solref[og][:2] = sro
        m.geom_solimp[og][:5] = sio
        m.geom_solref[tg][:2] = srf
        m.geom_solimp[tg][:5] = sif
        m.geom_priority[og] = prio
        rows[name] = sweep(m, forces, start_x, A.settle)

    hdr = "".join("%9s" % ("%gN" % f) for f in forces)
    for title, idx in (("overlap (mm)", 1), ("contacts", 2)):
        print(f"=== {title} ===")
        print("%-14s%s" % ("setting", hdr))
        print("-" * (14 + 9 * len(forces)))
        for name, vals in rows.items():
            print("%-14s%s" % (name, "".join(
                "%9s" % (("%.3f" % v[1]) if idx == 1 else v[2]) for v in vals)))
        print()

    print("=== growth across the force range (contacting levels only) ===")
    print("%-14s%12s%12s%10s" % ("setting", "low", "high", "growth"))
    print("-" * 48)
    for name, vals in rows.items():
        live = [v for v in vals if v[2] > 0]
        if len(live) < 2:
            print("%-14s   fewer than two force levels made contact" % name)
            continue
        lo, hi = live[0][1], live[-1][1]
        print("%-14s%12.3f%12.3f%10s" % (
            name, lo, hi, ("%.1fx" % (hi / lo)) if lo > 1e-6 else "n/a"))


main()
