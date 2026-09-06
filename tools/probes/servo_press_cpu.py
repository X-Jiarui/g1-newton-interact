"""How deep does a position servo bury the fingertip in the object, as a function of its authority?

The penetration was traced to the hand actuators, not to the contact: in a trained rollout the
fingertip sits 20.9 mm inside a 40 mm cube while the three object contacts carry 130, 193 and
332 N, both narrow phases agree on the depth, and gap, margin, masks and the contact budget all
measure healthy. Holding 654 N at the fingertip's ~20 mm moment arm costs ~13 N*m. mjlab gives the
hand joints stiffness 300 and a 30 N*m limit -- hip numbers. The scene's own Wuji actuators, which
mjlab renames `xml_motor_unused_*` and replaces, ask for kp 0.2-0.7 and 0.147 to 0.619 N*m.

This measures the consequence directly. The real fingertip collider is put on a slide joint driven
by a position actuator, the target is commanded well past the cube's surface, and the settled
overlap is read off. The target is IDENTICAL across settings, so nothing here can be explained by
the finger arriving somewhere else -- the confound that makes rollout-to-rollout penetration
comparisons worthless.

Joint quantities are converted to the tip: a hinge of stiffness k N*m/rad at a moment arm a metres
pushes with k/a^2 N/m and can supply at most effort/a newtons. a = 0.02 m here, the arm measured
between right_finger2_link4's contact point and its joint.

KNOWN LIMIT: the current mjlab setting cannot be measured on this rig. Converting kp 300 and
damping 8 through a 20 mm arm gives 750 kN/m against 20 kN*s/m on a 17 g link, and the slide joint
does not travel at all -- that is the conversion degenerating, not a physical result, so read the
first row as "no measurement" and take the current setting's depth from the real model instead
(-20.9 mm, with the contact carrying 330 N). The rows that do settle agree with what the real
model's own force-depth curve predicts: 122 N buys 3.1 mm, 209 N buys 12.1 mm, 330 N buys 20.9 mm,
so a 31 N tip cap should land near 1-3 mm, and it does.

Why a slide joint and a static cube: pressed against a convex fingertip with nothing to stop it
sideways, a free cube slips off and leaves the scene, and every cell reads zero contacts. Along one
axis the equilibrium is well posed, and the axis is the only direction the measurement is about.

    python tools/probes/servo_press_cpu.py
"""
from __future__ import annotations

import argparse
import os
import struct
import tempfile

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ap = argparse.ArgumentParser()
ap.add_argument("--xml", default=os.path.join(HERE, "assets/scene_stapler/scene.xml"))
ap.add_argument("--object-mesh", default=os.path.join(HERE, "traces/objects/cubesmall.stl"))
ap.add_argument("--finger-body", default="right_finger2_link4")
ap.add_argument("--arm", type=float, default=0.02, help="fingertip moment arm, metres")
ap.add_argument("--command", type=float, default=0.025,
                help="metres past first contact that the servo is told to reach")
ap.add_argument("--timestep", type=float, default=0.002)
ap.add_argument("--settle", type=int, default=6000)
A = ap.parse_args()

#   label,                     kp N*m/rad, damping, effort N*m
SETTINGS = [
    ("mjlab hand (current)", 300.0, 8.0, 30.0),
    ("effort capped only", 300.0, 8.0, 0.62),
    ("kp softened only", 0.5, 0.02, 30.0),
    ("Wuji spec (both)", 0.5, 0.02, 0.62),
    ("Wuji spec, weakest", 0.2, 0.01, 0.147),
]


def dump_obj(path, verts, faces):
    with open(path, "w") as fh:
        for v in verts:
            fh.write("v %.6f %.6f %.6f\n" % tuple(v))
        for f in faces:
            fh.write("f %d %d %d\n" % (f[0] + 1, f[1] + 1, f[2] + 1))


def extract_finger(xml, want, out_obj):
    m = mujoco.MjModel.from_xml_path(xml)
    B, M = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_MESH
    for g in range(m.ngeom):
        bn = mujoco.mj_id2name(m, B, int(m.geom_bodyid[g])) or ""
        if want not in bn or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        mid = int(m.geom_dataid[g])
        va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        dump_obj(out_obj, m.mesh_vert[va:va + vn], m.mesh_face[fa:fa + fn])
        return dict(name=mujoco.mj_id2name(m, M, mid), friction=m.geom_friction[g].copy(),
                    condim=int(m.geom_condim[g]), solref=m.geom_solref[g].copy(),
                    solimp=m.geom_solimp[g].copy(),
                    mass=float(m.body_mass[int(m.geom_bodyid[g])]),
                    half=float(np.abs(m.mesh_vert[va:va + vn]).max()))
    raise SystemExit(f"no mesh collider on a body matching {want!r}")


RIG = """
<mujoco model="servo_press">
  <option timestep="{dt}" gravity="0 0 0" integrator="implicitfast"/>
  <asset>
    <mesh name="tip" file="{tip}"/>
    <mesh name="obj" file="{obj}"/>
  </asset>
  <worldbody>
    <body name="objbody" pos="0 0 0">
      <geom name="objgeom" type="mesh" mesh="obj" condim="{condim}" friction="{f0} {f1} {f2}"
            solref="{osr0} {osr1}" solimp="{osi}"/>
    </body>
    <body name="tipbody" pos="{tx} 0 0">
      <joint name="slide" type="slide" axis="-1 0 0"/>
      <geom name="tipgeom" type="mesh" mesh="tip" condim="{condim}" friction="{f0} {f1} {f2}"
            solref="{hsr0} {hsr1}" solimp="{hsi}" mass="{tipm}"/>
    </body>
  </worldbody>
  <actuator>
    <position name="press" joint="slide" kp="{kp}" kv="{kv}" forcerange="-{fr} {fr}"
              ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


def main():
    tmp = tempfile.mkdtemp(prefix="servo_")
    tip_obj = os.path.join(tmp, "tip.obj")
    meta = extract_finger(A.xml, A.finger_body, tip_obj)

    with open(A.object_mesh, "rb") as fh:
        fh.read(80)
        n = struct.unpack("<I", fh.read(4))[0]
        raw = np.frombuffer(fh.read(n * 50), dtype=np.uint8).reshape(n, 50)
    tri = raw[:, 12:48].copy().view("<f4").reshape(n * 3, 3).astype(np.float64)
    verts, inv = np.unique(tri, axis=0, return_inverse=True)
    obj_obj = os.path.join(tmp, "obj.obj")
    dump_obj(obj_obj, verts, inv.reshape(n, 3))
    ohalf = float(np.abs(verts).max())

    # The object's solref is the training override; the hand keeps the MuJoCo default it ships with.
    osr = (0.004, 1.0)
    start = ohalf + meta["half"] + 0.001
    print("finger %s (half %.1f mm)  cube half %.1f mm  start gap 1.0 mm  arm %.0f mm" % (
        meta["name"], 1000 * meta["half"], 1000 * ohalf, 1000 * A.arm))
    print("commanded travel %.1f mm past first contact; %d steps at %g s\n" % (
        1000 * A.command, A.settle, A.timestep))
    print("tip link mass %.4f kg (the real one; an invented 0.02 kg made the stiffest setting "
          "numerically unstable and it ejected)\n" % meta["mass"])
    print("%-24s%12s%12s%14s%12s%10s%9s" % (
        "setting", "kp N*m/rad", "effort N*m", "tip force max N", "travel mm", "pen mm", "ncon"))
    print("-" * 92)

    for label, kp, kd, eff in SETTINGS:
        m = mujoco.MjModel.from_xml_string(RIG.format(
            dt=A.timestep, tip=tip_obj, obj=obj_obj, condim=meta["condim"],
            f0=meta["friction"][0], f1=meta["friction"][1], f2=meta["friction"][2],
            osr0=osr[0], osr1=osr[1], osi=" ".join("%g" % v for v in meta["solimp"][:5]),
            hsr0=0.02, hsr1=1.0, hsi=" ".join("%g" % v for v in meta["solimp"][:5]),
            tx=start, kp=kp / A.arm ** 2, kv=kd / A.arm ** 2, fr=eff / A.arm,
            tipm=meta["mass"]))
        d = mujoco.MjData(m)
        d.ctrl[0] = A.command                 # slide travel: +1 mm closes the gap, rest presses in
        og = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "objgeom")
        tg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "tipgeom")
        for _ in range(A.settle):
            mujoco.mj_step(m, d)
        pen = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            if {int(c.geom1), int(c.geom2)} == {og, tg}:
                pen = min(pen, float(c.dist))
        nc = sum(1 for i in range(d.ncon)
                 if {int(d.contact[i].geom1), int(d.contact[i].geom2)} == {og, tg})
        print("%-24s%12.3g%12.3g%14.1f%12.3f%10.3f%9d%s" % (
            label, kp, eff, eff / A.arm, 1000.0 * float(d.qpos[0]), -pen * 1000.0, nc,
            "   <- no contact: the tip never settled against the cube" if nc == 0 else ""))


main()
