"""Drop the real Wuji hand onto the real object and measure how far it goes in.

The earlier bench (`solver_contact_bench.py`) settled a cube on an anvil and found our contact
excellent -- 4 microns at 294 N. But that is object-against-table, and object-against-table was
never the complaint. **The problem is hand-against-object**, and a hand is not a cube: it is a
cluster of small convex-hulled colliders, several of which can touch at once, on a body far lighter
than the loads the anvil test applied.

So this puts the actual hand in. Every colliding mesh on the right hand is lifted out of the
training scene at a chosen pose, kept as the same geometry the training run collides (convex hulls,
via `approximate_meshes`), and welded into one rigid body. That body is dropped onto the object from
a series of heights. Height is the knob because it is the honest way to raise the contact impulse
without inventing a force: the further it falls the more momentum arrives in the one step where the
surfaces meet, and a wall that holds a resting hand may still let a falling one through.

The hand is rigid here, with no joints and no actuators. That is deliberate: it isolates the CONTACT
from the servo. The companion experiment in `docs/ik_press_bench_spec.md` does the opposite -- it
keeps the articulation and drives it with IK -- and the two together separate "the wall is soft"
from "the servo is strong".

Penetration is measured GEOMETRICALLY: hand vertices transformed into the object's frame and tested
against its faces. Not from `contact.dist`, and never from `efc_force` -- the point is to compare
solvers that do not share a contact representation, so the measurement must not belong to any of
them.

    python tools/probes/hand_drop_bench.py
    python tools/probes/hand_drop_bench.py --frame 285 --heights 0,0.002,0.01,0.05,0.2
"""
from __future__ import annotations

import argparse
import struct

import mujoco
import numpy as np
import warp as wp

import newton
from newton import ModelBuilder

ShapeConfig = ModelBuilder.ShapeConfig

ap = argparse.ArgumentParser()
ap.add_argument("--xml", default="assets/scene_stapler/scene.xml")
ap.add_argument("--object-mesh", default="traces/objects/cubesmall.stl")
ap.add_argument("--trace", default="traces/R30_CUBE_freerun.npz",
                help="supplies the hand pose; the hand is frozen in that shape and dropped")
ap.add_argument("--frame", type=int, default=285)
ap.add_argument("--heights", default="0,0.002,0.01,0.05,0.2", help="metres above first contact")
ap.add_argument("--dt", type=float, default=0.002)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--side", default="right")
ap.add_argument("--parts", default="all", choices=("all", "palm"),
                help="'all' freezes the whole hand in the trace pose -- but at frame 285 that pose "
                     "is WRAPPED round the cube, and the finger cage is narrower than the 40 mm "
                     "block, so some vertex is geometrically inside for ANY placement and no "
                     "contact model could prevent it. 'palm' drops the palm collider alone, which "
                     "presses a flat surface onto a face and is a valid wall test.")
ap.add_argument("--margin", type=float, default=0.002,
                help="metres of clear space between the hand and the object before the drop. The "
                     "hand must never START inside: a run that begins overlapped shows the solver "
                     "expelling an illegal initial condition, which is not a wall.")
ap.add_argument("--noise-floor", type=float, default=0.10,
                help="mm. Acceptance is the WORST overlap over the whole fall, not the settled one.")
ap.add_argument("--max-tri-pairs", type=int, default=12_000_000,
                help="Newton's default 1e6 triangle-pair buffer overflows on this scene -- the "
                     "object mesh is 107776 triangles against 21 hand hulls -- and everything past "
                     "the cap is silently dropped. The training env raises it to 12e6 for the same "
                     "reason (src/newton_vec_env.py, MAX_TRI_PAIRS).")
ap.add_argument("--device", default="cuda:0")
A = ap.parse_args()

HEIGHTS = [float(x) for x in A.heights.split(",")]
VERBOSE = [True]


def load_stl(path):
    with open(path, "rb") as fh:
        fh.read(80)
        n = struct.unpack("<I", fh.read(4))[0]
        raw = np.frombuffer(fh.read(n * 50), dtype=np.uint8).reshape(n, 50)
    normals = raw[:, 0:12].copy().view("<f4").astype(np.float64)
    tris = raw[:, 12:48].copy().view("<f4").reshape(n, 3, 3).astype(np.float64)
    return normals, tris


def hand_parts(frame: int):
    """Every colliding hand mesh, in the wrist frame, plus the hand's real mass.

    Taken from the compiled training scene at a recorded pose, so the shapes are the ones the
    training run actually collides -- not an idealisation of them.
    """
    m = mujoco.MjModel.from_xml_path(A.xml)
    d = mujoco.MjData(m)
    d.qpos[:] = np.load(A.trace, allow_pickle=True)["qpos"][frame]
    mujoco.mj_forward(m, d)
    B, Gm = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_GEOM

    want = ((f"{A.side}_palm_link",) if A.parts == "palm"
            else (f"{A.side}_palm_link", f"{A.side}_finger"))
    ref = next(i for i in range(m.nbody)
               if (mujoco.mj_id2name(m, B, i) or "").endswith(f"{A.side}_palm_link"))
    R_ref = d.xmat[ref].reshape(3, 3)
    p_ref = d.xpos[ref]

    parts, mass, bodies = [], 0.0, set()
    for g in range(m.ngeom):
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        bn = mujoco.mj_id2name(m, B, int(m.geom_bodyid[g])) or ""
        if not any(w in bn for w in want):
            continue
        mid = int(m.geom_dataid[g])
        va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        v = m.mesh_vert[va:va + vn].astype(np.float64)
        f = m.mesh_face[fa:fa + fn].astype(np.int32)
        # geom frame -> world -> wrist frame
        w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        loc = (w - p_ref) @ R_ref
        parts.append((loc.astype(np.float32), f, bn.split("_robot_")[-1]))
        bodies.add(int(m.geom_bodyid[g]))
    for b in bodies:
        mass += float(m.body_mass[b])
    return parts, mass


def object_planes():
    normals, tris = load_stl(A.object_mesh)
    key = np.round(normals, 4)
    _, idx = np.unique(key, axis=0, return_index=True)
    return [(normals[i], float(np.dot(normals[i], tris[i, 0]))) for i in sorted(idx)]


def deepest_inside(hand_xform, parts, obj_xform, planes):
    """Deepest hand vertex inside the object, mm. Solver-independent by construction."""
    hp, hq = np.array(hand_xform[:3]), np.array(hand_xform[3:])
    op, oq = np.array(obj_xform[:3]), np.array(obj_xform[3:])
    Rh, Ro = np.zeros(9), np.zeros(9)
    mujoco.mju_quat2Mat(Rh, np.array([hq[3], hq[0], hq[1], hq[2]]))
    mujoco.mju_quat2Mat(Ro, np.array([oq[3], oq[0], oq[1], oq[2]]))
    Rh, Ro = Rh.reshape(3, 3), Ro.reshape(3, 3)
    best = 0.0
    for loc, _f, _n in parts:
        w = loc.astype(np.float64) @ Rh.T + hp
        rel = (w - op) @ Ro
        depth = np.full(len(rel), np.inf)
        for nrm, off in planes:
            depth = np.minimum(depth, off - rel @ nrm)
        if depth.size and depth.max() > best:
            best = float(depth.max())
    return best * 1000.0


def clear_height(parts, planes, obj_z=0.02, margin=0.002):
    """The lowest hand height at which no hand vertex is inside the object, plus a margin.

    Scanned rather than guessed: the hand meshes are expressed in the wrist frame and straddle the
    origin, so the nominal 0.04 m start had the hand already buried in the block.
    """
    obj_x = (0.0, 0.0, obj_z, 0.0, 0.0, 0.0, 1.0)
    for k in range(0, 800):
        z = k * 0.0005
        if deepest_inside((0.0, 0.0, z, 0.0, 0.0, 0.0, 1.0), parts, obj_x, planes) <= 0.0:
            return z + margin
    raise SystemExit("no clear start height found")


def build(parts, hand_mass, height, obj_free=True, hull=True, base_z=0.04):
    b = ModelBuilder(up_axis=newton.Axis.Z, gravity=wp.vec3(0.0, 0.0, -9.81))
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)

    b.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, -0.05), wp.quat_identity()),
                    hx=0.3, hy=0.3, hz=0.05, cfg=ShapeConfig(mu=1.0))

    # The object, built the way src/grab_objects.py builds it for training: real mesh, sampled into
    # a signed distance field, added with force_sdf so the SDF is what collides.
    obj_cfg = ShapeConfig(density=0.364 / (0.04 ** 3), mu=1.0)
    obj_cfg.force_sdf = True
    obj = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.02), wp.quat_identity()), label="object")
    omesh = newton.Mesh.create_from_file(A.object_mesh)
    omesh.build_sdf(max_resolution=128)
    b.add_shape_mesh(obj, mesh=omesh, cfg=obj_cfg, label="apple_sdf")

    # The hand: every collider welded into one rigid body, so the drop tests the CONTACT and not
    # the finger servos. Its mass is the sum of the real link masses.
    hand = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, base_z + height), wp.quat_identity()),
                      mass=hand_mass, label="hand")
    hcfg = ShapeConfig(density=0.0, mu=1.0)
    for loc, f, name in parts:
        mesh = newton.Mesh(loc.copy(), f.flatten().copy())
        b.add_shape_mesh(hand, mesh=mesh, cfg=hcfg, label=f"hand_{name}")

    if hull:
        # Exactly what the training env does to every robot mesh collider. Verify it landed: with
        # the raw meshes in place the narrow phase reports "Triangle pair buffer overflowed
        # 2371072 > 1000000" and every result after that is meaningless.
        idx = [i for i in range(len(b.shape_type))
               if "hand_" in (b.shape_label[i] or "")]
        done = b.approximate_meshes(method="convex_hull", shape_indices=idx,
                                    keep_visual_shapes=True)
        if VERBOSE[0]:
            tri = []
            for i in idx:
                src = b.shape_source[i]
                tri.append(len(src.indices) // 3 if hasattr(src, "indices") else -1)
            print(f"[bench] hulled {len(done)} of {len(idx)} hand collider(s); "
                  f"triangles now min {min(tri)} max {max(tri)} total {sum(tri)}")
            osrc = [b.shape_source[i] for i in range(len(b.shape_type))
                    if "apple_sdf" in (b.shape_label[i] or "")]
            if osrc:
                o = osrc[0]
                print(f"[bench] object mesh {len(o.indices)//3} triangles, "
                      f"has_sdf={getattr(o, 'sdf', None) is not None}, "
                      f"force_sdf flag on cfg={obj_cfg.force_sdf}")
            VERBOSE[0] = False
    return b.finalize(device=A.device), hand, obj


def run(model, hand, obj, solver, parts, planes, steps):
    s0, s1 = model.state(), model.state()
    control = model.control()
    # `model.collide()` allocates the default 1e6 triangle-pair buffer, which this scene overflows
    # by 2.4x on every step ("Triangle pair buffer overflowed 2371072 > 1000000"). Everything past
    # the cap is dropped, so the contacts are a truncated subset and every number after it is
    # meaningless. A CollisionPipeline sized the way the training env sizes it fixes that, and has
    # the side benefit of being the same narrow phase training runs.
    from newton import CollisionPipeline
    pipeline = CollisionPipeline(model, reduce_contacts=True, broad_phase="nxn",
                                 max_triangle_pairs=A.max_tri_pairs)
    contacts = pipeline.contacts()
    worst = 0.0
    for _ in range(steps):
        pipeline.collide(s0, contacts)
        solver.step(s0, s1, control, contacts, A.dt)
        s0, s1 = s1, s0
        q = wp.to_torch(s0.body_q).detach().cpu().numpy()
        worst = max(worst, deepest_inside(q[hand], parts, q[obj], planes))
    q = wp.to_torch(s0.body_q).detach().cpu().numpy()
    return worst, deepest_inside(q[hand], parts, q[obj], planes), q[obj][:3]


def mj(model, native=True, solref=None):
    kw = dict(njmax=512, nconmax=512, iterations=100, ls_iterations=50,
              cone="pyramidal", impratio=20.0)
    if native:
        kw.update(use_mujoco_contacts=False, solver="newton", integrator="implicitfast")
    sv = newton.solvers.SolverMuJoCo(model, **kw)
    if solref is not None:
        mm = sv.mj_model
        for g in range(mm.ngeom):
            mm.geom_solref[g][:2] = solref
        wp.to_torch(sv.mjw_model.geom_solref)[:] = wp.to_torch(
            wp.array(mm.geom_solref, dtype=float))
    return sv


ROWS = [
    ("NATIVE contacts (our env)", lambda m: mj(m, native=True)),
    ("NATIVE + object solref .004", lambda m: mj(m, native=True, solref=(0.004, 1.0))),
    ("MuJoCo own narrow phase", lambda m: mj(m, native=False)),
    ("SolverXPBD it 20", lambda m: newton.solvers.SolverXPBD(m, iterations=20)),
]


def main():
    parts, hand_mass = hand_parts(A.frame)
    planes = object_planes()
    base_z = clear_height(parts, planes, margin=A.margin)
    start = deepest_inside((0.0, 0.0, base_z, 0.0, 0.0, 0.0, 1.0), parts,
                           (0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 1.0), planes)
    if start > 0.0:
        raise SystemExit(f"start pose is {1000*start:.3f} mm inside the object")
    print(f"clear start height {1000*base_z:.1f} mm (deepest hand vertex {1000*start:.3f} mm "
          f"inside, i.e. clear); every drop height is added on top of this")
    print(f"{A.side} hand: {len(parts)} colliding mesh(es), total mass {hand_mass:.4f} kg, "
          f"shape frozen at trace frame {A.frame}")
    print(f"object: {A.object_mesh} as a 128^3 SDF, free, resting on a static table")
    print(f"dt {1000*A.dt:.1f} ms, {A.steps} steps per drop; penetration measured geometrically\n")
    print("%-30s%s" % ("solver / setting",
                       "".join("%20s" % ("drop %.0f mm" % (1000 * h)) for h in HEIGHTS)))
    print("%-30s%s" % ("", "".join("%20s" % "WORST mm over fall" for _ in HEIGHTS)))
    print("-" * (30 + 20 * len(HEIGHTS)))
    for label, make in ROWS:
        cells = []
        for h in HEIGHTS:
            try:
                model, hand, obj = build(parts, hand_mass, h, base_z=base_z)
                worst, settled, _op = run(model, hand, obj, make(model), parts, planes, A.steps)
                cells.append("%.2f%s" % (worst, "" if worst <= A.noise_floor else "  FAIL"))
            except Exception as exc:
                cells.append(type(exc).__name__)
                if len(cells) == 1:
                    print("%-30s  %s" % (label, str(exc)[:80]))
                    break
        else:
            print("%-30s%s" % (label, "".join("%20s" % c for c in cells)))


main()
