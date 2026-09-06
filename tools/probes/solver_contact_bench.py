"""How hard a wall does each Newton solver give, on one identical scene?

The environment currently splits the contact in two: `--native-contacts` hands the narrow phase to
Newton's CollisionPipeline, and `SolverMuJoCo` integrates the result. So the contact RESPONSE -- how
hard the wall pushes back -- is MuJoCo's soft constraint, with its own parameters (solref/solimp),
while every contact parameter Newton exposes on a shape (`ke kd kf mu restitution`) belongs to its
penalty-based solvers and is ignored. That split is what this measures the cost of.

The test is the simplest contact there is, so nothing about a robot, a policy or a reference can
enter the answer: a free cube dropped onto a static anvil, settled under its own weight, with the
load swept by changing the cube's mass. The number reported is the settled overlap -- how far the
cube ends up inside the anvil -- and its slope against load, in mm per newton, which is the same
quantity `[pen-compliance]` reports during training.

One scene, built once by `newton.ModelBuilder`, handed to each solver in turn. Whatever differs
between the rows is the solver, because nothing else is allowed to change.

    python tools/probes/solver_contact_bench.py
    python tools/probes/solver_contact_bench.py --dt 0.001 --steps 6000
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import warp as wp

import newton
from newton import ModelBuilder

ShapeConfig = ModelBuilder.ShapeConfig

ap = argparse.ArgumentParser()
ap.add_argument("--dt", type=float, default=0.002, help="the training timestep")
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--half", type=float, default=0.02, help="cube half-extent, m (cubesmall is 40 mm)")
ap.add_argument("--masses", default="0.364,1,3,10,30")
ap.add_argument("--device", default="cuda:0")
A = ap.parse_args()

ANVIL_TOP = 0.0
MASSES = [float(x) for x in A.masses.split(",")]


def cube_mesh(half: float):
    """The same 40 mm cube, as a mesh -- so it can be collided through an SDF the way training is."""
    v = np.array([[x, y, z] for x in (-half, half) for y in (-half, half) for z in (-half, half)],
                 dtype=np.float32)
    f = np.array([[0,2,3],[0,3,1],[4,5,7],[4,7,6],[0,1,5],[0,5,4],
                  [2,6,7],[2,7,3],[0,4,6],[0,6,2],[1,3,7],[1,7,5]], dtype=np.int32)
    return newton.Mesh(v, f.flatten())


def build(mass: float, kind: str = "box"):
    """One cube on one anvil. Identical for every solver -- that is the point of the test."""
    b = ModelBuilder(up_axis=newton.Axis.Z, gravity=wp.vec3(0.0, 0.0, -9.81))
    # SolverMuJoCo stores solref/solimp as custom attributes; they have to be registered on the
    # builder before finalize or the solver has nowhere to read them from.
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    cfg = ShapeConfig(density=mass / (2 * A.half) ** 3, mu=1.0)
    b.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, ANVIL_TOP - 0.05), wp.quat_identity()),
                    hx=0.25, hy=0.25, hz=0.05, cfg=ShapeConfig(mu=1.0))
    # Start exactly touching. Dropping it from a height would measure the impact, not the wall.
    # add_body already attaches a free joint; adding one explicitly builds a second, parallel
    # joint between the same pair and Newton warns that the model is inconsistent.
    body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, ANVIL_TOP + A.half),
                                         wp.quat_identity()), label="cube")
    if kind == "sdf":
        # Built the way src/grab_objects.py builds the training object: real mesh, sampled into a
        # signed distance field, added with force_sdf so the SDF is what actually collides.
        mesh = cube_mesh(A.half)
        mesh.build_sdf(max_resolution=128)
        cfg.force_sdf = True
        b.add_shape_mesh(body, mesh=mesh, cfg=cfg, label="cube_sdf")
    else:
        b.add_shape_box(body, hx=A.half, hy=A.half, hz=A.half, cfg=cfg)
    return b.finalize(device=A.device), body


def settled_overlap(model, body, solver, steps):
    """Run to rest and report how far the cube sits inside the anvil, in mm."""
    s0, s1 = model.state(), model.state()
    control = model.control()
    contacts = None
    for i in range(steps):
        contacts = model.collide(s0)
        solver.step(s0, s1, control, contacts, A.dt)
        s0, s1 = s1, s0
    z = float(wp.to_torch(s0.body_q)[body, 2].item())
    return (ANVIL_TOP + A.half - z) * 1000.0


def mujoco_solver(model, solref=None, solimp=None, native=False):
    kw = dict(njmax=64, nconmax=64, iterations=100, ls_iterations=50,
              cone="pyramidal", impratio=20.0)
    if native:
        # Exactly what `--native-contacts` configures: Newton's CollisionPipeline supplies the
        # contacts and MuJoCo only integrates them. This is the split the whole question is about.
        kw.update(use_mujoco_contacts=False, solver="newton", integrator="implicitfast")
    sv = newton.solvers.SolverMuJoCo(model, **kw)
    if solref is not None:
        import warp as _wp
        mm = sv.mj_model
        for g in range(mm.ngeom):
            mm.geom_solref[g][:2] = solref
            if solimp is not None:
                mm.geom_solimp[g][:5] = solimp
        _wp.to_torch(sv.mjw_model.geom_solref)[:] = _wp.to_torch(
            _wp.array(mm.geom_solref, dtype=float))
        if solimp is not None:
            _wp.to_torch(sv.mjw_model.geom_solimp)[:] = _wp.to_torch(
                _wp.array(mm.geom_solimp, dtype=float))
    return sv


def newton_penalty_solver(cls, model, ke=None, kd=None):
    """Newton's own contact stiffness lives on the SHAPE, not on the solver."""
    if ke is not None:
        import warp as _wp
        _wp.to_torch(model.shape_material_ke)[:] = ke
        if kd is not None:
            _wp.to_torch(model.shape_material_kd)[:] = kd
    return cls(model)


ROWS = [
    ("SDF object, NATIVE (our env)",
     lambda m: mujoco_solver(m, native=True), "sdf"),
    ("SDF object, NATIVE, timeconst .004",
     lambda m: mujoco_solver(m, (0.004, 1.0), (0.9, 0.99, 0.001, 0.5, 2.0), native=True), "sdf"),
    ("SDF object, MuJoCo own narrow phase",
     lambda m: mujoco_solver(m), "sdf"),
    ("SDF object, SolverXPBD it 20",
     lambda m: newton.solvers.SolverXPBD(m, iterations=20), "sdf"),
    ("SolverMuJoCo  default solref",
     lambda m: mujoco_solver(m), "box"),
    ("SolverMuJoCo  timeconst .004",
     lambda m: mujoco_solver(m, (0.004, 1.0), (0.9, 0.99, 0.001, 0.5, 2.0)), "box"),
    ("SolverMuJoCo  direct k=1e5",
     lambda m: mujoco_solver(m, (-1e5, -1e3), (0.9, 0.99, 0.001, 0.5, 2.0)), "box"),
    ("SolverMuJoCo  direct k=1e6",
     lambda m: mujoco_solver(m, (-1e6, -3e3), (0.9, 0.99, 0.001, 0.5, 2.0)), "box"),
    ("MuJoCo NATIVE contacts (our env)",
     lambda m: mujoco_solver(m, native=True), "box"),
    ("MuJoCo NATIVE + timeconst .004",
     lambda m: mujoco_solver(m, (0.004, 1.0), (0.9, 0.99, 0.001, 0.5, 2.0), native=True), "box"),
    ("SolverXPBD    iterations 2",
     lambda m: newton.solvers.SolverXPBD(m), "box"),
    ("SolverXPBD    iterations 20",
     lambda m: newton.solvers.SolverXPBD(m, iterations=20), "box"),
    ("SolverSemiImplicit ke default",
     lambda m: newton_penalty_solver(newton.solvers.SolverSemiImplicit, m), "box"),
    ("SolverSemiImplicit ke 1e6",
     lambda m: newton_penalty_solver(newton.solvers.SolverSemiImplicit, m, 1e6, 1e3), "box"),
    ("SolverFeatherstone ke default",
     lambda m: newton_penalty_solver(newton.solvers.SolverFeatherstone, m), "box"),
    ("SolverFeatherstone ke 1e6",
     lambda m: newton_penalty_solver(newton.solvers.SolverFeatherstone, m, 1e6, 1e3), "box"),
]


def main():
    print(f"cube {2000*A.half:.0f} mm, anvil static, dt {1000*A.dt:.1f} ms, {A.steps} steps, "
          f"gravity only; load swept by mass")
    _m0, _ = build(MASSES[0])
    print(f"requested masses {MASSES} kg; the builder derives mass from shape density, "
          f"first row lands at {float(wp.to_torch(_m0.body_mass)[0]):.3f} kg")
    print(f"loads {[round(m * 9.81, 1) for m in MASSES]} N\n")
    hdr = "".join("%11s" % ("%.1fN" % (m * 9.81)) for m in MASSES)
    print("%-32s%s%13s" % ("solver / setting", hdr, "mm per N"))
    print("-" * (32 + 11 * len(MASSES) + 13))
    for label, make, kind in ROWS:
        cells, xs, ys = [], [], []
        for mass in MASSES:
            try:
                model, body = build(mass, kind)
                ov = settled_overlap(model, body, make(model), A.steps)
                if not math.isfinite(ov):
                    cells.append("blew up")
                    continue
                cells.append("%.3f" % ov)
                xs.append(mass * 9.81)
                ys.append(ov)
            except Exception as exc:  # a solver that cannot run this scene is a result too
                cells.append("n/a")
                if len(cells) == 1:
                    print("%-32s%s" % (label, ("  %s" % type(exc).__name__ + ": " + str(exc))[:90]))
                    break
        else:
            slope = (np.polyfit(xs, ys, 1)[0] if len(xs) >= 2 else float("nan"))
            print("%-32s%s%13.5f" % (label, "".join("%11s" % c for c in cells), slope))


main()
