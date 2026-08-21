"""Import mjlab's verified MJCF into Newton and dump the model Newton actually builds.

The round trip that matters:

    mjlab scene.xml  ->  ModelBuilder.add_mjcf  ->  Newton Model  ->  SolverMuJoCo
                                                                         |
                                                            save_to_mjcf v
                                                                    newton.xml  -> compile -> facts

Newton does not simulate the MJCF; it builds its own Model and *reconstructs* a MuJoCo model from
that. Everything the conversion drops, renames, reorders, or defaults differently lives in the gap
between the two ends of that arrow, and `save_to_mjcf` is what makes the gap measurable.

`register_custom_attributes` is called before parsing on purpose: without it the MuJoCo-specific
actuator fields (gaintype/biastype/gainprm/biasprm) have nowhere to be stored, and the position
servos -- which are the entire PD definition here -- would come back as something else.

No solver options are overridden by default. What Newton does unprompted with an <option> block is
worth measuring first; overriding immediately would hide whether `parse_mujoco_options` works.
"""
from __future__ import annotations
import argparse, os, sys, traceback
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--xml", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets/mjlab_scene/scene.xml"))
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets/newton_roundtrip.xml"))
ap.add_argument("--override-options", action="store_true",
                help="pass mjlab's solver settings to SolverMuJoCo explicitly")
A = ap.parse_args()

import newton
from newton.solvers import SolverMuJoCo

print(f"newton {newton.__version__}")
builder = newton.ModelBuilder()
SolverMuJoCo.register_custom_attributes(builder)
print("registered MuJoCo custom attributes on the builder")

# FIX 1 -- contact gap. Newton leaves shape gap unset when the MJCF omits a `gap` attribute (mjlab's
# does), and the conversion then fills in 0.1 rather than MuJoCo's default of 0. With margin=0 that
# makes the force threshold margin-gap = -0.1: every contact in the model stays inactive until 10 cm
# of penetration, so the robot falls through the floor and nothing can be grasped -- while the model
# still loads, steps, and reports plausible numbers. Measured on all 81 colliding geoms.
builder.default_shape_cfg.gap = 0.0
print("set default_shape_cfg.gap = 0.0 (Newton would otherwise use 0.1 and disable every contact)")

builder.add_mjcf(
  A.xml,
  # Structure-preserving choices. Both default this way in 1.5.0, but they are stated because each
  # silently changes the DOF layout the policy's 69 outputs are indexed against.
  collapse_fixed_joints=False,
  convert_3d_hinge_to_ball_joints=False,
  parse_mujoco_options=True,
  parse_meshes=True,
  parse_sites=True,
  enable_self_collisions=True,
  verbose=False,
)
print("parsed MJCF")

model = builder.finalize()

print(f"\nNewton Model:")
for attr in ("joint_count", "body_count", "shape_count", "joint_dof_count",
             "joint_coord_count", "particle_count", "world_count"):
  print(f"  {attr:20s} {getattr(model, attr, '-')}")

# FIX 3 -- multi-CCD. Newton disables it by default (disableflags gets mjDSBL_MULTICCD = 524288)
# while mjlab leaves it enabled. It caps a geom pair at one contact point instead of four, and the
# whole task is a multi-finger grasp, so contact-point count is not a detail here.
kw = dict(enable_multiccd=True)
if A.override_options:
  kw.update(iterations=10, ls_iterations=20, integrator="implicitfast",
            solver="newton", cone="pyramidal", impratio=1.0)
print(f"\nsolver options: {kw}")

Path(A.out).parent.mkdir(parents=True, exist_ok=True)
try:
  solver = SolverMuJoCo(model, save_to_mjcf=A.out, **kw)
except Exception:
  traceback.print_exc()
  sys.exit("SolverMuJoCo construction failed")

# FIX 2 -- passive joint damping, applied after conversion because it does not survive it.
# Newton's MJCF importer drops damping on free joints, so the object's damping="0.05" (all six DOFs)
# is lost and the apple floats with no passive resistance. Assigning Model.joint_damping before
# finalize is NOT enough -- measured: the value lands on the Newton Model and still arrives as 0.0 in
# the converted MuJoCo model -- so both converted models are patched directly. mjw_model is the one
# that actually simulates; mj_model is kept consistent so anything reading it agrees.
import mujoco as _mj
import numpy as _np
_ref = _mj.MjModel.from_xml_path(A.xml)
_want = _ref.dof_damping
for _holder, _label in ((solver.mj_model, "mj_model"), (getattr(solver, "mjw_model", None), "mjw_model")):
  if _holder is None:
    continue
  _dd = getattr(_holder, "dof_damping", None)
  if _dd is None:
    print(f"  !! {_label} has no dof_damping; skipped")
    continue
  _arr = _dd.numpy() if hasattr(_dd, "numpy") else _np.asarray(_dd)
  if _arr.size != _want.size:
    print(f"  !! {_label} dof_damping size {_arr.size} != reference {_want.size}; skipped")
    continue
  _fixed = int((_np.abs(_arr.flatten() - _want) > 1e-9).sum())
  _new = _want.astype(_arr.dtype).reshape(_arr.shape)
  if hasattr(_dd, "assign"):
    _dd.assign(_new)
  else:
    _dd[:] = _new
  print(f"  synced {_label}.dof_damping from source MJCF: {_fixed} of {_want.size} DOFs corrected")

# Newton drives dt from the simulation loop rather than the model, so the converted model keeps
# MuJoCo's default 0.002 while mjlab runs at 0.005. Both are set: the model so anything reading it
# agrees with mjlab, and the loop must still be stepped at the same dt.
solver.mj_model.opt.timestep = float(_ref.opt.timestep)
print(f"set timestep to {_ref.opt.timestep} on the CPU model "
      "(the warp model's timestep is written by Newton from the dt passed to step())")

# FIX 4 -- ground plane extent. mjlab's planes carry size (0, 0, 0.01); a zero half-extent means an
# infinite plane. Newton substitutes a finite 5 x 5. MuJoCo itself collides against the infinite
# plane either way, but mujoco_warp is a separate implementation and this is a cheap delta to remove
# rather than reason about.
_pl = [i for i in range(solver.mj_model.ngeom) if int(solver.mj_model.geom_type[i]) == 0]
if _pl:
  for _i in _pl:
    solver.mj_model.geom_size[_i] = _ref.geom_size[
      [j for j in range(_ref.ngeom) if int(_ref.geom_type[j]) == 0][0]]
  _mw = getattr(solver, "mjw_model", None)
  _gs = getattr(_mw, "geom_size", None) if _mw is not None else None
  if _gs is not None and hasattr(_gs, "numpy"):
    _a = _gs.numpy().copy()
    for _i in _pl:
      _a[..., _i, :] = solver.mj_model.geom_size[_i]
    _gs.assign(_a)
  print(f"reset {len(_pl)} plane geom size(s) to mjlab's infinite-plane form")

print(f"\nwrote {A.out}  ({Path(A.out).stat().st_size/1e6:.2f} MB)")

# The solver keeps the converted MuJoCo model; report what it thinks it built, since the saved XML
# is a re-serialisation and could differ from the live model.
# The saved MJCF is written inside the constructor, i.e. BEFORE the damping patch above, so it is a
# snapshot of the conversion rather than of what will be simulated. The fact-sheet that matters is
# taken from the live model here.
import sys as _sys, json as _json
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from model_facts import facts as _facts
_lf = _facts(solver.mj_model)
_lf["_source"] = dict(side="newton_live_model", xml=A.xml)
_lf["_versions"] = dict(mujoco=_mj.__version__, newton=newton.__version__)
_lp = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "docs/newton_live_facts.json")
Path(_lp).write_text(_json.dumps(_lf, indent=1))
print(f"wrote {_lp}  <-- compare against this, not the saved XML")

mj = getattr(solver, "mj_model", None)
if mj is not None:
  print(f"live converted model: nq={mj.nq} nv={mj.nv} nu={mj.nu} njnt={mj.njnt} "
        f"nbody={mj.nbody} ngeom={mj.ngeom}")
  print(f"  timestep={mj.opt.timestep} integrator={mj.opt.integrator} "
        f"solver={mj.opt.solver} iterations={mj.opt.iterations} ls_iterations={mj.opt.ls_iterations}")
