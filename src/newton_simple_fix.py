"""Restore MuJoCo's compressed mass-matrix layout on Newton's own model.

Newton offsets every massive body's centre of mass by 1 mm along X while compiling the MjSpec:

    compile_ipos[0] += 1.0e-3 if compile_ipos[0] >= 0.0 else -1.0e-3
    # A temporary COM offset forces qM storage that remains valid after inertia edits.

That is deliberate. The offset stops MuJoCo from marking any body "simple", so the compiler allocates
the general qM layout, which stays valid if inertia is later edited at runtime (domain randomisation,
`notify_model_changed(BODY_INERTIAL_PROPERTIES)`). Newton then writes the true COM back into both the
model and the spec.

The cost is paid every step: with no simple bodies, `nC` is the uncompressed size (1102 rather than
1087 on this scene) and the constraint solve has more to do. At this scene's 10 solver iterations
that is the difference between a grasp that holds and one that is marginal -- measured over repeated
runs, 8 of 11 held with the pessimised layout against 6 of 6 with the compressed one.

For a policy being *inferred*, nothing edits inertia, so the trade is not worth taking. And because
Newton restores the true COM onto the spec, the spec is already correct by the time the solver
finishes: recompiling it yields a model with the real inertial frames and therefore the compressed
layout, while keeping everything Newton built -- its geoms, actuators, naming and mocap bodies. That
is the difference between this and substituting a separately compiled MJCF, which discards Newton's
construction entirely.

Do NOT use this if you intend to edit inertia at runtime. The offset exists for that case.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np


class capture_spec:
  """Context manager that records the MjSpec a Newton solver compiles.

  `SolverMuJoCo` does not retain its spec, so it is captured by wrapping `MjSpec.compile` for the
  duration of solver construction. The wrapper is always removed, including on failure.
  """

  def __init__(self) -> None:
    self.spec: Any = None
    self._orig = None

  def __enter__(self) -> "capture_spec":
    self._orig = mujoco.MjSpec.compile
    outer = self

    def _wrapped(spec_self, *args, **kwargs):
      outer.spec = spec_self
      return outer._orig(spec_self, *args, **kwargs)

    mujoco.MjSpec.compile = _wrapped
    return self

  def __exit__(self, *exc) -> None:
    if self._orig is not None:
      mujoco.MjSpec.compile = self._orig
    return None


def restore_freejoint_damping(spec, xml_path: str, *, verbose: bool = True) -> int:
  """Put back the free-joint damping Newton's MJCF importer drops.

  `add_mjcf` does not carry `damping` on a `<freejoint>`: builder.joint_damping comes back all zeros
  and assigning it before `finalize()` has no effect either (measured: a marker value written there
  never reaches the model). In this scene the only damped dofs are the object's six, so the loss is
  invisible until an object is expected to settle rather than drift.

  The authored values are read from the scene file with MuJoCo's *parser* and written onto Newton's
  own spec, so they flow through Newton's compile rather than around it. Call before recompiling.
  """
  import mujoco as _mj

  authored = _mj.MjSpec.from_file(xml_path)
  want = {j.name: float(np.max(j.damping)) for j in authored.joints
          if j.type == _mj.mjtJoint.mjJNT_FREE and float(np.max(j.damping)) != 0.0}
  if not want:
    return 0
  fixed = 0
  for j in spec.joints:
    if j.type != _mj.mjtJoint.mjJNT_FREE:
      continue
    for name, value in want.items():
      # Newton renames joints to a flattened path, so match on suffix.
      if j.name == name or j.name.endswith(name.replace("/", "_")):
        # MjsJoint.damping is a 3-vector on the spec (per-axis), not the scalar the model exposes.
        j.damping = np.full(3, value, dtype=np.float64)
        fixed += 1
        break
  if verbose:
    print(f"[simple-fix] restored free-joint damping on {fixed} joint(s): "
          f"{ {k: v for k, v in want.items()} }")
  return fixed


def restore_simple_bodies(solver, spec, *, nworld: int = 1, nconmax: int = 256,
                          njmax: int = 2048, verbose: bool = True) -> dict:
  """Recompile `spec` and install the result on `solver`. Returns before/after `nC`.

  Raises if the recompiled model differs from Newton's in any dimension: that would mean the spec had
  changed in some way other than the restored COM, and silently swapping in a different model is the
  failure mode this whole project exists to avoid.
  """
  import mujoco_warp as mjw

  before = int(solver.mjw_model.nC)
  fixed = spec.compile()

  for field in ("nbody", "njnt", "nv", "nq", "nu", "ngeom", "nmocap", "nM"):
    a, b = getattr(solver.mj_model, field), getattr(fixed, field)
    if a != b:
      raise RuntimeError(
        f"recompiled spec disagrees with Newton's model on {field}: {a} vs {b}. The spec must "
        "differ by more than the restored centre of mass; refusing to install it."
      )

  solver.mj_model = fixed
  solver.mjw_model = mjw.put_model(fixed)
  solver.mjw_data = mjw.put_data(fixed, mujoco.MjData(fixed), nworld=nworld,
                                 nconmax=nconmax, njmax=njmax)
  after = int(solver.mjw_model.nC)
  if verbose:
    n_simple = int((fixed.body_simple != 0).sum())
    print(f"[simple-fix] recompiled Newton's spec: nC {before} -> {after}, "
          f"{n_simple} simple bodies restored (model dimensions unchanged)")
  return {"nC_before": before, "nC_after": after}
