"""Fix nC on the NATIVE path: recompile Newton's own spec after it restores the true COM.

Newton offsets every massive body's COM by 1 mm at compile time on purpose --
"A temporary COM offset forces qM storage that remains valid after inertia edits" -- so that inertia
can be edited at runtime without invalidating the compiled qM layout. The cost is that no body is
'simple', nC is the uncompressed size, and the contact solve is harder to converge at this scene's
10 iterations.

We never edit inertia, so the trade is not worth it here. And Newton writes the TRUE ipos back onto
the spec bodies right after compiling, which means the spec is already correct by the time the solver
finishes. Recompiling it therefore yields a model with the real inertial frames -- keeping everything
Newton built (its geoms, actuators, naming, mocap handling), unlike substituting the original MJCF.
"""
import os, sys, numpy as np, mujoco
sys.path.insert(0, os.path.expanduser("~/projects/g1-newton-interact/src"))
import mjw_compat; mjw_compat.apply()
import newton, warp as wp, mujoco_warp as mjw
from newton.solvers import SolverMuJoCo

XML = os.path.expanduser("~/projects/g1-newton-interact/assets/mjlab_scene/scene.xml")
ref = mujoco.MjModel.from_xml_path(XML)

captured = {}
_orig = mujoco.MjSpec.compile
def _capture(self, *a, **k):
    captured["spec"] = self
    return _orig(self, *a, **k)
mujoco.MjSpec.compile = _capture
try:
    b = newton.ModelBuilder(); SolverMuJoCo.register_custom_attributes(b)
    b.default_shape_cfg.gap = 0.0
    b.add_mjcf(XML, collapse_fixed_joints=False, parse_mujoco_options=True)
    sv = SolverMuJoCo(b.finalize(), enable_multiccd=True, update_data_interval=0,
                      njmax=2048, nconmax=256)
finally:
    mujoco.MjSpec.compile = _orig

spec = captured.get("spec")
print(f"spec captured: {spec is not None}")
print(f"before: nC={sv.mj_model.nC} body_simple={int((sv.mj_model.body_simple!=0).sum())} "
      f"nmocap={sv.mj_model.nmocap} nbody={sv.mj_model.nbody}")

fixed = spec.compile()
print(f"recompiled: nC={fixed.nC} body_simple={int((fixed.body_simple!=0).sum())} "
      f"nmocap={fixed.nmocap} nbody={fixed.nbody}  (reference nC={ref.nC}, "
      f"body_simple={int((ref.body_simple!=0).sum())})")

# The recompiled model must still be Newton's model: same sizes, same ordering.
same = all(getattr(fixed, f) == getattr(sv.mj_model, f)
           for f in ("nbody", "njnt", "nv", "nq", "nu", "ngeom", "nmocap"))
print(f"dimensions unchanged vs Newton's own model: {same}")
d_ipos = np.abs(fixed.body_ipos - sv.mj_model.body_ipos).max()
print(f"body_ipos max|diff| vs Newton's (post-restore): {d_ipos:.3g}")
wm = mjw.put_model(fixed)
print(f"put_model -> nC={int(wm.nC)}")
print("NATIVE_FIX_WORKS" if int(wm.nC) == int(ref.nC) and same else "NATIVE_FIX_FAILED")
