"""Is the invalidation the cause, or does the spec compile to nC=1102 in the first place?

_sync_mjw_inertias_to_mjc_cpu() zeroes body_simple/dof_simplenum whenever a body's inertia or
inertial frame differs between the CPU and warp models -- 60 bodies trip the iquat comparison, which
for float32-vs-float64 copies of the same data is round-off, not a real change. Disabling that one
method separates the two possible causes cleanly.
"""
import os, sys, numpy as np, mujoco
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
import mjw_compat; mjw_compat.apply()
import newton, warp as wp, mujoco_warp as mjw
from newton.solvers import SolverMuJoCo

XML = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets/mjlab_scene/scene.xml")
ref = mujoco.MjModel.from_xml_path(XML)

def build(disable_sync: bool):
    if disable_sync:
        SolverMuJoCo._sync_mjw_inertias_to_mjc_cpu = lambda self: None
    b = newton.ModelBuilder(); SolverMuJoCo.register_custom_attributes(b)
    b.default_shape_cfg.gap = 0.0
    b.add_mjcf(XML, collapse_fixed_joints=False, parse_mujoco_options=True)
    m = b.finalize()
    sv = SolverMuJoCo(m, enable_multiccd=True, update_data_interval=0, njmax=2048, nconmax=256)
    return sv

_orig = SolverMuJoCo._sync_mjw_inertias_to_mjc_cpu
sv = build(False)
print(f"stock            : mj_model.nC={sv.mj_model.nC}  mjw.nC={int(sv.mjw_model.nC)}  "
      f"body_simple={int((sv.mj_model.body_simple != 0).sum())}  "
      f"dof_simplenum={int((sv.mj_model.dof_simplenum != 0).sum())}")

SolverMuJoCo._sync_mjw_inertias_to_mjc_cpu = _orig
sv2 = build(True)
print(f"sync disabled    : mj_model.nC={sv2.mj_model.nC}  mjw.nC={int(sv2.mjw_model.nC)}  "
      f"body_simple={int((sv2.mj_model.body_simple != 0).sum())}  "
      f"dof_simplenum={int((sv2.mj_model.dof_simplenum != 0).sum())}")
print(f"reference        : nC={ref.nC}  body_simple={int((ref.body_simple != 0).sum())}  "
      f"dof_simplenum={int((ref.dof_simplenum != 0).sum())}")
print("INVALIDATION_IS_THE_CAUSE" if int(sv2.mjw_model.nC) == int(ref.nC)
      else "CAUSE_IS_UPSTREAM_OF_THE_INVALIDATION")
