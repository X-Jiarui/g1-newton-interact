"""Do the transplanted sensors survive Newton's compile and produce values?"""
import os, sys
import numpy as np, mujoco
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(HERE, "src"))
import mjw_compat; mjw_compat.apply()
import newton, warp as wp
from newton.solvers import SolverMuJoCo
from newton_sensors import transplant_sensors

XML = os.path.join(HERE, "assets/scene_stapler/scene.xml")
ref = mujoco.MjModel.from_xml_path(XML)
print(f"reference: nsensor={ref.nsensor} nsensordata={ref.nsensordata}")

_orig = mujoco.MjSpec.compile
def _hook(self, *a, **k):
    transplant_sensors(self, XML)
    return _orig(self, *a, **k)
mujoco.MjSpec.compile = _hook
try:
    b = newton.ModelBuilder(); SolverMuJoCo.register_custom_attributes(b)
    b.default_shape_cfg.gap = 0.0
    b.add_mjcf(XML, collapse_fixed_joints=False, parse_mujoco_options=True)
    sv = SolverMuJoCo(b.finalize(), enable_multiccd=True, update_data_interval=0,
                      njmax=2048, nconmax=256)
finally:
    mujoco.MjSpec.compile = _orig
print(f"newton after transplant: nsensor={sv.mj_model.nsensor} nsensordata={sv.mj_model.nsensordata}")
print(f"  mjw_data.sensordata shape: {sv.mjw_data.sensordata.numpy().shape}")
names = [mujoco.mj_id2name(sv.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(min(4, sv.mj_model.nsensor))]
print(f"  first sensors: {names}")
