"""Does Newton's model carry the contact sensors the grasp rewards read?

multi_tip_surface (weight 5.0), contact_duration (1.0) and object_hard_lift (2.0) are all exactly
zero in the Newton run and nonzero in the mjlab control with identical weights. Those terms are
contact-gated, and an earlier array comparison showed sensordata as (1,294) on the mjlab side against
(1,0) on Newton's -- a model with no sensors makes every one of them structurally zero, which is a
failure this project has already met once.
"""
import os, sys
import numpy as np, mujoco

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(HERE, "src"))
import mjw_compat; mjw_compat.apply()
import newton, warp as wp
from newton.solvers import SolverMuJoCo

XML = os.path.join(HERE, "assets/scene_stapler/scene.xml")
ref = mujoco.MjModel.from_xml_path(XML)
print(f"compiled MJCF : nsensor={ref.nsensor}  nsensordata={ref.nsensordata}")
from collections import Counter
kinds = Counter(int(ref.sensor_type[i]) for i in range(ref.nsensor))
print(f"  sensor types: {dict(kinds)}")
for i in range(min(6, ref.nsensor)):
    nm = mujoco.mj_id2name(ref, mujoco.mjtObj.mjOBJ_SENSOR, i) or f"<{i}>"
    print(f"    {nm:38s} type={int(ref.sensor_type[i])} dim={int(ref.sensor_dim[i])}")

b = newton.ModelBuilder(); SolverMuJoCo.register_custom_attributes(b)
b.default_shape_cfg.gap = 0.0
b.add_mjcf(XML, collapse_fixed_joints=False, parse_mujoco_options=True)
sv = SolverMuJoCo(b.finalize(), enable_multiccd=True, update_data_interval=0, njmax=2048, nconmax=256)
nt = sv.mj_model
print(f"\nnewton model  : nsensor={nt.nsensor}  nsensordata={nt.nsensordata}")
sd = sv.mjw_data.sensordata
print(f"  mjw_data.sensordata shape: {sd.numpy().shape}")
print(f"\n-> {'SENSORS PRESENT' if nt.nsensor else 'NEWTON HAS NO SENSORS: every contact-gated reward is structurally zero'}")
