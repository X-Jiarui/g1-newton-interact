"""Is mjlab's LIVE model the same structure as the XML this port was built from?

The compiled CPU models agree exactly (nM=1102, identical tree). But mjlab's live warp data carries
M with 1087 entries while Newton's carries 1102. Both cannot describe the same kinematic tree, so
either the export lost something structural, or the two libraries size M differently.

Which of those it is decides everything downstream: if the exported XML is not what mjlab actually
simulates, the Newton model was built from the wrong source and every later comparison inherited it.
"""
import os, sys
from pathlib import Path
import numpy as np, yaml as _yaml, torch, mujoco
sys.path.insert(0, os.path.expanduser("~/projects/g1-newton-interact/src"))
import mjw_compat; mjw_compat.apply()
import newton, warp as wp
from newton.solvers import SolverMuJoCo
import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.scripts.play import _apply_cfg_mapping

XML = os.path.expanduser("~/projects/g1-newton-interact/assets/mjlab_scene/scene.xml")
CKPT = os.path.expanduser("~/sweep_ckpts_r2/OF_00_apple_eat_1_SPHERE/model_7310.pt")
TASK = "Mjlab-ResidualInteract-G1"

cfg = load_env_cfg(TASK, play=True); cfg.scene.num_envs = 1
ag = load_rl_cfg(TASK)
_apply_cfg_mapping(ag, _yaml.unsafe_load((Path(CKPT).parent/"params"/"agent.yaml").open()))
_s = cfg.actions.get("sonic_action") if isinstance(cfg.actions, dict) else cfg.actions.sonic_action
_s.tracking_start_assist_gain = 0.0; _s.tracking_start_assist_steps = 0
from mjlab.tasks.residual_interact.env_cfgs import (set_astra_body_dynamics, install_astra_body_pd,
                                                    install_object_variant_sizes)
set_astra_body_dynamics(cfg)
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
u = env.unwrapped
install_astra_body_pd(u); install_object_variant_sizes(u)

live = u.sim.mj_model
exported = mujoco.MjModel.from_xml_path(XML)
print(f"mjlab LIVE cpu model : nbody={live.nbody} nv={live.nv} nM={live.nM} "
      f"ngeom={live.ngeom} nu={live.nu}")
print(f"exported XML compiled: nbody={exported.nbody} nv={exported.nv} nM={exported.nM} "
      f"ngeom={exported.ngeom} nu={exported.nu}")
print(f"  nM identical: {live.nM == exported.nM}")

for nm_ in ("nC", "nD", "nJ", "nM"):
    print(f"  {nm_}: live={getattr(live, nm_, '-')} exported={getattr(exported, nm_, '-')}")

M = u.sim.wp_data.M
print(f"\nmjlab warp data M shape: {M.numpy().shape}")
for nm_ in ("nM", "nC"):
    v = getattr(u.sim.wp_model, nm_, None)
    print(f"  mjlab wp_model.{nm_} = {v}")

b = newton.ModelBuilder(); SolverMuJoCo.register_custom_attributes(b)
b.default_shape_cfg.gap = 0.0
b.add_mjcf(XML, collapse_fixed_joints=False, parse_mujoco_options=True)
m2 = b.finalize()
sv = SolverMuJoCo(m2, enable_multiccd=True, update_data_interval=0, njmax=2048, nconmax=256)
print(f"newton warp data M shape: {sv.mjw_data.M.numpy().shape}")
for nm_ in ("nM", "nC"):
    print(f"  newton mjw_model.{nm_} = {getattr(sv.mjw_model, nm_, None)}")
print(f"newton cpu model nM = {sv.mj_model.nM}")

# If the live and exported CPU models disagree anywhere structural, name it.
if live.nM != exported.nM or live.nbody != exported.nbody:
    print("\nEXPORT LOST STRUCTURE -- comparing parents by name")
    ln = [mujoco.mj_id2name(live, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(live.nbody)]
    en = [mujoco.mj_id2name(exported, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(exported.nbody)]
    lp = {ln[i]: ln[int(live.body_parentid[i])] for i in range(live.nbody) if ln[i]}
    ep = {en[i]: en[int(exported.body_parentid[i])] for i in range(exported.nbody) if en[i]}
    for k in lp:
        if ep.get(k) != lp[k]:
            print(f"   {k}: live_parent={lp[k]} exported_parent={ep.get(k)}")
else:
    print("\nlive and exported CPU models agree on nM and nbody: the export is structurally faithful")
