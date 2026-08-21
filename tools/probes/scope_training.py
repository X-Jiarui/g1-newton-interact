"""What is actually missing to train on Newton? Try building mjlab's managers against the bridge.

The observation builders and the action term already run on Newton state. If the reward and
termination managers do too, the training loop is mostly assembly rather than a port. Anything they
need that the bridge does not provide will surface here as a named attribute, which is a scope
estimate rather than a guess.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np, torch, yaml as _yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
import mjw_compat; mjw_compat.apply()
import newton, mujoco, warp as wp
from newton.solvers import SolverMuJoCo
from newton_simple_fix import capture_spec, restore_simple_bodies, restore_freejoint_damping
import mjlab.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.scripts.play import _apply_cfg_mapping

XML = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets/mjlab_scene/scene.xml")
CKPT = os.path.expanduser("~/sweep_ckpts_r2/OF_00_apple_eat_1_SPHERE/model_7310.pt")
TASK = "Mjlab-ResidualInteract-G1"
N = 4

cfg = load_env_cfg(TASK, play=False); cfg.scene.num_envs = N
agent = load_rl_cfg(TASK)
_apply_cfg_mapping(agent, _yaml.unsafe_load((Path(CKPT).parent / "params" / "agent.yaml").open()))
_s = cfg.actions.get("sonic_action") if isinstance(cfg.actions, dict) else cfg.actions.sonic_action
_s.tracking_start_assist_gain = 0.0; _s.tracking_start_assist_steps = 0
from mjlab.tasks.residual_interact.env_cfgs import set_astra_body_dynamics
set_astra_body_dynamics(cfg)

scene = newton.ModelBuilder(); SolverMuJoCo.register_custom_attributes(scene)
scene.default_shape_cfg.gap = 0.0
scene.add_mjcf(XML, collapse_fixed_joints=False, parse_mujoco_options=True)
world = newton.ModelBuilder(); SolverMuJoCo.register_custom_attributes(world)
world.default_shape_cfg.gap = 0.0
world.replicate(scene, world_count=N)
nmodel = world.finalize()
with capture_spec() as cap:
    solver = SolverMuJoCo(nmodel, enable_multiccd=True, update_data_interval=0,
                          njmax=2048, nconmax=256)
ref = mujoco.MjModel.from_xml_path(XML)
restore_freejoint_damping(cap.spec, XML, verbose=False)
restore_simple_bodies(solver, cap.spec, nworld=N, nconmax=256, njmax=2048, verbose=False)

from newton_bridge import NewtonEnv
nenv = NewtonEnv(solver.mj_model, solver.mjw_data, N, "cuda:0", control=nmodel.control(),
                 rename_from=ref, physics_dt=0.005, decimation=4, solver=solver)
nenv.forward()
print(f"bridge built for {N} worlds: robot joints={len(nenv.scene['robot'].joint_names)}")
print(f"  joint_pos shape {tuple(nenv.scene['robot'].data.joint_pos.shape)}")
print(f"  object pos shape {tuple(nenv.scene['apple'].data.root_link_pos_w.shape)}")

# --- what do the managers need? ---
for label, mod_path, cfg_attr in (
    ("RewardManager", "mjlab.managers.reward_manager", "rewards"),
    ("TerminationManager", "mjlab.managers.termination_manager", "terminations"),
    ("EventManager", "mjlab.managers.event_manager", "events"),
):
    try:
        mod = __import__(mod_path, fromlist=["*"])
        cls = getattr(mod, label)
    except Exception as e:
        print(f"{label}: import failed ({type(e).__name__}: {e})")
        continue
    try:
        mgr = cls(getattr(cfg, cfg_attr), nenv)
        print(f"{label}: CONSTRUCTED ok  ({len(getattr(mgr, 'active_terms', []) or [])} terms)")
    except Exception as e:
        print(f"{label}: needs -> {type(e).__name__}: {str(e)[:130]}")
