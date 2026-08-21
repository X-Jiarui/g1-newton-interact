"""Bisect the first step: does the NaN come from the reference, the action term, or the solver?

After reset the state is clean, and one step produces 150 NaNs. That step is three things in
sequence -- process_actions, then per substep apply_actions (which during the startup hold writes
root and joint state straight from the reference), then solver.step. Each is checked separately, so
the answer is a stage rather than a guess.
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np, torch, yaml as _yaml

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--xml", default=os.path.join(HERE, "assets/scene_stapler/scene.xml"))
ap.add_argument("--reference-pkl", required=True)
ap.add_argument("--sdf-object", default=None)
A = ap.parse_args()

os.environ["APPLE_EAT_PKL"] = A.reference_pkl
sys.path.insert(0, os.path.join(HERE, "src"))
import mjw_compat; mjw_compat.apply()
import warp as wp
import mjlab.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.apple_eat import mdp as amdp
from newton_vec_env import NewtonVecEnv

# --- is the reference itself finite? ---
ref = amdp._ref("cuda:0")
print("=== reference clip ===")
for k, v in ref.items():
  if torch.is_tensor(v):
    n = int(torch.isnan(v).sum())
    print(f"  {k:26s} {tuple(v.shape)}  nan={n}  |max|={float(v.abs().max()) if v.numel() else 0:.4g}"
          + ("   <-- NaN IN REFERENCE" if n else ""))
  elif isinstance(v, (int, float)):
    print(f"  {k:26s} {v}")

cfg = load_env_cfg("Mjlab-ResidualInteract-G1", play=False); cfg.scene.num_envs = 4
_s = cfg.actions.get("sonic_action") if isinstance(cfg.actions, dict) else cfg.actions.sonic_action
_s.tracking_start_assist_gain = 0.0; _s.tracking_start_assist_steps = 0
from mjlab.tasks.residual_interact.env_cfgs import set_astra_body_dynamics
set_astra_body_dynamics(cfg)
env = NewtonVecEnv(cfg, A.xml, num_envs=4, device="cuda:0", sdf_object_stl=A.sdf_object)
env.reset()

q = lambda: wp.to_torch(env.solver.mjw_data.qpos)
v = lambda: wp.to_torch(env.solver.mjw_data.qvel)
c = lambda: wp.to_torch(env.control.mujoco.ctrl)
rep = lambda t: print(f"  {t:34s} qpos_nan={int(torch.isnan(q()).sum()):4d} "
                      f"qvel_nan={int(torch.isnan(v()).sum()):4d} "
                      f"ctrl_nan={int(torch.isnan(c()).sum()):4d} "
                      f"|qvel|max={float(v().abs().max()):.4g}")

print("\n=== first step, stage by stage ===")
rep("after reset")
a = torch.zeros(4, 69, device="cuda:0")          # zero action isolates the plumbing from the policy
env.action_manager.advance(a)
env.action_term.process_actions(a)
rep("after process_actions")
for sub in range(env.decimation):
  env.action_term.apply_actions()
  rep(f"after apply_actions (substep {sub})")
  env.solver.step(env.state_in, env.state_out, env.control, None, env.physics_dt)
  env.state_in, env.state_out = env.state_out, env.state_in
  rep(f"after solver.step  (substep {sub})")
  if int(torch.isnan(q()).sum()):
    break
