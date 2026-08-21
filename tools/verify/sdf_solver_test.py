"""Does SDF collision need its own solver parameters to be stable?

SolverMuJoCo exposes sdf_iterations and sdf_initpoints, both defaulting to None. Every general solver
setting failed identically at step 0, which is the signature of a collision routine producing garbage
rather than a solver failing to converge -- so the SDF-specific parameters are the ones left to try.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, torch

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--xml", default=os.path.join(HERE, "assets/scene_stapler/scene.xml"))
ap.add_argument("--reference-pkl", required=True)
ap.add_argument("--sdf-object", required=True)
ap.add_argument("--steps", type=int, default=30)
A = ap.parse_args()
os.environ["APPLE_EAT_PKL"] = A.reference_pkl
sys.path.insert(0, os.path.join(HERE, "src"))
import mjw_compat; mjw_compat.apply()
import warp as wp
import mjlab.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg
from newton_vec_env import NewtonVecEnv

def trial(label, **kw):
  cfg = load_env_cfg("Mjlab-ResidualInteract-G1", play=False); cfg.scene.num_envs = 4
  s = cfg.actions.get("sonic_action") if isinstance(cfg.actions, dict) else cfg.actions.sonic_action
  s.tracking_start_assist_gain = 0.0; s.tracking_start_assist_steps = 0
  from mjlab.tasks.residual_interact.env_cfgs import set_astra_body_dynamics
  set_astra_body_dynamics(cfg)
  try:
    env = NewtonVecEnv(cfg, A.xml, num_envs=4, device="cuda:0",
                       sdf_object_stl=A.sdf_object, solver_kwargs=kw)
  except Exception as e:
    print(f"  {label:44s} construction failed: {type(e).__name__}: {str(e)[:60]}")
    return
  env.reset()
  a = torch.zeros(4, 69, device="cuda:0")
  first = None
  for k in range(A.steps):
    env.action_manager.advance(a); env.action_term.process_actions(a)
    for _ in range(env.decimation):
      env.action_term.apply_actions()
      env.solver.step(env.state_in, env.state_out, env.control, None, env.physics_dt)
      env.state_in, env.state_out = env.state_out, env.state_in
    env._env.episode_length_buf += 1
    if torch.isnan(wp.to_torch(env.solver.mjw_data.qpos)).any():
      first = k; break
  v = wp.to_torch(env.solver.mjw_data.qvel)
  mv = float(v.abs().max()) if not torch.isnan(v).any() else float("nan")
  print(f"  {label:44s} first_nan={'-' if first is None else first:>4}  |qvel|max={mv:.4g}")
  del env

print("=== SDF-specific solver parameters ===")
trial("defaults (sdf params unset)")
trial("sdf_iterations=10", sdf_iterations=10)
trial("sdf_iterations=20, sdf_initpoints=10", sdf_iterations=20, sdf_initpoints=10)
trial("newton contacts (use_mujoco_contacts=False)", use_mujoco_contacts=False, solver="newton")
trial("newton contacts + elliptic + impratio",
      use_mujoco_contacts=False, solver="newton", cone="elliptic", impratio=1000.0,
      iterations=15, ls_iterations=100)
