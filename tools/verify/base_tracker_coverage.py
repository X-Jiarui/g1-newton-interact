"""How far is the frozen ASTRA base tracker from the reference, per clip, with no residual at all?

Training shows a body-link tracking distance of 14.2 cm that does not move, against a reward whose
scale is 3 cm -- flat enough that no policy change is measurable. The residual policy has limited
authority (clip 0.5), so if the frozen base tracker is already 14 cm off, no amount of training can
close it.

The base tracker was trained on apple_eat motion. Zero residual isolates what it can do on its own,
and comparing clips says whether 14 cm is this clip or this port.
"""
from __future__ import annotations
import argparse, os, sys
from dataclasses import asdict
from pathlib import Path
import numpy as np, torch, yaml as _yaml

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--xml", required=True)
ap.add_argument("--reference-pkl", required=True)
ap.add_argument("--sdf-object", default=None)
ap.add_argument("--label", default="")
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--agent-cfg-from", default=os.path.expanduser(
    "~/sweep_ckpts_r2/OF_00_apple_eat_1_SPHERE/model_7310.pt"))
A = ap.parse_args()
os.environ["APPLE_EAT_PKL"] = A.reference_pkl
sys.path.insert(0, os.path.join(HERE, "src"))
import mjw_compat; mjw_compat.apply()
import warp as wp
import mjlab.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.residual_interact import mdp as rmdp
from newton_vec_env import NewtonVecEnv

cfg = load_env_cfg("Mjlab-ResidualInteract-G1", play=False); cfg.scene.num_envs = 8
s = cfg.actions.get("sonic_action") if isinstance(cfg.actions, dict) else cfg.actions.sonic_action
s.tracking_start_assist_gain = 0.0; s.tracking_start_assist_steps = 0
from mjlab.tasks.residual_interact.env_cfgs import set_astra_body_dynamics
set_astra_body_dynamics(cfg)
env = NewtonVecEnv(cfg, A.xml, num_envs=8, device="cuda:0", sdf_object_stl=A.sdf_object)
env.reset()

zero = torch.zeros(8, 69, device="cuda:0")   # no residual: the base tracker alone
d_hist, w_hist = [], []
for k in range(A.steps):
  env.action_manager.advance(zero); env.action_term.process_actions(zero)
  for _ in range(env.decimation):
    env.action_term.apply_actions()
    env.solver.step(env.state_in, env.state_out, env.control, None, env.physics_dt)
    env.state_in, env.state_out = env.state_out, env.state_in
  env._env.episode_length_buf += 1
  d = float(rmdp._body_link_dist_mean_for_group(env._env, "all", use_tracking_weights=True).mean())
  d_hist.append(d)
  try:
    w_hist.append(float(rmdp._body_link_dist_mean_for_group(env._env, "right_wrist",
                                                            use_tracking_weights=False).mean()))
  except Exception:
    pass
d = np.array(d_hist)
print(f"\n=== {A.label or os.path.basename(A.reference_pkl)} : base tracker only, no residual ===")
print(f"  body-link tracking distance   mean {d.mean()*100:6.2f} cm   median {np.median(d)*100:6.2f}"
      f"   min {d.min()*100:6.2f}   max {d.max()*100:6.2f}")
if w_hist:
  w = np.array(w_hist)
  print(f"  right wrist distance          mean {w.mean()*100:6.2f} cm   min {w.min()*100:6.2f}")
print(f"  reward scale (distance_std) is 3.00 cm; reward = exp(-(d/std)^2)")
print(f"  -> reward at the mean distance: {np.exp(-(d.mean()/0.03)**2):.3g}")
