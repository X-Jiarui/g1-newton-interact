"""Is Newton's worse mean a tail of diverging environments, or everything being slightly worse?

mjlab and Newton have nearly the same median tracking distance (5.4 vs 6.4 cm) and very different
means (5.9 vs 11.9). That is the signature of a few environments failing rather than a uniform
degradation, and the two call for completely different fixes -- so the per-environment distribution
decides which.
"""
from __future__ import annotations
import argparse, os, sys
from dataclasses import asdict
from pathlib import Path
import numpy as np, torch, yaml as _yaml

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--reference-pkl", required=True)
ap.add_argument("--xml", default=os.path.join(HERE, "assets/scene_stapler/scene.xml"))
ap.add_argument("--checkpoint", default=os.path.expanduser(
    "~/sweep_ckpts_r2/OF_00_apple_eat_1_SPHERE/model_7310.pt"))
ap.add_argument("--steps", type=int, default=150)
ap.add_argument("--num-envs", type=int, default=32)
A = ap.parse_args()
os.environ["APPLE_EAT_PKL"] = A.reference_pkl
sys.path.insert(0, os.path.join(HERE, "src"))
import mjw_compat; mjw_compat.apply()
import warp as wp
import mjlab.tasks  # noqa: F401
from mjlab.rl import RslRlVecEnvWrapper, MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.scripts.play import _apply_cfg_mapping, _maybe_wrap_residual_action_stats_policy
from mjlab.tasks.residual_interact import mdp as rmdp
from mjlab.tasks.residual_interact.env_cfgs import set_astra_body_dynamics
from newton_vec_env import NewtonVecEnv

c = load_env_cfg("Mjlab-ResidualInteract-G1", play=False); c.scene.num_envs = A.num_envs
agent = load_rl_cfg("Mjlab-ResidualInteract-G1")
_apply_cfg_mapping(agent, _yaml.unsafe_load((Path(A.checkpoint).parent/"params"/"agent.yaml").open()))
s = c.actions.get("sonic_action") if isinstance(c.actions, dict) else c.actions.sonic_action
s.tracking_start_assist_gain = 0.0; s.tracking_start_assist_steps = 0
set_astra_body_dynamics(c)

env = NewtonVecEnv(c, A.xml, num_envs=A.num_envs, device="cuda:0")
env._env._force_reference_start_frame = 0
w = RslRlVecEnvWrapper(env)
r = (load_runner_cls("Mjlab-ResidualInteract-G1") or MjlabOnPolicyRunner)(w, asdict(agent), device="cuda:0")
r.load(A.checkpoint)
p = _maybe_wrap_residual_action_stats_policy("Mjlab-ResidualInteract-G1", r,
                                             r.get_inference_policy(device="cuda:0"))
obs, _ = env.reset()
per = []
pelv = []
for k in range(A.steps):
  with torch.inference_mode(): a = p(obs)
  for n_ in ("_residual_last_action_mean","_residual_last_astra_action_pkl","_residual_last_base_action",
             "_residual_last_decoder_body_delta","_residual_last_final_action",
             "_residual_last_raw_residual_action","_residual_last_residual_action",
             "_residual_last_token_delta"):
    v = getattr(env, n_, None)
    if v is not None: setattr(env._env, n_, v)
  obs, rew, term, trunc, ex = env.step(a)
  per.append(rmdp._body_link_dist_mean_for_group(env._env, "all",
                                                 use_tracking_weights=True).detach().cpu().numpy())
  pelv.append(env._env.scene["robot"].data.root_link_pos_w[:, 2].detach().cpu().numpy())

d = np.stack(per)            # (steps, envs)
z = np.stack(pelv)
per_env = d.mean(axis=0) * 100
print(f"\nper-environment mean tracking distance over {A.steps} steps ({A.num_envs} envs):")
print(f"  {np.round(np.sort(per_env), 1).tolist()}")
good = per_env < 10
print(f"\n  under 10 cm: {int(good.sum())}/{len(per_env)}   "
      f"their mean {per_env[good].mean():.2f} cm")
if (~good).any():
  print(f"  over  10 cm: {int((~good).sum())}/{len(per_env)}   "
        f"their mean {per_env[~good].mean():.2f} cm")
print(f"\n  overall mean {per_env.mean():.2f} cm   median {np.median(per_env):.2f} cm")
print(f"  final pelvis height: min {z[-1].min():.3f}  mean {z[-1].mean():.3f}  max {z[-1].max():.3f} m")
fallen = z[-1] < 0.5
print(f"  pelvis below 0.5 m at the end (fallen): {int(fallen.sum())}/{len(fallen)}")
