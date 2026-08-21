"""What is the trained policy actually doing, and why is the reward flat?

The reward is exp(-(d/0.03)^2) over weighted body-link tracking distance. At 3 cm std, a tracking
error of 20 cm puts the reward in a region so flat that no policy change is measurable -- the value
function learns a constant, entropy grows, and nothing improves. That is the shape of what the log
shows, so the tracking distance itself is the number to look at.

Also reports the physics the training log never recorded: object height, fingertip-object distance,
and active constraint count.
"""
from __future__ import annotations
import argparse, os, sys
from dataclasses import asdict
from pathlib import Path
import numpy as np, torch, yaml as _yaml

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--xml", default=os.path.join(HERE, "assets/scene_stapler/scene.xml"))
ap.add_argument("--reference-pkl", required=True)
ap.add_argument("--sdf-object", default=None)
ap.add_argument("--agent-cfg-from", default=os.path.expanduser(
    "~/sweep_ckpts_r2/OF_00_apple_eat_1_SPHERE/model_7310.pt"),
    help="checkpoint whose params/agent.yaml supplies the tracker config; our own training runs do "
         "not save one, so it comes from the checkpoint the config was taken from originally")
ap.add_argument("--num-envs", type=int, default=16)
ap.add_argument("--steps", type=int, default=260)
ap.add_argument("--every", type=int, default=40)
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
from newton_vec_env import NewtonVecEnv

TASK = "Mjlab-ResidualInteract-G1"
cfg = load_env_cfg(TASK, play=False); cfg.scene.num_envs = A.num_envs
agent = load_rl_cfg(TASK)
_apply_cfg_mapping(agent, _yaml.unsafe_load(
    (Path(A.agent_cfg_from).parent / "params" / "agent.yaml").open()))
_s = cfg.actions.get("sonic_action") if isinstance(cfg.actions, dict) else cfg.actions.sonic_action
_s.tracking_start_assist_gain = 0.0; _s.tracking_start_assist_steps = 0
from mjlab.tasks.residual_interact.env_cfgs import set_astra_body_dynamics
set_astra_body_dynamics(cfg)

env = NewtonVecEnv(cfg, A.xml, num_envs=A.num_envs, device="cuda:0",
                   sdf_object_stl=A.sdf_object)
wrapped = RslRlVecEnvWrapper(env)
runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(wrapped, asdict(agent), device="cuda:0")
runner.load(A.checkpoint)
policy = _maybe_wrap_residual_action_stats_policy(TASK, runner, runner.get_inference_policy(device="cuda:0"))
print(f"loaded {os.path.basename(A.checkpoint)}")

obs, _ = env.reset()
obj = env._env.scene["apple"]
print(f"\n{'step':>5} {'track_d_cm':>11} {'reward':>9} {'obj_z':>8} {'rise_cm':>8} {'h2o_m':>8} {'nefc':>7}")
print("-" * 62)
z0 = None
peak = {"rise": -99.0, "min_h2o": 9.9, "min_track": 99.0}
for k in range(A.steps):
  with torch.inference_mode():
    a = policy(obs)
  # the residual wrapper records its state on the runner's env; the feature builders read it here
  for n in ("_residual_last_action_mean", "_residual_last_astra_action_pkl",
            "_residual_last_base_action", "_residual_last_decoder_body_delta",
            "_residual_last_final_action", "_residual_last_raw_residual_action",
            "_residual_last_residual_action", "_residual_last_token_delta"):
    v = getattr(env, n, None)
    if v is not None: setattr(env._env, n, v)
  obs, rew, term, trunc, extras = env.step(a)

  d = float(rmdp._body_link_dist_mean_for_group(env._env, "all", use_tracking_weights=True).mean())
  z = float(obj.data.root_link_pos_w[:, 2].mean())
  if z0 is None: z0 = z
  rise = (z - z0) * 100
  try:    h2o = float(rmdp._tip_distances(env._env).min(dim=-1).values.mean())
  except Exception: h2o = float("nan")
  nefc = int(np.asarray(env.solver.mjw_data.nefc.numpy()).reshape(-1)[0])
  peak["rise"] = max(peak["rise"], rise)
  peak["min_track"] = min(peak["min_track"], d * 100)
  if h2o == h2o: peak["min_h2o"] = min(peak["min_h2o"], h2o)
  if k % A.every == 0 or k == A.steps - 1:
    print(f"{k:5d} {d*100:11.2f} {float(rew.mean()):9.4f} {z:8.3f} {rise:8.2f} {h2o:8.3f} {nefc:7d}")
print("-" * 62)
print(f"best tracking distance {peak['min_track']:.2f} cm   (reward std is 3.00 cm)")
print(f"max object rise {peak['rise']:.2f} cm   min fingertip-object {peak['min_h2o']:.3f} m")
