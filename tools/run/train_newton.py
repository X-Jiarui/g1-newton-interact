"""Train the residual policy with Newton doing the physics.

Reuses mjlab's PPO runner and its entire MDP -- observations, action term, rewards, terminations,
events -- against `NewtonVecEnv`. Only the simulator changes.

Validated before this script was written: all 20 observation groups agree with mjlab from the same
state, the ctrl vectors are identical for identical actions, and the reward matches mjlab exactly
while the two states agree (both 0 through the startup hold) and diverges only when the physics does.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml as _yaml

ap = argparse.ArgumentParser()
ap.add_argument("--num-envs", type=int, default=256)
ap.add_argument("--iterations", type=int, default=2000)
ap.add_argument("--xml", default=os.path.expanduser(
  "~/projects/g1-newton-interact/assets/mjlab_scene/scene.xml"))
ap.add_argument("--agent-cfg-from", default=os.path.expanduser(
  "~/sweep_ckpts_r2/OF_00_apple_eat_1_SPHERE/model_7310.pt"),
  help="checkpoint whose params/agent.yaml supplies the agent config (tracker, residual gains)")
ap.add_argument("--resume", default=None, help="checkpoint to warm-start from")
ap.add_argument("--run-name", default="NEWTON_NATIVE")
ap.add_argument("--log-root", default=os.path.expanduser(
  "~/projects/g1-newton-interact/logs/rsl_rl"))
ap.add_argument("--seed", type=int, default=42)
A = ap.parse_args()

sys.path.insert(0, os.path.expanduser("~/projects/g1-newton-interact/src"))
import mjw_compat
_p = mjw_compat.apply()
if _p:
  print(f"[compat] tolerating removed mujoco_warp options: {_p}")

import mjlab.tasks  # noqa: F401
from mjlab.rl import RslRlVecEnvWrapper, MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.scripts.play import _apply_cfg_mapping
from newton_vec_env import NewtonVecEnv

TASK = "Mjlab-ResidualInteract-G1"
torch.manual_seed(A.seed)
np.random.seed(A.seed)

cfg = load_env_cfg(TASK, play=False)
cfg.scene.num_envs = A.num_envs
agent_cfg = load_rl_cfg(TASK)
p = Path(A.agent_cfg_from).parent / "params" / "agent.yaml"
if not p.exists():
  raise SystemExit(f"missing {p}: the task default agent config points at another project's tracker")
_apply_cfg_mapping(agent_cfg, _yaml.unsafe_load(p.open()))

# Every candidate trained with the pelvis start-assist disabled; the task default is 1.5 for 120
# steps, which would prop the robot up with a wrench it never sees at evaluation.
_s = cfg.actions.get("sonic_action") if isinstance(cfg.actions, dict) else cfg.actions.sonic_action
_s.tracking_start_assist_gain = 0.0
_s.tracking_start_assist_steps = 0
if str(getattr(agent_cfg, "base_tracker_kind", "")).strip().lower() == "astra_onnx":
  from mjlab.tasks.residual_interact.env_cfgs import set_astra_body_dynamics
  set_astra_body_dynamics(cfg)

print(f"building {A.num_envs} Newton worlds ...")
env = NewtonVecEnv(cfg, A.xml, num_envs=A.num_envs, device="cuda:0")
print(f"  reward terms={len(env.reward_manager.active_terms)} "
      f"termination terms={len(env.termination_manager.active_terms)} "
      f"max_episode_length={env.max_episode_length}")

wrapped = RslRlVecEnvWrapper(env)
log_dir = Path(A.log_root) / "g1_residual_interact" / A.run_name
log_dir.mkdir(parents=True, exist_ok=True)
runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
runner = runner_cls(wrapped, asdict(agent_cfg), log_dir=str(log_dir), device="cuda:0")
if A.resume:
  runner.load(A.resume)
  print(f"warm-started from {A.resume}")

print(f"training for {A.iterations} iterations -> {log_dir}")
runner.learn(num_learning_iterations=A.iterations, init_at_random_ep_len=True)
