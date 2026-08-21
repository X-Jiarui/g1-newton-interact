"""mjlab against Newton on one clip: same policy, same object, same config. One number each.

body_link_dist_mean is a pure physics quantity -- it does not depend on reward weights -- so it is
the one metric that can attribute a 14 cm tracking error to the simulator rather than to the task.
Both sides get the analytic sphere so the object representation is not a second variable.
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
ap.add_argument("--num-envs", type=int, default=16)
A = ap.parse_args()
os.environ["APPLE_EAT_PKL"] = A.reference_pkl
sys.path.insert(0, os.path.join(HERE, "src"))
import mjw_compat; mjw_compat.apply()
import warp as wp
import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper, MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.scripts.play import _apply_cfg_mapping, _maybe_wrap_residual_action_stats_policy
from mjlab.tasks.residual_interact import mdp as rmdp
from mjlab.tasks.residual_interact.env_cfgs import (set_astra_body_dynamics, install_astra_body_pd,
                                                    install_object_variant_sizes)

def make_cfg(n):
  c = load_env_cfg("Mjlab-ResidualInteract-G1", play=False); c.scene.num_envs = n
  a = load_rl_cfg("Mjlab-ResidualInteract-G1")
  _apply_cfg_mapping(a, _yaml.unsafe_load((Path(A.checkpoint).parent/"params"/"agent.yaml").open()))
  s = c.actions.get("sonic_action") if isinstance(c.actions, dict) else c.actions.sonic_action
  s.tracking_start_assist_gain = 0.0; s.tracking_start_assist_steps = 0
  set_astra_body_dynamics(c)
  return c, a

def summarise(name, dists):
  d = np.array(dists)
  print(f"  {name:<26} mean {d.mean()*100:6.2f} cm   median {np.median(d)*100:6.2f}   "
        f"first10 {d[:10].mean()*100:6.2f}   min {d.min()*100:6.2f}")

# ---------------- mjlab ----------------
cfg, agent = make_cfg(A.num_envs)
mjenv = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
u = mjenv.unwrapped
install_astra_body_pd(u); install_object_variant_sizes(u)
u._force_reference_start_frame = 0
wrapped = RslRlVecEnvWrapper(mjenv)
runner = (load_runner_cls("Mjlab-ResidualInteract-G1") or MjlabOnPolicyRunner)(
    wrapped, asdict(agent), device="cuda:0")
runner.load(A.checkpoint)
policy = _maybe_wrap_residual_action_stats_policy("Mjlab-ResidualInteract-G1", runner,
                                                  runner.get_inference_policy(device="cuda:0"))
obs = wrapped.get_observations(); obs = obs[0] if isinstance(obs, tuple) else obs
dm = []
for k in range(A.steps):
  with torch.inference_mode(): a = policy(obs)
  out = wrapped.step(a); obs = out[0] if isinstance(out, tuple) else out
  dm.append(float(rmdp._body_link_dist_mean_for_group(u, "all", use_tracking_weights=True).mean()))
print(f"\n=== {os.path.basename(A.reference_pkl)} ===")
summarise("mjlab", dm)
mjenv.close()

# ---------------- newton ----------------
from newton_vec_env import NewtonVecEnv
cfg2, agent2 = make_cfg(A.num_envs)
nenv = NewtonVecEnv(cfg2, A.xml, num_envs=A.num_envs, device="cuda:0")
nenv._env._force_reference_start_frame = 0
w2 = RslRlVecEnvWrapper(nenv)
r2 = (load_runner_cls("Mjlab-ResidualInteract-G1") or MjlabOnPolicyRunner)(
    w2, asdict(agent2), device="cuda:0")
r2.load(A.checkpoint)
p2 = _maybe_wrap_residual_action_stats_policy("Mjlab-ResidualInteract-G1", r2,
                                              r2.get_inference_policy(device="cuda:0"))
obs2, _ = nenv.reset()
dn = []
for k in range(A.steps):
  with torch.inference_mode(): a = p2(obs2)
  for n_ in ("_residual_last_action_mean","_residual_last_astra_action_pkl","_residual_last_base_action",
             "_residual_last_decoder_body_delta","_residual_last_final_action",
             "_residual_last_raw_residual_action","_residual_last_residual_action",
             "_residual_last_token_delta"):
    v = getattr(nenv, n_, None)
    if v is not None: setattr(nenv._env, n_, v)
  obs2, rew, term, trunc, ex = nenv.step(a)
  dn.append(float(rmdp._body_link_dist_mean_for_group(nenv._env, "all", use_tracking_weights=True).mean()))
summarise("newton", dn)
