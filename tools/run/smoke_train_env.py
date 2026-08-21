"""Build NewtonVecEnv and step it. Finds what the assembly is still missing."""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np, torch, yaml as _yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
import mjw_compat; mjw_compat.apply()
import mjlab.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.scripts.play import _apply_cfg_mapping
from newton_vec_env import NewtonVecEnv

XML = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets/mjlab_scene/scene.xml")
CKPT = os.path.expanduser("~/sweep_ckpts_r2/OF_00_apple_eat_1_SPHERE/model_7310.pt")
TASK = "Mjlab-ResidualInteract-G1"
N = int(os.environ.get("SMOKE_ENVS", "4"))

cfg = load_env_cfg(TASK, play=False); cfg.scene.num_envs = N
agent = load_rl_cfg(TASK)
_apply_cfg_mapping(agent, _yaml.unsafe_load((Path(CKPT).parent / "params" / "agent.yaml").open()))
_s = cfg.actions.get("sonic_action") if isinstance(cfg.actions, dict) else cfg.actions.sonic_action
_s.tracking_start_assist_gain = 0.0; _s.tracking_start_assist_steps = 0
from mjlab.tasks.residual_interact.env_cfgs import set_astra_body_dynamics
set_astra_body_dynamics(cfg)

env = NewtonVecEnv(cfg, XML, num_envs=N, device="cuda:0")
print(f"env built: num_envs={env.num_envs} action_dim={env.action_manager.total_action_dim} "
      f"max_episode_length={env.max_episode_length} step_dt={env.step_dt}")
print(f"  reward terms={len(env.reward_manager.active_terms)} "
      f"termination terms={len(env.termination_manager.active_terms)}")

obs, _ = env.reset()
print(f"  obs groups={len(obs.keys())} example proprio_history={tuple(obs['proprio_history'].shape)}")

torch.manual_seed(0)
for k in range(6):
  a = (torch.rand(N, env.action_manager.total_action_dim, device="cuda:0") - 0.5) * 0.1
  obs, rew, term, extras = env.step(a)
  print(f"  step {k}: reward mean={float(rew.mean()):+.4f}  terminated={int(term.sum())}  "
        f"ep_len={env.episode_length_buf.tolist()}")
  if k == 5:
    # A reward of exactly 0.0000 is the tell for a term that never fires, so the breakdown is
    # inspected rather than trusted: which terms are nonzero, and what their weights are.
    log = env.extras.get("log", {})
    rt = {kk: float(vv.mean()) if hasattr(vv, "mean") else float(vv)
          for kk, vv in log.items() if "Reward" in kk or "reward" in kk}
    nz = {kk: v for kk, v in rt.items() if abs(v) > 1e-9}
    print(f"     reward log entries: {len(rt)}   nonzero: {len(nz)}")
    for kk, v in list(nz.items())[:8]:
      print(f"        {kk} = {v:+.5f}")
    w = {t: float(getattr(c, "weight", 0.0)) for t, c in
         zip(env.reward_manager.active_terms, env.reward_manager._term_cfgs)}
    nzw = {t: v for t, v in w.items() if abs(v) > 1e-12}
    print(f"     terms with nonzero weight: {len(nzw)}/{len(w)}")
    for t, v in list(nzw.items())[:10]:
      print(f"        {t}: weight={v}")
print("SMOKE_OK")
