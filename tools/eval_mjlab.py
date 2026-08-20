"""Measure a checkpoint's grasp performance in mjlab, following play.py's construction exactly.

Picking "the good checkpoint" by filename or iteration count is guesswork, and the numbers this
produces are also the baseline Newton has to reproduce -- so the construction has to be play.py's,
not an approximation of it. Custom rollouts have silently diverged from play.py here before and
produced three wrong verdicts. The steps that matter, in order:

  1. load_env_cfg(play=True) and overlay the checkpoint's OWN params/agent.yaml -- the task default
     points tracker_ckpt at another project's checkpoint and will not load
  2. pelvis start-assist gain -> 0.0. play.py defaults it to 1.5 for 120 steps while every candidate
     trained with it disabled, so the default would prop the robot up with a wrench it never saw
  3. set_astra_body_dynamics(env_cfg) BEFORE construction
  4. install_astra_body_pd(env) and install_object_variant_sizes(env) AFTER construction

lift_success is averaged over whatever start frames the RSI draws, so the window is reported with
the number rather than left implicit -- the same checkpoint scores differently under different
windows, and a bare percentage is not comparable to anything.
"""
from __future__ import annotations
import argparse, sys, json, os
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import numpy as np, torch, yaml as _yaml

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint")
ap.add_argument("--num-envs", type=int, default=64)
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--force-start-frame", type=int, default=None)
ap.add_argument("--out", default=None)
A = ap.parse_args()

# mjlab targets mujoco_warp 3.8 and sets options 3.9.1 removed. Newton 1.5 pins 3.11, so the shim
# goes in before any mjlab Simulation is built. It patches the new library, never mjlab: mjlab is the
# baseline, and editing it would mean the reference and the port stopped running identical code.
sys.path.insert(0, os.path.expanduser("~/projects/g1-newton-interact/src"))
try:
  import mjw_compat as _mjw_compat
  _patched = _mjw_compat.apply()
  if _patched:
    print(f"[compat] tolerating removed mujoco_warp options: {_patched}")
except Exception as _e:
  print(f"[compat] shim unavailable ({type(_e).__name__}: {_e})")

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper, MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.scripts.play import _apply_cfg_mapping, _maybe_wrap_residual_action_stats_policy

TASK = "Mjlab-ResidualInteract-G1"
ck = Path(A.checkpoint)
env_cfg = load_env_cfg(TASK, play=True)
env_cfg.scene.num_envs = A.num_envs
agent_cfg = load_rl_cfg(TASK)
cfgp = ck.parent / "params" / "agent.yaml"
if not cfgp.exists():
  raise SystemExit(f"missing {cfgp}")
_apply_cfg_mapping(agent_cfg, _yaml.unsafe_load(cfgp.open()))

_act = getattr(env_cfg, "actions", None)
_sonic = _act.get("sonic_action") if isinstance(_act, dict) else getattr(_act, "sonic_action", None)
_sonic.tracking_start_assist_gain = 0.0
_sonic.tracking_start_assist_steps = 0

astra = str(getattr(agent_cfg, "base_tracker_kind", "")).strip().lower() == "astra_onnx"
if astra:
  from mjlab.tasks.residual_interact.env_cfgs import set_astra_body_dynamics
  set_astra_body_dynamics(env_cfg)

env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode=None)
u = env.unwrapped
if astra:
  from mjlab.tasks.residual_interact.env_cfgs import install_astra_body_pd, install_object_variant_sizes
  install_astra_body_pd(u)
  install_object_variant_sizes(u)

if A.force_start_frame is not None:
  u._force_reference_start_frame = int(A.force_start_frame)

wrapped = RslRlVecEnvWrapper(env)
runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(wrapped, asdict(agent_cfg), device="cuda:0")
runner.load(str(ck))
policy = runner.get_inference_policy(device="cuda:0")
# play.py wraps the inference policy here, and the wrapper is not cosmetic: it feeds last_residual
# and last_final_action back into the observation. Without it those two groups stay stale, the policy
# acts on a world it is not in, and nothing reports a problem -- measured as the hand ending up 0.73 m
# from the object where play.py tracks the reference wrist to 1.5 cm.
policy = _maybe_wrap_residual_action_stats_policy(TASK, runner, policy)

KEYS = ["PhaseA/lift_success", "PhaseA/lift_duration_s", "PhaseA/sequence_success",
        "PhaseA/live_contact_006", "PhaseA/object_mpjpe_mm", "Stage/physical_contact",
        "Metric/hand_to_obj_dist", "Metric/hand_to_obj_under_005_frac"]

env.reset()
obs = wrapped.get_observations()
obs = obs[0] if isinstance(obs, tuple) else obs
acc = defaultdict(list)
start_frames = set()
for k in range(A.steps):
  with torch.inference_mode():
    a = policy(obs)
  out = wrapped.step(a)
  obs = out[0] if isinstance(out, tuple) else out
  log = u.extras.get("log", {})
  for key in KEYS:
    v = log.get(key)
    if v is not None:
      acc[key].append(float(v))
  sf = getattr(u, "_reference_start_frame", None)
  if sf is not None and k == 5:
    start_frames.update(np.unique(sf.detach().cpu().numpy()).tolist())

res = {k: (float(np.mean(v)) if v else None) for k, v in acc.items()}
res["_peak_lift_success"] = float(np.max(acc["PhaseA/lift_success"])) if acc["PhaseA/lift_success"] else None
res["_final_lift_success"] = float(acc["PhaseA/lift_success"][-1]) if acc["PhaseA/lift_success"] else None
res["_checkpoint"] = str(ck)
res["_num_envs"] = A.num_envs
res["_steps"] = A.steps
res["_forced_start_frame"] = A.force_start_frame
res["_rsi_start_frames_seen"] = sorted(start_frames)[:12]

label = f"{ck.parent.parent.name}/{ck.parent.name}/{ck.name}" if ck.parent.name != "params" else str(ck)
print("\n" + "=" * 78)
print(f"CKPT {label}")
print(f"  envs={A.num_envs} steps={A.steps} forced_start_frame={A.force_start_frame} "
      f"rsi_frames_seen={res['_rsi_start_frames_seen']}")
for k in KEYS:
  v = res.get(k)
  print(f"  {k:36s} {'--' if v is None else f'{v:.4f}'}")
print(f"  {'lift_success (peak / final)':36s} "
      f"{res['_peak_lift_success']} / {res['_final_lift_success']}")
if A.out:
  Path(A.out).write_text(json.dumps(res, indent=1))
env.close()
