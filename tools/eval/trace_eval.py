"""Time-resolved rollout: does the episode survive long enough to reach the grasp, and does it grasp?

Averaging over a 600-step rollout hides the answer. The grasp in this clip happens around reference
frame 380-457, so an episode that starts near frame 0 has to survive ~380 steps before lift_success
can be anything but zero -- and a mean over the whole run divides any success by 600 regardless.
Worse, if episodes terminate and reset early they never reach the grasp at all, which looks identical
to "the policy cannot grasp".

So this reports the series, not the mean: episode length, reference frame, hand-object distance,
contact and lift, sampled over time.
"""
from __future__ import annotations
import argparse, sys, os
from dataclasses import asdict
from pathlib import Path
import numpy as np, torch, yaml as _yaml

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint")
ap.add_argument("--num-envs", type=int, default=32)
ap.add_argument("--steps", type=int, default=700)
ap.add_argument("--every", type=int, default=50)
ap.add_argument("--no-terminations", action="store_true",
                help="what record_policy_video.sh/view_run.sh use: let the episode run the whole "
                     "clip instead of resetting on a fall")
ap.add_argument("--force-start-frame", type=int, default=None)
ap.add_argument("--dump-qpos", default=None, help="npz of per-step qpos + mocap, for rendering")
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
env_cfg = load_env_cfg(TASK, play=True); env_cfg.scene.num_envs = A.num_envs
agent_cfg = load_rl_cfg(TASK)
_apply_cfg_mapping(agent_cfg, _yaml.unsafe_load((ck.parent/"params"/"agent.yaml").open()))
_act = getattr(env_cfg, "actions", None)
_s = _act.get("sonic_action") if isinstance(_act, dict) else getattr(_act, "sonic_action", None)
_s.tracking_start_assist_gain = 0.0; _s.tracking_start_assist_steps = 0
astra = str(getattr(agent_cfg, "base_tracker_kind", "")).strip().lower() == "astra_onnx"
if astra:
  from mjlab.tasks.residual_interact.env_cfgs import set_astra_body_dynamics
  set_astra_body_dynamics(env_cfg)
if A.no_terminations:
  env_cfg.terminations = {}
env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode=None)
u = env.unwrapped
if astra:
  from mjlab.tasks.residual_interact.env_cfgs import install_astra_body_pd, install_object_variant_sizes
  install_astra_body_pd(u); install_object_variant_sizes(u)
if A.force_start_frame is not None:
  u._force_reference_start_frame = int(A.force_start_frame)
wrapped = RslRlVecEnvWrapper(env)
runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(wrapped, asdict(agent_cfg), device="cuda:0")
runner.load(str(ck)); policy = runner.get_inference_policy(device="cuda:0")
# play.py wraps the inference policy here, and the wrapper is not cosmetic: it feeds last_residual
# and last_final_action back into the observation. Without it those two groups stay stale, the policy
# acts on a world it is not in, and nothing reports a problem -- measured as the hand ending up 0.73 m
# from the object where play.py tracks the reference wrist to 1.5 cm.
policy = _maybe_wrap_residual_action_stats_policy(TASK, runner, policy)

from mjlab.tasks.apple_eat import object_pool as _pool
_obj = _pool.active(u)
_z0 = None
_qlog, _mlog = [], []
env.reset()
obs = wrapped.get_observations(); obs = obs[0] if isinstance(obs, tuple) else obs
print(f"\n{'step':>5} {'ep_len':>8} {'obj_z':>8} {'dz_cm':>8} {'h2o_m':>8} {'contact':>8} "
      f"{'lift':>7} {'resets':>7}")
print("-" * 74)
prev_ep = u.episode_length_buf.clone()
resets_total = 0
peak = {"lift": 0.0, "contact": 0.0, "min_h2o": 9.9}
for k in range(A.steps):
  with torch.inference_mode():
    a = policy(obs)
  out = wrapped.step(a); obs = out[0] if isinstance(out, tuple) else out
  ep = u.episode_length_buf
  resets_total += int((ep < prev_ep).sum())
  prev_ep = ep.clone()
  log = u.extras.get("log", {})
  g = lambda key: float(log.get(key, float("nan")))
  peak["lift"] = max(peak["lift"], g("PhaseA/lift_success") if log.get("PhaseA/lift_success") is not None else 0.0)
  peak["contact"] = max(peak["contact"], g("Stage/physical_contact") if log.get("Stage/physical_contact") is not None else 0.0)
  if log.get("Metric/hand_to_obj_dist") is not None:
    peak["min_h2o"] = min(peak["min_h2o"], g("Metric/hand_to_obj_dist"))
  if A.dump_qpos:
    _qlog.append(u.sim.data.qpos[0].detach().float().cpu().numpy().copy())
    _mlog.append((u.sim.data.mocap_pos[0].detach().float().cpu().numpy().copy(),
                  u.sim.data.mocap_quat[0].detach().float().cpu().numpy().copy()))
  z = float(_obj.data.root_link_pos_w[:, 2].mean())
  if _z0 is None:
    _z0 = z
  peak["max_dz_cm"] = max(peak.get("max_dz_cm", -99.0), (z - _z0) * 100.0)
  if k % A.every == 0 or k == A.steps - 1:
    print(f"{k:5d} {float(ep.float().mean()):8.1f} {z:8.3f} {(z-_z0)*100:8.2f} "
          f"{g('Metric/hand_to_obj_dist'):8.3f} {g('Stage/physical_contact'):8.3f} "
          f"{g('PhaseA/lift_success'):7.3f} {resets_total:7d}")
print("-" * 68)
print(f"max object rise = {peak.get('max_dz_cm', float('nan')):.2f} cm  (lift threshold is 3 cm)")
if A.dump_qpos:
  np.savez_compressed(A.dump_qpos, qpos=np.stack(_qlog),
                      mocap_pos=np.stack([m[0] for m in _mlog]),
                      mocap_quat=np.stack([m[1] for m in _mlog]))
  print(f"wrote {A.dump_qpos} ({len(_qlog)} frames)")
print(f"peak lift_success={peak['lift']:.3f}  peak physical_contact={peak['contact']:.3f}  "
      f"min hand_to_obj={peak['min_h2o']:.3f} m  total env-resets={resets_total}")
env.close()
