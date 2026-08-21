"""Find where the NaN comes from, and record what it looks like.

The diagnostic rollout reported NaN object positions from the first step while the training log
showed a perfectly steady reward -- mjlab's observation groups are configured nan_policy="sanitize",
so NaNs become zeros before anything sees them. That is why 3576 iterations reported a flat 0.041 and
a value function that converged to a constant: it was fitting a sanitised placeholder.

Checks qpos/qvel directly at each stage of construction, so the first NaN is attributed to a step
rather than guessed at.
"""
from __future__ import annotations
import argparse, os, sys
from dataclasses import asdict
from pathlib import Path
import numpy as np, torch, yaml as _yaml

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--xml", default=os.path.join(HERE, "assets/scene_stapler/scene.xml"))
ap.add_argument("--reference-pkl", required=True)
ap.add_argument("--sdf-object", default=None)
ap.add_argument("--checkpoint", default=None)
ap.add_argument("--agent-cfg-from", default=os.path.expanduser(
    "~/sweep_ckpts_r2/OF_00_apple_eat_1_SPHERE/model_7310.pt"))
ap.add_argument("--num-envs", type=int, default=4)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--video", default=None)
A = ap.parse_args()

os.environ["APPLE_EAT_PKL"] = A.reference_pkl
sys.path.insert(0, os.path.join(HERE, "src"))
import mjw_compat; mjw_compat.apply()
import warp as wp
import mjlab.tasks  # noqa: F401
from mjlab.rl import RslRlVecEnvWrapper, MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.scripts.play import _apply_cfg_mapping, _maybe_wrap_residual_action_stats_policy
from newton_vec_env import NewtonVecEnv

TASK = "Mjlab-ResidualInteract-G1"
cfg = load_env_cfg(TASK, play=False); cfg.scene.num_envs = A.num_envs
agent = load_rl_cfg(TASK)
_apply_cfg_mapping(agent, _yaml.unsafe_load((Path(A.agent_cfg_from).parent/"params"/"agent.yaml").open()))
_s = cfg.actions.get("sonic_action") if isinstance(cfg.actions, dict) else cfg.actions.sonic_action
_s.tracking_start_assist_gain = 0.0; _s.tracking_start_assist_steps = 0
from mjlab.tasks.residual_interact.env_cfgs import set_astra_body_dynamics
set_astra_body_dynamics(cfg)

env = NewtonVecEnv(cfg, A.xml, num_envs=A.num_envs, device="cuda:0", sdf_object_stl=A.sdf_object)

def state_report(tag):
  q = wp.to_torch(env.solver.mjw_data.qpos)
  v = wp.to_torch(env.solver.mjw_data.qvel)
  print(f"  {tag:28s} qpos nan={int(torch.isnan(q).sum())} |max|={float(q.abs().max()):.4g}   "
        f"qvel nan={int(torch.isnan(v).sum())} |max|={float(v.abs().max()):.4g}")

print("\n=== state at each construction stage ===")
state_report("after build")
env._env.forward(); state_report("after forward")
obs, _ = env.reset(); state_report("after reset")

policy = None
if A.checkpoint:
  wrapped = RslRlVecEnvWrapper(env)
  runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(wrapped, asdict(agent), device="cuda:0")
  runner.load(A.checkpoint)
  policy = _maybe_wrap_residual_action_stats_policy(TASK, runner, runner.get_inference_policy(device="cuda:0"))
  print(f"  policy: {os.path.basename(A.checkpoint)}")

viewer = None
if A.video:
  os.environ.setdefault("PYGLET_HEADLESS", "1")
  import pyglet as _pg; _pg.options["headless"] = True
  import newton.viewer as nv, newton
  viewer = nv.ViewerGL(width=1100, height=680, headless=True)
  flags = wp.to_torch(env.nmodel.shape_flags); sbody = wp.to_torch(env.nmodel.shape_body)
  VIS = int(newton.ShapeFlags.VISIBLE)
  for b in torch.unique(sbody):
    idx = (sbody == b).nonzero(as_tuple=True)[0]
    if int((flags[idx] & VIS).sum()) == 0: flags[idx] |= VIS
  viewer.set_model(env.nmodel)
  cam = viewer.camera
  cam.pos = np.array([1.9, -1.5, 1.4], dtype=np.float32)
  cam.look_at(np.array([0.6, 0.0, 0.85], dtype=np.float32))
  frames = []

print("\n=== rollout ===")
print(f"{'step':>5} {'qpos_nan':>9} {'qvel_max':>10} {'reward':>9} {'ep_len':>7}")
first_nan = None
for k in range(A.steps):
  if policy is not None:
    with torch.inference_mode(): a = policy(obs)
  else:
    a = (torch.rand(A.num_envs, 69, device="cuda:0") - 0.5) * 0.1
  obs, rew, term, trunc, ex = env.step(a)
  q = wp.to_torch(env.solver.mjw_data.qpos); v = wp.to_torch(env.solver.mjw_data.qvel)
  nn = int(torch.isnan(q).sum())
  if nn and first_nan is None:
    first_nan = k
    print(f"  >>> first NaN in qpos at step {k}")
  if k % 20 == 0 or k == A.steps - 1:
    print(f"{k:5d} {nn:9d} {float(v.abs().max()):10.4g} {float(rew.mean()):9.4f} "
          f"{float(env.episode_length_buf.float().mean()):7.1f}")
  if viewer is not None:
    bq = wp.to_torch(env.state_in.body_q)
    xp = wp.to_torch(env.solver.mjw_data.xpos); xq = wp.to_torch(env.solver.mjw_data.xquat)
    nb = xp.shape[1] - 1
    if bq.shape[0] == xp.shape[0] * nb:
      bq[:, 0:3] = xp[:, 1:1+nb].reshape(-1, 3)
      bq[:, 3:7] = xq[:, 1:1+nb].reshape(-1, 4)[:, [1, 2, 3, 0]]
    viewer.begin_frame(k * env.step_dt); viewer.log_state(env.state_in); viewer.end_frame()
    img = viewer.get_frame()
    arr = np.asarray(img.numpy() if hasattr(img, "numpy") else img)
    if arr.dtype != np.uint8: arr = np.clip(arr*255, 0, 255).astype(np.uint8)
    frames.append(arr[..., :3].copy())

print(f"\nfirst NaN step: {first_nan if first_nan is not None else 'none'}")
if viewer is not None and frames:
  import imageio.v2 as im
  w = im.get_writer(A.video, fps=25, codec="libx264", macro_block_size=None, quality=8)
  for f in frames: w.append_data(f)
  w.close(); viewer.close()
  print(f"wrote {A.video} ({len(frames)} frames)")
