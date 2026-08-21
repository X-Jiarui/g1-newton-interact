"""Which solver settings make SDF mesh contact stable?

Same scene and clip are stable with the sphere collider and blow up in the first step with the SDF
mesh, so the mesh contact is what the solver cannot resolve. mjlab's scene is authored for convex
primitives -- pyramidal cone, 10 solver iterations, impratio 1 -- while Newton's own SDF/hydroelastic
example uses an elliptic cone, impratio 1000 and far more iterations. This sweeps those.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, torch

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--xml", default=os.path.join(HERE, "assets/scene_stapler/scene.xml"))
ap.add_argument("--reference-pkl", required=True)
ap.add_argument("--sdf-object", required=True)
ap.add_argument("--steps", type=int, default=40)
A = ap.parse_args()

os.environ["APPLE_EAT_PKL"] = A.reference_pkl
sys.path.insert(0, os.path.join(HERE, "src"))
import mjw_compat; mjw_compat.apply()
import warp as wp
import mjlab.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg
from newton_vec_env import NewtonVecEnv

CONE = {"pyramidal": 0, "elliptic": 1}

def trial(label, iterations=None, ls_iterations=None, cone=None, impratio=None, sdf_iters=None):
  cfg = load_env_cfg("Mjlab-ResidualInteract-G1", play=False); cfg.scene.num_envs = 4
  s = cfg.actions.get("sonic_action") if isinstance(cfg.actions, dict) else cfg.actions.sonic_action
  s.tracking_start_assist_gain = 0.0; s.tracking_start_assist_steps = 0
  from mjlab.tasks.residual_interact.env_cfgs import set_astra_body_dynamics
  set_astra_body_dynamics(cfg)
  env = NewtonVecEnv(cfg, A.xml, num_envs=4, device="cuda:0", sdf_object_stl=A.sdf_object)

  o = env.solver.mjw_model.opt
  def setopt(name, val):
    if val is None: return
    a = getattr(o, name, None)
    if a is None: return
    if hasattr(a, "fill_"): a.fill_(val)
    else: setattr(o, name, val)
  setopt("iterations", iterations); setopt("ls_iterations", ls_iterations)
  setopt("cone", cone); setopt("impratio", impratio)

  env.reset()
  q = lambda: wp.to_torch(env.solver.mjw_data.qpos)
  a = torch.zeros(4, 69, device="cuda:0")
  first_nan, maxv = None, 0.0
  for k in range(A.steps):
    env.action_manager.advance(a); env.action_term.process_actions(a)
    for _ in range(env.decimation):
      env.action_term.apply_actions()
      env.solver.step(env.state_in, env.state_out, env.control, None, env.physics_dt)
      env.state_in, env.state_out = env.state_out, env.state_in
    env._env.episode_length_buf += 1
    v = wp.to_torch(env.solver.mjw_data.qvel)
    if not torch.isnan(v).any():
      maxv = max(maxv, float(v.abs().max()))
    if torch.isnan(q()).any() and first_nan is None:
      first_nan = k; break
  print(f"  {label:44s} first_nan={'-' if first_nan is None else first_nan:>4}  "
        f"|qvel|max={maxv:.4g}")
  del env

print("=== SDF mesh contact stability ===")
trial("mjlab defaults (pyramidal, 10 iter, impratio 1)")
trial("iterations=50", iterations=50)
trial("iterations=100, ls=50", iterations=100, ls_iterations=50)
trial("elliptic cone", cone=CONE["elliptic"])
trial("elliptic + impratio=100", cone=CONE["elliptic"], impratio=100.0)
trial("elliptic + impratio=1000 + 15/100 iters",
      cone=CONE["elliptic"], impratio=1000.0, iterations=15, ls_iterations=100)
