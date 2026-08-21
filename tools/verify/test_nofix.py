"""Is the simple-body fix destroying the SDF wiring?

restore_simple_bodies replaces solver.mjw_model and solver.mjw_data with put_model/put_data of a
recompiled CPU model. Newton attaches SDF volumes to the warp model when it builds the solver, so
replacing that model wholesale would leave the mesh collider with no field behind it -- which fits
the evidence exactly: SDF meshes work standalone, fail here, and fail identically under every solver
setting because nothing about the solver is wrong.

The fix is also unnecessary for a mesh object: nC=1102 is the correct answer once the body is no
longer a simple free body.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, torch

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--xml", default=os.path.join(HERE, "assets/scene_stapler/scene.xml"))
ap.add_argument("--reference-pkl", required=True)
ap.add_argument("--sdf-object", required=True)
ap.add_argument("--steps", type=int, default=60)
A = ap.parse_args()
os.environ["APPLE_EAT_PKL"] = A.reference_pkl
sys.path.insert(0, os.path.join(HERE, "src"))
import mjw_compat; mjw_compat.apply()
import warp as wp
import mjlab.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg
import newton_vec_env as nve
from newton_vec_env import NewtonVecEnv
import newton_simple_fix as nsf

def trial(label, skip_simple_fix):
  orig = nsf.restore_simple_bodies
  if skip_simple_fix:
    nsf.restore_simple_bodies = lambda *a, **k: {"nC_before": -1, "nC_after": -1}
    nve.restore_simple_bodies = nsf.restore_simple_bodies
  try:
    cfg = load_env_cfg("Mjlab-ResidualInteract-G1", play=False); cfg.scene.num_envs = 4
    s = cfg.actions.get("sonic_action") if isinstance(cfg.actions, dict) else cfg.actions.sonic_action
    s.tracking_start_assist_gain = 0.0; s.tracking_start_assist_steps = 0
    from mjlab.tasks.residual_interact.env_cfgs import set_astra_body_dynamics
    set_astra_body_dynamics(cfg)
    env = NewtonVecEnv(cfg, A.xml, num_envs=4, device="cuda:0", sdf_object_stl=A.sdf_object)
    env.reset()
    a = torch.zeros(4, 69, device="cuda:0")
    first = None
    for k in range(A.steps):
      env.action_manager.advance(a); env.action_term.process_actions(a)
      for _ in range(env.decimation):
        env.action_term.apply_actions()
        env.solver.step(env.state_in, env.state_out, env.control, None, env.physics_dt)
        env.state_in, env.state_out = env.state_out, env.state_in
      env._env.episode_length_buf += 1
      if torch.isnan(wp.to_torch(env.solver.mjw_data.qpos)).any():
        first = k; break
    v = wp.to_torch(env.solver.mjw_data.qvel)
    mv = float(v.abs().max()) if not torch.isnan(v).any() else float("nan")
    obj_z = float(env._env.scene["apple"].data.root_link_pos_w[:, 2].mean())
    print(f"  {label:38s} first_nan={'-' if first is None else first:>4}  "
          f"|qvel|max={mv:<9.4g} obj_z={obj_z:.4f}")
    del env
  finally:
    nsf.restore_simple_bodies = orig
    nve.restore_simple_bodies = orig

print("=== is the spec recompile dropping the SDF? ===")
trial("with simple-body fix (current)", False)
trial("WITHOUT simple-body fix", True)
