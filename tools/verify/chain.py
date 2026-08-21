"""Walk the causal chain at the first divergent step and bisect the physics step itself.

Four links, each tested rather than inferred:

  1. same observations?      all 20 groups, both sides, at the step where they still agree
  2. same policy output?     the SAME policy object run on each side's observations
  3. same ctrl?              after process_actions/apply_actions, what reached mjw_data.ctrl
  4. same step result?       ONE physics substep from state that has been made bit-identical

Link 4 is the bisection that matters. Everything mjlab's mjw_data holds is copied into Newton's --
every array they share by name and shape -- so the two simulators enter the substep from provably the
same state. If they still part company, the difference is in the step itself (model or kernel path),
and the fields that could NOT be copied are named, because a shape mismatch there is a difference in
the model rather than the state.
"""
from __future__ import annotations
import os, sys
from dataclasses import asdict
from pathlib import Path
import numpy as np, torch, yaml as _yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
import mjw_compat; mjw_compat.apply()
import newton, mujoco, warp as wp
from newton.solvers import SolverMuJoCo
import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper, MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.scripts.play import _apply_cfg_mapping, _maybe_wrap_residual_action_stats_policy
from mjlab.tasks.apple_eat import mdp as amdp

XML = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets/mjlab_scene/scene.xml")
CKPT = os.path.expanduser("~/sweep_ckpts_r2/OF_00_apple_eat_1_SPHERE/model_7310.pt")
TASK = "Mjlab-ResidualInteract-G1"
WARMUP = int(os.environ.get("CHAIN_WARMUP", "31"))   # steps of identical hard-hold before the test

cfg = load_env_cfg(TASK, play=True); cfg.scene.num_envs = 1
agent_cfg = load_rl_cfg(TASK)
_apply_cfg_mapping(agent_cfg, _yaml.unsafe_load((Path(CKPT).parent/"params"/"agent.yaml").open()))
_s = cfg.actions.get("sonic_action") if isinstance(cfg.actions, dict) else cfg.actions.sonic_action
_s.tracking_start_assist_gain = 0.0; _s.tracking_start_assist_steps = 0
from mjlab.tasks.residual_interact.env_cfgs import (set_astra_body_dynamics, install_astra_body_pd,
                                                    install_object_variant_sizes)
set_astra_body_dynamics(cfg); cfg.terminations = {}
mjenv = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
u = mjenv.unwrapped
install_astra_body_pd(u); install_object_variant_sizes(u)
u._force_reference_start_frame = 0
wrapped = RslRlVecEnvWrapper(mjenv)
runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(wrapped, asdict(agent_cfg), device="cuda:0")
runner.load(CKPT)
policy = _maybe_wrap_residual_action_stats_policy(TASK, runner, runner.get_inference_policy(device="cuda:0"))

b = newton.ModelBuilder(); SolverMuJoCo.register_custom_attributes(b)
b.default_shape_cfg.gap = 0.0
b.add_mjcf(XML, collapse_fixed_joints=False, parse_mujoco_options=True)
nm = b.finalize()
sv = SolverMuJoCo(nm, enable_multiccd=True, update_data_interval=0, njmax=2048, nconmax=256,
                  ccd_iterations=50)
refmj = mujoco.MjModel.from_xml_path(XML)
dd = sv.mjw_model.dof_damping
dd.assign(refmj.dof_damping.astype(dd.numpy().dtype).reshape(dd.numpy().shape))
control = nm.control()
from newton_bridge import NewtonEnv
nenv = NewtonEnv(sv.mj_model, sv.mjw_data, 1, "cuda:0", control=control, rename_from=refmj,
                 physics_dt=0.005, decimation=4, solver=sv)
nenv._force_reference_start_frame = 0

ev = cfg.events if isinstance(cfg.events, dict) else vars(cfg.events)
rc = ev["reset_to_residual_interact_curriculum"]; rp = dict(getattr(rc, "params", {}) or {})
mjenv.reset(); rc.func(u, None, **rp)
rc.func(nenv, None, **rp); nenv.forward()

nt_cfg = cfg.actions.get("sonic_action") if isinstance(cfg.actions, dict) else cfg.actions.sonic_action
nt_term = amdp.Sonic53Action(nt_cfg, nenv)
nenv.bind_action_manager(nt_term.action_dim, {"sonic_action": nt_term})

obs_cfg = cfg.observations if isinstance(cfg.observations, dict) else vars(cfg.observations)
builders_mj, builders_nt = {}, {}
for g, gc in obs_cfg.items():
  terms = getattr(gc, "terms", None) or (gc if isinstance(gc, dict) else None)
  if not terms: continue
  t = terms.get("policy") or next(iter(terms.values()))
  try:
    builders_mj[g] = t.func(t, u); builders_nt[g] = t.func(t, nenv)
  except Exception:
    pass

from tensordict import TensorDict as _TD

def _nt_obs():
  # Call EVERY Newton-side builder each step, exactly as the real runner does. Some of these groups
  # carry internal history (astra_obs is the tracker's input), so a builder that is only invoked at
  # the end of a warmup has no history and cannot be compared against one that has been running.
  return _TD({g: builders_nt[g](nenv) for g in builders_nt}, batch_size=[1])

obs = wrapped.get_observations(); obs = obs[0] if isinstance(obs, tuple) else obs
for k in range(WARMUP):
  _nt_obs()                      # keep Newton's builders in step with mjlab's
  with torch.inference_mode():
    a = policy(obs)
  nenv.action_manager.advance(a); nt_term.process_actions(a.clone())
  for _ in range(4):
    nt_term.apply_actions()
    sv.step(nm.state(), nm.state(), control, None, 0.005)
  nenv.episode_length_buf += 1
  out = wrapped.step(a); obs = out[0] if isinstance(out, tuple) else out

qm = lambda: u.sim.data.qpos[0].detach().float().cpu().numpy()
qn = lambda: wp.to_torch(sv.mjw_data.qpos)[0].detach().float().cpu().numpy()
print(f"\nafter {WARMUP} warmup steps: |dqpos|max = {np.abs(qm()-qn()).max():.6g}")

# ---------------- link 1: observations -------------------------------------
print("\n[1] observations")
worst = (0.0, "")
for g in builders_mj:
  try:
    vm = builders_mj[g](u)[0].detach().float().cpu().numpy()
    vn = builders_nt[g](nenv)[0].detach().float().cpu().numpy()
  except Exception:
    continue
  if vm.shape != vn.shape: print(f"    {g}: SHAPE {vm.shape} vs {vn.shape}"); continue
  d = float(np.abs(vm-vn).max())
  if d > worst[0]: worst = (d, g)
print(f"    worst group: {worst[1]} at {worst[0]:.6g}")
ACTOR = ["proprio_history", "tracker_action", "reference_phase", "reference_preview", "object_state",
         "hand_object_geometry", "object_future", "contact_features", "tracking_error",
         "last_residual", "last_final_action", "astra_obs"]
print("    groups the policy and tracker actually read:")
for g in ACTOR:
  if g not in builders_mj: continue
  try:
    vm = builders_mj[g](u)[0].detach().float().cpu().numpy()
    vn = builders_nt[g](nenv)[0].detach().float().cpu().numpy()
    print(f"       {g:24s} {float(np.abs(vm-vn).max()):.6g}")
  except Exception as e:
    print(f"       {g:24s} ERROR {type(e).__name__}")
if worst[0] > 1e-4:
  g = worst[1]
  vm = builders_mj[g](u)[0].detach().float().cpu().numpy()
  vn = builders_nt[g](nenv)[0].detach().float().cpu().numpy()
  bad = np.flatnonzero(np.abs(vm - vn) > 1e-4)
  print(f"    {g}: {len(bad)}/{len(vm)} slots differ, first {bad[:10].tolist()}")
  for i in bad[:6]:
    print(f"       [{int(i)}] mjlab={vm[i]:+.5f} newton={vn[i]:+.5f}")

# ---------------- link 2: policy output ------------------------------------
print("[2] policy output")
obs_nt = _nt_obs()
# The inference policy is wrapped by _maybe_wrap_residual_action_stats_policy, which keeps running
# state. Calling it once per side in sequence therefore compares call 1 against call 2 of a stateful
# object, not mjlab's observations against Newton's. Determinism is checked first so the number below
# means what it appears to mean.
with torch.inference_mode():
  a1 = policy(obs)
  a2 = policy(obs)
det = float((a1 - a2).abs().max())
print(f"    same obs twice -> |d action|max = {det:.6g} "
      f"({'DETERMINISTIC' if det < 1e-6 else 'STATEFUL: sequential comparison is invalid'})")

# A second, independent policy with the same weights, so each side gets a fresh one.
runner2 = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(wrapped, asdict(agent_cfg), device="cuda:0")
runner2.load(CKPT)
policy2 = _maybe_wrap_residual_action_stats_policy(TASK, runner2, runner2.get_inference_policy(device="cuda:0"))
with torch.inference_mode():
  a_mj = policy2(obs)
runner3 = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(wrapped, asdict(agent_cfg), device="cuda:0")
runner3.load(CKPT)
policy3 = _maybe_wrap_residual_action_stats_policy(TASK, runner3, runner3.get_inference_policy(device="cuda:0"))
with torch.inference_mode():
  a_nt = policy3(obs_nt)
print(f"    fresh policy per side: |d action|max = {float((a_mj-a_nt).abs().max()):.6g}   "
      f"norm mj={float(a_mj.norm()):.4f} nt={float(a_nt.norm()):.4f}")
with torch.inference_mode():
  a_ctl = policy3(obs) if False else None

# ---------------- link 3: ctrl ---------------------------------------------
print("[3] action -> ctrl")
act = a_mj.clone()
mj_term = u.action_manager.get_term("sonic_action")
mj_term.process_actions(act.clone()); mj_term.apply_actions()
u.scene["robot"]._apply_actuator_controls()
nenv.action_manager.advance(act); nt_term.process_actions(act.clone()); nt_term.apply_actions()
c_mj = u.sim.data.ctrl[0].detach().float().cpu().numpy()
c_nt = wp.to_torch(control.mujoco.ctrl).view(-1).detach().float().cpu().numpy()
print(f"    |d ctrl|max = {np.abs(c_mj-c_nt).max():.6g}")

# ---------------- link 4: one physics substep from identical data -----------
print("[4] one substep from bit-identical data")
md, nd = u.sim.wp_data, sv.mjw_data
copied, skipped = [], []
for name in sorted(set(dir(md)) & set(dir(nd))):
  if name.startswith("_"): continue
  A_, B_ = getattr(md, name, None), getattr(nd, name, None)
  if not (hasattr(A_, "numpy") and hasattr(B_, "numpy")): continue
  try:
    a_np = A_.numpy(); b_np = B_.numpy()
  except Exception:
    continue
  if a_np.shape != b_np.shape or a_np.dtype != b_np.dtype:
    skipped.append(f"{name}{a_np.shape}vs{b_np.shape}"); continue
  try:
    wp.to_torch(B_).copy_(wp.to_torch(A_)); copied.append(name)
  except Exception:
    skipped.append(name + "(copy failed)")
print(f"    copied {len(copied)} data arrays; could not copy {len(skipped)}")
if skipped:
  print(f"      not copied: {skipped[:10]}")
print(f"    after copy: |dqpos|max={np.abs(qm()-qn()).max():.6g} "
      f"|dqvel|max={np.abs(u.sim.data.qvel[0].detach().float().cpu().numpy() - wp.to_torch(sv.mjw_data.qvel)[0].detach().cpu().numpy()).max():.6g}")

import mujoco_warp as mjw
with wp.ScopedDevice(nm.device):
  mjw.step(u.sim.wp_model, md)
  mjw.step(sv.mjw_model, nd)

def g(d, n):
  x = getattr(d, n, None)
  return None if x is None else np.asarray(x.numpy()).reshape(-1)
for f in ("qpos", "qvel", "qacc", "qfrc_constraint", "qfrc_smooth", "qfrc_bias", "ncon", "nefc"):
  A_, B_ = g(md, f), g(nd, f)
  if A_ is None or B_ is None or A_.shape != B_.shape:
    print(f"    {f:18s} shape {None if A_ is None else A_.shape} vs {None if B_ is None else B_.shape}")
    continue
  print(f"    {f:18s} max|diff| = {np.abs(A_.astype(np.float64)-B_.astype(np.float64)).max():.6g}")
