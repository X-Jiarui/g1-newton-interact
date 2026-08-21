"""Diff Newton's observations against mjlab's, group by group, from the same reset state.

The Newton rollout holds the apple on the table correctly but the arm approaches to 0.22 m and then
retreats, where mjlab closes to 0.035 m and grasps. That is what a policy does when its observations
are subtly wrong -- it is still a working policy, acting on a world that is not the one it is in.

Both sides are reset to reference frame 0 by the same mjlab event, so their states should agree before
a single step is taken. Any group that differs at step 0 is a bridge defect, and the size of the
difference says whether it is a unit/frame error or a wholesale mismatch.

This is only possible because mjlab now runs under mujoco_warp 3.11 alongside Newton -- one process,
one set of kernels, so a difference cannot be blamed on the version.
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
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.scripts.play import _apply_cfg_mapping

XML = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets/mjlab_scene/scene.xml")
CKPT = os.path.expanduser("~/sweep_ckpts_r2/OF_00_apple_eat_1_SPHERE/model_7310.pt")
TASK = "Mjlab-ResidualInteract-G1"

def make_env_cfg():
  c = load_env_cfg(TASK, play=True); c.scene.num_envs = 1
  a = load_rl_cfg(TASK)
  _apply_cfg_mapping(a, _yaml.unsafe_load((Path(CKPT).parent / "params" / "agent.yaml").open()))
  act = getattr(c, "actions", None)
  s = act.get("sonic_action") if isinstance(act, dict) else getattr(act, "sonic_action")
  s.tracking_start_assist_gain = 0.0; s.tracking_start_assist_steps = 0
  if str(getattr(a, "base_tracker_kind", "")).strip().lower() == "astra_onnx":
    from mjlab.tasks.residual_interact.env_cfgs import set_astra_body_dynamics
    set_astra_body_dynamics(c)
  return c, a

env_cfg, agent_cfg = make_env_cfg()
mjenv = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode=None)
u = mjenv.unwrapped
from mjlab.tasks.residual_interact.env_cfgs import install_astra_body_pd, install_object_variant_sizes
install_astra_body_pd(u); install_object_variant_sizes(u)
u._force_reference_start_frame = 0
mjenv.reset()

ev = env_cfg.events if isinstance(env_cfg.events, dict) else vars(env_cfg.events)
rcfg = ev["reset_to_residual_interact_curriculum"]
rparams = dict(getattr(rcfg, "params", {}) or {})
rcfg.func(u, None, **rparams)
u.sim.forward() if hasattr(u.sim, "forward") else None

# ---- Newton side ----
ncfg, _ = make_env_cfg()
b = newton.ModelBuilder(); SolverMuJoCo.register_custom_attributes(b)
b.default_shape_cfg.gap = 0.0
b.add_mjcf(XML, collapse_fixed_joints=False, parse_mujoco_options=True)
nm = b.finalize()
sv = SolverMuJoCo(nm, enable_multiccd=True, update_data_interval=0, njmax=2048, nconmax=256)
refmj = mujoco.MjModel.from_xml_path(XML)
dd = sv.mjw_model.dof_damping
dd.assign(refmj.dof_damping.astype(dd.numpy().dtype).reshape(dd.numpy().shape))
control = nm.control()
from newton_bridge import NewtonEnv
nenv = NewtonEnv(sv.mj_model, sv.mjw_data, 1, "cuda:0", control=control, rename_from=refmj,
                 physics_dt=0.005, decimation=4, solver=sv)
nenv._force_reference_start_frame = 0
rcfg.func(nenv, None, **rparams)
nenv.forward()

# ---- state diff first: if the states differ, the observations must ----
print("\n=== state after reset ===")
for label, e_mj, e_nt in (("robot", u.scene["robot"], nenv.scene["robot"]),
                          ("apple", u.scene["apple"], nenv.scene["apple"])):
  for field in ("joint_pos", "joint_vel", "root_link_pos_w", "root_link_quat_w",
                "root_link_lin_vel_w", "root_link_ang_vel_w", "root_link_ang_vel_b",
                "projected_gravity_b"):
    try:
      a = getattr(e_mj.data, field)[0].detach().float().cpu().numpy()
      bb = getattr(e_nt.data, field)[0].detach().float().cpu().numpy()
      d = np.abs(a - bb).max()
      print(f"  {label}.{field:20s} max|diff| = {d:.6g}")
    except Exception as ex:
      print(f"  {label}.{field:20s} -- {type(ex).__name__}: {ex}")

# ---- observation diff ----
obs_cfg = env_cfg.observations if isinstance(env_cfg.observations, dict) else vars(env_cfg.observations)
print("\n=== observation groups ===")
print(f"{'group':<26}{'dim':>6}{'max|diff|':>14}   verdict")
print("-" * 62)
worst = []
for g, gcfg in obs_cfg.items():
  terms = getattr(gcfg, "terms", None) or (gcfg if isinstance(gcfg, dict) else None)
  if not terms: continue
  t = terms.get("policy") or next(iter(terms.values()))
  try:
    vm = t.func(t, u)(u)[0].detach().float().cpu().numpy()
    vn = t.func(t, nenv)(nenv)[0].detach().float().cpu().numpy()
  except Exception as ex:
    print(f"{g:<26}{'--':>6}{'--':>14}   {type(ex).__name__}: {str(ex)[:40]}")
    continue
  if vm.shape != vn.shape:
    print(f"{g:<26}{'--':>6}{'--':>14}   SHAPE {vm.shape} vs {vn.shape}")
    continue
  d = float(np.abs(vm - vn).max())
  worst.append((d, g))
  verdict = "same" if d < 1e-4 else ("close" if d < 1e-2 else "DIFFERS")
  print(f"{g:<26}{vm.shape[0]:>6}{d:>14.6g}   {verdict}")
  if d >= 1e-4:
    # Which slots differ? Contiguous runs point at a specific sub-field of the group.
    bad = np.flatnonzero(np.abs(vm - vn) > 1e-4)
    runs, start = [], bad[0] if len(bad) else None
    for i in range(1, len(bad) + 1):
      if i == len(bad) or bad[i] != bad[i - 1] + 1:
        runs.append((int(start), int(bad[i - 1]))); start = bad[i] if i < len(bad) else None
    print(f"      differing slots ({len(bad)}/{vm.shape[0]}): "
          f"{[f'{a}-{b}' if a != b else str(a) for a, b in runs][:8]}")
    for i in bad[:5]:
      print(f"        [{int(i)}] mjlab={vm[i]:+.5f}  newton={vn[i]:+.5f}  d={vm[i]-vn[i]:+.5f}")
print("-" * 62)
for d, g in sorted(worst, reverse=True)[:3]:
  print(f"  largest: {g} ({d:.4g})")

# ---- action path: same action in, same ctrl out? -------------------------------
# Observations agreeing at reset does not mean the rollout agrees. If the 69 action values are routed
# to different actuators on the two sides, the policy is correct and the robot still does the wrong
# thing -- the exact failure this port is built to rule out. Feed both action terms an IDENTICAL
# action and diff the resulting ctrl vectors.
print("\n=== action -> ctrl ===")
from mjlab.tasks.apple_eat import mdp as amdp
n_act = 69
torch.manual_seed(0)
act = (torch.rand(1, n_act, device="cuda:0") - 0.5) * 0.2

mj_term = u.action_manager.get_term("sonic_action")
nt_cfg = (ncfg.actions.get("sonic_action") if isinstance(ncfg.actions, dict)
          else getattr(ncfg.actions, "sonic_action"))
nt_term = amdp.Sonic53Action(nt_cfg, nenv)
nenv.bind_action_manager(nt_term.action_dim, {"sonic_action": nt_term})

mj_term.process_actions(act.clone()); mj_term.apply_actions()
# mjlab stages position targets in data.joint_pos_target; an actuator layer then maps them onto ctrl
# slots via explicit (target_ids -> ctrl_ids) groups. Without this call mjlab's ctrl reads all zeros
# and the comparison is meaningless.
u.scene["robot"]._apply_actuator_controls()
nt_term.process_actions(act.clone()); nt_term.apply_actions()

ctrl_mj = u.sim.data.ctrl[0].detach().float().cpu().numpy()
ctrl_nt = wp.to_torch(control.mujoco.ctrl).view(-1).detach().float().cpu().numpy()
print(f"  ctrl dims: mjlab {ctrl_mj.shape} newton {ctrl_nt.shape}")
d = np.abs(ctrl_mj - ctrl_nt)
print(f"  max|diff| = {d.max():.6g}   nonzero slots: mjlab {int((ctrl_mj!=0).sum())} "
      f"newton {int((ctrl_nt!=0).sum())}")
bad = np.flatnonzero(d > 1e-4)
if len(bad):
    print(f"  {len(bad)} of {len(d)} ctrl slots differ; first few:")
    for i in bad[:8]:
        nm = mujoco.mj_id2name(u.sim.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(i)) or f"<{i}>"
        print(f"    [{int(i):3d}] {nm:44s} mjlab={ctrl_mj[i]:+.5f} newton={ctrl_nt[i]:+.5f}")
else:
    print("  ctrl vectors agree -- the action path is not the divergence")

# Which actuator does mjlab consider real for each joint? The bridge picks by widest forcerange;
# mjlab knows from its actuator config. A disagreement here routes every command one slot off.
try:
  rb = u.scene["robot"]
  mj_ctrl_ids = rb.indexing.ctrl_ids.detach().cpu().numpy().tolist()
  nt_ctrl_ids = nenv.scene["robot"].indexing.ctrl_ids.detach().cpu().numpy().tolist()
  print(f"\n  mjlab entity ctrl_ids: {len(mj_ctrl_ids)}  newton: {len(nt_ctrl_ids)}")
  print(f"  identical set: {sorted(mj_ctrl_ids) == sorted(nt_ctrl_ids)}")
  only_mj = sorted(set(mj_ctrl_ids) - set(nt_ctrl_ids))[:6]
  only_nt = sorted(set(nt_ctrl_ids) - set(mj_ctrl_ids))[:6]
  if only_mj or only_nt:
    print(f"  only mjlab: {only_mj}   only newton: {only_nt}")
except Exception as ex:
  print(f"  ctrl id comparison unavailable: {type(ex).__name__}: {ex}")
