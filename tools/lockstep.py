"""Step mjlab and Newton on the SAME actions and watch where their states part company.

Observations agree at reset and the ctrl vectors agree for an identical action, yet the rollouts end
differently: mjlab lifts the apple 50 cm, Newton's hand closes to 0.20 m and retreats. So the
divergence is not in what the policy sees or in how commands are routed -- it appears while stepping.

Feeding both simulators one shared action sequence separates the two possibilities that remain:
a difference that shows up on the very first step is in the step call itself (options, contacts,
control application); one that grows from nothing is ordinary chaos, and its growth rate says whether
it can explain a qualitative change of outcome.

Actions come from the mjlab side so that Newton is never asked to produce them -- this measures the
simulators, not the policy.
"""
from __future__ import annotations
import os, sys
from dataclasses import asdict
from pathlib import Path
import numpy as np, torch, yaml as _yaml

sys.path.insert(0, os.path.expanduser("~/projects/g1-newton-interact/src"))
import mjw_compat; mjw_compat.apply()

import newton, mujoco, warp as wp
from newton.solvers import SolverMuJoCo
import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper, MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.scripts.play import _apply_cfg_mapping, _maybe_wrap_residual_action_stats_policy
from mjlab.tasks.apple_eat import mdp as amdp

XML = os.path.expanduser("~/projects/g1-newton-interact/assets/mjlab_scene/scene.xml")
CKPT = os.path.expanduser("~/sweep_ckpts_r2/OF_00_apple_eat_1_SPHERE/model_7310.pt")
TASK = "Mjlab-ResidualInteract-G1"
STEPS = int(os.environ.get("LOCKSTEP_STEPS", "120"))

cfg = load_env_cfg(TASK, play=True); cfg.scene.num_envs = 1
agent_cfg = load_rl_cfg(TASK)
_apply_cfg_mapping(agent_cfg, _yaml.unsafe_load((Path(CKPT).parent/"params"/"agent.yaml").open()))
_s = cfg.actions.get("sonic_action") if isinstance(cfg.actions, dict) else cfg.actions.sonic_action
_s.tracking_start_assist_gain = 0.0; _s.tracking_start_assist_steps = 0
from mjlab.tasks.residual_interact.env_cfgs import (set_astra_body_dynamics, install_astra_body_pd,
                                                    install_object_variant_sizes)
set_astra_body_dynamics(cfg)
mjenv = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
u = mjenv.unwrapped
install_astra_body_pd(u); install_object_variant_sizes(u)
u._force_reference_start_frame = 0
cfg.terminations = {}
wrapped = RslRlVecEnvWrapper(mjenv)
runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(wrapped, asdict(agent_cfg), device="cuda:0")
runner.load(CKPT)
policy = _maybe_wrap_residual_action_stats_policy(TASK, runner, runner.get_inference_policy(device="cuda:0"))

b = newton.ModelBuilder(); SolverMuJoCo.register_custom_attributes(b)
b.default_shape_cfg.gap = 0.0
b.add_mjcf(XML, collapse_fixed_joints=False, parse_mujoco_options=True)
nm = b.finalize()
# ccd_iterations measured at 35 on the Newton side against mjlab's 50. Contact resolution is exactly
# where a grasp lives, so it is matched rather than left to each library's default.
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
mjenv.reset()
rc.func(u, None, **rp)
rc.func(nenv, None, **rp); nenv.forward()

nt_cfg = cfg.actions.get("sonic_action") if isinstance(cfg.actions, dict) else cfg.actions.sonic_action
nt_term = amdp.Sonic53Action(nt_cfg, nenv)
nenv.bind_action_manager(nt_term.action_dim, {"sonic_action": nt_term})

# The compiled CPU models were verified field-for-field, but the thing that actually steps is the
# WARP model, and its opt block carries fields the CPU model does not (broadphase, graph conditional,
# collision toggles, iteration caps). A difference there changes the physics while every earlier
# comparison still says the models match.
print("\n=== warp model opt: mjlab vs newton ===")
mo, no = u.sim.wp_model.opt, sv.mjw_model.opt
def _v(o, n):
    a = getattr(o, n, "<absent>")
    try:
        a = a.numpy() if hasattr(a, "numpy") else a
        a = np.asarray(a).reshape(-1)
        return a[0] if a.size == 1 else tuple(a[:4].tolist())
    except Exception:
        return a
fields = sorted(set([n for n in dir(mo) if not n.startswith("_")]) |
                set([n for n in dir(no) if not n.startswith("_")]))
ndiff = 0
for n in fields:
    try:
        a, bv = _v(mo, n), _v(no, n)
    except Exception:
        continue
    if callable(a) or callable(bv):
        continue
    same = (str(a) == str(bv))
    if not same:
        ndiff += 1
        print(f"  {n:28s} mjlab={str(a)[:28]:<30} newton={str(bv)[:28]}")
print(f"  ({ndiff} differing opt fields)")

# The opt block is not the whole warp model. Everything the CPU comparison covered is re-checked here
# at the warp level, because that is the model the kernels actually read.
print("\n=== warp model arrays ===")
arrs = ["actuator_ctrllimited", "actuator_ctrlrange", "actuator_gear", "actuator_trntype",
        "actuator_gaintype", "actuator_biastype", "dof_armature", "dof_damping", "dof_frictionloss", "jnt_range", "jnt_stiffness",
        "actuator_gainprm", "actuator_biasprm", "actuator_forcerange", "actuator_ctrlrange",
        "body_mass", "body_inertia", "geom_solref", "geom_solimp", "geom_friction",
        "geom_margin", "geom_gap", "geom_condim", "geom_priority", "geom_size"]
for n in arrs:
    A = getattr(u.sim.wp_model, n, None); B = getattr(sv.mjw_model, n, None)
    if A is None or B is None:
        print(f"  {n:22s} absent on {'mjlab' if A is None else 'newton'}"); continue
    a = np.asarray(A.numpy() if hasattr(A, "numpy") else A).reshape(-1).astype(np.float64)
    bb = np.asarray(B.numpy() if hasattr(B, "numpy") else B).reshape(-1).astype(np.float64)
    if a.shape != bb.shape:
        print(f"  {n:22s} SHAPE {a.shape} vs {bb.shape}"); continue
    d = np.abs(a - bb).max() if a.size else 0.0
    if d > 1e-5:
        print(f"  {n:22s} max|diff| = {d:.6g}   <-- DIFFERS")
        if n == "body_inertia":
            aa, bbb = a.reshape(-1, 3), bb.reshape(-1, 3)
            dd_ = np.abs(aa - bbb).max(axis=1)
            for i in np.argsort(-dd_)[:6]:
                nm_ = mujoco.mj_id2name(refmj, mujoco.mjtObj.mjOBJ_BODY, int(i)) or f"<{i}>"
                print(f"        body {int(i):3d} {nm_:28s} mjlab={np.round(aa[i],6).tolist()} "
                      f"newton={np.round(bbb[i],6).tolist()}")
        if n == "jnt_range":
            aa, bbb = a.reshape(-1, 2), bb.reshape(-1, 2)
            dd_ = np.abs(aa - bbb).max(axis=1)
            lim = u.sim.wp_model.jnt_limited
            lim = np.asarray(lim.numpy() if hasattr(lim, "numpy") else lim).reshape(-1)
            for i in np.argsort(-dd_)[:4]:
                nm_ = mujoco.mj_id2name(refmj, mujoco.mjtObj.mjOBJ_JOINT, int(i)) or f"<{i}>"
                print(f"        joint {int(i):3d} {nm_[:34]:34s} limited={int(lim[i])} "
                      f"mjlab={np.round(aa[i],3).tolist()} newton={np.round(bbb[i],3).tolist()}")
print("  (only differences shown)")

obs = wrapped.get_observations(); obs = obs[0] if isinstance(obs, tuple) else obs
qm = lambda: u.sim.data.qpos[0].detach().float().cpu().numpy()
qn = lambda: wp.to_torch(sv.mjw_data.qpos)[0].detach().float().cpu().numpy()

# Pelvis height on both sides. A 1.3 rad joint divergence under a kp=40 position servo is not
# tracking error -- it is the robot losing its posture. If Newton's pelvis drops while mjlab's holds,
# the story is a fall, and every other symptom (hand stalling at 0.2 m, then drifting to 0.65 m and
# staying there) follows from it.
print(f"\n{'step':>5} {'|dqpos|max':>12} {'|dq_hinge|':>12} {'mj_pelvz':>9} {'nt_pelvz':>9} "
      f"{'mj_objz':>9} {'nt_objz':>9}")
print("-" * 72)
hinge = [i for i in range(refmj.njnt) if int(refmj.jnt_type[i]) == 3]
hq = np.array([int(refmj.jnt_qposadr[i]) for i in hinge])
for k in range(STEPS):
  with torch.inference_mode():
    a = policy(obs)
  nenv.action_manager.advance(a)
  nt_term.process_actions(a.clone())
  for _ in range(4):
    nt_term.apply_actions()
    sv.step(nm.state(), nm.state(), control, None, 0.005)
  nenv.episode_length_buf += 1
  out = wrapped.step(a); obs = out[0] if isinstance(out, tuple) else out
  if k == 31:
    # What actually reached mjw_data.ctrl after each side applied its control. mjlab's real
    # actuators have ctrllimited=0, so its targets are never clipped; if Newton's conversion clamps
    # to ctrlrange, the arm is commanded short of where the policy asked and under-reaches exactly
    # the way this rollout does.
    cm = u.sim.data.ctrl[0].detach().float().cpu().numpy()
    cn = wp.to_torch(sv.mjw_data.ctrl).view(-1).detach().float().cpu().numpy()
    dc = np.abs(cm - cn)
    print(f"   [ctrl@31] max|diff|={dc.max():.6g}  nonzero mj={int((cm!=0).sum())} nt={int((cn!=0).sum())}")
    for i in np.argsort(-dc)[:6]:
      nm_ = mujoco.mj_id2name(refmj, mujoco.mjtObj.mjOBJ_ACTUATOR, int(i)) or f"<{i}>"
      rng = refmj.actuator_ctrlrange[int(i)]
      print(f"      [{int(i):3d}] {nm_:34s} mj={cm[i]:+.5f} nt={cn[i]:+.5f} range={np.round(rng,3).tolist()}")
  if k % 10 == 0 or k in (31, 35, 40) or k == STEPS - 1:
    A, B = qm(), qn()
    d = np.abs(A - B)
    pm = float(u.scene["robot"].data.root_link_pos_w[0, 2])
    pn = float(nenv.scene["robot"].data.root_link_pos_w[0, 2])
    print(f"{k:5d} {d.max():12.6g} {d[hq].max():12.6g} {pm:9.4f} {pn:9.4f} "
          f"{A[76]:9.4f} {B[76]:9.4f}")
