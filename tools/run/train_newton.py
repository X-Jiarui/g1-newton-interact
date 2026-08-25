"""Train the residual policy with Newton doing the physics.

Reuses mjlab's PPO runner and its entire MDP -- observations, action term, rewards, terminations,
events -- against `NewtonVecEnv`. Only the simulator changes.

Validated before this script was written: all 20 observation groups agree with mjlab from the same
state, the ctrl vectors are identical for identical actions, and the reward matches mjlab exactly
while the two states agree (both 0 through the startup hold) and diverges only when the physics does.
"""

from __future__ import annotations

import argparse
import json as _json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml as _yaml

ap = argparse.ArgumentParser()
ap.add_argument("--num-envs", type=int, default=256)
ap.add_argument("--profile-step", type=int, default=0,
                help="time each part of one control step over N steps and exit; says whether the cost is in warp kernels or in the torch-side managers")
ap.add_argument("--state-digest", type=int, default=0,
                help="step N times with zero residual and print checksums of qpos, qvel and reward; run twice with and without --cuda-graph to prove the graph changes no number")
ap.add_argument("--effortless-action", action="store_true",
                help="drop the PD torque law from the action term; it is discarded on this backend and costs a host sync per substep")
ap.add_argument("--object-solref", default="0.004,1.0",
                help="solref for the object collider. Default 0.004,1.0: measured, the shared default of 0.02 lets the stapler settle 1.88mm into the table (and 11.6mm in transient) against mjlab's analytic sphere at 0.37mm; 0.004 gives 0.04mm for the stapler and 0.28mm for the mug. Pass an empty string to keep the scene default.")
ap.add_argument("--solver-kwargs", default=None,
                help="JSON merged into the SolverMuJoCo kwargs, e.g. "
                     "'{\"impratio\": 1.0, \"cone\": \"pyramidal\"}'. The native path defaults "
                     "to cone=elliptic with impratio=1000, copied from Newton's hydroelastic "
                     "example; that ratio makes the tangential (friction) constraints three "
                     "orders softer than the normal ones, which is a candidate explanation for "
                     "'the hand touches the object but the object does not move'.")
ap.add_argument("--rigid-object-table", action="store_true",
                help="With --native-contacts, collide the object and table rigidly instead of "
                     "through the hydroelastic SDF. Measured at 1024 env: object/table contacts "
                     "30-37 -> 4, per-world total 48-79 -> 19-25, 15142 -> 20011 env-steps/s "
                     "(the MuJoCo-contact path is 22095), resting penetration 0.001 -> 0.110 mm. "
                     "The object keeps its real STL collider either way, which is what the "
                     "hand-object grasp actually collides against.")
ap.add_argument("--table-sdf-resolution", type=int, default=None,
                help="Separate SDF resolution for the table. Free to lower: at 8 the resting "
                     "penetration is still -0.002 mm. It does not reduce the contact count, "
                     "which follows the object's resolution.")
ap.add_argument("--native-contacts", action="store_true",
                help="use Newton's SDF hydroelastic collision pipeline instead of "
                     "MuJoCo's own collision (SolverMuJoCo still integrates)")
ap.add_argument("--table-under-object", action="store_true",
                help="move the mocap table so the object's true collider rests where the reference places it, instead of letting the object settle away from it")
ap.add_argument("--newton-video", default=None,
                help="record an mp4 with Newton's own renderer, which draws the "
                     "model the physics holds -- real object mesh, real table")
ap.add_argument("--video-size", default="960x720")
ap.add_argument("--video-steps", type=int, default=500)
ap.add_argument("--video-cam", default=None,
                help="camera as \"ex,ey,ez,tx,ty,tz\"; default frames the hands and the object from 0.9m")
ap.add_argument("--dump-qpos", default=None,
                help="npz of env-0 qpos and mocap per control step, for rendering "
                     "a video of the run with tools/run/render_traj.py")
ap.add_argument("--dump-steps", type=int, default=600,
                help="how many control steps --dump-qpos records")
ap.add_argument("--cuda-graph", action="store_true",
                help="replay the physics substep from a captured CUDA graph")
ap.add_argument("--sensor-probe", type=int, default=0,
                help="step N times with zero residual and report which sensordata slots ever "
                     "read nonzero, then exit; verifies contact sensors are live")
ap.add_argument("--iterations", type=int, default=2000)
ap.add_argument("--xml", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets/mjlab_scene/scene.xml"))
ap.add_argument("--agent-cfg-from", default=os.path.expanduser(
  "~/sweep_ckpts_r2/OF_00_apple_eat_1_SPHERE/model_7310.pt"),
  help="checkpoint whose params/agent.yaml supplies the agent config (tracker, residual gains)")
ap.add_argument("--resume", default=None, help="checkpoint to warm-start from")
ap.add_argument("--run-name", default="NEWTON_NATIVE")
ap.add_argument("--log-root", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs/rsl_rl"))
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--sdf-object", default=None,
                help="STL whose SDF replaces the scene's sphere collider, e.g. the GRAB mesh for "
                     "this clip's object. Without it the object stays mjlab's 4cm analytic sphere.")
ap.add_argument("--sdf-resolution", type=int, default=128)
ap.add_argument("--viser-port", type=int, default=None,
                help="serve a live view of training on this port (tunnel and open in a browser)")
ap.add_argument("--render-every", type=int, default=4,
                help="control steps between rendered frames; rendering every step costs throughput")
ap.add_argument("--reference-pkl", default=None,
                help="sets APPLE_EAT_PKL before the task modules read it")
A = ap.parse_args()

if A.reference_pkl:
  # Has to be set before mjlab's task modules import: the clip path is read at module level.
  os.environ["APPLE_EAT_PKL"] = A.reference_pkl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
import mjw_compat
_p = mjw_compat.apply()
if _p:
  print(f"[compat] tolerating removed mujoco_warp options: {_p}")

import mjlab.tasks  # noqa: F401
from mjlab.rl import RslRlVecEnvWrapper, MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.scripts.play import _apply_cfg_mapping
from newton_vec_env import NewtonVecEnv
from reward_cfg_from_checkpoint import reward_weights_from_env_yaml, apply_reward_weights

TASK = "Mjlab-ResidualInteract-G1"
torch.manual_seed(A.seed)
np.random.seed(A.seed)

cfg = load_env_cfg(TASK, play=False)
cfg.scene.num_envs = A.num_envs
agent_cfg = load_rl_cfg(TASK)
p = Path(A.agent_cfg_from).parent / "params" / "agent.yaml"
if not p.exists():
  raise SystemExit(f"missing {p}: the task default agent config points at another project's tracker")
_apply_cfg_mapping(agent_cfg, _yaml.unsafe_load(p.open()))

# The agent config is not the whole story: the checkpoint's env.yaml carries the reward weights it
# trained with, and load_env_cfg returns the task default where only `tracking` has weight. Training
# against the default is training with no grasping reward at all.
_env_yaml = Path(A.agent_cfg_from).parent / "params" / "env.yaml"
if _env_yaml.exists():
  apply_reward_weights(cfg, reward_weights_from_env_yaml(_env_yaml))
else:
  print(f"[reward-cfg] WARNING: no env.yaml beside the checkpoint; training with the task default, "
        f"where only one reward term carries weight")

# Every candidate trained with the pelvis start-assist disabled; the task default is 1.5 for 120
# steps, which would prop the robot up with a wrench it never sees at evaluation.
_s = cfg.actions.get("sonic_action") if isinstance(cfg.actions, dict) else cfg.actions.sonic_action
_s.tracking_start_assist_gain = 0.0
_s.tracking_start_assist_steps = 0
if str(getattr(agent_cfg, "base_tracker_kind", "")).strip().lower() == "astra_onnx":
  from mjlab.tasks.residual_interact.env_cfgs import set_astra_body_dynamics
  set_astra_body_dynamics(cfg)

print(f"building {A.num_envs} Newton worlds ...")
env = NewtonVecEnv(cfg, A.xml, num_envs=A.num_envs, device="cuda:0",
                   sdf_object_stl=A.sdf_object, sdf_resolution=A.sdf_resolution,
                   native_contacts=A.native_contacts,
                   hydro_object_table=not A.rigid_object_table,
                   table_sdf_resolution=A.table_sdf_resolution,
                   viser_port=A.viser_port, render_every=A.render_every,
                   cuda_graph=A.cuda_graph,
                   effortless_action=A.effortless_action,
                   table_under_object=A.table_under_object,
                   object_solref=A.object_solref,
                   dump_qpos=A.dump_qpos, dump_steps=A.dump_steps,
                   newton_video=A.newton_video, video_size=A.video_size,
                   video_steps=A.video_steps, video_cam=A.video_cam,
                   solver_kwargs=(_json.loads(A.solver_kwargs) if A.solver_kwargs else None))

if os.environ.get("HULL_PROBE"):
  import numpy as _np, mujoco as _mj
  _m = env.solver.mj_model
  _g = [g for g in range(_m.ngeom)
        if "apple" in (_mj.mj_id2name(_m, _mj.mjtObj.mjOBJ_GEOM, g) or "")][0]
  _did = int(_m.geom_dataid[_g])
  _va, _vn = int(_m.mesh_vertadr[_did]), int(_m.mesh_vertnum[_did])
  _V = _m.mesh_vert[_va:_va + _vn].reshape(-1, 3).astype(_np.float64)
  print("HULL full mesh verts=%d  z=[%.5f, %.5f]" % (_vn, _V[:, 2].min(), _V[:, 2].max()))
  _ga = int(_m.mesh_graphadr[_did])
  if _ga >= 0:
    _g_ = _m.mesh_graph
    _numvert = int(_g_[_ga]); _numface = int(_g_[_ga + 1])
    print("HULL convex hull verts=%d faces=%d" % (_numvert, _numface))
    # hull vertex indices follow the two counts, after the edge tables
    _idx = _g_[_ga + 2 + _numvert:_ga + 2 + _numvert + _numvert]
    _idx = _np.asarray(_idx, dtype=_np.int64)
    _idx = _idx[(_idx >= 0) & (_idx < _vn)]
    if len(_idx):
      _H = _V[_idx]
      print("HULL hull z=[%.5f, %.5f]  lowest is %.2f mm ABOVE the true mesh bottom"
            % (_H[:, 2].min(), _H[:, 2].max(), 1000.0 * (_H[:, 2].min() - _V[:, 2].min())))
  else:
    print("HULL no convex graph on this mesh")
  raise SystemExit(0)

if os.environ.get("NCON_PROBE"):
  import numpy as _np, torch as _t, warp as _wp, mujoco as _mj
  _m = env.solver.mj_model
  _oid = [g for g in range(_m.ngeom)
          if "apple" in (_mj.mj_id2name(_m, _mj.mjtObj.mjOBJ_GEOM, g) or "")]
  _tid = [g for g in range(_m.ngeom)
          if "table" in (_mj.mj_id2name(_m, _mj.mjtObj.mjOBJ_GEOM, g) or "")]
  print("NCON object geoms=%s table geoms=%s" % (_oid, _tid))
  env.reset()
  _act = _t.zeros(env.num_envs, env.action_manager.total_action_dim, device="cuda:0")
  for _k in range(1, 201):
    env.step(_act)
    if _k not in (1, 5, 20, 60, 150, 200): continue
    _d = env.solver.mjw_data
    _n = int(_wp.to_torch(_d.ncon)[0]) if hasattr(_d, "ncon") else -1
    _g1 = _wp.to_torch(_d.contact.geom)[:, 0].cpu().numpy() if hasattr(_d.contact, "geom") else None
    _pairs = 0
    _depth = []
    if _g1 is not None:
      _gg = _wp.to_torch(_d.contact.geom).cpu().numpy()
      _dist = _wp.to_torch(_d.contact.dist).cpu().numpy()
      for _i in range(min(_n if _n > 0 else len(_gg), len(_gg))):
        _a, _b = int(_gg[_i][0]), int(_gg[_i][1])
        if (_a in _oid and _b in _tid) or (_b in _oid and _a in _tid):
          _pairs += 1
          _depth.append(float(_dist[_i]))
    print("NCON step=%-4d total_con=%-4d object_table_contacts=%d depths_mm=%s"
          % (_k, _n, _pairs, [round(1000*x, 2) for x in sorted(_depth)[:6]]))
  raise SystemExit(0)

if os.environ.get("SDF_CHECK"):
  import mujoco as _mj, numpy as _np, warp as _wp
  _m = env.solver.mj_model
  print("SDF nplugin=%d" % getattr(_m, "nplugin", -1))
  for _g in range(_m.ngeom):
    _n = _mj.mj_id2name(_m, _mj.mjtObj.mjOBJ_GEOM, _g) or ""
    if "apple" not in _n: continue
    _did = int(_m.geom_dataid[_g])
    print("SDF geom %s type=%d dataid=%d" % (_n, _m.geom_type[_g], _did))
    if hasattr(_m, "geom_plugin"):
      print("SDF   geom_plugin=%d" % int(_m.geom_plugin[_g]))
    if _did >= 0:
      print("SDF   mesh verts=%d faces=%d graphadr=%d"
            % (int(_m.mesh_vertnum[_did]), int(_m.mesh_facenum[_did]),
               int(_m.mesh_graphadr[_did]) if hasattr(_m, "mesh_graphadr") else -1))
  _w = env.solver.mjw_model
  for _a in ("nsdf", "sdf_geom", "geom_sdf", "mesh_sdf", "nmeshsdf"):
    if hasattr(_w, _a):
      _v = getattr(_w, _a)
      print("SDF mjw_model.%s = %s" % (_a, getattr(_v, "shape", _v)))
  print("SDF newton shapes with sdf:",
        [i for i, s_ in enumerate(env.nmodel.shape_source) if s_ is not None
         and getattr(s_, "has_sdf", None)][:5] if hasattr(env.nmodel, "shape_source") else "n/a")
  raise SystemExit(0)

if os.environ.get("CPARAM_PROBE"):
  import mujoco as _mj, numpy as _np
  _m = env.solver.mj_model
  def _bodyid(sfx):
    for b in range(_m.nbody):
      n = (_mj.mj_id2name(_m, _mj.mjtObj.mjOBJ_BODY, b) or "").replace("/", "_")
      if n.endswith(sfx): return b
    return -1
  for _label, _sfx in (("object", "apple_apple"), ("table", "table_table")):
    _b = _bodyid(_sfx)
    print("CP %s body=%d name=%s" % (_label, _b, _mj.mj_id2name(_m, _mj.mjtObj.mjOBJ_BODY, _b)))
    print("CP    mass=%.4f" % float(_m.body_mass[_b]))
    for _g in range(_m.ngeom):
      if _m.geom_bodyid[_g] != _b: continue
      print("CP    geom%-4d type=%d contype=%d conaffinity=%d condim=%d priority=%d"
            % (_g, _m.geom_type[_g], _m.geom_contype[_g], _m.geom_conaffinity[_g],
               _m.geom_condim[_g], _m.geom_priority[_g]))
      print("CP      solref=%s solimp=%s" % (_np.round(_m.geom_solref[_g], 6).tolist(),
                                             _np.round(_m.geom_solimp[_g], 4).tolist()))
      print("CP      margin=%.5f gap=%.5f friction=%s"
            % (_m.geom_margin[_g], _m.geom_gap[_g], _np.round(_m.geom_friction[_g], 4).tolist()))
  print("CP timestep=%.5f  o_solref=%s o_solimp=%s"
        % (_m.opt.timestep, _np.round(_m.opt.o_solref, 6).tolist(),
           _np.round(_m.opt.o_solimp, 4).tolist()))
  print("CP gravity=%s  impratio=%.3f" % (_np.round(_m.opt.gravity, 3).tolist(), _m.opt.impratio))
  raise SystemExit(0)



if A.state_digest:
  import torch as _t, warp as _wp
  _e = env
  _e.reset()
  _act = _t.zeros(_e.num_envs, _e.action_manager.total_action_dim, device="cuda:0")
  for _k in range(int(A.state_digest)):
    _obs, _rew, *_ = _e.step(_act)
  _qp = _wp.to_torch(_e.solver.mjw_data.qpos).double()
  _qv = _wp.to_torch(_e.solver.mjw_data.qvel).double()
  print("DIGEST steps=%d graph=%s" % (A.state_digest, bool(A.cuda_graph)))
  for _n, _x in (("qpos", _qp), ("qvel", _qv), ("reward", _rew.double())):
    print("DIGEST   %-7s sum=%+.12e  absmax=%.12e  mean=%+.12e"
          % (_n, float(_x.sum()), float(_x.abs().max()), float(_x.mean())))
  raise SystemExit(0)

if A.profile_step:
  import time as _time, torch as _t
  _e = env
  _e.reset()
  _act = _t.zeros(_e.num_envs, _e.action_manager.total_action_dim, device="cuda:0")
  _acc = {}

  def _tick(_key, _fn):
    _t.cuda.synchronize(); _t0 = _time.perf_counter()
    _r = _fn()
    _t.cuda.synchronize()
    _acc[_key] = _acc.get(_key, 0.0) + (_time.perf_counter() - _t0)
    return _r

  for _ in range(10):        # warm up kernels and any lazy compilation
    _e.step(_act)

  _N = int(A.profile_step)
  for _ in range(_N):
    _tick("action_manager.advance", lambda: _e.action_manager.advance(_act))
    _tick("action_term.process", lambda: _e.action_term.process_actions(_act))
    def _phys():
      for _ in range(_e.decimation):
        _e.action_term.apply_actions()
        _e.solver.step(_e.state_in, _e.state_out, _e.control, None, _e.physics_dt)
        _e.state_in, _e.state_out = _e.state_out, _e.state_in
    _tick("physics (decimation loop)", _phys)
    def _phys_apply():
      for _ in range(_e.decimation):
        _e.action_term.apply_actions()
    _tick("  of which apply_actions", _phys_apply)
    _tick("observation_manager", lambda: _e.observation_manager.compute())
    _e._env.reward_buf = _tick("reward_manager", lambda: _e.reward_manager.compute(dt=_e.step_dt))
    _tick("termination_manager", lambda: _e.termination_manager.compute())
    _tick("metrics_manager", lambda: _e.metrics_manager.compute()
          if hasattr(_e.metrics_manager, "compute") else None)

  _total = sum(v for k, v in _acc.items() if not k.startswith("  "))
  print("PROF envs=%d decimation=%d steps=%d" % (_e.num_envs, _e.decimation, _N))
  for _k, _v in sorted(_acc.items(), key=lambda kv: -kv[1]):
    _pct = 100.0 * _v / _total if not _k.startswith("  ") else float("nan")
    print("PROF   %-30s %8.2f ms/step  %s" % (
      _k, 1000.0 * _v / _N, ("%5.1f%%" % _pct) if _pct == _pct else "(subset)"))
  print("PROF   %-30s %8.2f ms/step" % ("TOTAL (accounted)", 1000.0 * _total / _N))
  raise SystemExit(0)

if A.sensor_probe:
  import numpy as _np, torch as _t, warp as _wp
  _m = env.solver.mjw_model
  env.reset()
  _peak = _np.zeros(int(_m.nsensordata))
  for _ in range(A.sensor_probe):
    env.step(_t.zeros(env.num_envs, env.action_manager.total_action_dim, device="cuda:0"))
    _sd = _wp.to_torch(env.solver.mjw_data.sensordata).detach().cpu().numpy()
    _peak = _np.maximum(_peak, _np.abs(_sd).max(axis=0))
  _nz = int((_peak > 1e-9).sum())
  _ty = _wp.to_torch(env.solver.mjw_model.sensor_type).cpu().numpy()
  _ad = _wp.to_torch(env.solver.mjw_model.sensor_adr).cpu().numpy()
  _dm = _wp.to_torch(env.solver.mjw_model.sensor_dim).cpu().numpy()
  _cmask = _np.zeros(len(_peak), bool)
  for _t_, _a_, _d_ in zip(_ty, _ad, _dm):
    if int(_t_) == 42:
      _cmask[int(_a_):int(_a_) + int(_d_)] = True
  _cp = _peak[_cmask]
  print(f"PROBE_CONTACT slots={_cmask.sum()} nonzero={int((_cp>1e-9).sum())} max={_cp.max():.4f}")
  print(f"PROBE steps={A.sensor_probe} nonzero_slots={_nz}/{len(_peak)} max={_peak.max():.4f}")
  raise SystemExit(0)
print(f"  reward terms={len(env.reward_manager.active_terms)} "
      f"termination terms={len(env.termination_manager.active_terms)} "
      f"max_episode_length={env.max_episode_length}")

wrapped = RslRlVecEnvWrapper(env)
log_dir = Path(A.log_root) / "g1_residual_interact" / A.run_name
log_dir.mkdir(parents=True, exist_ok=True)
runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
runner = runner_cls(wrapped, asdict(agent_cfg), log_dir=str(log_dir), device="cuda:0")
if A.resume:
  runner.load(A.resume)
  print(f"warm-started from {A.resume}")

print(f"training for {A.iterations} iterations -> {log_dir}")

runner.learn(num_learning_iterations=A.iterations, init_at_random_ep_len=True)
