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
ap.add_argument("--reward-cfg", default=None,
  help="yaml whose `rewards:` block supplies the reward weights, instead of the "
       "checkpoint's params/env.yaml. Same block format: two spaces name, four spaces weight")
ap.add_argument("--resume", default=None, help="checkpoint to warm-start from")
ap.add_argument("--rollout-free-run", action="store_true",
                help="record the policy without early termination: clears the termination terms and "
                     "lifts the episode time limit, and sizes the rollout from the clip's own length "
                     "instead of a fixed step count. A reset mid-grasp makes the video show a "
                     "sequence of restarts rather than one attempt at the task.")
ap.add_argument("--rollout-pad", type=float, default=1.2,
                help="with --rollout-free-run, run this multiple of the clip length so the policy "
                     "is still being watched after the reference ends")
ap.add_argument("--rollout-steps", type=int, default=0,
                help="instead of training, roll the loaded checkpoint out for this many steps and "
                     "exit. Reuses the env and runner built above, so the contact recipe, object "
                     "mesh, table and reference are the ones the run was TRAINED with -- a "
                     "separate eval script re-deriving those flags is how earlier rollouts came "
                     "to be judged in a scene the policy never saw. Pair with --dump-qpos.")
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
ap.add_argument("--reference-pkls", default=None,
                help="comma-separated clips for mixed training; sets APPLE_EAT_PKL_MIX. Each env "
                     "trains on exactly one of them, so the batch carries several tasks at once.")
ap.add_argument("--clip-env-counts", default=None,
                help="comma-separated env count per clip, in --reference-pkls order. Must sum to "
                     "--num-envs. This is how a PMCP segment starts from the previous segment's "
                     "allocation; without it the split is equal.")
ap.add_argument("--sdf-objects", default=None,
                help="comma-separated object meshes, one per --reference-pkls entry IN THE SAME "
                     "ORDER. Checked against each clip's own obj_name rather than trusted.")
A = ap.parse_args()

MIX_PKLS: list[str] = []
MIX_STLS: list[str] = []
if A.reference_pkls:
  import pickle as _pickle
  MIX_PKLS = [x.strip() for x in A.reference_pkls.split(",") if x.strip()]
  MIX_STLS = [x.strip() for x in (A.sdf_objects or "").split(",") if x.strip()]
  if len(MIX_PKLS) < 2:
    raise SystemExit("--reference-pkls needs at least two clips; use --reference-pkl for one")
  if len(MIX_STLS) != len(MIX_PKLS):
    raise SystemExit(f"--sdf-objects has {len(MIX_STLS)} entries for {len(MIX_PKLS)} clip(s)")
  # Pairing the wrong mesh with a clip does not crash -- the robot just reaches for a shape that is
  # not there, and the clip never learns. Each pkl records its own object, so check rather than
  # trust the order the caller typed.
  for _pkl, _stl in zip(MIX_PKLS, MIX_STLS):
    with open(_pkl, "rb") as _f:
      _want = str(_pickle.load(_f).get("obj_name", "")).strip().lower()
    _got = os.path.splitext(os.path.basename(_stl))[0].strip().lower()
    if _want and _got != _want:
      raise SystemExit(f"clip {os.path.basename(_pkl)} is about {_want!r} but was paired with "
                       f"{os.path.basename(_stl)!r}; --reference-pkls and --sdf-objects must line up")
  os.environ["APPLE_EAT_PKL_MIX"] = ",".join(MIX_PKLS)
  # Several mjlab code paths still read the singular variable; point it at the first clip so they
  # resolve to a real file rather than whatever was left in the environment.
  os.environ["APPLE_EAT_PKL"] = MIX_PKLS[0]
  # This port groups environments by object and replicates each group, so an environment carries
  # exactly one object -- its own. Declare that layout so the scene config authors ONE object
  # entity and one sensor set instead of one per clip; mjlab's default fan-out assumes every
  # environment holds every object and parks the unused ones, which this scene has no room for.
  # Only scene authoring changes: mix_clip_count() still reports the real clip count, so the clip
  # map, the per-clip gates and the terminations are untouched.
  os.environ["APPLE_OBJECT_PER_WORLD"] = "1"
  print(f"[train] MIX: {len(MIX_PKLS)} clips " +
        ", ".join(f"{os.path.basename(p)}->{os.path.basename(t)}"
                  for p, t in zip(MIX_PKLS, MIX_STLS)))
elif A.reference_pkl:
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
_env_yaml = (Path(A.reward_cfg) if A.reward_cfg
             else Path(A.agent_cfg_from).parent / "params" / "env.yaml")
if A.reward_cfg and not _env_yaml.exists():
  raise SystemExit(f"--reward-cfg {_env_yaml} does not exist")
if _env_yaml.exists():
  _w = reward_weights_from_env_yaml(_env_yaml)
  # A yaml that parses to nothing is the failure this guard exists for: the flow style
  # `name: {weight: 1.0}` matches neither regex, so the file reads as empty and training
  # silently falls back to the task default where only `tracking` carries weight.
  if not _w:
    raise SystemExit(f"[reward-cfg] {_env_yaml} parsed to zero reward terms. The block "
                     f"format is `  <name>:` then `    weight: <float>` on the next line.")
  print(f"[reward-cfg] source: {_env_yaml}")
  apply_reward_weights(cfg, _w)
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

if A.rollout_free_run:
  from mjlab.tasks.apple_eat import mdp as _fr_mdp
  _fr_n = int(_fr_mdp._ref("cpu")["n_frames"])
  # An explicit --rollout-steps wins; the clip-derived length is only the fallback. Every video
  # being the same length makes them comparable side by side, which is usually what you want.
  _fr_steps = int(A.rollout_steps) if int(A.rollout_steps) > 0 \
      else max(1, int(round(_fr_n * float(A.rollout_pad))))
  A.rollout_steps = _fr_steps
  A.dump_steps = _fr_steps
  A.video_steps = _fr_steps
  # An episode that is never cut short is the whole point; both halves matter, because a time-out
  # resets just as surely as a termination does.
  _fr_terms = cfg.terminations if isinstance(cfg.terminations, dict) else vars(cfg.terminations)
  _fr_dropped = sorted(_fr_terms.keys())
  if isinstance(cfg.terminations, dict):
    cfg.terminations = {}
  else:
    for _k in _fr_dropped:
      setattr(cfg.terminations, _k, None)
  cfg.episode_length_s = 1.0e9
  print(f"[free-run] clip is {_fr_n} frames -> {_fr_steps} steps; "
        f"terminations disabled: {_fr_dropped}", flush=True)

print(f"building {A.num_envs} Newton worlds ...")
env = NewtonVecEnv(cfg, A.xml, num_envs=A.num_envs, device="cuda:0",
                   sdf_object_stl=A.sdf_object, sdf_resolution=A.sdf_resolution,
                   sdf_object_stls=(MIX_STLS or None),
                   clip_env_counts=([int(x) for x in A.clip_env_counts.split(",")]
                                    if A.clip_env_counts else None),
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

if A.rollout_steps:
  # Deterministic inference, not the stochastic rollout `learn` would collect: the point is to see
  # what the policy does, not what it explores.
  import torch as _rt
  from mjlab.scripts.play import _maybe_wrap_residual_action_stats_policy as _wrap_stats
  if not A.resume:
    raise SystemExit("--rollout-steps needs --resume; there is nothing to roll out otherwise")
  _pol = runner.get_inference_policy(device="cuda:0")
  # Without this wrapper a working checkpoint silently produces a policy that does nothing --
  # the residual action statistics live outside the actor and have to be reattached.
  _pol = _wrap_stats(TASK, runner, _pol)
  # Judge every checkpoint from the same place: the reference start. The RSI window would
  # otherwise drop each rollout at a random frame and the clips would not be comparable.
  env._env._force_reference_start_frame = 0
  env.reset()
  _obs = env.get_observations()
  print(f"[rollout] {A.rollout_steps} deterministic steps from reference frame 0")
  # Calibration, not decoration: a rollout that merely *looks* wrong is an opinion, but a rollout
  # whose contact/lift/stand disagree with the run's own training row is a measurement. Without
  # this the first eval harness shipped a scene the policy never saw and nobody could tell.
  _acc, _nacc, _dones = {}, 0, 0
  for _k in range(int(A.rollout_steps)):
    with _rt.inference_mode():
      _act = _pol(_obs)
    _obs, _rw, _tm, _to, _ex = env.step(_act)
    _dones += int((_tm | _to).sum())
    if _k in (0, 1, 5, 50, 200):
      _a = _act.detach().float()
      print(f"[rollout] action step={_k} shape={tuple(_a.shape)} "
            f"absmean={_a.abs().mean().item():.5f} absmax={_a.abs().max().item():.5f} "
            f"nonzero={(_a.abs() > 1e-8).float().mean().item():.3f}", flush=True)
    _lg = (env.extras.get("log") or {})
    for _kk, _vv in _lg.items():
      try:
        _acc[_kk] = _acc.get(_kk, 0.0) + float(_vv)
      except (TypeError, ValueError):
        continue
    _nacc += 1
    if _k % 100 == 0:
      print(f"[rollout] step {_k}", flush=True)
  print(f"[rollout] metrics over {_nacc} steps, {_dones} episode ends:", flush=True)
  for _kk in sorted(_acc):
    if any(_t in _kk for _t in ("physical_contact", "lift_success", "stable_not_fallen",
                                "sequence_success", "Termination", "ep_len", "object_mpjpe")):
      print(f"[rollout]   {_kk} = {_acc[_kk] / max(_nacc, 1):.4f}", flush=True)
  print("[rollout] done")
  raise SystemExit(0)

print(f"training for {A.iterations} iterations -> {log_dir}")

if os.environ.get("MIX_VERIFY"):
  import torch as _t, numpy as _np
  import mujoco as _mj
  from mjlab.tasks.apple_eat import mdp as _am, object_pool as _op
  _e = env._env
  _ref = _am._ref(str(_e.device))
  _n = int(_ref["n_frames"])
  _cid = env.clip_id
  print("\n[V] ===================== mixed-env verification =====================")

  # (4) RSI: does each env start at a DIFFERENT frame of its own clip, or all at frame 0?
  _sf = _am._reference_start_frame(_e, _n)
  _lo, _hi = _am._clip_bounds(_e, _n)
  _local = (_sf - _lo)
  print(f"[V] start frame, local to each clip: {[int(x) for x in _local[:8]]}")
  print(f"[V]   distinct local start frames  : {sorted(set(int(x) for x in _local))[:12]}")
  print(f"[V]   per-clip band [lo,hi]        : {[(int(a),int(b)) for a,b in zip(_lo[:4],_hi[:4])]}")

  # (5) object mass per world: does each clip's object carry its own mass?
  _mm = env.solver.mj_model
  _oid = [g for g in range(_mm.nbody)
          if "apple" in (_mj.mj_id2name(_mm, _mj.mjtObj.mjOBJ_BODY, g) or "")]
  print(f"[V] object bodies in mj_model: {[_mj.mj_id2name(_mm,_mj.mjtObj.mjOBJ_BODY,g) for g in _oid]}")
  for _b in _oid:
    print(f"[V]   body {_b} mass={float(_mm.body_mass[_b]):.5f} kg "
          f"inertia={[round(float(x),6) for x in _mm.body_inertia[_b]]}")
  try:
    import warp as _wp
    _bm = _wp.to_torch(env.solver.mjw_model.body_mass)
    print(f"[V] mjw_model.body_mass shape {tuple(_bm.shape)}")
    if _bm.ndim == 2:
      for _b in _oid:
        _col = _bm[:, _b]
        print(f"[V]   per-world mass of body {_b}: first8={[round(float(x),5) for x in _col[:8]]} "
              f"distinct={sorted(set(round(float(x),5) for x in _col))[:6]}")
    else:
      print(f"[V]   body_mass is shared across worlds (ndim={_bm.ndim}) -> every clip uses the same mass")
  except Exception as _ex:
    print(f"[V]   mjw body_mass read failed: {type(_ex).__name__}: {_ex}")

  # object collider extent per world, as an independent check that the MESH really differs
  try:
    _gid = [g for g in range(_mm.ngeom)
            if "apple" in (_mj.mj_id2name(_mm, _mj.mjtObj.mjOBJ_GEOM, g) or "")]
    for _g in _gid:
      print(f"[V] geom {_mj.mj_id2name(_mm,_mj.mjtObj.mjOBJ_GEOM,_g)} type={int(_mm.geom_type[_g])} "
            f"size={[round(float(x),4) for x in _mm.geom_size[_g]]} "
            f"rbound={float(_mm.geom_rbound[_g]):.4f}")
  except Exception as _ex:
    print(f"[V]   geom read failed: {type(_ex).__name__}")
  raise SystemExit(0)
if os.environ.get("MIX_FAR_STATS"):
  import torch as _st
  from mjlab.tasks.residual_interact import omnigrasp_faithful_mdp as _ofm
  from mjlab.tasks.apple_eat import mdp as _sam, object_pool as _sop
  from mjlab.tasks.residual_interact import mdp as _srm
  _orig_far = _ofm.og_object_far_termination
  _st_state = {"n": 0}
  def _far_stats(env, *a, **kw):
    out = _orig_far(env, *a, **kw)
    try:
      _st_state["n"] += 1
      if _st_state["n"] % 200 == 0:
        ref = _sam._ref(str(env.device))
        cid = _sam._clip_id(env)
        obj = _sop.active(env)
        fr = _sam._tracking_frame(env, ref["n_frames"])
        rp = _srm._reference_object_pos_w(env, ref, fr)
        d = (obj.data.root_link_pos_w - rp).norm(dim=-1)
        nc = int(cid.max().item()) + 1
        lines = []
        for c in range(nc):
          m = cid == c
          if not bool(m.any()): continue
          fires = out[m]
          lines.append("c%d fire=%.4f d_mean=%.3f d_p95=%.3f d_max=%.3f elb=%.1f" % (
            c, fires.float().mean().item(), d[m].mean().item(),
            d[m].quantile(0.95).item(), d[m].max().item(),
            env.episode_length_buf[m].float().mean().item()))
        print("[FARSTAT %d] " % _st_state["n"] + " | ".join(lines), flush=True)
        if _st_state["n"] == 200:
          try:
            _m = getattr(obj.data, "default_mass", None)
            print("[FARSTAT mass] shape=%s" % (None if _m is None else tuple(_m.shape),), flush=True)
            if _m is not None and _m.ndim >= 2:
              _pm = _m.flatten(1).sum(dim=-1)
              for c in range(nc):
                mm = cid == c
                vv = sorted(set(round(float(x), 5) for x in _pm[mm].tolist()))
                print("[FARSTAT mass] clip %d n=%d distinct=%s" % (c, int(mm.sum()), vv[:5]), flush=True)
          except Exception as _e2:
            print("[FARSTAT mass] failed %s: %s" % (type(_e2).__name__, _e2), flush=True)
          try:
            base = env._reference_start_frame
            for c in range(nc):
              mm = cid == c
              vv = sorted(set(int(x) for x in base[mm].tolist()))
              print("[FARSTAT rsi] clip %d n=%d ndistinct=%d %s" % (c, int(mm.sum()), len(vv), vv[:8]), flush=True)
          except Exception as _e3:
            print("[FARSTAT rsi] failed %s: %s" % (type(_e3).__name__, _e3), flush=True)
          try:
            for c in range(nc):
              mm = cid == c
              vv = sorted(set(round(float(x), 4) for x in d[mm].tolist()))
              print("[FARSTAT dist] clip %d n=%d ndistinct=%d top=%s" % (c, int(mm.sum()), len(vv), vv[-6:]), flush=True)
          except Exception as _e4:
            print("[FARSTAT dist] failed %s: %s" % (type(_e4).__name__, _e4), flush=True)
          try:
            import torch as _T
            hit = (d - 0.1216).abs() < 2e-4
            print("[FARSTAT hit] n=%d of %d, clips=%s" % (
              int(hit.sum()), d.numel(),
              sorted(set(int(x) for x in cid[hit].tolist()))), flush=True)
            _op_ = obj.data.root_link_pos_w
            _eo = env.scene.env_origins
            idx = _T.nonzero(hit).flatten()[:6]
            for k in idx.tolist():
              print("[FARSTAT hit] env=%d clip=%d frame=%d obj=%s ref=%s delta=%s origin=%s elb=%d" % (
                k, int(cid[k]), int(fr[k]),
                [round(float(x),4) for x in _op_[k].tolist()],
                [round(float(x),4) for x in rp[k].tolist()],
                [round(float(x),4) for x in (_op_[k]-rp[k]).tolist()],
                [round(float(x),3) for x in _eo[k].tolist()],
                int(env.episode_length_buf[k])), flush=True)
            miss = ~hit
            idx2 = _T.nonzero(miss).flatten()[:4]
            for k in idx2.tolist():
              print("[FARSTAT ok ] env=%d clip=%d frame=%d obj=%s ref=%s delta=%s origin=%s elb=%d" % (
                k, int(cid[k]), int(fr[k]),
                [round(float(x),4) for x in _op_[k].tolist()],
                [round(float(x),4) for x in rp[k].tolist()],
                [round(float(x),4) for x in (_op_[k]-rp[k]).tolist()],
                [round(float(x),3) for x in _eo[k].tolist()],
                int(env.episode_length_buf[k])), flush=True)
          except Exception as _e5:
            print("[FARSTAT hit] failed %s: %s" % (type(_e5).__name__, _e5), flush=True)
          try:
            import torch as _T
            tb = env.scene["table"]
            tz = tb.data.root_link_pos_w[:, 2]
            hit = (d - 0.1216).abs() < 2e-4
            for c in range(nc):
              mm = cid == c
              vv = sorted(set(round(float(x), 4) for x in tz[mm].tolist()))
              nh = int((mm & hit).sum())
              print("[FARSTAT table] clip %d n=%d nhit=%d distinct_tablez=%d %s" % (
                c, int(mm.sum()), nh, len(vv), vv[:6]), flush=True)
            print("[FARSTAT table] hit tablez mean=%.4f  ok tablez mean=%.4f" % (
              float(tz[hit].mean()), float(tz[~hit].mean())), flush=True)
            vz = obj.data.root_link_lin_vel_w[:, 2] if hasattr(obj.data, "root_link_lin_vel_w") else None
            if vz is not None:
              print("[FARSTAT table] hit objvz mean=%.3f  ok objvz mean=%.3f" % (
                float(vz[hit].mean()), float(vz[~hit].mean())), flush=True)
          except Exception as _e6:
            print("[FARSTAT table] failed %s: %s" % (type(_e6).__name__, _e6), flush=True)
    except Exception as _e:
      print("[FARSTAT] failed: %s: %s" % (type(_e).__name__, _e), flush=True)
    return out
  _tm = env.termination_manager
  _i = _tm._term_names.index("og_object_far")
  _orig_far = _tm._term_cfgs[_i].func
  _tm._term_cfgs[_i].func = _far_stats
  print("[FARSTAT] installed on the live termination manager", flush=True)
if os.environ.get("MIX_FAR_PROBE"):
  import torch as _t
  import mujoco as _mj
  from mjlab.tasks.apple_eat import mdp as _am, object_pool as _op
  from mjlab.tasks.residual_interact import mdp as _rmdp
  _e = env._env
  _ref = _am._ref(str(_e.device))
  _n = int(_ref["n_frames"])
  _cid = env.clip_id
  _obj = _op.active(_e)
  _names = [os.path.basename(p) for p in os.environ.get("APPLE_EAT_PKL_MIX", "").split(",")]
  _act = _t.zeros((_e.num_envs, env.action_manager.total_action_dim), device=_e.device)
  print("\n[FAR] step | clip | localframe | objz | refz | dist(mm) | elb")
  for _k in range(24):
    _fr = _am._tracking_frame(_e, _n)
    _lf = _am.local_tracking_frame(_e, _n)
    _rp = _rmdp._reference_object_pos_w(_e, _ref, _fr)
    _op_w = _obj.data.root_link_pos_w
    _d = (_op_w - _rp).norm(dim=-1)
    if _k in (0, 1, 2, 3, 5, 8, 12, 20, 23):
      for _i in range(_e.num_envs):
        _c = int(_cid[_i])
        _flag = "  <-- FAR" if float(_d[_i]) > 0.12 else ""
        print(f"[FAR] {_k:3d} | {_c} {_names[_c][:22]:<22} | {int(_lf[_i]):5d} | "
              f"{float(_op_w[_i][2]):.4f} | {float(_rp[_i][2]):.4f} | "
              f"{float(_d[_i])*1000:8.1f}{_flag}   elb={int(_e.episode_length_buf[_i])}")
      print()
    env.step(_act)
  raise SystemExit(0)
runner.learn(num_learning_iterations=A.iterations, init_at_random_ep_len=True)
