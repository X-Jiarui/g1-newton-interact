"""Infer the mjlab apple_eat_1 checkpoint with Newton doing the physics.

Division of labour, stated plainly because it is the whole point of the exercise:

  Newton    builds the model (ModelBuilder.add_mjcf -> SolverMuJoCo, verified field-for-field
            against mjlab's compiled model) and advances it (solver.step at dt = 0.005)
  mjlab     supplies the trained policy, the observation assembly (ResidualFeatureGroupObs) and the
            action term (Sonic53Action), driven against Newton state through src/newton_bridge.py

Nothing about the observation or action pipeline is re-derived. That is deliberate: the mjlab eval
harness lost a working 0.645-lift checkpoint to a single missing policy wrapper, with no error and no
warning -- just a policy that looked like it could not grasp. 1328 observation dims across 20 groups
offer many more chances to make that same mistake quietly.

The mjlab environment built here is used ONLY to construct the runner and load the checkpoint (the
runner needs an env to resolve observation and action dimensions). It is never stepped. Every
observation fed to the policy comes from Newton's state.
"""

from __future__ import annotations

import argparse, sys, json, os, sys
from dataclasses import asdict
from pathlib import Path

import numpy as np, torch, yaml as _yaml

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default=os.path.expanduser(
  "~/sweep_ckpts_r2/OF_00_apple_eat_1_SPHERE/model_7310.pt"))
ap.add_argument("--xml", default=os.path.expanduser(
  "~/projects/g1-newton-interact/assets/mjlab_scene/scene.xml"))
ap.add_argument("--steps", type=int, default=400, help="control steps (decimation 4 => 4 physics each)")
ap.add_argument("--every", type=int, default=20)
ap.add_argument("--until-ref-end", action="store_true",
                help="run the whole reference clip instead of a fixed step count: the length is read "
                     "from the clip itself, plus the startup hold before the reference starts "
                     "advancing, and the rollout stops on the last frame.")
ap.add_argument("--dump-qpos", default=None, help="npz of per-step qpos, for rendering the video")
ap.add_argument("--newton-video", default=None,
                help="record an mp4 with NEWTON's own renderer (ViewerGL, headless). This shows "
                     "Newton's scene graph -- the 81 collision geoms of the converted MJCF, with no "
                     "visual-only meshes -- rather than mjlab's visual model.")
ap.add_argument("--video-size", default="960x720")
ap.add_argument("--compare-obs", action="store_true",
                help="also build mjlab's own observations at step 0 and diff them group by group")
A = ap.parse_args()

sys.path.insert(0, os.path.expanduser("~/projects/g1-newton-interact/src"))

# ---------------------------------------------------------------- Newton side
import newton, warp as wp, mujoco
from newton.solvers import SolverMuJoCo

print(f"newton {newton.__version__}  warp {wp.config.version}  mujoco {mujoco.__version__}")
builder = newton.ModelBuilder()
SolverMuJoCo.register_custom_attributes(builder)
builder.default_shape_cfg.gap = 0.0            # else every contact needs 10 cm penetration
builder.add_mjcf(A.xml, collapse_fixed_joints=False, convert_3d_hinge_to_ball_joints=False,
                 parse_mujoco_options=True, parse_meshes=True, parse_sites=True,
                 enable_self_collisions=True)
nmodel = builder.finalize()
# update_data_interval=0 pins mjw_data as the authority: SolverMuJoCo otherwise re-syncs qpos from
# Newton's State every step, which would erase the direct joint/root writes Sonic53Action performs
# during the first 30 steps of the startup hold.
# Constraint budget copied from mjlab's SimulationCfg (njmax 2048, nconmax 256). Newton estimates
# these from the INITIAL state, which for this scene has almost no contacts -- it chose a budget that
# overflowed the moment the hand and table came into contact ("nefc overflow - please increase njmax
# to 65"), and an overflowing solver silently DROPS constraints: the apple fell straight through the
# table it was resting on. The warning scrolls past in a log nobody reads; the symptom looks like bad
# physics rather than a budget.
solver = SolverMuJoCo(nmodel, enable_multiccd=True, update_data_interval=0,
                      njmax=2048, nconmax=256)
control = nmodel.control()

_ref_mj = mujoco.MjModel.from_xml_path(A.xml)

# Newton reconstructs an MjModel from its own Model rather than compiling the MJCF, and that
# reconstruction mislabels which bodies are "simple": apple/apple and table/table -- the two free
# bodies -- come back with body_simple=0, and the apple's six free dofs get dof_simplenum=0 instead
# of MuJoCo's 6,5,4,3,2,1. That changes nC (the size of MuJoCo's compressed mass-matrix layout) from
# 1087 to 1102, so the two engines solve with different sparse structures for the SAME tree.
#
# The effect is precisely what the divergence looked like: smooth dynamics agreed to 1e-5 while
# qfrc_constraint differed by 2e-3 from the first free-dynamics step. It is also invisible to every
# check that compares the compiled models -- they agree on nM, on the tree, on every per-body field.
#
# dof_damping is the other field Newton's conversion drops (free-joint damping specifically), and it
# has to be re-applied AFTER put_model, since put_model copies from the CPU model.
# Patching body_simple/dof_simplenum on the reconstructed model does not help: nC and the index
# arrays that go with it (M_rowadr, M_colind, mapM2M) are computed by MuJoCo's compiler and stored on
# the MjModel, so put_model copies the stale nC straight through. The derived structure has to come
# from a compiled model.
#
# Newton's solver still does everything it does: control application, collision, the step. Only the
# MjModel handed to it is MuJoCo's compilation of the same MJCF rather than Newton's reconstruction
# of it -- and the two were verified identical in tree topology, body/joint/actuator counts and
# order, so this substitutes a correctly-derived model, not a different robot.
import mujoco_warp as _mjw
_before = (int(solver.mjw_model.nC), int(solver.mj_model.nmocap))
solver.mj_model = _ref_mj
solver.mjw_model = _mjw.put_model(_ref_mj)
solver.mjw_data = _mjw.put_data(_ref_mj, mujoco.MjData(_ref_mj),
                                nworld=1, nconmax=256, njmax=2048)
print(f"warp model rebuilt from the compiled MJCF: nC {_before[0]} -> {int(solver.mjw_model.nC)}, "
      f"nmocap {_before[1]} -> {int(_ref_mj.nmocap)}")
assert int(solver.mjw_model.nC) == int(_ref_mj.nC)

print(f"newton model: nq={solver.mj_model.nq} nv={solver.mj_model.nv} nu={solver.mj_model.nu} "
      f"ngeom={solver.mj_model.ngeom}")

DT = float(_ref_mj.opt.timestep)
DECIMATION = 4

# ---------------------------------------------------------------- mjlab side
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
from mjlab.tasks.residual_interact import mdp as rmdp
from mjlab.tasks.apple_eat import mdp as amdp

TASK = "Mjlab-ResidualInteract-G1"
ck = Path(A.checkpoint)
env_cfg = load_env_cfg(TASK, play=True); env_cfg.scene.num_envs = 1
agent_cfg = load_rl_cfg(TASK)
_apply_cfg_mapping(agent_cfg, _yaml.unsafe_load((ck.parent / "params" / "agent.yaml").open()))
_act = getattr(env_cfg, "actions", None)
_sonic_cfg = _act.get("sonic_action") if isinstance(_act, dict) else getattr(_act, "sonic_action")
_sonic_cfg.tracking_start_assist_gain = 0.0      # every candidate trained with the assist disabled
_sonic_cfg.tracking_start_assist_steps = 0
astra = str(getattr(agent_cfg, "base_tracker_kind", "")).strip().lower() == "astra_onnx"
if astra:
  from mjlab.tasks.residual_interact.env_cfgs import set_astra_body_dynamics
  set_astra_body_dynamics(env_cfg)

mjenv = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode=None)
if astra:
  from mjlab.tasks.residual_interact.env_cfgs import install_astra_body_pd, install_object_variant_sizes
  install_astra_body_pd(mjenv.unwrapped); install_object_variant_sizes(mjenv.unwrapped)
wrapped = RslRlVecEnvWrapper(mjenv)
runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(wrapped, asdict(agent_cfg), device="cuda:0")
runner.load(str(ck))
policy = runner.get_inference_policy(device="cuda:0")
policy = _maybe_wrap_residual_action_stats_policy(TASK, runner, policy)
print("policy loaded (mjlab env is for construction only and is never stepped)")

# ------------------------------------------------------------- bridge + terms
from newton_bridge import NewtonEnv

nenv = NewtonEnv(solver.mj_model, solver.mjw_data, num_envs=1, device="cuda:0",
                 control=control, rename_from=None, physics_dt=DT, decimation=DECIMATION,
                 solver=solver)
print(f"bridge: robot joints={len(nenv.scene['robot'].joint_names)} "
      f"bodies={len(nenv.scene['robot'].body_names)} entities={sorted(nenv.scene)}")

# Build each observation group from the env config's OWN term cfg -- func and params taken as
# written -- instead of hand-constructing a group param. Two groups are not ResidualFeatureGroupObs
# at all (astra_obs is AstraObs136, sonic_obs_or_latent is a Sonic encoder), and the residual groups
# carry a ref_preview_steps parameter that a hand-rolled cfg would silently omit and change the
# reference_preview width. Reading the config is the only way to be sure they match mjlab.
obs_cfg = env_cfg.observations if isinstance(env_cfg.observations, dict) else vars(env_cfg.observations)
builders, skipped = {}, {}
for gname, gcfg in obs_cfg.items():
  terms = getattr(gcfg, "terms", None) or (gcfg if isinstance(gcfg, dict) else None)
  if not terms:
    skipped[gname] = "no terms"; continue
  tcfg = terms.get("policy") or next(iter(terms.values()))
  try:
    builders[gname] = tcfg.func(tcfg, nenv)
  except Exception as e:
    skipped[gname] = f"{type(e).__name__}: {e}"
print(f"observation builders: {len(builders)} built, {len(skipped)} skipped")
for g, why in skipped.items():
  print(f"   skipped {g}: {str(why)[:90]}")

action_term = amdp.Sonic53Action(_sonic_cfg, nenv)
amgr = nenv.bind_action_manager(action_term.action_dim, {"sonic_action": action_term})
print(f"action term: dim={action_term.action_dim}")

# ------------------------------------------------------------- newton viewer
viewer = None
frames = []
if A.newton_video:
  # pyglet creates a shadow window even when the viewer is asked for headless, which fails on a box
  # with no display. Its own headless path uses EGL and has to be selected before pyglet is imported.
  os.environ.setdefault("PYGLET_HEADLESS", "1")
  import pyglet as _pyglet
  _pyglet.options["headless"] = True
  import newton.viewer as _nv
  _w, _h = (int(x) for x in A.video_size.lower().split("x"))
  # headless=True renders offscreen; get_frame() then returns the rendered image as a warp array.
  viewer = _nv.ViewerGL(width=_w, height=_h, headless=True)

  # Newton's convention is that a collider is not drawn -- the visual mesh is. mjlab's robot has 145
  # visual-only geoms so it renders fine, but the apple and the table are each a SINGLE geom that is
  # both visual and collidable, which arrives here as collide-but-not-visible. The result is a video
  # of a robot grasping thin air above an invisible table.
  #
  # The rule applied is per body rather than per shape: if a body has no visible shape at all, make
  # its colliders visible. That reveals the apple and table without double-drawing the robot.
  _flags = wp.to_torch(nmodel.shape_flags)
  _sbody = wp.to_torch(nmodel.shape_body)
  _VIS = int(newton.ShapeFlags.VISIBLE)
  _revealed = 0
  for _bid in torch.unique(_sbody):
    _idx = (_sbody == _bid).nonzero(as_tuple=True)[0]
    if int((_flags[_idx] & _VIS).sum()) == 0:
      _flags[_idx] |= _VIS
      _revealed += len(_idx)
  print(f"  made {_revealed} collider shape(s) visible (bodies that had no visual geometry)")

  viewer.set_model(nmodel)
  cam = getattr(viewer, "camera", None)
  if cam is not None:
    # Camera.pos must be an array -- get_view_matrix divides it by a scale factor, so a plain tuple
    # raises inside the first render. look_at() takes the target only; the eye comes from pos.
    try:
      cam.pos = np.array([1.75, -1.35, 1.40], dtype=np.float32)
      cam.look_at(np.array([0.80, -0.06, 0.86], dtype=np.float32))
      print(f"  camera at {np.round(np.asarray(cam.pos), 2).tolist()} looking at the table")
    except Exception as e:
      print(f"  camera placement failed ({type(e).__name__}: {e}); using the default view")
  print(f"newton ViewerGL: {_w}x{_h} headless, camera attrs "
        f"{[a for a in dir(cam) if not a.startswith('_')][:8] if cam else 'none'}")

# ------------------------------------------------------------------- rollout
nenv._force_reference_start_frame = 0

# Place robot, object and table at the reference start frame using mjlab's OWN reset event. Without
# it the scene starts at the model's default qpos: measured, the apple began 13 cm high and fell 82 cm
# to the floor, because Sonic53Action's startup hold writes the robot but not the object.
# With _force_reference_start_frame set, the curriculum event short-circuits to reset_to_apple_eat_frame.
ev = env_cfg.events if isinstance(env_cfg.events, dict) else vars(env_cfg.events)
reset_cfg = ev.get("reset_to_residual_interact_curriculum")
reset_params = dict(getattr(reset_cfg, "params", {}) or {})
print(f"reset event params: {reset_params}")
reset_cfg.func(nenv, None, **reset_params)
nenv.forward()      # qpos was written; xpos/xquat are stale until forward kinematics runs
print("scene reset to reference frame 0 (+ forward kinematics)")

state_in, state_out = nmodel.state(), nmodel.state()

from tensordict import TensorDict

# The residual policy wrapper records its per-step state by setting attributes on the env it was
# CONSTRUCTED with -- rl.py's _set_residual_action_stats does `env = self.env.unwrapped` -- and that
# is the mjlab env, because the runner needs one to resolve observation and action dimensions.
#
# Three observation groups read those attributes back: tracker_action, last_residual and astra_obs.
# On the Newton env they were never set, so mdp.py silently took its fallback path (teacher_action
# instead of the tracker's actual output) and those three groups were wrong by ~4.2 while every other
# group agreed to 1e-4. Nothing raised; the policy simply acted on a different world.
#
# The values themselves are correct -- the policy was fed Newton's observations to produce them --
# so they only need to be moved onto the env the feature builders read.
_RESIDUAL_STATS_ATTRS = (
  "_residual_last_action_mean", "_residual_last_astra_action_pkl", "_residual_last_base_action",
  "_residual_last_decoder_body_delta", "_residual_last_final_action",
  "_residual_last_raw_residual_action", "_residual_last_residual_action",
  "_residual_last_token_delta",
)


def _sync_residual_stats() -> int:
  src = mjenv.unwrapped
  n = 0
  for name in _RESIDUAL_STATS_ATTRS:
    v = getattr(src, name, None)
    if v is not None:
      setattr(nenv, name, v)
      n += 1
  return n


def _obs_dict():
  # mjlab hands the policy a TensorDict, not a plain dict (see evaluate_astra_body_only.py).
  out = {}
  for g, b in builders.items():
    try:
      out[g] = b(nenv)
    except Exception as e:
      raise RuntimeError(f"observation group {g!r} failed at runtime: {type(e).__name__}: {e}") from e
  return TensorDict(out, batch_size=[nenv.num_envs])

obj = nenv.scene["apple"]
qpos_log, mocap_log, rows = [], [], []
z0 = None
peak = {"rise": -99.0, "min_h2o": 9.9, "contact": 0.0}

N_FRAMES = int(amdp._ref(str(nenv.device))["n_frames"])
if A.until_ref_end:
  # The reference does not advance during the startup hold, so the clip ends about 40 control steps
  # later than its frame count; the loop also breaks on the last frame, so the margin only has to be
  # generous enough not to cut the clip short.
  A.steps = N_FRAMES + 60
  print(f"reference clip has {N_FRAMES} frames -> running {A.steps} control steps")

print(f"\nstepping Newton: dt={DT} decimation={DECIMATION} -> control dt {DT*DECIMATION}")
print(f"\n{'step':>5} {'frame':>6} {'obj_z':>8} {'rise_cm':>8} {'h2o_m':>8} {'nefc':>7}")
print("-" * 52)

for k in range(A.steps):
  obs = _obs_dict()
  with torch.inference_mode():
    action = policy(obs)
  n_sync = _sync_residual_stats()
  if k == 0:
    print(f"  residual-stats attributes synced onto the Newton env: {n_sync}/"
          f"{len(_RESIDUAL_STATS_ATTRS)}")
  amgr.advance(action)          # keeps last_final_action / last_residual live, as mjlab does
  action_term.process_actions(action)

  # apply_actions() goes INSIDE the decimation loop, exactly as mjlab's step() does:
  #   for _ in range(decimation): apply_action(); write_data_to_sim(); sim.step(); scene.update()
  # It is not just a ctrl write. Every call re-applies the startup root/joint hold, the table pose
  # and the reference object tracking, so calling it once per control step instead of once per
  # substep runs all of that at a quarter of its intended rate.
  for _ in range(DECIMATION):
    action_term.apply_actions()
    solver.step(state_in, state_out, control, None, DT)
    state_in, state_out = state_out, state_in

  nenv.episode_length_buf += 1

  if viewer is not None:
    # Drive the render from mjw_data, not from whatever Newton's State happens to hold. The table is
    # a mocap body: its pose is written to mocap_pos and never enters State, so it renders at the
    # origin and the apple appears to float in mid-air with nothing under it. Syncing every body
    # makes the picture match the simulation by construction rather than by coincidence.
    #
    # Both conventions here were measured, not assumed (tools/probes/probe_bodymap.py):
    # Newton body i is MuJoCo body i+1 (MuJoCo's body 0 is the world), and Newton stores quaternions
    # xyzw against MuJoCo's wxyz.
    _bq = wp.to_torch(state_in.body_q)
    _xp = wp.to_torch(solver.mjw_data.xpos)[0]
    _xq = wp.to_torch(solver.mjw_data.xquat)[0]
    _n = _bq.shape[0]
    _bq[:, 0:3] = _xp[1:1 + _n]
    _bq[:, 3:7] = _xq[1:1 + _n][:, [1, 2, 3, 0]]
    viewer.begin_frame(k * DT * DECIMATION)
    viewer.log_state(state_in)
    viewer.end_frame()
    img = viewer.get_frame()
    frames.append(np.asarray(img.numpy() if hasattr(img, "numpy") else img).copy())

  z = float(obj.data.root_link_pos_w[0, 2])
  if z0 is None:
    z0 = z
  rise = (z - z0) * 100.0
  peak["rise"] = max(peak["rise"], rise)

  # fingertip-to-object distance, using mjlab's own helper so it means the same thing it does there
  try:
    d = rmdp._tip_distances(nenv)
    h2o = float(d.min())
  except Exception:
    h2o = float("nan")
  if h2o == h2o:
    peak["min_h2o"] = min(peak["min_h2o"], h2o)

  # mujoco_warp exposes no scalar contact count here (`contact` is a fixed 256-slot struct), so the
  # active constraint-row count is reported instead. It is the quantity that actually matters for the
  # budget: this is what overflowed njmax and silently dropped the table contact (defect 4).
  _ne = solver.mjw_data.nefc
  nefc = int(np.asarray(_ne.numpy() if hasattr(_ne, "numpy") else _ne).reshape(-1)[0])
  if A.dump_qpos:
    qpos_log.append(wp.to_torch(solver.mjw_data.qpos)[0].detach().cpu().numpy().copy())
    # The table is a mocap body, so its pose lives outside qpos. Without it the render puts the
    # table at the origin and the apple appears to float.
    mocap_log.append((wp.to_torch(solver.mjw_data.mocap_pos)[0].detach().cpu().numpy().copy(),
                      wp.to_torch(solver.mjw_data.mocap_quat)[0].detach().cpu().numpy().copy()))
  frame = int(rmdp._tracking_frame(nenv, int(amdp._ref(str(nenv.device))["n_frames"]))[0])
  rows.append(dict(step=k, frame=frame, obj_z=z, rise_cm=rise, h2o=h2o, nefc=nefc))
  if k % A.every == 0 or k == A.steps - 1:
    print(f"{k:5d} {frame:6d} {z:8.3f} {rise:8.2f} {h2o:8.3f} {nefc:7d}")
  if A.until_ref_end and frame >= N_FRAMES - 1:
    print(f"{k:5d} {frame:6d} {z:8.3f} {rise:8.2f} {h2o:8.3f} {nefc:7d}   <- reference end")
    break

print("-" * 52)
print(f"max object rise = {peak['rise']:.2f} cm   (mjlab@3.8 49.72, mjlab@3.11 50.13)")
print(f"min fingertip-object distance = {peak['min_h2o']:.3f} m   (mjlab 0.032-0.035)")

if viewer is not None and frames:
  import imageio.v2 as _imageio
  arr = np.stack(frames)
  if arr.dtype != np.uint8:
    arr = np.clip(arr * (255.0 if arr.max() <= 1.01 else 1.0), 0, 255).astype(np.uint8)
  if arr.shape[-1] == 4:
    arr = arr[..., :3]
  # ViewerGL already returns rows top-down; flipping produced an upside-down video. Kept as an env
  # switch because framebuffer origin conventions are the sort of thing that changes between releases.
  if bool(int(os.environ.get("NEWTON_VIDEO_FLIP", "0"))):
    arr = arr[:, ::-1]
  wr = _imageio.get_writer(A.newton_video, fps=int(round(1.0 / (DT * DECIMATION))),
                           codec="libx264", macro_block_size=None, quality=8)
  for f_ in arr:
    wr.append_data(f_)
  wr.close()
  viewer.close()
  print(f"wrote {A.newton_video}  ({len(frames)} frames rendered by Newton's own viewer)")

if A.dump_qpos:
  np.savez_compressed(A.dump_qpos, qpos=np.stack(qpos_log),
                      mocap_pos=np.stack([m[0] for m in mocap_log]),
                      mocap_quat=np.stack([m[1] for m in mocap_log]),
                      rows=json.dumps(rows))
  print(f"wrote {A.dump_qpos}  ({len(qpos_log)} frames of qpos for rendering)")
