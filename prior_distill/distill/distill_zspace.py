#!/usr/bin/env python
"""zspace distillation: compress the whole-body grasp policy into a latent action VAE (DAgger).

What runs here, and how it maps onto Omnigrasp's PULSE-X distillation
(ZhengyiLuo/Omnigrasp: phc/env/tasks/humanoid_im_distill.py + phc/learning/amp_agent.py):

  PULSE-X                                       here
  ------------------------------------------    ---------------------------------------------------
  teacher: frozen PHC-X (3-primitive PNN)       teacher: a frozen ResidualInteractActorModel checkpoint
                                                (frozen ASTRA trunk + trained residual), 69-D mean action
  student drives the sim, teacher labels every  identical: the executed action is the student's (or, with
  state it reaches (DAgger, only_kin_loss=True)  beta > 0, the teacher's -- see --beta-*), the label is the
                                                teacher's deterministic mean on the SAME observation
  self_obs -> decoder & prior                   --self-groups   (default proprio_history: pure proprioception)
  self_obs + task_obs -> encoder                --task-groups   (goal / object / reference groups)
  gt_action = teacher PD target                 a_teacher = teacher `final_action` (69 = 29 body + 40 hand), in
                                                mjlab action units, exactly what the env's action term consumes
  loss: RMSE + kld*KL(q||p) + ar1 + regu        src/zspace.py:kin_loss, same terms, same defaults
  z_noise stored at act time, reused in update  same
  running mean/std on inputs                    same (RunningMeanStd on s and g)

Two things PULSE-X did not have to deal with:

  * The env's action term overrides the joint target with the reference pose for the first 36
    control steps of every episode (Sonic53Action.apply_actions: episode_length_buf <= 36).  The
    teacher's label there is never applied and carries no information, so those samples are
    masked out of every loss term (`valid`).
  * Our teacher cannot get up.  A randomly initialised student drives the humanoid into states the
    teacher never saw, and its labels there are noise.  --beta-start/--beta-end let the teacher
    drive a fraction of the envs early on (classic DAgger beta-mixing); the label is always the
    teacher's, the observation is always what actually happened.

The env attributes the observation groups read back (`_residual_last_final_action`,
`_residual_last_astra_action_pkl`, ...) are published for the EXECUTED action, not the teacher's,
so `astra_obs[5]`, `proprio_history[-69:]` and `last_final_action` stay truthful under DAgger.

Launch exactly like train_newton.py (same --config / --reference-pkls / --sdf-objects / env vars),
plus --resume <teacher checkpoint>.  Machine-local values (GMR_ROOT, SEED_CUBE, PYTHONPATH to the
mjlab tree the teacher was trained with) come from the environment, as for every other launch.
"""

from __future__ import annotations

import argparse
import json as _json
import math
import os
import sys
import time
from pathlib import Path

import yaml as _yaml

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
# --- env / teacher: the same flag names as train_newton.py so one --config serves both ---------
ap.add_argument("--config", default=None, help="YAML with env:/args: blocks (see train_newton.py)")
ap.add_argument("--num-envs", type=int, default=64)
ap.add_argument("--xml", default=os.path.join(_REPO, "assets/mjlab_scene/scene.xml"))
ap.add_argument("--agent-cfg-from", default=os.path.expanduser(
  "~/sweep_ckpts_r2/OF_00_apple_eat_1_SPHERE/model_7310.pt"),
  help="checkpoint whose params/agent.yaml supplies the agent config (tracker, residual arch)")
ap.add_argument("--reward-cfg", default=None)
ap.add_argument("--resume", default=None, help="TEACHER checkpoint (model_*.pt of the residual run); "
  "required in --mode full, unused in --mode body (the teacher is the frozen ASTRA base itself)")
ap.add_argument("--mode", choices=("full", "body"), default="full",
  help="full: distil the residual grasp policy (69-D label = body + hand). "
       "body: distil the frozen ASTRA tracker alone (29-D body label, hand executed at zero) -- "
       "the pure-tracking line, run on body-only clips (prior_distill/distill/make_body_clips.py)")
ap.add_argument("--reference-list", default=None,
  help="text file with one reference pkl path per line; alternative to --reference-pkls for large sets")
ap.add_argument("--sdf-object-all", default=None,
  help="one object mesh used for EVERY clip (body mode: the parked cube); replaces --sdf-objects")
ap.add_argument("--ref-root-term-m", type=float, default=0.0,
  help="PULSE-X's terminationDistance: end the episode once the root is this far (m) from the "
       "reference root, so the student never leaves the teacher's competence. 0 = off. Needs "
       "--keep-terminations (it takes over one of the silenced term slots).")
ap.add_argument("--keep-terminations", default="",
  help="comma list; when set, every other termination term is proxied to never fire "
       "(e.g. fell_over,fell_over_early,time_out for the body line). Physics unchanged.")
ap.add_argument("--reference-pkl", default=None)
ap.add_argument("--reference-pkls", default=None)
ap.add_argument("--sdf-objects", default=None)
ap.add_argument("--clip-env-counts", default=None)
ap.add_argument("--sdf-object", default=None)
ap.add_argument("--sdf-resolution", type=int, default=128)
ap.add_argument("--table-sdf-resolution", type=int, default=None)
ap.add_argument("--native-contacts", action="store_true")
ap.add_argument("--rigid-object-table", action="store_true")
ap.add_argument("--table-under-object", action="store_true")
ap.add_argument("--cuda-graph", action="store_true")
ap.add_argument("--object-solref", default="0.004,1.0")
ap.add_argument("--solver-kwargs", default=None)
ap.add_argument("--seed", type=int, default=1)
# accepted so a train_newton config's args: block applies cleanly; not used by the distiller
ap.add_argument("--iterations", type=int, default=None, help="(ignored; use --distill-iters)")
ap.add_argument("--log-root", default=None, help="(ignored; use --out-dir)")
ap.add_argument("--run-name", default=None, help="(ignored; use --out-dir)")
# --- distillation -----------------------------------------------------------------------------
ap.add_argument("--out-dir", required=True)
ap.add_argument("--distill-iters", type=int, default=2000)
ap.add_argument("--horizon", type=int, default=32, help="rollout steps per iteration (PULSE horizon_length)")
ap.add_argument("--epochs", type=int, default=5, help="gradient epochs over each collected batch")
ap.add_argument("--num-minibatches", type=int, default=4, help="env-chunks per epoch; AR(1) needs whole env rows")
ap.add_argument("--lr", type=float, default=5e-5, help="PULSE used 2e-5 at 3072 envs")
ap.add_argument("--grad-norm", type=float, default=50.0)
ap.add_argument("--latent", type=int, default=48)
ap.add_argument("--enc-units", default="1536,1024,512")
ap.add_argument("--dec-units", default="3096,2048,1024")
ap.add_argument("--self-groups", default="proprio_history",
  help="observation groups forming s (decoder + prior input). Proprioception only.")
ap.add_argument("--task-groups",
  default="astra_obs,reference_phase,reference_preview,object_state,hand_object_geometry,"
          "object_future,contact_features,tracking_error",
  help="observation groups forming g (encoder-only input)")
ap.add_argument("--kld-coef", type=float, default=0.01)
ap.add_argument("--kld-coef-min", type=float, default=0.001)
ap.add_argument("--kld-anneal-iters", type=int, default=None, help="default: --distill-iters")
ap.add_argument("--ar1-coef", type=float, default=0.005)
ap.add_argument("--regu-coef", type=float, default=0.0, help="0.005 = PULSE's use_vae_prior_regu weight")
ap.add_argument("--w-body", type=float, default=1.0)
ap.add_argument("--w-hand", type=float, default=1.0)
ap.add_argument("--beta-start", type=float, default=1.0, help="fraction of envs the TEACHER drives at iter 0")
ap.add_argument("--beta-end", type=float, default=0.0)
ap.add_argument("--beta-anneal-iters", type=int, default=None, help="default: --distill-iters // 3")
ap.add_argument("--startup-steps", type=int, default=36, help="labels with episode_length_buf <= this are void")
ap.add_argument("--save-every", type=int, default=50)
ap.add_argument("--eval-every", type=int, default=50)
ap.add_argument("--eval-steps", type=int, default=300, help="deterministic student-only rollout length")
ap.add_argument("--resume-student", default=None, help="continue from a student_*.pt")
ap.add_argument("--eval-teacher-ref", type=int, default=1,
  help="1: after each student-only eval, run the same rollout with the teacher driving and log it as eval_teacher_* (the ceiling)")
ap.add_argument("--probe-prior", default=None,
  help="diagnostic, no training: roll the (resumed) student's PRIOR alone (z = mu_p(s), no goal) and "
       "compare with a prior sample, the teacher, and a zero action; save per-step root height / fall "
       "data to this .npy, print a summary, exit. Answers 'does the prior keep the humanoid standing'")
ap.add_argument("--probe-steps", type=int, default=300)
ap.add_argument("--probe-root-err", default=None,
  help="diagnostic, no training: roll the (resumed) student and then the teacher for 240 steps, "
       "save per-step root error with steps-since-reset and clip id to this .npy, print bins, exit")
ap.add_argument("--teacher-check-steps", type=int, default=0,
  help="before training, roll the teacher alone for N steps and report its action spread and metrics")
A = ap.parse_args()

# ---------------------------------------------------------------------------------------------
# Config file: applied HERE, before any mjlab import (several env vars are read at import time).
# Same semantics as train_newton.py; unknown args keys are ignored with a note instead of
# refusing, so a training config can be reused unchanged.
# ---------------------------------------------------------------------------------------------
if A.config:
  _cfg_path = os.path.abspath(os.path.expanduser(A.config))
  if not os.path.exists(_cfg_path):
    raise SystemExit(f"--config {_cfg_path} does not exist")
  with open(_cfg_path) as _cf:
    _cfg = _yaml.safe_load(_cf) or {}
  _env_set, _env_kept = [], []
  for _k, _v in (_cfg.get("env") or {}).items():
    if _v is None:
      continue
    _sv = ("1" if _v is True else "0" if _v is False else str(_v))
    if os.environ.get(_k) not in (None, ""):
      _env_kept.append(f"{_k}={os.environ[_k]}")
    else:
      os.environ[_k] = _sv
      _env_set.append(f"{_k}={_sv}")
  _known = {a.dest for a in ap._actions}
  _arg_set, _arg_kept, _arg_ign = [], [], []
  for _k, _v in (_cfg.get("args") or {}).items():
    if _k not in _known:
      _arg_ign.append(_k)
      continue
    if f"--{_k.replace('_', '-')}" in sys.argv or f"--{_k}" in sys.argv:
      _arg_kept.append(_k)
      continue
    if _k == "solver_kwargs" and isinstance(_v, dict):
      _v = _json.dumps(_v)
    if isinstance(_v, str) and ("$" in _v or _v.startswith("~")):
      _exp = os.path.expandvars(os.path.expanduser(_v))
      if "$" in _exp:
        raise SystemExit(f"--config {_cfg_path}: args.{_k} = {_v!r} references an unset variable")
      _v = _exp
    setattr(A, _k, _v)
    _arg_set.append(f"{_k}={_v}")
  print(f"[config] {_cfg_path} :: {_cfg.get('name', '(unnamed)')}", flush=True)
  print(f"[config]   env  : {len(_env_set)} set" +
        (f"; kept from environment: {', '.join(_env_kept)}" if _env_kept else ""), flush=True)
  print(f"[config]   args : {len(_arg_set)} set" +
        (f"; command line wins for: {', '.join(_arg_kept)}" if _arg_kept else "") +
        (f"; ignored here: {', '.join(_arg_ign)}" if _arg_ign else ""), flush=True)

if A.mode == "full" and not A.resume:
  raise SystemExit("--resume <teacher checkpoint> is required in --mode full: there is nothing to distil otherwise")

MIX_PKLS: list[str] = []
MIX_STLS: list[str] = []
if A.reference_list:
  with open(os.path.expanduser(A.reference_list)) as _lf:
    MIX_PKLS = [ln.strip() for ln in _lf if ln.strip() and not ln.startswith("#")]
elif A.reference_pkls:
  MIX_PKLS = [x.strip() for x in A.reference_pkls.split(",") if x.strip()]
if MIX_PKLS:
  import pickle as _pickle
  if A.sdf_object_all:
    MIX_STLS = [A.sdf_object_all] * len(MIX_PKLS)
  else:
    MIX_STLS = [x.strip() for x in (A.sdf_objects or "").split(",") if x.strip()]
  if len(MIX_PKLS) < 2:
    raise SystemExit("--reference-pkls needs at least two clips; use --reference-pkl for one")
  if len(MIX_STLS) != len(MIX_PKLS):
    raise SystemExit(f"--sdf-objects has {len(MIX_STLS)} entries for {len(MIX_PKLS)} clip(s)")
  _missing = [p for p in MIX_PKLS if not os.path.exists(p)]
  if _missing:
    raise SystemExit(f"{len(_missing)} reference clips do not exist, e.g. {_missing[:3]}")
  if not A.sdf_object_all:  # a shared parked mesh is deliberately not the clip's object
    for _pkl, _stl in zip(MIX_PKLS, MIX_STLS):
      with open(_pkl, "rb") as _f:
        _want = str(_pickle.load(_f).get("obj_name", "")).strip().lower()
      _got = os.path.splitext(os.path.basename(_stl))[0].strip().lower()
      if _want and _got != _want:
        raise SystemExit(f"clip {os.path.basename(_pkl)} is about {_want!r} but was paired with "
                         f"{os.path.basename(_stl)!r}")
  os.environ["APPLE_EAT_PKL_MIX"] = ",".join(MIX_PKLS)
  os.environ["APPLE_EAT_PKL"] = MIX_PKLS[0]
  os.environ["APPLE_OBJECT_PER_WORLD"] = "1"
  print(f"[distill] MIX: {len(MIX_PKLS)} clips " + (
    ", ".join(f"{os.path.basename(p)}->{os.path.basename(t)}" for p, t in zip(MIX_PKLS, MIX_STLS))
    if len(MIX_PKLS) <= 16 else f"(first {os.path.basename(MIX_PKLS[0])} .. last {os.path.basename(MIX_PKLS[-1])}, "
    f"mesh {os.path.basename(MIX_STLS[0])} for all)"))
elif A.reference_pkl:
  os.environ["APPLE_EAT_PKL"] = A.reference_pkl

sys.path.insert(0, os.path.join(_REPO, "src"))
import mjw_compat  # noqa: E402
_p = mjw_compat.apply()
if _p:
  print(f"[compat] tolerating removed mujoco_warp options: {_p}")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from dataclasses import asdict  # noqa: E402

import mjlab.tasks  # noqa: F401,E402
from mjlab.rl import RslRlVecEnvWrapper, MjlabOnPolicyRunner  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402
from mjlab.scripts.play import _apply_cfg_mapping  # noqa: E402
from mjlab.tasks.residual_interact import mdp as ri_mdp  # noqa: E402
from newton_vec_env import NewtonVecEnv  # noqa: E402
from reward_cfg_from_checkpoint import reward_weights_from_env_yaml, apply_reward_weights  # noqa: E402
from zspace import (  # noqa: E402
  KinLossWeights, ZSpaceVAE, ZSpaceConfig, kin_loss, kld_weight, load_student, save_student,
)

TASK = "Mjlab-ResidualInteract-G1"
DEVICE = "cuda:0"
torch.manual_seed(A.seed)
np.random.seed(A.seed)
NUM_BODY = int(ri_mdp.NUM_BODY)
ACTION_DIM = int(ri_mdp.ACTION_DIM)

out_dir = Path(A.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------------------------
# env + frozen teacher, built the way train_newton.py builds them
# ---------------------------------------------------------------------------------------------
cfg = load_env_cfg(TASK, play=False)
cfg.scene.num_envs = A.num_envs
agent_cfg = load_rl_cfg(TASK)
_agent_yaml = Path(A.agent_cfg_from).parent / "params" / "agent.yaml"
if not _agent_yaml.exists():
  raise SystemExit(f"missing {_agent_yaml}: --agent-cfg-from must point at a checkpoint with params/")
_apply_cfg_mapping(agent_cfg, _yaml.unsafe_load(_agent_yaml.open()))

_env_yaml = (Path(A.reward_cfg) if A.reward_cfg else Path(A.agent_cfg_from).parent / "params" / "env.yaml")
if _env_yaml.exists():
  _w = reward_weights_from_env_yaml(_env_yaml)
  if not _w:
    raise SystemExit(f"[reward-cfg] {_env_yaml} parsed to zero reward terms")
  print(f"[reward-cfg] source: {_env_yaml}")
  apply_reward_weights(cfg, _w)
else:
  print("[reward-cfg] no env.yaml beside the agent checkpoint; rewards are only logged here anyway")

_s = cfg.actions.get("sonic_action") if isinstance(cfg.actions, dict) else cfg.actions.sonic_action
_s.tracking_start_assist_gain = 0.0
_s.tracking_start_assist_steps = 0
if str(getattr(agent_cfg, "base_tracker_kind", "")).strip().lower() == "astra_onnx":
  from mjlab.tasks.residual_interact.env_cfgs import set_astra_body_dynamics
  set_astra_body_dynamics(cfg)

print(f"[distill] building {A.num_envs} Newton worlds ...", flush=True)
env = NewtonVecEnv(
  cfg, A.xml, num_envs=A.num_envs, device=DEVICE,
  sdf_object_stl=A.sdf_object, sdf_resolution=A.sdf_resolution,
  sdf_object_stls=(MIX_STLS or None),
  clip_env_counts=([int(x) for x in A.clip_env_counts.split(",")] if A.clip_env_counts else None),
  native_contacts=A.native_contacts,
  hydro_object_table=not A.rigid_object_table,
  table_sdf_resolution=A.table_sdf_resolution,
  cuda_graph=A.cuda_graph,
  table_under_object=A.table_under_object,
  object_solref=A.object_solref,
  solver_kwargs=(_json.loads(A.solver_kwargs) if A.solver_kwargs else None),
)
if A.keep_terminations:
  # Same mechanism as train_newton.py's ROLLOUT_NO_TERM, but keeping a named subset. Physics is
  # untouched; only whether an episode is reset changes.
  _keep = {x.strip() for x in A.keep_terminations.split(",") if x.strip()}

  class _NeverFires:
    def __init__(self, inner): self._inner = inner
    def __getattr__(self, k): return getattr(self._inner, k)
    def __call__(self, env_, *a, **k):
      return torch.zeros(env_.num_envs, dtype=torch.bool, device=env_.device)

  _tm = env.termination_manager
  _off = []
  for _ni, _nn in enumerate(_tm._term_names):
    if _nn in _keep:
      continue
    _tm._term_cfgs[_ni].func = _NeverFires(_tm._term_cfgs[_ni].func)
    _off.append(_nn)
  _unknown = _keep - set(_tm._term_names)
  if _unknown:
    raise SystemExit(f"--keep-terminations names unknown terms {sorted(_unknown)}; have {_tm._term_names}")
  print(f"[distill] terminations kept: {sorted(_keep)}; never fire: {_off}", flush=True)

  if A.ref_root_term_m > 0.0:
    from mjlab.tasks.apple_eat import mdp as _apple_mdp

    class _RootFar:
      """PHC/PULSE `terminationDistance`: root further than d from the reference root -> reset.

      Inactive during the 36-step startup override (the root is hard-held there anyway).
      """
      def __init__(self, inner, d: float):
        self._inner = inner
        self.d = float(d)
      def __getattr__(self, k): return getattr(self._inner, k)
      def __call__(self, env_, *a, **k):
        ref = _apple_mdp._ref(env_.device)
        frame = _apple_mdp._tracking_frame(env_, int(ref["n_frames"]))
        robot = env_.scene["robot"]
        ref_root = ref["root_pos"][frame] + env_.scene.env_origins
        dist = (robot.data.root_link_pose_w[:, :3] - ref_root).norm(dim=-1)
        return (dist > self.d) & (env_.episode_length_buf > A.startup_steps)

    if not _off:
      raise SystemExit("--ref-root-term-m needs a silenced termination slot; keep fewer terms")
    _slot = _tm._term_names.index(_off[0])
    _tm._term_cfgs[_slot].func = _RootFar(_tm._term_cfgs[_slot].func, A.ref_root_term_m)
    print(f"[distill] reference-root termination at {A.ref_root_term_m} m installed in slot '{_off[0]}'", flush=True)

wrapped = RslRlVecEnvWrapper(env)
runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
runner = runner_cls(wrapped, asdict(agent_cfg), log_dir=str(out_dir / "teacher_runner"), device=DEVICE)
if A.resume:
  runner.load(A.resume)
  print(f"[distill] teacher checkpoint: {A.resume}", flush=True)
actor = runner.alg.actor
base_tracker = actor.base_tracker
if not hasattr(base_tracker, "sync_last_astra_action_from_mjlab"):
  raise SystemExit("teacher's base tracker is not the ASTRA actor; this distiller assumes base_tracker_kind=astra_onnx")
_full_policy = runner.get_inference_policy(device=DEVICE)  # deterministic mean, actor in eval mode

if A.mode == "full":
  LABEL_DIM = ACTION_DIM
  label_fn = _full_policy  # 69-D final_action of the residual policy

  def to_executed(a_label: torch.Tensor) -> torch.Tensor:
    return a_label
else:
  LABEL_DIM = NUM_BODY
  if A.resume:
    raise SystemExit("--mode body takes no --resume: the teacher is the frozen ASTRA base, not a residual checkpoint")

  def label_fn(obs) -> torch.Tensor:
    # The residual actor with NO checkpoint loaded: its residual head is zero-initialised, so the
    # 69-D output is exactly [ASTRA body(29) | zero hand(40)] -- verified once below against the
    # bare base tracker. Running the full actor (rather than base_tracker alone) keeps every
    # `last_*` attribute the env metrics read at the right (N, ...) shape.
    with torch.no_grad():
      return _full_policy(obs)[:, :NUM_BODY]

  _o = wrapped.get_observations()
  with torch.no_grad():
    _a_full = _full_policy(_o)
    _a_base = base_tracker(_o)
  _dbody = float((_a_full[:, :NUM_BODY] - _a_base[:, :NUM_BODY]).abs().max())
  _dhand = float(_a_full[:, NUM_BODY:].abs().max())
  if _dbody > 1e-5 or _dhand > 1e-5:
    raise SystemExit(f"body mode expects a zero residual: actor-vs-ASTRA body diff {_dbody:.2e}, hand |a| {_dhand:.2e}")
  print(f"[distill] body mode check: actor == ASTRA base (body diff {_dbody:.1e}, hand {_dhand:.1e})", flush=True)

  def to_executed(a_label: torch.Tensor) -> torch.Tensor:
    return torch.cat([a_label, torch.zeros(a_label.shape[0], ACTION_DIM - NUM_BODY,
                                           device=a_label.device, dtype=a_label.dtype)], dim=-1)
  print("[distill] mode=body: teacher = frozen ASTRA tracker (29-D), hand executed at zero", flush=True)


def publish_executed(executed: torch.Tensor) -> None:
  """Publish the env-side action cache for the action that was actually applied.

  The runner's `_set_residual_action_stats` fills every `_residual_last_*` attribute the
  observation groups read from the actor's own state after a forward pass.  Two of those must
  describe the EXECUTED action rather than the teacher's: `_residual_last_final_action` (it takes
  `actions` as given) and `_residual_last_astra_action_pkl`, which it copies from
  `base_tracker.last_astra_action_pkl` -- so that is re-derived from the executed body action
  first (`sync_last_astra_action_from_mjlab` is the exact inverse of the ASTRA->mjlab decode).
  """
  base_tracker.sync_last_astra_action_from_mjlab(executed[:, :NUM_BODY])
  runner._set_residual_action_stats(actor, executed)


# ---------------------------------------------------------------------------------------------
# student
# ---------------------------------------------------------------------------------------------
self_groups = tuple(x.strip() for x in A.self_groups.split(",") if x.strip())
task_groups = tuple(x.strip() for x in A.task_groups.split(",") if x.strip())


def _parse_group(spec: str) -> tuple[str, slice]:
  """'name' or 'name[a:b]' -> (name, slice). Used to take e.g. the reference-command part of astra_obs."""
  if "[" in spec:
    name, rng = spec[:-1].split("[")
    a, b = rng.split(":")
    return name, slice(int(a) if a else None, int(b) if b else None)
  return spec, slice(None)


def _group_tensor(obs, spec: str) -> torch.Tensor:
  name, sl = _parse_group(spec)
  t = obs[name]
  return t.reshape(t.shape[0], -1)[:, sl]


def split_obs(obs) -> tuple[torch.Tensor, torch.Tensor]:
  s = torch.cat([_group_tensor(obs, g) for g in self_groups], dim=-1)
  g_ = torch.cat([_group_tensor(obs, g) for g in task_groups], dim=-1)
  return s.float(), g_.float()


obs, _ = wrapped.reset()
missing = [g for g in (*self_groups, *task_groups) if _parse_group(g)[0] not in obs.keys()]
if missing:
  raise SystemExit(f"observation groups not produced by the env: {missing}; available: {sorted(obs.keys())}")
s0, g0 = split_obs(obs)
group_dims = {g: int(_group_tensor(obs, g).shape[-1]) for g in (*self_groups, *task_groups)}
print(f"[distill] self_obs = {self_groups} -> {s0.shape[-1]} dims; task_obs = {task_groups} -> {g0.shape[-1]} dims",
      flush=True)
print(f"[distill] group dims: {group_dims}", flush=True)

if A.resume_student:
  student, _payload = load_student(A.resume_student, DEVICE)
  student.self_norm.frozen = False
  student.task_norm.frozen = False
  if student.cfg.self_dim != s0.shape[-1] or student.cfg.task_dim != g0.shape[-1]:
    raise SystemExit("--resume-student dims do not match the current observation groups")
  start_iter = int(_payload.get("iteration", 0)) + 1
  print(f"[distill] resumed student from {A.resume_student} at iteration {start_iter}", flush=True)
else:
  student = ZSpaceVAE(ZSpaceConfig(
    self_dim=int(s0.shape[-1]), task_dim=int(g0.shape[-1]), action_dim=LABEL_DIM,
    latent_dim=A.latent,
    enc_units=tuple(int(x) for x in A.enc_units.split(",")),
    dec_units=tuple(int(x) for x in A.dec_units.split(",")),
    self_groups=self_groups, task_groups=task_groups,
  )).to(DEVICE)
  start_iter = 0
student.train()
n_params = sum(p.numel() for p in student.parameters())
print(f"[distill] student: latent={A.latent} enc={A.enc_units} dec={A.dec_units} params={n_params:,}", flush=True)
opt = torch.optim.Adam(student.parameters(), lr=A.lr)
weights = KinLossWeights(kld=A.kld_coef, ar1=A.ar1_coef, regu=A.regu_coef, w_body=A.w_body, w_hand=A.w_hand)
kld_anneal_iters = A.kld_anneal_iters if A.kld_anneal_iters is not None else A.distill_iters
beta_anneal_iters = A.beta_anneal_iters if A.beta_anneal_iters is not None else max(A.distill_iters // 3, 1)

with open(out_dir / "distill_config.json", "w") as f:
  _json.dump({"args": vars(A), "group_dims": group_dims, "self_groups": self_groups,
              "task_groups": task_groups, "action_dim": LABEL_DIM, "num_body": NUM_BODY,
              "mode": A.mode, "n_clips": len(MIX_PKLS) or 1}, f, indent=2)


def beta_at(it: int) -> float:
  frac = min(it / float(beta_anneal_iters), 1.0)
  return A.beta_start + (A.beta_end - A.beta_start) * frac


N = A.num_envs
H = A.horizon
E = A.latent
ep_len_buf = env.episode_length_buf  # the live tensor; read before every step
drive_teacher = torch.zeros(N, dtype=torch.bool, device=DEVICE)


def redraw_drive(mask: torch.Tensor, beta: float) -> None:
  if mask.any():
    drive_teacher[mask] = torch.rand(int(mask.sum()), device=DEVICE) < beta


def harvest_log(extras: dict, acc: dict[str, list]) -> None:
  for k, v in (extras.get("log") or {}).items():
    if "lift_success" in k or "Episode_Termination" in k or k.endswith("episode_length"):
      try:
        acc.setdefault(k, []).append(float(v))
      except Exception:
        pass


# ---------------------------------------------------------------------------------------------
# optional: teacher-only sanity run
# ---------------------------------------------------------------------------------------------
if A.teacher_check_steps > 0:
  acts, acc = [], {}
  for _ in range(A.teacher_check_steps):
    with torch.no_grad():
      a_t = to_executed(label_fn(obs))
    acts.append(a_t.detach())
    publish_executed(a_t)
    obs, _, dones, extras = wrapped.step(a_t)
    harvest_log(extras, acc)
  acts_t = torch.stack(acts)  # [T, N, A]
  spread = acts_t.std(dim=0).mean().item()
  print(f"[teacher-check] {A.teacher_check_steps} steps, action std over time (mean over envs/dims) = {spread:.4f}"
        + ("  <-- CONSTANT ACTIONS: the observation cache trap; use more envs" if spread < 1e-4 else ""), flush=True)
  for k in sorted(acc):
    print(f"[teacher-check]   {k}: {np.mean(acc[k]):.4f} over {len(acc[k])} reports", flush=True)
  obs, _ = wrapped.reset()

if A.probe_prior:
  # PULSE's first sanity check on a learned prior: with no goal at all, z = mu_p(s) should keep the
  # humanoid upright and moving like a person; if it falls, the prior is not a usable action space.
  # Four drivers on identical starts (RSI over all clips): the prior mean, a prior sample, the
  # teacher (upper bound: it sees the goal), and a zero action (lower bound: the PD default pose).
  robot = env.scene["robot"]
  _rows = []
  student.eval()
  for _mode in ("prior_mean", "prior_sample", "teacher", "zero"):
    obs, _ = wrapped.reset()
    _fell = torch.zeros(N, dtype=torch.bool, device=DEVICE)
    _t_fall = torch.full((N,), float(A.probe_steps), device=DEVICE)
    for _t in range(A.probe_steps):
      with torch.no_grad():
        _s, _g = split_obs(obs)
        if _mode == "prior_mean":
          _a = student.act_from_prior(_s, sample=False)
        elif _mode == "prior_sample":
          _a = student.act_from_prior(_s, sample=True)
        elif _mode == "teacher":
          _a = label_fn(obs)
        else:
          _a = torch.zeros(N, LABEL_DIM, device=DEVICE)
        _ax = to_executed(_a)
      _z = robot.data.root_link_pose_w[:, 2]
      _v = robot.data.root_link_lin_vel_w[:, :2].norm(dim=-1)
      _qd = robot.data.joint_vel[:, :NUM_BODY].abs().mean(-1)
      _rows.append(np.stack([np.full(N, ("prior_mean", "prior_sample", "teacher", "zero").index(_mode)),
                             np.full(N, _t), _z.cpu().numpy(), _v.cpu().numpy(), _qd.cpu().numpy()], 1))
      publish_executed(_ax)
      obs, _, _d, _x = wrapped.step(_ax)
      _now = _d.bool() & ~_fell
      if _t + 1 < A.probe_steps:  # a done before the last step is a fall (only fell_over/time_out are live)
        _t_fall[_now & (_t_fall >= A.probe_steps)] = float(_t + 1)
        _fell |= _now
    _R = np.concatenate(_rows[-A.probe_steps:])
    _late = _R[_R[:, 1] >= A.probe_steps - 60]
    print(f"[probe-prior] {_mode:12s}: fell within {A.probe_steps} steps {float(_fell.float().mean()):.3f} "
          f"(median time-to-fall of those {float(_t_fall[_fell].median()) if _fell.any() else float('nan'):.0f}) | "
          f"root z last 60 steps {float(_late[:, 2].mean()):.3f} m | xy speed {float(_late[:, 3].mean()):.3f} m/s | "
          f"mean |qd| {float(_late[:, 4].mean()):.3f} rad/s", flush=True)
  np.save(A.probe_prior, np.concatenate(_rows))
  raise SystemExit(0)

if A.probe_root_err:
  # Is the eval's flat root error an offset present from the first step (frame/init), or drift?
  # Same rollout as the eval; the teacher is run the same way as the control.
  _ref_ = ri_mdp.apple_mdp._ref(DEVICE)
  _ncl = int(_ref_.get("n_clips", 1))
  _clip = torch.arange(N, device=DEVICE) % _ncl
  _rows = []
  for _mode in ("student", "teacher"):
    obs, _ = wrapped.reset()
    student.eval()
    for _t in range(240):
      with torch.no_grad():
        if _mode == "student":
          _s, _g = split_obs(obs)
          _a, _ = student(_s, _g, deterministic=True)
        else:
          _a = label_fn(obs)
        _ax = to_executed(_a)
      _te = obs["tracking_error"].reshape(obs["tracking_error"].shape[0], -1)
      _r = _te[:, :3].norm(dim=-1)
      _rows.append(np.stack([np.full(N, 0 if _mode == "student" else 1),
                             env.episode_length_buf.cpu().numpy(), _clip.cpu().numpy(),
                             _r.cpu().numpy(), _te[:, 9:9 + NUM_BODY].abs().mean(-1).cpu().numpy()], 1))
      publish_executed(_ax)
      obs, _, _d, _x = wrapped.step(_ax)
  _R = np.concatenate(_rows)
  np.save(A.probe_root_err, _R)
  for _m, _name in ((0, "student"), (1, "teacher")):
    _X = _R[_R[:, 0] == _m]
    print(f"[probe] {_name}: root err (m) / joint err (rad) by steps-since-reset")
    for _lo in range(0, 240, 40):
      _w = _X[(_X[:, 1] >= _lo) & (_X[:, 1] < _lo + 40)]
      if len(_w):
        print(f"[probe]   step {_lo:3d}-{_lo+39:3d}: root {_w[:, 3].mean():.3f}  joint {_w[:, 4].mean():.3f}  (n={len(_w)})")
    _pc = np.array([_X[_X[:, 2] == c, 3].mean() for c in range(_ncl)])
    print(f"[probe]   per-clip root err: median {np.median(_pc):.3f} p10 {np.percentile(_pc, 10):.3f} "
          f"p90 {np.percentile(_pc, 90):.3f} max {_pc.max():.3f} (clip {int(_pc.argmax())})", flush=True)
  raise SystemExit(0)

# ---------------------------------------------------------------------------------------------
# DAgger loop
# ---------------------------------------------------------------------------------------------
Ds, Dg = int(s0.shape[-1]), int(g0.shape[-1])
buf_s = torch.zeros(H, N, Ds, device=DEVICE)
buf_g = torch.zeros(H, N, Dg, device=DEVICE)
buf_a = torch.zeros(H, N, LABEL_DIM, device=DEVICE)
buf_noise = torch.zeros(H, N, E, device=DEVICE)
buf_len = torch.zeros(H, N, dtype=torch.long, device=DEVICE)
buf_student = torch.zeros(H, N, dtype=torch.bool, device=DEVICE)
buf_student_err = torch.zeros(H, N, device=DEVICE)

log_f = open(out_dir / "log.jsonl", "a")
redraw_drive(torch.ones(N, dtype=torch.bool, device=DEVICE), beta_at(start_iter))
t_start = time.time()

for it in range(start_iter, A.distill_iters):
  beta = beta_at(it)
  weights.kld = kld_weight(it, A.kld_coef, A.kld_coef_min, kld_anneal_iters)
  t0 = time.time()
  acc: dict[str, list] = {}
  done_lens: list[float] = []
  student.eval()  # act; dropout-free anyway, but keeps BN-style semantics explicit
  with torch.no_grad():
    for t in range(H):
      ep_len = ep_len_buf.clone()
      a_t = label_fn(obs)  # label; also refreshes the actor's internal last_* state
      s, g_ = split_obs(obs)
      a_s, info = student(s, g_, deterministic=False)
      executed = to_executed(torch.where(drive_teacher.unsqueeze(-1), a_t, a_s))
      buf_s[t], buf_g[t], buf_a[t], buf_noise[t], buf_len[t] = s, g_, a_t, info["noise"], ep_len
      buf_student[t] = ~drive_teacher
      buf_student_err[t] = torch.norm(a_s - a_t, dim=-1)
      publish_executed(executed)
      obs, _, dones, extras = wrapped.step(executed)
      harvest_log(extras, acc)
      d = dones.bool()
      if d.any():
        done_lens.extend((ep_len[d] + 1).float().tolist())
        redraw_drive(d, beta)
  t_collect = time.time() - t0

  # --- update -------------------------------------------------------------------------------
  t1 = time.time()
  student.train()
  valid = buf_len > A.startup_steps  # [H, N]
  consecutive = (buf_len[1:] == buf_len[:-1] + 1) & valid[1:] & valid[:-1]  # [H-1, N]
  with torch.no_grad():
    v_flat = valid.reshape(-1)
    if v_flat.any():
      student.self_norm.update(buf_s.reshape(-1, Ds)[v_flat])
      student.task_norm.update(buf_g.reshape(-1, Dg)[v_flat])
  # env-major views so a minibatch is a set of whole env rows (AR(1) needs the time axis)
  S_ = buf_s.transpose(0, 1)
  G_ = buf_g.transpose(0, 1)
  A_ = buf_a.transpose(0, 1)
  Z_ = buf_noise.transpose(0, 1)
  V_ = valid.transpose(0, 1)
  C_ = consecutive.transpose(0, 1)
  B = max(N // A.num_minibatches, 1)
  infos: list[dict] = []
  for _ep in range(A.epochs):
    perm = torch.randperm(N, device=DEVICE)
    for i in range(0, N, B):
      idx = perm[i:i + B]
      if not V_[idx].any():
        continue
      loss, info = kin_loss(student, S_[idx], G_[idx], A_[idx], Z_[idx], V_[idx], C_[idx], weights, NUM_BODY)
      opt.zero_grad(set_to_none=True)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(student.parameters(), A.grad_norm)
      opt.step()
      infos.append(info)
  t_update = time.time() - t1

  # --- report -------------------------------------------------------------------------------
  mean_info = {k: float(np.mean([d[k] for d in infos])) for k in infos[0]} if infos else {}
  sv = (valid & buf_student)
  student_err = float(buf_student_err[sv].mean()) if sv.any() else float("nan")
  row = {
    "iter": it, "beta": beta, "kld_w": weights.kld,
    "valid_frac": float(valid.float().mean()),
    "student_driven_frac": float(buf_student.float().mean()),
    "act_rmse_student_driven": student_err,
    "mean_done_len": float(np.mean(done_lens)) if done_lens else float("nan"),
    "n_done": len(done_lens),
    "t_collect": t_collect, "t_update": t_update,
    "fps": H * N / max(t_collect, 1e-6),
    **mean_info,
    **{k: float(np.mean(v)) for k, v in acc.items()},
  }
  log_f.write(_json.dumps(row) + "\n")
  log_f.flush()
  lift = {k.split("/")[-1]: v for k, v in acc.items() if "lift_success" in k}
  if not infos:
    # every env still inside the startup override (fresh reset + horizon < startup_steps):
    # nothing to fit yet, which is the expected shape of iteration 0, not a failure
    print(f"[it {it:5d}] no valid labels this iteration (valid {row['valid_frac']:.2f}; all envs in the "
          f"{A.startup_steps}-step startup override) | beta {beta:.2f} | {t_collect:.1f}s {row['fps']:.0f} fps",
          flush=True)
    continue
  print(
    f"[it {it:5d}] loss {row.get('loss', float('nan')):.4f} act {row.get('kin_action_loss', float('nan')):.4f} "
    f"(body {row.get('kin_body_rmse', float('nan')):.4f} hand {row.get('kin_hand_rmse', float('nan')):.4f}) "
    f"KLD {row.get('kin_KLD', float('nan')):.3f}*{weights.kld:.4f} ar1 {row.get('kin_ar1', float('nan')):.4f} "
    f"| beta {beta:.2f} valid {row['valid_frac']:.2f} rmse@student {student_err:.4f} "
    f"| done {len(done_lens)} len {row['mean_done_len']:.0f} "
    f"| {t_collect:.1f}s+{t_update:.1f}s {row['fps']:.0f} fps"
    + (" | lift " + " ".join(f"{k}={np.mean(v):.2f}" for k, v in sorted(lift.items())) if lift else ""),
    flush=True,
  )

  if (it + 1) % A.save_every == 0 or it + 1 == A.distill_iters:
    extra = {"iteration": it, "teacher": A.resume, "group_dims": group_dims}
    save_student(str(out_dir / f"student_{it + 1:06d}.pt"), student, extra)
    save_student(str(out_dir / "student_latest.pt"), student, extra)

  # --- eval: deterministic student (z = mu_q), student drives every env --------------------------
  if A.eval_every > 0 and ((it + 1) % A.eval_every == 0 or it + 1 == A.distill_iters):
    student.eval()

    def _eval_rollout(driver: str):
      """Roll `driver` ('student' | 'teacher') for A.eval_steps; return the eval metrics dict."""
      global obs
      errs_b, errs_h, eacc, elens = [], [], {}, []
      root_errs, q_errs = [], []  # metres / radians against the reference, from tracking_error
      with torch.no_grad():
        for _t in range(A.eval_steps):
          ep_len = ep_len_buf.clone()
          a_t = label_fn(obs)
          if driver == "student":
            s, g_ = split_obs(obs)
            a_d, _ = student(s, g_, deterministic=True)
          else:
            a_d = a_t
          ok = ep_len > A.startup_steps
          if ok.any():
            errs_b.append(torch.norm((a_d - a_t)[ok, :NUM_BODY], dim=-1).mean().item())
            errs_h.append(torch.norm((a_d - a_t)[ok, NUM_BODY:], dim=-1).mean().item())
            if "tracking_error" in obs.keys():
              te = obs["tracking_error"].reshape(obs["tracking_error"].shape[0], -1)[ok]
              root_errs.append(te[:, :3].norm(dim=-1).mean().item())          # root_pos_err (heading-local, m)
              q_errs.append(te[:, 9:9 + NUM_BODY].abs().mean().item())        # q_err body joints (rad)
          a_x = to_executed(a_d)
          publish_executed(a_x)
          obs, _, dones, extras = wrapped.step(a_x)
          harvest_log(extras, eacc)
          d = dones.bool()
          if d.any():
            elens.extend((ep_len[d] + 1).float().tolist())
      return {"body_rmse": float(np.mean(errs_b)) if errs_b else float("nan"),
              "hand_rmse": float(np.mean(errs_h)) if errs_h else float("nan"),
              "mean_done_len": float(np.mean(elens)) if elens else float("nan"),
              "n_done": len(elens),
              "root_err_m": float(np.mean(root_errs)) if root_errs else float("nan"),
              "body_q_err_rad": float(np.mean(q_errs)) if q_errs else float("nan"),
              **{k: float(np.mean(v)) for k, v in eacc.items()}}

    _es = _eval_rollout("student")
    # Same rollout with the teacher driving: the ceiling every student number is judged against
    # (ASTRA drifts ~20 cm from the reference within 4 s on GRAB body clips, so absolute root
    # error and survival are bounded by the teacher, not the student).
    _et = _eval_rollout("teacher") if A.eval_teacher_ref else {}
    erow = {"iter": it, "eval": True,
            **{"eval_" + k: v for k, v in _es.items()},
            **{"eval_teacher_" + k: v for k, v in _et.items()}}
    log_f.write(_json.dumps(erow) + "\n")
    log_f.flush()
    elift = {k.split("/")[-1]: v for k, v in _es.items() if "lift_success" in k}
    print(f"[eval it {it:5d}] student-only {A.eval_steps} steps: body rmse {_es['body_rmse']:.4f} "
          f"hand rmse {_es['hand_rmse']:.4f} | root err {_es['root_err_m']:.3f} m q err {_es['body_q_err_rad']:.3f} rad"
          f" | episodes {_es['n_done']} mean len {_es['mean_done_len']:.0f}"
          + (f" || teacher: root err {_et['root_err_m']:.3f} m q err {_et['body_q_err_rad']:.3f} rad "
             f"episodes {_et['n_done']} mean len {_et['mean_done_len']:.0f}" if _et else "")
          + (" | lift " + " ".join(f"{k}={v:.2f}" for k, v in sorted(elift.items())) if elift else ""), flush=True)
    # the env continues from wherever the eval left it; redraw who drives with the current beta
    redraw_drive(torch.ones(N, dtype=torch.bool, device=DEVICE), beta)

print(f"[distill] done: {A.distill_iters - start_iter} iterations in {(time.time() - t_start) / 60:.1f} min; "
      f"student at {out_dir / 'student_latest.pt'}", flush=True)
