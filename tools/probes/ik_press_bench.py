"""A scripted press bench for the hand-object penetration problem. No policy, no reward, no RSI.

Why this exists: every comparison of two ROLLOUTS under different contact settings is confounded.
A setting that makes the hand weaker also changes where the policy puts its hand, so a rollout whose
grasp got worse stands further off, presses more lightly, and scores a SHALLOWER overlap for
entirely the wrong reason. Three verdicts have already been retracted for exactly that.

So nothing here is left to a policy. The scene is the training scene -- same MJCF, same SDF object
collider, same solver kwargs, same contact recipe as R30/R31 -- built at one world. The object is
placed at the centroid of the five right fingertips, clear of contact, by construction. The finger
joints are then driven open-loop to a commanded target. The command is a literal constant, so any
difference in settled overlap is the physics and nothing else.

Two control modes, because they fail differently:

  * POSITION: command the finger joints a chosen distance PAST the surface. A physically correct
    simulator flattens -- commanding deeper stops producing deeper. Whatever depth survives is the
    wall failing.
  * FORCE: cap the finger actuator torque so the tip force is bounded and known, and sweep it.
    Gives overlap versus force in mm/N, the one quantity comparable across settings with no
    confound at all.

Every setting is applied to ONE built model in place and read back off the compiled model before
the press runs (`--mode facts` dumps the whole table). Two full A/B training runs were wasted on
switches that printed their new value and simulated the old one; nothing here is trusted because it
printed.

    python tools/probes/ik_press_bench.py --mode facts
    python tools/probes/ik_press_bench.py --mode position
    python tools/probes/ik_press_bench.py --mode force
"""
from __future__ import annotations

import argparse
import json as _json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="facts", choices=("facts", "position", "force", "pose"))
ap.add_argument("--xml", default=os.path.join(ROOT, "assets/scene_stapler/scene.xml"))
ap.add_argument("--sdf-object", default=os.path.expanduser(
    "~/jiarui/scaled_grab_dataset_wuji_all/meshes/cubesmall.stl"))
ap.add_argument("--reference-pkl", default=os.path.expanduser(
    "~/a85_cube/step6_grasp_blended/s1/cubesmall_lift.pkl"))
ap.add_argument("--agent-cfg-from", default=os.path.expanduser(
    "~/sweep_ckpts_r2/OF_00_apple_eat_1_SPHERE/model_7310.pt"))
ap.add_argument("--reward-cfg", default=os.path.join(ROOT, "configs/rewards/staged_cf_r24.yaml"))
ap.add_argument("--solver-kwargs",
                default='{"impratio": 20.0, "cone": "pyramidal", "iterations": 100, '
                        '"ls_iterations": 50}',
                help="R30/R31's own value; kept identical so the bench is the trained physics")
ap.add_argument("--sdf-resolution", type=int, default=128)
ap.add_argument("--native-contacts", type=int, default=1)
ap.add_argument("--settle", type=int, default=600, help="substeps to settle before reading")
ap.add_argument("--hold", type=int, default=100, help="substeps over which rest is checked")
ap.add_argument("--depths", default="1,3,5,10,20", help="commanded depths, mm (position mode)")
ap.add_argument("--forces", default="0.05,0.15,0.62,2,10,30",
                help="finger actuator torque caps, N*m (force mode)")
ap.add_argument("--settings", default="baseline",
                help="comma-separated condition names from CONDITIONS, or 'all'")
ap.add_argument("--object-offset", default="0,0,0",
                help="extra offset of the object from the fingertip centroid, m (x,y,z world)")
ap.add_argument("--out", default=None, help="write the table as csv here as well")
A = ap.parse_args()

# Read straight off R30_CUBE's own /proc/<pid>/environ, so the bench scene is the trained scene.
# APPLE_HAND_KIND in particular is not optional: the default 'xhand' rejects this clip's 69 dof
# columns outright, and a wrong hand kind would silently bench a different robot.
os.environ.setdefault("APPLE_HAND_KIND", "wuji")
os.environ.setdefault("APPLE_OBJECT_PER_WORLD", "1")
os.environ.setdefault("APPLE_SCENE_Z_OFFSET", "-0.03")
os.environ.setdefault("APPLE_EAT_PKL", A.reference_pkl)
os.environ.setdefault("PEN_LOG", "0")
sys.path.insert(0, os.path.join(ROOT, "src"))

import mjw_compat  # noqa: E402

mjw_compat.apply()

import mujoco  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402
import yaml as _yaml  # noqa: E402
from pathlib import Path  # noqa: E402

import mjlab.tasks  # noqa: F401,E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: E402
from mjlab.scripts.play import _apply_cfg_mapping  # noqa: E402
from newton_vec_env import NewtonVecEnv  # noqa: E402
from reward_cfg_from_checkpoint import reward_weights_from_env_yaml, apply_reward_weights  # noqa: E402

TASK = "Mjlab-ResidualInteract-G1"


# --------------------------------------------------------------------------------------- helpers
def gname(m, g):
  return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f"<geom{g}>"


def jname(m, j):
  return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(j)) or f"<jnt{j}>"


def bname(m, b):
  return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b)) or f"<body{b}>"


GEOM_TYPE = {0: "plane", 1: "hfield", 2: "sphere", 3: "capsule", 4: "ellipsoid",
             5: "cylinder", 6: "box", 7: "mesh", 8: "sdf"}


def push(solver, field, host_array):
  """Copy a host mj_model field onto the compiled mjw_model, and report the mjw shape.

  Setting mj_model alone changes nothing: mjWarp runs off its own device copy. Every switch in
  newton_vec_env pushes; this is the same push, so the bench and training move the same value.
  """
  dev = getattr(solver.mjw_model, field, None)
  if dev is None:
    return "MISSING"
  t = wp.to_torch(dev)
  src = torch.as_tensor(np.asarray(host_array), dtype=t.dtype, device=t.device)
  if src.shape != t.shape:
    src = src.reshape((1,) * (t.dim() - src.dim()) + tuple(src.shape)).expand_as(t)
  t[:] = src
  return tuple(t.shape)


def readback(solver, field, idx):
  """Read one row of a field off the COMPILED device model -- never off the host copy we wrote."""
  dev = getattr(solver.mjw_model, field, None)
  if dev is None:
    return None
  t = wp.to_torch(dev).cpu().numpy()
  while t.ndim > 1 and t.shape[0] == 1:      # mjWarp broadcasts world-invariant fields as (1, n, ...)
    t = t[0]
  return t[idx]


# --------------------------------------------------------------------------------------- scene
def build_env():
  cfg = load_env_cfg(TASK, play=False)
  cfg.scene.num_envs = 1
  agent_cfg = load_rl_cfg(TASK)
  p = Path(A.agent_cfg_from).parent / "params" / "agent.yaml"
  _apply_cfg_mapping(agent_cfg, _yaml.unsafe_load(p.open()))
  w = reward_weights_from_env_yaml(Path(A.reward_cfg))
  apply_reward_weights(cfg, w)
  _s = cfg.actions.get("sonic_action") if isinstance(cfg.actions, dict) else cfg.actions.sonic_action
  _s.tracking_start_assist_gain = 0.0
  _s.tracking_start_assist_steps = 0
  if str(getattr(agent_cfg, "base_tracker_kind", "")).strip().lower() == "astra_onnx":
    from mjlab.tasks.residual_interact.env_cfgs import set_astra_body_dynamics
    set_astra_body_dynamics(cfg)
  env = NewtonVecEnv(cfg, A.xml, num_envs=1, device="cuda:0",
                     sdf_object_stl=A.sdf_object, sdf_resolution=A.sdf_resolution,
                     native_contacts=bool(A.native_contacts),
                     hydro_object_table=False,          # --rigid-object-table
                     table_under_object=True,
                     object_solref="0.004,1.0",         # train_newton's default
                     cuda_graph=False,
                     solver_kwargs=_json.loads(A.solver_kwargs))
  return env


class Rig:
  """Indices, masks and the open-loop command. Built once; every condition reuses it."""

  def __init__(self, env):
    self.env = env
    self.sv = env.solver
    self.m = env.solver.mj_model
    self.d = env.solver.mjw_data
    m = self.m

    self.obj_geoms = [g for g in range(m.ngeom) if "apple" in gname(m, g)]
    self.hand_geoms, self.tip_geoms = [], []
    for g in range(m.ngeom):
      b = bname(m, int(m.geom_bodyid[g]))
      if "robot" in b and ("finger" in b or "palm" in b or "hand" in b):
        self.hand_geoms.append(g)
        if "right_finger" in b and "link4" in b:
          self.tip_geoms.append(g)
    if not self.obj_geoms:
      raise RuntimeError("no object geom; mjlab flattens BODY names, so key on geoms")
    if not self.tip_geoms:
      raise RuntimeError("no right fingertip geom matched")

    self.tip_bodies = sorted({int(m.geom_bodyid[g]) for g in self.tip_geoms})

    # Trap 7: mjlab's compiled actuators are unnamed. Identify them through the joint they drive.
    self.jnt_of_act = [int(m.actuator_trnid[a, 0]) for a in range(m.nu)]
    self.finger_acts, self.other_acts = [], []
    for a in range(m.nu):
      if "right_finger" in jname(m, self.jnt_of_act[a]):
        self.finger_acts.append(a)
      else:
        self.other_acts.append(a)
    # Trap 9: every finger joint carries TWO actuators. The weak one is the scene's own Wuji
    # motor, which mjlab renames but never removes and never writes; left at ctrl 0 it pulls the
    # finger toward zero. Both are driven here, so the command is the whole command.
    self.strong_finger_acts = [a for a in self.finger_acts
                               if float(m.actuator_gainprm[a, 0]) > 10.0]
    self.weak_finger_acts = [a for a in self.finger_acts if a not in self.strong_finger_acts]

    self.jadr = {j: int(m.jnt_qposadr[j]) for j in range(m.njnt)}
    self.obj_jnt = [j for j in range(m.njnt)
                    if "apple" in jname(m, j) and int(m.jnt_type[j]) == 0]
    if not self.obj_jnt:
      raise RuntimeError("object free joint not found")
    self.obj_qadr = self.jadr[self.obj_jnt[0]]
    self.obj_body = int(m.jnt_bodyid[self.obj_jnt[0]])
    self.base_jnt = [j for j in range(m.njnt)
                     if int(m.jnt_type[j]) == 0 and "floating_base" in jname(m, j)]
    self.base_vadr = int(m.jnt_dofadr[self.base_jnt[0]]) if self.base_jnt else None

    self.half = float(np.max(np.abs(self._obj_mesh_extent()))) if True else 0.02

  def _obj_mesh_extent(self):
    m = self.m
    g = self.obj_geoms[0]
    if int(m.geom_type[g]) == 7 and int(m.geom_dataid[g]) >= 0:
      did = int(m.geom_dataid[g])
      va, vn = int(m.mesh_vertadr[did]), int(m.mesh_vertnum[did])
      V = m.mesh_vert[va:va + vn].reshape(-1, 3)
      return 0.5 * (V.max(0) - V.min(0))
    return m.geom_size[g][:3]

  # ---- state access ------------------------------------------------------------------
  def qpos(self):
    return wp.to_torch(self.d.qpos)

  def qvel(self):
    return wp.to_torch(self.d.qvel)

  def ctrl(self):
    return wp.to_torch(self.d.ctrl)

  def xpos(self):
    return wp.to_torch(self.d.xpos)

  def contacts(self):
    """Real hand-object contacts as (geom1, geom2, dist, solref, solimp, includemargin, force).

    Trap 8: the array is padded. Rows with geom1 == geom2 and dist exactly 0 are empty slots.
    Trap 1: efc_force[efc_address] is one PYRAMID EDGE, not the normal force. efc_address is 2-D,
    (ncon, nedges); the normal force is the sum across the second axis.
    """
    c = self.d.contact
    n = int(wp.to_torch(self.d.ncon).cpu().numpy().reshape(-1)[0])
    geom = wp.to_torch(c.geom).cpu().numpy()
    dist = wp.to_torch(c.dist).cpu().numpy()
    solref = wp.to_torch(c.solref).cpu().numpy()
    solimp = wp.to_torch(c.solimp).cpu().numpy()
    incm = wp.to_torch(c.includemargin).cpu().numpy()
    adr = wp.to_torch(c.efc_address).cpu().numpy()
    try:
      efc = wp.to_torch(self.d.efc.force).cpu().numpy().reshape(-1)
    except Exception:
      efc = None
    out = []
    for i in range(min(n, geom.shape[0])):
      g1, g2 = int(geom[i, 0]), int(geom[i, 1])
      if g1 == g2 and float(dist[i]) == 0.0:
        continue                                   # padded slot
      f = float("nan")
      if efc is not None:
        rows = np.atleast_1d(adr[i])
        rows = rows[(rows >= 0) & (rows < efc.size)]
        f = float(efc[rows].sum()) if rows.size else 0.0
      out.append(dict(g1=g1, g2=g2, dist=float(dist[i]), solref=solref[i].copy(),
                      solimp=solimp[i].copy(), incm=float(incm[i]), force=f))
    return out, n

  def hand_object_contacts(self):
    cons, n = self.contacts()
    og, hg = set(self.obj_geoms), set(self.hand_geoms)
    keep = [c for c in cons
            if (c["g1"] in og and c["g2"] in hg) or (c["g2"] in og and c["g1"] in hg)]
    return keep, cons, n

  def overlap_mm(self):
    keep, _, _ = self.hand_object_contacts()
    if not keep:
      return float("nan"), 0, float("nan")
    d = np.array([c["dist"] for c in keep])
    f = np.array([c["force"] for c in keep])
    return -1000.0 * float(d.min()), len(keep), float(np.nansum(f))

  # ---- the command -------------------------------------------------------------------
  def snapshot(self):
    self._q0 = self.qpos().clone()
    self._v0 = self.qvel().clone()
    self._c0 = self.ctrl().clone()

  def restore(self):
    self.qpos()[:] = self._q0
    self.qvel()[:] = self._v0
    self.ctrl()[:] = self._c0

  def place_object(self, offset=(0.0, 0.0, 0.0), clearance=0.0):
    """Put the object at the centroid of the five right fingertips, clear of contact.

    Deterministic and condition-independent: the same three numbers in every run, so the geometry
    the press starts from cannot differ between settings.
    """
    xp = self.xpos().cpu().numpy()[0]
    c = xp[self.tip_bodies].mean(axis=0) + np.asarray(offset, dtype=float)
    q = self.qpos()
    q[0, self.obj_qadr:self.obj_qadr + 3] = torch.tensor(c, dtype=q.dtype, device=q.device)
    q[0, self.obj_qadr + 3:self.obj_qadr + 7] = torch.tensor([1.0, 0.0, 0.0, 0.0],
                                                             dtype=q.dtype, device=q.device)
    v = self.qvel()
    va = int(self.m.jnt_dofadr[self.obj_jnt[0]])
    v[0, va:va + 6] = 0.0
    return c

  def hold_command(self, finger_delta_rad=0.0):
    """Every actuator commanded to hold where it is; the right-hand fingers commanded to close
    `finger_delta_rad` further. Written straight to mjw_data.ctrl, so no action manager, no
    residual, no tracker enters the number."""
    m = self.m
    q = self.qpos().cpu().numpy()[0]
    c = self.ctrl()
    tgt = np.zeros(m.nu, dtype=np.float64)
    for a in range(m.nu):
      j = self.jnt_of_act[a]
      tgt[a] = q[self.jadr[j]]
    for a in self.finger_acts:
      tgt[a] += finger_delta_rad
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    lim = m.actuator_ctrllimited.astype(bool)
    tgt = np.where(lim, np.clip(tgt, lo, hi), tgt)
    c[0, :] = torch.tensor(tgt, dtype=c.dtype, device=c.device)
    self._cmd = tgt.copy()
    return tgt

  def step_settled(self, nsteps, hold, pin_base=True):
    """Step to rest and report the SETTLED overlap, not a transient (trap 3: a frozen pose says
    nothing about whether integrating diverges).

    The base is pinned at the VELOCITY level, not by writing qpos: writing the pose of a jointed
    robot every step fights the integrator and produced `Nan, Inf or huge value in QACC` within
    80 ms on this very scene. Zeroing the six base dofs is equivalent to infinite damping and
    leaves the integrator's own solution alone.
    """
    hist = []
    for i in range(nsteps):
      if pin_base and self.base_vadr is not None:
        self.qvel()[0, self.base_vadr:self.base_vadr + 6] = 0.0
      self.ctrl()[0, :] = torch.tensor(self._cmd, dtype=self.ctrl().dtype,
                                       device=self.ctrl().device)
      self.env._physics_step()
      self.env.state_in, self.env.state_out = self.env.state_out, self.env.state_in
      if i >= nsteps - hold:
        hist.append(self.overlap_mm()[0])
    ov, n, f = self.overlap_mm()
    h = np.array([x for x in hist if np.isfinite(x)])
    drift = float(h.max() - h.min()) if h.size >= 2 else float("nan")
    q = self.qpos().cpu().numpy()
    ok = bool(np.isfinite(q).all())
    return dict(overlap_mm=ov, ncon=n, force_N=f, drift_mm=drift, finite=ok)


# --------------------------------------------------------------------------------------- switches
def apply_setting(rig, name):
  """Apply one condition to the built model IN PLACE, then read every value back off the compiled
  device model. Nothing is trusted because it printed."""
  m, sv = rig.m, rig.sv
  notes = []

  def set_geom(field, geoms, vals):
    arr = getattr(m, field)
    v = np.atleast_1d(np.asarray(vals, dtype=arr.dtype if arr.dtype != np.int32 else np.int32))
    for g in geoms:
      if arr.ndim == 1:
        arr[g] = v[0]
      else:
        arr[g][:v.size] = v
    push(sv, field, arr)
    got = readback(sv, field, geoms[0])
    notes.append(f"{field}[{gname(m, geoms[0])}] -> {np.atleast_1d(got)[:v.size]}")

  def set_force(acts, lim):
    m.actuator_forcerange[acts, 0] = -abs(lim)
    m.actuator_forcerange[acts, 1] = abs(lim)
    m.actuator_forcelimited[acts] = 1
    push(sv, "actuator_forcerange", m.actuator_forcerange)
    push(sv, "actuator_forcelimited", m.actuator_forcelimited)
    got = readback(sv, "actuator_forcerange", acts[0])
    notes.append(f"actuator_forcerange[{jname(m, rig.jnt_of_act[acts[0]])}] -> {got}")

  if name == "baseline":
    pass
  elif name.startswith("objsolref="):
    v = [float(x) for x in name.split("=", 1)[1].split(",")]
    set_geom("geom_solref", rig.obj_geoms, v)
  elif name.startswith("handsolref="):
    v = [float(x) for x in name.split("=", 1)[1].split(",")]
    set_geom("geom_solref", rig.hand_geoms, v)
  elif name.startswith("bothsolref="):
    v = [float(x) for x in name.split("=", 1)[1].split(",")]
    set_geom("geom_solref", rig.obj_geoms + rig.hand_geoms, v)
  elif name.startswith("objsolimp="):
    v = [float(x) for x in name.split("=", 1)[1].split(",")]
    set_geom("geom_solimp", rig.obj_geoms, v)
  elif name.startswith("priority="):
    set_geom("geom_priority", rig.obj_geoms, [int(name.split("=", 1)[1])])
  elif name.startswith("ffl="):
    set_force(rig.finger_acts, float(name.split("=", 1)[1]))
  elif name.startswith("timestep="):
    dt = float(name.split("=", 1)[1])
    m.opt.timestep = dt
    try:
      sv.mjw_model.opt.timestep = dt
    except Exception:
      pass
    rig.env.physics_dt = dt
    notes.append(f"opt.timestep -> host {m.opt.timestep} "
                 f"device {getattr(sv.mjw_model.opt, 'timestep', 'MISSING')} "
                 f"newton steps with {rig.env.physics_dt}")
  else:
    raise SystemExit(f"unknown setting {name!r}")
  return notes


# --------------------------------------------------------------------------------------- facts
def dump_facts(rig):
  m, sv, env = rig.m, rig.sv, rig.env
  print("\n=== timestep and options (read off the COMPILED model, not the config) ===")
  dev_dt = getattr(sv.mjw_model.opt, "timestep", None)
  dev_dt = float(wp.to_torch(dev_dt).cpu().numpy().reshape(-1)[0]) if hasattr(dev_dt, "dtype") \
      else dev_dt
  print(f"  cfg physics_dt (what Newton integrates)  {env.physics_dt}")
  print(f"  mj_model.opt.timestep  (host)            {float(m.opt.timestep)}")
  print(f"  mjw_model.opt.timestep (device)          {dev_dt}")
  print(f"  decimation {env.decimation}  control rate {1.0/(env.physics_dt*env.decimation):.1f} Hz")
  print(f"  cone {int(m.opt.cone)}  impratio {float(m.opt.impratio)}  "
        f"iterations {int(m.opt.iterations)}  ls_iterations {int(m.opt.ls_iterations)}  "
        f"integrator {int(m.opt.integrator)}")
  refsafe = int(m.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)
  clamp = 2.0 * float(m.opt.timestep)
  print(f"  disableflags {int(m.opt.disableflags)}  REFSAFE disabled: {bool(refsafe)}")
  print(f"  -> solref timeconst is clamped up to 2*opt.timestep = {clamp*1000:.2f} ms "
        f"unless REFSAFE is disabled")
  if abs(float(m.opt.timestep) - float(env.physics_dt)) > 1e-9:
    print(f"  !! MISMATCH: the clamp uses opt.timestep ({1000*float(m.opt.timestep):.2f} ms) but "
          f"Newton integrates {1000*float(env.physics_dt):.2f} ms")

  print("\n=== object collider ===")
  for g in rig.obj_geoms:
    print(f"  {gname(m,g):40s} type={GEOM_TYPE.get(int(m.geom_type[g]),'?'):6s} "
          f"solref={m.geom_solref[g][:2]} solimp={m.geom_solimp[g][:5]} "
          f"prio={int(m.geom_priority[g])} solmix={float(m.geom_solmix[g]):.3f} "
          f"margin={float(m.geom_margin[g]):.4f} gap={float(m.geom_gap[g]):.4f} "
          f"condim={int(m.geom_condim[g])} fric={m.geom_friction[g][:3]} "
          f"con/aff={int(m.geom_contype[g])}/{int(m.geom_conaffinity[g])}")
    print(f"    device readback  solref={readback(sv,'geom_solref',g)} "
          f"solimp={readback(sv,'geom_solimp',g)} prio={readback(sv,'geom_priority',g)}")
  print(f"  object half-extent from the compiled mesh: "
        f"{np.round(1000*rig._obj_mesh_extent(),2)} mm")
  print(f"  object body mass {float(m.body_mass[rig.obj_body]):.4f} kg")

  print("\n=== right fingertip colliders (5 of "
        f"{len(rig.hand_geoms)} hand geoms) ===")
  for g in rig.tip_geoms[:5]:
    print(f"  {gname(m,g)[-46:]:46s} type={GEOM_TYPE.get(int(m.geom_type[g]),'?'):5s} "
          f"solref={m.geom_solref[g][:2]} solimp={m.geom_solimp[g][:3]} "
          f"prio={int(m.geom_priority[g])} margin={float(m.geom_margin[g]):.4f} "
          f"gap={float(m.geom_gap[g]):.4f} condim={int(m.geom_condim[g])}")
    print(f"    device readback  solref={readback(sv,'geom_solref',g)} "
          f"prio={readback(sv,'geom_priority',g)}")

  og, tg = rig.obj_geoms[0], rig.tip_geoms[0]
  p1, p2 = int(m.geom_priority[og]), int(m.geom_priority[tg])
  s1, s2 = m.geom_solref[og][:2], m.geom_solref[tg][:2]
  x1, x2 = float(m.geom_solmix[og]), float(m.geom_solmix[tg])
  mix = 1.0 if p1 > p2 else (0.0 if p2 > p1 else x1 / max(x1 + x2, 1e-15))
  eff = mix * np.asarray(s1) + (1 - mix) * np.asarray(s2)
  print(f"\n  PREDICTED pair solref for object<->fingertip: mix={mix:.3f} -> {eff} "
        f"(clamped by REFSAFE to >= {clamp:.4f})")

  print("\n=== finger actuators (identified through actuator_trnid; mjlab's are unnamed) ===")
  by_joint = {}
  for a in rig.finger_acts:
    by_joint.setdefault(jname(m, rig.jnt_of_act[a]), []).append(a)
  k = sorted(by_joint)[0] if by_joint else None
  print(f"  {len(rig.finger_acts)} actuators on {len(by_joint)} right-finger joints "
        f"({len(rig.finger_acts)/max(len(by_joint),1):.1f} per joint)")
  if k:
    for a in by_joint[k]:
      print(f"  {k:38s} act{a:4d} gain={float(m.actuator_gainprm[a,0]):9.4f} "
            f"bias={m.actuator_biasprm[a][:3]} "
            f"forcerange={m.actuator_forcerange[a]} limited={int(m.actuator_forcelimited[a])}")
      print(f"      device readback forcerange={readback(sv,'actuator_forcerange',a)} "
            f"gainprm={np.atleast_1d(readback(sv,'actuator_gainprm',a))[:1]}")
  print(f"  strong (mjlab) {len(rig.strong_finger_acts)}   weak (xml_motor_unused) "
        f"{len(rig.weak_finger_acts)}"
        + ("   <-- trap 9: the weak set pulls the fingers toward ctrl=0 unless written"
           if rig.weak_finger_acts else ""))

  print("\n=== Newton-side contact override ===")
  ct = env.contacts
  if ct is None:
    print("  MuJoCo's own narrow phase (no Newton contacts)")
  else:
    for f in ("rigid_contact_stiffness", "rigid_contact_damping", "rigid_contact_friction"):
      arr = getattr(ct, f, None)
      if arr is None:
        print(f"  {f}: absent -> geom solref/solimp is what the pair uses")
      else:
        v = wp.to_torch(arr).cpu().numpy()
        nz = int((v != 0).sum())
        print(f"  {f}: present, {nz}/{v.size} nonzero, max {float(np.nanmax(v)) if v.size else 0}"
              + ("   <-- NONZERO OVERRIDES geom solref entirely" if nz else ""))
    mode = getattr(getattr(env.nmodel, "mujoco", None), "solref_mode", None)
    if mode is not None:
      v = wp.to_torch(mode).cpu().numpy()
      print(f"  shape solref_mode: unique {np.unique(v).tolist()} "
            f"(2 = FORCE_SPACE, which recomputes solref from shape ke/kd)")


def dump_live_contacts(rig, limit=12):
  m = rig.m
  keep, cons, n = rig.hand_object_contacts()
  print(f"\n  contact array: ncon={n}, real rows {len(cons)}, hand<->object {len(keep)}")
  for c in keep[:limit]:
    print(f"    {gname(m,c['g1'])[-34:]:34s} <-> {gname(m,c['g2'])[-24:]:24s} "
          f"dist={1000*c['dist']:+8.3f} mm  solref={np.round(c['solref'],5)} "
          f"solimp={np.round(c['solimp'],3)} incm={1000*c['incm']:+.3f} mm  "
          f"Fn={c['force']:9.3f} N")


# --------------------------------------------------------------------------------------- main
def main():
  print(f"building one Newton world: {A.xml}\n  object {A.sdf_object}", flush=True)
  env = build_env()
  rig = Rig(env)
  env.reset()
  rig.snapshot()

  if A.mode == "facts":
    dump_facts(rig)
    off = tuple(float(x) for x in A.object_offset.split(","))
    c = rig.place_object(off)
    rig.hold_command(0.0)
    print(f"\n=== after 200 substeps holding the reset pose, object at fingertip centroid "
          f"{np.round(c,4)} ===")
    r = rig.step_settled(200, 50)
    print(f"  {r}")
    dump_live_contacts(rig)
    return

  if A.mode == "pose":
    xp = rig.xpos().cpu().numpy()[0]
    c = xp[rig.tip_bodies].mean(axis=0)
    print(f"\nfingertip bodies {[bname(rig.m,b)[-28:] for b in rig.tip_bodies]}")
    print(f"positions\n{np.round(xp[rig.tip_bodies],4)}")
    print(f"centroid {np.round(c,4)}   object now at "
          f"{np.round(rig.qpos().cpu().numpy()[0][rig.obj_qadr:rig.obj_qadr+3],4)}")
    print(f"spread (max pairwise) "
          f"{np.max(np.linalg.norm(xp[rig.tip_bodies][:,None]-xp[rig.tip_bodies][None],axis=-1)):.4f} m")
    print(f"object half-extent {np.round(1000*rig._obj_mesh_extent(),2)} mm")
    return

  settings = [s.strip() for s in A.settings.split(",") if s.strip()]
  off = tuple(float(x) for x in A.object_offset.split(","))
  arm = 0.02   # right_finger*_link4 contact point to its joint, measured

  rows = []
  if A.mode == "position":
    sweep = [float(x) / 1000.0 for x in A.depths.split(",")]
    hdr = "commanded depth (mm)"
  else:
    sweep = [float(x) for x in A.forces.split(",")]
    hdr = "finger torque cap (N*m)"

  print(f"\n{'setting':28s}{hdr:>24s}")
  for name in settings:
    notes = apply_setting(rig, name)
    for t in notes:
      print(f"  [readback] {t}")
    cells = []
    for v in sweep:
      rig.restore()
      rig.place_object(off)
      if A.mode == "position":
        rig.hold_command(v / arm)          # a tip depth of v needs v/arm rad of extra flexion
      else:
        apply_setting(rig, f"ffl={v}")
        rig.hold_command(0.35)             # a fixed, generous closure; the CAP is the variable
      r = rig.step_settled(A.settle, A.hold)
      cells.append(r)
      rows.append(dict(setting=name, sweep=v, **r))
      print(f"    {name:24s} {v:8.4f}  overlap {r['overlap_mm']:8.3f} mm  "
            f"ncon {r['ncon']:4d}  F {r['force_N']:9.2f} N  drift {r['drift_mm']:.4f} mm  "
            f"finite {r['finite']}", flush=True)

  if A.out:
    import csv
    with open(A.out, "w", newline="") as fh:
      w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
      w.writeheader()
      w.writerows(rows)
    print(f"\nwrote {A.out}")


main()
