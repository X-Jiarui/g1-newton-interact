"""A scripted press bench for the hand-object penetration problem. No policy, no reward, no RSI.

Why this exists: every comparison of two ROLLOUTS under different contact settings is confounded.
A setting that makes the hand weaker also changes where the policy puts its hand, so a rollout whose
grasp got worse stands further off, presses more lightly, and scores a SHALLOWER overlap for
entirely the wrong reason. Three verdicts have already been retracted for exactly that.

So nothing here is left to a policy. The scene is the training scene -- same MJCF, the same object
STL collided through the same 128^3 SDF, the same solver kwargs and the same contact recipe as
R30_CUBE and R31_CUBE -- built at one world. The robot is left where reset puts it, the hand is
opened to its joint limits, the cube is placed at the point the CLOSED hand's fingertips converge
on, and the fingers are then driven shut by an open-loop command written straight into Newton's
Control. The command is a literal constant, so any difference in the settled overlap is the physics
and nothing else.

Commanded depth is not asserted, it is MEASURED: the same finger command is first run with the cube
parked 5 m away, and the depth to which the free-flying fingertips would have entered the cube is
computed from the collider hulls. That is the "commanded" column. The "settled" column is the same
command with the cube present. A physically correct simulator flattens -- commanding deeper stops
producing deeper.

Two control modes, because they fail differently:

  * POSITION: sweep the commanded depth. Whatever settled depth survives is the wall failing.
  * FORCE: cap the finger actuator torque so the tip force is bounded, sweep the cap, and read the
    contact's own normal force. Gives overlap versus force in mm/N with no confound at all.

Nothing is trusted because it printed. Every switch is written to mj_model, pushed to mjw_model and
then read back off the compiled device model; the command is checked against mjw_data.ctrl after the
first step. Two full A/B training runs were wasted on switches that logged a new value and simulated
the old one.

    python tools/probes/ik_press_bench.py --mode facts
    python tools/probes/ik_press_bench.py --mode position --settings baseline,priority=1
    python tools/probes/ik_press_bench.py --mode force --settings baseline
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
ap.add_argument("--mode", default="facts", choices=("facts", "position", "force"))
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
ap.add_argument("--settle", type=int, default=300, help="substeps to settle before reading")
ap.add_argument("--hold", type=int, default=60, help="substeps over which rest is checked")
ap.add_argument("--closures", default="0.10,0.25,0.45,0.70,1.00,1.40",
                help="extra finger flexion beyond the open pose, rad (position mode)")
ap.add_argument("--forces", default="0.15,0.62,2,10,30",
                help="finger actuator torque caps, N*m (force mode)")
ap.add_argument("--force-closure", type=float, default=1.4,
                help="the fixed, generous closure used in force mode; the CAP is the variable")
ap.add_argument("--settings", default="baseline")
ap.add_argument("--out", default=None, help="write the rows as csv here as well")
A = ap.parse_args()

# Read straight off R30_CUBE's own /proc/<pid>/environ, so the bench scene is the trained scene.
# APPLE_HAND_KIND is not optional: the default 'xhand' rejects this clip's 69 dof columns outright,
# and a wrong hand kind would silently bench a different robot.
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
GEOM_TYPE = {0: "plane", 1: "hfield", 2: "sphere", 3: "capsule", 4: "ellipsoid",
             5: "cylinder", 6: "box", 7: "mesh", 8: "sdf"}


def gname(m, g):
  return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f"<geom{g}>"


def jname(m, j):
  return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(j)) or f"<jnt{j}>"


def bname(m, b):
  return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b)) or f"<body{b}>"


def sq(a):
  """mjWarp lays model and data arrays out per world; at one world the leading axis is a singleton."""
  v = wp.to_torch(a).cpu().numpy() if not isinstance(a, np.ndarray) else a
  while v.ndim > 1 and v.shape[0] == 1:
    v = v[0]
  return v


def push(solver, field, host_array):
  """Copy a host mj_model field onto the compiled mjw_model. Setting mj_model alone changes
  nothing: mjWarp runs off its own device copy."""
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
  return sq(dev)[idx]


def build_env():
  cfg = load_env_cfg(TASK, play=False)
  cfg.scene.num_envs = 1
  agent_cfg = load_rl_cfg(TASK)
  p = Path(A.agent_cfg_from).parent / "params" / "agent.yaml"
  _apply_cfg_mapping(agent_cfg, _yaml.unsafe_load(p.open()))
  apply_reward_weights(cfg, reward_weights_from_env_yaml(Path(A.reward_cfg)))
  _s = cfg.actions.get("sonic_action") if isinstance(cfg.actions, dict) else cfg.actions.sonic_action
  _s.tracking_start_assist_gain = 0.0
  _s.tracking_start_assist_steps = 0
  if str(getattr(agent_cfg, "base_tracker_kind", "")).strip().lower() == "astra_onnx":
    from mjlab.tasks.residual_interact.env_cfgs import set_astra_body_dynamics
    set_astra_body_dynamics(cfg)
  return NewtonVecEnv(cfg, A.xml, num_envs=1, device="cuda:0",
                      sdf_object_stl=A.sdf_object, sdf_resolution=A.sdf_resolution,
                      native_contacts=bool(A.native_contacts),
                      hydro_object_table=False,          # --rigid-object-table
                      table_under_object=True,
                      object_solref="0.004,1.0",         # train_newton's own default
                      cuda_graph=False,
                      solver_kwargs=_json.loads(A.solver_kwargs))


class Rig:
  def __init__(self, env):
    self.env, self.sv = env, env.solver
    self.m, self.d = env.solver.mj_model, env.solver.mjw_data
    m = self.m

    self.obj_geoms = [g for g in range(m.ngeom) if "apple" in gname(m, g)]
    self.hand_geoms, self.tip_geoms = [], []
    for g in range(m.ngeom):
      b = bname(m, int(m.geom_bodyid[g]))
      if "robot" in b and ("finger" in b or "palm" in b or "hand" in b):
        self.hand_geoms.append(g)
        if "right_finger" in b and "link4" in b:
          self.tip_geoms.append(g)
    if not self.obj_geoms or not self.tip_geoms:
      raise RuntimeError("mjlab flattens BODY names into one path string; key on GEOM names")
    self.tip_bodies = sorted({int(m.geom_bodyid[g]) for g in self.tip_geoms})

    # Trap 7: mjlab's compiled actuators are unnamed. Identify them through the joint they drive.
    self.jnt_of_act = [int(m.actuator_trnid[a, 0]) for a in range(m.nu)]
    self.finger_acts = [a for a in range(m.nu)
                        if "right_finger" in jname(m, self.jnt_of_act[a])]
    # Trap 9: every finger joint carries TWO actuators. The weak one is the scene's own Wuji motor,
    # which mjlab renames `xml_motor_unused_*` but never removes and never writes; left at ctrl 0 it
    # pulls the finger toward zero with up to 0.577 N*m. Both are commanded here, so the finger
    # torque budget is exactly what the sweep says it is.
    self.strong = [a for a in self.finger_acts if float(m.actuator_gainprm[a, 0]) > 10.0]
    self.weak = [a for a in self.finger_acts if a not in self.strong]

    self.jadr = {j: int(m.jnt_qposadr[j]) for j in range(m.njnt)}
    self.obj_jnt = [j for j in range(m.njnt)
                    if "apple" in jname(m, j) and int(m.jnt_type[j]) == 0][0]
    self.obj_qadr = self.jadr[self.obj_jnt]
    self.obj_vadr = int(m.jnt_dofadr[self.obj_jnt])
    self.obj_body = int(m.jnt_bodyid[self.obj_jnt])
    base = [j for j in range(m.njnt)
            if int(m.jnt_type[j]) == 0 and "floating_base" in jname(m, j)]
    self.base_vadr = int(m.jnt_dofadr[base[0]]) if base else None
    self._hv = None
    self.h = self.obj_half()

  # ---------------------------------------------------------------- state
  def qpos(self):
    return wp.to_torch(self.d.qpos)

  def qvel(self):
    return wp.to_torch(self.d.qvel)

  def ctrl(self):
    """mjw_data.ctrl -- READ ONLY. SolverMuJoCo calls _apply_mjc_control at the top of every step,
    so a direct write here is overwritten before it is ever used. The first version of this bench
    wrote here, the fingers never moved, and every condition read zero hand-object contacts."""
    return wp.to_torch(self.d.ctrl)

  def cmd(self):
    """Newton's Control object -- what the solver actually reads."""
    return wp.to_torch(self.env.control.mujoco.ctrl).view(self.env.num_envs, -1)

  def obj_half(self):
    m, g = self.m, self.obj_geoms[0]
    if int(m.geom_type[g]) == 7 and int(m.geom_dataid[g]) >= 0:
      did = int(m.geom_dataid[g])
      va, vn = int(m.mesh_vertadr[did]), int(m.mesh_vertnum[did])
      V = m.mesh_vert[va:va + vn].reshape(-1, 3)
      return 0.5 * (V.max(0) - V.min(0))
    return np.asarray(m.geom_size[g][:3])

  def hull_verts(self):
    if self._hv is None:
      m, out = self.m, []
      for g in self.hand_geoms:
        if int(m.geom_type[g]) != 7 or int(m.geom_dataid[g]) < 0:
          continue
        did = int(m.geom_dataid[g])
        va, vn = int(m.mesh_vertadr[did]), int(m.mesh_vertnum[did])
        out.append((g, m.mesh_vert[va:va + vn].reshape(-1, 3).astype(np.float64)))
      self._hv = out
    return self._hv

  def hand_world_verts(self):
    gx, gm = sq(self.d.geom_xpos), sq(self.d.geom_xmat)
    return {g: hv @ gm[g].reshape(3, 3).T + gx[g] for g, hv in self.hull_verts()}

  def depth_into_cube_mm(self, centre):
    """How far the hand's collider surface lies inside a cube at `centre`, with NO physics at all.

    The cube is axis-aligned (the bench sets its quaternion to identity) and convex, so testing a
    hull vertex against its six faces is exact. This is the ground truth `contact.dist` is checked
    against -- an SDF collider's reported distance has been suspected of not being the geometric
    interpenetration it looks like.
    """
    best, who = 0.0, None
    for g, P in self.hand_world_verts().items():
      d = self.h[None, :] - np.abs(P - np.asarray(centre)[None, :])
      k = float(d.min(axis=1).max()) if d.size else -1.0
      if k > best:
        best, who = k, g
    return 1000.0 * best, who

  # ---------------------------------------------------------------- contacts
  def contact_rows(self):
    c = self.d.contact
    cnt = getattr(self.d, "nacon", None) or getattr(self.d, "ncon")
    n = int(sq(cnt).reshape(-1)[0])
    geom, dist = sq(c.geom), sq(c.dist)
    solref, solimp = sq(c.solref), sq(c.solimp)
    incm, adr = sq(c.includemargin), sq(c.efc_address)
    try:
      efc = sq(self.d.efc.force).reshape(-1)
    except Exception:
      efc = None
    out = []
    for i in range(min(n, geom.shape[0])):
      g1, g2 = int(geom[i, 0]), int(geom[i, 1])
      if g1 == g2 and float(dist[i]) == 0.0:
        continue                       # trap 8: padded slot, not a contact
      f = float("nan")
      if efc is not None:
        # Trap 1: efc_force[efc_address] is ONE PYRAMID EDGE, not the normal force. efc_address is
        # 2-D, (ncon, nedges); the normal force is the sum across the second axis. Reading a single
        # edge produced a fictitious 654 N where the truth was 15.8 N.
        rows = np.atleast_1d(adr[i])
        rows = rows[(rows >= 0) & (rows < efc.size)]
        f = float(efc[rows].sum()) if rows.size else 0.0
      out.append(dict(g1=g1, g2=g2, dist=float(dist[i]), solref=np.asarray(solref[i]).copy(),
                      solimp=np.asarray(solimp[i]).copy(), incm=float(incm[i]), force=f))
    return out, n

  def hand_object(self):
    rows, n = self.contact_rows()
    og, hg = set(self.obj_geoms), set(self.hand_geoms)
    keep = [c for c in rows
            if (c["g1"] in og and c["g2"] in hg) or (c["g2"] in og and c["g1"] in hg)]
    return keep, rows, n

  # ---------------------------------------------------------------- command
  def snapshot(self):
    self._q0, self._v0, self._c0 = self.qpos().clone(), self.qvel().clone(), self.cmd().clone()

  def restore(self):
    self.qpos()[:] = self._q0
    self.qvel()[:] = self._v0
    self.cmd()[:] = self._c0

  def target_from_qpos(self, extra=None):
    m = self.m
    q = sq(self.qpos())
    tgt = np.array([q[self.jadr[self.jnt_of_act[a]]] for a in range(m.nu)], dtype=np.float64)
    if extra is not None:
      tgt[self.finger_acts] = extra
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    lim = m.actuator_ctrllimited.astype(bool)
    return np.where(lim, np.clip(tgt, lo, hi), tgt)

  def set_fingers(self, absolute_rad):
    """Every actuator holds where it is; the right-hand fingers go to `absolute_rad`."""
    self._cmd = self.target_from_qpos(extra=absolute_rad)
    return self._cmd

  def run(self, nsteps, hold, park_object=False, measure_centre=None):
    m_nu = self.m.nu
    if park_object:
      self.qpos()[0, self.obj_qadr + 2] += 5.0
      self.qvel()[0, self.obj_vadr:self.obj_vadr + 6] = 0.0
    hist = []
    for i in range(nsteps):
      if self.base_vadr is not None:
        # Velocity-level pin. Writing the POSE of a jointed robot every step fights the integrator
        # and produced `Nan, Inf or huge value in QACC` within 80 ms on this very scene; zeroing the
        # six base dofs is equivalent to infinite damping and leaves the integrator alone.
        self.qvel()[0, self.base_vadr:self.base_vadr + 6] = 0.0
      self.cmd()[0, :] = torch.tensor(self._cmd, dtype=self.cmd().dtype, device=self.cmd().device)
      self.env._physics_step()
      self.env.state_in, self.env.state_out = self.env.state_out, self.env.state_in
      if i == 0:
        got = sq(self.ctrl()).reshape(-1)[:m_nu]
        err = float(np.max(np.abs(got - self._cmd)))
        if err > 1e-4:
          raise RuntimeError(f"the command never reached mjw_data.ctrl (max error {err:.6f}); "
                             f"writing mjw_data.ctrl directly does not work")
      if i >= nsteps - hold:
        hist.append(self.overlap_mm()[0] if measure_centre is None
                    else self.depth_into_cube_mm(measure_centre)[0])
    h = np.array([x for x in hist if np.isfinite(x)])
    return float(h.max() - h.min()) if h.size >= 2 else float("nan")

  def overlap_mm(self):
    keep, _, _ = self.hand_object()
    if not keep:
      return float("nan"), 0, 0.0, float("nan")
    d = np.array([c["dist"] for c in keep])
    f = np.array([c["force"] for c in keep])
    return (-1000.0 * float(d.min()), len(keep), float(np.nansum(f)),
            float(keep[int(np.argmin(d))]["solref"][0]))

  # ---------------------------------------------------------------- setup
  def calibrate(self, settle=250):
    """Two poses, both derived from the scene rather than guessed, both identical in every
    condition: where the CLOSED fingertips converge (the press target), and the OPEN hand the press
    starts from. The cube starts at the target with the hand open, so it is never initialised inside
    the hand -- doing that ejected it at ~0.5 m/s and left every condition with zero contacts."""
    m = self.m
    hi = np.array([m.actuator_ctrlrange[a, 1] for a in self.finger_acts])
    lo = np.array([m.actuator_ctrlrange[a, 0] for a in self.finger_acts])

    self.restore()
    self.set_fingers(hi)
    self.run(settle, 0, park_object=True)
    xp = sq(self.d.xpos)
    self.target = xp[self.tip_bodies].mean(axis=0)
    self.closed_spread = float(np.max(np.linalg.norm(
        xp[self.tip_bodies][:, None] - xp[self.tip_bodies][None], axis=-1)))
    self.closed_cmd = self.target_from_qpos(extra=hi)[self.finger_acts]

    self.restore()
    self.set_fingers(lo)
    self.run(settle, 0, park_object=True)
    self.open_cmd = self.target_from_qpos(extra=lo)[self.finger_acts]
    self.open_clear, _ = self.depth_into_cube_mm(self.target)
    # The open hand IS the press start: snapshot it so every condition begins identically.
    self.snapshot()
    return self.target, self.closed_spread, self.open_clear

  def gravcomp_object(self, on=True):
    """Cancel the cube's own weight. A 0.364 kg cube in mid-air falls out of the hand before the
    contact settles; removing 3.6 N is small beside the press forces and identical in every
    condition, so it cannot move a comparison."""
    m = self.m
    if not hasattr(m, "body_gravcomp"):
      return None
    m.body_gravcomp[self.obj_body] = 1.0 if on else 0.0
    push(self.sv, "body_gravcomp", m.body_gravcomp)
    return float(np.atleast_1d(readback(self.sv, "body_gravcomp", self.obj_body))[0])

  def place_object(self):
    q, v = self.qpos(), self.qvel()
    q[0, self.obj_qadr:self.obj_qadr + 3] = torch.tensor(self.target, dtype=q.dtype, device=q.device)
    q[0, self.obj_qadr + 3:self.obj_qadr + 7] = torch.tensor([1.0, 0.0, 0.0, 0.0],
                                                             dtype=q.dtype, device=q.device)
    v[0, self.obj_vadr:self.obj_vadr + 6] = 0.0

  # ---------------------------------------------------------------- one press
  def press(self, closure_frac, park=False):
    """One press at a commanded closure. `park=True` runs the SAME command with the cube 5 m away,
    which is how the commanded depth is measured rather than asserted."""
    self.restore()
    if not park:
      self.place_object()
    tgt = self.open_cmd + closure_frac * (self.closed_cmd - self.open_cmd)
    self.set_fingers(tgt)
    drift = self.run(A.settle, A.hold, park_object=park,
                     measure_centre=(self.target if park else None))
    if park:
      d, who = self.depth_into_cube_mm(self.target)
      return dict(commanded_mm=d, drift_mm=drift, deepest_geom=who)
    ov, n, f, tc = self.overlap_mm()
    geo, _ = self.depth_into_cube_mm(sq(self.d.geom_xpos)[self.obj_geoms[0]])
    moved = 1000.0 * float(np.linalg.norm(
        sq(self.qpos())[self.obj_qadr:self.obj_qadr + 3] - self.target))
    return dict(settled_mm=ov, geom_mm=geo, ncon=n, force_N=f, pair_timeconst=tc,
                drift_mm=drift, object_moved_mm=moved,
                finite=bool(np.isfinite(sq(self.qpos())).all()))


# ---------------------------------------------------------------------------------- switches
def apply_setting(rig, name):
  m, sv = rig.m, rig.sv
  notes = []

  def set_geom(field, geoms, vals):
    arr = getattr(m, field)
    v = np.atleast_1d(np.asarray(vals))
    for g in geoms:
      if arr.ndim == 1:
        arr[g] = v[0]
      else:
        arr[g][:v.size] = v
    push(sv, field, arr)
    notes.append(f"{field}[{gname(m, geoms[0])[-28:]}] device -> "
                 f"{np.atleast_1d(readback(sv, field, geoms[0]))[:max(v.size,1)]}")

  if name == "baseline":
    notes.append("as R30_CUBE runs it")
  elif name.startswith("objsolref="):
    set_geom("geom_solref", rig.obj_geoms, [float(x) for x in name.split("=", 1)[1].split(",")])
  elif name.startswith("handsolref="):
    set_geom("geom_solref", rig.hand_geoms, [float(x) for x in name.split("=", 1)[1].split(",")])
  elif name.startswith("bothsolref="):
    set_geom("geom_solref", rig.obj_geoms + rig.hand_geoms,
             [float(x) for x in name.split("=", 1)[1].split(",")])
  elif name.startswith("objsolimp="):
    set_geom("geom_solimp", rig.obj_geoms, [float(x) for x in name.split("=", 1)[1].split(",")])
  elif name.startswith("bothsolimp="):
    set_geom("geom_solimp", rig.obj_geoms + rig.hand_geoms,
             [float(x) for x in name.split("=", 1)[1].split(",")])
  elif name.startswith("priority="):
    set_geom("geom_priority", rig.obj_geoms, [int(name.split("=", 1)[1])])
  elif name.startswith("impratio="):
    m.opt.impratio = float(name.split("=", 1)[1])
    try:
      sv.mjw_model.opt.impratio.fill_(float(name.split("=", 1)[1]))
      got = float(sq(sv.mjw_model.opt.impratio).reshape(-1)[0])
    except Exception:
      got = "MISSING"
    notes.append(f"opt.impratio device -> {got}")
  elif name.startswith("ffl="):
    lim = float(name.split("=", 1)[1])
    m.actuator_forcerange[rig.finger_acts, 0] = -lim
    m.actuator_forcerange[rig.finger_acts, 1] = lim
    m.actuator_forcelimited[rig.finger_acts] = 1
    push(sv, "actuator_forcerange", m.actuator_forcerange)
    push(sv, "actuator_forcelimited", m.actuator_forcelimited)
    notes.append(f"actuator_forcerange device -> "
                 f"{readback(sv, 'actuator_forcerange', rig.finger_acts[0])} "
                 f"on {len(rig.finger_acts)} finger actuators")
  else:
    raise SystemExit(f"unknown setting {name!r}")
  return notes


def reset_settings(rig):
  """Put every swept field back to the value the model compiled with, so conditions do not stack."""
  m, sv = rig.m, rig.sv
  for f, v in rig._pristine.items():
    getattr(m, f)[:] = v
    push(sv, f, getattr(m, f))
  m.opt.impratio = rig._pristine_impratio
  try:
    sv.mjw_model.opt.impratio.fill_(rig._pristine_impratio)
  except Exception:
    pass


# ---------------------------------------------------------------------------------- facts
def dump_facts(rig):
  m, sv, env = rig.m, rig.sv, rig.env
  print("\n=== timestep and options, read off the COMPILED model ===")
  print(f"  cfg physics_dt (what Newton integrates)  {env.physics_dt}")
  print(f"  mj_model.opt.timestep before any step    {float(m.opt.timestep)}")
  env._physics_step()
  env.state_in, env.state_out = env.state_out, env.state_in
  dev = float(sq(sv.mjw_model.opt.timestep).reshape(-1)[0])
  print(f"  mjw_model.opt.timestep after ONE step    {dev}   "
        f"(SolverMuJoCo.step does mjw_model.opt.timestep.fill_(dt))")
  print(f"  decimation {env.decimation}, control rate "
        f"{1.0/(env.physics_dt*env.decimation):.1f} Hz")
  print(f"  cone {int(m.opt.cone)} (0=pyramidal)  impratio {float(m.opt.impratio)}  "
        f"iterations {int(m.opt.iterations)}  ls_iterations {int(m.opt.ls_iterations)}  "
        f"integrator {int(m.opt.integrator)}")
  refsafe = int(m.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)
  print(f"  REFSAFE disabled: {bool(refsafe)}  ->  solref timeconst is clamped UP to "
        f"2*dt = {2000*dev:.2f} ms. Nothing shorter can be asked for.")

  print("\n=== object collider ===")
  for g in rig.obj_geoms:
    print(f"  {gname(m,g)[-40:]:40s} {GEOM_TYPE.get(int(m.geom_type[g]),'?'):5s} "
          f"solref={np.round(readback(sv,'geom_solref',g),5)} "
          f"solimp={np.round(readback(sv,'geom_solimp',g),4)} "
          f"prio={readback(sv,'geom_priority',g)} solmix={float(m.geom_solmix[g]):.2f} "
          f"margin={float(m.geom_margin[g]):.4f} gap={float(m.geom_gap[g]):.4f} "
          f"condim={int(m.geom_condim[g])}")
  print(f"  half-extent from the compiled mesh {np.round(1000*rig.h,2)} mm, "
        f"body mass {float(m.body_mass[rig.obj_body]):.4f} kg")

  print("\n=== right fingertip colliders (2 of "
        f"{len(rig.hand_geoms)} hand geoms) ===")
  for g in rig.tip_geoms[:2]:
    print(f"  {gname(m,g)[-40:]:40s} {GEOM_TYPE.get(int(m.geom_type[g]),'?'):5s} "
          f"solref={np.round(readback(sv,'geom_solref',g),5)} "
          f"solimp={np.round(readback(sv,'geom_solimp',g),4)} "
          f"prio={readback(sv,'geom_priority',g)} margin={float(m.geom_margin[g]):.4f} "
          f"gap={float(m.geom_gap[g]):.4f} condim={int(m.geom_condim[g])}")

  og, tg = rig.obj_geoms[0], rig.tip_geoms[0]
  s1, s2 = np.asarray(m.geom_solref[og][:2]), np.asarray(m.geom_solref[tg][:2])
  x1, x2 = float(m.geom_solmix[og]), float(m.geom_solmix[tg])
  p1, p2 = int(m.geom_priority[og]), int(m.geom_priority[tg])
  mix = 1.0 if p1 > p2 else (0.0 if p2 > p1 else x1 / max(x1 + x2, 1e-15))
  print(f"\n  MuJoCo mixes solref between two geoms of EQUAL priority: mix={mix:.3f} -> "
        f"{mix*s1 + (1-mix)*s2}. The object's tuned {s1} never reaches this pair.")

  print("\n=== finger actuators (found through actuator_trnid; mjlab's are unnamed) ===")
  by_j = {}
  for a in rig.finger_acts:
    by_j.setdefault(jname(m, rig.jnt_of_act[a]), []).append(a)
  print(f"  {len(rig.finger_acts)} actuators on {len(by_j)} right-finger joints "
        f"({len(rig.finger_acts)/max(len(by_j),1):.1f} per joint) -- "
        f"{len(rig.strong)} mjlab, {len(rig.weak)} xml_motor_unused")
  for a in by_j[sorted(by_j)[0]]:
    print(f"    act{a:4d} gain={float(m.actuator_gainprm[a,0]):9.4f} "
          f"forcerange(host)={m.actuator_forcerange[a]} "
          f"forcerange(device)={readback(sv,'actuator_forcerange',a)}")
  print(f"  weak set total torque budget {float(np.abs(m.actuator_forcerange[rig.weak,1]).sum()):.3f} "
        f"N*m; strong set {float(np.abs(m.actuator_forcerange[rig.strong,1]).sum()):.1f} N*m")

  print("\n=== Newton-side contact override ===")
  ct = env.contacts
  for f in ("rigid_contact_stiffness", "rigid_contact_damping"):
    arr = getattr(ct, f, None) if ct is not None else None
    print(f"  {f}: " + ("absent -> the geom solref/solimp is what the pair uses"
                        if arr is None else
                        f"present, {int((sq(arr)!=0).sum())} nonzero -- WOULD OVERRIDE geom solref"))
  mode = getattr(getattr(env.nmodel, "mujoco", None), "solref_mode", None)
  if mode is not None:
    names = {0: "FORCE_SPACE (recomputed from shape ke/kd)", 1: "RAW",
             2: "MJCF_DEFAULT (the geom's authored solref is used)"}
    print("  shape solref_mode: "
          + ", ".join(f"{int(k)}={names.get(int(k),'?')}" for k in np.unique(sq(mode))))


# ---------------------------------------------------------------------------------- main
def main():
  print(f"one Newton world: {A.xml}\n  object {A.sdf_object}", flush=True)
  env = build_env()
  rig = Rig(env)
  env.reset()
  rig.snapshot()
  rig._pristine = {f: getattr(rig.m, f).copy()
                   for f in ("geom_solref", "geom_solimp", "geom_priority",
                             "actuator_forcerange", "actuator_forcelimited")}
  rig._pristine_impratio = float(rig.m.opt.impratio)

  if A.mode == "facts":
    dump_facts(rig)

  gc = rig.gravcomp_object(True)
  t, spread, clear = rig.calibrate()
  print(f"\n=== press rig ===")
  print(f"  press target (closed fingertip centroid) {np.round(t,4)} m; closed tip spread "
        f"{1000*spread:.1f} mm against a {np.round(2000*rig.h,1)} mm cube")
  print(f"  with the hand fully OPEN the cube at that point is clear by "
        f"{-clear:.2f} mm (positive would mean it starts inside the hand)")
  print(f"  object gravity compensation readback {gc}")

  if A.mode == "facts":
    r = rig.press(1.0)
    print(f"  full closure onto the cube: {r}")
    keep, rows, n = rig.hand_object()
    print(f"\n  contact array: nacon={n}, real rows {len(rows)}, hand<->object {len(keep)}")
    for c in (keep or rows)[:10]:
      print(f"    {gname(rig.m,c['g1'])[-34:]:34s} <-> {gname(rig.m,c['g2'])[-30:]:30s} "
            f"dist={1000*c['dist']:+8.3f} mm solref={np.round(c['solref'],5)} "
            f"solimp={np.round(c['solimp'],3)} Fn={c['force']:9.3f} N")
    return

  settings = [s.strip() for s in A.settings.split(",") if s.strip()]
  rows = []
  if A.mode == "position":
    sweep = [float(x) for x in A.closures.split(",")]
    print(f"\n{'setting':22s}{'closure':>8s}{'commanded':>11s}{'settled':>10s}"
          f"{'geom':>9s}{'ncon':>6s}{'Fn (N)':>10s}{'tau':>8s}{'moved':>8s}{'drift':>8s}")
    print("-" * 100)
    for name in settings:
      reset_settings(rig)
      for note in apply_setting(rig, name):
        print(f"  [readback] {note}")
      for v in sweep:
        c = rig.press(v, park=True)
        s = rig.press(v)
        rows.append(dict(setting=name, closure=v, **c, **s))
        print(f"{name[:22]:22s}{v:8.2f}{c['commanded_mm']:11.3f}{s['settled_mm']:10.3f}"
              f"{s['geom_mm']:9.3f}{s['ncon']:6d}{s['force_N']:10.3f}"
              f"{s['pair_timeconst']:8.4f}{s['object_moved_mm']:8.2f}{s['drift_mm']:8.3f}",
              flush=True)
  else:
    sweep = [float(x) for x in A.forces.split(",")]
    print(f"\n{'setting':22s}{'cap N*m':>9s}{'settled':>10s}{'geom':>9s}{'ncon':>6s}"
          f"{'Fn (N)':>10s}{'mm/N':>10s}{'moved':>8s}{'drift':>8s}")
    print("-" * 95)
    for name in settings:
      reset_settings(rig)
      for note in apply_setting(rig, name):
        print(f"  [readback] {note}")
      xs, ys = [], []
      for v in sweep:
        for note in apply_setting(rig, f"ffl={v}"):
          pass
        s = rig.press(A.force_closure)
        rows.append(dict(setting=name, cap=v, **s))
        if np.isfinite(s["settled_mm"]) and np.isfinite(s["force_N"]) and s["force_N"] > 0:
          xs.append(s["force_N"])
          ys.append(s["settled_mm"])
        slope = (np.polyfit(xs, ys, 1)[0] / 1.0) if len(xs) >= 2 else float("nan")
        print(f"{name[:22]:22s}{v:9.3f}{s['settled_mm']:10.3f}{s['geom_mm']:9.3f}"
              f"{s['ncon']:6d}{s['force_N']:10.3f}{slope:10.5f}"
              f"{s['object_moved_mm']:8.2f}{s['drift_mm']:8.3f}", flush=True)

  if A.out and rows:
    import csv
    keys = sorted({k for r in rows for k in r})
    with open(A.out, "w", newline="") as fh:
      w = csv.DictWriter(fh, fieldnames=keys)
      w.writeheader()
      w.writerows(rows)
    print(f"\nwrote {A.out}")


main()
