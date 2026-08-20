"""A canonical, comparable fact-sheet of a compiled MuJoCo model.

Both sides of this migration end up as a compiled `mujoco.MjModel`: mjlab compiles an MjSpec, and
Newton can export the model it built via `SolverMuJoCo(save_to_mjcf=...)`, which we then compile the
same way. Comparing the two compiled models -- not the two XML files -- is the honest test, because
the compiler is what resolves defaults, class inheritance, and auto-limits. Two very different XMLs
can compile to the same physics, and two similar-looking XMLs can compile to different physics.

Everything the residual policy's behaviour depends on is captured, keyed by NAME rather than index so
that a reordering shows up as a reordering instead of silently shifting every comparison by one:

    joints      type, dof/qpos addresses, range, armature, damping, stiffness, frictionloss
    actuators   trntype, target, gaintype/biastype, gainprm/biasprm, ctrlrange, forcerange, gear
                -- a MuJoCo position servo is gaintype=FIXED, biastype=AFFINE, gainprm[0]=kp,
                   biasprm[1]=-kp, biasprm[2]=-kv, and that is the entire PD definition
    geoms       type, group, contype/conaffinity, condim, priority, solref/solimp, friction, margin,
                gap, size -- the contact parameters the boxstack colliders were tuned around
    bodies      mass, inertia, ipos
    options     timestep, integrator, solver, iterations, cone, impratio, gravity, tolerances

The joint ORDER is recorded as an explicit list as well, because it is the migration's single
highest-risk quantity: a permutation raises nothing and simply routes every actuator command to the
wrong joint.
"""

from __future__ import annotations

import numpy as np


def _name(m, objtype, i):
  import mujoco
  n = mujoco.mj_id2name(m, objtype, i)
  return n if n else f"<unnamed_{objtype}_{i}>"


def _f(x):
  a = np.asarray(x).tolist()
  return a if isinstance(a, list) else float(a)


def facts(m) -> dict:
  import mujoco
  T = mujoco.mjtObj

  jnt = {}
  jnt_order = []
  for i in range(m.njnt):
    n = _name(m, T.mjOBJ_JOINT, i)
    jnt_order.append(n)
    jnt[n] = dict(
      index=i,
      type=int(m.jnt_type[i]),
      body=_name(m, T.mjOBJ_BODY, int(m.jnt_bodyid[i])),
      qposadr=int(m.jnt_qposadr[i]),
      dofadr=int(m.jnt_dofadr[i]),
      limited=int(m.jnt_limited[i]),
      range=_f(m.jnt_range[i]),
      armature=_f(m.dof_armature[m.jnt_dofadr[i]]),
      damping=_f(m.dof_damping[m.jnt_dofadr[i]]),
      frictionloss=_f(m.dof_frictionloss[m.jnt_dofadr[i]]),
      stiffness=_f(m.jnt_stiffness[i]),
    )

  act = {}
  act_order = []
  for i in range(m.nu):
    n = _name(m, T.mjOBJ_ACTUATOR, i)
    act_order.append(n)
    trnid = int(m.actuator_trnid[i][0])
    trntype = int(m.actuator_trntype[i])
    # Resolve what the actuator drives, so "same kp on a different joint" cannot look like a match.
    if trntype == int(mujoco.mjtTrn.mjTRN_JOINT):
      target = _name(m, T.mjOBJ_JOINT, trnid)
    else:
      target = f"<trntype{trntype}_id{trnid}>"
    act[n] = dict(
      index=i,
      trntype=trntype,
      target=target,
      gaintype=int(m.actuator_gaintype[i]),
      biastype=int(m.actuator_biastype[i]),
      gainprm=_f(m.actuator_gainprm[i][:3]),
      biasprm=_f(m.actuator_biasprm[i][:3]),
      ctrllimited=int(m.actuator_ctrllimited[i]),
      ctrlrange=_f(m.actuator_ctrlrange[i]),
      forcelimited=int(m.actuator_forcelimited[i]),
      forcerange=_f(m.actuator_forcerange[i]),
      gear=_f(m.actuator_gear[i][:3]),
      # The servo law in the terms we actually reason about. Only meaningful for FIXED/AFFINE.
      kp=float(m.actuator_gainprm[i][0]),
      kv=float(-m.actuator_biasprm[i][2]),
    )

  geom = {}
  for i in range(m.ngeom):
    n = _name(m, T.mjOBJ_GEOM, i)
    geom[n] = dict(
      index=i,
      type=int(m.geom_type[i]),
      body=_name(m, T.mjOBJ_BODY, int(m.geom_bodyid[i])),
      group=int(m.geom_group[i]),
      contype=int(m.geom_contype[i]),
      conaffinity=int(m.geom_conaffinity[i]),
      condim=int(m.geom_condim[i]),
      priority=int(m.geom_priority[i]),
      solref=_f(m.geom_solref[i]),
      solimp=_f(m.geom_solimp[i]),
      friction=_f(m.geom_friction[i]),
      margin=_f(m.geom_margin[i]),
      gap=_f(m.geom_gap[i]),
      size=_f(m.geom_size[i]),
      pos=_f(m.geom_pos[i]),
    )

  body = {}
  for i in range(m.nbody):
    n = _name(m, T.mjOBJ_BODY, i)
    body[n] = dict(
      index=i,
      parent=_name(m, T.mjOBJ_BODY, int(m.body_parentid[i])),
      mass=_f(m.body_mass[i]),
      inertia=_f(m.body_inertia[i]),
      ipos=_f(m.body_ipos[i]),
      pos=_f(m.body_pos[i]),
    )

  opt = m.opt
  options = dict(
    timestep=_f(opt.timestep),
    integrator=int(opt.integrator),
    solver=int(opt.solver),
    iterations=int(opt.iterations),
    ls_iterations=int(opt.ls_iterations),
    cone=int(opt.cone),
    jacobian=int(opt.jacobian),
    impratio=_f(opt.impratio),
    tolerance=_f(opt.tolerance),
    ls_tolerance=_f(opt.ls_tolerance),
    gravity=_f(opt.gravity),
    density=_f(opt.density),
    viscosity=_f(opt.viscosity),
    disableflags=int(opt.disableflags),
    enableflags=int(opt.enableflags),
  )

  return dict(
    sizes=dict(nq=int(m.nq), nv=int(m.nv), nu=int(m.nu), njnt=int(m.njnt),
               nbody=int(m.nbody), ngeom=int(m.ngeom), nmesh=int(m.nmesh)),
    joint_order=jnt_order,
    actuator_order=act_order,
    joints=jnt,
    actuators=act,
    geoms=geom,
    bodies=body,
    options=options,
  )
