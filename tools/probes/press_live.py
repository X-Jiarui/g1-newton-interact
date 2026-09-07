"""Watch the hand press down onto the block, live, with the physics running.

This is the demonstration the offline rigs could not be. The robot stands on the floor, the block
rests on the table under gravity exactly as it does in training, and only the right arm moves --
along a planned straight line, from clear space above the block, straight down through where the
block is. Nothing is pinned to the air, nothing is teleported, no policy is involved. If the contact
works, the hand stops at the surface and the block reacts; if it does not, you watch it go in.

Why the earlier rigs were not this, each of which produced a wrong conclusion first:

  * they PINNED the block and drove it into a stationary hand, so the block hung in mid-air with its
    own physics switched off -- which is why it looked like it was floating;
  * one seeded the arm from a training rollout frame, so the run began 9.4 mm INSIDE the block and
    then showed the solver repairing an illegal initial state, which is not a wall;
  * two "passes" turned out to be runs where nothing ever touched, because the collider filter
    matched the LEFT hand while the block travelled down the right hand's axis. A run with no
    contact scores a perfect penetration number.

All three are guarded here: the block is free and resting on the table, step 0 is asserted clear of
it, and a run that never registers contact is reported as INVALID rather than as a clean number.

Open the printed URL. The 3-D view is orbit / zoom / pan with the mouse, so you can put the camera
on the fingertip, and the metrics stream beside it as live plots: penetration, contact count,
normal force, and the depth being commanded.

    python tools/probes/press_live.py --port 8080
    SIM_TIMESTEP=0.001 python tools/probes/press_live.py --port 8080 --settings stack
"""
from __future__ import annotations

import argparse
import os
import sys

# ik_press_bench parses argv at import time, so hand it a clean one and keep ours.
_OURS, sys.argv = sys.argv[1:], [sys.argv[0], "--mode", "facts"]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ik_press_bench as B  # noqa: E402

sys.argv = [sys.argv[0]] + _OURS

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8080)
ap.add_argument("--settings", default="stack",
                help="'baseline', or 'stack' = dt 1 ms + object solref .002 + priority 1 + finger "
                     "torque 0.62 N-m + the leftover xml_motor_unused_* actuators neutralised")
ap.add_argument("--standoff", type=float, default=0.08,
                help="metres above the block's top face where the descent starts")
ap.add_argument("--through", type=float, default=0.03,
                help="metres BELOW the top face the fingertip is commanded to. Impossible on "
                     "purpose: correct physics simply refuses it")
ap.add_argument("--approach-s", type=float, default=6.0,
                help="seconds of descent. Slow: the position servo lags badly, and a 29 cm reach "
                     "in 2 s left 38-207 mm of tracking error so the hand never arrived")
ap.add_argument("--hold-s", type=float, default=8.0,
                help="seconds held at the bottom, long enough for the servo to converge")
ap.add_argument("--settle-s", type=float, default=1.5,
                help="seconds before the press, so the block lands on the table and the robot "
                     "stands: mjlab's reset event is what puts the table under the block")
ap.add_argument("--reach-s", type=float, default=6.0,
                help="seconds to travel from wherever the arm settled to the standoff point")
ap.add_argument("--ik-iters", type=int, default=120,
                help="DLS iterations per waypoint, warm-started from the previous one")
ap.add_argument("--damping", type=float, default=0.05, help="damped-least-squares lambda")
ap.add_argument("--once", action="store_true", help="stop after one press instead of looping")
ap.add_argument("--pose", default="palm_down", choices=("palm_down", "palm_in"),
                help="palm_down: palm faces the floor, fingers naturally curled, index and middle "
                     "left straighter so they meet the top face first. "
                     "palm_in: palm faces the robot, index extended, the rest closed into a fist, "
                     "poking straight down.")
ap.add_argument("--move-robot", type=float, default=0.36,
                help="metres in front of the block to stand the robot, written ONCE into the base "
                     "free joint after the settle. From where the reference clip leaves it the "
                     "block is 80 cm from the settled fingertip -- 675 mm of IK residual even with "
                     "the waist in the chain -- so the arm can never track and the run measures "
                     "reachability. Moving the ROBOT (rather than the table) leaves the table, the "
                     "block and their contact exactly as training has them. 0 disables.")
ap.add_argument("--place-block", type=int, default=0,
                help="move the table and block to a comfortable reach directly under the hand. "
                     "Where the reference clip puts them is 80 cm from the settled fingertip -- "
                     "past the G1's reach even with the waist in the chain -- so the arm can never "
                     "track the plan and the run measures reachability, not the wall. Moving the "
                     "table changes nothing about the contact being tested.")
ap.add_argument("--drop-below", type=float, default=0.16,
                help="metres below the settled fingertip to put the block's top face")
ap.add_argument("--corrections", type=int, default=6,
                help="rounds of static-error correction on the descent. The servo's droop depends "
                     "on the arm's configuration, so one feedforward cannot cover a 110 mm reach")
ap.add_argument("--max-push", type=float, default=0.05,
                help="metres of extra command per correction round; clamped so one bad round "
                     "cannot run away")
ap.add_argument("--correct-s", type=float, default=1.5, help="seconds per correction ramp")
ap.add_argument("--dump", default=None, help="record qpos/mocap per control step for render_traj")
ap.add_argument("--w-rot", type=float, default=0.35,
                help="weight on the palm-orientation half of the IK task")
ap.add_argument("--no-waist", action="store_true",
                help="plan with the arm alone. It does not reach the block from a standing rest "
                     "pose -- kept so that fact stays reproducible")
A = ap.parse_args()

ARM = ("right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
       "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
       "right_wrist_yaw_joint")
# The right arm ALONE cannot reach the block: the reference clip puts it 25 cm in front, and the
# first plan left 131 mm of residual. In training the robot gets there by leaning, not by stretching
# the arm, so the waist joins the chain. The feet stay planted either way -- the base is pinned.
WAIST = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")


def build_env(port: int):
    """B.build_env, plus the live viewer. Kept here rather than edited there so the bench and this
    demo cannot break each other."""
    from pathlib import Path
    import json as _json
    import yaml as _yaml
    cfg = B.load_env_cfg(B.TASK, play=False)
    cfg.scene.num_envs = 1
    agent_cfg = B.load_rl_cfg(B.TASK)
    B._apply_cfg_mapping(agent_cfg, _yaml.unsafe_load(
        (Path(B.A.agent_cfg_from).parent / "params" / "agent.yaml").open()))
    B.apply_reward_weights(cfg, B.reward_weights_from_env_yaml(Path(B.A.reward_cfg)))
    s = cfg.actions.get("sonic_action") if isinstance(cfg.actions, dict) else cfg.actions.sonic_action
    s.tracking_start_assist_gain = 0.0
    s.tracking_start_assist_steps = 0
    if str(getattr(agent_cfg, "base_tracker_kind", "")).strip().lower() == "astra_onnx":
        from mjlab.tasks.residual_interact.env_cfgs import set_astra_body_dynamics
        set_astra_body_dynamics(cfg)
    return B.NewtonVecEnv(cfg, B.A.xml, num_envs=1, device="cuda:0",
                          nconmax=B.A.nconmax, njmax=B.A.njmax,
                          sdf_object_stl=B.A.sdf_object, sdf_resolution=B.A.sdf_resolution,
                          native_contacts=bool(B.A.native_contacts),
                          hydro_object_table=False, table_under_object=True,
                          object_solref="0.004,1.0", cuda_graph=False,
                          viser_port=port, render_every=1,
                          solver_kwargs=_json.loads(B.A.solver_kwargs))


class Press:
    def __init__(self, rig):
        import mujoco
        self.rig, m = rig, rig.m
        # Arm joints matched on the name's TAIL. mjlab flattens the whole body path into every
        # joint name, and it joins with underscores, not slashes -- so splitting on "/" returns the
        # entire path and matches nothing, while a bare `"right_shoulder" in name` would match all
        # twenty finger joints. endswith on the joint's own name is the one test that is both.
        chain = ARM if A.no_waist else ARM + WAIST
        self.arm_j = [j for j in range(m.njnt)
                      if any((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").endswith(x)
                             for x in chain)]
        if len(self.arm_j) != len(chain):
            raise SystemExit(f"matched {len(self.arm_j)} of {len(chain)} chain joints")
        print(f"[press-live] IK chain: {len(chain)} joints "
              f"({'arm only' if A.no_waist else 'arm + waist'})")
        self.arm_q = np.array([int(m.jnt_qposadr[j]) for j in self.arm_j])
        self.arm_v = np.array([int(m.jnt_dofadr[j]) for j in self.arm_j])
        # TWO actuators per joint here as well, not just on the fingers: mjlab adds its own servo
        # and leaves the scene's `xml_motor_unused_*` one in place. Both must be commanded, or the
        # leftover one holds the joint at its own target and fights the press.
        slot = {j: i for i, j in enumerate(self.arm_j)}
        self.arm_a = [a for a in range(m.nu) if rig.jnt_of_act[a] in slot]
        self.arm_a_slot = np.array([slot[rig.jnt_of_act[a]] for a in self.arm_a])
        self._q_ref = B.sq(rig.qpos()).copy()
        # The INDEX finger, not whatever sorts first -- that is finger1, the thumb. The Wuji hand
        # numbers thumb..pinky as finger1..finger5.
        def _body(sub):
            hits = [b for b in range(m.nbody)
                    if sub in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "")]
            if not hits:
                raise SystemExit(f"no body matching {sub!r}")
            return hits[0]
        self.index_body = _body("right_finger2_link4")
        self.middle_body = _body("right_finger3_link4")
        self.palm_body = _body("right_palm_link")
        self.tip_body = self.index_body
        self.cpu = mujoco.MjData(m)
        # Which way does the palm face? Defined from the hand's own motion rather than guessed:
        # the fingers curl TOWARD the palm, so the displacement of the fingertips between fully
        # open and fully closed, expressed in the palm frame, is the palm normal.
        self.palm_axis_local = self._palm_axis()

    def _palm_axis(self):
        import mujoco
        m, rig = self.rig.m, self.rig
        fj = sorted({rig.jnt_of_act[a] for a in rig.finger_acts})
        qa = [int(m.jnt_qposadr[j]) for j in fj]
        lo = np.array([m.jnt_range[j][0] for j in fj])
        hi = np.array([m.jnt_range[j][1] for j in fj])
        tips = sorted(rig.tip_bodies)

        def mean_tip(vals):
            self.cpu.qpos[:] = self._q_ref
            for a, v in zip(qa, vals):
                self.cpu.qpos[a] = v
            mujoco.mj_forward(m, self.cpu)
            return np.mean([self.cpu.xpos[b] for b in tips], axis=0)

        d = mean_tip(hi) - mean_tip(lo)
        self.cpu.qpos[:] = self._q_ref
        mujoco.mj_forward(m, self.cpu)
        R = self.cpu.xmat[self.palm_body].reshape(3, 3)
        a = R.T @ d
        return a / max(np.linalg.norm(a), 1e-9)

    def palm_axis_world(self, q):
        import mujoco
        self.cpu.qpos[:] = q
        mujoco.mj_forward(self.rig.m, self.cpu)
        return self.cpu.xmat[self.palm_body].reshape(3, 3) @ self.palm_axis_local

    def tip_world(self, q):
        import mujoco
        self.cpu.qpos[:] = q
        mujoco.mj_forward(self.rig.m, self.cpu)
        return self.cpu.xpos[self.tip_body].copy()

    def finger_targets(self, pose):
        """The hand SHAPE is part of the plan, not an afterthought: the fingers are commanded into
        it before the descent starts, so what meets the block is the surface we intended.

        Fractions of each joint's own range, so the same numbers mean the same shape on every
        finger regardless of its individual limits.
        """
        m, rig = self.rig.m, self.rig
        if pose == "palm_down":
            # Naturally curled, with index and middle left straighter so their tips are the lowest
            # thing on the hand and meet the top face first.
            frac = {1: 0.55, 2: 0.18, 3: 0.18, 4: 0.55, 5: 0.55}
        else:
            # A fist with the index extended: one fingertip pokes straight down.
            frac = {1: 0.85, 2: 0.00, 3: 0.90, 4: 0.90, 5: 0.90}
        out = {}
        for a in rig.finger_acts:
            n = B.jname(m, rig.jnt_of_act[a])
            k = next((i for i in frac if f"right_finger{i}_joint" in n), None)
            if k is None:
                continue
            lo, hi = m.actuator_ctrlrange[a]
            out[a] = float(lo + frac[k] * (hi - lo))
        return out

    def solve(self, q, target, iters=None, axis_des=None):
        """Damped least squares on the arm chain only, run to convergence.

        Position task only: adding an orientation task made the solver swing 1.26 rad to move the
        tip 16 mm. Joint limits are respected, because an IK answer outside them is one the servo
        will never reach and the residual would silently be blamed on the contact.
        """
        import mujoco
        m = self.rig.m
        q = q.copy()
        err = np.zeros(3)
        for _ in range(iters or A.ik_iters):
            self.cpu.qpos[:] = q
            mujoco.mj_forward(m, self.cpu)
            err = target - self.cpu.xpos[self.tip_body]
            jacp = np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, self.cpu, jacp, None, self.tip_body)
            if axis_des is None:
                if np.linalg.norm(err) < 1e-5:
                    break
                J, e6 = jacp[:, self.arm_v], err
            else:
                # Align the palm's own normal with a world direction. An AXIS task, not a full
                # orientation task: it leaves the spin about that axis free, so the solver has room
                # to satisfy the position task as well. A full orientation target here made the
                # solver swing 1.26 rad to move the tip 16 mm.
                jacr = np.zeros((3, m.nv))
                mujoco.mj_jacBody(m, self.cpu, None, jacr, self.palm_body)
                a_cur = self.cpu.xmat[self.palm_body].reshape(3, 3) @ self.palm_axis_local
                rot_err = np.cross(a_cur, axis_des)
                if np.linalg.norm(err) < 1e-5 and np.linalg.norm(rot_err) < 1e-4:
                    break
                J = np.vstack([jacp[:, self.arm_v], A.w_rot * jacr[:, self.arm_v]])
                e6 = np.concatenate([err, A.w_rot * rot_err])
            dq = J.T @ np.linalg.solve(J @ J.T + (A.damping ** 2) * np.eye(J.shape[0]), e6)
            q[self.arm_q] += np.clip(dq, -0.05, 0.05)
            for k, j in enumerate(self.arm_j):
                if m.jnt_limited[j]:
                    lo, hi = m.jnt_range[j]
                    q[self.arm_q[k]] = float(np.clip(q[self.arm_q[k]], lo, hi))
        return q[self.arm_q].copy(), float(np.linalg.norm(err))

    def plan(self, q0, waypoints, iters, axis_des=None):
        """Solve the whole path offline, each waypoint warm-started from the last.

        Re-solving from the LIVE qpos every control step does not work: the servo lags the command,
        so the solver keeps starting from behind and the residual never closes -- the first version
        of this sat at 104-148 mm of error for the entire run and never touched the block. Planning
        once and playing the trajectory is also what "a planned motion" is supposed to mean.
        """
        q = q0.copy()
        out, worst = [], 0.0
        for w in waypoints:
            arm, e = self.solve(q, w, iters, axis_des=axis_des)
            q[self.arm_q] = arm
            out.append(arm.copy())
            worst = max(worst, e)
        return out, worst


def signed_depth_mm(rig, centre, geoms):
    """Signed: positive inside, NEGATIVE clear, in mm.

    `Rig.depth_into_cube_mm` starts its running maximum at 0.0, so it returns exactly +0.000 for
    every non-penetrating pose and cannot distinguish "just touching" from "30 cm away". That made
    the step-0 clearance assertion vacuous and hid a run in which the arm never moved.
    """
    best = -1e9
    c = np.asarray(centre)[None, :]
    for _g, P in rig.hand_world_verts(geoms).items():
        d = rig.h[None, :] - np.abs(P - c)
        if d.size:
            best = max(best, float(d.min(axis=1).max()))
    return 1000.0 * best


def main():
    env = build_env(A.port)
    rig = B.Rig(env)
    B.apply_setting(rig, A.settings)
    press = Press(rig)
    m = rig.m
    dt = env.physics_dt * env.decimation
    print(f"[press-live] dt {1000*env.physics_dt:.2f} ms x decimation {env.decimation} "
          f"= {1000*dt:.1f} ms per control step; setting '{A.settings}'")

    def rest(cmd_vec, body, tol_mm_s=2.0, max_s=6.0):
        """Hold a command until the body stops moving, and report how long it took.

        The position servo has a steady-state error under gravity, so the arm keeps creeping for
        seconds after the command stops changing. Reading the fingertip before it has settled put
        the block 7 cm to the side of where the hand actually ended up.
        """
        prev = B.sq(rig.d.xpos)[body].copy()
        for i in range(int(max_s / 0.2)):
            advance(cmd_vec, int(0.2 / dt))
            now = B.sq(rig.d.xpos)[body].copy()
            v = 1000.0 * float(np.linalg.norm(now - prev)) / 0.2
            prev = now
            if v < tol_mm_s:
                return (i + 1) * 0.2, v
        return max_s, v

    def advance(cmd_vec, n=1):
        c = rig.cmd()
        t = torch.tensor(cmd_vec, dtype=c.dtype, device=c.device)
        for _ in range(n):
            # Velocity-level pin on the floating base: the G1 stands still. Writing its POSE every
            # step fights the integrator and produced NaN in QACC on this very scene.
            if rig.base_vadr is not None:
                rig.qvel()[0, rig.base_vadr:rig.base_vadr + 6] = 0.0
            rig.cmd()[0, :] = t
            env._physics_step()
            env.state_in, env.state_out = env.state_out, env.state_in

    # 1. Settle through the FULL env.step, not the bare physics step.
    #
    #    The table is a mocap body and only mjlab's mdp writes its runtime pose; a fresh model
    #    carries the authored pose, which is z = 0 -- the table sits inside the floor and the block
    #    falls straight through it. The first version of this settled with `_physics_step` alone and
    #    the block ended up on the ground at z = 0.02, 80 cm below where the hand was pressing, with
    #    the run dutifully reporting "gap -500 mm, 0 contacts".
    #
    #    Mocap poses persist once written, so stepping the real pipeline for the settle is enough:
    #    afterwards the press can drive the arm directly and the table stays put.
    nact = int(env._env.action_manager.total_action_dim)
    zero_act = torch.zeros((env.num_envs, nact), device="cuda:0")
    for _ in range(int(A.settle_s / dt)):
        with torch.inference_mode():
            env.step(zero_act)
    # Stand the robot next to the block. Written once, as an initial condition, not per step -- and
    # to the BASE free joint only, so the table, the block and every contact stay as training has
    # them. Moving the table instead was tried first and left the block in free fall: its SDF
    # collider did not follow the mocap, and the block fell 1.3 m through where the table should be.
    if A.move_robot > 0:
        obj0 = B.sq(rig.d.xpos)[rig.obj_body].copy()
        base_j = [j for j in range(m.njnt) if int(m.jnt_type[j]) == 0
                  and "floating_base" in B.jname(m, j)][0]
        bq = int(m.jnt_qposadr[base_j])
        q = rig.qpos()
        cur = B.sq(q)[bq:bq + 3].copy()
        tgt = np.array([obj0[0] - A.move_robot, obj0[1], cur[2]])
        q[0, bq:bq + 3] = torch.tensor(tgt, dtype=q.dtype, device=q.device)
        rig.qvel()[0, :] = 0.0
        for _ in range(int(1.5 / dt)):
            if rig.base_vadr is not None:
                rig.qvel()[0, rig.base_vadr:rig.base_vadr + 6] = 0.0
            rig.cmd()[0, :] = torch.tensor(rig.target_from_qpos(), dtype=rig.cmd().dtype,
                                           device=rig.cmd().device)
            env._physics_step()
            env.state_in, env.state_out = env.state_out, env.state_in
        print(f"[press-live] robot moved {np.round(cur,3)} -> {np.round(tgt,3)}, "
              f"{100*A.move_robot:.0f} cm in front of the block at {np.round(obj0,3)}")
    rig.snapshot(freeze_hold=True)
    hold = rig.target_from_qpos()
    tb = [b for b in range(m.nbody) if int(m.body_mocapid[b]) >= 0]
    print("[press-live] mocap bodies after settle: "
          + ", ".join(f"{B.bname(m, b).split('_')[-1]} z={B.sq(rig.d.xpos)[b][2]:.4f}" for b in tb))

    # 2. Shape the hand FIRST. The pose is part of the plan: what meets the block has to be the
    #    surface we chose, not whatever shape the reset happened to leave.
    fing = press.finger_targets(A.pose)
    for a, v in fing.items():
        hold[a] = v
    n_shape = int(1.5 / dt)
    advance(hold, n_shape)
    print(f"[press-live] pose '{A.pose}': commanded {len(fing)} finger actuator(s); "
          f"palm axis in the palm frame {np.round(press.palm_axis_local, 3)}")

    # 3. Orient the palm. palm_down points its normal at the floor; palm_in points it back at the
    #    robot, computed from the pelvis rather than assumed to be -x.
    q = B.sq(rig.qpos()).copy()
    tip = press.tip_world(q)
    if A.pose == "palm_down":
        axis_des = np.array([0.0, 0.0, -1.0])
    else:
        pelvis = [b for b in range(m.nbody) if B.bname(m, b).endswith("robot_pelvis")]
        pv = B.sq(rig.d.xpos)[pelvis[0]] - tip if pelvis else np.array([-1.0, 0.0, 0.0])
        pv[2] = 0.0
        axis_des = pv / max(np.linalg.norm(pv), 1e-9)
    n_orient = int(A.reach_s / dt)
    orient_traj, e_or = press.plan(q, [tip] * n_orient, A.ik_iters, axis_des=axis_des)
    for k in range(n_orient):
        cmd = hold.copy()
        cmd[press.arm_a] = orient_traj[k][press.arm_a_slot]
        advance(cmd)
    hold[press.arm_a] = orient_traj[-1][press.arm_a_slot]
    t_rest, v_rest = rest(hold, press.index_body)
    q = B.sq(rig.qpos()).copy()
    tip = press.tip_world(q)
    print(f"[press-live] arm came to rest after {t_rest:.1f} s ({v_rest:.2f} mm/s)")
    got = press.palm_axis_world(q)
    print(f"[press-live] palm axis now {np.round(got,3)} vs wanted {np.round(axis_des,3)} "
          f"(angle {np.degrees(np.arccos(np.clip(got @ axis_des, -1, 1))):.1f} deg); "
          f"index fingertip at {np.round(tip,4)}")

    # 4. Put the block where the hand can actually reach it.
    #
    #    Where the reference clip leaves it, the settled fingertip is 80 cm away -- past the G1's
    #    reach even with the waist in the chain. The arm then never tracks the plan (error grew from
    #    21 mm to 125 mm against a STATIONARY target) and the run measures reachability, not the
    #    wall. The table is a mocap body, so it moves by writing mocap_pos; mocap persists, and the
    #    press drives the arm directly afterwards.
    if A.place_block:
        mocap = [b for b in range(m.nbody) if int(m.body_mocapid[b]) >= 0
                 and "table" in B.bname(m, b)]
        if not mocap:
            raise SystemExit("no table mocap body found")
        slot = int(m.body_mocapid[mocap[0]])
        mp = B.sq(rig.d.mocap_pos).copy()
        # The table's top surface, MEASURED, not derived from the collider mesh. Reading the z
        # extent of table_box.stl gave 105 mm against the authored half-height of 20 mm, which put
        # the table 77 mm too low; the block was still in free fall when the descent was planned and
        # the hand pressed 47 mm above where the block actually came to rest.
        # After the settle the block rests ON the table, so the top is exactly one half-extent below
        # the block's centre, and the offset to the mocap body follows.
        obj0 = B.sq(rig.d.xpos)[rig.obj_body].copy()
        top_now = obj0[2] - float(rig.h[2])
        off = top_now - float(mp[slot][2])

        top = tip - np.array([0.0, 0.0, A.drop_below])
        centre = top - np.array([0.0, 0.0, float(rig.h[2])])
        mp[slot] = np.array([centre[0], centre[1], centre[2] - float(rig.h[2]) - off])
        mt = wp.to_torch(rig.d.mocap_pos)
        mt[:] = torch.as_tensor(mp, dtype=mt.dtype, device=mt.device).reshape(mt.shape)
        qq = rig.qpos()
        qq[0, rig.obj_qadr:rig.obj_qadr + 3] = torch.tensor(
            centre + np.array([0.0, 0.0, 0.002]), dtype=qq.dtype, device=qq.device)
        qq[0, rig.obj_qadr + 3:rig.obj_qadr + 7] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], dtype=qq.dtype, device=qq.device)
        rig.qvel()[0, rig.obj_vadr:rig.obj_vadr + 6] = 0.0
        # Wait until it is actually AT REST. A fixed 1 s wait was not enough with the table
        # misplaced, and planning against a falling block is how the hand came to press at thin air.
        for _ in range(30):
            advance(hold, int(0.2 / dt))
            v = B.sq(rig.qvel())[rig.obj_vadr:rig.obj_vadr + 3]
            if float(np.linalg.norm(v)) < 1e-3:
                break
        obj_c = B.sq(rig.d.xpos)[rig.obj_body].copy()
        print(f"[press-live] table top measured at z {top_now:.4f} (mocap offset "
              f"{1000*off:.1f} mm); block placed and settled at {np.round(obj_c, 4)}, "
              f"speed {1000*float(np.linalg.norm(B.sq(rig.qvel())[rig.obj_vadr:rig.obj_vadr+3])):.3f} mm/s")

    # 5. Plan the descent from where the hand and the block actually are.
    rest(hold, press.index_body)
    q = B.sq(rig.qpos()).copy()
    obj_c = B.sq(rig.d.xpos)[rig.obj_body].copy()
    top = obj_c + np.array([0.0, 0.0, float(rig.h[2])])
    tip = press.tip_world(q)
    start = np.array([top[0], top[1], tip[2]])
    goal = top - np.array([0.0, 0.0, A.through])
    pen0 = signed_depth_mm(rig, obj_c, rig.press_geoms)
    print(f"[press-live] block centre {np.round(obj_c,4)}  top face z {top[2]:.4f}")
    print(f"[press-live] fingertip {np.round(tip,4)} -> over the block {np.round(start,4)} "
          f"-> commanded {np.round(goal,4)} ({1000*A.through:.0f} mm below the face, on purpose)")
    print(f"[press-live] step-0 gap {pen0:+.3f} mm "
          f"({'CLEAR' if pen0 <= 0.0 else 'ALREADY INSIDE -- the run would be meaningless'})")
    if pen0 > 0.0:
        raise SystemExit("step 0 is inside the block; refusing to run")

    # Plan DISPLACEMENTS from the pose the arm is being COMMANDED to, not from where it actually
    # is. The servo sits below its command under gravity; solving IK for the actual tip therefore
    # asks the arm to climb back up to it, and the first version of this drifted 7 cm sideways and
    # 160 mm of tracking error while the command was nominally constant. Feeding the command the
    # same displacement we want the tip to make keeps that offset constant instead of fighting it.
    q_cmd = q.copy()
    for a, sl in zip(press.arm_a, press.arm_a_slot):
        q_cmd[press.arm_q[sl]] = hold[a]
    tip_cmd = press.tip_world(q_cmd)
    print(f"[press-live] commanded tip {np.round(tip_cmd,4)} vs actual {np.round(tip,4)}: the "
          f"servo sits {1000*float(np.linalg.norm(tip_cmd-tip)):.1f} mm from its own target")

    n_reach = int(A.reach_s / dt)
    n_down, n_hold = int(A.approach_s / dt), int(A.hold_s / dt)
    reach_pts = [tip_cmd + (start - tip) * (i + 1) / n_reach for i in range(n_reach)]
    down_pts = [tip_cmd + (start - tip) + (goal - start) * min(1.0, (i + 1) / n_down)
                for i in range(n_down + n_hold)]
    pts = reach_pts + down_pts
    traj, worst_ik = press.plan(q_cmd, pts, A.ik_iters, axis_des=axis_des)
    print(f"[press-live] planned {len(traj)} waypoints; worst IK residual {1000*worst_ik:.2f} mm")
    if worst_ik > 0.005:
        print(f"[press-live] WARNING the plan does not close: {1000*worst_ik:.1f} mm of residual "
              f"means any contact result would be about reachability, not the wall.")

    worst, contact_steps, step, t0 = 0.0, 0, 0, 0.0
    rec_q, rec_mp, rec_mq, rec_pen, rec_n, rec_f = [], [], [], [], [], []

    def tick(cmd, target):
        """One control step, plus every measurement and the live plots."""
        nonlocal worst, contact_steps, step, t0
        advance(cmd)
        obj_c = B.sq(rig.d.xpos)[rig.obj_body].copy()
        pen = signed_depth_mm(rig, obj_c, rig.press_geoms)
        tip_now = press.tip_world(B.sq(rig.qpos()))
        track = 1000.0 * float(np.linalg.norm(tip_now - target))
        keep, _rows, _n = rig.hand_object()
        fn = float(sum(c["force"] for c in keep)) if keep else 0.0
        worst = max(worst, pen)
        contact_steps += 1 if keep else 0
        if A.dump:
            rec_q.append(B.sq(rig.qpos()).copy())
            rec_mp.append(B.sq(rig.d.mocap_pos).copy())
            rec_mq.append(B.sq(rig.d.mocap_quat).copy())
            rec_pen.append(pen)
            rec_n.append(len(keep))
            rec_f.append(fn)
        if env.viewer is not None:
            env.viewer.begin_frame(t0)
            env.viewer.log_state(env.state_in)
            for nm, val in (("penetration_mm", pen), ("hand_object_contacts", len(keep)),
                            ("normal_force_N", fn), ("worst_penetration_mm", worst),
                            ("commanded_depth_mm", 1000.0 * (top[2] - target[2])),
                            ("tracking_error_mm", track)):
                env.viewer.log_array(nm, np.array([val], dtype=np.float32))
            env.viewer.end_frame()
        t0 += dt
        if step % 25 == 0:
            print(f"[press-live] {step:5d}  gap {pen:+8.2f} mm  worst {worst:+7.3f}  "
                  f"contacts {len(keep):3d}  Fn {fn:8.2f} N  tip {np.round(tip_now,3)}  "
                  f"block {np.round(obj_c,3)}  reach err {track:6.1f} mm", flush=True)
        step += 1
        return tip_now

    def ramp_to(arm_target, secs, target_pt):
        """Interpolate the arm COMMAND in joint space and hold it there."""
        n = max(1, int(secs / dt))
        a0 = hold[press.arm_a].copy()
        a1 = arm_target[press.arm_a_slot]
        for i in range(n):
            c = hold.copy()
            c[press.arm_a] = a0 + (a1 - a0) * (i + 1) / n
            tick(c, target_pt)
        hold[press.arm_a] = a1

    # Reach across to sit over the block, then press down through it.
    arm_reach, _e = press.solve(q_cmd, pts[n_reach - 1], A.ik_iters, axis_des=axis_des)
    ramp_to(arm_reach, A.reach_s, start)
    arm_down, _e = press.solve(q_cmd, pts[-1], A.ik_iters, axis_des=axis_des)
    ramp_to(arm_down, A.approach_s, goal)

    # The position servo droops under gravity and the droop DEPENDS on the arm's configuration, so
    # one feedforward offset cannot cover a 110 mm descent: the arm was commanded down 110 mm and
    # moved 4 mm. Correct it the way a static error is always corrected -- measure what is left and
    # add it to the command -- and repeat. Because the goal is INSIDE the block, this keeps pushing
    # until the CONTACT is what balances the servo, which is exactly the position-control test:
    # commanded deeper, and the wall is the only thing that can refuse.
    # VERTICAL only, and always re-solved from the same commanded pose with an accumulated offset.
    # Correcting the full 3-D error instead pushed in x and y as well: the targets left the
    # reachable set (IK residual 293 mm), the arm swung sideways, and it swept the block off the
    # table without ever pressing on it. Each push is clamped so one bad round cannot run away.
    push_z = 0.0
    for it in range(A.corrections):
        tip_now = press.tip_world(B.sq(rig.qpos()))
        dz = float(goal[2] - tip_now[2])
        if abs(dz) < 1e-4:
            break
        push_z += float(np.clip(dz, -A.max_push, A.max_push))
        want = pts[-1] + np.array([0.0, 0.0, push_z])
        arm_c, res = press.solve(q_cmd, want, A.ik_iters, axis_des=axis_des)
        if res > 0.005:
            print(f"[press-live] correction {it+1} abandoned: the command needed "
                  f"({np.round(want,4)}) is {1000*res:.1f} mm outside the arm's reach", flush=True)
            break
        print(f"[press-live] correction {it+1}: {1000*dz:+.1f} mm short vertically, command now "
              f"{1000*push_z:+.1f} mm below plan (IK residual {1000*res:.2f} mm)", flush=True)
        ramp_to(arm_c, A.correct_s, goal)
        for _ in range(int(A.hold_s / dt)):
            tick(hold, goal)

    if A.dump and rec_q:
        names = [x for _, x in sorted(zip(
            [int(m.body_mocapid[b]) for b in range(m.nbody) if m.body_mocapid[b] >= 0],
            [B.bname(m, b) for b in range(m.nbody) if m.body_mocapid[b] >= 0]))]
        np.savez_compressed(A.dump, qpos=np.stack(rec_q), mocap_pos=np.stack(rec_mp),
                            mocap_quat=np.stack(rec_mq), mocap_names=np.array(names),
                            overlap_mm=np.asarray(rec_pen), ncon=np.asarray(rec_n, dtype=float),
                            contact_force_N=np.asarray(rec_f))
        print(f"[press-live] wrote {A.dump} ({len(rec_q)} frames)")

    verdict = ("INVALID -- nothing ever touched, so the penetration number means nothing"
               if contact_steps < 50 else f"worst penetration {worst:+.3f} mm")
    print(f"[press-live] {verdict}  (contact on {contact_steps} of {step} steps)")


if __name__ == "__main__":
    main()
