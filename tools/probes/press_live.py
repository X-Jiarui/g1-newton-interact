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
ap.add_argument("--approach-s", type=float, default=3.0, help="seconds of descent")
ap.add_argument("--hold-s", type=float, default=5.0, help="seconds held down afterwards")
ap.add_argument("--settle-s", type=float, default=1.5,
                help="seconds before the press, so the block lands on the table and the robot "
                     "stands: mjlab's reset event is what puts the table under the block")
ap.add_argument("--ik-iters", type=int, default=8, help="DLS corrections per control step")
ap.add_argument("--damping", type=float, default=0.05, help="damped-least-squares lambda")
ap.add_argument("--once", action="store_true", help="stop after one press instead of looping")
A = ap.parse_args()

ARM = ("right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
       "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
       "right_wrist_yaw_joint")


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
        self.arm_j = [j for j in range(m.njnt)
                      if any((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").endswith(x)
                             for x in ARM)]
        if len(self.arm_j) != len(ARM):
            raise SystemExit(f"matched {len(self.arm_j)} of {len(ARM)} arm joints")
        self.arm_q = np.array([int(m.jnt_qposadr[j]) for j in self.arm_j])
        self.arm_v = np.array([int(m.jnt_dofadr[j]) for j in self.arm_j])
        # TWO actuators per joint here as well, not just on the fingers: mjlab adds its own servo
        # and leaves the scene's `xml_motor_unused_*` one in place. Both must be commanded, or the
        # leftover one holds the joint at its own target and fights the press.
        slot = {j: i for i, j in enumerate(self.arm_j)}
        self.arm_a = [a for a in range(m.nu) if rig.jnt_of_act[a] in slot]
        self.arm_a_slot = np.array([slot[rig.jnt_of_act[a]] for a in self.arm_a])
        self.tip_body = sorted(rig.tip_bodies)[0]
        self.cpu = mujoco.MjData(m)

    def tip_world(self, q):
        import mujoco
        self.cpu.qpos[:] = q
        mujoco.mj_forward(self.rig.m, self.cpu)
        return self.cpu.xpos[self.tip_body].copy()

    def solve(self, q, target):
        """Damped least squares on the arm chain only. Position task only: adding orientation made
        the solver swing 1.26 rad to move the tip 16 mm."""
        import mujoco
        q = q.copy()
        for _ in range(A.ik_iters):
            self.cpu.qpos[:] = q
            mujoco.mj_forward(self.rig.m, self.cpu)
            err = target - self.cpu.xpos[self.tip_body]
            if np.linalg.norm(err) < 1e-4:
                break
            jacp = np.zeros((3, self.rig.m.nv))
            mujoco.mj_jacBody(self.rig.m, self.cpu, jacp, None, self.tip_body)
            J = jacp[:, self.arm_v]
            dq = J.T @ np.linalg.solve(J @ J.T + (A.damping ** 2) * np.eye(3), err)
            q[self.arm_q] += dq
        return q[self.arm_q].copy(), float(np.linalg.norm(err))


def main():
    env = build_env(A.port)
    rig = B.Rig(env)
    B.apply_setting(rig, A.settings)
    press = Press(rig)
    m = rig.m
    dt = env.physics_dt * env.decimation
    print(f"[press-live] dt {1000*env.physics_dt:.2f} ms x decimation {env.decimation} "
          f"= {1000*dt:.1f} ms per control step; setting '{A.settings}'")

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

    # 1. Settle. The table only arrives under the block when mjlab's reset event runs, and the robot
    #    needs a moment to stand. Everything holds its reset pose.
    rig.snapshot(freeze_hold=True)
    hold = rig.target_from_qpos()
    advance(hold, int(A.settle_s / dt))

    # 2. Plan, from where things actually ended up.
    q = B.sq(rig.qpos()).copy()
    obj_c = B.sq(rig.d.xpos)[rig.obj_body].copy()
    top = obj_c + np.array([0.0, 0.0, float(rig.h[2])])
    tip = press.tip_world(q)
    start = top + np.array([0.0, 0.0, A.standoff])
    goal = top - np.array([0.0, 0.0, A.through])
    pen0, _ = rig.depth_into_cube_mm(obj_c, rig.press_geoms)
    print(f"[press-live] block centre {np.round(obj_c,4)}  top face z {top[2]:.4f}")
    print(f"[press-live] fingertip now {np.round(tip,4)}  ->  start {np.round(start,4)}  "
          f"->  commanded {np.round(goal,4)}  ({1000*A.through:.0f} mm below the face, on purpose)")
    print(f"[press-live] step-0 overlap {pen0:+.3f} mm "
          f"({'CLEAR' if pen0 <= 0.0 else 'ALREADY INSIDE -- the run would be meaningless'})")

    n_down, n_hold = int(A.approach_s / dt), int(A.hold_s / dt)
    worst, contact_steps, step = -1e9, 0, 0
    t0 = 0.0

    while env.viewer is None or env.viewer.is_running():
        phase = step % (n_down + n_hold)
        frac = min(1.0, phase / max(1, n_down))
        target = start + (goal - start) * frac

        q = B.sq(rig.qpos()).copy()
        arm_target, err = press.solve(q, target)
        cmd = hold.copy()
        cmd[press.arm_a] = arm_target[press.arm_a_slot]
        advance(cmd)

        obj_c = B.sq(rig.d.xpos)[rig.obj_body].copy()
        pen, _who = rig.depth_into_cube_mm(obj_c, rig.press_geoms)
        keep, _rows, _n = rig.hand_object()
        fn = float(sum(c["force"] for c in keep)) if keep else 0.0
        worst = max(worst, pen)
        contact_steps += 1 if keep else 0

        if env.viewer is not None:
            env.viewer.begin_frame(t0)
            env.viewer.log_state(env.state_in)
            for name, val in (("penetration_mm", pen),
                              ("hand_object_contacts", len(keep)),
                              ("normal_force_N", fn),
                              ("commanded_depth_mm", 1000.0 * (top[2] - target[2])),
                              ("worst_penetration_mm", worst),
                              ("block_moved_mm", 1000.0 * float(np.linalg.norm(obj_c - top +
                                                np.array([0, 0, float(rig.h[2])]))))):
                env.viewer.log_array(name, np.array([val], dtype=np.float32))
            env.viewer.end_frame()
        t0 += dt

        if step % 20 == 0:
            print(f"[press-live] {step:5d}  commanded {1000*(top[2]-target[2]):+7.2f} mm  "
                  f"into block {pen:+7.3f} mm  worst {worst:+7.3f}  contacts {len(keep):3d}  "
                  f"Fn {fn:9.2f} N  ik err {1000*err:6.2f} mm", flush=True)
        step += 1
        if A.once and step >= n_down + n_hold:
            break

    verdict = ("INVALID -- nothing ever touched, so the penetration number means nothing"
               if contact_steps < 50 else f"worst penetration {worst:+.3f} mm over {step} steps")
    print(f"[press-live] {verdict}  (contact on {contact_steps} of {step} steps)")


main()
