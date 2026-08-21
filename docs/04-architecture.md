# How the port is put together

## The decision that shapes everything

The residual policy reads **1328 dims across 20 observation groups**, assembled by mjlab's
`ResidualFeatureGroupObs` and `AstraObs136`. Its action term, `Sonic53Action`, does considerably more
than emit ctrl: during the first 30 steps it writes root and joint state directly, through step 36 it
overrides targets with the reference pose, and on every call it drives the table pose and the
reference object tracking.

None of that is reimplemented here. mjlab's own code runs, and an adapter gives it Newton state to
read and write.

The reason is in [03-defects.md](03-defects.md): a missing or stale observation group does not raise.
It produces a policy that looks a bit worse. Defect 8 cost 50 cm of lift and presented as "the hand
can't quite reach" — and that was *one* group being stale out of twenty. Re-deriving all 1328 dims
would be twenty more chances at the same thing.

What makes the adapter thin rather than a rewrite: Newton's `SolverMuJoCo` runs **mujoco_warp**
underneath — the same engine mjlab uses — and the two models were verified identical in joint, body
and actuator ordering. So `mjw_data` has mjlab's exact layout (qpos 83, qvel 81, ctrl 138, xpos/xquat
92) and the index maps are shared.

## Reading `mjw_data` sidesteps Newton's conventions entirely

Newton's `Model`/`State` API stores quaternions xyzw, reports linear velocity at the centre of mass
and angular velocity in the world frame. MuJoCo uses wxyz, the body-frame origin, and body-frame
angular velocity. That table of differences is real — and it does not apply here, because `mjw_data`
is MuJoCo's own structure and still carries MuJoCo's conventions.

Warp arrays are exposed through `wp.to_torch`, which shares memory rather than copying, so reads are
views on live simulator state exactly as they are in mjlab.

The one place a frame conversion is still needed is writing the root state, because MuJoCo's free
joint mixes frames internally (defect 6).

## The bridge surface

`src/newton_bridge.py` implements only what was measured to be used:

```
NewtonEnv          num_envs, device, scene, episode_length_buf, step_dt, physics_dt,
                   action_manager, extras, forward()
NewtonEntity       data.{joint_pos, joint_vel, root_link_*, body_link_*, projected_gravity_b, ...}
                   find_bodies / find_joints / joint_names / body_names
                   set_joint_position_target, set_joint_effort_target,
                   write_joint_state_to_sim, write_root_state_to_sim, write_root_pose_to_sim,
                   write_mocap_pose_to_sim, write_external_wrench_to_sim, is_mocap
_ActionManagerView action, prev_action, get_term, advance
```

Two design choices are load-bearing:

**`_EntityData.__getattr__` raises.** Anything not explicitly mapped is a gap in the adapter, not a
zero. Returning `None` or falling through to the IsaacLab/Newton spelling would put wrong numbers
into the observation silently. This is how defect 6 surfaced as an `AttributeError` naming the field
instead of as a plausible-looking velocity.

**`set_joint_effort_target` is a deliberate no-op.** Measured in mjlab with the trained policy: ctrl
on every `xml_motor_unused_*` actuator stays exactly 0.0 and never varies, so that torque law reaches
no actuator that can produce force. Implementing it here would *add* a force mjlab does not apply.

## Control path

Measured, not assumed:

- `control.mujoco.ctrl` (138 entries) is the only entry point. Its indices are identity with the
  model's actuator order, which is identical to mjlab's.
- Writing `mjw_data.ctrl` directly does **not** work: `SolverMuJoCo.step` calls `_apply_mjc_control`
  every step and overwrites it (measured: 0.777 → 0.0 after one step).
- `control.joint_target_q` does **not** work either: these actuators are imported as MJCF general
  actuators using `CtrlSource.CTRL_DIRECT`, not `JOINT_TARGET`. Writing all 81 DOF slots reached
  0 of 138 actuators.
- `update_data_interval=0` pins `mjw_data` as the authority, so the direct joint/root writes
  `Sonic53Action` performs during the startup hold survive instead of being overwritten from Newton's
  `State` each step.

## The loop

```python
obs = {group: builder(nenv) for group in builders}      # mjlab's builders, Newton's state
action = policy(obs)                                     # mjlab's policy
sync_residual_stats()                                    # defect 8
action_term.process_actions(action)
for _ in range(4):                                       # decimation, defect 7
    action_term.apply_actions()
    solver.step(state_in, state_out, control, None, 0.005)
episode_length_buf += 1
```

The mjlab environment that exists in the same process is used **only** to construct the runner and
load the checkpoint — the runner needs an env to resolve observation and action dimensions. It is
never stepped. Every observation fed to the policy comes from Newton's state.

## Both libraries in one process

mjlab pins `mujoco-warp>=3.8.0` and Newton 1.5 pins 3.11, so they coexist: mjlab is installed into
the Newton environment and both run on mujoco_warp 3.11. `src/mjw_compat.py` patches the *new* library
to tolerate the option mjlab's 3.8-era code still sets, never mjlab itself — mjlab is the baseline,
and editing it would mean the reference and the port stopped running identical code.

This also removes the version confound from every comparison, and makes the control experiment
possible: same code, same checkpoint, only the mujoco_warp version changed → 0.4 cm.

## Verification ladder

Each rung isolates a different class of error, and each was needed:

| tool | question |
|---|---|
| `tools/verify/compare_facts.py` | do the compiled models agree field for field? |
| `tools/verify/compare_obs.py` | do all 20 observation groups agree from the same reset state? |
| `tools/verify/chain.py` | obs → policy output → ctrl → one substep, each link separately |
| `tools/verify/lockstep.py` | driven by identical actions, where do the trajectories part? |
| `tools/eval/trace_eval.py` | does mjlab still do what the baseline says? |

`chain.py` is the one that found defects 8 and 9, because it tests the links **separately** instead of
comparing end states. A rollout that ends differently tells you nothing about which link broke.
