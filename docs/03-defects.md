# The nine defects, and the measurement that found each

Every one of these is silent. None raised an exception, none printed a warning that anybody would
read as an error, and the model comparisons kept saying the two simulators matched. What they
produced instead was a policy that looked merely worse — the hand reaching to 0.2 m and stopping,
the apple sinking through a table, a robot that stood up fine and simply failed at the task.

That is the failure mode this port had to be built against, and it is why the observation and action
code is reused from mjlab rather than reimplemented. The list is ordered as encountered.

---

## 1. Contact needed 10 cm of penetration before it existed

`ModelBuilder.default_shape_cfg.gap` defaults to `0.1`. The MJCF specifies `margin = 0`, and MuJoCo's
`gap` is the width of the band inside `margin` in which contacts are *detected but generate no force*.
With gap at 0.1 m, two geoms had to overlap by 10 cm before the solver would push them apart.

**Presented as:** the hand passing through the apple; the apple resting inside the table.
**Found by:** comparing `geom_gap` field-for-field against the compiled mjlab model.
**Fix:** `builder.default_shape_cfg.gap = 0.0` before `add_mjcf`.

## 2. Free-joint damping dropped, and the obvious fix silently does nothing

Newton's conversion does not carry `dof_damping` for free joints. Worse, assigning it on the builder
before `finalize()` has no effect — the value is regenerated — so the natural fix reads as applied
and is not.

**Presented as:** the robot and object drifting more than they should.
**Found by:** field comparison, then re-checking after the "fix" and finding the value unchanged.
**Fix:** assign `dof_damping` on the warp model *after* the solver is constructed.

## 3. Plane geoms resized

Two plane geoms came through with sizes that differ from the compiled model.

**Presented as:** nothing visible. Planes are infinite in MuJoCo regardless of `geom_size`, so this
one is cosmetic — recorded because it was found by the same sweep and because a reader comparing
field dumps will see it.
**Fix:** copy `geom_size` for plane geoms from the reference model.

## 4. The constraint buffer overflowed and dropped contacts

Newton sizes `njmax`/`nconmax` from the state at construction time. In this scene almost nothing is
in contact at construction, so it chose a budget that overflowed as soon as the hand and table met:

```
nefc overflow - please increase njmax to 65
```

An overflowing solver does not fail — it discards constraints. The apple fell through the table it
had been resting on.

**Presented as:** the object dropping 82 cm to the floor around step 200.
**Found by:** the warning, which scrolls past in a log nobody reads, next to a symptom that looks
like bad physics rather than a budget.
**Fix:** `njmax=2048, nconmax=256`, matching mjlab's `SimulationCfg`.

## 5. Writing qpos does not move anything

`xpos` and `xquat` are outputs of forward kinematics. After a reset — or any direct joint/root write —
they still describe the previous pose, and the observation builders read the derived fields, not
qpos.

**Presented as:** `joint_pos` matching mjlab exactly while `root_link_pos_w` was off by 1.7 cm and
the object's orientation was flatly wrong (quaternion diff 2.0). The policy opened on a world that
did not exist.
**Found by:** diffing state field-by-field after the reset, where joint values agreed and derived
values did not — a pattern that names its own cause.
**Fix:** `NewtonEnv.forward()` (mujoco_warp `fwd_position` + `fwd_velocity`) after the reset. Inside
the rollout loop a `solver.step()` follows every write and does this implicitly.

## 6. Root angular velocity written in the wrong frame

mjlab's 13-element root state is entirely world-frame. MuJoCo's free joint stores linear velocity in
the world frame but **angular velocity in the body frame**, and `Entity.write_root_velocity` converts
before writing. The bridge wrote the world-frame value straight through.

**Presented as:** 0.34 rad/s of error that landed on exactly three observation slots — `astra_obs`
0–2, the ASTRA tracker's gyro input — and on nothing else. Every other group looked perfect.
**Found by:** a per-group observation diff, then a per-slot diff of the one group that disagreed.
**Fix:** `quat_apply_inverse(quat_w, ang_vel_w)` before writing qvel, as mjlab does.

## 7. `apply_actions` belongs inside the decimation loop

mjlab's `step()` is:

```python
for _ in range(decimation):
    self.action_manager.apply_action()
    self.scene.write_data_to_sim()
    self.sim.step()
    self.scene.update(dt=self.physics_dt)
self.episode_length_buf += 1
```

`apply_action` runs four times per control step, not once. And it is not just a ctrl write:
`Sonic53Action` re-applies the startup root/joint hold, the table pose and the reference object
tracking on every call.

**Presented as:** all of that running at a quarter of its intended rate.
**Found by:** reading mjlab's step loop instead of assuming the conventional once-per-control-step
placement.
**Fix:** call `apply_actions()` inside the substep loop.

## 8. The policy wrote its per-step state onto the wrong environment — *the one that mattered*

`rl.py`'s `_set_residual_action_stats` records the tracker's output by setting attributes on
`self.env.unwrapped` — the env the **runner was constructed with**. That has to be the mjlab env,
because the runner needs one to resolve observation and action dimensions.

Three observation groups read those attributes back, and mdp.py's readers fall back silently:

```python
def _last_residual_action(env):
    value = getattr(env, "_residual_last_residual_action", None)
    if value is None:
        return _zeros(env, ACTION_DIM)      # ← Newton took this path, every step
```

So `tracker_action`, `last_residual` and `astra_obs` were computed from `teacher_action` instead of
the tracker's actual output. The values themselves were correct — the policy had been fed Newton's
observations to produce them — they simply landed on an object nothing read.

**Presented as:** the hand reaching to 0.20 m and retreating. A working policy, acting on a world it
was not in.
**Found by:** printing the diff for **every** observation group rather than the worst one. The
pattern was unmistakable once laid out:

```
reference_phase      0
object_future        3.9e-06
contact_features     7.4e-05
object_state         0.000213      ... everything at ~1e-4
─────────────────────────────
tracker_action       4.21
last_residual        4.21
astra_obs            4.24          ... and three groups, all tracker-related
```

**Fix:** copy the eight `_residual_last_*` attributes onto the Newton env after each policy call.
**This was the defect that decided the outcome**: fixing it took the rollout from 0.196 m and no
lift to 50.6 cm and a held grasp.

## 9. Newton's reconstructed `MjModel` mislabels which bodies are "simple"

Newton does not compile the MJCF into an `MjModel`; it reconstructs one from its own `Model`. That
reconstruction sets `body_simple = 0` for `apple/apple` and `table/table` — the scene's only two free
bodies — and `dof_simplenum = 0` for the apple's six free dofs where MuJoCo gives `6,5,4,3,2,1`.

The consequence is `nC`, the size of MuJoCo's compressed mass-matrix layout:

```
compiled MjModel :  nM = 1102,  nC = 1087
Newton rebuilt   :  nM = 1102,  nC = 1102
```

Same tree, same `nM`, different sparse structure to solve in.

**Presented as:** smooth dynamics agreeing to 1e-5 while `qfrc_constraint` differed by 2.3e-3 from
the first free-dynamics step.
**Found by:** copying *every* shared `mjw_data` array from mjlab into Newton, stepping once, and
diffing the force terms separately. The arrays that could **not** be copied named the cause: `M` was
`(1,1087)` on one side and `(1,1102)` on the other.
**Invisible to:** every comparison of the compiled models. They agree on `nM`, on the tree, on
`dof_parentid`, on every per-body field.
**Fix:** build the warp model from a compiled `MjModel`. Patching `body_simple` on the reconstruction
does not work — `nC` and its index arrays (`M_rowadr`, `M_colind`, `mapM2M`) are computed by MuJoCo's
compiler and stored on the model, so `put_model` copies the stale value through.

Worth reporting upstream to Newton.

---

## Not a defect: `ls_parallel`

mjlab sets `opt.ls_parallel = True`; mujoco_warp removed the option in 3.9.1 and now raises from a
property that exists only to produce that error. `src/mjw_compat.py` swallows the write **and records
it**, so the difference stays visible instead of hiding inside a compatibility layer. Measured cost:
the 3.8 → 3.11 control run lifts 50.13 cm against 49.72 cm, so the removed parallel line search is
worth about 0.4 cm here.

## What generalises

Three habits caught seven of the nine:

- **Diff every element, not the aggregate.** Defect 8 was invisible in "worst group = 4.2" and obvious
  in the full table, because the *pattern* — three tracker groups high, everything else at 1e-4 —
  is what named the cause.
- **Compare the thing that runs.** The compiled models agreed on everything while the warp models did
  not (defect 9), and the warp model is what the kernels read.
- **Make the adapter raise on anything unmapped.** `_EntityData.__getattr__` raises rather than
  returning `None`, which is how defect 6 surfaced as an error instead of as a plausible wrong number.
