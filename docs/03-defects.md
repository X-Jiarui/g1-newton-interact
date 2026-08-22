# The twelve defects, and the measurement that found each

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

## 10. Newton drops every `<sensor>`, and the rewards that read them return 0.0 instead of failing

`import_mjcf.py` and `solver_mujoco.py` contain no handling of `<sensor>` at all, so every scene
Newton compiles has `nsensor=0`. Nothing errors. The three reward terms that gate on tip contact —
`multi_tip_surface` (weight 5.0), `contact_duration` (1.0), `object_hard_lift` (2.0) — simply return
0.0 on every step of every run, and PPO happily optimises the remaining terms.

**The measurement.** Running mjlab and Newton on the same clip with the same config and printing
`nsensor`: mjlab 140, Newton 0. This defect is a repeat of one already recorded on the mjlab side
(geomless tip bodies made contact-gated rewards silently zero for 2900 iterations); the same class
of bug survived the port because the failure mode is a zero, not an exception.

**The fix.** `src/newton_sensors.py` copies all 140 sensors from the authored scene into Newton's own
`MjSpec` just before it compiles, remapping body names by longest-suffix match. Newton's importer
also drops the `<site>` elements the 4 IMU sensors mount on, so those sites are recreated on the
matching body — pure geometry, no change to the kinematic tree.

**Parity after the fix**, same clip, 150 steps, zero residual:

| | mjlab | Newton |
|---|---|---|
| `nsensor` / contact / `nsensordata` | 140 / 136 / 294 | 140 / 136 / 294 |
| sensordata slots ever nonzero | 16 / 294 | 16 / 294 |
| peak contact magnitude | 3.23 | 8.83 |

The magnitudes differ because Newton collides against the exact SDF of the mesh where mjlab uses a
convex hull — that difference is the point of the port, not an error.

**A correction to an earlier reading.** `object_hard_lift` sitting at 0.0 was listed as evidence the
sensors were missing. It is not: the mjlab control holds it at exactly 0.0 through 163 iterations
too, because nothing lifts the stapler that early. Only `multi_tip_surface` (0.0023 at iteration 0
in mjlab, rising to 3.09) is a usable early signal that contact sensing is live.

**Guardrails.** `NewtonVecEnv` now prints `nsensor`/contact/`nsensordata` at construction on every
run, and both training entries take `--sensor-probe N`, which steps N times with zero residual and
reports how many sensordata slots ever read nonzero. An `nsensor=0` regression is now loud. It
caught one within the hour: a bad variable name in this very fix.

## 11. Every reward term matched, and the total was 49x too small

`RewardManager` was constructed as `RewardManager(cfg.rewards, env)`, which takes the constructor
default `scale_by_dt=True`. mjlab passes it explicitly -- `manager_based_rl_env.py:330` reads
`scale_by_dt=self.cfg.scale_rewards_by_dt` -- and this task sets that flag `False`. So every Newton
reward was multiplied by an extra `dt = 0.02`.

**Why it survived a term-by-term parity check.** A uniform scale is invisible term by term. The
measurement that caught it evaluated each active term on both backends at the same state *and*
printed what `compute()` returned:

| term | mjlab | Newton |
|---|---|---|
| `tracking` | +0.001077 | +0.001163 |
| `left_wrist_tracking` | +0.001197 | +0.001101 |
| `right_wrist_tracking` | +0.000131 | +0.000232 |
| `omnigrasp_style` | +0.012214 | +0.012218 |
| **`compute()` returned** | **0.726881** | **0.014713** |

Four significant figures of agreement on every term, and a 49.7x gap in the total -- and 49.7 is
1/0.02. The ratio named the bug.

## 12. `_reset_idx` called no manager's `reset()`

mjlab's reset runs `reset(env_ids)` on every manager, in an order its source marks as sensitive.
The Newton env ran none of them. The visible symptom was a missing `Episode_Reward/*` log group,
which is the least of it:

- `reward_manager.reset` zeroes the per-episode sums,
- `metrics_manager.reset` clears episodic accumulators such as `lift_success`,
- `action_manager.reset` drops the previous action that the two action-history observation groups
  are built from -- so a new episode began by observing the end of the old one.

All of that leaked across every episode boundary of every run. The bridge's `_ActionManagerView`
gained a matching `reset()`; the task has no curriculum or command manager (both cfg dicts are
empty), so mjlab's calls to those have no counterpart.

**Parity after 11 and 12**, iteration 40, same clip, 128 envs, same analytic-sphere object:

| | mjlab | Newton |
|---|---|---|
| `Train/mean_reward` | 53.05 | 49.64 |
| episode length | 98.90 | 98.93 |
| `body_tracking` | 0.5983 | 0.6055 |
| min tip distance | 0.2671 | 0.2599 |
| `object_tracking_raw` | 0.1401 | 0.1445 |

**A correction.** Before these two fixes Newton read `mean_reward` 0.73 against mjlab's 53.05, and
its fingertips sat 15cm further from the object. That gap was attributed here to the object
representation -- Newton colliding against the exact mesh where mjlab uses a 4cm sphere. It was
not: the sphere-vs-mesh runs differ from each other by almost nothing, and the whole gap closed
when the reward scale and the resets were fixed.

**What generalises.** Both defects are the same shape as defect 10: a wrong value that is still a
*valid* value. Nothing raises. The guard that works is not a stricter type -- it is a control run
of the thing you are porting from, compared on the aggregate rather than the parts.

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
