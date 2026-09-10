# Training parameters

Every knob that changes what the policy trains against, what it is scored on, or what the physics
does. Written because the run identity used to live in a wall of `env VAR=...` prefixes that no two
launches ever spelled the same way — see `configs/train/best.yaml` and `--config`.

Each row carries an **evidence** column, because a surprising number of these were set by copying
another example and have never been varied:

| mark | meaning |
|---|---|
| **measured** | varied in a controlled run, effect quantified |
| **settled** | measured and the decision is not open |
| **retracted** | a previous conclusion about it was withdrawn; the current value is a fallback |
| **untested** | inherited or guessed; never varied |

Machine-local values (`CUDA_VISIBLE_DEVICES`, `GMR_ROOT`, the data root) are deliberately kept out
of configs.

---

## 1. Collision and contact

| knob | current | evidence | notes |
|---|---|---|---|
| object collider mesh | `cubesmall_box.stl`, **12 triangles** | **settled** | The GRAB mesh is a flat-faced box tessellated into **107,776** triangles. It overflowed the triangle-pair buffer by up to **123x** in every run ever measured, silently dropping most contact candidates. Swapping it: overflow gone, mean penetration 0.0046→0.0006 mm, contacts detected **up 2x**, iteration time halved. Geometry is bit-identical. |
| `MAX_TRI_PAIRS` | 12,000,000 (default) | **settled** | Raised from Newton's 1e6 once, which covered 0.8% of the real need. With the 12-triangle cube the requirement drops ~9000x and the buffer is never touched. Check `grep -c "Triangle pair buffer overflowed"` before trusting any contact number. |
| `--native-contacts` | on | **measured** | Newton's `CollisionPipeline` for detection, mjWarp for response. MuJoCo's own narrow phase on the same mesh gives 4.734 mm. Newton's own Allegro example uses the same split. |
| `SHAPE_MARGIN` | **absent** | **retracted** | Previously called "the largest lever" for penetration. It is not. At 0.01 it fires contacts while geometries are up to 10 mm apart; penetration is `(-dist).clamp(min=0)` so those score exactly 0.0000 mm while carrying 125 N. Three arms ran with it and reported perfect penetration; lift_success was 0.1134 / 0.0017 / 0.0000 against 0.7794 without it. It is a force field holding the hand off the object. |
| `SHAPE_GAP` | 0.01 | **untested** | Broad-phase AABB expansion, same mechanism as margin but never varied on its own since the mesh fix. Worth a controlled run. |
| `HAND_SOLREF` | `0.004,1.0` | **measured** | Must be set: equal-priority geoms have their solref averaged, so leaving the hand at the default let MuJoCo average away the object's stiff setting. |
| `--object-solref` | `0.004,1.0` | **measured** | Resting penetration 0.04 mm (stapler) against 1.88 mm at the shared default. `timeconst` is clamped to `2*dt` by REFSAFE; the negative `(-stiffness,-damping)` form is unclamped but violently unstable. |
| `OBJECT_PRIORITY` | off | **measured** | Takes 9.8 mm to 1.6 mm but collapses contact on soft-trained checkpoints. Not used. |
| `SIM_TIMESTEP` | 0.002 | **measured** | Control rate stays 50 Hz; decimation absorbs the change. Contact stiffness is timestep-bound — at 2 ms no contact parameter buys 2x, while halving dt bought 51x. Cost: dt 2 ms runs ~1.9x slower than 5 ms. A clean no-margin 2 ms vs 5 ms pair is running now (`H9_DT2` / `H9_DT5`); before it, every 5 ms run also carried margin. |
| `NCONMAX` | 512 | **measured** | 1024 and 2048 gave identical results. An earlier "~1000 contacts dropped per step" claim was retracted — that figure described a hydroelastic config. |
| solver `impratio` / `cone` / `iterations` / `ls_iterations` | 20.0 / pyramidal / 100 / 50 | **untested** | Copied from the hydroelastic example. Note the cone choice changes how contact force must be read: under a pyramidal cone the normal force is the **sum across `efc_address` rows**, not `efc_force[adr]`. |
| `HAND_FRICTION` / `OBJECT_FRICTION` | mu 0.6 (scene XML) | **untested** | An earlier sweep set only `OBJECT_FRICTION` and concluded "friction does not matter" from a sweep in which friction never changed — the hand's value won. The solver's own `contact.friction` is what settles it. |
| `HAND_COLLISION_FIX` | on | **settled** | wrist↔palm is a real Newton bug: 7200 contacts and −9 mm weld overlap without it. |
| `LINK2_ISOLATE` | off | **retracted** | MuJoCo contype masks do nothing here — Newton's broad phase does not read them. The Newton-side filter over-matched (`"link2" in label` also caught link3/4 through ancestor paths) and drove contact to 0.0000, which reads like a triumph and is a dead run. |
| convex hull of robot meshes | on | **settled** | 65 meshes per world, up to 25,662 verts on one torso shape. MuJoCo convexifies every mesh geom for collision anyway, so this is parity, not a loss of fidelity. The object and table are excluded **by name**; keying off the hydroelastic flag silently swept them in when `--rigid-object-table` was passed. |

## 2. Actuation

| knob | current | evidence | notes |
|---|---|---|---|
| `FINGER_FORCE_LIMIT` | **1.0 N·m** | **measured** | The Wuji spec in the scene XML is **0.15–0.62 N·m** per finger joint; mjlab's actuator overrides it with **30 N·m** — 48–200x over spec. Rollouts of the V4 checkpoint: 30 → 1.0 N·m took mean penetration 0.0492 → 0.0131 mm, max 3.57 → 1.98 mm, the >1 mm contact fraction down **173x**, and lift did not drop. At 0.6 N·m: 0.0101 mm mean, 1.42 mm max, >1 mm down 650x. 1.0 N·m is 33 N at a 30 mm lever against the 4.2 N needed to hold the object. |
| `APPLE_HAND_EFFORT` | **do not use** | **settled** | Edits the actuator cfg, and the cfg value never reaches the compiled model — the entity spec is cached, so a run prints the new limit and simulates the old one. A guard now raises and points at `FINGER_FORCE_LIMIT`. |
| `NEUTRALISE_LEFTOVER` | on | **measured** | Every finger joint carries **two** actuators: mjlab's and the scene's own `xml_motor_unused_*`, which mjlab renames but never removes and never writes. At ctrl 0 they are position servos pulling the finger **open** — 12.384 N·m of budget across 40 actuators, mean 0.157 N·m. Against 30 N·m that is noise (measured: A ≈ B in the effort ladder). Against a 0.6–1.0 N·m cap it is a quarter of the budget, so it must be off for any finger-torque experiment. |
| `HAND_PD` (`"kp,kd"`) | **10 / 0.2** | **measured** | Writes `gainprm[0]`, `biasprm[1] = -kp`, `biasprm[2] = -kd` into both `mj_model` and `mjw_model`, then reads back and raises if it did not take. Four-point sweep off `MIX8_pinned.pt`, mean penetration (mm): kp 300 → 0.0040–0.0041, kp 50 → **0.0074–0.0075 (1.83x worse)**, kp 10 → 0.0045–0.0046 (1.12x), kp 0.5 (VIRAL) → launched and retired without a verdict. **Non-monotonic: softening the hand does not reduce penetration**, and contact was flat at 0.47–0.49 across every arm. Settled on 10 / 0.2 because it is CoorDex's published G1+Wuji number and is exactly critically damped here (ζ = 1.0000, f_n = 35.6 Hz on armature 2.0e-4), not because it won on penetration — the 1.12x it costs is inside the arm-to-arm noise this metric shows. |
| `APPLE_HAND_KP` / `KD` | **do not use** | **settled** | Same cached-entity-spec defect as `APPLE_HAND_EFFORT`: `APPLE_HAND_KP=10` still compiles to kp 300.0. Three arms of a stiffness sweep were burned on identical hands before a compiled-gain print caught it. `[newton-env] compiled hand actuators: n= kp= kd= effort=` now prints on every run — read it, do not trust the cfg. |
| `CLAMP_TARGETS` | off (report only) | **open** | The target guard measures commands **+0.88 rad (+50.4°) past the actuator's saturation range**, consistently. Joint limits then carry the load, and a limit constraint has no torque ceiling. Turning the clamp on is untested. |
| `--effortless-action` | off | **settled** | `set_joint_effort_target` is a verified no-op here; the robot is a position servo. |

## 3. Object and scene

| knob | current | evidence |
|---|---|---|
| `APPLE_HAND_KIND` | `wuji` | **settled** — the Wuji retarget grasps better than xhand's |
| `APPLE_OBJECT_PER_WORLD` | 1 | **settled** |
| `APPLE_SCENE_Z_OFFSET` | −0.03 | **untested** |
| `--rigid-object-table` | on | **settled** — hydroelastic contact is off; both meshes are plain colliders |
| `--table-under-object` | on | **settled** — without the reset event the table sits under the robot's feet; three probes were wrong because of it |
| `--sdf-resolution` | 128 | **untested** |
| `APPLE_OBJECT_HULL` / `_COLLISION` / `_BOX_GROUP` | default | **measured elsewhere** — box primitives beat mesh hulls on worst-case penetration; the stapler's 5-box collider bought its first non-zero lift |

## 4. Reset, curriculum, episode

| knob | current | evidence | notes |
|---|---|---|---|
| `RSI_ANCHOR_CF` | 1 | **settled** | Start relative to the contact frame. A policy trained this way never sees frames 0..cf−20, so a frame-0 rollout scores it out of its own distribution — use `ROLLOUT_START_FRAME=rsi`. |
| `RSI_CF_OFFSET_START` / `_END` | −20 / −10 | **untested** | |
| `TABLE_REMOVE_AFTER_CF` | 50 (1.0 s at 50 Hz) | **measured** | 1.0 s of margin after cf bought the first sustained lift. |
| `TABLE_REMOVE_DROP` | 0.30 | **untested** | |
| `OBJ_REF_STILL_UNTIL_CF` | 1 | **untested** | |
| `OBJ_REF_WINDOW_ENABLE` | 0 | **untested** | |
| `TIP_CF_MISS_ENABLE` | 0 | **untested** | |
| `MIX_PMCP_*` | off (`MIX_PMCP_EVERY=0`) | n/a | Multi-clip curriculum; single-clip runs leave it off. If enabled, the env→clip map must be **read**, not recomputed — recomputing it collapsed ep_len from 107 to 8. |

## 5. RL / runner

| knob | current | evidence |
|---|---|---|
| `--num-envs` | 1024 | **measured** — 4096 and 8192 both die with `cudaErrorIllegalAddress`, alone on an idle GPU. Real ceiling, not contention. |
| steps per env per iteration | 128 (from the runner) | fixed |
| `--seed` | 1 | Rollouts vary **6–14%**; never judge a change from one seed. |
| `--iterations` | 12000 | |
| `--resume` | `model_500.pt` | **measured** — chosen because at iteration 500 penetration was 1.22 mm with lift 0.0000, while by 1000 it was 1.70 mm with lift 0.2411: the policy learns to grasp *by* penetrating, so resuming later imports that habit. |
| `--reward-cfg` | `staged_cf_r24.yaml` | |
| `RESIDUAL_SAFE_PPO_LOG_RATIO_CLIP` | default | **untested** |
| action clip / exploration std | in the agent cfg | **settled** — two ceilings (clip 6x too small, then std 70x too small) once made every run match its own untrained `model_0` |

## 6. Instrumentation — not training parameters

`PEN_LOG`, `PEN_LOG_EVERY`, `PEN_PROBE`, `CONTACT_CENSUS`, `CENSUS_*`, `GRASP_FORCE_PROBE`,
`PRESS_*`, `NCON_PROBE`, `SENSOR_PROBE`, `HULL_PROBE`, `CPARAM_PROBE`, `SDF_CHECK`, `MIX_VERIFY`,
`MIX_FAR_*`, `VISER_COLLIDERS`, `ROLLOUT_START_FRAME`, `--viser-port`, `--render-every`,
`--newton-video`, `--dump-qpos`, `--profile-step`, `--state-digest`.

`PEN_LOG=1` is kept on in the pinned config: it is the only continuous read on penetration.

**`VISER_COLLIDERS=1`** draws the shapes the solver actually collides instead of the render meshes.
The hand's colliders are convex hulls and a hull is fatter than the mesh drawn over it, so judging
penetration from the default view reports a number the physics never computed.

## 7. Reading the numbers

- **`Episode_Metrics/*` from `--rollout-steps` are not comparable to the training row.** The
  rollout divides by total steps while these are logged only on steps where an episode ended, so
  arms with more episode ends read higher across the board. Measured: the same checkpoint under its
  own training config gives 0.1170 in the rollout against 0.9953 in training.
- **`lift_success` is RSI-averaged.** Name the start-frame distribution when quoting it.
- **Contact force under a pyramidal cone is the sum across the pyramid rows.** Reading
  `efc_force[adr]` returns one edge — 0.00 N next to a true 15.76 N — and once produced a 654 N
  fiction that survived a whole round of conclusions.
- **`[pen-compliance]` (mm/N) is the confound-free metric** for physics changes: a grasp that
  merely got worse shrinks numerator and denominator together. It is **not** valid for
  force-capping runs, where the checks are contact count and whether deep force clamps at the new
  ceiling.
