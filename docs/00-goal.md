# Migrating the mjlab residual-grasp stack to Newton

## Acceptance criterion

Load the mjlab-trained checkpoint (`OF_00_apple_eat_1_SPHERE/model_2010.pt`) in Newton and get
behaviour consistent with mjlab.

## What "consistent" can and cannot mean

Bitwise parity is **not achievable**, and it is worth being precise about why rather than discovering
it as a failed assert later:

| | mjlab | Newton 1.5.0 |
|---|---|---|
| warp-lang | 1.12.0 | 1.16.0 |
| mujoco-warp | 3.8.0.2 | 3.11.0 |
| mujoco | 3.8.1 | 3.11.0 |

Newton's primary backend *is* mujoco_warp -- the same engine mjlab runs on -- which is why this port
is far more tractable than the IsaacLab/PhysX one. But Newton 1.5 pins mujoco_warp 3.11 against
mjlab's 3.8. Three minor versions of contact and solver kernels separate them, so identical inputs
will not produce identical trajectories, and a contact-rich grasp amplifies small differences fast.

Therefore the criterion is evaluated in two layers:

1. **Model fidelity** -- exact, and demanded exactly. The compiled MuJoCo model Newton produces must
   match mjlab's field for field (joint order, actuator gains, armature, contact parameters, solver
   options). This is a static comparison with no physics in it, so there is no tolerance to argue
   about. `SolverMuJoCo(save_to_mjcf=...)` makes Newton's model dumpable, and both sides are then
   compared as *compiled* models via `src/model_facts.py` -- not as XML, because the compiler is what
   resolves defaults and class inheritance, and two different XMLs can compile to identical physics.

2. **Behavioural parity** -- statistical, against a measured noise floor. Before attributing any
   divergence to Newton, run the *same* MJCF under plain MuJoCo 3.8 and 3.11 to measure what the
   version bump alone costs. Only divergence above that floor is Newton's doing. Task-level metrics
   (`lift_success`, `lift_duration_s`, object trajectory) are the verdict; per-step state equality is
   not.

## The four orderings (measured, not assumed)

The single highest-risk quantity in this port. A permutation raises nothing: the policy runs and
every command lands on the wrong joint. There are **four distinct orders**, and mjlab already
converts between them internally:

| order | length | what it indexes | source |
|---|---|---|---|
| joint order | 69 | `qpos`, all observations | `robot.joint_names` |
| action order | 69 | the policy's 69 outputs | `BODY_29_DOF_NAMES + HAND_*_DOF_NAMES` |
| reference/PKL order | 29 (body) | the retargeted clip | `IL_FOR_PKL` in `Sonic53Action` |
| ctrl/actuator order | 138 | `data.ctrl` | model actuator ids |

**Measured: action -> joint differs in 27 of 69 slots.** Example: `action[22]` drives
`right_shoulder_pitch_joint`, while `joint_order[22]` is `left_finger1_joint1`. Action order is
"29 body then 40 hand"; joint order interleaves the two hands. Recorded in `docs/ctrl_probe.json`
(`action_to_joint_index`) and `docs/action_mapping.json`.

## The actuator / PD definition (measured)

`nu = 138` for `njnt = 71` (69 hinges + 2 free joints: robot base and object). Every hinge carries
**two** actuators:

* `robot/<joint>` -- the real position servo. `gaintype=FIXED, biastype=AFFINE`, i.e.
  `gainprm[0]=kp, biasprm[1]=-kp, biasprm[2]=-kv`. That triple *is* the PD definition. Example:
  `left_hip_pitch` kp=40.18, kv=2.56, forcerange ±(25..50); the 40 hand joints use forcerange ±30.
* `robot/xml_motor_unused_<joint>` -- leftovers from the XML. **Measured with the trained policy
  running, ctrl for all 69 of these is exactly 0.0 and never varies**, which is also why
  `Sonic53Action`'s visible torque law is dead code: `set_joint_effort_target` reaches no actuator
  that can produce force. Only the 69 real servos carry ctrl (|ctrl|max ≈ 1.93, all 69 varying).

  Caveat still to confirm in physics: the 29 body leftovers have `forcerange=[0,0]` and are
  genuinely inert, but the 40 hand leftovers are AFFINE servos with `ctrllimited=1`,
  `ctrlrange=[0.0475, 1.603]` and `forcerange=±0.4452`. MuJoCo clamps ctrl into ctrlrange when
  computing actuator force, so a ctrl of 0 becomes an effective target of 0.0475 -- a passive
  restoring spring of kp≈0.41 on every finger joint, about 1.5% of the real actuator's authority.
  In a task where the thumb was measured carrying 1/300 of the load, that is not automatically
  negligible. **To verify by reading `actuator_force`, not by argument.**

## Solver options that must be set explicitly

mjlab's compiled model, versus what `SolverMuJoCo` would otherwise default to:

| option | mjlab | Newton default | action |
|---|---|---|---|
| timestep | 0.005 | -- | set |
| integrator | `implicitfast` (3) | `implicitfast` | agrees |
| solver | `newton` (2) | `newton` | agrees |
| cone | `pyramidal` (0) | `pyramidal` | agrees |
| **iterations** | **10** | **100** | **must override** |
| **ls_iterations** | **20** | **50** | **must override** |
| impratio | 1.0 | 1.0 | agrees |
| tolerance / ls_tolerance | 1e-8 / 0.01 | same | agrees |

The two iteration counts are the ones that silently change contact behaviour.

## Known conversion hazards in Newton's MJCF path

From Newton's own docs and importer source, the items that apply to this asset:

* **Quaternion order**: Newton is `xyzw`, MuJoCo is `wxyz`. Every root and object orientation in the
  observation passes through this.
* **Velocity reference point**: MuJoCo reports linear velocity at the body frame origin, Newton at
  the CoM; MuJoCo angular velocity is body-frame, Newton's is world-frame. `SolverMuJoCo` converts,
  but the observation builders must be fed the mjlab convention.
* **Geom classification**: Newton's importer keys colliders off MJCF *class names*
  (`collider_classes=("collision",)`, `no_class_as_colliders=True`), not geom groups. It does honour
  `contype`/`conaffinity`, and mjlab's model sets them properly -- 81 of 226 geoms collide, 145 are
  visual-only -- so this should classify correctly, **but it must be verified on our XML**, because
  misclassification would silently add collision geometry the boxstack work was tuned to avoid.
* **Structure-changing importer defaults** to leave alone: `collapse_fixed_joints=False`,
  `convert_3d_hinge_to_ball_joints=False`.
* Meshes are convex-hulled by MuJoCo's compiler. Our objects are analytic spheres and boxstacks, so
  this is mostly moot -- which is a second dividend from that work.

## Reference artifacts

| file | what |
|---|---|
| `docs/mjlab_facts.json` | the compiled mjlab model, field by field -- the baseline |
| `docs/mjlab_joint_order.json` | 69 hinge order + 69 real-actuator order |
| `docs/action_mapping.json` | joint / action / ctrl orders and the permutations |
| `docs/ctrl_probe.json` | measured ctrl per actuator class + `action_to_joint_index` |
| `src/model_facts.py` | the comparator, used on **both** sides |

## Stages

1. ~~Measure the mjlab baseline: model, orderings, PD, what reaches ctrl.~~ **done**
2. Import mjlab's MJCF into Newton; dump via `save_to_mjcf`; compile; diff fact-sheets; fix deltas
   until the model comparison is clean.
3. Measure the version noise floor (same MJCF, MuJoCo 3.8 vs 3.11).
4. Port the observation assembly, reusing mjlab's own builders through an adapter rather than
   re-deriving 1328 dims.
5. Run the checkpoint in Newton; compare task metrics against mjlab and against the noise floor.
