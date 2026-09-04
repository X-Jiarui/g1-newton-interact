# G1 Wuji — mjlab → Newton port

Trains and evaluates an mjlab residual grasping policy with [Newton](https://github.com/newton-physics/newton)
1.5 doing the physics, and reproduces mjlab's behaviour.

Two contact paths are supported and are meant to be compared:

| | collision | object collider | throughput at 2048 env |
|---|---|---|---|
| **MuJoCo contacts** (default) | MuJoCo's own narrow phase inside `SolverMuJoCo` | MuJoCo's convex hull, 124 verts for the stapler | 40429 env-steps/s |
| **Newton native** (`--native-contacts --rigid-object-table`) | Newton's `CollisionPipeline` | the real STL mesh, 19991 verts | 32437 env-steps/s (1.25x slower) |

Which one learns faster is **not settled** — see [Status](#status) before drawing a conclusion
from either. Earlier revisions of this file quoted 2.1x and 1.10x for the native path; both were
measured before `--rigid-object-table` or at a different env count. 1.25x at 2048 env is the
current number.

---

## The recipe to start from

For a new single-clip run, use the configuration in **[docs/RECIPE.md](docs/RECIPE.md)** rather
than the `--native-contacts` defaults. The two solver options it sets (`cone=pyramidal`,
`impratio=1`) are the only ones that differ between the two contact paths, and on the stapler clip
they moved the first lift from "never cleanly" to iteration 542.

Beyond the single-clip sweep: **[2.5 mixed-clip training](#25-mixed-clip-training-one-policy-many-clips-and-objects)**
trains one policy over several clips and objects at once, **[2.6](#26-recording-a-video-per-run)**
records a video per run inside Newton, and **[2.7](#27-comparing-checkpoints-in-the-browser)**
compares checkpoints in a browser.

To bring up a fresh box and start a sweep:

```bash
git clone https://github.com/X-Jiarui/g1-newton-interact.git && cd g1-newton-interact
./tools/setup/bootstrap_box.sh --from user@source-box:/path      # deps + mjlab + data
./tools/run/launch_clip_sweep.sh --top 8 --iterations 6000       # one clip per GPU
./tools/run/monitor_runs.sh logs/monitor.txt 600 NAME1 NAME2 ...
```

`data/top100_travel.json` is the clip ranking the sweep draws from: every clip in the 1324-clip
retarget set, sorted by how far the object actually travels, jitter-suspect clips dropped. Rebuild
it with `tools/pipeline/rank_sequences_by_travel.py`.

---

## 1. Install on a fresh box

Everything below assumes a CUDA machine with a recent NVIDIA driver.

### 1.1 What must already exist

| | where it comes from | note |
|---|---|---|
| Python env with `mjlab`, `newton==1.5`, `warp==1.16`, `mujoco_warp`, `torch`, `viser` | project image or conda env | `newton` and `mujoco_warp` versions matter; 1.5 / 3.11 is what this was measured on |
| **mjlab source** with the `residual_interact` and `apple_eat` tasks | the mjlab checkout this project pairs with | **verify it matches**, see 1.3 |
| reference clips (`grab_g1_wuji_aligned`) | the GRAB retarget pipeline | 1324 clips, 17 GB |
| object meshes (`scaled_grab_wuji_all_o70/meshes`, `scaled_grab_dataset_wuji/meshes`) | same | the second holds the full 41-object GRAB set |
| an agent config to start from — a checkpoint directory containing `params/agent.yaml` and `params/env.yaml` | a previous sweep | see 1.4, this is not optional |

### 1.2 Clone

```bash
git clone https://github.com/X-Jiarui/g1-newton-interact.git
cd g1-newton-interact
```

### 1.3 Verify the mjlab you import is the one you think

The task code defines the observation layout and the reward weights. A mismatched checkout will
train or evaluate a *different* agent without raising anything.

```bash
python -c "import mjlab, os; print(os.path.dirname(mjlab.__file__))"
# then compare against the box the checkpoints came from:
cd $(python -c "import mjlab,os;print(os.path.dirname(mjlab.__file__))") && \
  find tasks -name '*.py' | sort | xargs md5sum | md5sum
```

Run the same two lines on both machines and compare the final hash. This has caught a real
mismatch in this project.

### 1.4 Environment variables

Every command below needs these. They are read at import time by the task modules, so exporting
them after the process starts does nothing.

```bash
export APPLE_HAND_KIND=wuji
export APPLE_SCENE_Z_OFFSET=-0.03
export ASTRA_BASE_BACKEND=torch
```

`APPLE_EAT_PKL` is set for you by `--reference-pkl`. Configuration for this task lives in ~35
`APPLE_*` environment variables; a run does not record which ones it used, so if a result is
surprising, check the environment before the code.

---

## 2. Training

### 2.1 Newton native contacts

```bash
python tools/run/train_newton.py \
  --xml assets/scene_stapler/scene.xml \
  --reference-pkl $DATA/grab_g1_wuji_aligned/s8/stapler_pass_1.pkl \
  --sdf-object   $DATA/scaled_grab_wuji_all_o70/meshes/stapler.stl \
  --native-contacts --rigid-object-table --table-under-object --cuda-graph \
  --num-envs 2048 --iterations 6000 --seed 1 \
  --run-name MY_RUN
```

### 2.2 The MuJoCo-contact control

Identical minus two flags. Use this whenever you need a line that is known to learn:

```bash
python tools/run/train_newton.py \
  --xml assets/scene_stapler/scene.xml \
  --reference-pkl $DATA/grab_g1_wuji_aligned/s8/stapler_pass_1.pkl \
  --sdf-object   $DATA/scaled_grab_wuji_all_o70/meshes/stapler.stl \
  --table-under-object --cuda-graph \
  --num-envs 2048 --iterations 6000 --seed 1 \
  --run-name MY_CONTROL
```

### 2.3 Flags that matter, and why

| flag | effect |
|---|---|
| `--native-contacts` | switches to Newton's `CollisionPipeline`. Also convex-hulls the robot's 65 mesh colliders, swaps the table to a mesh collider, raises `nconmax`/`njmax`/`contact_sensor_maxmatch`/`max_triangle_pairs`, and enables the world-welded body pose sync. It is a bundle, not one knob. |
| `--rigid-object-table` | the object/table pair collides rigidly instead of through the hydroelastic SDF. Hydroelastic was that pair's only consumer; turning it off cut per-world contacts from 48-79 to 19-25 and took the path from 1.46x to 1.10x of the MuJoCo line at 1024 env (1.25x at 2048). The object keeps its real mesh either way. |
| `--table-under-object` | places the table under the object using the reference clip and the true collider extents. Without it the object starts in the air. |
| `--cuda-graph` | 2.15x at 2048 env. The body-pose sync had to be written as a warp kernel to stay capturable. |
| `--solver-kwargs '{"impratio": 1.0, "cone": "pyramidal"}'` | override the solver. The native default is `elliptic` / `impratio=1000`, copied from Newton's hydroelastic example; the MuJoCo line runs `pyramidal` / `impratio=1`. |
| `--table-sdf-resolution` | free to lower — at 8 the resting penetration is still -0.002 mm. It does not change the contact count, which follows the *object's* resolution. |
| `--resume` | warm-start from a checkpoint. rsl_rl keeps only the last five rolling checkpoints plus every 500th, so copy the one you want somewhere safe before using it. |

### 2.4 Watching a run

The numbers that carry information, in order of usefulness:

- `Episode_Metrics/object_motion_frac` — the object is actually moving. The clearest early separator.
- `Stage/physical_contact` — the hand is touching it.
- `Episode_Metrics/lift_success` — fraction of *finished episodes* that ever lifted 3 cm and held 0.5 s.
- `PhaseA/lift_success` — the same criterion sampled *every step over all envs*, so it is much lower
  and rises as the policy lifts *earlier*. `lift` 0.97 with `liftA` 0.20 is not a contradiction.
- `Health/nonfinite_worlds` — only logged when non-zero. Silence is good.

`Episode_Metrics/ep_len` is **not** the average episode length. Its term uses the default
`reduce="mean"`, which averages `episode_length_buf` over the steps *within* an episode, so for an
episode of length L it reports about L/2; it is also averaged only over the envs being reset that
step, which oversamples short episodes. Useful as a trend, misleading as an absolute.

**Lift arrives as a phase transition, and when it arrives depends on the collider.** Two older
baselines using mjlab's decimated `cir160` collider sat at exactly zero for ~4400 iterations and
then turned contact and lift on together (stapler 4447, mug 5992) — and a second mug seed never
lifted at all in 7295 iterations. A later control line on the same task, differing only in that
the object is handed to MuJoCo as the real mesh (which MuJoCo then hulls to 124 verts), first
lifted at **iteration 918**. So do not apply a fixed "wait 4500 iterations" rule: watch for the
transition instead, and treat any single seed with suspicion.

---

## 2.5 Mixed-clip training (one policy, many clips and objects)

A single run can train across several clips with **different objects**. Each clip gets a
contiguous block of environments, its own object mesh, and its own SDF collider; the object is
replicated per world, so clip *i*'s worlds contain clip *i*'s object and nothing else.

```bash
python tools/run/train_newton.py \
  --xml assets/scene_stapler/scene.xml \
  --reference-pkls $DATA/grab_g1_wuji_aligned/s1/hammer_use_2.pkl,$DATA/grab_g1_wuji_aligned/s1/binoculars_lift.pkl \
  --sdf-objects    $MESH/hammer.stl,$MESH/binoculars.stl \
  --mix-env-split  1024,1024 \
  --agent-cfg-from $SWEEP/OF_00_apple_eat_1_SPHERE/model_7310.pt \
  --table-under-object --native-contacts --rigid-object-table \
  --object-solref 0.004,1.0 \
  --num-envs 2048 --iterations 6000 --run-name MIX2
```

`--reference-pkls` and `--sdf-objects` are **positionally paired** and validated at startup — a
mismatched count or a mesh whose name does not appear in its clip's path aborts before the model is
built. `--mix-env-split` must sum to `--num-envs`; omit it for an equal share.

Every per-clip metric is logged under `PhaseA/<metric>/clip<i>`, in `--reference-pkls` order. Read
those, not the aggregate: the aggregate is an env-count-weighted mean, so a curriculum that moves
environments between clips changes it even when no clip improved.

### The env → clip map has exactly one owner

The map from environment index to clip is built once, in `newton_vec_env`, and everything else
**reads** it. Recomputing it — even with what looks like the same formula — is how a mixed run once
collapsed to 8-step episodes: mjlab's side derived `env_id % n_clips` while the env used contiguous
blocks, so nearly every environment was scored against another clip's reference. `clip_frame0_rows`
in mjlab's `apple_eat/mdp.py` now takes `env` and defers to `_clip_id(env)`, and raises if the
length disagrees rather than broadcasting.

mjlab is not versioned with this repo, so that patch ships as
`tools/setup/patch_mjlab.py` and is applied by `bootstrap_box.sh` after the mjlab rsync. It is
idempotent. **A fresh box that skips it will train a mixed run that looks alive and learns
nothing.**

### Curriculum: PMCP and graduation

Off by default. Both reallocate environments between clips while training runs.

| variable | default | what |
|---|---|---|
| `MIX_PMCP_EVERY` | `0` (off) | reallocation period, in control steps |
| `MIX_PMCP_METRIC` | `PhaseA/lift_success` | the metric clips are ranked on |
| `MIX_PMCP_QUOTA` | `0` | per-clip env floor; 0 means an equal share |
| `MIX_PMCP_TAU` / `MIX_PMCP_EMA` | `8.0` / `0.1` | softmax temperature, metric smoothing |
| `MIX_PMCP_RULE` | `graduation` | `graduation` moves envs off clips that have met their bar |
| `MIX_GRAD_CONTACT` / `MIX_GRAD_LIFT` | `0.2` / `0.1` | the bars a clip must clear to graduate |
| `MIX_GRAD_HOLD` | `3` | consecutive **windows** (not steps) the bars must hold |
| `MIX_GRAD_WARMUP` | `3` | windows before any graduation can fire |
| `MIX_GRAD_RELEASE` | `0.5` | fraction of a graduated clip's envs given up per stage |

Three defects here were real and are fixed; the shapes are worth knowing because they all produced
a run that looked healthy:

- **`HOLD` counts windows, not steps.** Counting steps makes the bar trivially easy to clear.
- **The window boundary gates the update, not just the reallocation.** Updating the hold counters
  every step while only reallocating on the boundary lets a clip graduate off a single lucky step.
- **A stage change that moves no environment is printed**, as `(stage only; no environment could be
  moved inside the object block)`. It used to return silently, which made the `[grad]` lines vanish
  and led to the wrong conclusion that graduation was never firing.

Environments only move **within an object block** — worlds are built with a specific object mesh
and cannot be reassigned to a clip that uses a different one. Give-back is distributed evenly
rather than piled onto `clips[0]`.

---

## 2.6 Recording a video per run

One command records every run on a box, through **Newton's own renderer**, with no early
termination:

```bash
tools/run/record_runs.sh --out /root/nvideos                 # every live run on this box
```

```bash
tools/run/record_runs.sh --out ~/nvideos --log-root logs/rsl_rl \
  --runs RUN_A,RUN_B --pkl $DATA/s8/stapler_pass_1.pkl --stl $MESH/stapler.stl
```

```bash
tools/run/record_runs.sh --out /workspace/nvideos --mix MIX8_BIG \
  --data $DATA --mesh $MESH \
  --clips s1/hammer_use_2:hammer,s1/binoculars_lift:binoculars      # one video per clip
```

Defaults to 500 frames / 10 s; `--steps N` changes it, `--python` picks the interpreter.

**Both properties are load-bearing, and both were wrong in the first version:**

*The scene must be Newton's.* Dumping `qpos` and replaying it through mjlab's MuJoCo model gives
the right trajectory in the wrong world — mjlab's model carries the visual meshes while Newton's
carries the real object mesh, the real table and the SDF colliders. The replay silently swaps the
object and the collision geometry, so the video stops being evidence about the run. `--newton-video`
draws the model the physics integrates. An analysis of finger penetration computed in the replay
model was retracted for exactly this reason.

*The episode must not be cut short.* With terminations live, a grasp that slips resets mid-clip and
the video becomes a sequence of restarts. `--rollout-free-run` clears the termination terms **and**
lifts `episode_length_s` — a time-out resets just as surely as a termination — and it must run
before the env is built, because the video only flushes once `video_steps` frames exist.

The rollout prints its own contact / lift / stand / sequence averages when it finishes. Compare
those against the run's training row before trusting what the video looks like; a video that merely
looks wrong is an opinion, those numbers are a measurement.

`pyglet` is required for the headless `ViewerGL` and is not in every image.

---

## 2.7 Comparing checkpoints in the browser

`tools/pipeline/policy_gallery.py` serves a viser page with a dropdown over recorded traces, so
several runs (or several checkpoints of one run) can be stepped through side by side.

```bash
python tools/pipeline/policy_gallery.py --traces /root/traces --port 8099
ssh -N -L 8099:localhost:8099 <user>@<host>      # then open http://localhost:8099
```

Start every evaluation at **reference frame 0**, and keep one viser alive at a time.

A mocap body that the render scene does not have is skipped, not fatal — Newton's scene carries a
terrain plane the render model lacks. A body matching *several* candidates still aborts: picking one
is how the table once ended up drawn under the robot's feet. If **no** recorded mocap body maps into
the scene the loader refuses outright, because that is the case where the table is silently drawn at
the origin.

---

## 3. Headed evaluation

Serves a live viser view of a rollout, with a dropdown that hot-swaps checkpoints.

```bash
python tools/run/run_newton.py \
  --xml assets/scene_stapler/scene.xml \
  --reference-pkl $DATA/grab_g1_wuji_aligned/s8/stapler_pass_1.pkl \
  --sdf-object   $DATA/scaled_grab_wuji_all_o70/meshes/stapler.stl \
  --agent-cfg-from $SWEEP/OF_00_apple_eat_1_SPHERE/model_7310.pt \
  --table-under-object \
  --checkpoint  $CK/model_2000.pt \
  --checkpoints "$CK/model_2000.pt,$CK/model_1500.pt,$CK/model_1000.pt" \
  --viser-port 8099 --steps 400 --start-frame 0
```

From your laptop:

```bash
ssh -N -L 8099:localhost:8099 <user>@<host>     # then open http://localhost:8099
```

The GUI panel carries the checkpoint dropdown, a *restart rollout* button, and a status line.
Selecting a checkpoint reloads the policy into the existing runner and restarts from reference
frame 0, so the comparison between checkpoints is fair.

**`--agent-cfg-from` is required, not optional.** `train_newton.py` takes its agent config from a
default checkpoint elsewhere and writes no `params/` into its own run directory. Pointing the eval
at the checkpoint under evaluation would silently evaluate a different agent.

Note this path runs **MuJoCo contacts** — `run_newton.py` builds its own solver and has no
`--native-contacts`. A policy trained natively and evaluated here is being scored under a different
contact engine.

---

## 3.5 Grasp retargeting: fixing the hand at the contact frame

GMR gives a good body and a bad hand: at GRAB's own contact frame the retargeted fingertips are a
median **9.1 cm** from the object points the human touched with them, and the palm is **41.9 deg**
off the human's orientation. [`docs/07-grasp-retarget.md`](docs/07-grasp-retarget.md) re-solves the
arm and the operating hand there, against targets read out of GRAB.

**Adopted route: mesh fingertips + a palm-orientation objective.**

| | orient | shape | residual | penetration | 4+ finger contact |
|---|---|---|---|---|---|
| GMR, no IK | 41.9 deg | 16.8 deg | — | 0.86 cm | 58.6 % |
| joint + 2 cm, rotation 0.3 | 12.8 deg | 12.8 deg | 1.49 cm | 0.40 cm | 59.1 % |
| **mesh tip + rotation 0.3** | **12.3 deg** | **12.4 deg** | **1.60 cm** | **0.55 cm** | **71.7 %** |

198 s1 clips, 198/198 solved, no arm joint left within 2 % of a limit.

```bash
X=$GMR_ROOT/assets/g1_wuji/g1_mocap_29dof_with_wuji_hands.xml

python tools/retarget/shrink_object_radius.py --dataset-root $SRC --out-root $WORK/shifted \
  --robot-xml $X --mode adaptive --grab-dir $GRAB_DIR --margin 0.03

python tools/retarget/extract_grasp_targets.py --grab-dir $GRAB_DIR \
  --smplx-dir $GMR_ROOT/assets/body_models --dataset-root $WORK/shifted \
  --out $WORK/targets.npz --back 0

python tools/retarget/solve_arm_ik.py --dataset-root $WORK/shifted --grab-dir $GRAB_DIR \
  --robot-xml $X --targets mesh --grasp-npz $WORK/targets.npz --free-fingers \
  --rot-weight 0.3 --out-root $WORK/solved --summary $WORK/solved.csv
```

Swap `--targets mesh` for `--targets human --tip-extend 0.020` to run the joint-based route, which
the mesh route replaced. Every parameter, sweep and rejected alternative is recorded in
[`configs/retarget/grasp_ik.yaml`](configs/retarget/grasp_ik.yaml) — including a global object scale,
a larger reach margin and three other target definitions, all measured and all worse.

The IK has **no collision term**, so fingers still pass through the object at a median 0.55 cm, and
only the contact frame is solved. See the doc for what that leaves open.

## 4. Objects: SDF pipeline and gallery

### 4.1 Build and validate a collider for every object

```bash
python tools/pipeline/build_object_sdfs.py \
  --mesh-dir $DATA/scaled_grab_dataset_wuji/meshes \
  --out artifacts/pipeline/sdf_manifest.json \
  --cache-dir ~/.cache/sdf
```

Two stages. **build** bakes the SDF and records geometry facts; **drop test** puts the object on a
table in a minimal Newton scene and measures how it settles. The second stage is the point: an SDF
that bakes without error can still be unusable — at resolution 16 the stapler builds fine and then
rests 2.7 mm inside the table.

Result over the 41 GRAB objects: 41/41 built, 41/41 accepted, |penetration| median 0.0096 mm, worst
0.4533 mm, 7.7 s total.

Resolution is chosen **per object** from a 1 mm target voxel, not fixed. `max_resolution` divides
the *longest* axis, so a fixed 64 gives a 22 cm knife a 3.8 mm voxel and its blade one or two voxels
of thickness (knife 1.06 mm into the table, flute 1.17, gamecontroller 0.76). Raising it globally is
wrong in the other direction: at 256 the stapler produced zero contacts where 64 and 128 rest at
-0.0005 mm.

### 4.2 Look at the colliders

```bash
python tools/pipeline/object_gallery.py \
  --mesh-dir $DATA/scaled_grab_dataset_wuji/meshes \
  --manifest artifacts/pipeline/sdf_manifest.json \
  --port 8100
ssh -N -L 8100:localhost:8100 <user>@<host>     # then open http://localhost:8100
```

Dropdown over all 41 objects, each resting on the table, drawn as the geometry the physics
collides. A checkbox overlays the convex hull — what the MuJoCo path collides instead. On the
stapler the hull fills in the cavity under the handle; that difference is the leading explanation
for why the two contact paths learn this object at different rates.

### 4.3 Rank clips by how far the object travels

```bash
python tools/pipeline/rank_sequences_by_travel.py \
  --dataset $DATA/grab_g1_wuji_aligned \
  --out artifacts/pipeline/top100.json --top 100 --exclude-jitter
```

Reports path length, net displacement, furthest-from-start and vertical range, and de-duplicates by
`(subject, sequence)`. Path length alone is not enough: `stapler_lift` accumulates 1.853 m of path
while never getting more than 0.124 m from its start, so it is flagged jitter-suspect rather than
ranked above clips that genuinely move.

---

---

## Status

Current as of the runs described below; **everything in this section is one seed per
configuration and several runs are still going**, so treat it as the state of an investigation,
not as a result.

### The native path's default solver settings are probably wrong

`--native-contacts` sets `cone="elliptic"` with `impratio=1000`, copied from
`newton/examples/robot/example_robot_panda_hydro.py`. That example runs hydroelastic contact; we
do not (`--rigid-object-table`). `impratio=1000` makes the tangential constraints three orders of
magnitude softer than the normal ones.

An audit of every contact-related parameter found that **these two options are the only solver
settings that differ between the two paths** — the MuJoCo line already runs `pyramidal` with
`impratio=1`, and every per-geom and per-contact parameter (friction, solref, solimp, condim,
margin, gap) is identical on both. Setting the native path to the same two values makes every
per-contact parameter match exactly.

Three runs are testing it, all with `--solver-kwargs '{"impratio": 1.0, "cone": "pyramidal"}'`:

| | iterations since the change | `lift_success` | same-iteration control |
|---|---|---|---|
| stapler, resumed from a native checkpoint at iter 2000 | 258 | 0.22 | 0.021 |
| mug, resumed from a native checkpoint at iter 2000 | 181 | 0.006 | 0 |
| stapler, from scratch | 326 | still 0 (too early) | — |

The stapler resume went from 0.008 to 0.22 in 258 iterations while its unchanged twin moved from
0.008 to 0.021 over the same span. That is a large effect and the direction has held for twelve
consecutive samples, but it is one seed and `lift_success` is high-variance.

### What is not explained by the solver settings

With friction aligned, the native path's `object_motion_frac` is ~0.23 against the control's
~0.64. Something still differs. The remaining candidates, in the order they are worth testing:

1. **Object collider geometry** — real mesh vs 124-vertex convex hull. On the stapler the hull
   fills in the cavity under the handle; `tools/pipeline/object_gallery.py` shows the difference.
   This also fits the object dependence: the mug, a solid cylinder, shows no such gap between the
   paths, while the concave stapler does.
2. **Table collider** — the native path swaps the table to `assets/meshes/table_box.stl`; the
   MuJoCo path keeps the scene's original geom. Same contact parameters, different shape.
3. **The collision engine itself** — at one sampled step the native path held 30 hand/object
   contact points against the control's 8, under identical friction parameters.

### Ruled out by measurement, do not re-suspect

- **Contact sensors.** The ratio of "sensor reports contact" to "hand is geometrically within
  5 cm" tracks within 1% between the two paths across the whole run (0.910/0.921 at iteration 500,
  0.952/0.963 at 1750). Newton's injected contacts are read faithfully by `mjSENS_CONTACT`.
- **Reward definitions.** All 37 reward terms are present on both paths with the same weights;
  both load the same `env.yaml`.
- **Termination conditions.** The same set, both firing.
- **Numerical stability.** `Health/nonfinite_worlds` has stayed at zero for every sample of every
  run since the guard was added.

## 5. Things that will bite you

Each cost real time here. [docs/03-defects.md](docs/03-defects.md) and
[docs/06-native-newton-migration.md](docs/06-native-newton-migration.md) carry the full accounts.

- **A metric reading exactly `0.0000` often means NaN.** `_safe_log` computes
  `nan_to_num(value.mean(), nan=0.0)` — mean first — so one non-finite world zeroes a whole metric.
- **Non-finite worlds do not heal.** The NaN sits in `qacc` and the solver warm start, which the
  reset events never write. `_nonfinite_worlds` / `_clear_world_state` detect and clear them.
- **`nconmax`, `njmax` and the SDF grid are all per world.** Sizes that are right for a one-env
  probe OOM at 2048.
- **Anything `O(n^2)` over shapes explodes after replication.** A pair-completion loop that added
  0 pairs at one env generated 8.4 M at 2048.
- **`contact_sensor_maxmatch` defaults to 64** and silently truncates the sensors the grasp rewards
  read. It is raised to 256 on the native path.
- **`broad_phase="explicit"` does no AABB culling.** It narrow-phased 9.2 M pairs per call for
  42-83 actual contacts; `nxn` is the default now.
- **The robot is hard-held for the first 30 steps** (`episode_length_buf <= 30`), and the reference
  frame is driven by that same counter. Any probe that bypasses `env.step()` must advance it by hand.
- **The table is placed by a reset event** and sits under the robot's feet before the first reset.
- **Verify geometry by vertex count, not by log lines.** A log line here asserted the object kept
  its real geometry while it was being hulled to 64 vertices.
- **Never run a probe on a GPU that has a training process.** It has already OOM-killed a run.
- **Never `pkill -f` / `pgrep -f` a pattern that appears in your own command line.** `ps` lists your
  own process; the pattern matches itself. This killed the controlling ssh session three times and
  left a liveness check permanently reporting "alive".

---

## 6. Repo map

| path | what |
|---|---|
| `src/newton_vec_env.py` | the training env: builds the Newton model, both contact paths, the manager loop |
| `src/newton_bridge.py` | lets mjlab's observation and action code read and write Newton state |
| `src/grab_objects.py` | swaps an entity's collider for a real STL and bakes its SDF |
| `src/newton_table.py`, `src/newton_extents.py` | table placement from measured collider extents |
| `tools/run/train_newton.py` | training entry point |
| `tools/run/run_newton.py` | rollout / headed eval with the checkpoint dropdown |
| `tools/pipeline/build_object_sdfs.py` | per-object SDF build + drop-test validation |
| `tools/pipeline/object_gallery.py` | viser gallery of collider geometry |
| `tools/run/record_runs.sh` | one video per run, Newton's renderer, no early termination |
| `tools/pipeline/policy_gallery.py` | viser dropdown over recorded traces, for comparing checkpoints |
| `tools/pipeline/rank_sequences_by_travel.py` | clip ranking by object travel |
| `tools/retarget/shrink_object_radius.py` | pulls each object radially toward the pelvis until the arm can reach it |
| `tools/retarget/extract_grasp_targets.py` | the human's fingertips at the contact frame, in the object's frame |
| `tools/retarget/solve_arm_ik.py` | re-solves the arm and hand at the contact frame with Newton's IK |
| `configs/retarget/grasp_ik.yaml` | every retargeting parameter, its sweep, and what was rejected |
| `tools/setup/patch_mjlab.py` | ships the env->clip map fix into the unversioned mjlab checkout |
| `docs/` | goal, parity, baseline, defects, architecture, reproduction, native migration, grasp retargeting |
