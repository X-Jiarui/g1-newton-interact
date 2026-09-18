# ASTRA base model + main training: spec sheet

Written for someone about to write distillation code against this project, who has not worked
on it before. Everything here was read off the running code and the live training logs on
2026-09-18, not from memory.

## 0. Scope, and what to ignore

Read this file, then these three source files:

| what | path (inside the mjlab tree, see §1) |
|---|---|
| ASTRA wrapper: loads the ONNX, builds the torch copy, decodes the action | `src/mjlab/tasks/residual_interact/residual_actor.py` |
| the 136-D ASTRA observation, and every other observation group | `src/mjlab/tasks/residual_interact/mdp.py` (`class AstraObs136`) |
| env config: actions, rewards, terminations, sim rate | `src/mjlab/tasks/residual_interact/env_cfgs.py` |

**Ignore `src/mjlab/tasks/astra_distill/` and `docs/astra_body_dagger_distill.md`.** They are an
earlier, unrelated distillation attempt (body-only DAgger against a different teacher). Do not
read them, do not copy their structure, do not reuse their obs/action conventions. Start from
the interface in this document.

## 1. Where the code lives

Two trees, and they are NOT the same code:

- **`g1-newton-interact`** (this repo) — launcher, configs, the Newton/MuJoCo env wrapper.
  Entry point is `tools/run/train_newton.py`. Configs in `configs/train/`.
- **`mjlab`** — the task itself (observations, actions, rewards, the ASTRA wrapper). It is a
  separate checkout selected by `PYTHONPATH`. On the H200 it is `/home/jrxu/mjlab-run`.
  This repo carries a whole-file overlay of the task under
  `tools/setup/mjlab_overlay/residual_interact/`, applied by `tools/setup/apply_mjlab_overlay.py`.

Which mjlab is live is decided by `PYTHONPATH` at launch, and getting it wrong silently runs
different code. Always confirm with `tr '\0' '\n' < /proc/<pid>/environ | grep PYTHONPATH`.

## 2. ASTRA: the frozen base policy

ASTRA (a.k.a. Humanoid-GPT) is a **frozen 29-DoF whole-body motion tracker**. It is never
trained here. It knows nothing about objects, hands, or grasping — it tracks a body reference.

**Weights:** `pns_wo_priv216.onnx` (11.7 MB). On the H200: `/home/jrxu/onnx/pns_wo_priv216.onnx`.
Resolved by `ASTRA_ROOT` (default `<ASTRA_ROOT>/storage/ckpts/pns_wo_priv216.onnx`), see
`rl_cfg.py: DEFAULT_ASTRA_ONNX`. The `216` in the filename is wrong — the obs is 136-D.

**ONNX signature** (read with `onnx.load`):

```
IN   obs                 [batch, 136]   float32
OUT  continuous_actions  [batch, 29]    float32
OUT  std_param           [29]           float32   (unused here; we run the mean)
```

Two backends, chosen by `ASTRA_BASE_BACKEND` (`torch` | `onnx_cuda` | `onnx_cpu`). Our runs use
**`torch`**: the wrapper reads the ONNX initializers and rebuilds the MLP in torch as
`136 -> 2048 -> ... -> 512 -> 29`, so the 512-D penultimate hidden state is reachable. The
`onnx_*` backends only give you the 29-D output.

### 2.1 The 136-D observation, in order

Built by `AstraObs136.__call__`. All body quantities are in **ASTRA/PKL joint order**, which is
`BODY_29_DOF_NAMES` (§2.3). `frame_next = min(current_reference_frame + 1, clip_end)`.

| # | dims | what |
|---|---|---|
| 1 | 3 | root angular velocity, base frame (`root_link_ang_vel_b`) |
| 2 | 3 | gravity direction rotated into the base frame |
| 3 | 29 | `q - ASTRA_DEFAULT_BODY_PKL` (measured joint pos, offset by ASTRA's default pose) |
| 4 | 29 | `qd` (measured joint velocity) |
| 5 | 29 | last ASTRA action, in ASTRA's own units (`_astra_last_action_pkl`) |
| 6 | 29 | `reference_dof_pos[frame_next, :29] - ASTRA_DEFAULT_BODY_PKL` |
| 7 | 1 | reference root height z, relative to the env origin |
| 8 | 3 | gravity rotated into the **reference** root frame |
| 9 | 6 | reference root velocity, linear + angular (`_astra_ref_root_cvel[frame_next]`) |
| 10 | 2 | `[cos(yaw_err), sin(yaw_err)]`, reference yaw minus measured yaw, wrapped to ±pi |
| 11 | 2 | reference root xy minus measured root xy, rotated into the heading-local frame |

Total 136. The builder asserts this and raises if it drifts. Output is `nan_to_num` + clamped
to ±1e6.

### 2.2 The 29-D action, and how it becomes a joint target

ASTRA's 29 outputs are **not** joint targets. `residual_actor.py:_astra_to_mjlab_body_action`
converts ASTRA units to mjlab units:

```python
target_pkl       = ASTRA_DEFAULT_BODY_PKL + a_astra * ASTRA_ACTION_SCALE_PKL * 0.25
body_action_pkl  = (target_pkl - mjlab_default_pkl) / mjlab_scale_pkl
body_action_il   = body_action_pkl[pkl_for_il]          # PKL order -> mjlab "IL" order
```

`0.25` is `ASTRA_POLICY_ACTION_SCALE`. The per-joint constants `ASTRA_DEFAULT_BODY_PKL`,
`ASTRA_ACTION_SCALE_PKL`, `ASTRA_EFFORT_BODY_PKL`, `ASTRA_KP_BODY_PKL`, `ASTRA_KD_BODY_PKL`
are literal arrays at the top of `mdp.py`. **The PKL->IL permutation is real and easy to get
wrong** — a wrong permutation gives a robot that stands but tracks nothing.

The env's action term (`Sonic53Action.apply_actions`) then turns the mjlab action into a
position target:

```python
body_target = (body_default_pkl + a_body_pkl * body_scale_pkl * body_action_scale).clamp(-3.14, 3.14)
hand_target = reference_default_hand + a_hand * hand_action_scale
```

and those go to the actuators as position setpoints (the robot is a **position servo** — any
visible torque law in the action term is dead code).

**Startup override:** for the first 36 control steps the target is replaced by the reference
pose outright, and for the first 30 the root is hard-held. Any imitation dataset must either
skip those steps or reproduce them, or the student learns to sit still for 0.7 s.

### 2.3 Joint orders (copy these exactly)

`BODY_29_DOF_NAMES` — ASTRA/PKL order, and the first 29 of our action vector:

```
left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_ankle_pitch, left_ankle_roll,
right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee, right_ankle_pitch, right_ankle_roll,
waist_yaw, waist_roll, waist_pitch,
left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow,
left_wrist_roll, left_wrist_pitch, left_wrist_yaw,
right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow,
right_wrist_roll, right_wrist_pitch, right_wrist_yaw
```
(all with a `_joint` suffix in the MJCF)

Hand: **Wuji gen-1, 20 DoF per side, 40 total**, `left_finger{1..5}_joint{1..4}` then
`right_finger{1..5}_joint{1..4}`. `finger1` is the thumb. Selected by `APPLE_HAND_KIND=wuji`
(the other option, `xhand`, is 12/side and changes every dimension below — check it).

So `NUM_BODY=29`, `NUM_HAND=40`, **`ACTION_DIM=69`**.

## 3. What we actually train

ASTRA is frozen. We train a **residual** on top of it, for a table-top grasp task.

- **Task id:** `Mjlab-ResidualInteract-G1`. Physics: **Newton** (not the MuJoCo CPU path).
- **Sim:** `timestep=0.005`, `decimation=4` -> **control at 50 Hz**. `episode_length_s=12.0`,
  but real episodes end at ~60-85 control steps on terminations (§3.3).
- **Reference data:** 8 retargeted GRAB clips, mixed in one run, in this fixed order
  (clip index matters — per-clip state is indexed by it, and reordering changes results):

  | clip | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
  |---|---|---|---|---|---|---|---|---|
  | object | cubesmall | phone | gamecontroller | binoculars | hammer | camera | banana | flashlight |

  Each env is assigned one clip; envs are grouped by clip in blocks (with 64 envs and 8 clips,
  clip k owns worlds 8k..8k+7). **Do not assume the blocks are in clip order — look it up from
  the live `clip_id` array.**
- **Episode shape:** the object starts on a table. `RSI_ANCHOR_CF=1` starts the episode at a
  frame relative to the clip's contact frame `cf`, drawn from a window. The table is removed
  `TABLE_REMOVE_AFTER_CF=50` control steps (1.0 s) after `cf`, which is when a grasp is actually
  tested — before that the object is supported.

### 3.1 The residual architecture (`arch=astra_hidden_residual`)

This is the part that surprises people. The residual does **not** simply add to the 69-D action.

```
astra_obs(136) --[frozen ASTRA trunk]--> hidden(512)
features(1328) --[residual MLP]--------> 552 = [ token_delta(512) , hand_action(40) ]
(hidden + token_delta) --[frozen ASTRA head 512->29]--> a_astra(29) --decode--> body_action(29)
final_action(69) = concat(body_action(29), hand_action(40))
```

- `token_delta` is clipped to **±0.1** (`token_clip`); the body residual acts *inside* ASTRA.
- The hand 40 dims are produced directly by the residual — **ASTRA contributes nothing to the
  hand** (`base_hand_mode=zero`, i.e. the hand base is all-zero).
- `residual_gain=1.0`, `residual_action_clip=0.5`, `zero_init=True`.
- PPO sees a Gaussian whose mean is the final 69-D action.
- The 1328-D residual feature vector is the concatenation of these observation groups:
  `sonic_obs_or_latent, astra_obs, proprio_history, tracker_action, reference_phase,
  reference_preview, object_state, hand_object_geometry, object_future, contact_features,
  tracking_error, last_residual, last_final_action`. Each group's builder is in `mdp.py`; the
  log prints `active feature groups: name=dim` at startup — read it there rather than summing
  by hand.

**If you are distilling the whole policy, the student's target is the 69-D `final_action` and
its input should be whatever subset of those groups you choose to keep.** You do not need to
reproduce the hidden-state trick; it is an implementation detail of how the teacher was trained.

### 3.2 Main-training hyperparameters that matter

| knob | value | note |
|---|---|---|
| envs | 4096 (H200) / 1024 (5090) | `num_steps_per_env=128`, so 524288 / 131072 samples per iteration |
| throughput | ~2000-2500 env-steps/s | one iteration is ~210-260 s at 4096 envs |
| `HAND_PD` | `10,0.2` on the current best run | see the warning below |
| RSI window | wide: `RSI_CF_OFFSET_START=-1000 RSI_CF_OFFSET_END=-20` | i.e. `[clip start, cf-20]`; the narrow default is `[cf-20, cf-10]` |
| `WRIST_TARGET_FAR` | `0` (off) on the current best run | |
| `TABLE_REMOVE_AFTER_CF` | 50 | 1.0 s; shorter never produced a sustained lift |

**`HAND_PD` is contested, and `configs/train/best.yaml` currently argues against it.** kp 10 /
kd 0.2 zeroes the grasp when it is applied to a checkpoint trained at kp 300 — that is what the
comment in `best.yaml` records. But the current best run (`PD10_WIDE_S`) trains **from scratch**
at kp 10 / 0.2 and does grasp (camera 0.61, gamecontroller 0.61, phone 0.48). Treat the PD as
"depends on whether you switch it mid-training", and read it off the run you are distilling
rather than off the config.

### 3.3 Terminations (why episodes are short)

Set in `env_cfgs.py: terminations`. The ones that actually fire: `og_object_far` (object more
than 0.12 m from its reference), `object_reference_window`, `tip_cf_miss`, `fell_over`,
`time_out`. `og_object_far` is the dominant one, and it fires about 1 s **after** a successful
grasp, because the reference keeps moving while the policy holds still.

For generating imitation data this matters a lot: `ROLLOUT_NO_TERM=1` (in
`tools/run/train_newton.py`) proxies every non-`time_out` termination to never-fire, changing
no physics, so a rollout runs the full `--rollout-steps` instead of being cut at the grasp.
With it, episodes can run to the 600-step `time_out`. Note that beyond ~150 steps the policy is
far outside its trained horizon and tends to drop the object — do not train a student on that
tail and call it a failure of the teacher.

### 3.4 Success metric

`PhaseA/lift_success/clip<k>` — object lifted `lift_height_m=0.03` above its own start and held
`hold_duration_s=0.5` with at least 2 fingertips in contact, averaged over the RSI start-frame
distribution. It is **not** comparable across different RSI settings, and per-clip values are
the only meaningful ones (the aggregate mixes 8 tasks).

## 4. Teacher checkpoints worth distilling

Current best, all on the 8-clip mix, `lift_success` per object:

| run | envs | cube | phone | gamectl | binoc | camera | banana | checkpoint |
|---|---|---|---|---|---|---|---|---|
| `K0_V_G` | 1024 | 0.51 | 0.77 | – | – | 0.75 | – | `model_13100.pt` (finished) |
| `REF_BASE_G` | 1024 | – | – | – | 0.08 | 0.69 | 0.58 | live, ~it 8700 |
| `PD10_WIDE_S` | 4096 | – | 0.48 | 0.59 | 0.04 | 0.61 | – | live, ~it 2200 |

hammer and flashlight have never been lifted by any run. These three runs differ in ~6
variables at once, so they are three separate recipes, not an ablation. `K0_V_G` is the only
finished one and the natural first teacher.

Each `logs/rsl_rl/g1_residual_interact/<RUN>/` holds `model_*.pt` plus `params/` with the
`env.yaml` / `agent.yaml` that run used. **`env.yaml` is not reloaded by `play.py`** — it
restores the agent config only, so an eval silently runs the default RSI window unless you
re-supply the env vars. Read them off the training process, not off the config.

## 5. Running a rollout to generate data

`tools/run/train_newton.py --iterations 0 --rollout-steps N --resume <ckpt>` runs deterministic
inference (the policy mean). Four things must be right or you will record a working policy
failing:

1. `ROLLOUT_START_FRAME=rsi` — otherwise the rollout forces reference frame 0, which a policy
   trained with `RSI_ANCHOR_CF` has never seen. Setting the `RSI_*` env vars alone does nothing.
2. Pass the **exact** `--reference-pkls` / `--sdf-objects` lists in training order.
3. Use enough envs (64 is fine). mjlab caches observations until a reset, so a 1-env rollout
   runs on frame-0 observations and emits one constant action.
4. Re-supply the run's own env vars (`HAND_PD`, `RSI_*`, `WRIST_TARGET_FAR`, `OG_FAR_AFTER_TABLE`,
   `APPLE_HAND_KIND`, `GMR_ROOT`, `SEED_CUBE`) and its own `--config`. Copy them from
   `/proc/<pid>/environ` of the live run rather than retyping.

Useful extra switches in the same file: `ROLLOUT_NO_TERM=1` (§3.3),
`ROLLOUT_PER_WORLD_LIFT=1` (prints per-world `max/final` object height, so you can pick a world
that actually succeeded), `VIDEO_CLIP=<k>` / `VIDEO_ENV=<w>` + `VIDEO_FOLLOW=1` for video.

Rendering note: Newton's ViewerGL has no GPU GL context on the H200 and falls back to software
(~3 min per frame). Render on a box with a real GL context.
