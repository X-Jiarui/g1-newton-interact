# zspace: distilling the grasp policy into a latent action space

`src/zspace.py` + `prior_distill/distill/distill_zspace.py`. Written 2026-09-18 against the K0_V_G tree
(`/workspace/g1-eigen` + `PYTHONPATH=/workspace/mjlab-eigen/src` on the 2x5090 box); the
interfaces it depends on are the ones in `docs/ASTRA_BASE_AND_TRAINING_HANDOFF.md`.

## What it does

The trained grasp policy (frozen ASTRA trunk + residual, 1328-D features -> 69-D `final_action`)
is the **teacher**. The **student** is a conditional VAE over the same 69-D action:

```
encoder  q(z | s, g)   s = proprioception, g = goal/object/reference groups    -> 48-D z
prior    p(z | s)      proprioception only
decoder  a = D(s, z)   proprioception + z                                       -> 69-D action
```

`s` never contains the goal. Everything the action needs to know about the task has to travel
through `z`, which is what makes `z` a usable action space afterwards: a downstream policy outputs
48 numbers, `z = prior_mu(s) + a_policy`, the frozen decoder turns that into the 69 joint targets
(`ZSpaceVAE.act_from_latent`). `a_policy = 0` is "what the prior expects a body in this state to do".

Training is DAgger, not RL: the student drives the simulator, the teacher labels every state the
student reaches with its deterministic mean action on the same observation, and the loss is
supervised. The method is the PULSE-X distillation from Omnigrasp (`humanoid_im_distill.py` +
`amp_agent.py:_optimize_kin`, `only_kin_loss=True`), ported term for term:

| term | definition | default |
|---|---|---|
| action loss | `mean_b ‖a_pred − a_teacher‖₂` (RMSE, not MSE) | 1 |
| KL | `KL(q(z\|s,g) ‖ p(z\|s))`, analytic, both logvars clamped to [−5, 2] | 0.01 → 0.001 linear over the run |
| AR(1) | `mean ‖μ_t − 0.99·μ_{t−1}‖₂` over consecutive steps of one env | 0.005 |
| regu | 0.001·(μ², logvar²) on posterior and prior | 0 (PULSE default off; 0.005 turns it on) |

The reparameterisation noise sampled when the student *acted* is stored and reused in the update,
so the gradient goes through the exact `z` that produced the executed action. Inputs are
normalised by a running mean/std updated from each collected batch. Network shapes are PULSE's:
encoder/prior trunks (1536, 1024, 512) → 5·E → μ/logvar heads, decoder (3096, 2048, 1024), SiLU;
18.0 M parameters at E = 48.

## Two things the port had to add

1. **The startup override.** For the first 36 control steps of every episode the action term
   replaces the joint target with the reference pose (`Sonic53Action.apply_actions`,
   `episode_length_buf <= 36`). The teacher's label there is never applied and carries no
   information, so those samples are masked out of every loss term. This is why iteration 0
   after a full reset reports "no valid labels" — every env is inside the window.
2. **Our teacher cannot get up.** PULSE's teacher was distilled with fall-recovery episodes; ours
   has never seen a fallen humanoid, and a fresh student falls. `--beta-start/--beta-end` let the
   teacher drive a fraction of the envs early on (drawn per episode, annealed to 0). The label is
   always the teacher's; the observation is always what actually happened.

The env-side action cache the observation groups read back (`_residual_last_final_action`,
`_residual_last_astra_action_pkl`, `proprio_history[-69:]`) is published for the **executed**
action, not the teacher's — `publish_executed()` re-derives the ASTRA-native last action from the
executed body action with `sync_last_astra_action_from_mjlab` before calling the runner's
`_set_residual_action_stats`. Otherwise `astra_obs[5]` would claim the teacher's action was
applied while the student's was.

## Inputs

| | groups | dims |
|---|---|---|
| `s` (`--self-groups`) | `proprio_history` | 646 = 4 × (body q, qd, hand q, qd, root vel, gravity) + root z + last executed action |
| `g` (`--task-groups`) | `astra_obs, reference_phase, reference_preview, object_state, hand_object_geometry, object_future, contact_features, tracking_error` | 611 |

Left out of `g` on purpose: `tracker_action` and `last_residual` are the teacher's own internal
quantities (what ASTRA alone would do, and the teacher's last residual). A downstream policy has
no such thing, and the encoder is only ever used during distillation. `astra_obs[0:93]` duplicates
proprioception; that is harmless.

## Two lines

| | line 1: grasp (`--mode full`) | line 2: body tracking (`--mode body`) |
|---|---|---|
| teacher | residual grasp checkpoint (`--resume`), 69-D | frozen ASTRA alone, 29-D; hand executed at zero |
| clips | the teacher's own 8 GRAB clips with real objects | 1324 GRAB clips, body only: `prior_distill/distill/make_body_clips.py` parks the object 15 m away, pins cf to the last frame (the office-walk recipe), drops everything else (17 GB → 144 MB) |
| `g` | object / reference / contact groups (611) | `astra_obs[93:136]` — ASTRA's own reference command (43) |
| terminations | the task's | `--keep-terminations fell_over,fell_over_early,time_out` |
| config | `configs/train/best.yaml` | `prior_distill/distill/body_track_distill.yaml` |
| where | 2x5090 box GPU 1, `/workspace/distill_full.sh`, out `/workspace/zspace_K0VG_8clip` | 4x5090 box GPU 1, `/workspace/distill_body.sh`, out `/workspace/zspace_body_grab` |

Line 2 is PULSE-X's own setting (AMASS bodies, no objects) with GRAB standing in for AMASS. In
body mode the label comes from the residual actor with **no** checkpoint loaded: its residual head
is zero-initialised, so its output is exactly ASTRA's — asserted at startup against the bare base
tracker (`body mode check: ... diff 0.0e+00`). Running the whole actor rather than the base alone
keeps the env's `_residual_*` metric plumbing at the right shapes (the bare base tracker left them
at their `(1, ·)` construction defaults and `MetricsManager` refused the step).

`--reference-list <file>` takes one pkl per line (1324 paths do not belong on a command line) and
`--sdf-object-all <stl>` pairs every clip with the same parked mesh; `num_envs` must be a multiple
of the clip count (1324 envs = one per clip).

## Running

Same launch shape as `train_newton.py` — same `--config`, `--reference-pkls`, `--sdf-objects`,
same environment variables (the config's `env:` block is exported before mjlab is imported, as
there) — plus `--resume <teacher checkpoint>` and `--out-dir`. A config's `args:` keys that
mean nothing here (`iterations`, `log_root`, `run_name`) are ignored with a note.

```bash
python prior_distill/distill/distill_zspace.py --config configs/train/best.yaml \
    --reference-pkls "$PKLS" --sdf-objects "$STLS" \
    --num-envs 1024 --resume logs/rsl_rl/g1_residual_interact/K0_V_G/model_13100.pt \
    --out-dir logs/zspace/K0_V_G --distill-iters 4000 --horizon 32 \
    --beta-start 1.0 --beta-end 0.0 --beta-anneal-iters 500
```

On the 2x5090 box `/workspace/distill_smoke.sh` wraps exactly this with K0_V_G's own environment
(`launch_k0vg.sh`'s `env -i` block), `OUT=... NENV=... /workspace/distill_smoke.sh <extra flags>`.

Outputs in `--out-dir`: `log.jsonl` (one row per iteration, `eval: true` rows for the evals),
`student_<iter>.pt` / `student_latest.pt` (self-contained: config, weights, normaliser stats,
group names and dims), `distill_config.json`.

Per-iteration line: action RMSE split body/hand, KL and its current weight, AR(1), `beta`, the
fraction of valid labels, the RMSE between student and teacher on the student-driven envs, episode
ends, and whatever `lift_success` the env reports. `[eval]` lines are a student-only deterministic
rollout (`z = μ_q`, `beta = 0`): body/hand RMSE against the teacher and the lifts the *student*
achieves — that is the number that says whether the distillation worked.

Loading a student elsewhere:

```python
from zspace import load_student
student, meta = load_student("student_latest.pt", "cuda")
a = student.act_from_latent(s, a_latent)     # downstream: 48-D latent action -> 69-D action
a = student.act_from_prior(s)                # diagnostic: the prior's mean behaviour
```

## Smoke test, 2026-09-18

K0_V_G `model_13100.pt`, 8-clip mix, 64 envs, GPU 1 of the 2x5090 box, 12 iterations × 32 steps,
2 epochs, `beta` 1 → 0 over 6 iterations, evals of 120 student-only steps at iterations 6 and 12.

- 64 worlds build and run at ~7.5 s per 32-step iteration (~270 env-steps/s; the step is
  clock-bound, so more envs are nearly free — 256 envs cost ~9 s).
- Teacher sanity: action std over time 0.37 (not the constant-action observation-cache trap).
  Teacher-driven envs reproduce K0_V_G's known per-clip lifts (cube, phone, camera).
- Label masking behaves: valid fraction 0.00 (iteration 0, all envs in startup) → 0.84 → 1.00 →
  0.45 after a wave of episode ends → back up.
- Loss falls from the first update: action RMSE 9.10 → 7.13 over iterations 1–5 (body 8.63 → 6.68,
  hand 2.80 → 2.46), KL weight anneals 0.0100 → 0.0063, AR(1) reported.
- Checkpoints save and reload (`load_student`), normaliser counts carried, `act_from_latent`
  returns 69-D and the latent moves the action.

- Whole run: 12 iterations in 2.6 min including two 120-step evals; three checkpoints written.
- Student-only evals (`z = μ_q`, student drives all 64 envs): body RMSE 12.47 → 6.80, hand
  4.17 → 2.31 between iterations 6 and 12; mean episode length ~105 steps, i.e. the barely
  trained student already stands and approaches for as long as the teacher's episodes last
  (the teacher's own end at ~95–105 by `og_object_far`). Student lifts are ~0 at this point, as
  expected after 12 iterations — the teacher needed 13 100 PPO iterations.

Iteration 8 shows the DAgger regime once `beta` hits 0: 50 episodes end in one iteration (the
student's own mistakes), the label RMSE on those fresh states jumps (10.3), and the next
iteration is back at 7.5. That is the intended loop — the student is being labelled on its own
state distribution.

A 300-iteration, 256-env run (`/workspace/distill_demo`, `lr 1e-4`, `beta` 1 → 0 over 100
iterations, 300-step evals every 50) was started right after the smoke to show the curve over a
longer horizon; its `log.jsonl` is the thing to read for the trend.
