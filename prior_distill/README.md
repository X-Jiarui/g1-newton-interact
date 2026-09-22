# prior_distill — a latent action prior for the G1 grasp policy (PULSE-X / Omnigrasp port)

Everything from the September 2026 exploration of *"can we drop the body reference and drive the
humanoid through a distilled latent prior, the way Omnigrasp does?"* lives here, plus the one
library module it needs (`src/zspace.py`, kept in `src/` because the mjlab actor imports it by
name). Two stages, mirroring PULSE-X → Omnigrasp:

1. **Distill** (`distill/`): DAgger-distil the frozen ASTRA body tracker into a 48-d
   conditional VAE — encoder `q(z|s,g)`, prior `p(z|s)`, decoder `D(s,z)` — on 1324 GRAB body
   clips. `s` = proprioception only, `g` = ASTRA's reference command. Result: a prior that keeps
   the humanoid standing and calm with no goal at all, and a decoder that turns 48 numbers into
   29 joint targets. Details, losses, and the run log: [`distill/DISTILL.md`](distill/DISTILL.md).
2. **RL through the latent** (`rl/`): the grasp policy outputs a 48-d residual on the prior's
   mean, the frozen decoder produces the body action, hand held at zero. Seven arms were trained
   on the same recipe to isolate three questions: *is z a better residual space than ASTRA's
   hidden token?* (no — a tie), *does anchoring on a goal-free prior work?* (yes, once
   exploration happens in z), and *does the policy need the body reference at all?* (no).

The arm to keep is **`ZS_ZP_ZN_NOREF`**: a policy that sees no reference body motion, only
proprioception plus the object, and matches or beats every reference-conditioned arm at equal
budget. It is the first policy in this repo that could run from an object trajectory alone.

## The arms, and how ACT / ZS_Z / ZS_ZP relate

All arms share the main-train recipe (`PD10_WIDE_S`: `HAND_PD=10,0.2`, `RSI_ANCHOR_CF=1`,
`RSI_CF_OFFSET_START=-1000`, `RSI_CF_OFFSET_END=-20`, `WRIST_TARGET_FAR=0`, `configs/train/best.yaml`,
`staged_cf_r24` rewards, 4096 envs, seed 1) with the hand network switched off
(`fixed_hand_action_frame: 0`, `hand_init_std: 1e-4`, `FIXED_HAND_ZERO=1`), so the comparison is
body-only. Each arm changes exactly one thing relative to the previous row.

| arm | policy output | anchor at zero residual | goal in the anchor? | exploration | optimiser | actor sees body reference? |
|---|---|---|---|---|---|---|
| `ZS_ACT` | 512-d delta on ASTRA's hidden token (the main train) | ASTRA(obs) | yes (ASTRA's reference command) | Gaussian over the 69-d action | ours (lr 5e-5 adaptive, entropy 5e-4) | yes |
| `ZS_Z` | 48-d delta on z | encoder mean μ_q(s, g) | yes (g = reference command) | action space | ours | yes |
| `ZS_ZP` | 48-d delta on z | **prior mean μ_p(s)** | **no** | action space | ours | yes |
| `ZS_ZP03` | as ZS_ZP, residual clip 0.3 instead of 1.0 | prior | no | action space | ours | yes |
| `ZS_ZP_ZN` | 48-d delta on z, **Gaussian over z** (σ = e⁻¹ fixed, PPO ratio in z) | prior | no | **z space** | PULSE (lr 2e-5 fixed, entropy 0) | yes |
| `ZS_ZP_ZN_OURLR` | as ZS_ZP_ZN | prior | no | z space | ours | yes |
| **`ZS_ZP_ZN_NOREF`** | as ZS_ZP_ZN_OURLR | prior | no | z space | ours | **no** |

How the three named spaces relate:

- **ACT** is the residual-on-ASTRA design this repo trains by default. The frozen ASTRA tracker is
  goal-conditioned: given the reference command it already produces a tracking action, and the
  policy only nudges its hidden token. The anchor does the tracking for free.
- **ZS_Z** swaps the 512-d hidden-token residual for a 48-d residual in the distilled latent, but
  keeps a goal-conditioned anchor (the encoder, which sees `g`). It is the same design in a smaller
  space, and it trains identically to ACT (a tie within ±5 % through 500 iterations; videos are
  indistinguishable). Conclusion: the latent is neither a better nor a worse residual space when the
  anchor tracks.
- **ZS_ZP** is Omnigrasp's setting: the anchor is the prior `μ_p(s)`, which has never seen a goal,
  so *every* task-relevant decision has to travel through the 48-d residual. The reward pays
  nothing for tracking (`tracking 0`, `staged_lower_tracking 2`), so ZS_ZP tracks loosely and
  spends the freedom on the object — it came out ahead of ACT/Z on reward but was unstable when
  the exploration noise stayed in action space (five collapses of 30–75 % in 900 iterations, KL
  2–4× the target, ratio spikes to 12–14). Shrinking the clip (`ZS_ZP03`) made it slower and it
  collapsed anyway.
- **ZS_ZP_ZN\*** fixes that by sampling in z, as PULSE does: the policy's Gaussian is over the
  latent residual with a fixed σ = 0.37, the sampled z is decoded deterministically, and PPO's
  ratio is computed in z. With PULSE's lr (2e-5 fixed) it learns 2.5× slower than our optimiser;
  with ours (`_OURLR`) it is monotone, never collapsed in 700 iterations, and reaches ZS_ZP's
  best reward in 40 % of the iterations.
- **NOREF** removes `reference_phase`, `reference_preview` and `tracking_error` from the actor's
  input (the critic is unchanged). The actor's 1015-d input is then `proprio_history`,
  `tracker_action` (the prior's decoded base action — goal-free in the z arms), `object_state`,
  `hand_object_geometry`, `object_future` (the object's reference trajectory, which is the task
  specification, as in Omnigrasp), `contact_features`, `last_residual`, `last_final_action`.
  Nothing in that list encodes the human body motion.

## NOREF vs ZS_ZP

Same prior, same decoder, same rewards, same envs and seed. Two differences: NOREF explores in z
(like `ZS_ZP_ZN_OURLR`) and it cannot see the reference body motion. Numbers are 10-iteration
means from the training logs on the Moonlight H200 (gpu1), 2026-09-21/22.

**At equal iteration**

| iteration | arm | reward | obj<5cm | obj<15cm | wrist err (m) | body err (m) | contact | falls/it |
|---|---|---|---|---|---|---|---|---|
| 300 | ZS_ZP | 854 | 0.825 | 0.938 | 0.133 | 0.096 | 0.313 | 0.00 |
| 300 | ZS_ZP_ZN_OURLR | 1003 | 0.942 | 0.968 | 0.133 | 0.092 | 0.333 | 0.00 |
| 300 | **ZS_ZP_ZN_NOREF** | **1015** | **0.943** | 0.969 | 0.125 | 0.089 | 0.336 | 0.00 |
| 500 | ZS_ZP | 1018 | 0.953 | 0.996 | 0.131 | 0.086 | 0.350 | 0.00 |
| 500 | ZS_ZP_ZN_OURLR | 1250 | 0.977 | 0.993 | 0.121 | 0.081 | 0.359 | 0.00 |
| 500 | **ZS_ZP_ZN_NOREF** | **1309** | **0.990** | 1.000 | 0.117 | 0.081 | 0.371 | 0.00 |
| 590 | ZS_ZP | 769 (in a collapse) | 0.677 | 0.996 | 0.125 | 0.083 | 0.294 | 0.00 |
| 590 | ZS_ZP_ZN_OURLR | 1317 | 0.984 | 0.998 | 0.119 | 0.079 | 0.362 | 0.00 |
| 590 | **ZS_ZP_ZN_NOREF** | **1416** | **0.993** | 1.000 | 0.116 | 0.079 | 0.375 | 0.00 |
| final | ZS_ZP @910 | 1240 | 0.944 | 1.000 | 0.122 | 0.085 | 0.375 | 0.00 |
| final | ZS_ZP_ZN_OURLR @705 | 1409 | 0.994 | 1.000 | 0.115 | 0.077 | 0.372 | 0.00 |
| final | **ZS_ZP_ZN_NOREF @594** | **1425** | 0.992 | 1.000 | 0.116 | 0.079 | 0.375 | 0.00 |

**Best reward reached**

| arm | best reward | at iteration | collapses (>15 % drop) |
|---|---|---|---|
| ZS_ACT (stopped) | 972 | 572 | 0 |
| ZS_Z (stopped) | 979 | 552 | 0 |
| ZS_ZP | 1288 | 870 | 5 |
| ZS_ZP_ZN_OURLR | 1445 | 696 | 0 |
| **ZS_ZP_ZN_NOREF** | **1422** | **568** | 0 |

`obj<5cm` is the sticky per-episode "fingertip ever within 5 cm of the object" fraction,
`contact` is the physical-contact fraction, `body err` the mean body-link tracking error against
the reference. Read the table this way:

- **The reference input buys nothing.** NOREF tracks OURLR (same design plus the reference) to
  within noise at every iteration, and is 5–9 % ahead from iteration 450 on. The residual policy
  had the reference available and did not use it; removing it did not slow learning.
- **NOREF beats ZS_ZP by 25–30 % at equal budget and reaches ZS_ZP's all-time best in 500
  iterations instead of 870**, because sampling in z is what makes the goal-free anchor trainable.
  ZS_ZP with action-space noise was the same anchor with a 2–4× too large PPO step.
- **Tracking error is lower without the reference**, not higher (body 0.079 vs ZS_ZP's 0.083–0.096
  m; wrist 0.116 vs 0.125–0.133 m). The policy has to stay near the human motion to reach the
  object, and the prior keeps it there; the reference input was not what held the body on track.
- **Beyond the training horizon** (300-step rollouts, no early termination, `rl/vid_zs.sh`,
  4x5090 box, 2026-09-21): NOREF@220 and OURLR@290/@330 approach the object in 62–66 % of envs at
  steps 80–119, lose it when the table drops (the hand is held at zero, so nothing can be
  grasped), and only 4–9 % of envs have fallen by step 300. ZS_ZP@390 on the same clips: 77 %
  fallen. The z-sampled policies degrade gracefully past what they were trained on; the
  action-sampled one does not. NOREF uses a smaller z residual (norm 1.3–1.8 vs 1.5–3.3), i.e. it
  leans more on the prior for the same approach.

What NOREF does *not* prove: nothing here grasps (hand fixed at zero) and nothing here is scored
past the table drop, so lift is untested. The reward still has no term for tracking, so a
goal-free policy is free to drift once the object is reached; a `tracking` term would be the
first thing to add before trusting it on a longer clip.

## Where the artefacts are

Nothing large is in git. Persistent copies (Lustre, shared across the Ropedia nodes):
`/mnt/ddn/jrxu/zspace_prior/` — `MANIFEST.md5` lists everything.

| item | path |
|---|---|
| distilled student (line 2, arm B, iteration 4000) | `student_B4000.pt` (md5 `cb8bacea…`), also `/workspace/zspace_body_grab_B/student_004000.pt` on the 4x5090 box |
| re-basing constants for the 1324-clip mix (see `ZSPACE_REF_DEFAULT` below) | `ref_default_1324.npz` |
| NOREF checkpoints | `ckpts/ZS_ZP_ZN_NOREF_590.pt` (final), `ckpts/ZS_NOREF_220.pt` (the filmed one) |
| OURLR / ZP checkpoints | `ckpts/ZS_ZP_ZN_OURLR_700.pt` (final), `ckpts/ZS_OURLR_290.pt`, `ckpts/ZS_OURLR_330.pt`, `ckpts/ZS_ZP_910.pt` (final), `ckpts/ZS_ZP_900.pt` |
| the three agent configs | `agents/agent_{handoff,pulse,noref}.yaml` (same files as `rl/agents/` here) |
| final checkpoints (runs stopped 2026-09-22: ZS_ZP @910, OURLR @705, NOREF @594) | `ckpts/ZS_ZP_910.pt`, `ckpts/ZS_ZP_ZN_OURLR_700.pt`, `ckpts/ZS_ZP_ZN_NOREF_590.pt`; run dirs on gpu1 `~/zs/h200/g1-newton-interact/logs/rsl_rl/g1_residual_interact/ZS_*` (last 5 checkpoints each) |
| body clips for the distillation | 4x5090 box `/workspace/grab_body_clips/` (`clips.txt`, `meshes/cubesmall.stl`); rebuild with `distill/make_body_clips.py` |
| rollout videos | 4x5090 box `/workspace/vidshot/ZS_*_{phone,camera}.mp4` + per-step `.csv` timelines |

## Layout

```
prior_distill/
  distill/
    distill_zspace.py        DAgger distiller (--mode full: grasp teacher; --mode body: frozen ASTRA)
    make_body_clips.py       grab_g1_wuji_aligned -> body-only clips (object parked 15 m away)
    body_track_distill.yaml  env recipe for --mode body
    DISTILL.md               method, losses, the two lines, results
  rl/
    mjlab_overlay/residual_interact/
      residual_actor.py      the actor with the z-space anchor / decode / sampling-in-z
      rl.py                  runner: executed-action stats for latent-D actions, token-clip fix
    train_newton.py          ZSPACE_SAMPLE_Z env wrapper, rollout decode, ROLLOUT_TIMELINE
    patches/*.diff           the same three files as unified diffs against the gpu3 running tree
    agents/                  agent_handoff (ours) / agent_pulse (PULSE lr) / agent_noref (no reference)
    go.sh                    the launcher used for every arm (gpu1 paths)
    vid_zs.sh                no-termination rollout + video for any arm (4x5090 box paths)
    monitor/                 log parsers: per-arm status table, PPO internals, reward-term split
src/zspace.py                ZSpaceVAE: encode / prior / decode, kin_loss, load_student
```

`rl/mjlab_overlay/` and `rl/train_newton.py` are **whole-file copies of the gpu3 running tree as of
2026-09-20 plus the z-space changes**, not of `tools/setup/mjlab_overlay/` at this commit (the two
trees had diverged before this work started). To port the changes onto another base, apply
`rl/patches/*.diff` — they contain only the z-space additions. `train_newton.py` additionally
carries the rollout-loop fix that decodes a latent action before stepping the raw env (without it
a `ZSPACE_SAMPLE_Z` checkpoint crashes with `size of tensor a (69) must match ... (48)`).

## Running an arm

`rl/go.sh <run-name> <gpu> <seed> [VAR=val ...]` reproduces the recipe; the z-space arms are its
extra variables:

| variable | meaning |
|---|---|
| `ZSPACE_STUDENT=<student.pt>` | enable the latent: the actor loads the frozen VAE and the residual lives in z |
| `ZSPACE_REF_DEFAULT=<npz>` | re-base `proprio_history`'s joint columns to the clip set the student was trained on (required whenever the RL clip list differs from the distillation list — the default pose is a dataset constant) |
| `ZSPACE_ANCHOR=encoder\|prior` | zero-residual anchor: μ_q(s,g) (ZS_Z) or μ_p(s) (ZS_ZP and all ZN arms) |
| `ZSPACE_TOKEN_CLIP` | clip on the z residual, default 1.0 (0.3 = ZS_ZP03) |
| `ZSPACE_SAMPLE_Z=1` | Gaussian over z (σ `ZSPACE_Z_STD`, default e⁻¹; `ZSPACE_LEARN_Z_STD=1` to learn it); PPO in z |
| `FIXED_HAND_ZERO=1` | hand action held at zero (the student is body-only) |
| `AGENT=<dir with params/agent.yaml>` | agent config: `agents/agent_handoff` (ours), `agent_pulse` (lr 2e-5 fixed), `agent_noref` (no reference groups) |

```bash
ZS="FIXED_HAND_ZERO=1 ZSPACE_STUDENT=$H/student_B4000.pt ZSPACE_REF_DEFAULT=$H/ref_default_1324.npz"
go.sh ZS_ACT           0 1                                                  # main train, hand off
go.sh ZS_Z             1 1 $ZS ZSPACE_ANCHOR=encoder
go.sh ZS_ZP            2 1 $ZS ZSPACE_ANCHOR=prior
go.sh ZS_ZP_ZN         3 1 $ZS ZSPACE_ANCHOR=prior ZSPACE_SAMPLE_Z=1   AGENT=$H/agent_pulse
go.sh ZS_ZP_ZN_OURLR   4 1 $ZS ZSPACE_ANCHOR=prior ZSPACE_SAMPLE_Z=1
go.sh ZS_ZP_ZN_NOREF   5 1 $ZS ZSPACE_ANCHOR=prior ZSPACE_SAMPLE_Z=1   AGENT=$H/agent_noref
```

Evaluating a z-space checkpoint needs the same variables (the actor is rebuilt from them) and, for
NOREF, `--agent-cfg-from` pointing at `agent_noref` — the checkpoint does not record its feature
groups. Verify any eval by the log line `loaded residual config: ... token_clip=1.0` and by the
`tok_norm` column of `ROLLOUT_TIMELINE` (a z residual of norm ~0.5 instead of ~2–4 means the
runner's config clip was re-applied on load and the policy is running as the bare prior).

## Things that cost a day each

- `proprio_history` subtracts the **mean pose of the loaded clip set**, so a student distilled on
  1324 clips reads shifted joints on an 8-clip mix and tracks 2× worse. `ZSPACE_REF_DEFAULT`
  re-bases the columns. `astra_obs` is immune (literal default).
- The runner re-applies the agent config's `token_residual_clip` (0.1, a hidden-token unit) on
  construction and on checkpoint load, over the actor's own z clip. Every z-space eval before the
  fix ran the bare prior. `rl.py` now takes the clip from `ZSPACE_TOKEN_CLIP` when a student is set.
- The runner keeps 5 checkpoints; copy one aside the moment an equal-iteration comparison is planned.
- The distillation's root accuracy comes from the 0.25 m drift reset (PULSE's
  `terminationDistance`), not from teacher-mixing (`beta`); ASTRA itself drifts 20 cm in 4 s on
  GRAB body clips, which caps every absolute number.
