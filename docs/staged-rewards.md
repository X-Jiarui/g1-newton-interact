# Staged rewards on the contact frame — handover

## What changed, in one line

Four of the ten weighted reward terms are replaced by cf-staged equivalents and one new term is
added, so that the reward stops asking the policy to track an arm trajectory that stops being data
after the grasp, and starts reading the finger joints the retarget now solves for.

## Where the code is

| | |
|---|---|
| reward term implementations | `mjlab-astra-dagger-distill-20260625/src/mjlab/tasks/residual_interact/staged_mdp.py` |
| term registration (params, weight 0) | same repo, `residual_interact/env_cfgs.py` |
| **the weights that actually train** | `g1-newton-interact/configs/rewards/staged_cf.yaml` → copy into `<ckpt>/params/env.yaml` |

The Newton repo supplies physics and the bridge only. `src/newton_vec_env.py` imports mjlab's
`RewardManager` and the `residual_interact` terms, and `docs/05-reproduce.md:9` pins the mjlab source
to the dagger-distill checkout (`pip install -e`). Editing any other mjlab tree has no effect on
training — including `~/mjlab-src` (mjlab-astra-interact), which is the older mjlab-only line.

**Weights only take effect through the env.yaml override.** `tools/run/train_newton.py:107` calls
`apply_reward_weights(cfg, reward_weights_from_env_yaml(<ckpt>/params/env.yaml))`, which overwrites
the base cfg. A term registered in `env_cfgs.py` but missing from env.yaml trains at the base weight.

## Why stage at all

The reference became accurate in two different ways, in two different places, and one term cannot
spend both:

| | before cf | after cf |
|---|---|---|
| what is data | the whole arm and hand — fingertips on the human's own contact points (median 1.6 cm, taken from the SMPL-X **mesh**), palm within 12.3° of the human's hand frame against 41.9° for raw GMR, hand closing over the 30 frames before cf | the object trajectory |
| what is a guess | — | the arm: one solved frame carried by a constant joint offset |

Tracking the arm after cf therefore fights the object reward wherever the carry is imperfect, which
is everywhere. `cf` is GRAB's own first-contact label, resampled with the reference and kept per clip.

## The swap, term by term

| out | in | why |
|---|---|---|
| `tracking` 1.0 | `staged_lower_tracking` 1.0 | `tracking` is one Gaussian over the whole body, so the upper half cannot be released separately. The lower body stays always-on: GMR's feet carry the highest IK weights, and the upper body needs a base once free. |
| `right_wrist_tracking` 3.0 + `left_wrist_tracking` 0.5 | `staged_upper_tracking` 3.5 | same job, the whole upper chain rather than two wrist points, released after cf |
| `object_trajectory_tracking` 2.0 | `staged_object_tracking` 2.0 | same quantity plus two gates: active only after cf, and paid only when a fingertip is within 6 cm of the object |
| — | **`staged_hand_pose` 1.0 (new)** | nothing in the baseline reads the finger joints at all |
| unchanged | `multi_tip_surface` 5.0, `object_lift_hold` 2.0, `object_hard_lift` 2.0, `contact_duration` 1.0, `stability` 1.0, `omnigrasp_style` 0.5 | task terms, orthogonal to staging |

Ten weighted terms before, ten after; weight sum 18.0 → 18.5, deliberately close so the value scale
does not move and a matched-iteration comparison stays honest.

## `staged_hand_pose` — the one genuinely new signal

Step 5 of the data pipeline solves the 20 finger joints of the operating hand against the human's own
fingertips taken from the SMPL-X mesh; step 6 ramps them in over the 30 frames before cf. So the
reference hand is open while reaching and closed on the object at cf, and reproducing that joint
trajectory *is* reproducing the grasp.

Joint space rather than fingertip positions, for two reasons: no forward kinematics in the reward
loop, and the joint vector is what the retarget actually solved for. A fingertip position target
would re-introduce the hand-size mismatch — the Wuji hand is 1.24× a human hand — that the
mesh-target IK exists to absorb.

`post_weight` 0.2: after cf the hand must stay closed, and letting the term vanish invites the policy
to open it the moment the object term takes over.

## One recommendation against the brief

**Do not replace `multi_tip_surface`.** It looks like the natural counterpart to a staged fingertip
term, and it is the largest weight in the set, but it is not a simple distance: it carries contact,
grasp, force, drift-gate, top-k and opposition-gate sub-terms and measures to the object **surface**.
`staged_tip_object`, which is registered here, is a plain Gaussian on tip-to-object-**centre** with a
near-mask. Swapping them would lose five working sub-terms to gain one mask.

`staged_tip_object` is therefore left at weight 0. The mask in it — Omnigrasp's `close_hand_flag`,
scoring only the tips the reference brings near the object — is worth folding into
`multi_tip_surface` as a follow-up, but that is a second change and should not ride along with this
one.

## Two things taken from Omnigrasp, one deliberately not

**Taken — the contact gate.** `staged_object_tracking` pays nothing unless a fingertip is within
6 cm. Without it a policy is paid for standing next to an object that is where the reference says
because nobody moved it. This is Omnigrasp's `check_contact` and it is the load-bearing idea in that
codebase.

**Taken — per-sequence staging.** `staged_mdp.cf_local` reads each clip's own contact frame.

**Not taken — Omnigrasp's stage boundary.** Its shipping version switches at a fixed frame 30
(`grasp_start_frame`) compared against *absolute motion time*, so on a clip whose contact is late the
pregrasp reward never fires at all. Its own earlier `humanoid_omnigrab.py` used each sequence's
contact time. Ours must: cf lands at frame 47 on the median clip and as early as 22.

## Every post_weight is small, not zero

Omnigrasp drops its body term to exactly zero because its action space is a pretrained motion latent
that cannot produce a non-human pose. Ours is joint residuals on a frozen tracker — nothing
structurally stops the arm windmilling once the term is released. `staged_upper_tracking` post_weight
is 0.15, `staged_hand_pose` 0.2, `staged_tip_object` 0.3. Sweep these up from zero if the arm
misbehaves, rather than starting at zero.

## How to evaluate

This is an acceleration change, not a bug fix — the existing 10-term set does learn to lift
(`s9/flashlight` in R13 climbs seven rounds from iter 760 to 0.06). So it needs a **matched-iteration
A/B**: same clip, same seed, same iteration count, baseline env.yaml vs this one.

Watch these logs, all new:

```
Metric/before_cf                 fraction of steps in the pre-cf phase (sanity: should be ~cf/len)
Metric/cf_local                  the per-clip contact frame the gate resolved to
Metric/staged_hand_pose_err_deg  mean finger-joint error against the reference
ResidualReward/staged_*          the four staged terms
Metric/staged_object_contact_gate  fraction of steps the object term was actually paid
```

**First check before trusting any of it:** `Metric/cf_local` must be a per-clip constant in the
20–60 range. If it is 0 or equals `n_frames`, `gt_contact_frames` did not survive into the reference
and every staged term is running with a degenerate gate.
