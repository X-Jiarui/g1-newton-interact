# Rounds R28-R31: moving the task toward Omnigrasp's shape

Everything here was read off `ZhengyiLuo/Omnigrasp` (`phc/env/tasks/humanoid_omnigrasp.py` and
`phc/data/cfg/env/env_x_grab_z.yaml`) and measured against our own runs. Each switch is an
environment variable defaulting to OFF, so a run that does not set one is unchanged.
Launcher: `tools/run/launch/r30_omnigrasp_shape.sh`.

## What Omnigrasp actually does

**It does not hand the policy a lift.** The object is a full rigid body under gravity
(`fix_base_link = False`, `density = 1000`); withdraw the table and an unheld object falls and the
episode ends. A claim that removal makes the lift free is wrong.

**It does not teleport the hand onto the object either.** `_set_env_state` writes the humanoid's
root pose, every DOF and every rigid-body pose from the motion library, and `_reset_target` writes
the object and table from the same frame -- but the sampled time is uniform over the whole motion,
then clamped to `contact_time - 0.5 s`. The mass above the clamp piles onto that one instant; the
rest genuinely covers the approach, frame 0 included. And `flags.test` sets `motion_times[:] = 0`,
so **evaluation always starts at the beginning and the humanoid walks the whole way** -- which is
what the demo videos show.

**The four things that make removal survivable:**

1. The object target does not move before removal:
   `TrajGenerator3D(..., starting_still_dt=table_remove_frame * dt)`. There is no window in which
   "the reference lifted the object and the policy did not".
2. Removal is at a **fixed 45 frames (1.5 s at its 30 Hz)**, not synced to contact. The line that
   syncs it, `table_remove_frame = (contact_time - motion_times)/dt`, sits inside
   `if use_stage_reward:` which `env_x_grab_z.yaml` never sets. So the clamped cohort has contact
   at 0.5 s and support withdrawn at 1.5 s: **1.000 s of margin**.
3. One termination, `grab_termination_distance = 0.12`, object against its reference. No fall check.
4. Holding is the only source of stage-2 reward: every tracking term is multiplied by
   `contact_filter`, plus `w_conctact = 0.1` per step for contact alone.

Under-reported elsewhere and worth keeping in view: `obj_rand_pos: True`, `episode_length: 900`
(30 s), `power_reward` + `penality_slippage`, an object curriculum (`auto_pmcp_soft`, `obj_pmcp`),
3072 envs, and a policy that acts only through a pretrained 48-d VAE latent -- a constraint, not
only a freebie.

## What we measured on our side

**R28** (fixed alpha 0.85 data + removal at cf). `object_reference_window` at 0.05 m took
**70-100 % of every reset**, and fired before the grasp was due: the reference lifts the object
20-40 frames ahead of cf while the real one stays on the table. `time_out`, `fell_over`,
`wrist_target_far`, `no_progress`, `object_drift`, `object_leash` were all exactly 0.

**R29** (that window off, og_object_far alone, reference frozen until cf, RSI at cf-20..-10).
`og_object_far` became the only terminator at ~100 % of resets, and `ep_len` pinned:

    36 startup + 15 approach (RSI mean cf-15) + 7.8 free fall = ~59 steps

7.8 is `sqrt(2*0.12/9.81) / 0.02` -- the object clearing the 0.12 m threshold at our 50 Hz.
Over eleven consecutive monitoring cycles `contact` rose every single time (0.075-0.117 ->
0.19-0.25) while `ep_len` never moved. Touching improved monotonically; holding did not improve
at all, because the policy got ~5 control steps of post-cf experience per episode, ~8 % of samples.

**R30** (`TABLE_REMOVE_AFTER_CF=50`, i.e. 1.000 s, Omnigrasp's own margin at our rate). One switch:

| | R29 last | R30 |
|---|---|---|
| ep_len | 50.9 / 58.3 / 56.9 / 59.9 | **81.1 / 105.9 / 101.0 / 93.1** |
| contact | 0.191 / 0.156 / 0.185 / 0.247 | **0.315 / 0.414 / 0.296 / 0.420** |
| atcf | 0.145 / 0.163 / 0.166 / 0.181 | **0.427 / 0.532 / 0.524 / 0.468** |
| lift_success | 0 / 0 / 0 / 0 | 0 / **0.0129** / 0.0006 / 0.0016 |

(order: apple / binoculars / camera / toothpaste)

Binoculars held non-zero on **fourteen consecutive records**, 1.2-1.9 %. The lifts happen while
the table is still there -- `tblrm` is 0.001-0.075, cf lands at step ~51 and removal at ~101 while
ep_len is 81-106 -- so they are real grasps, not the withdrawal doing the work.

## R31: the pregrasp reward, ported whole (written, not yet run)

`compute_pregrasp_reward_time` does not score fingertip-to-object distance. It scores every body in
`hand_bodies` -- per hand the wrist plus three joints of each finger, 16 a side -- against that
body's **world pose at the contact frame, position and orientation both**. Our `staged_tip_cf`
scored five fingertip positions, and the same five points are reachable with quite different finger
curls, so the hand SHAPE was unconstrained.

`staged_hand_cf_reward` ports it: Wuji maps one for one, `palm_link` for the wrist and
`finger{1..5}_link{2,3,4}` for the three joints (`link1` has zero travel, `_tip` is a geomless
derived frame). Verified with `tools/probes/hand_body_mask.py`: all 32 bodies resolve, and the
0.20 m mask selects **exactly 16 -- one hand's worth, the right one -- on all four clips**, with no
side flag, exactly as the original does.

Two corrections worth keeping:

* **There is no seam at `close_distance`.** Their position error is `(diff**2).mean(dim=-1)`, a
  mean over the three XYZ components, i.e. `|d|^2/3` -- an implicit `k_pos/3`. At 0.20 m that pays
  `exp(-100*0.04/3) = 0.264` against a far-field ceiling of 0.100, so crossing inward is a 2.4x
  raise. Reading it as `exp(-100*|d|^2) = 0.018` invents a pay cut that is not there.
* **The endgame is flat**: 0.920 at 5 cm, 0.987 at 2 cm, so the last three centimetres buy 6.7 %.
  That is the shape already measured stalling this task once. `endgame_std > 0` mixes in the narrow
  Gaussian that fixed it; 0.0 keeps the port faithful. Watch `Metric/staged_hand_cf_dist`: if it
  settles at 3-6 cm and stops, turn it on.

Weights: `configs/rewards/staged_cf_r31.yaml` sets `staged_tip_cf: 0.0` and `staged_hand_cf: 5.0`
and is otherwise byte-identical to R24 -- same job, same [0,1] range, so the total reward scale and
every other term's share are unchanged and the rounds stay comparable.

## Reading the metrics (two traps paid for here)

* **`atcf` and `arrF` are time shares, not coverage.** They go through `_safe_log`, which writes a
  per-control-step scalar that the logger averages over the episode, so they read "what fraction of
  the episode was already past cf / already arrived", not "what fraction of envs got there".
* **`ep_len` is not comparable across RSI settings.** With RSI at 0-50 the episode contains 22-92
  frames of approach; with `RSI_ANCHOR_CF` the pre-cf part is at most 36 + 20 = 56 steps, so 47-62
  there is the floor, not a collapse. Restate any ep_len threshold whenever RSI moves.
* `tipcfD`'s published floors were measured on the adaptive-alpha data and do not transfer to
  re-solved clips.
