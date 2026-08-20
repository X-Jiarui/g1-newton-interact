# The mjlab baseline Newton has to reproduce

## The checkpoint

`~/sweep_ckpts_r2/OF_00_apple_eat_1_SPHERE/model_7310.pt`

Chosen by measurement, not by filename. `~/sweep_rank.tsv` (the best-lift-first ranking from the
24-run sweep) puts `OF_00_apple_eat_1_SPHERE` at **lift_success 0.6450**, second of 24 and the only
apple run that lifts at all -- `OF_01_apple_eat_1` with a 6x3 boxstack is exactly 0.0000. Round 2's
checkpoint (7310 iterations) is the one verified below; round 1's `model_2010.pt` is the same run
earlier in training.

## Verified behaviour (mjlab, MuJoCo 3.8.1 / mujoco_warp 3.8.0.2)

8 envs, terminations off, forced reference start frame 0 -- the configuration
`~/viz/view_run.sh` uses:

| step | object z | rise | hand-object | contact | lift_success |
|---|---|---|---|---|---|
| 100 | 0.707 | +0.0 cm | 0.087 m | 0.00 | 0.00 |
| 120 | 0.707 | -0.0 cm | 0.035 m | 1.00 | 0.00 |
| 180 | 0.917 | +21.0 cm | 0.044 m | 1.00 | **1.00** |
| 240 | 1.203 | **+49.6 cm** | 0.045 m | 1.00 | **1.00** |
| 399 | 1.166 | +45.9 cm | 0.045 m | 1.00 | **1.00** |

The clip's grasp is at reference frame 76-78 (`global_min_frame=76`, `global_min_dist=0.0084 m`).
The first ~37 steps are the startup hold that replays the reference, after which `frame = step - 37`.
Peak lift 49.7 cm against a 3 cm success threshold, held to the end of the rollout, zero resets.

## The environment is not the default one

None of this reproduces under the task defaults. Required:

```
APPLE_HAND_KIND=wuji
APPLE_SCENE_Z_OFFSET=-0.03                  # shifts apple AND table down 3 cm
ASTRA_BASE_BACKEND=torch
GMR_ROOT=$HOME/jiarui/GMR
APPLE_EAT_PKL=$HOME/jiarui/scaled_grab_wuji_all_o70/s1/apple_eat_1.pkl   # o70, not the default set
```

and, in code, `--start-assist-gain 0.0`: play.py defaults the pelvis start-assist wrench to 1.5 for
120 steps, while every candidate trained with `tracking_start_assist_gain: 0.0`. The default props the
robot up with a wrench it never trained with.

The reference clip matters concretely: the default dataset puts this clip's closest hand-object
approach at frame 457, the o70 dataset at frame 76. A frame index means nothing without naming the
clip.

## The one line that decided everything

The first evaluations reported `lift_success` of exactly 0.0000 for every checkpoint, with the hand
stalling 0.62-0.73 m from the object. The checkpoints were fine. The harness was missing one line
that `play.py` has:

```python
policy = _maybe_wrap_residual_action_stats_policy(task_id, runner, policy)
```

That wrapper feeds `last_residual` and `last_final_action` back into the observation. Without it those
two groups stay stale, the policy acts on a world it is not in, and **nothing raises** -- it simply
looks like a policy that cannot grasp. Adding it took the same checkpoint from 0.00 to a sustained
1.00 with a 49.7 cm lift.

This is the failure mode the whole port has to be defended against, and it is why the Newton side
reuses mjlab's own observation assembly through an adapter rather than re-deriving 1328 dims: a
missing or stale group does not announce itself.

Calibration against `play.py` is what caught it. Run directly, play.py showed the wrist tracking the
reference to 1.5 cm at the grasp frame while the harness had the hand 0.73 m away -- two numbers that
cannot both describe the same rollout.

## Acceptance target for Newton

Same checkpoint, same clip and scene settings, stepped at dt = 0.005:

* contact established around reference frame 76-80
* object lifted well past the 3 cm threshold and **held** to the end of the rollout
* hand-object distance settling at 0.04-0.05 m during the carry

Exact per-step agreement is not the target and is not achievable -- Newton runs mujoco_warp 3.11
against mjlab's 3.8. The noise floor between those two versions is measured separately before any
divergence is attributed to the port.
