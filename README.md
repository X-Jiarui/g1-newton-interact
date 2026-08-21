# G1 Wuji — mjlab → Newton port

Runs an mjlab-trained residual grasping policy under [Newton](https://github.com/newton-physics/newton)
1.5 and reproduces mjlab's behaviour.

## Result

The `apple_eat_1` checkpoint, unchanged, driven by Newton's `SolverMuJoCo`:

| | mjlab @ mujoco_warp 3.8 | mjlab @ 3.11 | **Newton 1.5** |
|---|---|---|---|
| peak object rise | 49.72 cm | 50.13 cm | **50.60 cm** |
| min fingertip–object distance | 0.032 m | 0.035 m | **0.035 m** |
| contact established | frame ~85 | frame ~85 | **frame ~85** |
| hand–object during carry | 0.044–0.045 m | 0.044–0.045 m | **0.044–0.046 m** |
| lift held to end of rollout | yes | yes | **yes** |

The lift threshold is 3 cm; all three lift roughly 50. The mjlab 3.8 → 3.11 column is the control:
it is the same code and the same checkpoint with only the mujoco_warp version changed, and it moves
the answer by 0.4 cm. Newton sits 0.47 cm from mjlab 3.11, i.e. inside that noise floor.

Per-step agreement is not claimed and was never the target — two engines diverge within a few steps
of contact-rich motion. What is claimed, and measured, is that the same weights produce the same
behaviour.

## What is Newton's and what is mjlab's

| | |
|---|---|
| **Newton** | builds the model (`ModelBuilder.add_mjcf` → `SolverMuJoCo`) and advances it (`solver.step`, dt = 0.005), including collision and control application |
| **mjlab** | supplies the trained policy, the observation assembly (`ResidualFeatureGroupObs`, `AstraObs136`) and the action term (`Sonic53Action`) |
| **this repo** | `src/newton_bridge.py`, which lets mjlab's code read and write Newton state |

The observation and action code is reused rather than reimplemented. The policy reads 1328 dims
across 20 groups; re-deriving them is 1328 chances at a mistake that does not raise — as
[docs/03-defects.md](docs/03-defects.md) shows, that failure mode cost this project several days.

One deviation is worth naming up front: the warp model is rebuilt from a **compiled** `MjModel`
rather than Newton's reconstruction of one, because the reconstruction mislabels which bodies are
"simple" and changes the mass matrix layout (defect 7). Stepping, contacts and control are still
Newton's.

## Repo map

```
docs/00-goal.md          acceptance criteria and why requirement 2 comes first
docs/01-model-parity.md  the model comparison: what was checked and what it caught
docs/02-baseline.md      the mjlab baseline — which checkpoint, and the environment it needs
docs/03-defects.md       the eight silent defects, each with the measurement that found it
docs/04-architecture.md  how the bridge works and what surface it covers
docs/05-reproduce.md     exact commands
docs/data/               machine-generated fact dumps (model fields, orderings, eval results)

src/newton_bridge.py     mjlab's Entity/Env surface backed by mujoco_warp Data
src/mjw_compat.py        lets mjlab's 3.8-era code run on mujoco_warp 3.11
src/model_facts.py       field-level model extraction, shared by both sides

tools/build/             mjlab MJCF out, Newton model in
tools/run/               the port itself, and video rendering
tools/verify/            parity checks: model fields, observations, lockstep stepping
tools/eval/              the mjlab baseline these numbers are measured against
tools/probes/            one-off investigations, kept because each settled a specific question
```

`assets/mjlab_scene/assets/` (48 MB of STL) is gitignored; `tools/build/collect_assets.py`
regenerates it from the mjlab checkout.

## Quick start

See [docs/05-reproduce.md](docs/05-reproduce.md) for the full path. The short version, from a
Python environment with both Newton and mjlab installed:

```bash
export APPLE_HAND_KIND=wuji APPLE_SCENE_Z_OFFSET=-0.03 ASTRA_BASE_BACKEND=torch
export APPLE_EAT_PKL=$HOME/jiarui/scaled_grab_wuji_all_o70/s1/apple_eat_1.pkl
python tools/run/run_newton.py --steps 400 --dump-qpos /tmp/newton_qpos.npz
```

The environment variables are not optional and none of them are defaults — the reference clip, the
3 cm scene offset and the torch ASTRA backend all change the result. `docs/02-baseline.md` explains
what each one does.

## Status

Requirement 2 (infer the mjlab checkpoint in Newton) is met. Requirement 1 (train in Newton to
mjlab's numbers at comparable iterations) has not been started.
