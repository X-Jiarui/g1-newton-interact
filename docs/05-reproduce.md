# Reproducing the result

## Environment

One Python environment holds both libraries. mjlab pins `mujoco-warp>=3.8.0` (a lower bound), so
Newton's 3.11 satisfies it:

```bash
pip install -e ~/projects/mjlab-astra-dagger-distill-20260625   # into the newton env
```

Verified versions:

```
newton 1.5.0   mujoco 3.11.0   mujoco_warp 3.11.0   warp 1.16.0   torch 2.11.0+cu128
```

`src/mjw_compat.py` handles the one API mjlab still uses that 3.9.1 removed (`ls_parallel`). It is
imported by every tool here before mjlab builds a `Simulation`.

## The environment variables are not optional

```bash
export APPLE_HAND_KIND=wuji                  # 40 hand dofs; without it the model is xhand's 24
export APPLE_SCENE_Z_OFFSET=-0.03            # moves apple AND table down 3 cm
export ASTRA_BASE_BACKEND=torch
export GMR_ROOT=$HOME/jiarui/GMR
export APPLE_EAT_PKL=$HOME/jiarui/scaled_grab_wuji_all_o70/s1/apple_eat_1.pkl
export MUJOCO_GL=egl                         # rendering only
```

None of these are defaults, and each changes the answer. The clip especially: the **default** dataset
puts this clip's closest hand–object approach at frame 457, the **o70** dataset at frame 76. A frame
index means nothing without naming the clip.

The checkpoint also trained with the pelvis start-assist disabled
(`tracking_start_assist_gain: 0.0`), while play.py defaults it to 1.5 for 120 steps. Every tool here
sets it to 0; if you run `play` by hand, pass `--start-assist-gain 0.0` or you are watching a robot
propped up by a wrench it never trained with.

## 1. Build the Newton-side scene

Only needed once, or after the mjlab scene changes.

```bash
python tools/build/export_mjlab_mjcf.py       # mjlab's compiled model -> assets/mjlab_scene/scene.xml
python tools/build/collect_assets.py          # the 48 MB of STL the MJCF references (gitignored)
python tools/build/patch_scene_options.py     # solver options MjSpec.to_xml() omits
python tools/build/hoist_plane_geoms.py       # plane geoms out of the wrapper body
python tools/build/import_to_newton.py        # round-trip check: Newton model vs mjlab, field by field
```

`export_mjlab_mjcf.py` round-trips to `FACTS_MATCH`; `import_to_newton.py` writes
`docs/data/newton_live_facts.json` and reports any field that differs.

## 2. The mjlab baseline

```bash
python tools/eval/trace_eval.py \
  $HOME/sweep_ckpts_r2/OF_00_apple_eat_1_SPHERE/model_7310.pt \
  --num-envs 8 --steps 400 --every 40 --no-terminations --force-start-frame 0
```

Expected: `max object rise ≈ 50 cm`, `min hand_to_obj ≈ 0.032–0.035 m`, `peak lift_success 1.000`.

Which checkpoint and why: see [02-baseline.md](02-baseline.md). Briefly — `~/sweep_rank.tsv` ranks
the 24-run sweep by `lift_success` and `OF_00_apple_eat_1_SPHERE` is 0.6450, the only apple run that
lifts at all.

## 3. Newton

```bash
python tools/run/run_newton.py --steps 400 --every 20 --dump-qpos /tmp/newton_qpos.npz
```

Expected:

```
warp model rebuilt from the compiled MJCF: nC 1102 -> 1087, nmocap 3 -> 1
residual-stats attributes synced onto the Newton env: 8/8
...
max object rise = 50.60 cm   (mjlab@3.8 49.72, mjlab@3.11 50.13)
min fingertip-object distance = 0.035 m   (mjlab 0.032-0.035)
```

Both banner lines matter. The first is defect 9, the second is defect 8; if either number is wrong
the rollout will still run and will simply fail to grasp.

## 4. Verification

```bash
python tools/verify/compare_obs.py     # all 20 observation groups from the same reset state
python tools/verify/chain.py           # obs -> policy -> ctrl -> one substep, link by link
python tools/verify/lockstep.py        # identical actions into both, where do they part?
```

`compare_obs.py` should report every group at ≤1e-6 and `|d ctrl|max = 0`.

`chain.py`'s link-4 bisection copies every shared `mjw_data` array from mjlab into Newton before
stepping, so the two enter the substep from provably identical state. It also prints the arrays it
could **not** copy — that list is where defect 9 was found.

## 5. Video

```bash
python tools/eval/trace_eval.py <ckpt> --num-envs 1 --steps 400 \
    --no-terminations --force-start-frame 0 --dump-qpos /tmp/mjlab_qpos.npz
python tools/run/run_newton.py --steps 400 --dump-qpos /tmp/newton_qpos.npz
python tools/run/render_compare.py --right-mocap-idx 0 --out ~/newton_vs_mjlab.mp4
```

Both panels are drawn with mjlab's **full visual model** (226 geoms) and differ only in the state
driving them. Newton's converted model carries 81 collision geoms and no visual-only geometry, so
rendering it directly would show a different-looking robot and invite the difference to be blamed on
the renderer.

`--right-mocap-idx` exists because the table is a mocap body whose index differs between the two
models — 0 with the compiled model, 2 with Newton's reconstruction, which turns every world-attached
fixed body into a mocap body. Copying slot-for-slot puts the table at the origin and the apple
appears to float.

## Known-good numbers

| | mjlab @ 3.8 | mjlab @ 3.11 | Newton 1.5 |
|---|---|---|---|
| peak rise | 49.72 cm | 50.13 cm | 50.60 cm |
| min hand–object | 0.032 m | 0.035 m | 0.035 m |
| lockstep divergence, first free step | — | — | 5.3e-4 rad (pre-fix) |
| observation groups agreeing | — | — | 20/20 at ≤8.3e-07 |
| ctrl agreement | — | — | exact (0) |
