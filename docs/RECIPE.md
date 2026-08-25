# The from-scratch recipe

This is the configuration to start a new single-clip run with. It is written down because the
default that ships with `--native-contacts` is **not** this, and the difference is large.

```bash
python tools/run/train_newton.py \
  --xml assets/scene_stapler/scene.xml \
  --reference-pkl  $DATA/grab_g1_wuji_aligned/<subject>/<sequence>.pkl \
  --sdf-object     $DATA/scaled_grab_dataset_wuji/meshes/<object>.stl \
  --native-contacts --rigid-object-table --table-under-object --cuda-graph \
  --solver-kwargs '{"impratio": 1.0, "cone": "pyramidal"}' \
  --num-envs 2048 --iterations 3000 --seed 1 \
  --log-root logs/rsl_rl --run-name <NAME>
```

`--xml` is the same file for every object. `assets/scene_stapler/scene.xml` and
`assets/scene_mug/scene.xml` are byte-identical (same md5): the scene carries a placeholder sphere
body named `apple/apple`, and the real collider is swapped in at load time from `--sdf-object`.
There is no per-object scene to generate.

## Why the two solver options

`--native-contacts` sets `cone="elliptic"` with `impratio=1000`, inherited from
`newton/examples/robot/example_robot_panda_hydro.py`. That example runs hydroelastic contact; this
project does not (`--rigid-object-table`). `impratio=1000` makes the tangential (friction)
constraints three orders of magnitude softer than the normal ones, so the hand pushes the object
around instead of holding it.

An audit of every contact-related parameter found that **cone and impratio are the only solver
options that differ** between the native path and the MuJoCo-contact path. The MuJoCo path already
runs `pyramidal` with `impratio=1`. Setting the native path to those two values makes every
per-geom and per-contact parameter (friction, solref, solimp, condim, margin, gap) identical on
both paths.

## What it bought

Three runs on `s8/stapler_pass_1`, all else equal. `lift_success` is the fraction of environments
that raised the object 3 cm above its reference start height, with contact, held for 0.5 s.

| | iterations | `lift_success` |
|---|---|---|
| native default (`elliptic`, `impratio=1000`) | 3998 | 0.20 |
| native + these two options, resumed from that run's iter-2000 checkpoint | 3332 | 0.78 |
| native + these two options, **from scratch** | 1372 | 0.75 |

The from-scratch line is the one to trust: same seed, same code, same data, same starting weights
as the unchanged run — only the two solver options differ. It first lifted at **iteration 542**;
the MuJoCo-contact control line on the same clip first lifted at **918**, and the unchanged native
line has no clean transition at all, it creeps up from noise over thousands of iterations.

At the iteration where the control line produced its *first* non-zero lift (918), the friction line
was already at 0.42.

## What is still open

- `sequence_success` (completing the whole clip) reached 0.31 on the MuJoCo-contact control line
  and 0.12 on the friction resume, but is still ~0.001 on the from-scratch friction line at
  iteration 1372. It rises much later than `lift_success`.
- On the mug clip (`s1/mug_drink_4`, 498 steps vs the stapler's 359) the MuJoCo-contact control
  line is still at `lift_success` 0.0 after 4770 iterations, while the friction run reaches 0.49.
  Nothing about that clip is understood yet beyond "it is longer and harder".
- `object_contact` **falls** as a run improves — the control line went from 0.68 to 0.35 while
  `sequence_success` went from 0.0015 to 0.31, ending up *below* a run that never learned to lift.
  Do not read contact as a progress signal.
