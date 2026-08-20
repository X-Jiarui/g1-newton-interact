# Stage 2 result: the Newton model matches mjlab's

Comparison is between mjlab's compiled model and **the live model `SolverMuJoCo` actually
simulates** -- not the MJCF it exports. `save_to_mjcf` runs inside the constructor, before the
post-conversion repairs below, so the saved file is a snapshot of the raw conversion and disagrees
with what runs. `docs/newton_live_facts.json` is the artifact that counts.

Both sides compiled under MuJoCo 3.11 (`tools/compare_facts.py --colliding-geoms-only --tol 1e-5`):

| section | result |
|---|---|
| joints | **identical order** (71), all 71 agree on every field |
| actuators | **identical order** (138), all 138 agree on every field |
| geoms (colliding) | all 81 agree on every field |
| bodies | all 92 agree on every field |
| options | no differences |
| sizes | nq/nv/nu/njnt/nbody identical; `ngeom` 226 vs 81 and `nmesh` 145 vs 65 **by design** |

The `ngeom`/`nmesh` gap is `skip_visual_only_geoms=True`: Newton drops geoms that cannot collide.
mjlab's model has exactly 145 non-colliding and 81 colliding geoms, and all 81 came across intact.

**Joint order and the PD definition survived the conversion untouched**, which were the two
highest-risk quantities going in.

## Control: the MuJoCo version bump changes no structure

The same mjlab XML compiled under 3.8.1 and 3.11.0 gives `FACTS_MATCH`. So structural differences are
attributable to Newton's conversion alone, and the 3.8-vs-3.11 gap is confined to solver *behaviour*
-- which is measured separately, in the noise-floor step, not guessed at.

## Four defects found in the conversion, and the repairs

Each was measured, and each is silent: the model loads, steps, and reports plausible numbers either
way. They are applied in `tools/import_to_newton.py`.

**1. Every contact was disabled.** Newton leaves shape `gap` unset when the MJCF omits the attribute
(mjlab's does), and the conversion then fills 0.1 instead of MuJoCo's default 0. With `margin=0` the
force threshold `margin - gap` becomes **-0.1**: contacts only become active at 10 cm of penetration.
All 81 colliding geoms were affected -- the robot would fall through the floor and no grasp could
form. Fix: `builder.default_shape_cfg.gap = 0.0` before `add_mjcf`.

**2. Free-joint damping was dropped.** The object's `damping="0.05"` on all six DOFs arrived as 0,
so the apple had no passive resistance. Assigning `Model.joint_damping` before `finalize()` is *not*
sufficient -- measured: the value lands on the Newton model and still converts to 0.0 -- so
`dof_damping` is patched on both `mj_model` and `mjw_model` after the solver exists. `mjw_model` is
the one that simulates.

**3. Multi-CCD was disabled.** Newton sets `mjDSBL_MULTICCD` (disableflags 524288) by default while
mjlab leaves it enabled; it caps a geom pair at one contact point instead of four. This task is a
five-finger grasp, so contact-point count is not a detail. Fix: `enable_multiccd=True`.
mujoco_warp warns that cylinder-cylinder/box/mesh pairs still yield one contact regardless.

**4. Solver options and timestep.** `parse_mujoco_options=True` does carry `<option>` through --
integrator, solver, cone, iterations and ls_iterations all arrived correctly -- **but only because
the option block was put into the XML first**. mjlab does not keep solver settings in its spec; it
applies them to the compiled model afterwards, so its exported MJCF describes Euler at 2 ms with 100
iterations rather than implicitfast at 5 ms with 10 (`tools/patch_scene_options.py`). Timestep is
separate again: Newton drives dt from the simulation loop, so the model kept 0.002 and is now set
explicitly -- **and the rollout loop must still step at 0.005**.

## Two changes made to the exported MJCF, both deliberate

* **`<option>` injected** (`tools/patch_scene_options.py`) -- see defect 4. Without it the file does
  not describe the physics mjlab runs.
* **The ground plane hoisted out of its wrapper body** (`tools/hoist_plane_geoms.py`). Newton refuses
  a plane on a non-static body; mjlab wraps its plane in a jointless `<body name="terrain">`, which
  MuJoCo treats as welded to the world but Newton gives a FIXED joint and calls dynamic.
  `collapse_fixed_joints=True` is the documented lever and is **not usable here**: 20 of 91 bodies are
  jointless and **all 10 fingertip bodies are among them**, so collapsing would erase exactly the
  bodies `_tip_distances` looks up by name for 16 reward and gate sites. Hoisting only affects
  wrapper bodies with no joint, no transform and no children, where it is semantics-preserving.

Plane extent is also normalised after conversion: mjlab's `(0, 0, 0.01)` means an infinite plane,
Newton substitutes a finite 5x5.

## What is verified, and what is not

Verified: the two models are the same model. That is a static claim about structure and parameters,
and it is now exact.

Not yet verified: that they *behave* the same. Nothing here has stepped physics. mujoco_warp 3.8 vs
3.11 remains unmeasured, and the observation/action plumbing is untouched.
