# SolverMuJoCo: free bodies lose `body_simple`, giving the mass matrix a different sparse layout (`nC`)

## Summary

`SolverMuJoCo` reconstructs an `MjModel` from the Newton `Model` rather than using MuJoCo's compiled
one. The reconstruction does not set `body_simple` for free-floating bodies, so `dof_simplenum` stays
0 and `nC` — the size of MuJoCo's compressed mass-matrix layout — comes out larger than MuJoCo
computes for the same model. The solver then runs on a different sparse structure than stock
mujoco_warp does for the same MJCF.

## Reproduction

`minimal_free_body.xml` — one free body:

```xml
<mujoco model="minimal">
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="ball" pos="0 0 1">
      <freejoint name="ball_free"/>
      <geom name="ball_geom" type="sphere" size="0.05" mass="0.3"/>
    </body>
  </worldbody>
</mujoco>
```

```python
import numpy as np, mujoco, newton
from newton.solvers import SolverMuJoCo

path = "minimal_free_body.xml"
ref = mujoco.MjModel.from_xml_path(path)

b = newton.ModelBuilder()
SolverMuJoCo.register_custom_attributes(b)
b.add_mjcf(path)
sv = SolverMuJoCo(b.finalize())
nt = sv.mj_model

for f in ("nbody", "nv", "nM", "nC"):
    print(f"{f:16s} compiled={getattr(ref, f):4d}  newton={getattr(nt, f):4d}")
for f in ("body_simple", "dof_simplenum"):
    print(f"{f:16s} compiled={np.asarray(getattr(ref, f)).tolist()}  "
          f"newton={np.asarray(getattr(nt, f)).tolist()}")
print("mjw_model.nC =", int(sv.mjw_model.nC), " (compiled model says", ref.nC, ")")
```

## Output

```
nbody            compiled=   2  newton=   2
nv               compiled=   6  newton=   6
nM               compiled=  21  newton=  21
nC               compiled=   6  newton=  21
body_simple      compiled=[1, 1]              newton=[1, 0]
dof_simplenum    compiled=[6, 5, 4, 3, 2, 1]  newton=[0, 0, 0, 0, 0, 0]
mjw_model.nC = 21  (compiled model says 6)
```

`nM` and the kinematic tree agree; only the compressed layout differs.

## Why it matters

Ported an mjlab-trained policy onto `SolverMuJoCo` with a 91-body humanoid plus two free bodies
(a grasped object and a table). Both models were verified identical field-for-field — same tree, same
`nM`, same `dof_parentid`, same per-body mass and inertia, same actuators — and the rollouts still
diverged.

Stepping both from bit-identical state with identical `ctrl`, one substep apart:

```
qfrc_smooth       max|diff| = 1.2e-05
qfrc_bias         max|diff| = 1.0e-05
qfrc_constraint   max|diff| = 2.3e-03      <-- constraint solve
qacc              max|diff| = 9.4e-03
```

Smooth dynamics agree; the constraint solve does not. Rebuilding the warp model from a compiled
`MjModel` brings `nC` to the expected value and reproduces stock mujoco_warp to `1.5e-08` over 20
steps.

The difficulty is that this is invisible to model comparison: `nM`, topology and every per-body field
match. It only shows up in `Data.M`, which is `(1, nC)` — different lengths on the two sides.

### Behavioural cost

Running the same policy with everything else held fixed, changing only whether the warp model comes
from Newton's reconstruction or from a compiled `MjModel`:

Same policy, same scene, same everything else; only the source of the warp model changes. Repeated
runs, because GPU contact solving is not bit-reproducible and a single run does not characterise a
configuration:

| | native Newton model (`nC=1102`) | compiled `MjModel` (`nC=1087`) |
|---|---|---|
| runs holding the object | **8 of 11** | 6 of 6 |
| peak lift when it holds | 49.8 – 52.2 cm | 49.8 – 50.8 cm |
| peak lift when it fails | 5.7 / 8.2 / 16.9 cm, then dropped | — |

So the defect does not break the grasp outright; it makes it **marginal**. The mislabelled body is the
grasped object itself, so its mass matrix — the one the contact solve acts through — is the one
carrying the wrong layout.

Raising `opt.iterations` from the scene's 10 to 200 recovers the lift on the native model
(51.8 cm), which suggests the different layout is harder to converge rather than describing different
physics. That is consistent with the force diff above being a solver residual rather than a modelling
error.

## Note on workarounds

Setting `body_simple` / `dof_simplenum` on the reconstructed `MjModel` and calling `put_model` again
does **not** help: `nC` and its index arrays (`M_rowadr`, `M_colind`, `mapM2M`) are computed by
MuJoCo's compiler and stored on the model, so the stale `nC` is copied through. The only workaround
we found is to build the warp model from a compiled `MjModel`.

## Environment

```
newton 1.5.0   mujoco 3.11.0   mujoco_warp 3.11.0   warp 1.16.0
Python 3.12    CUDA 12.8       RTX 4090            Ubuntu 24.04
```
