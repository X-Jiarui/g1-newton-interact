"""Exhaustive field diff on the two-body minimal model, to find what makes MuJoCo decline 'simple'.

Everything the obvious rule depends on -- sameframe, ipos, iquat, joint type -- is identical between
the two models, so the deciding field is something else. With two bodies and one joint the whole
model is small enough to compare field by field.
"""
import os, sys, numpy as np, mujoco
sys.path.insert(0, os.path.expanduser("~/projects/g1-newton-interact/src"))
import mjw_compat; mjw_compat.apply()
import newton
from newton.solvers import SolverMuJoCo

XML = """
<mujoco model="minimal">
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="ball" pos="0 0 1">
      <freejoint name="ball_free"/>
      <geom name="ball_geom" type="sphere" size="0.05" mass="0.3"/>
    </body>
  </worldbody>
</mujoco>
"""
p = "/tmp/minimal_free_body.xml"; open(p, "w").write(XML)
ref = mujoco.MjModel.from_xml_path(p)
b = newton.ModelBuilder(); SolverMuJoCo.register_custom_attributes(b)
b.add_mjcf(p)
sv = SolverMuJoCo(b.finalize())
mm = sv.mj_model

print(f"ref  : nC={ref.nC} body_simple={ref.body_simple.tolist()} dof_simplenum={ref.dof_simplenum.tolist()}")
print(f"newton: nC={mm.nC} body_simple={mm.body_simple.tolist()} dof_simplenum={mm.dof_simplenum.tolist()}")

skip = ("names", "paths", "text_data", "name")
print("\nfields differing beyond float32 round-off:")
n_diff = 0
for f in sorted(dir(ref)):
    if f.startswith("_") or any(k in f for k in skip):
        continue
    try:
        a = getattr(ref, f); c = getattr(mm, f)
    except Exception:
        continue
    if callable(a) or not isinstance(a, (np.ndarray, int, float, np.integer, np.floating)):
        continue
    a = np.asarray(a).reshape(-1).astype(np.float64)
    c = np.asarray(c).reshape(-1).astype(np.float64)
    if a.shape != c.shape:
        print(f"  {f:24s} SHAPE {a.shape} vs {c.shape}"); n_diff += 1; continue
    if a.size == 0:
        continue
    scale = np.maximum(np.abs(a), 1.0)
    rel = np.abs(a - c) / scale
    if rel.max() > 1e-6:            # float32 eps is ~1.2e-7; this only catches real differences
        n_diff += 1
        i = int(np.argmax(rel))
        print(f"  {f:24s} rel={rel.max():.3g}  ref[{i}]={a[i]:.9g}  newton[{i}]={c[i]:.9g}")
print(f"\n{n_diff} field(s) differ beyond round-off")
