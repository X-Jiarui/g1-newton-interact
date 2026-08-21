"""Minimal reproduction: does the body_simple / nC mismatch appear on a two-body scene?

If it needs a 91-body humanoid it is hard to act on. A single free-floating sphere is the smallest
thing that can carry MuJoCo's "simple body" flag, so that is what is tried first.
"""
import numpy as np, mujoco, newton
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
path = "/tmp/minimal_free_body.xml"
open(path, "w").write(XML)

ref = mujoco.MjModel.from_xml_path(path)
b = newton.ModelBuilder()
SolverMuJoCo.register_custom_attributes(b)
b.add_mjcf(path)
m = b.finalize()
sv = SolverMuJoCo(m)
nt = sv.mj_model

print(f"{'':22s} {'compiled':>10} {'newton':>10}")
for f in ("nbody", "nv", "nM", "nC"):
    print(f"  {f:20s} {getattr(ref, f):>10} {getattr(nt, f):>10}")
for f in ("body_simple", "dof_simplenum"):
    a = np.asarray(getattr(ref, f)).reshape(-1).tolist()
    c = np.asarray(getattr(nt, f)).reshape(-1).tolist()
    print(f"  {f:20s} {str(a):>10} {str(c):>10}")

wm = sv.mjw_model
print(f"\n  mjw_model.nC          {'':>10} {int(wm.nC):>10}   (compiled model says {ref.nC})")
print("REPRODUCES" if ref.nC != nt.nC else "does not reproduce on this model")
