"""Does the shape gap/margin default decide whether MuJoCo marks a body simple?

The exhaustive diff on the minimal model left only two input-side differences: body_margin and
geom_gap, both 0.1 on the Newton side (its default) against 0 in the MJCF. Everything else that
differed -- nC, body_simple, dof_simplenum, the M_* index arrays -- is downstream of the compiler's
decision. So the test is direct: build the same model with gap forced to 0 and see what the compiler
says.
"""
import os, sys, numpy as np, mujoco
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
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
print(f"reference        : nC={ref.nC} body_simple={ref.body_simple.tolist()} "
      f"body_margin={ref.body_margin.tolist()}")

for gap in (None, 0.0):
    b = newton.ModelBuilder(); SolverMuJoCo.register_custom_attributes(b)
    if gap is not None:
        b.default_shape_cfg.gap = gap
    b.add_mjcf(p)
    sv = SolverMuJoCo(b.finalize())
    mm = sv.mj_model
    tag = "newton default gap" if gap is None else f"newton gap={gap}"
    print(f"{tag:17s}: nC={mm.nC} body_simple={mm.body_simple.tolist()} "
          f"body_margin={np.round(mm.body_margin,4).tolist()}")

# and the real scene, where gap is already forced to 0
SCENE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets/mjlab_scene/scene.xml")
sref = mujoco.MjModel.from_xml_path(SCENE)
b = newton.ModelBuilder(); SolverMuJoCo.register_custom_attributes(b)
b.default_shape_cfg.gap = 0.0
b.add_mjcf(SCENE, collapse_fixed_joints=False, parse_mujoco_options=True)
sv = SolverMuJoCo(b.finalize(), enable_multiccd=True, njmax=2048, nconmax=256)
mm = sv.mj_model
ap, tb = 89, 91
print(f"\nreal scene, gap already 0:")
print(f"  nC ref={sref.nC} newton={mm.nC}")
for i, nmm in ((ap, "apple/apple"), (tb, "table/table")):
    print(f"  {nmm:12s} body_margin ref={sref.body_margin[i]:.4g} newton={mm.body_margin[i]:.4g}"
          f"   simple ref={int(sref.body_simple[i])} newton={int(mm.body_simple[i])}")
print(f"  bodies with nonzero margin: ref={int((sref.body_margin!=0).sum())} "
      f"newton={int((mm.body_margin!=0).sum())}")
