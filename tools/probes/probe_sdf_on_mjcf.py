"""Can SDF collision be switched on for an object that came in through add_mjcf?

Newton's own example builds the SDF before adding the shape: mesh.build_sdf(...) then
add_shape_mesh(cfg=...is_hydroelastic/force_sdf). Our object arrives via add_mjcf, so the mesh object
is created inside the importer. If the builder still exposes it afterwards, SDF can be enabled in
place; if not, the object has to be added separately from the robot.
"""
import os, sys, numpy as np, trimesh
sys.path.insert(0, os.path.expanduser("~/projects/g1-newton-interact/src"))
import mjw_compat; mjw_compat.apply()
import newton
from newton.solvers import SolverMuJoCo

XML = os.path.expanduser("~/projects/g1-newton-interact/assets/mjlab_scene/scene.xml")
b = newton.ModelBuilder(); SolverMuJoCo.register_custom_attributes(b)
b.default_shape_cfg.gap = 0.0
b.add_mjcf(XML, collapse_fixed_joints=False, parse_mujoco_options=True)

print("builder shape-related attrs:",
      [a for a in dir(b) if a.startswith("shape_") and "source" in a or a in
       ("shape_geo_src", "shape_source", "shape_type", "shape_flags", "shape_key")][:10])
src = getattr(b, "shape_source", None) or getattr(b, "shape_geo_src", None)
print("shape source list present:", src is not None, "len:", len(src) if src is not None else None)
if src is not None:
    kinds = {}
    for s in src:
        kinds[type(s).__name__] = kinds.get(type(s).__name__, 0) + 1
    print("  source kinds:", kinds)
    meshes = [(i, s) for i, s in enumerate(src) if isinstance(s, newton.Mesh)]
    print(f"  Mesh sources: {len(meshes)}")
    if meshes:
        i, m0 = meshes[0]
        print(f"   first: shape {i}, verts={len(m0.vertices)}, has sdf={getattr(m0,'sdf',None) is not None}")
        print(f"   build_sdf available: {hasattr(m0, 'build_sdf')}")

# ShapeConfig knobs that would turn it on
cfgc = newton.ModelBuilder.ShapeConfig()
for f in ("force_sdf", "configure_sdf", "is_hydroelastic", "sdf_max_resolution",
          "sdf_narrow_band_range", "sdf_padding", "kh"):
    print(f"  ShapeConfig.{f} default = {getattr(cfgc, f, '<absent>')}")

flags = np.asarray(b.shape_flags)
print(f"\nbuilder shape_flags: {len(flags)} shapes, distinct {sorted(set(flags.tolist()))}")
