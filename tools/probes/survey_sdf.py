"""How are SDF and hydroelastic shapes actually built and which solvers accept them?

The capability list is promising -- MESH distinct from CONVEX_MESH, a real SDF type, and a
HydroelasticSDF -- but a capability that only one unusable solver supports is not a plan. This looks
at how each is constructed, what it costs, and whether SolverMuJoCo (the one that reproduces our
trained policy) can use it.
"""
import inspect
import newton
from newton.geometry import HydroelasticSDF

print("=== newton.SDF ===")
print("  __init__:", str(inspect.signature(newton.SDF.__init__))[:200])
print("  methods:", [a for a in dir(newton.SDF) if not a.startswith("_")][:16])
for cname in ("from_mesh", "from_file", "from_numpy", "create"):
    f = getattr(newton.SDF, cname, None)
    if f is not None:
        print(f"  SDF.{cname}{str(inspect.signature(f))[:160]}")

print("\n=== HydroelasticSDF ===")
print("  __init__:", str(inspect.signature(HydroelasticSDF.__init__))[:220])
print("  methods:", [a for a in dir(HydroelasticSDF) if not a.startswith("_")][:16])
print("  doc:", (HydroelasticSDF.__doc__ or "").strip()[:400])

print("\n=== ShapeConfig: what per-shape knobs exist ===")
sc = newton.ModelBuilder.ShapeConfig
print("  fields:", [f for f in dir(sc) if not f.startswith("_")][:28])

print("\n=== add_shape_mesh signature (full) ===")
print(" ", inspect.signature(newton.ModelBuilder.add_shape_mesh))

print("\n=== newton.Mesh: does it carry an SDF / hydroelastic option? ===")
print("  __init__:", str(inspect.signature(newton.Mesh.__init__))[:260])
print("  attrs:", [a for a in dir(newton.Mesh) if not a.startswith("_")][:20])

print("\n=== which solvers mention hydroelastic ===")
import newton.solvers as S
for name in dir(S):
    if not name.startswith("Solver"):
        continue
    cls = getattr(S, name)
    try:
        sig = inspect.signature(cls.__init__)
    except Exception:
        continue
    hits = [p for p in sig.parameters if "hydro" in p.lower() or "sdf" in p.lower()]
    doc = (cls.__doc__ or "")
    if hits or "hydroelastic" in doc.lower():
        print(f"   {name}: params={hits}  doc_mentions={'hydroelastic' in doc.lower()}")
