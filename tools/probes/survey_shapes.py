"""What object representations does Newton 1.5 actually offer, and which solver supports each?

In mjlab every object had to become convex: analytic spheres, convex hulls, or a stack of <=20 boxes
fitted by hand. That is a modelling compromise forced by the contact model, and it is what the new
object representation is meant to escape. So: what does Newton support natively, how is it enabled,
and what does it need.
"""
import inspect
import numpy as np
import newton

print("=== GeoType ===")
gt = {k: int(v) for k, v in vars(newton.GeoType).items() if not k.startswith("_") and isinstance(v, int)}
print("  ", gt)

print("\n=== ShapeFlags ===")
sf = {k: int(v) for k, v in vars(newton.ShapeFlags).items() if not k.startswith("_") and isinstance(v, int)}
print("  ", sf)

print("\n=== ModelBuilder.add_shape_* ===")
adders = [a for a in dir(newton.ModelBuilder) if a.startswith("add_shape")]
for a in adders:
    try:
        sig = str(inspect.signature(getattr(newton.ModelBuilder, a)))
    except Exception:
        sig = "?"
    print(f"   {a}{sig[:120]}")

print("\n=== SDF / mesh types ===")
for name in ("SDF", "Mesh", "SdfVolume", "Volume"):
    o = getattr(newton, name, None)
    if o is None:
        print(f"   newton.{name}: absent"); continue
    try:
        print(f"   newton.{name}{str(inspect.signature(o.__init__))[:130]}")
    except Exception:
        print(f"   newton.{name}: {type(o)}")

print("\n=== hydroelastic ===")
hits = []
for mod_name in ("newton", "newton.geometry", "newton.solvers"):
    try:
        mod = __import__(mod_name, fromlist=["*"])
    except Exception:
        continue
    for a in dir(mod):
        if "hydro" in a.lower():
            hits.append(f"{mod_name}.{a}")
print("   symbols:", hits or "none at top level")

from newton.solvers import SolverMuJoCo
mj_params = inspect.signature(SolverMuJoCo.__init__).parameters
print("\n=== SolverMuJoCo options mentioning contact/sdf/hydro ===")
for n, p in mj_params.items():
    if any(k in n.lower() for k in ("contact", "sdf", "hydro", "ccd", "mesh", "convex")):
        print(f"   {n} = {p.default}")

print("\n=== other solvers available ===")
import newton.solvers as S
print("  ", [a for a in dir(S) if a.startswith("Solver")])
