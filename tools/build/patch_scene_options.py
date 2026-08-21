"""Write mjlab's solver options into the exported MJCF so the file describes the real physics.

`scene.write()` serialises the MjSpec, but mjlab does not keep its solver settings in the spec: it
applies them to the *compiled* model afterwards (`Simulation`'s cfg apply step). The exported XML
therefore compiles to MuJoCo's defaults -- Euler at 2 ms with 100 solver iterations -- while mjlab
actually runs implicitfast at 5 ms with 10. Anyone importing the bare XML, Newton included, would be
simulating a different system than the policy was trained in, and nothing about the file would say so.

Values are taken from the measured baseline (`docs/mjlab_facts.json`), not retyped, so this cannot
drift away from what mjlab actually ran.
"""
from __future__ import annotations
import json, os, re, sys
from pathlib import Path

FACTS = Path(os.path.expanduser("~/projects/g1-newton-interact/docs/mjlab_facts.json"))
XML = Path(os.path.expanduser("~/projects/g1-newton-interact/assets/mjlab_scene/scene.xml"))

INTEGRATOR = {0: "Euler", 1: "RK4", 2: "implicit", 3: "implicitfast"}
SOLVER = {0: "PGS", 1: "CG", 2: "Newton"}
CONE = {0: "pyramidal", 1: "elliptic"}
JACOBIAN = {0: "dense", 1: "sparse", 2: "auto"}

o = json.load(FACTS.open())["options"]
attrs = {
  "timestep": repr(float(o["timestep"])),
  "integrator": INTEGRATOR[int(o["integrator"])],
  "solver": SOLVER[int(o["solver"])],
  "cone": CONE[int(o["cone"])],
  "jacobian": JACOBIAN[int(o["jacobian"])],
  "iterations": str(int(o["iterations"])),
  "ls_iterations": str(int(o["ls_iterations"])),
  "impratio": repr(float(o["impratio"])),
  "tolerance": repr(float(o["tolerance"])),
  "ls_tolerance": repr(float(o["ls_tolerance"])),
  "gravity": " ".join(repr(float(v)) for v in o["gravity"]),
  "density": repr(float(o["density"])),
  "viscosity": repr(float(o["viscosity"])),
}
block = "<option " + " ".join(f'{k}="{v}"' for k, v in attrs.items()) + " />"
print(block)

text = XML.read_text()
if re.search(r"<option\b", text):
  text = re.sub(r"<option\b[^>]*/>", block, text, count=1)
  text = re.sub(r"<option\b[^>]*>.*?</option>", block, text, count=1, flags=re.S)
  how = "replaced existing <option>"
else:
  # Insert immediately after <compiler .../>, which every mjlab export carries.
  m = re.search(r"<compiler\b[^>]*/>", text)
  if not m:
    sys.exit("no <compiler> element found; refusing to guess where <option> belongs")
  text = text[: m.end()] + "\n  " + block + text[m.end():]
  how = "inserted after <compiler>"
XML.write_text(text)
print(f"{how} in {XML}")
