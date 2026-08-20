"""Hoist plane geoms out of jointless wrapper bodies so Newton will accept them.

Newton's SolverMuJoCo refuses a plane on a non-static body. mjlab wraps its ground plane in
`<body name="terrain">`, which MuJoCo treats as static -- a body with no joint is welded to the world
-- but Newton's importer gives every such body a FIXED joint, which makes it dynamic, and the
conversion aborts.

The obvious lever, `collapse_fixed_joints=True`, is NOT usable here and that was checked rather than
assumed: 20 of the 91 bodies are jointless, and **all 10 fingertip bodies are among them**. Collapsing
would erase exactly the bodies `_tip_distances` looks up by name to feed 16 reward and gate sites.

So the edit is surgical: move plane geoms up into `<worldbody>` verbatim, and only from wrapper bodies
that are genuinely inert -- no joint, no pos, no quat, no child bodies. Under those conditions the
wrapper contributes nothing but a name, so hoisting is semantics-preserving. Anything else is left
alone and reported.

Known, intended consequence: the hoisted geom's owning body changes from the wrapper to `world`, so a
fact-sheet comparison will show that one field moving. That is a real difference and is meant to be
visible rather than papered over.
"""
from __future__ import annotations
import os, sys
import xml.etree.ElementTree as ET
from pathlib import Path

XML = Path(os.path.expanduser(
  sys.argv[1] if len(sys.argv) > 1
  else "~/projects/g1-newton-interact/assets/mjlab_scene/scene.xml"))

tree = ET.parse(XML)
root = tree.getroot()
world = root.find("worldbody")
if world is None:
  sys.exit("no <worldbody>")

hoisted, refused = [], []
for body in list(world.findall("body")):
  planes = [g for g in body.findall("geom") if g.get("type") == "plane"]
  if not planes:
    continue
  inert = (
    not body.findall("joint")
    and not body.findall("freejoint")
    and not body.findall("body")
    and body.get("pos") in (None, "0 0 0")
    and body.get("quat") in (None, "1 0 0 0")
  )
  if not inert:
    refused.append((body.get("name"), len(planes)))
    continue
  for g in planes:
    body.remove(g)
    world.append(g)          # attributes carried over untouched
    hoisted.append((g.get("name"), body.get("name")))

for name, wrapper in hoisted:
  print(f"hoisted plane {name!r} out of body {wrapper!r} into <worldbody>")
for name, n in refused:
  print(f"REFUSED body {name!r} ({n} planes): not inert (has joint/pos/quat/child bodies)")
if not hoisted and not refused:
  print("no plane geoms inside wrapper bodies; nothing to do")

tree.write(XML, encoding="unicode")
print(f"wrote {XML}")
sys.exit(2 if refused else 0)
