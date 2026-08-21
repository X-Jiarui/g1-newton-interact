"""Compile an MJCF and write its fact-sheet. The same step is applied to both sides of the port:
mjlab's exported scene.xml, and the MJCF that Newton's SolverMuJoCo emits via save_to_mjcf.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
import mujoco
from model_facts import facts

ap = argparse.ArgumentParser()
ap.add_argument("xml"); ap.add_argument("out")
ap.add_argument("--label", default="")
a = ap.parse_args()

m = mujoco.MjModel.from_xml_path(a.xml)
f = facts(m)
f["_source"] = dict(side=a.label or Path(a.xml).stem, xml=os.path.abspath(a.xml))
f["_versions"] = dict(mujoco=mujoco.__version__)
Path(a.out).write_text(json.dumps(f, indent=1))
s = f["sizes"]
print(f"[{a.label or Path(a.xml).stem}] mujoco {mujoco.__version__}  "
      f"nq={s['nq']} nv={s['nv']} nu={s['nu']} njnt={s['njnt']} nbody={s['nbody']} ngeom={s['ngeom']}")
print(f"wrote {a.out}")
