"""Materialise the mesh files the exported MJCF references, from the authoritative source.

`scene.write()` emits scene.xml with `meshdir="assets"` but does not copy the meshes: mjlab's specs
reference them by path through their own meshdir, and the exporter only carries assets embedded in
the spec. The 145 referenced files therefore have to be gathered by hand.

Picking them by filename alone would be unsafe -- `left_ankle_pitch_link.STL` exists in five places
on this machine, and quietly taking the wrong copy would change collision geometry while everything
still compiled. So roots are searched in priority order, mjlab's own asset directory first because
that is what mjlab actually compiled, and every file found in more than one root is hashed. A
same-name/different-content collision is reported loudly instead of being resolved silently.
"""
from __future__ import annotations
import hashlib, os, shutil, sys
from pathlib import Path

XML = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets/mjlab_scene/scene.xml"))
DEST = XML.parent / "assets"
ROOTS = [  # priority order: mjlab's own tree wins
  Path(os.path.expanduser("~/projects/mjlab-astra-dagger-distill-20260625/src/mjlab/asset_zoo/robots/unitree_g1/xmls/assets")),
  Path(os.path.expanduser("~/jiarui/GMR/assets/g1_wuji/meshes")),
  Path(os.path.expanduser("~/jiarui/GMR/assets/g1_xhand/meshes")),
]
for r in ROOTS:
  print(f"root {'OK ' if r.is_dir() else 'MISSING'} {r}")

import re
need = sorted(set(re.findall(r'file="([^"]+)"', XML.read_text())))
print(f"referenced files: {len(need)}")

def md5(p):
  return hashlib.md5(p.read_bytes()).hexdigest()

DEST.mkdir(parents=True, exist_ok=True)
copied = missing = []
copied, missing, ambiguous = [], [], []
for rel in need:
  hits = []
  for r in ROOTS:
    for cand in (r / rel, r / Path(rel).name):
      if cand.is_file() and cand not in hits:
        hits.append(cand)
  if not hits:
    missing.append(rel); continue
  digests = {md5(h) for h in hits}
  if len(digests) > 1:
    ambiguous.append((rel, [str(h) for h in hits]))
  src = hits[0]
  dst = DEST / rel
  dst.parent.mkdir(parents=True, exist_ok=True)
  shutil.copyfile(src, dst)
  copied.append(rel)

print(f"copied   : {len(copied)}")
print(f"missing  : {len(missing)}")
for r in missing[:10]:
  print(f"    MISSING {r}")
print(f"ambiguous (same name, different content in >1 root): {len(ambiguous)}")
for rel, hits in ambiguous[:10]:
  print(f"    {rel}")
  for h in hits:
    print(f"        {h}")
if ambiguous:
  print("\nNOTE: the first root won for each of the above. mjlab's own asset tree has top priority,")
  print("so this is the same file mjlab compiled -- but it is reported because a wrong pick here")
  print("would change collision geometry silently.")
sys.exit(1 if missing else 0)
