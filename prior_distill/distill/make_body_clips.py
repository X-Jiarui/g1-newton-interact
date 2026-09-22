#!/usr/bin/env python
"""Turn GRAB retargeted clips into body-only tracking clips (GRAB schema, object parked out of reach).

For the ASTRA-tracking distillation line the object is irrelevant: only the human body motion is
wanted, the way PULSE-X treats AMASS. The residual_interact task still needs an object and a table
in every clip, so the same trick as the office-walk tracking run is applied, without any code change
on the task side:

  * the object and its table are parked PARK_M metres to the side of the clip's start and never
    move, so the live object sits exactly on its own reference and none of the object terminations
    (object_drift, og_object_far, object_leash) can ever fire;
  * the GRAB contact label is rewritten so the first contact frame is the LAST frame: everything
    phase-gated on contact treats the whole episode as pre-contact and the table is never removed;
  * the object mesh is the 12-triangle cubesmall for every clip (the launcher's obj_name check is
    skipped in body mode).

Everything not needed by apple_eat/mdp.py:_load_ref_single is dropped, which takes a clip from
10-27 MB to ~100-300 KB, so the whole 1324-clip set moves over a slow link in minutes.

Usage:  python make_body_clips.py <src_root> <dst_root> [--park-m 15] [--min-frames 60]
        (walks <src_root>/s*/*.pkl, mirrors the layout under <dst_root>)
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--park-m", type=float, default=15.0)
ap.add_argument("--min-frames", type=int, default=60)
ap.add_argument("--stl-name", default="cubesmall.stl")
A = ap.parse_args()

src, dst = Path(A.src), Path(A.dst)
paths = sorted(src.glob("s*/*.pkl"))
if not paths:
  raise SystemExit(f"no s*/*.pkl under {src}")
dst.mkdir(parents=True, exist_ok=True)
(dst / "meshes").mkdir(exist_ok=True)
kept, skipped, listing = 0, [], []
for p in paths:
  with p.open("rb") as f:
    d = pickle.load(f)
  r53 = d["robot_53dof"]
  root_pos = np.asarray(r53["root_pos"], dtype=np.float32)
  root_rot = np.asarray(r53["root_rot"], dtype=np.float32)
  dof = np.asarray(r53["dof_pos"], dtype=np.float32)
  n = int(dof.shape[0])
  if n < A.min_frames or root_pos.shape[0] != n or root_rot.shape[0] != n:
    skipped.append((p.name, f"frames={n}"))
    continue
  if not (np.isfinite(root_pos).all() and np.isfinite(root_rot).all() and np.isfinite(dof).all()):
    skipped.append((p.name, "non-finite"))
    continue
  obj = d["object"]
  table = d["table"]
  obj_pos0 = np.asarray(obj["pos_mj"], dtype=np.float32)[0]
  obj_quat0 = np.asarray(obj["quat_wxyz_mj"], dtype=np.float32)[0]
  tab_pos0 = np.asarray(table["pos_mj"], dtype=np.float32)[0]
  tab_quat0 = np.asarray(table["quat_wxyz_mj"], dtype=np.float32)[0]
  # Park sideways relative to the start heading: the loader canonicalises every clip so the root
  # starts at the origin facing +x, so "+y in the start frame" is a fixed direction for all clips.
  # The start-frame heading is the yaw of root_rot[0] (xyzw in robot_53dof).
  x, y, z, w = root_rot[0]
  yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  side = np.array([-np.sin(yaw), np.cos(yaw), 0.0], dtype=np.float32) * float(A.park_m)
  park_obj = np.tile(obj_pos0 + side, (n, 1))
  park_obj[:, 2] = obj_pos0[2]
  park_tab = np.tile(tab_pos0 + side, (n, 1))
  park_tab[:, 2] = tab_pos0[2]
  contact = np.zeros((n, 1), dtype=np.float32)
  contact[-1, 0] = 1.0  # first contact = last frame -> cf pinned to the end
  out = {
    "sequence_name": d.get("sequence_name", p.stem),
    "subject": d.get("subject", p.parent.name),
    "obj_name": "cubesmall",
    "source_obj_name": d.get("obj_name", ""),
    "fps": float(d.get("fps", 30.0)),
    "n_frames": n,
    "body_only": True,
    "robot_53dof": {"root_pos": root_pos, "root_rot": root_rot, "dof_pos": dof,
                    "dof_names": list(r53.get("dof_names", []))},
    "object": {"pos_mj": park_obj, "quat_wxyz_mj": np.tile(obj_quat0, (n, 1)),
               "stl_path": A.stl_name},
    "table": {"pos_mj": park_tab, "quat_wxyz_mj": np.tile(tab_quat0, (n, 1)),
              "stl_path": str(table.get("stl_path", ""))},
    "contact": {"object": contact, "body": None, "threshold": 0.0},
  }
  o = dst / p.parent.name / p.name
  o.parent.mkdir(parents=True, exist_ok=True)
  with o.open("wb") as f:
    pickle.dump(out, f, protocol=4)
  kept += 1
  listing.append(str(o.resolve()))
  if kept % 100 == 0:
    print(f"{kept} clips ...", flush=True)

(dst / "clips.txt").write_text("\n".join(listing) + "\n")
print(f"wrote {kept} clips to {dst} ({len(skipped)} skipped); list in {dst / 'clips.txt'}")
for name, why in skipped[:20]:
  print(f"  skipped {name}: {why}")
print(f"put the object mesh at {dst / 'meshes' / A.stl_name} (the loader searches <clip>/../../meshes)")
