#!/usr/bin/env python3
"""Translate one clip object (and its table) by a fixed world offset, one copy per offset.

Why the table moves with the object: this pipeline already settled that question. Moving the
object alone changed the object-table relationship from a 2.9 cm spread to 7.9 cm -- objects
floating up to 3 cm or sunk 4.9 cm -- so `align_object_to_hand.py` translates the table by the
SAME vector and the pair moves as a unit. Doing anything else here would invent a second,
contradictory convention.

The filename and its parent directory are NOT changed. `extract_grasp_targets.py` locates the
GRAB source as `<grab_dir>/grab/<parent dir name>/<file stem>.npz`, so a renamed variant simply
fails to find its own annotations. Each variant therefore gets its own DATA_ROOT holding the same
relative path, and the renaming happens only after step 6, for training, which never reads GRAB.

Nothing else in the clip is touched. In particular the contact labels are per-vertex on the
object mesh and a translation does not change which vertices the human touched, which is the
whole reason re-solving the IK against them is legitimate.
"""
import argparse, pathlib, pickle, shutil
import numpy as np

# 3 cm in six axis directions plus the two horizontal diagonals, all the same magnitude so no
# variant is a harder task merely by being further away.
D = 0.03
Q = float(D / np.sqrt(2.0))
OFFSETS = {
    # The control. Zero offset, but regenerated through the SAME pipeline invocation as the eight
    # variants: the existing step-6 copy of this clip was produced by another environment, and a
    # control that differs from the treatment by anything other than the offset is not a control.
    "orig": (0.0,  0.0, 0.0),
    "xp":  ( D,   0.0, 0.0), "xm":  (-D,   0.0, 0.0),
    "yp":  (0.0,  D,   0.0), "ym":  (0.0, -D,   0.0),
    "zp":  (0.0,  0.0,  D),  "zm":  (0.0,  0.0, -D),
    "dpp": ( Q,   Q,   0.0), "dmm": (-Q,  -Q,   0.0),
}

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="step-3 clip, e.g. .../grab_wuji_o70_full/s1/cubesmall_lift.pkl")
ap.add_argument("--work", required=True, help="directory to hold one DATA_ROOT per variant")
ap.add_argument("--only", default="", help="comma-separated variant tags; default all")
a = ap.parse_args()

src = pathlib.Path(a.src).resolve()
rel = pathlib.Path(src.parent.name) / src.name          # s1/cubesmall_lift.pkl -- preserved exactly
work = pathlib.Path(a.work); work.mkdir(parents=True, exist_ok=True)
tags = [t for t in a.only.split(",") if t] or list(OFFSETS)

with src.open("rb") as f:
    base = pickle.load(f)
o0 = np.asarray(base["object"]["pos_mj"], np.float64)
t0 = np.asarray(base["table"]["pos_mj"], np.float64)
print(f"src {src}  frames {o0.shape[0]}  object[0] {np.round(o0[0], 4)}  table[0] {np.round(t0[0], 4)}")

for tag in tags:
    dx = np.asarray(OFFSETS[tag], np.float64)
    d = pickle.loads(pickle.dumps(base))
    d["object"]["pos_mj"] = (o0 + dx).astype(np.float32)
    d["table"]["pos_mj"] = (t0 + dx).astype(np.float32)
    d["placement_randomization"] = {"tool": "make_variants", "tag": tag,
                                    "offset_m": [float(v) for v in dx],
                                    "moved_table_with_object": True, "src": str(src)}
    out = work / tag / "in" / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(d, f, protocol=4)
    chk = np.asarray(d["object"]["pos_mj"], np.float64) - o0
    print(f"  {tag:4s} offset {np.round(dx*100,2)} cm  realised {np.round(chk[0]*100,2)} cm  -> {out}")
