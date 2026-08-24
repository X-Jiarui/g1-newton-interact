#!/usr/bin/env python3
"""Rank reference clips by how far the object actually travels.

The point of the ranking is to find the sequences where object tracking is the task -- a clip in
which the object barely leaves its start pose cannot show whether a policy is tracking it. Four
numbers are reported per clip because they disagree in useful ways:

  path_len_m     summed frame-to-frame displacement. The headline number: total distance travelled.
                 Sensitive to jitter, so a stationary noisy object accumulates some of this.
  net_disp_m     |last - first|. Zero for a clip that returns the object to where it started,
                 which is most "pass" and "inspect" clips.
  max_from_start furthest the object ever gets from its start pose. Immune to jitter and to
                 round trips, which makes it the best tie-breaker against path_len_m.
  z_range_m      vertical extent. Separates lifts from slides.

Ranking is by path_len_m, but the manifest carries all four and flags any clip whose path length
looks like accumulated jitter (path_len_m large while max_from_start is small).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np

JITTER_RATIO = 8.0  # path_len / max_from_start above this reads as jitter, not travel


def clip_stats(path: str) -> dict | None:
  try:
    with open(path, "rb") as f:
      d = pickle.load(f)
  except Exception as e:
    return {"path": path, "error": f"{type(e).__name__}: {e}"}
  obj = d.get("object")
  if not isinstance(obj, dict) or "pos_mj" not in obj:
    return {"path": path, "error": "no object.pos_mj"}
  p = np.asarray(obj["pos_mj"], dtype=np.float64)
  if p.ndim != 2 or p.shape[0] < 2:
    return {"path": path, "error": f"object.pos_mj has shape {p.shape}"}
  steps = np.linalg.norm(np.diff(p, axis=0), axis=1)
  from_start = np.linalg.norm(p - p[0], axis=1)
  fps = float(d.get("fps") or 0.0)
  n = int(p.shape[0])
  path_len = float(steps.sum())
  max_from_start = float(from_start.max())
  return {
    "sequence": d.get("sequence_name") or os.path.basename(path)[:-4],
    "subject": d.get("subject", ""),
    "object": d.get("obj_name", ""),
    "intent": d.get("motion_intent", ""),
    "frames": n,
    "fps": fps,
    "duration_s": round(n / fps, 3) if fps else None,
    "path_len_m": round(path_len, 5),
    "net_disp_m": round(float(np.linalg.norm(p[-1] - p[0])), 5),
    "max_from_start_m": round(max_from_start, 5),
    "z_range_m": round(float(p[:, 2].max() - p[:, 2].min()), 5),
    "max_step_m": round(float(steps.max()), 5),
    "jitter_suspect": bool(max_from_start > 1e-9 and path_len / max_from_start > JITTER_RATIO),
    "path": path,
  }


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--dataset", required=True, help="directory searched recursively for *.pkl")
  ap.add_argument("--out", required=True, help="JSON manifest to write")
  ap.add_argument("--top", type=int, default=100)
  ap.add_argument("--min-frames", type=int, default=2)
  ap.add_argument("--exclude-jitter", action="store_true",
                  help="drop clips whose path length looks like accumulated jitter")
  a = ap.parse_args()

  paths = []
  for root, _, files in os.walk(a.dataset):
    for f in sorted(files):
      if f.endswith(".pkl"):
        paths.append(os.path.join(root, f))
  print(f"[rank] {len(paths)} clip(s) under {a.dataset}")

  rows, bad = [], []
  for p in paths:
    r = clip_stats(p)
    if r is None or "error" in r:
      bad.append(r)
      continue
    if r["frames"] < a.min_frames:
      bad.append({**r, "error": f"only {r['frames']} frames"})
      continue
    rows.append(r)
  if bad:
    print(f"[rank] {len(bad)} clip(s) unusable; first few: "
          + "; ".join(f"{os.path.basename(b['path'])}: {b.get('error')}" for b in bad[:3]))

  # The same clip can sit in several dataset directories (the working copies on this box carry
  # three copies of apple_lift). Rank each (subject, sequence) once, keeping the longest version.
  best: dict[tuple, dict] = {}
  for r in rows:
    key = (r["subject"], r["sequence"])
    if key not in best or r["frames"] > best[key]["frames"]:
      best[key] = r
  if len(best) != len(rows):
    print(f"[rank] {len(rows)} clip files collapse to {len(best)} distinct (subject, sequence)")
  rows = list(best.values())

  kept = [r for r in rows if not (a.exclude_jitter and r["jitter_suspect"])]
  kept.sort(key=lambda r: r["path_len_m"], reverse=True)
  top = kept[:a.top]

  print(f"\n[rank] top {min(a.top, len(top))} of {len(kept)} by object path length\n")
  print(f"{'#':>4} {'sequence':32s} {'obj':14s} {'path_m':>8} {'net_m':>8} "
        f"{'maxfs_m':>8} {'z_m':>7} {'frames':>7}")
  for i, r in enumerate(top[:40], 1):
    print(f"{i:4d} {r['sequence'][:32]:32s} {r['object'][:14]:14s} "
          f"{r['path_len_m']:8.3f} {r['net_disp_m']:8.3f} {r['max_from_start_m']:8.3f} "
          f"{r['z_range_m']:7.3f} {r['frames']:7d}"
          + ("  <- jitter?" if r["jitter_suspect"] else ""))
  if len(top) > 40:
    print(f"     ... {len(top) - 40} more in the manifest")

  if rows:
    pl = np.array([r["path_len_m"] for r in rows])
    print(f"\n[rank] path length over all {len(rows)} usable clips: "
          f"median {np.median(pl):.3f} m, p90 {np.percentile(pl, 90):.3f} m, max {pl.max():.3f} m")
    n_jit = sum(1 for r in rows if r["jitter_suspect"])
    print(f"[rank] {n_jit} clip(s) flagged jitter-suspect "
          f"(path_len / max_from_start > {JITTER_RATIO})")

  os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
  with open(a.out, "w") as f:
    json.dump({"dataset": a.dataset, "clips_found": len(paths), "usable": len(rows),
               "unusable": bad, "ranked_by": "path_len_m", "top_n": a.top,
               "top": top, "all": rows}, f, indent=2)
  print(f"[rank] manifest -> {a.out}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
