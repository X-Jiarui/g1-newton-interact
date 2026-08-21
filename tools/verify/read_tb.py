"""Read the real task metrics out of the tensorboard events.

`Mean reward` is one number over 37 terms of which one carries weight; it cannot say whether the hand
is reaching, whether contact happens, or whether the object ever leaves the table. Those are logged
separately, and they are what decides whether a flat reward means "not yet" or "never".
"""
import glob, os, sys
from collections import defaultdict

run = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/projects/g1-newton-interact/logs/rsl_rl/g1_residual_interact/HEADED_STAPLER")
files = sorted(glob.glob(os.path.join(run, "events.out.tfevents.*")))
if not files:
    raise SystemExit(f"no events under {run}")

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
acc = EventAccumulator(files[-1], size_guidance={"scalars": 0})
acc.Reload()
tags = acc.Tags()["scalars"]
print(f"{os.path.basename(run)}: {len(tags)} scalar tags\n")

WANT = ["lift", "contact", "hand_to_obj", "success", "mpjpe", "reach", "tracking", "object"]
sel = [t for t in tags if any(w in t.lower() for w in WANT)]

print(f"{'tag':<44}{'first':>10}{'last':>10}{'max':>10}")
print("-" * 74)
for t in sorted(sel)[:26]:
    ev = acc.Scalars(t)
    if not ev:
        continue
    vals = [e.value for e in ev]
    n = max(1, len(vals) // 10)
    first = sum(vals[:n]) / n
    last = sum(vals[-n:]) / n
    print(f"{t:<44}{first:>10.4f}{last:>10.4f}{max(vals):>10.4f}")
print(f"\nsteps logged: {len(acc.Scalars(sel[0])) if sel else 0}")
