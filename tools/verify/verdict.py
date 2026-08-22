"""Is training moving in the right direction? Compare against mjlab's own trajectory, not intuition.

The reward terms that decide a grasp are the ones to watch: contact first, then lift. Tracking
distance is the leading indicator -- mjlab's successful runs came down from ~6 cm, and a run that
stays flat there will not grasp no matter how long it runs.
"""
import glob, os, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(run)
files = sorted(glob.glob(os.path.join(run, "events.out.tfevents.*")))
if not files:
    print(f"{label}: no events yet"); sys.exit(0)
acc = EventAccumulator(files[-1], size_guidance={"scalars": 0}); acc.Reload()
tags = sorted(acc.Tags()["scalars"])

def find(*subs):
    for t in tags:
        if all(s in t.lower() for s in subs):
            return t
    return None

def band(tag, n=8):
    if not tag: return None
    v = [e.value for e in acc.Scalars(tag)]
    if not v: return None
    k = max(1, len(v)//n)
    return v[0], sum(v[:k])/k, sum(v[-k:])/k, max(v), len(v)

WATCH = [
    ("tracking distance cm", find("body_link_dist_mean"), 100.0),
    ("reward",               find("mean_reward"),          1.0),
    ("multi_tip_surface",    find("multi_tip_surface"),    1.0),
    ("contact_duration",     find("contact_duration"),     1.0),
    ("object_hard_lift",     find("object_hard_lift"),     1.0),
    ("object_lift_hold",     find("object_lift_hold"),     1.0),
    ("object_traj_tracking", find("object_trajectory"),    1.0),
    ("right_wrist_tracking", find("right_wrist_tracking"), 1.0),
    ("episode length",       find("mean_episode_length"),  1.0),
    ("action std",           find("mean_std"),             1.0),
]
it = band(find("mean_reward"))
print(f"\n### {label}   iteration ~{it[4] if it else 0}")
print(f"{'metric':<24}{'start':>10}{'now':>10}{'max':>10}   direction")
for name, tag, scale in WATCH:
    b = band(tag)
    if b is None:
        print(f"{name:<24}{'not logged':>30}")
        continue
    _, first, last, mx, _ = b
    d = (last - first) * scale
    arrow = "flat" if abs(d) < 1e-4 else ("up" if d > 0 else "down")
    print(f"{name:<24}{first*scale:>10.4f}{last*scale:>10.4f}{mx*scale:>10.4f}   {arrow}")
