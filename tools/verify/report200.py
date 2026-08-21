"""Three-layer read of the task metrics: is the physics alive, is it approaching, is it learning?

Aggregate reward cannot answer any of these -- it is one number over 37 terms of which one carries
weight, and a sanitised NaN looks exactly like a steady policy. These are the terms that separate
"not yet" from "never".
"""
import glob, os, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = sys.argv[1]
files = sorted(glob.glob(os.path.join(run, "events.out.tfevents.*")))
if not files:
    print(f"{os.path.basename(run)}: no events"); sys.exit(0)
acc = EventAccumulator(files[-1], size_guidance={"scalars": 0}); acc.Reload()
tags = set(acc.Tags()["scalars"])

def pick(*subs):
    for t in sorted(tags):
        low = t.lower()
        if all(s in low for s in subs):
            return t
    return None

def trend(tag):
    if tag is None: return None
    ev = acc.Scalars(tag)
    if not ev: return None
    v = [e.value for e in ev]
    n = max(1, len(v)//8)
    return (sum(v[:n])/n, sum(v[-n:])/n, max(v), len(v))

LAYERS = [
    ("1 physics", [("object height",  pick("object", "z")), ("object mpjpe mm", pick("mpjpe")),
                   ("episode length", pick("episode_length"))]),
    ("2 approach", [("hand-object dist", pick("hand_to_obj_dist")),
                    ("hand<5cm frac",    pick("hand_to_obj_under")),
                    ("contact",          pick("physical_contact")),
                    ("live contact",     pick("live_contact"))]),
    ("3 learning", [("lift_success",   pick("lift_success")),
                    ("lift_duration",  pick("lift_duration")),
                    ("seq success",    pick("sequence_success")),
                    ("reward",         pick("mean_reward"))]),
]
print(f"\n### {os.path.basename(run)}   ({len(tags)} tags)")
print(f"{'':<12}{'metric':<20}{'first':>11}{'last':>11}{'max':>11}")
for name, items in LAYERS:
    for label, tag in items:
        t = trend(tag)
        if t is None:
            print(f"{name:<12}{label:<20}{'not logged':>33}")
        else:
            f, l, m, n = t
            print(f"{name:<12}{label:<20}{f:>11.4f}{l:>11.4f}{m:>11.4f}")
    name = ""
