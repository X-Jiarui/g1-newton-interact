"""What did body_link_dist_mean look like early in an mjlab run that eventually worked?

The Newton run sits at 14.2 cm and flat after 200 iterations. Whether that is a problem depends
entirely on what mjlab looked like at the same point -- if mjlab was also at 14 cm and only came down
later, this is 'not yet'; if mjlab was already falling, this is 'never'.
"""
import glob, os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = os.path.expanduser(
    "~/projects/mjlab-astra-dagger-distill-20260625/logs/rsl_rl/g1_residual_interact")
runs = sorted(glob.glob(os.path.join(ROOT, "*")))
print(f"{len(runs)} mjlab runs; looking for ones with tracking metrics and >=200 iterations\n")

hits = []
for r in runs:
    ev = sorted(glob.glob(os.path.join(r, "events.out.tfevents.*")))
    if not ev:
        continue
    try:
        acc = EventAccumulator(ev[-1], size_guidance={"scalars": 0}); acc.Reload()
        tags = acc.Tags()["scalars"]
    except Exception:
        continue
    tag = next((t for t in tags if "body_link_dist_mean" in t), None)
    rew = next((t for t in tags if "mean_reward" in t and "/time" not in t), None)
    if not tag:
        continue
    s = acc.Scalars(tag)
    if len(s) < 100:
        continue
    hits.append((len(s), os.path.basename(r), acc, tag, rew))

hits.sort(reverse=True)
for n, name, acc, tag, rew in hits[:6]:
    d = [e.value for e in acc.Scalars(tag)]
    r = [e.value for e in acc.Scalars(rew)] if rew else []
    def at(v, i):
        return v[min(i, len(v)-1)] if v else float("nan")
    print(f"=== {name[:56]}  ({n} iterations) ===")
    print(f"   body_link_dist_mean   it0={at(d,0)*100:6.2f}cm  it100={at(d,100)*100:6.2f}  "
          f"it200={at(d,200)*100:6.2f}  it500={at(d,500)*100:6.2f}  final={d[-1]*100:6.2f}")
    if r:
        print(f"   mean_reward           it0={at(r,0):7.4f}  it100={at(r,100):7.4f}  "
              f"it200={at(r,200):7.4f}  it500={at(r,500):7.4f}  final={r[-1]:7.4f}")
