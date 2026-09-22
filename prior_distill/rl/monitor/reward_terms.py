import re, sys
paths = sys.argv[1:-1]; it0 = int(sys.argv[-1])
rows = {}
keys = []
for p in paths:
    t = open(p, "rb").read().decode("utf-8", "replace")
    bl = re.split(r"Learning iteration (\d+)/\d+", t)
    acc = {}
    n = 0
    for i in range(1, len(bl) - 1, 2):
        it = int(bl[i])
        if it0 - 4 <= it <= it0 + 4:
            for m in re.finditer(r"(Episode_Reward/\S+|Mean reward|Metrics/(?:body_link_dist_mean|left_wrist_link_dist_mean|hand_to_obj_under_005_frac|physical_contact|ep_len)|Termination/(?:fell_over|og_object_far|time_out)):\s*([\d.\-e]+)", bl[i + 1]):
                acc.setdefault(m.group(1), []).append(float(m.group(2)))
            n += 1
    name = p.split("/")[-1].replace(".log", "")
    rows[name] = {k: sum(v) / len(v) for k, v in acc.items()}
    for k in acc:
        if k not in keys: keys.append(k)
print("term".ljust(46) + "".join(n.rjust(12) for n in rows))
for k in keys:
    print(k.ljust(46) + "".join(("%.3f" % rows[n].get(k, float("nan"))).rjust(12) for n in rows))
