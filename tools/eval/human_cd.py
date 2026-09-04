"""Human C and D at the contact frame, computed with grasp_cd.py's own definitions.

The robot's C and D are not quantities to minimise. The human demonstration has its own C and D
for that clip and that object; a robot D of 0 where the human's is 2 cm is not a better grasp, it
is a different one. What we want is |robot - human|.

The human landmarks come from grasp_targets.npz, which extract_grasp_targets.py writes in the
OBJECT frame -- so the object centre is the origin and both numbers fall out directly.
"""
import sys, csv, numpy as np

sys.path.insert(0, "/home/jiarui/grab-g1-wuji-pipeline/tools/eval")
from grasp_cd import hull_distance  # the same Frank-Wolfe hull distance the robot side uses

z = np.load(sys.argv[1], allow_pickle=True)
seqs = [str(s) for s in z["seqs"]]
tips = z["tips_skin"]          # (n,5,3) human fingertip skin points, object frame
wrist = z["wrist"]             # (n,3)

human = {}
for i, s in enumerate(seqs):
    six = np.vstack([tips[i], wrist[i][None]])
    human[s] = (float(np.linalg.norm(six.mean(axis=0))) * 100,
                hull_distance(six, np.zeros(3)) * 100)

def load(p):
    return {r["seq"]: (float(r["C_cm"]), float(r["D_cm"]), int(r["n_fingers"]))
            for r in csv.DictReader(open(p))}

runs = [(lab, load(p)) for lab, p in zip(sys.argv[2::2], sys.argv[3::2])]

hdr = f"{'seq':<22}{'C_hum':>7}{'D_hum':>7}"
for lab, _ in runs:
    hdr += f"{'C_'+lab:>9}{'dC':>7}{'D_'+lab:>9}{'dD':>7}"
print(hdr); print('-' * len(hdr))
tot = {lab: [0.0, 0.0] for lab, _ in runs}
for s in sorted(human):
    ch, dh = human[s]
    line = f"{s:<22}{ch:>7.2f}{dh:>7.2f}"
    for lab, r in runs:
        if s not in r:
            line += f"{'-':>9}{'-':>7}{'-':>9}{'-':>7}"; continue
        c, d, _ = r[s]
        line += f"{c:>9.2f}{abs(c-ch):>7.2f}{d:>9.2f}{abs(d-dh):>7.2f}"
        tot[lab][0] += abs(c - ch); tot[lab][1] += abs(d - dh)
    print(line)
print()
for lab, r in runs:
    n = max(len(r), 1)
    print(f"  {lab:<12} mean |C-C_human| = {tot[lab][0]/n:.2f} cm   "
          f"mean |D-D_human| = {tot[lab][1]/n:.2f} cm")
