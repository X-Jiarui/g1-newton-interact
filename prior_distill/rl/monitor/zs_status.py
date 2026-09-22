import re, os, sys, statistics as st
ARMS = ("ZS_ZP", "ZS_ZP_ZN_OURLR", "ZS_ZP_ZN_NOREF")
KEYS = {"reward": r"Mean reward: ([\d.\-]+)", "eplen": r"Mean episode length: ([\d.]+)",
        "obj5": r"Metrics/hand_to_obj_under_005_frac: ([\d.]+)", "obj15": r"Metrics/hand_to_obj_under_015_frac: ([\d.]+)",
        "wrist": r"Metrics/left_wrist_link_dist_mean: ([\d.]+)", "body": r"Metrics/body_link_dist_mean: ([\d.]+)",
        "contact": r"Metrics/physical_contact: ([\d.]+)", "fell": r"Termination/fell_over: ([\d.]+)",
        "coll": r"Collection time: ([\d.]+)s", "learn": r"Learning time: ([\d.]+)s", "fps": r"Steps/s: ([\d.]+)|fps: ([\d.]+)|([\d.]+) steps/s"}
def parse(path):
    txt = open(path, "rb").read().decode("utf-8", "replace")
    blocks = re.split(r"Learning iteration (\d+)/\d+", txt)
    rows = []
    for i in range(1, len(blocks) - 1, 2):
        it = int(blocks[i]); body = blocks[i + 1]
        r = {"iter": it}
        for k, pat in KEYS.items():
            m = re.search(pat, body)
            if m:
                r[k] = float(next(g for g in m.groups() if g is not None))
        rows.append(r)
    return rows
def m(rows, k):
    v = [r[k] for r in rows if k in r]
    return st.mean(v) if v else float("nan")
out = []
for a in ARMS:
    p = os.path.expanduser(f"~/zs/logs/{a}.log")
    if not os.path.exists(p):
        out.append(f"{a}: no log"); continue
    rows = parse(p)
    alive = os.popen(f"pgrep -u jrxu -f 'run-name {a}$' | head -1").read().strip() != ""
    if not rows:
        out.append(f"{a}: it=0 alive={int(alive)} (no iteration yet)"); continue
    w = rows[-10:]
    sit = m(w, "coll") + m(w, "learn")
    out.append(f"{a}: it={rows[-1]['iter']} alive={int(alive)} reward={m(w,'reward'):.1f} eplen={m(w,'eplen'):.0f} obj<5cm={m(w,'obj5'):.3f} obj<15cm={m(w,'obj15'):.3f} "
               f"wrist={m(w,'wrist'):.3f} body={m(w,'body'):.3f} contact={m(w,'contact'):.3f} fell={m(w,'fell'):.3f} s/it={sit:.0f}")
# equal-budget rows: every arm that is AHEAD of another, evaluated at that other arm's newest iteration
alls = {a: parse(os.path.expanduser(f"~/zs/logs/{a}.log")) for a in ARMS if os.path.exists(os.path.expanduser(f"~/zs/logs/{a}.log"))}
alls = {a: r for a, r in alls.items() if r}
newest = {a: r[-1]["iter"] for a, r in alls.items()}
for ib in sorted(set(newest.values()))[:-1]:
    for a, rows in alls.items():
        if newest[a] <= ib:
            continue
        w = [r for r in rows if ib - 9 <= r["iter"] <= ib]
        if w:
            out.append(f"{a}@it{ib}: reward={m(w,'reward'):.1f} obj<5cm={m(w,'obj5'):.3f} obj<15cm={m(w,'obj15'):.3f} wrist={m(w,'wrist'):.3f} body={m(w,'body'):.3f} contact={m(w,'contact'):.3f} fell={m(w,'fell'):.3f}")
print("\n".join(out))
