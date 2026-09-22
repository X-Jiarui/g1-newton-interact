import re, sys
t = open(sys.argv[1], "rb").read().decode("utf-8", "replace")
bl = re.split(r"Learning iteration (\d+)/\d+", t)
lo = int(sys.argv[2]) if len(sys.argv) > 2 else 0
step = int(sys.argv[3]) if len(sys.argv) > 3 else 2
P = {"reward": r"Mean reward: ([\d.\-]+)", "kl": r"Mean kl_mean loss: ([\d.e\-]+)", "klmax": r"Mean kl_max loss: ([\d.e\-]+)",
     "ratio_max": r"Mean ratio_max loss: ([\d.e\-]+)", "lograt": r"Mean log_ratio_abs_max loss: ([\d.e\-]+)",
     "vloss": r"Mean value loss: ([\d.e\-]+)", "surr": r"Mean surrogate loss: ([\d.\-e]+)", "ent": r"Mean entropy loss: ([\d.\-e]+)",
     "std": r"Mean action noise std: ([\d.]+)", "tok": r"token_residual_norm: ([\d.]+)", "fell": r"Termination/fell_over: ([\d.]+)"}
def g(b, p):
    m = re.search(p, b); return float(m.group(1)) if m else float("nan")
for i in range(1, len(bl) - 1, 2):
    it = int(bl[i])
    if it >= lo and it % step == 0:
        b = bl[i + 1]
        v = {k: g(b, p) for k, p in P.items()}
        print("it=%d reward=%.0f kl=%.4f klmax=%.3f ratio_max=%.2f lograt_max=%.1f vloss=%.0f surr=%.3f ent=%.0f std=%.3f tok=%.2f fell=%.2f" % (
            it, v["reward"], v["kl"], v["klmax"], v["ratio_max"], v["lograt"], v["vloss"], v["surr"], v["ent"], v["std"], v["tok"], v["fell"]))
