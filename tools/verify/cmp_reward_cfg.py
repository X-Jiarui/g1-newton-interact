"""Which reward configuration did the checkpoint that worked actually train with?

Our runs take the reward config from load_env_cfg -- the task default -- where exactly one of 37
terms carries weight and its value is bounded by 1. mjlab runs that reached a working grasp logged
mean rewards in the hundreds, which that config cannot produce. The checkpoint ships its own
params/env.yaml; that is the configuration it was trained under.
"""
import os, re, sys

P = os.path.expanduser("~/sweep_ckpts_r2/OF_00_apple_eat_1_SPHERE/params/env.yaml")
txt = open(P).read().split("\n")

# Parse the rewards block textually: the yaml carries Python objects that will not load outside a
# built env, and only names and weights are needed here.
out, name, depth = {}, None, None
in_rewards = False
for i, ln in enumerate(txt):
    if re.match(r"^rewards:", ln): in_rewards = True; continue
    if in_rewards and re.match(r"^[a-z_]+:", ln): break
    if not in_rewards: continue
    m = re.match(r"^  ([a-z0-9_]+):\s*$", ln)
    if m: name = m.group(1); continue
    m = re.match(r"^    weight:\s*([-0-9.eE]+)", ln)
    if m and name: out[name] = float(m.group(1))

nz = {k: v for k, v in out.items() if abs(v) > 1e-12}
print(f"checkpoint env.yaml: {len(out)} reward terms, {len(nz)} with nonzero weight")
for k, v in sorted(nz.items(), key=lambda t: -abs(t[1])):
    print(f"   {k:<40} {v}")

sys.path.insert(0, os.path.expanduser("~/projects/g1-newton-interact/src"))
import mjw_compat; mjw_compat.apply()
import mjlab.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg
cfg = load_env_cfg("Mjlab-ResidualInteract-G1", play=False)
rw = cfg.rewards if isinstance(cfg.rewards, dict) else vars(cfg.rewards)
rw = {k: v for k, v in rw.items() if hasattr(v, "weight")}
nz2 = {k: float(v.weight) for k, v in rw.items() if abs(float(v.weight)) > 1e-12}
print(f"\ntask default (what our training uses): {len(rw)} terms, {len(nz2)} with nonzero weight")
for k, v in sorted(nz2.items(), key=lambda t: -abs(t[1])):
    print(f"   {k:<40} {v}")

only_ckpt = sorted(set(nz) - set(nz2))
print(f"\nactive in the checkpoint but NOT in our config: {len(only_ckpt)}")
for k in only_ckpt[:20]:
    print(f"   {k:<40} weight {nz[k]}")
