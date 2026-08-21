"""Do mjlab and the Newton training env produce the same reward for the same actions?

The Newton env reports a reward of exactly 0.0000 for its first steps. That is either correct -- only
one of the 37 terms carries weight, and it may genuinely be zero during the startup hold -- or it is
a term that never fires, which is the failure this project keeps meeting. mjlab is the reference, so
the answer is a comparison, not an argument.

Both envs are reset to the same reference frame and driven by an identical action sequence.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np, torch, yaml as _yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
import mjw_compat; mjw_compat.apply()
import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.scripts.play import _apply_cfg_mapping
from newton_vec_env import NewtonVecEnv

XML = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets/mjlab_scene/scene.xml")
CKPT = os.path.expanduser("~/sweep_ckpts_r2/OF_00_apple_eat_1_SPHERE/model_7310.pt")
TASK = "Mjlab-ResidualInteract-G1"
N = 4
STEPS = 40

def make_cfg():
    c = load_env_cfg(TASK, play=False); c.scene.num_envs = N
    a = load_rl_cfg(TASK)
    _apply_cfg_mapping(a, _yaml.unsafe_load((Path(CKPT).parent / "params" / "agent.yaml").open()))
    s = c.actions.get("sonic_action") if isinstance(c.actions, dict) else c.actions.sonic_action
    s.tracking_start_assist_gain = 0.0; s.tracking_start_assist_steps = 0
    from mjlab.tasks.residual_interact.env_cfgs import set_astra_body_dynamics
    set_astra_body_dynamics(c)
    return c

cfg_mj = make_cfg()
mjenv = ManagerBasedRlEnv(cfg=cfg_mj, device="cuda:0", render_mode=None)
u = mjenv.unwrapped
from mjlab.tasks.residual_interact.env_cfgs import install_astra_body_pd, install_object_variant_sizes
install_astra_body_pd(u); install_object_variant_sizes(u)
u._force_reference_start_frame = 0
mjenv.reset()

nenv = NewtonVecEnv(make_cfg(), XML, num_envs=N, device="cuda:0")
nenv._env._force_reference_start_frame = 0
nenv.reset()

print(f"{'step':>5} {'mjlab reward':>14} {'newton reward':>14} {'|diff|':>10}")
print("-" * 48)
torch.manual_seed(0)
acts = [(torch.rand(N, 69, device="cuda:0") - 0.5) * 0.1 for _ in range(STEPS)]
worst = 0.0
for k, a in enumerate(acts):
    out_mj = mjenv.step(a.clone())
    r_mj = out_mj[1] if len(out_mj) > 1 else None
    _, r_nt, _, _ = nenv.step(a.clone())
    dm, dn = float(r_mj.mean()), float(r_nt.mean())
    worst = max(worst, abs(dm - dn))
    if k % 5 == 0 or k == STEPS - 1:
        print(f"{k:5d} {dm:14.6f} {dn:14.6f} {abs(dm-dn):10.3g}")
print("-" * 48)
print(f"worst |reward diff| over {STEPS} steps: {worst:.6g}")
print(f"mjlab reward all-zero: {all(float(mjenv.step(a.clone())[1].mean()) == 0.0 for a in acts[:3])}")
