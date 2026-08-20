"""Does install_astra_body_pd change the PD that set_astra_body_dynamics already put in place?

play.py does BOTH: set_astra_body_dynamics(env_cfg) before the env exists, then
install_astra_body_pd(env) after. The second one rewrites actuator_gainprm/biasprm/forcerange for the
29 body joints directly on the compiled model. Every model fact-sheet recorded so far was taken
without it, so if the two disagree the baseline -- and the Newton model matched against it -- describe
a system that is not the one play.py runs.

Measured, not argued: dump the gains before and after the call and diff them.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np, yaml as _yaml

CKPT = os.path.expanduser(
  "~/projects/mjlab-astra-dagger-distill-20260625/logs/rsl_rl/g1_residual_interact/REGRESS_EAT1/model_8500.pt")

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.scripts.play import _apply_cfg_mapping
import mujoco

TASK = "Mjlab-ResidualInteract-G1"
env_cfg = load_env_cfg(TASK, play=True); env_cfg.scene.num_envs = 1
agent_cfg = load_rl_cfg(TASK)
_apply_cfg_mapping(agent_cfg, _yaml.unsafe_load((Path(CKPT).parent/"params"/"agent.yaml").open()))

# play.py: assist gain override. Every candidate trained with it at 0.0.
_sonic = getattr(env_cfg.actions, "sonic_action", None) or env_cfg.actions["sonic_action"]
_sonic.tracking_start_assist_gain = 0.0
_sonic.tracking_start_assist_steps = 0

from mjlab.tasks.residual_interact.env_cfgs import set_astra_body_dynamics, install_astra_body_pd
set_astra_body_dynamics(env_cfg)

env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode=None)
u = env.unwrapped
m = u.sim.mj_model

before = dict(gainprm=m.actuator_gainprm.copy(), biasprm=m.actuator_biasprm.copy(),
              forcerange=m.actuator_forcerange.copy())
term = u.action_manager.get_term("sonic_action")
t_before = (term._kp.clone(), term._kd.clone(), term._effort_limit.clone())

install_astra_body_pd(u)

after = dict(gainprm=m.actuator_gainprm, biasprm=m.actuator_biasprm, forcerange=m.actuator_forcerange)
print("=== compiled model actuator params, before vs after install_astra_body_pd ===")
total = 0
for k in before:
    d = np.abs(before[k] - after[k])
    n = int((d > 1e-9).sum())
    total += n
    print(f"  {k:12s} entries changed: {n:5d}   max |delta| = {d.max():.6g}")
print(f"  TOTAL changed entries: {total}")

print("\n=== action term buffers ===")
for name, old, new in (("_kp", t_before[0], term._kp), ("_kd", t_before[1], term._kd),
                       ("_effort_limit", t_before[2], term._effort_limit)):
    d = (old - new).abs()
    print(f"  {name:14s} changed dims: {int((d > 1e-9).sum()):3d} / {d.numel()}   max |delta| = {float(d.max()):.6g}")

if total == 0:
    print("\nIDEMPOTENT: set_astra_body_dynamics already produced these values; the recorded "
          "baseline describes what play.py runs.")
else:
    print("\nNOT IDEMPOTENT: the runtime install changes the PD. Every fact-sheet taken without it "
          "is wrong, and the Newton model was matched against the wrong target.")
    ids = np.argwhere(np.abs(before["gainprm"] - after["gainprm"]) > 1e-9)
    for i in ids[:6]:
        a = int(i[0])
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        print(f"    {nm}: kp {before['gainprm'][a,0]:.4f} -> {after['gainprm'][a,0]:.4f}")
env.close()
