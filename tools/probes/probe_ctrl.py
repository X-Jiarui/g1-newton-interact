"""What actually reaches physics: the ctrl vector, measured with the trained policy running.

Sonic53Action calls BOTH set_joint_position_target and set_joint_effort_target. Which of them moves
the robot is not readable from the action term -- it depends on what the 138 model actuators do with
those writes. The compiled model says the 29 body `xml_motor_unused_*` actuators have
forcerange=[0,0], so effort written there cannot produce force. But the 40 finger `xml_motor_unused_*`
are position servos with a real forcerange, so the same reasoning does NOT extend to them.

So it gets measured: run the real policy, then read ctrl back per actuator class. Whatever is
non-constant here is what the Newton port must reproduce; whatever is pinned is a passive spring that
must still be present, and whatever is zero is genuinely dead weight.

Also records the action-order -> joint-index permutation (Sonic53Action._target_ids), which is the
mapping the port needs and which exists nowhere else in a readable form.
"""
from __future__ import annotations
import json, os
from dataclasses import asdict
from pathlib import Path
import numpy as np, torch, yaml as _yaml

CKPT = os.path.expanduser("~/sweep_ckpts_r2/OF_00_apple_eat_1_SPHERE/model_7310.pt")
OUT = os.path.expanduser("~/projects/g1-newton-interact/docs/ctrl_probe.json")

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper, MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.scripts.play import _apply_cfg_mapping
import mujoco

TASK = "Mjlab-ResidualInteract-G1"
env_cfg = load_env_cfg(TASK, play=True); env_cfg.scene.num_envs = 1
agent_cfg = load_rl_cfg(TASK)
_apply_cfg_mapping(agent_cfg, _yaml.unsafe_load((Path(CKPT).parent/"params"/"agent.yaml").open()))
if str(getattr(agent_cfg,"base_tracker_kind","")).strip().lower()=="astra_onnx":
  from mjlab.tasks.residual_interact.env_cfgs import set_astra_body_dynamics
  set_astra_body_dynamics(env_cfg)
env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode=None)
u = env.unwrapped; robot = u.scene["robot"]; m = u.sim.mj_model
wrapped = RslRlVecEnvWrapper(env)
runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(wrapped, asdict(agent_cfg), device="cuda:0")
runner.load(CKPT); policy = runner.get_inference_policy(device="cuda:0")

term = u.action_manager._terms["sonic_action"]
tids = term._target_ids.detach().cpu().numpy().tolist()
jnames = list(robot.joint_names)
action_order = [jnames[i] for i in tids]

ctrl_ids = robot.indexing.ctrl_ids.detach().cpu().numpy().tolist()
anames = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, int(i)) for i in ctrl_ids]

env.reset()
o = wrapped.get_observations(); o = o[0] if isinstance(o, tuple) else o
hist = []
for k in range(60):
  with torch.inference_mode():
    a = policy(o)
  r = wrapped.step(a); o = r[0] if isinstance(r, tuple) else r
  if k >= 40:
    hist.append(u.sim.data.ctrl[0].detach().float().cpu().numpy().copy())
C = np.stack(hist)                      # (20, nu_global)
loc = C[:, np.asarray(ctrl_ids)]        # entity-local, aligned with anames

def cls(n):
  if "xml_motor_unused" not in n: return "real"
  return "unused_finger" if "finger" in n else "unused_body"

rep = {}
for c in ("real", "unused_body", "unused_finger"):
  idx = [i for i, n in enumerate(anames) if cls(n) == c]
  if not idx: continue
  sub = loc[:, idx]
  rep[c] = dict(
    n=len(idx),
    absmax=float(np.abs(sub).max()),
    varies_over_time=bool((sub.std(axis=0) > 1e-9).any()),
    n_dofs_varying=int((sub.std(axis=0) > 1e-9).sum()),
    mean_first=[round(float(v), 5) for v in sub[0][:4]],
  )
  print(f"{c:15s} n={len(idx):3d} |ctrl|max={rep[c]['absmax']:10.4f} "
        f"varies={rep[c]['varies_over_time']} ({rep[c]['n_dofs_varying']} of {len(idx)}) "
        f"first4={rep[c]['mean_first']}")

# Does effort actually move anything? compare the two writes' destinations.
print(f"\naction_order == joint_order : {action_order == jnames}")
d = [i for i, v in enumerate(tids) if v != i]
print(f"action->joint permutation is identity: {len(d)==0}  ({len(d)} of {len(tids)} differ)")
if d:
  print(f"  first 6 differing action slots: {d[:6]}")
  for i in d[:4]:
    print(f"    action[{i}] -> joint index {tids[i]} = {jnames[tids[i]]}   (joint_order[{i}]={jnames[i]})")

Path(OUT).write_text(json.dumps(dict(
  action_order=action_order, joint_order=jnames, action_to_joint_index=tids,
  actuator_order_entity_local=anames, ctrl_report=rep), indent=1))
print(f"\nwrote {OUT}")
env.close()
