"""The definitive action -> joint -> actuator mapping, read out of a live mjlab environment.

Reading this off the source is not good enough. The compiled model already proves the two orders
differ -- the 69 real actuators are NOT in the same order as the 69 hinge joints -- so any port that
assumes "action i drives joint i drives ctrl i" is scrambled, and nothing raises: the policy runs and
every command lands on the wrong joint.

Three orders are recorded, plus the permutations between them:

    joint order      robot.joint_names            what qpos / observations are indexed by
    action order     the action term's joint ids  what the policy's 69 outputs mean
    actuator order   model actuator ids           what ctrl is indexed by

The Newton port has to reproduce the composition of these, not any single one.
"""

from __future__ import annotations

import json, os, sys
from dataclasses import asdict
from pathlib import Path

import numpy as np, yaml as _yaml

CKPT = os.path.expanduser("~/sweep_ckpts_r2/OF_00_apple_eat_1_SPHERE/model_7310.pt")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "docs/action_mapping.json")

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.scripts.play import _apply_cfg_mapping

TASK = "Mjlab-ResidualInteract-G1"
env_cfg = load_env_cfg(TASK, play=True); env_cfg.scene.num_envs = 1
agent_cfg = load_rl_cfg(TASK)
_apply_cfg_mapping(agent_cfg, _yaml.unsafe_load((Path(CKPT).parent / "params" / "agent.yaml").open()))
if str(getattr(agent_cfg, "base_tracker_kind", "")).strip().lower() == "astra_onnx":
  from mjlab.tasks.residual_interact.env_cfgs import set_astra_body_dynamics
  set_astra_body_dynamics(env_cfg)

env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode=None)
u = env.unwrapped
robot = u.scene["robot"]
import mujoco
m = u.sim.mj_model

joint_order = list(robot.joint_names)
act_names = list(robot.actuator_names)
print(f"robot.joint_names   : {len(joint_order)}")
print(f"robot.actuator_names: {len(act_names)}")

# ctrl_ids: entity-local actuator index -> global model actuator id
ctrl_ids = robot.indexing.ctrl_ids.detach().cpu().numpy().tolist()
ctrl_actuator_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, int(i)) for i in ctrl_ids]

# what joint does each of those actuators drive, in ctrl order
def act_target(gid):
  tid = int(m.actuator_trnid[gid][0])
  return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, tid)

ctrl_target_joints = [act_target(int(i)) for i in ctrl_ids]

print("\n--- action manager terms ---")
info = {}
for name, term in u.action_manager._terms.items():
  d = {k: v for k, v in vars(term).items()
       if isinstance(v, (int, float, str, bool, list, tuple))}
  ids = getattr(term, "_joint_ids", None)
  names = getattr(term, "_joint_names", None)
  print(f"  {name}: dim={getattr(term,'action_dim','?')} "
        f"joint_ids={type(ids).__name__} joint_names={type(names).__name__}")
  if ids is not None and not isinstance(ids, slice):
    ids = np.asarray(getattr(ids, "cpu", lambda: ids)().numpy() if hasattr(ids, "cpu") else ids).tolist()
  info[name] = dict(action_dim=int(getattr(term, "action_dim", -1)),
                    joint_ids=ids if not isinstance(ids, slice) else "slice(None)",
                    joint_names=list(names) if names is not None else None,
                    scalars=d)

# action order = joint names indexed by the term's joint_ids
action_order = None
for name, t in info.items():
  ids = t["joint_ids"]
  if ids == "slice(None)":
    action_order = joint_order[: t["action_dim"]]; break
  if isinstance(ids, list) and ids:
    action_order = [joint_order[i] for i in ids]; break

def perm(src, dst):
  """index into src for each element of dst, or None if the sets differ."""
  if src is None or dst is None or set(src) != set(dst):
    return None
  pos = {n: i for i, n in enumerate(src)}
  return [pos[n] for n in dst]

res = dict(
  joint_order=joint_order,
  action_order=action_order,
  ctrl_order_actuator_names=ctrl_actuator_names,
  ctrl_order_target_joints=ctrl_target_joints,
  action_terms=info,
  perm_joint_to_action=perm(joint_order, action_order),
  perm_joint_to_ctrltarget=perm(joint_order, ctrl_target_joints),
  perm_action_to_ctrltarget=perm(action_order, ctrl_target_joints),
)
Path(OUT).write_text(json.dumps(res, indent=1))

print(f"\naction_order resolved  : {action_order is not None} "
      f"({len(action_order) if action_order else 0})")
print(f"joint_order == action_order       : {joint_order == action_order}")
print(f"joint_order == ctrl_target_joints : {joint_order == ctrl_target_joints}")
print(f"action_order == ctrl_target_joints: {action_order == ctrl_target_joints}")
p = res["perm_joint_to_ctrltarget"]
if p is not None:
  ident = p == list(range(len(p)))
  print(f"perm joint->ctrl is identity: {ident}")
  if not ident:
    diff = [i for i, v in enumerate(p) if v != i]
    print(f"  {len(diff)} of {len(p)} positions differ; first 10 -> {diff[:10]}")
    for i in diff[:5]:
      print(f"    ctrl slot {i}: drives {ctrl_target_joints[i]}  (joint_order[{i}]={joint_order[i]})")
print(f"\nwrote {OUT}")
env.close()
