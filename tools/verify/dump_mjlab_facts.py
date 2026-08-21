"""Dump the mjlab reference: the compiled MuJoCo model the successful checkpoint was trained in.

This is the baseline every Newton-side number is compared against, so the environment is built the
way play.py builds it, not the way the task defaults describe it. Two steps matter and both were
learned the hard way:

  * the agent config comes from the checkpoint's own params/agent.yaml -- the task default points
    tracker_ckpt at another project's GR00T checkpoint and simply fails to load
  * set_astra_body_dynamics(env_cfg) runs BEFORE the env is constructed, because it rewrites the
    robot's PD gains and body dynamics for the ASTRA base tracker. Dumping without it would record a
    different actuator model than the policy was trained against -- which is the exact class of
    silent mismatch this whole exercise exists to prevent.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import yaml as _yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default=os.path.expanduser(
  "~/sweep_ckpts_r2/OF_00_apple_eat_1_SPHERE/model_7310.pt"))
ap.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "docs"))
ARGS = ap.parse_args()

import mjlab.tasks  # noqa: F401,E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: E402
from mjlab.scripts.play import _apply_cfg_mapping  # noqa: E402

from model_facts import facts  # noqa: E402

TASK = "Mjlab-ResidualInteract-G1"
env_cfg = load_env_cfg(TASK, play=True)
env_cfg.scene.num_envs = 1

agent_cfg = load_rl_cfg(TASK)
cfg_path = Path(ARGS.checkpoint).parent / "params" / "agent.yaml"
if not cfg_path.exists():
  raise SystemExit(f"missing {cfg_path}")
_apply_cfg_mapping(agent_cfg, _yaml.unsafe_load(cfg_path.open()))
print(f"agent cfg   : {cfg_path}")
print(f"base_tracker: {getattr(agent_cfg, 'base_tracker_kind', None)}")

if str(getattr(agent_cfg, "base_tracker_kind", "")).strip().lower() == "astra_onnx":
  from mjlab.tasks.residual_interact.env_cfgs import set_astra_body_dynamics
  set_astra_body_dynamics(env_cfg)
  print("applied     : set_astra_body_dynamics")

env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode=None)
u = env.unwrapped
m = u.sim.mj_model

os.makedirs(ARGS.outdir, exist_ok=True)
f = facts(m)
f["_source"] = dict(side="mjlab", task=TASK, checkpoint=ARGS.checkpoint)

import mujoco  # noqa: E402
f["_versions"] = dict(mujoco=mujoco.__version__)
try:
  import mujoco_warp, warp  # noqa: E402
  f["_versions"].update(mujoco_warp=getattr(mujoco_warp, "__version__", "?"),
                        warp=warp.__version__)
except Exception:
  pass

out = Path(ARGS.outdir) / "mjlab_facts.json"
out.write_text(json.dumps(f, indent=1))
print(f"\nwrote {out}")

# The XML is a convenience for reading, not the comparison basis. mjlab keeps the spec it compiled;
# if this version does not expose one, the fact-sheet above is still complete.
spec = None
for attr in ("spec", "_spec"):
  spec = getattr(u.sim, attr, None)
  if spec is not None:
    break
if spec is not None:
  try:
    xml = Path(ARGS.outdir) / "mjlab_model.xml"
    xml.write_text(spec.to_xml())
    print(f"wrote {xml}")
  except Exception as e:
    print(f"spec.to_xml failed ({type(e).__name__}: {e}); fact-sheet is unaffected")
else:
  print("no MjSpec exposed by this mjlab Simulation; fact-sheet only")

s = f["sizes"]
print(f"\nnq={s['nq']} nv={s['nv']} nu={s['nu']} njnt={s['njnt']} "
      f"nbody={s['nbody']} ngeom={s['ngeom']}")

# The PD definition, summarised: if these are not all FIXED/AFFINE position servos, the migration's
# actuator story is more complicated than "copy kp and kv" and we need to know that now.
kinds = {}
for n, a in f["actuators"].items():
  kinds[(a["gaintype"], a["biastype"])] = kinds.get((a["gaintype"], a["biastype"]), 0) + 1
print(f"actuator (gaintype,biastype) histogram: {kinds}")
kps = sorted({round(a['kp'], 3) for a in f['actuators'].values()})
kvs = sorted({round(a['kv'], 3) for a in f['actuators'].values()})
print(f"distinct kp values ({len(kps)}): {kps[:12]}{' ...' if len(kps) > 12 else ''}")
print(f"distinct kv values ({len(kvs)}): {kvs[:12]}{' ...' if len(kvs) > 12 else ''}")

groups = {}
for g in f["geoms"].values():
  key = (g["group"], g["contype"], g["conaffinity"])
  groups[key] = groups.get(key, 0) + 1
print(f"geom (group,contype,conaffinity) histogram: {groups}")

env.close()
