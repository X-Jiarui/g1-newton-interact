"""Export the exact MJCF mjlab compiles, then prove the export round-trips.

Newton's importer needs a real MJCF on disk, and it has to be the one that carries the ASTRA PD
gains -- so the env is built with the same recipe as the fact dump (checkpoint agent.yaml overlay,
then set_astra_body_dynamics before construction).

The round-trip check is the point of this script, not a bonus. mjlab compiles an MjSpec it builds in
memory; `scene.write()` serialises that spec to XML. If serialise -> reload -> compile does not
reproduce the same model, then every later disagreement with Newton would be contaminated by an
export bug, and I would be debugging the wrong side.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import yaml as _yaml

sys.path.insert(0, os.path.expanduser("~/projects/g1-newton-interact/src"))

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default=os.path.expanduser(
  "~/sweep_ckpts_r2/OF_00_apple_eat_1_SPHERE/model_7310.pt"))
ap.add_argument("--outdir", default=os.path.expanduser("~/projects/g1-newton-interact/assets/mjlab_scene"))
A_ = ap.parse_args()

import sys, os
sys.path.insert(0, os.path.expanduser("~/projects/g1-newton-interact/src"))
try:
  import mjw_compat as _c; _c.apply()
except Exception:
  pass
import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.scripts.play import _apply_cfg_mapping
import mujoco
from model_facts import facts

TASK = "Mjlab-ResidualInteract-G1"
env_cfg = load_env_cfg(TASK, play=True); env_cfg.scene.num_envs = 1
agent_cfg = load_rl_cfg(TASK)
_apply_cfg_mapping(agent_cfg, _yaml.unsafe_load((Path(A_.checkpoint).parent/"params"/"agent.yaml").open()))
if str(getattr(agent_cfg, "base_tracker_kind", "")).strip().lower() == "astra_onnx":
  from mjlab.tasks.residual_interact.env_cfgs import set_astra_body_dynamics
  set_astra_body_dynamics(env_cfg)
  print("applied set_astra_body_dynamics")

env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode=None)
u = env.unwrapped

out = Path(A_.outdir)
out.parent.mkdir(parents=True, exist_ok=True)
u.scene.write(out)
xml = out / "scene.xml"
print(f"exported {xml}  ({xml.stat().st_size / 1e6:.2f} MB)")
n_assets = len(list((out / "assets").glob("*"))) if (out / "assets").exists() else 0
print(f"assets: {n_assets}")

# Round-trip: reload the exported XML and compile it with the same mujoco that built the reference.
m2 = mujoco.MjModel.from_xml_path(str(xml))
f2 = facts(m2)
f2["_source"] = dict(side="mjlab_exported_xml", xml=str(xml))
f2["_versions"] = dict(mujoco=mujoco.__version__)
rt = Path(os.path.expanduser("~/projects/g1-newton-interact/docs/mjlab_exported_facts.json"))
rt.write_text(json.dumps(f2, indent=1))
print(f"wrote {rt}")
s = f2["sizes"]
print(f"reloaded: nq={s['nq']} nv={s['nv']} nu={s['nu']} njnt={s['njnt']} ngeom={s['ngeom']}")
env.close()
