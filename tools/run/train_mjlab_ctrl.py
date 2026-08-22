"""The control: identical setup, mjlab's simulator instead of Newton's.

Everything is shared with train_newton -- the same clip, the same checkpoint-derived agent config,
the same ten reward weights, the same runner. Only the environment class differs. If mjlab's tracking
distance falls where Newton's sits at 14 cm, the gap is the port; if mjlab is also stuck, the gap is
the task and no amount of Newton debugging will move it.

mjlab's own train.py is bypassed on purpose: its CLI resolves the tracker from the task default,
which points at a checkpoint that does not exist here.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml as _yaml

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--num-envs", type=int, default=128)
ap.add_argument("--sensor-probe", type=int, default=0,
                help="mjlab-side twin of train_newton.py --sensor-probe, for parity comparison")
ap.add_argument("--iterations", type=int, default=400)
ap.add_argument("--reference-pkl", required=True)
ap.add_argument("--agent-cfg-from", default=os.path.expanduser(
  "~/sweep_ckpts_r2/OF_00_apple_eat_1_SPHERE/model_7310.pt"))
ap.add_argument("--run-name", default="MJLAB_CTRL")
ap.add_argument("--log-root", default=os.path.join(HERE, "logs/rsl_rl"))
ap.add_argument("--seed", type=int, default=42)
A = ap.parse_args()

os.environ["APPLE_EAT_PKL"] = A.reference_pkl
sys.path.insert(0, os.path.join(HERE, "src"))
import mjw_compat; mjw_compat.apply()

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper, MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.scripts.play import _apply_cfg_mapping
from reward_cfg_from_checkpoint import reward_weights_from_env_yaml, apply_reward_weights

TASK = "Mjlab-ResidualInteract-G1"
torch.manual_seed(A.seed); np.random.seed(A.seed)

cfg = load_env_cfg(TASK, play=False); cfg.scene.num_envs = A.num_envs
agent_cfg = load_rl_cfg(TASK)
p = Path(A.agent_cfg_from).parent / "params" / "agent.yaml"
_apply_cfg_mapping(agent_cfg, _yaml.unsafe_load(p.open()))
env_yaml = Path(A.agent_cfg_from).parent / "params" / "env.yaml"
apply_reward_weights(cfg, reward_weights_from_env_yaml(env_yaml))

_s = cfg.actions.get("sonic_action") if isinstance(cfg.actions, dict) else cfg.actions.sonic_action
_s.tracking_start_assist_gain = 0.0
_s.tracking_start_assist_steps = 0
if str(getattr(agent_cfg, "base_tracker_kind", "")).strip().lower() == "astra_onnx":
  from mjlab.tasks.residual_interact.env_cfgs import set_astra_body_dynamics
  set_astra_body_dynamics(cfg)

env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
u = env.unwrapped
if str(getattr(agent_cfg, "base_tracker_kind", "")).strip().lower() == "astra_onnx":
  from mjlab.tasks.residual_interact.env_cfgs import install_astra_body_pd, install_object_variant_sizes
  install_astra_body_pd(u); install_object_variant_sizes(u)

if os.environ.get("PEN_PROBE"):
  import numpy as _np, torch as _t, warp as _wp, mujoco as _mj
  _u = env.unwrapped
  _obj = _u.scene["apple"]
  _m = _u.sim.mj_model
  # mjlab's object is the 4cm sphere placeholder, so its lowest point is centre minus the radius
  _rad = None
  for _g in range(_m.ngeom):
    _n = _mj.mj_id2name(_m, _mj.mjtObj.mjOBJ_GEOM, _g) or ""
    if _n.endswith("apple_geom"):
      _rad = float(_m.geom_size[_g][0]); break
  env.reset()
  _act = _t.zeros(_u.num_envs, _u.action_manager.total_action_dim, device="cuda:0")
  _mp = _wp.to_torch(_u.sim.wp_data.mocap_pos)[0].detach().cpu().numpy()
  _top = float(_mp[-1][2]) + 0.02
  print("PEN mjlab sphere radius=%.4f table_top=%.5f" % (_rad, _top))
  _done = 0
  for _step in (0, 5, 20, 60, 150):
    while _done < _step:
      env.step(_act); _done += 1
    _z = float(_obj.data.root_link_pos_w[0, 2])
    print("PEN step=%-4d obj_z=%.5f obj_lowest=%.5f penetration=%+.2f mm"
          % (_step, _z, _z - _rad, 1000.0 * (_top - (_z - _rad))))
  raise SystemExit(0)

if A.sensor_probe:
  import numpy as _np, torch as _t, warp as _wp
  _u = env.unwrapped if hasattr(env, "unwrapped") else env
  _mdl = _u.sim.wp_model
  _dat = _u.sim.wp_data
  print(f"[mjlab-ctrl] sensors: nsensor={int(_mdl.nsensor)} "
        f"contact={int((_wp.to_torch(_mdl.sensor_type)==42).sum())} "
        f"nsensordata={int(_mdl.nsensordata)}")
  env.reset()
  _peak = _np.zeros(int(_mdl.nsensordata))
  _n = _u.action_manager.total_action_dim
  for _ in range(A.sensor_probe):
    env.step(_t.zeros(_u.num_envs, _n, device="cuda:0"))
    _sd = _wp.to_torch(_dat.sensordata).detach().cpu().numpy()
    _peak = _np.maximum(_peak, _np.abs(_sd).max(axis=0))
  _ty = _wp.to_torch(_mdl.sensor_type).cpu().numpy()
  _ad = _wp.to_torch(_mdl.sensor_adr).cpu().numpy()
  _dm = _wp.to_torch(_mdl.sensor_dim).cpu().numpy()
  _cmask = _np.zeros(len(_peak), bool)
  for _t_, _a_, _d_ in zip(_ty, _ad, _dm):
    if int(_t_) == 42:
      _cmask[int(_a_):int(_a_) + int(_d_)] = True
  _cp = _peak[_cmask]
  print(f"PROBE_CONTACT slots={_cmask.sum()} nonzero={int((_cp>1e-9).sum())} max={_cp.max():.4f}")
  print(f"PROBE steps={A.sensor_probe} nonzero_slots={int((_peak>1e-9).sum())}/{len(_peak)} "
        f"max={_peak.max():.4f}")
  raise SystemExit(0)

wrapped = RslRlVecEnvWrapper(env)
log_dir = Path(A.log_root) / "g1_residual_interact" / A.run_name
log_dir.mkdir(parents=True, exist_ok=True)
runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(wrapped, asdict(agent_cfg),
                                                        log_dir=str(log_dir), device="cuda:0")
print(f"mjlab control: {A.num_envs} envs, {A.iterations} iterations -> {log_dir}")
runner.learn(num_learning_iterations=A.iterations, init_at_random_ep_len=True)
