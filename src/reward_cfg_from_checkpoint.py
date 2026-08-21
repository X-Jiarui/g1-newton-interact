"""Apply a checkpoint's own reward weights to a freshly loaded env config.

`load_env_cfg` returns the task DEFAULT, where one of 37 reward terms carries weight. The checkpoint
that reached a working grasp trained with ten: multi_tip_surface 5.0, right_wrist_tracking 3.0,
object_trajectory_tracking / object_lift_hold / object_hard_lift 2.0, tracking / contact_duration /
stability 1.0, left_wrist_tracking / omnigrasp_style 0.5. Training against the default means training
with no grasping reward at all -- no lift term, no contact term, no fingertip-surface term -- which
is why the reward sat flat while every other curve looked healthy.

The weights are parsed textually rather than loaded: params/env.yaml carries Python objects that do
not resolve outside a built env (`cannot find OriginType in mjlab.viewer.viewer_config`), and only
names and weights are needed.
"""

from __future__ import annotations

import re
from pathlib import Path


def reward_weights_from_env_yaml(path: str | Path) -> dict[str, float]:
  """Read `rewards: <name>: weight:` pairs out of a checkpoint's env.yaml."""
  out: dict[str, float] = {}
  name = None
  in_rewards = False
  for line in Path(path).read_text().split("\n"):
    if re.match(r"^rewards:", line):
      in_rewards = True
      continue
    if in_rewards and re.match(r"^[a-z_]+:", line):
      break
    if not in_rewards:
      continue
    m = re.match(r"^  ([a-z0-9_]+):\s*$", line)
    if m:
      name = m.group(1)
      continue
    m = re.match(r"^    weight:\s*([-0-9.eE]+)", line)
    if m and name:
      out[name] = float(m.group(1))
  return out


def apply_reward_weights(cfg, weights: dict[str, float], *, verbose: bool = True) -> int:
  """Set the weights on cfg's reward terms. Returns how many were changed.

  Terms present in the checkpoint but absent from the current task raise: a silently dropped reward
  term is the failure this function exists to prevent.
  """
  rewards = cfg.rewards if isinstance(cfg.rewards, dict) else vars(cfg.rewards)
  terms = {k: v for k, v in rewards.items() if hasattr(v, "weight")}
  missing = [k for k, w in weights.items() if abs(w) > 1e-12 and k not in terms]
  if missing:
    raise KeyError(
      f"the checkpoint weights name reward terms this task does not define: {missing}. "
      "Training would silently drop them."
    )
  changed = []
  for k, w in weights.items():
    if k in terms and abs(float(terms[k].weight) - w) > 1e-12:
      terms[k].weight = w
      changed.append((k, w))
  if verbose:
    nz = {k: w for k, w in weights.items() if abs(w) > 1e-12}
    print(f"[reward-cfg] applied {len(changed)} weight change(s); "
          f"{len(nz)} terms now carry weight: "
          f"{dict(sorted(nz.items(), key=lambda t: -abs(t[1])))}")
  return len(changed)
