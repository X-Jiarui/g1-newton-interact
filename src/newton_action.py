"""Sonic53Action with the dead torque law cut out, for the Newton backend.

`set_joint_effort_target` is a no-op on the Newton bridge, and deliberately so: measured in mjlab
with the trained policy, ctrl on every `xml_motor_unused_*` actuator stays exactly 0.0, so the
torque law reaches no actuator that can produce force. The robot is a position servo.

The base `apply_actions` still computes that torque on every substep -- it reads joint_pos and
joint_vel, evaluates a 53-DOF PD law, clamps it against the effort limits, and performs a host
synchronisation (`hard_hold_mask.any()`) to zero it for the environments still in the startup
hold. All of it is discarded. At four substeps per control step that is four wasted PD evaluations
and four pipeline stalls.

Rather than copy the 80-line method into this repo -- a copy would silently drift the day mjlab's
version changes -- this removes exactly the dead region from the base class's own source and
recompiles it. The region is matched literally, so if upstream edits any line of it the surgery
fails loudly instead of cutting the wrong thing.
"""

from __future__ import annotations

import inspect
import textwrap

# The dead region: everything between publishing the position target and the pelvis band. Matched
# literally against the base class source; any upstream edit here makes _build() raise.
_DEAD_REGION = """    q = self._entity.data.joint_pos[:, self._target_ids]
    qd = self._entity.data.joint_vel[:, self._target_ids]
    if bool(getattr(self, "_teacher_velocity_feedforward", False)):
      frame = _tracking_frame(self._env, ref["n_frames"])
      target_qd = ref["dof_vel"][frame, : NUM_BODY + NUM_HAND]
      effort = self._kp.unsqueeze(0) * (target - q) + self._kd.unsqueeze(0) * (
        target_qd - qd
      )
    else:
      effort = self._kp.unsqueeze(0) * (target - q) - self._kd.unsqueeze(0) * qd
    effort = effort.clamp(
      -self._effort_limit.unsqueeze(0), self._effort_limit.unsqueeze(0)
    )
    if hard_hold_mask.any():
      effort[hard_hold_mask] = 0.0
    self._entity.set_joint_effort_target(effort, joint_ids=self._target_ids)
"""


def build_effortless_apply_actions(base_cls):
  """Return a replacement `apply_actions` for `base_cls` with the dead torque law removed."""
  # Match against the raw source, whose body sits at the four-space indent _DEAD_REGION is written
  # at; dedent only afterwards, so the method compiles at module level.
  src = inspect.getsource(base_cls.apply_actions)
  dead = _DEAD_REGION
  if dead not in src:
    raise RuntimeError(
      f"{base_cls.__name__}.apply_actions no longer contains the torque block this fork removes. "
      "Upstream changed it -- re-read the method and update _DEAD_REGION rather than skipping "
      "the surgery, because the block may no longer be dead.")

  trimmed = textwrap.dedent(src.replace(dead, ""))
  module = inspect.getmodule(base_cls)
  ns = dict(vars(module))          # the method closes over module-level names like NUM_BODY
  exec(compile(trimmed, f"<{base_cls.__name__}.apply_actions:effortless>", "exec"), ns)
  return ns["apply_actions"]


def make_effortless(base_cls):
  """Subclass `base_cls` with the torque law removed from apply_actions."""
  return type(f"{base_cls.__name__}Effortless", (base_cls,),
              {"apply_actions": build_effortless_apply_actions(base_cls),
               "__doc__": __doc__})
