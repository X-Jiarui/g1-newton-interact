"""One independent entity per object, instead of one body whose geoms are swapped per world.

WHY NOT VariantEntityCfg

A merged variant body shares every geom index between objects, and only some model fields are
scattered per world. Measured on the mixed apple+stapler scene (tools/eval/audit_world_isolation.py):
13 fields are genuinely per-world, 9 are shared outright, and 7 more have a per-world SHAPE but a
stride of 0, so a per-world write silently lands on every world. On top of that the scatter loop in
`variants.py` only writes mesh slots -- non-mesh primitives keep the template variant's values in
every world. That last one put a 4 cm sphere inside the stapler in every stapler world, lifting it
off the table and tilting it 20 degrees, while the same run's apple was fine.

With one entity per object none of that applies. Each object has its own body, its own free joint and
its own geoms, so `geom_type`, `contype`, `priority`, `solimp` and friends are simply different
indices rather than one index that has to mean two things. The object's pose is qpos -- state, not
model -- which is per-world by construction and needs no scatter at all.

The cost is that every world carries every object. The ones a world does not use are parked far below
the floor, where they touch nothing.

SINGLE CLIP IS UNCHANGED. With one object this module returns the one entity's tensors directly, so
runs that do not use APPLE_EAT_PKL_MIX behave exactly as before.
"""

from __future__ import annotations

import torch

# Where an object goes in the worlds that are not using it.
#
# NOT below the floor. The terrain is an infinite plane at z=0, so parking at z=-5 buries the object
# 5 m inside it; the contact solver sees that penetration and launches it out. Measured: a parked
# object went -5.00 -> -3.62 -> -2.09 -> -0.43 -> +0.86 over four steps, decaying under the free
# joint's damping, and ended up back in the scene looking like something had teleported it.
#
# Park it sideways instead, resting ON the plane, far beyond env_spacing (6 m) so it cannot reach any
# world's robot or table. It settles immediately and stays put.
PARK_DX = 50.0
PARK_Z = 0.05

# The first object keeps the historical entity name so single-clip runs, their checkpoints and every
# existing config keep working untouched.
BASE_NAME = "apple"


def entity_names(n_objects: int) -> list[str]:
  """Scene entity name per object index."""
  return [BASE_NAME] + [f"{BASE_NAME}_c{i}" for i in range(1, int(n_objects))]


def sensor_name(obj_index: int, base: str = "hand_apple_contact") -> str:
  """Scene key of `base`'s sensor for object `obj_index`.

  env_cfgs._object_sensors names the first object's sensors without a suffix, so single-clip runs
  and their checkpoints keep the historical keys.
  """
  return base if obj_index == 0 else f"{base}_c{obj_index}"


def n_objects(env) -> int:
  """How many object ENTITIES the scene holds -- not how many clips the reference carries.

  mjlab authors one object per clip inside every environment and parks the unused ones. A host may
  instead give each environment a single world carrying only its own object: the Newton port groups
  environments by object so identical worlds can be replicated in one contiguous block, which is
  what makes a heterogeneous scene affordable. There the scene holds exactly ONE object entity and
  it is already this environment own object, so the pool must collapse to the single-entity path --
  active() hands back the real entity and write_root_state skips the parking loop.

  Returning n_clips on such a host is not merely wasteful. write_root_state would iterate the same
  entity n times, writing the real pose for the envs that own object i and the PARK pose for all the
  others, so the last pass teleports almost every environment object out of the scene.
  """
  from mjlab.tasks.apple_eat import mdp as amdp

  return amdp.object_entity_count()


def entities(env) -> list:
  """The object entities, in clip order."""
  return [env.scene[name] for name in entity_names(n_objects(env))]


def env_object_index(env) -> torch.Tensor:
  """Which object entity each env uses. Identical to the clip assignment by construction.

  Deliberately delegates to apple_eat.mdp._clip_id rather than recomputing `arange % n`: two places
  independently deriving the same round-robin is how an env ends up running clip A against object B
  with nothing raising.
  """
  from mjlab.tasks.apple_eat import mdp as amdp

  return amdp._clip_id(env)


def gather(env, attr: str) -> torch.Tensor:
  """Per-env view of `attr` on whichever object that env is using.

  `attr` is any tensor on the entity's `.data` whose first dimension is num_envs.
  """
  ents = entities(env)
  if len(ents) == 1:
    return getattr(ents[0].data, attr)
  idx = env_object_index(env)
  stacked = torch.stack([getattr(e.data, attr) for e in ents], dim=0)
  rows = torch.arange(stacked.shape[1], device=stacked.device)
  return stacked[idx.to(stacked.device), rows]


def write_root_state(env, state: torch.Tensor, env_ids: torch.Tensor | None) -> None:
  """Place each env's own object at `state[env]`, and park every other object for those envs.

  Parking matters: an unused object left at the reference pose would sit in the scene, collide with
  the table and the hand, and be picked up by any metric that reads it.
  """
  ents = entities(env)
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=state.device)
  env_ids = env_ids.to(state.device)
  if len(ents) == 1:
    ents[0].write_root_state_to_sim(state, env_ids=env_ids)
    return

  idx = env_object_index(env).to(state.device)[env_ids]
  parked = state.clone()
  parked[:, 0] = parked[:, 0] + PARK_DX
  parked[:, 2] = PARK_Z
  parked[:, 7:] = 0.0  # no velocity on a parked object
  for i, ent in enumerate(ents):
    mine = idx == i
    if mine.any():
      ent.write_root_state_to_sim(state[mine], env_ids=env_ids[mine])
    if (~mine).any():
      ent.write_root_state_to_sim(parked[~mine], env_ids=env_ids[~mine])


def gather_sensor(env, attr: str, base: str = "hand_apple_contact"):
  """Per-env view of a contact-sensor field, one sensor per object."""
  n = n_objects(env)
  sensors = [env.scene[sensor_name(i, base)] for i in range(n)]
  vals = []
  for s in sensors:
    v = getattr(s.data, attr, None)
    if v is None:
      return None
    vals.append(v)
  if n == 1:
    return vals[0]
  idx = env_object_index(env)
  stacked = torch.stack(vals, dim=0)
  rows = torch.arange(stacked.shape[1], device=stacked.device)
  return stacked[idx.to(stacked.device), rows]


##
# An entity-shaped view of "this env's object".
#
# Every reward, observation and termination reads the object the same way:
#     obj = env.scene["apple"]; ... obj.data.root_link_pos_w ...
# With one entity per object that line silently reads clip 0's object in every world. `active(env)`
# keeps the call shape and gathers per env instead, so the 29 call sites change by one token.
##


class _ActiveData:
  """`.data` of the active object, gathered per env."""

  def __init__(self, env, ents):
    object.__setattr__(self, "_env", env)
    object.__setattr__(self, "_ents", ents)

  def __getattr__(self, name):
    vals = [getattr(e.data, name) for e in self._ents]
    first = vals[0]
    if len(vals) == 1 or not torch.is_tensor(first):
      # Non-tensor fields (names, ids, counts) are structural and identical across entities,
      # because every object entity is built by the same spec function.
      return first
    idx = env_object_index(self._env)
    stacked = torch.stack(vals, dim=0)
    rows = torch.arange(stacked.shape[1], device=stacked.device)
    return stacked[idx.to(stacked.device), rows]


class _ActiveObject:
  """Entity-shaped proxy. Reads gather per env; writes raise rather than hit the wrong object."""

  def __init__(self, env):
    ents = entities(env)
    object.__setattr__(self, "_env", env)
    object.__setattr__(self, "_ents", ents)
    object.__setattr__(self, "data", _ActiveData(env, ents))

  def __getattr__(self, name):
    if name.startswith("write_"):
      raise AttributeError(
        f"{name} is not available on the active-object view: a write has to choose WHICH object "
        f"entity it targets. Use object_pool.write_root_state(env, state, env_ids), which places "
        f"each env's own object and parks the others."
      )
    # Structural attributes are identical across entities; take them from the first.
    return getattr(self._ents[0], name)


def active(env):
  """The object this env is using, as something that quacks like the old single entity."""
  ents = entities(env)
  if len(ents) == 1:
    return ents[0]  # single clip: hand back the real entity, so nothing changes at all
  return _ActiveObject(env)


class _ActiveSensorData:
  """`.data` of this env's own contact sensor, gathered per env."""

  def __init__(self, env, sensors):
    object.__setattr__(self, "_env", env)
    object.__setattr__(self, "_sensors", sensors)

  def __getattr__(self, name):
    vals = [getattr(s.data, name, None) for s in self._sensors]
    first = vals[0]
    if first is None or len(vals) == 1 or not torch.is_tensor(first):
      # A field the sensor did not populate, or structural metadata: identical across sensors,
      # because every pair is built from the same ContactSensorCfg template.
      return first
    if any(v is None for v in vals):
      return None
    idx = env_object_index(self._env)
    stacked = torch.stack(vals, dim=0)
    rows = torch.arange(stacked.shape[1], device=stacked.device)
    return stacked[idx.to(stacked.device), rows]


class _ActiveSensor:
  """ContactSensor-shaped proxy over one sensor per object."""

  def __init__(self, env, base):
    n = n_objects(env)
    sensors = [env.scene[sensor_name(i, base)] for i in range(n)]
    object.__setattr__(self, "_env", env)
    object.__setattr__(self, "_sensors", sensors)
    object.__setattr__(self, "data", _ActiveSensorData(env, sensors))

  def __getattr__(self, name):
    # primary_names, slot counts and the rest are structural and identical across the pairs.
    return getattr(self._sensors[0], name)


def active_sensor(env, base: str = "hand_apple_contact"):
  """The contact sensor for the object THIS env is using.

  Single clip returns the real sensor, so nothing changes for runs without APPLE_EAT_PKL_MIX.
  """
  if n_objects(env) <= 1:
    return env.scene[sensor_name(0, base)]
  return _ActiveSensor(env, base)
