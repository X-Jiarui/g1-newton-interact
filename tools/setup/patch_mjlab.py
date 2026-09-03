#!/usr/bin/env python3
"""Make mjlab's scene anchor use the SAME clip assignment as everything else.

mjlab computed the per-environment clip twice: `_clip_id` returned it, and `clip_frame0_rows`
recomputed it as `arange(num_envs) % n_clips` on the assumption that the two always agree. This
port breaks that assumption on purpose -- `newton_vec_env` injects `env._reference_clip_id` so
environments sharing an object can be replicated in one contiguous block, which is what makes a
heterogeneous scene affordable to author.

With the two maps disagreeing, the reference, the object reset and the terminations all followed
the injected map while the every-step table write followed the round-robin one. Seven of every
eight environments were handed another clip's table height, so their object spawned over empty
space and fell. Measured on an 8-clip mix: each clip's environments held all 8 table heights
instead of one; the object dropped 121.7mm in free fall, crossed the 0.12m `og_object_far`
threshold on step 8 (sqrt(2*0.1217/9.81) = 0.157s = 7.9 steps at 50Hz), terminated and reset into
the same fall. Mean episode length was 8.0 against 95-127 for the same clip trained alone; after
this patch the same mix runs 104-116.

Single-clip training never saw it: n_clips == 1 makes both maps all-zero.

Idempotent -- running it twice reports "already applied" rather than stacking.
"""

import sys
from pathlib import Path

OLD_FN = '''def clip_frame0_rows(ref: dict, num_envs: int, device) -> torch.Tensor:
  """Global row of frame 0 of each environment's own clip; zeros without a mix.

  Same assignment as _clip_id (arange % n_clips), so the scene anchor and the reference agree.
  """
  n_clips = int(ref.get("n_clips", 1))
  idx = torch.arange(num_envs, device=device, dtype=torch.long)
  if n_clips <= 1:
    return torch.zeros_like(idx)
  return (idx % n_clips) * int(ref["n_frames"])'''

NEW_FN = '''def clip_frame0_rows(ref: dict, num_envs: int, device, env=None) -> torch.Tensor:
  """Global row of frame 0 of each environment's own clip; zeros without a mix.

  The clip assignment MUST come from _clip_id, not be recomputed here. This used to hardcode
  `arange % n_clips` on the assumption that _clip_id always returns that. A host that injects its
  own `env._reference_clip_id` -- the Newton port groups environments by object so identical
  worlds can be replicated in one block -- breaks the assumption silently: the reference, the
  object reset and the terminations all follow the injected map while the every-step table write
  followed the round-robin one, so 7 of every 8 environments got another clip's table height and
  their object fell through empty space. Pass `env` wherever one is available.
  """
  n_clips = int(ref.get("n_clips", 1))
  if n_clips <= 1:
    return torch.zeros(num_envs, device=device, dtype=torch.long)
  if env is not None:
    cid = _clip_id(env)
    if cid.shape[0] != num_envs:
      raise RuntimeError(
        f"_clip_id gave {cid.shape[0]} entries for {num_envs} environments; the scene anchor and "
        "the reference would disagree")
    return cid.to(device=device, dtype=torch.long) * int(ref["n_frames"])
  idx = torch.arange(num_envs, device=device, dtype=torch.long)
  return (idx % n_clips) * int(ref["n_frames"])'''

OLD_CALL = '''        clip_frame0_rows(ref, self._env.num_envs, self._env.scene.env_origins.device),'''
NEW_CALL = '''        clip_frame0_rows(ref, self._env.num_envs, self._env.scene.env_origins.device,
                         env=self._env),'''

OLD_INIT = '''def _initial_cuboid_scene_poses(
  ref: dict,
  env_origins: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  return _cuboid_scene_poses_at_frame(
    ref, env_origins, clip_frame0_rows(ref, env_origins.shape[0], env_origins.device)
  )'''
NEW_INIT = '''def _initial_cuboid_scene_poses(
  ref: dict,
  env_origins: torch.Tensor,
  env=None,
) -> tuple[torch.Tensor, torch.Tensor]:
  return _cuboid_scene_poses_at_frame(
    ref, env_origins,
    clip_frame0_rows(ref, env_origins.shape[0], env_origins.device, env=env),
  )'''

OLD_RI = '''def _initial_cuboid_scene_poses(ref: dict, env_origins: torch.Tensor):
  return apple_mdp._initial_cuboid_scene_poses(ref, env_origins)'''
NEW_RI = '''def _initial_cuboid_scene_poses(ref: dict, env_origins: torch.Tensor, env=None):
  return apple_mdp._initial_cuboid_scene_poses(ref, env_origins, env=env)'''



OLD_NOBJ = """def n_objects(env) -> int:
  from mjlab.tasks.apple_eat import mdp as amdp

  return int(amdp._ref(str(env.device)).get("n_clips", 1))"""

NEW_NOBJ = '''def n_objects(env) -> int:
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

  return amdp.object_entity_count()'''


# --- one object entity per world -------------------------------------------------------------
# mjlab decides how many object ENTITIES to author from the number of mixed clips: every
# environment gets one object per clip and parks the ones it is not using. That is the right
# layout for a host that builds a single world template and clones it.
#
# This port does not. It groups environments by object and `replicate`s each group, so an
# environment carries exactly ONE object -- its own. Omnigrasp does the same thing
# (`create_actor(env_ptr, self._target_assets[self.env_to_obj_name[env_id]], ...)`), so this is the
# ordinary layout for the task family, not an eccentricity of this port.
#
# Under that layout the cfg-time fan-out is not merely wasteful, it does not bind: the scene has no
# `apple_c1`, and no `hand_apple_contact_c1` sensor for it either. `APPLE_OBJECT_PER_WORLD=1`
# declares the host's layout so the fan-out collapses to one entity while the reference keeps its
# real `n_clips` -- the clip map, the per-clip gates and the terminations are unaffected.
OLD_OEC = """def mix_clip_count() -> int:
  return len(_mix_paths())"""

NEW_OEC = """def mix_clip_count() -> int:
  return len(_mix_paths())


def object_entity_count() -> int:
  '''How many object entities the scene config should author.

  One per clip for a host that clones a single world template and parks the unused objects; ONE
  when the host authors a separate world per object and hands each environment only its own
  (`APPLE_OBJECT_PER_WORLD=1`). Do not use this for anything but scene authoring -- the number of
  CLIPS is still mix_clip_count(), and the reference, the gates and the terminations all key off
  that.
  '''
  import os as _os

  if _os.environ.get("APPLE_OBJECT_PER_WORLD", "").strip() in ("1", "true", "yes", "on"):
    return 1
  return mix_clip_count()"""

OLD_FANOUT = """mdp.apple_mdp.mix_clip_count()"""
NEW_FANOUT = """mdp.apple_mdp.object_entity_count()"""

OLD_TCM = '    mean_d, has_tips, cf = _tip_cf_distance(env, near_threshold=near_threshold)\n    active = (cf >= 0) & has_tips & (frame >= (cf + int(grace_frames)))'

NEW_TCM = '    # TIP_CF_MISS_ENABLE=0 turns this off without touching the termination list, so a run with it\n    # disabled keeps every other termination index identical and the two remain directly comparable.\n    # Measured reason for the switch: this kill takes 99% of all episodes and `time_out` never fires,\n    # while the round predating it (R14) led on approach progress on all four clips.\n    if os.environ.get("TIP_CF_MISS_ENABLE", "1").strip() in ("0", "off", "false"):\n        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)\n\n    mean_d, has_tips, cf = _tip_cf_distance(env, near_threshold=near_threshold)\n    active = (cf >= 0) & has_tips & (frame >= (cf + int(grace_frames)))'

def apply(root: Path) -> int:
  edits = [
    (root / "mjlab/tasks/apple_eat/mdp.py", [(OLD_FN, NEW_FN), (OLD_CALL, NEW_CALL),
                                             (OLD_INIT, NEW_INIT)]),
    (root / "mjlab/tasks/residual_interact/mdp.py", [(OLD_RI, NEW_RI)]),
    (root / "mjlab/tasks/apple_eat/object_pool.py", [(OLD_NOBJ, NEW_NOBJ)]),
    (root / "mjlab/tasks/apple_eat/mdp.py", [(OLD_OEC, NEW_OEC)]),
    (root / "mjlab/tasks/residual_interact/staged_mdp.py", [(OLD_TCM, NEW_TCM)]),
    (root / "mjlab/tasks/residual_interact/env_cfgs.py", [(OLD_FANOUT, NEW_FANOUT, -1)]),
  ]
  changed = 0
  for path, subs in edits:
    if not path.exists():
      raise SystemExit(f"not an mjlab source tree: {path} is missing")
    text = path.read_text()
    for sub in subs:
      old, new = sub[0], sub[1]
      count = sub[2] if len(sub) > 2 else 1
      if new in text:
        continue
      if old not in text:
        raise SystemExit(
          f"{path}: cannot find the code this patch replaces, and the patched form is not there "
          "either -- mjlab has moved on and this patch needs rewriting rather than forcing")
      text = text.replace(old, new, count)
      changed += 1
    path.write_text(text)
  return changed


if __name__ == "__main__":
  root = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/mjlab-run/src")
  n = apply(root)
  print(f"[patch_mjlab] {'applied ' + str(n) + ' edit(s)' if n else 'already applied'} under {root}")
