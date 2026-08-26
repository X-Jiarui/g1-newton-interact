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


def apply(root: Path) -> int:
  edits = [
    (root / "mjlab/tasks/apple_eat/mdp.py", [(OLD_FN, NEW_FN), (OLD_CALL, NEW_CALL),
                                             (OLD_INIT, NEW_INIT)]),
    (root / "mjlab/tasks/residual_interact/mdp.py", [(OLD_RI, NEW_RI)]),
  ]
  changed = 0
  for path, subs in edits:
    if not path.exists():
      raise SystemExit(f"not an mjlab source tree: {path} is missing")
    text = path.read_text()
    for old, new in subs:
      if new in text:
        continue
      if old not in text:
        raise SystemExit(
          f"{path}: cannot find the code this patch replaces, and the patched form is not there "
          "either -- mjlab has moved on and this patch needs rewriting rather than forcing")
      text = text.replace(old, new, 1)
      changed += 1
    path.write_text(text)
  return changed


if __name__ == "__main__":
  root = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/mjlab-run/src")
  n = apply(root)
  print(f"[patch_mjlab] {'applied ' + str(n) + ' edit(s)' if n else 'already applied'} under {root}")
