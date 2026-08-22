"""Give Newton's model the MJCF's sensors, by adding them to its MjSpec before it compiles.

Newton has no sensor support at all -- neither its MJCF importer nor its solver's spec construction
mentions them -- so a scene that defines 140 sensors arrives with nsensor=0. For this task that is
not cosmetic: 136 of them are contact sensors, and the reward terms that decide a grasp read them.
multi_tip_surface (weight 5.0), contact_duration (1.0) and object_hard_lift (2.0) were all exactly
zero for every step of training because the sensors they gate on did not exist.

Newton does build an MjSpec and compile it, so the sensors are added there: MuJoCo's own sensors,
compiled by MuJoCo's compiler, evaluated by mujoco_warp. Nothing about contact semantics is
reimplemented, which is the part that would be easy to get subtly and silently wrong.

The one translation needed is names: Newton renames bodies to a flattened path, so a sensor attached
to `robot/left_finger1_link4` has to be re-pointed at Newton's spelling of the same body.
"""

from __future__ import annotations

from typing import Any

import mujoco

_COPY_FIELDS = ("type", "objtype", "reftype", "intprm", "datatype",
                "needstage", "cutoff", "noise", "interval", "nsample", "interp", "delay")


def _name_map(dst_spec, src_names: list[str]) -> dict[str, str]:
  """Map source body names onto Newton's flattened ones by longest-suffix match.

  Longest wins because `..._left_hip_roll_joint` also ends with `_roll_joint`; a first or shortest
  match would silently attach a sensor to the wrong body.
  """
  dst = [b.name for b in dst_spec.bodies]
  flat = sorted(((s.replace("/", "_"), s) for s in src_names if s), key=lambda t: -len(t[0]))
  out: dict[str, str] = {}
  for src_flat, src in flat:
    for d in dst:
      if d == src or d.endswith(src_flat):
        out[src] = d
        break
  return out


def transplant_sensors(dst_spec, src_xml: str, *, verbose: bool = True) -> int:
  """Copy every sensor from `src_xml`'s spec into `dst_spec`. Call before dst_spec.compile()."""
  src = mujoco.MjSpec.from_file(src_xml)
  src_sensors = list(src.sensors)
  if not src_sensors:
    return 0

  body_names = [b.name for b in src.bodies]
  site_names = [s.name for b in src.bodies for s in b.sites]
  mapping = _name_map(dst_spec, body_names)
  dst_sites = {s.name for b in dst_spec.bodies for s in b.sites}
  dst_bodies = {b.name: b for b in dst_spec.bodies}

  added, skipped = 0, []
  for s in src_sensors:
    obj = s.objname
    ref = s.refname
    # sites are not renamed by Newton's importer the way bodies are; only remap what needs it
    if s.objtype == mujoco.mjtObj.mjOBJ_BODY:
      obj = mapping.get(s.objname, s.objname)
    if s.reftype == mujoco.mjtObj.mjOBJ_BODY and s.refname:
      ref = mapping.get(s.refname, s.refname)
    if s.objtype == mujoco.mjtObj.mjOBJ_SITE and s.objname not in dst_sites:
      # Newton's MJCF importer keeps bodies but not their <site> children, so an IMU sensor would
      # be dropped for want of its mounting frame. The site is pure geometry -- a named pose on a
      # body -- so it can be recreated on the matching body without touching the kinematic tree.
      if _clone_site(dst_spec, src, s.objname, mapping, dst_bodies):
        dst_sites.add(s.objname)   # cloned, so this sensor is added below like any other
      else:
        skipped.append((s.name, f"site {s.objname!r} absent and its body has no counterpart"))
        continue

    n = dst_spec.add_sensor()
    n.name = s.name
    n.objname = obj
    n.refname = ref
    for f in _COPY_FIELDS:
      try:
        setattr(n, f, getattr(s, f))
      except Exception:
        pass
    added += 1

  if verbose:
    from collections import Counter
    kinds = Counter(int(s.type) for s in src_sensors)
    print(f"[sensors] transplanted {added}/{len(src_sensors)} sensors into Newton's spec "
          f"(types {dict(kinds)})")
    if skipped:
      print(f"[sensors] skipped {len(skipped)}: {skipped[:4]}")
  return added


def _clone_site(dst_spec, src_spec, site_name: str, mapping: dict, dst_bodies: dict):
  """Recreate one src site on the destination body Newton imported for its parent. Returns False if
  that body has no counterpart, in which case the caller drops the sensor rather than guessing."""
  for b in src_spec.bodies:
    for st in b.sites:
      if st.name != site_name:
        continue
      tgt_name = mapping.get(b.name, b.name)
      tgt = dst_bodies.get(tgt_name)
      if tgt is None:
        return False
      new = tgt.add_site()
      new.name = st.name
      for f in ("pos", "quat", "size", "type", "group"):
        try:
          setattr(new, f, getattr(st, f))
        except (AttributeError, ValueError, TypeError):
          pass
      return True
  return False
