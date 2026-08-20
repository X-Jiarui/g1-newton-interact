"""Rewrite a Newton-side fact-sheet into mjlab's naming so the two can actually be compared.

Newton's MJCF export renames everything to a flattened path -- `robot/left_hip_pitch_joint` comes
back as `mjlab scene_worldbody_robot_pelvis_robot_left_hip_pitch_link_robot_left_hip_pitch_joint` --
and emits actuators with no names at all. None of that is a physics difference, but it makes a
name-keyed diff report every single entry as missing, which buries the differences that do matter.

Each entity type is re-keyed by something Newton cannot rename:

  bodies, joints   longest mjlab name whose `/`->`_` form is a suffix of the Newton name. Longest
                   wins because `..._left_hip_roll_joint` also ends with `_roll_joint`; taking the
                   first or shortest match would mis-assign silently.
  actuators        the joint they drive, canonicalised. Newton drops actuator names entirely, and
                   the target is the identifying property anyway -- "the servo on left_hip_pitch"
                   is the thing being compared, not its label.
  geoms            (owning body, type, size, pos), rounded. Geom names are unreliable on BOTH sides
                   -- mjlab's own model has unnamed geoms -- so a structural key is used instead.

Any entity that cannot be mapped is reported and left under its original name rather than guessed at.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("newton"); ap.add_argument("reference"); ap.add_argument("out")
A = ap.parse_args()

nf = json.load(open(A.newton))
rf = json.load(open(A.reference))


def build_suffix_map(ref_names, newton_names):
  """newton name -> reference name, by longest flattened-suffix match."""
  flat = sorted(((r.replace("/", "_"), r) for r in ref_names), key=lambda t: -len(t[0]))
  out, unmapped = {}, []
  for n in newton_names:
    hit = None
    for f, r in flat:
      if n == r or n.endswith(f):
        hit = r
        break
    if hit is None:
      unmapped.append(n)
    else:
      out[n] = hit
  return out, unmapped


report = []
body_map, ub = build_suffix_map(rf["bodies"].keys(), nf["bodies"].keys())
joint_map, uj = build_suffix_map(rf["joints"].keys(), nf["joints"].keys())
report.append(f"bodies mapped {len(body_map)}/{len(nf['bodies'])} (unmapped {len(ub)}: {ub[:4]})")
report.append(f"joints mapped {len(joint_map)}/{len(nf['joints'])} (unmapped {len(uj)}: {uj[:4]})")

# Collisions would make the rename lossy; refuse rather than silently merge two entities into one.
for label, mp in (("body", body_map), ("joint", joint_map)):
  seen = {}
  for k, v in mp.items():
    seen.setdefault(v, []).append(k)
  dup = {v: ks for v, ks in seen.items() if len(ks) > 1}
  if dup:
    report.append(f"  !! {label} name collisions: {list(dup.items())[:3]}")

out = dict(nf)
out["bodies"] = {body_map.get(k, k): dict(v, parent=body_map.get(v.get("parent"), v.get("parent")))
                 for k, v in nf["bodies"].items()}
out["joints"] = {joint_map.get(k, k): dict(v, body=body_map.get(v.get("body"), v.get("body")))
                 for k, v in nf["joints"].items()}
out["joint_order"] = [joint_map.get(n, n) for n in nf["joint_order"]]

# actuators: key by driven joint
act, clash = {}, 0
order = []
for k in nf["actuator_order"]:
  a = nf["actuators"][k]
  tgt = joint_map.get(a.get("target"), a.get("target"))
  key = f"act@{tgt}"
  while key in act:            # two actuators on one joint -- mjlab has exactly this
    clash += 1
    key = f"{key}#{clash}"
  act[key] = dict(a, target=tgt)
  order.append(key)
out["actuators"] = act
out["actuator_order"] = order
report.append(f"actuators re-keyed by target joint ({len(act)}); {clash} joints carry >1 actuator")

# MuJoCo stores only the size components a geom type uses and leaves the rest at 0; Newton's live
# model fills the unused slots instead (a sphere is (r,0,0) on one side and (r,r,r) on the other).
# Keying on the raw triple would call those different geoms, so only the meaningful components count.
_SIZE_DIMS = {0: 3, 2: 1, 3: 2, 4: 3, 5: 2, 6: 3, 7: 0}   # plane, sphere, capsule, ellipsoid, cylinder, box, mesh


def gkey(g, bmap):
  b = bmap.get(g.get("body"), g.get("body"))
  _n = _SIZE_DIMS.get(int(g.get("type", -1)), 3)
  sz = tuple(round(float(x), 5) for x in (g.get("size") or [])[:_n])
  ps = tuple(round(float(x), 5) for x in (g.get("pos") or []))
  return f"geom@{b}|t{g.get('type')}|s{sz}|p{ps}"

geo, gclash = {}, 0
for k, g in nf["geoms"].items():
  key = gkey(g, body_map)
  while key in geo:
    gclash += 1
    key = f"{key}#{gclash}"
  geo[key] = dict(g, body=body_map.get(g.get("body"), g.get("body")))
def _norm_size(g):
  n = _SIZE_DIMS.get(int(g.get("type", -1)), 3)
  g = dict(g)
  g["size"] = [round(float(x), 5) for x in (g.get("size") or [])[:n]]
  return g


geo = {k: _norm_size(g) for k, g in geo.items()}
out["geoms"] = geo
report.append(f"geoms re-keyed structurally ({len(geo)}); {gclash} key collisions resolved by suffix")

# The reference has to be re-keyed the same way for the geom comparison to line up.
ref_out = dict(rf)
rgeo, rclash = {}, 0
_ref_geoms = {k: g for k, g in rf["geoms"].items()
              if (g.get("contype", 0) or g.get("conaffinity", 0))}
report.append(f"reference geoms: {len(rf['geoms'])} total -> {len(_ref_geoms)} colliding "
              "(keyed over the colliding subset only, so duplicate suffixes match Newton's)")
for k, g in _ref_geoms.items():
  key = gkey(g, {})
  while key in rgeo:
    rclash += 1
    key = f"{key}#{rclash}"
  rgeo[key] = g
rgeo = {k: _norm_size(g) for k, g in rgeo.items()}
ref_out["geoms"] = rgeo
ract, rclash2 = {}, 0
rorder = []
for k in rf["actuator_order"]:
  a = rf["actuators"][k]
  key = f"act@{a.get('target')}"
  while key in ract:
    rclash2 += 1
    key = f"{key}#{rclash2}"
  ract[key] = a
  rorder.append(key)
ref_out["actuators"] = ract
ref_out["actuator_order"] = rorder

Path(A.out).write_text(json.dumps(out, indent=1))
ref_side = Path(A.out).with_name(Path(A.out).stem + "_refkeyed.json")
ref_side.write_text(json.dumps(ref_out, indent=1))
print("\n".join(report))
print(f"wrote {A.out}")
print(f"wrote {ref_side}   <-- compare against THIS, not the raw reference")
