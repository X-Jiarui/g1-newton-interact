#!/usr/bin/env python3
"""Turn every object mesh in a dataset into a validated SDF collider.

This is the flow that got the stapler and the mug working, written down so it can be run over a
whole dataset instead of rediscovered per object. Two stages:

  build     load the mesh, bake an SDF with the parameters that were measured to work, record
            geometry facts (vertex count, watertightness, AABB, voxel size, build time)
  validate  drop the object onto a table in a minimal Newton scene and measure how it settles

The build parameters are not defaults and not guesses. They come from a bisect against
newton/examples/robot/example_robot_panda_hydro.py:

  max_resolution=64, narrow_band_range=(-0.01, 0.01), margin=0.01

Newton's own defaults (band +-0.1, margin 0.05, res 128) give identical physics -- an object at
rest sits at 0.000 mm penetration either way -- but the sparse grid is allocated *per world*, and
at those defaults 2048 worlds cannot be allocated on a 32 GiB card at any contact budget.

The validation stage is the part worth keeping. An SDF that builds without error can still be
unusable: at resolution 16 the stapler builds fine and then rests 2.7 mm inside the table. The
acceptance criterion is what the object does, not whether the bake returned.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import replace

import numpy as np

SDF_RESOLUTION = 64          # floor; the real choice is TARGET_VOXEL_M below
SDF_RESOLUTION_MAX = 256
TARGET_VOXEL_M = 0.001       # 1 mm of surface detail, whatever the object's size
SDF_NARROW_BAND = (-0.01, 0.01)
SDF_MARGIN = 0.01
CONTACT_KH = 1e11
CONTACT_GAP = 0.01

# Anything a GRAB dataset directory carries that is not an object to be manipulated.
NON_OBJECT_STEMS = {"table"}
# Ablation copies of one object, kept in the dataset for collider experiments. They are built
# only with --include-variants, because otherwise a single apple contributes 18 rows.
VARIANT_MARKERS = ("_cir", "_col", "_ins", "_truehull", "_collision")


def resolution_for(longest_axis_m: float, target_voxel_m: float) -> int:
  """Pick the grid so the voxel is ~target, not so the *longest axis* gets a fixed count.

  max_resolution divides the longest AABB axis, so a fixed 64 gives a 22 cm knife a 3.8 mm voxel
  and its blade ends up one or two voxels thick -- measured: the knife rests 1.06 mm inside the
  table at 64 and 0.25 mm at 256; the flute 1.17 -> 0.20; the gamecontroller 0.76 -> 0.10. Simply
  raising the number for everything is wrong in the other direction: at 256 the stapler produced
  zero contacts, where at 64 and 128 it rests at -0.0005 mm. Scale with the object instead.
  """
  span = longest_axis_m + 2 * SDF_MARGIN
  res = int(round(span / max(target_voxel_m, 1e-6) / 8.0)) * 8   # build_sdf wants a multiple of 8
  return max(SDF_RESOLUTION, min(SDF_RESOLUTION_MAX, res))


def discover_meshes(mesh_dir: str, include_variants: bool) -> list[str]:
  out = []
  for name in sorted(os.listdir(mesh_dir)):
    if not name.endswith(".stl"):
      continue
    stem = name[:-4]
    if stem in NON_OBJECT_STEMS:
      continue
    if not include_variants and any(m in stem for m in VARIANT_MARKERS):
      continue
    out.append(os.path.join(mesh_dir, name))
  return out


def build_one(path: str, resolution: int | None, cache_dir: str | None,
              target_voxel_m: float = TARGET_VOXEL_M) -> dict:
  """Bake the SDF and report what the mesh actually is. Never raises; failures are recorded."""
  import newton

  rec: dict = {"mesh": os.path.basename(path), "path": path}
  try:
    t0 = time.time()
    mesh = newton.Mesh.create_from_file(path)
    v = np.asarray(mesh.vertices, dtype=np.float64)
    lo, hi = v.min(axis=0), v.max(axis=0)
    rec.update(
      verts=int(len(v)),
      extent_m=[round(float(x), 5) for x in (hi - lo)],
      longest_axis_m=round(float((hi - lo).max()), 5),
      watertight=bool(getattr(mesh, "is_watertight", None)) if hasattr(mesh, "is_watertight") else None,
    )
    res = resolution or resolution_for(rec["longest_axis_m"], target_voxel_m)
    rec["resolution"] = res
    kw = dict(max_resolution=res, narrow_band_range=SDF_NARROW_BAND, margin=SDF_MARGIN)
    if cache_dir:
      kw["cache_dir"] = cache_dir
    mesh.build_sdf(**kw)
    rec["build_s"] = round(time.time() - t0, 2)
    sdf = getattr(mesh, "sdf", None)
    if sdf is None:
      rec.update(ok=False, error="build_sdf returned but mesh.sdf is None")
      return rec
    # A mesh whose longest axis is much smaller than the margin is all margin and no object.
    rec["voxel_m"] = round(float((rec["longest_axis_m"] + 2 * SDF_MARGIN) / res), 6)
    rec["voxels_across_object"] = int(rec["longest_axis_m"] / rec["voxel_m"])
    rec["ok"] = True
  except Exception as e:  # recorded, not raised: one bad mesh must not stop the dataset
    rec.update(ok=False, error=f"{type(e).__name__}: {e}")
  return rec


def drop_test(path: str, resolution: int, seconds: float = 1.0,
              table_half: float = 0.5) -> dict:
  """Drop the object on a table and report how it settles.

  The scene is deliberately minimal -- one object, one box table, no robot -- so the number
  measures the object's collider and nothing else. Both sides are hydroelastic, which is the
  configuration under which an SDF collider is worth having at all.
  """
  import newton
  import warp as wp
  from newton.geometry import HydroelasticSDF

  rec = {"mesh": os.path.basename(path)}
  try:
    cfg = newton.ModelBuilder.ShapeConfig(kh=CONTACT_KH, gap=CONTACT_GAP,
                                          mu_torsional=0.0, mu_rolling=0.0)
    cfg_h = replace(cfg, is_hydroelastic=True)
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.default_shape_cfg = cfg

    # Half a metre of table each way. The first version used 0.25 m and rejected the watch,
    # the toothpaste and the eyeglasses for drifting 74-171 mm -- they had simply rolled off, which
    # is correct physics for a watch and says nothing about its SDF.
    table = newton.Mesh.create_box(table_half, table_half, 0.05, duplicate_vertices=True,
                                   compute_normals=False, compute_uvs=False, compute_inertia=True)
    table.build_sdf(max_resolution=resolution, narrow_band_range=SDF_NARROW_BAND, margin=SDF_MARGIN)
    top = float(np.asarray(table.vertices)[:, 2].max())
    b.add_shape_mesh(body=-1, mesh=table,
                     xform=wp.transform(wp.vec3(0.0, 0.0, 0.10 - top), wp.quat_identity()),
                     cfg=cfg_h)

    obj = newton.Mesh.create_from_file(path)
    obj.build_sdf(max_resolution=resolution, narrow_band_range=SDF_NARROW_BAND, margin=SDF_MARGIN)
    verts = np.asarray(obj.vertices, dtype=np.float64)
    lowest = float(verts[:, 2].min())
    drop = 0.003
    body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.10 - lowest + drop),
                                         wp.quat_identity()), label="object")
    b.add_shape_mesh(body=body, mesh=obj, cfg=cfg_h)
    b.add_joint_free(body)

    model = b.finalize()
    s0, s1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    control = model.control()
    pipe = newton.CollisionPipeline(model, reduce_contacts=True, broad_phase="nxn",
                                    sdf_hydroelastic_config=HydroelasticSDF.Config())
    contacts = pipe.contacts()
    solver = newton.solvers.SolverMuJoCo(
      model, use_mujoco_contacts=False, disable_sensors=True, solver="newton",
      integrator="implicitfast", cone="elliptic", njmax=500, nconmax=500,
      iterations=15, ls_iterations=100, impratio=1000.0)

    dt = 1.0 / 600.0
    steps = int(round(seconds / dt))
    for i in range(steps):
      pipe.collide(s0, contacts)
      solver.step(s0, s1, control, contacts, dt)
      s0, s1 = s1, s0

    bq = wp.to_torch(s0.body_q).cpu().numpy()[0]
    x, y, zq, w = bq[3:7]
    R = np.array([[1 - 2 * (y * y + zq * zq), 2 * (x * y - zq * w), 2 * (x * zq + y * w)],
                  [2 * (x * y + zq * w), 1 - 2 * (x * x + zq * zq), 2 * (y * zq - x * w)],
                  [2 * (x * zq - y * w), 2 * (y * zq + x * w), 1 - 2 * (x * x + y * y)]])
    world_z = (verts @ R.T + bq[0:3])[:, 2]
    rec.update(
      penetration_mm=round(float(1000 * (0.10 - world_z.min())), 4),
      xy_drift_mm=round(float(1000 * math.hypot(bq[0], bq[1])), 3),
      contacts=int(wp.to_torch(contacts.rigid_contact_count)[0]),
      mass_kg=round(float(wp.to_torch(model.body_mass).cpu().numpy()[0]), 5),
      finite=bool(np.isfinite(bq).all()),
    )
    # Acceptance. The stapler and the mug both land at |penetration| < 0.01 mm with 30+ contacts
    # and no drift; resolution 16 was rejected on this exact test at 2.7 mm.
    # Drift is reported, not judged: a watch rolls, and that is the object behaving correctly.
    # What must hold is that the collider is solid (it does not sink) and present (it is still in
    # contact at the end). "Left the table" is caught by contacts == 0.
    rec["left_table"] = bool(rec["contacts"] == 0)
    rec["accepted"] = bool(rec["finite"] and abs(rec["penetration_mm"]) < 0.5
                           and not rec["left_table"])
  except Exception as e:
    rec.update(accepted=False, error=f"{type(e).__name__}: {e}")
  return rec


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--mesh-dir", required=True)
  ap.add_argument("--out", required=True, help="manifest JSON to write")
  ap.add_argument("--resolution", type=int, default=0,
                  help="fixed grid for every object; 0 (default) picks per object from "
                       "--target-voxel")
  ap.add_argument("--target-voxel", type=float, default=TARGET_VOXEL_M,
                  help="metres of surface detail to resolve, per object")
  ap.add_argument("--table-half", type=float, default=0.5)
  ap.add_argument("--cache-dir", default=None, help="reuse baked SDFs across runs")
  ap.add_argument("--include-variants", action="store_true",
                  help="also process the *_cir/_col/_ins collider ablation copies")
  ap.add_argument("--no-validate", action="store_true", help="skip the drop test")
  ap.add_argument("--limit", type=int, default=0)
  a = ap.parse_args()

  meshes = discover_meshes(a.mesh_dir, a.include_variants)
  if a.limit:
    meshes = meshes[:a.limit]
  print(f"[sdf-pipeline] {len(meshes)} object mesh(es) in {a.mesh_dir}")
  print(f"[sdf-pipeline] resolution {a.resolution or 'per-object from %.1f mm voxels' % (1000 * a.target_voxel)}"
        f", narrow band {SDF_NARROW_BAND}, margin {SDF_MARGIN}, table half {a.table_half} m")

  rows = []
  for i, m in enumerate(meshes, 1):
    rec = build_one(m, a.resolution or None, a.cache_dir, a.target_voxel)
    if rec.get("ok") and not a.no_validate:
      rec.update({k: v for k, v in drop_test(m, rec["resolution"],
                                             table_half=a.table_half).items() if k != "mesh"})
    rows.append(rec)
    flag = "ok " if rec.get("ok") else "BUILD-FAIL"
    if rec.get("ok") and not a.no_validate:
      flag = "PASS" if rec.get("accepted") else "REJECT"
    print(f"  [{i:3d}/{len(meshes)}] {rec['mesh']:28s} {flag:10s} "
          f"verts={rec.get('verts', '?'):>7} "
          f"pen={rec.get('penetration_mm', float('nan')):8.4f}mm "
          f"res={rec.get('resolution', '?'):>4} "
          f"contacts={rec.get('contacts', '?'):>4} "
          f"build={rec.get('build_s', float('nan')):5.1f}s"
          + (f"  {rec.get('error')}" if rec.get("error") else ""))

  ok = [r for r in rows if r.get("ok")]
  passed = [r for r in rows if r.get("accepted")]
  summary = {
    "mesh_dir": a.mesh_dir, "resolution": a.resolution or "per-object",
    "target_voxel_m": a.target_voxel,
    "narrow_band": list(SDF_NARROW_BAND), "margin": SDF_MARGIN,
    "total": len(rows), "built": len(ok),
    "validated": None if a.no_validate else len(passed),
    "objects": rows,
  }
  os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
  with open(a.out, "w") as f:
    json.dump(summary, f, indent=2)
  print(f"\n[sdf-pipeline] built {len(ok)}/{len(rows)}"
        + ("" if a.no_validate else f", accepted {len(passed)}/{len(ok)}")
        + f"; manifest -> {a.out}")
  rejected = [r["mesh"] for r in rows if r.get("ok") and not r.get("accepted")]
  if rejected and not a.no_validate:
    print(f"[sdf-pipeline] rejected by the drop test: {rejected}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
