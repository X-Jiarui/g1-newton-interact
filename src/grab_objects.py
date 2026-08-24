"""Load GRAB objects straight into Newton as SDF colliders -- no MJCF, no MuJoCo geom conversion.

The object path in mjlab went mesh -> convex decomposition -> a stack of boxes or a hull, because the
contact model needed convex pieces. Newton takes the mesh itself: `Mesh.create_from_file` reads the
STL, `build_sdf` samples it into a signed distance field, and the shape is added with `force_sdf` so
collision uses the field rather than a convex approximation of it.

Nothing here goes through MJCF. The robot still does -- it is authored that way -- but the object,
which is the thing whose shape was being thrown away, does not.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

import numpy as np

GRAB_MESHES = os.path.expanduser("~/jiarui/scaled_grab_wuji_all_o70/meshes")


def grab_mesh_path(name: str) -> str:
  """Resolve a GRAB object name to its mesh, preferring the full-resolution one.

  The dataset carries simplified variants alongside the originals (`*_cir160.stl`). Those are already
  convex for several objects -- the stapler's side groove is gone in its cir160 form -- so the
  full-resolution mesh is the one worth handing to an SDF.
  """
  for candidate in (f"{name}.stl", f"{name}_cir160.stl"):
    p = os.path.join(GRAB_MESHES, candidate)
    if os.path.exists(p):
      return p
  raise FileNotFoundError(f"no mesh for {name!r} in {GRAB_MESHES}")


def add_grab_object(builder, name: str, *, pos=(0.0, 0.0, 0.0), quat=(0.0, 0.0, 0.0, 1.0),
                    mass: float = 0.3, sdf_resolution: int = 128, hydroelastic: bool = False,
                    density: float | None = None, verbose: bool = True) -> tuple[int, int, Any]:
  """Add a GRAB object as a free body whose collider is its own SDF. Returns (body, shape, mesh)."""
  import newton
  import warp as wp

  path = grab_mesh_path(name)
  mesh = newton.Mesh.create_from_file(path)
  # The SDF is built before the shape is added: Newton's own hydroelastic example does the same,
  # because the field has to exist when the shape is registered.
  mesh.build_sdf(max_resolution=sdf_resolution)

  # Mesh-backed shapes reject cfg.sdf_* outright ("Build and attach an SDF on the mesh via
  # mesh.build_sdf()"): the field comes from the mesh, and only is_hydroelastic is a shape-level
  # choice. Primitives are the opposite -- they configure their SDF through the cfg.
  cfg = replace(builder.default_shape_cfg, is_hydroelastic=hydroelastic)
  if density is not None:
    cfg = replace(cfg, density=density)

  # add_body already creates the free joint. Adding another produced a second, parallel FREE joint
  # per object -- Newton warned about it and then dropped it as an unmappable loop closure, which
  # means the object silently carried redundant degrees of freedom.
  body = builder.add_body(xform=wp.transform(wp.vec3(*pos), wp.quat(*quat)), mass=mass, label=name)
  shape = builder.add_shape_mesh(body=body, mesh=mesh, cfg=cfg, label=f"{name}_sdf")

  if verbose:
    v = np.asarray(mesh.vertices)
    ext = v.max(axis=0) - v.min(axis=0)
    print(f"  {name:14s} {len(v):6d} verts  extents {np.round(ext, 3).tolist()} m  "
          f"sdf={mesh.sdf is not None} res={sdf_resolution}"
          f"{'  hydroelastic' if hydroelastic else ''}")
  return body, shape, mesh


def swap_collider_to_sdf(builder, mj_reference, body_name: str, stl_path: str, *,
                         resolution: int = 128, hydroelastic: bool = False,
                         verbose: bool = True) -> int:
  """Replace a body's imported collider with the object's real mesh, collided through an SDF.

  The scene comes from mjlab, where the object is a 4 cm analytic sphere -- and mjlab's own mesh path
  is no better for this purpose: it insists on the `*_cir160` collision mesh, which is the convex
  hull decimated to 68 vertices. For the stapler that hull is 1.7x the true volume and the side
  opening is gone entirely.

  So the body is kept -- with it every name, the entity lookup, the reference tracking and the reward
  wiring stay exactly as mjlab built them -- and only its collision geometry is swapped. The sphere
  is left in place as a visual but stripped of COLLIDE_SHAPES, so nothing collides twice.

  Body index mapping: Newton body i corresponds to MuJoCo body i+1 (MuJoCo body 0 is the world).
  Verified in tools/probes/probe_bodymap.py rather than assumed.
  """
  import mujoco
  import newton
  import numpy as np
  import warp as wp

  names = [mujoco.mj_id2name(mj_reference, mujoco.mjtObj.mjOBJ_BODY, i) or ""
           for i in range(mj_reference.nbody)]
  if body_name not in names:
    raise ValueError(f"{body_name!r} not in the reference model; have {names[-4:]}")
  newton_body = names.index(body_name) - 1

  sb = np.asarray(builder.shape_body)
  flags = np.asarray(builder.shape_flags)
  COLLIDE = int(newton.ShapeFlags.COLLIDE_SHAPES) | int(newton.ShapeFlags.COLLIDE_PARTICLES)
  existing = np.flatnonzero(sb == newton_body)
  for i in existing:
    builder.shape_flags[int(i)] = int(flags[int(i)]) & ~COLLIDE

  mesh = newton.Mesh.create_from_file(stl_path)
  mesh.build_sdf(max_resolution=resolution)
  from dataclasses import replace as _replace
  cfg = _replace(builder.default_shape_cfg, is_hydroelastic=hydroelastic)
  shape = builder.add_shape_mesh(body=newton_body, mesh=mesh, cfg=cfg,
                                 label=f"{body_name}_sdf")
  if verbose:
    v = np.asarray(mesh.vertices)
    print(f"[sdf] {body_name}: {len(existing)} imported collider(s) disabled, "
          f"replaced by {os.path.basename(stl_path)} ({len(v)} verts, sdf res {resolution})")
  return shape


def flag_hydroelastic(builder, mj_reference, body_names) -> int:
    """Mark every shape on the given bodies as hydroelastic, before the scene is replicated.

    Hydroelastic contact is defined between two hydroelastic surfaces. Flagging only the object and
    leaving the table and ground as plain rigid shapes produced a mixed pairing that injected a
    large impulse: the stapler stood itself upright around step 110 with nothing touching it, was
    ejected by step 135, and the state went partly non-finite in between.

    Primitives need no SDF -- newton/examples/robot/example_robot_panda_hydro.py notes that meshes
    require an explicit build_sdf while primitive SDFs come from the shape config -- so the table
    box and the ground plane only need the flag.
    """
    import numpy as np
    import mujoco
    import newton

    names = [mujoco.mj_id2name(mj_reference, mujoco.mjtObj.mjOBJ_BODY, i)
             for i in range(mj_reference.nbody)]
    sb = np.asarray(builder.shape_body)
    flagged = 0
    for want in body_names:
        if want not in names:
            raise RuntimeError(f"body {want!r} not in the reference model; "
                               f"present: {[n for n in names if n]}")
        newton_body = names.index(want) - 1        # MuJoCo body 0 is the world
        idx = np.flatnonzero(sb == newton_body)
        if len(idx) == 0:
            raise RuntimeError(f"body {want!r} carries no shape to flag hydroelastic")
        for i in idx:
            builder.shape_flags[int(i)] |= int(newton.ShapeFlags.HYDROELASTIC)
            flagged += 1
    return flagged
