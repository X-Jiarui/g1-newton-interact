"""A vectorised RL environment that steps in Newton and reuses mjlab's managers.

Everything except the physics comes from mjlab: the observation terms, the action term, and the
reward / termination / event managers all construct against `NewtonEnv` unchanged (37 reward terms,
9 termination terms). That is deliberate -- the point of the port is to change the simulator, not to
re-derive an MDP, and a reward that differs silently is indistinguishable from a policy that learns
badly.

Physics is Newton's: `ModelBuilder.replicate` for parallel worlds, `SolverMuJoCo.step` at dt=0.005,
with the two spec repairs from `newton_simple_fix` (free-joint damping, compressed mass-matrix
layout) applied to Newton's own model.

The step order follows mjlab's `ManagerBasedRlEnv.step` exactly, including applying the action on
every physics substep rather than once per control step -- `Sonic53Action` re-applies the startup
hold, the table pose and the reference object tracking on each call, so calling it once per control
step runs all of that at a quarter rate.
"""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
import torch
import warp as wp


@wp.kernel
def _sync_body_q_kernel(xpos: wp.array2d(dtype=wp.vec3),
                        xquat: wp.array2d(dtype=wp.quat),
                        world: wp.array(dtype=wp.int32),
                        mjc_body: wp.array(dtype=wp.int32),
                        newton_body: wp.array(dtype=wp.int32),
                        body_q: wp.array(dtype=wp.transform)):
  """MuJoCo pose -> Newton State.body_q, w-first quaternion to w-last.

  A warp kernel rather than torch indexing because this runs inside the CUDA graph capture that
  `--cuda-graph` performs. The torch version made capture fail outright ("operation would make
  the legacy stream depend on a capturing blocking stream"), which silently cost the whole graph
  speedup on the native path.
  """
  i = wp.tid()
  w = world[i]
  b = mjc_body[i]
  p = xpos[w, b]
  q = xquat[w, b]
  body_q[newton_body[i]] = wp.transform(p, wp.quat(q[1], q[2], q[3], q[0]))


def mujoco_legal_shape_pairs(solver, pairs):
  """Keep only the shape pairs MuJoCo itself would collide.

  Newton's CollisionPipeline does not import MJCF's collision filtering, so with
  use_mujoco_contacts=False the robot collides with itself everywhere MuJoCo forbids it. Measured
  on this scene: 1566 contacts per step of which essentially all were adjacent links --
  left_wrist_yaw <-> left_palm (86), right_wrist_yaw <-> right_palm (85), finger3_link4 <->
  finger4_link4 (57), elbow <-> wrist_pitch (28). Every one of them carries kh=1e11, so the hand
  was being blown apart from the inside and whatever it held was kicked away.

  The rules are MuJoCo's own, in mj_collideGeoms order: welded bodies never collide, parent and
  child never collide while mjDSBL_FILTERPARENT is off, <exclude> pairs never collide, and the two
  geoms must pass the contype/conaffinity mask test. Explicit <pair> entries bypass all of it.
  """
  import mujoco
  import numpy as np
  import warp as wp

  if solver.newton_shape_to_mjc_geom is None:
    solver._create_inverse_shape_mapping()
  s2g = wp.to_torch(solver.newton_shape_to_mjc_geom).cpu().numpy()
  m = solver.mj_model
  gb, weld, parent = m.geom_bodyid, m.body_weldid, m.body_parentid
  contype, conaff = m.geom_contype, m.geom_conaffinity
  filter_parent = not bool(m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
  excluded = set()
  for k in range(m.nexclude):
    sig = int(m.exclude_signature[k])
    b1, b2 = sig >> 16, sig & 0xFFFF
    excluded.add((b1, b2)); excluded.add((b2, b1))
  explicit = {(int(m.pair_geom1[k]), int(m.pair_geom2[k])) for k in range(m.npair)}
  explicit |= {(b, a) for a, b in explicit}

  kept, dropped, unmapped = [], 0, 0
  for a, b in pairs:
    ga, gbx = int(s2g[a]), int(s2g[b])
    if ga < 0 or gbx < 0:
      # Shapes we added ourselves after the MuJoCo conversion have no geom; never drop those.
      unmapped += 1; kept.append([a, b]); continue
    if (ga, gbx) in explicit:
      kept.append([a, b]); continue
    b1, b2 = int(gb[ga]), int(gb[gbx])
    w1, w2 = int(weld[b1]), int(weld[b2])
    if w1 == w2:
      dropped += 1; continue
    if filter_parent and w1 != 0 and w2 != 0 and (
        int(weld[parent[w1]]) == w2 or int(weld[parent[w2]]) == w1):
      dropped += 1; continue
    if (b1, b2) in excluded:
      dropped += 1; continue
    if not ((int(contype[ga]) & int(conaff[gbx])) or (int(contype[gbx]) & int(conaff[ga]))):
      dropped += 1; continue
    kept.append([a, b])
  print(f"[newton-env] MuJoCo collision filtering: {len(pairs)} pairs -> {len(kept)} "
        f"({dropped} dropped as welded/parent-child/excluded/masked, {unmapped} kept unmapped)")
  return kept


class NewtonVecEnv:
  """Duck-types enough of mjlab's ManagerBasedRlEnv for `RslRlVecEnvWrapper`."""

  def __init__(self, cfg, xml: str, num_envs: int, device: str = "cuda:0",
               njmax: int = 2048, nconmax: int = 256, object_entity: str = "apple",
               sdf_object_stl: str | None = None, sdf_resolution: int = 64,
               sdf_hydroelastic: bool = False, native_contacts: bool = False,
               table_sdf_resolution: int | None = None,
               hydro_object_table: bool = True,
               convex_hull_robot: bool = True,
               hydro_kh: float | None = None, hydro_grid_size: int | None = None,
               broad_phase: str = "nxn",
               viser_port: int | None = None,
               render_every: int = 4, solver_kwargs: dict | None = None,
               cuda_graph: bool = False, effortless_action: bool = False,
               table_under_object: bool = False, object_solref: str | None = None,
               dump_qpos: str | None = None, dump_steps: int = 600,
               newton_video: str | None = None, video_size: str = "960x720",
               object_collider_view: bool = False,
               video_steps: int = 500, video_cam: str | None = None) -> None:
    import mujoco
    import newton
    from newton.solvers import SolverMuJoCo
    from newton_simple_fix import capture_spec, restore_simple_bodies, restore_freejoint_damping
    from newton_sensors import transplant_sensors
    from newton_bridge import NewtonEnv

    self.cfg = cfg
    self.device = device
    self.num_envs = int(num_envs)
    self.render_mode = None
    self.extras: dict[str, Any] = {}

    self.physics_dt = float(cfg.sim.mujoco.timestep) if hasattr(cfg.sim, "mujoco") else 0.005
    self.decimation = int(cfg.decimation)
    self.step_dt = self.physics_dt * self.decimation
    self.max_episode_length = int(math.ceil(float(cfg.episode_length_s) / self.step_dt))

    # --- Newton model: one authored scene, replicated into parallel worlds -----
    scene = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(scene)
    scene.default_shape_cfg.gap = 0.0          # Newton's default of 0.1 needs 10cm of penetration
    if native_contacts:
      # Hydroelastic needs a contact margin: `gap` is what the SDF narrow band is built around, and
      # with gap=0 the pair never registers a contact at all -- measured, the hydroelastic counter
      # stayed at 0 for every step while the object fell straight through the table. The official
      # panda_hydro example uses gap=0.01 with a +/-1cm narrow band at resolution 64.
      # Zero for everything; swap_collider_to_sdf gives the object and the table their own 0.01.
      # See the note there: a scene-wide gap turned every knuckle pair inside the hand into a
      # contact candidate and made 74% of the per-world contacts hand-against-itself.
      scene.default_shape_cfg.gap = 0.0
    if hydro_kh is not None:
      # pressure = -kh * signed_depth, and the two sides combine in series
      # ((k_a*k_b)/(k_a+k_b)), so raising one alone leaves the softer side in control. Newton's
      # default is 1e10; the official panda_hydro grasping example uses 1e11.
      scene.default_shape_cfg.kh = float(hydro_kh)
      print(f"[newton-env] hydroelastic stiffness kh = {hydro_kh:.0e} on both contact sides")
    scene.add_mjcf(xml, collapse_fixed_joints=False, parse_mujoco_options=True)

    # Swap the object's collider before replicating, so every world gets it. mjlab authors the object
    # as a 4 cm sphere and its own mesh path uses the *_cir160 convex hull instead of the real shape.
    self._ref_mj_pre = mujoco.MjModel.from_xml_path(xml)
    if sdf_object_stl:
      from grab_objects import swap_collider_to_sdf
      swap_collider_to_sdf(scene, self._ref_mj_pre, f"{object_entity}/{object_entity}",
                           sdf_object_stl, resolution=sdf_resolution,
                           hydroelastic=((sdf_hydroelastic or native_contacts) and hydro_object_table))

    world = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(world)
    world.default_shape_cfg.gap = 0.0
    # Hydroelastic contact is pairwise: narrow_phase.py routes a pair to the SDF pipeline only when
    # BOTH shapes carry ShapeFlags.HYDROELASTIC, and otherwise falls through to the rigid path.
    # With only the object flagged and use_mujoco_contacts=False, the object-table pair had no
    # working contact at all -- filmed: the stapler sank into the table, stood up on its end and was
    # ejected, and the state went NaN around step 110.
    #
    # Boxes need no SDF build; primitive shapes are configured through the flag alone, per
    # newton/examples/robot/example_robot_panda_hydro.py.
    if native_contacts:
      # The table needs a mesh collider with an SDF, not just the flag: hydroelastic requires
      # texture SDF data, which a box primitive does not carry. Same route the official
      # panda_hydro example takes for its own table.
      _table_stl = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "assets/meshes/table_box.stl")
      if not os.path.exists(_table_stl):
        raise RuntimeError(f"missing {_table_stl}; generate it from the table's half-extents first")
      # The table is a flat box that only ever meets the object on its top face, so its SDF
      # carries no shape worth resolving -- but it does help set the tessellation of the contact
      # patch, and object/table is half of every contact in the scene (30-37 of 48-79 per world).
      # The object keeps its own resolution; only the table's is dialled down.
      _tres = int(table_sdf_resolution or sdf_resolution)
      swap_collider_to_sdf(scene, self._ref_mj_pre, "table/table", _table_stl,
                           resolution=_tres, hydroelastic=hydro_object_table)
      if _tres != sdf_resolution:
        print(f"[newton-env] table SDF resolution {_tres}, object stays at {sdf_resolution}")
      _n = sum(1 for f in scene.shape_flags if f & int(newton.ShapeFlags.HYDROELASTIC))
      print(f"[newton-env] hydroelastic shapes before replicate: {_n} "
            f"(object + table; a pair routes to SDF only when both sides are flagged)")

      # Convex-hull every colliding mesh that is NOT hydroelastic, exactly as
      # newton/examples/robot/example_robot_panda_hydro.py does for its non-finger shapes. This
      # scene carries 65 such meshes per world -- 48 in the Wuji hand alone, up to 25662 verts on
      # a single torso shape, 150k verts in total -- and Newton narrow-phases them as full meshes.
      #
      # This is parity with the baseline, not a loss of fidelity: MuJoCo already convexifies every
      # mesh geom for collision (its convex-hull graph is what `graphadr` indexes), so the
      # MuJoCo-contact path we are trying to match has been colliding hulls all along. The object
      # and the table keep their real geometry, which is the whole point of the SDF path.
      _H = int(newton.ShapeFlags.HYDROELASTIC)
      _C = int(newton.ShapeFlags.COLLIDE_SHAPES)
      # Exclude the object and the table by NAME, not by the hydroelastic flag. Keying off the
      # flag meant that turning hydroelastic off (--rigid-object-table) silently swept them into
      # the hull pass: the count went 65 -> 67 and the stapler's collider became a 64-vertex hull,
      # coarser than the 68-vertex cir160 hull this whole path exists to get away from. The real
      # mesh survived only as a visual. Verified by shape_source vertex counts, not by the log
      # line -- which cheerfully claimed the object kept its real geometry either way.
      _keep_real = ("_sdf",)
      _to_hull = [i for i in range(len(scene.shape_type))
                  if int(scene.shape_type[i]) == int(newton.GeoType.MESH)
                  and int(scene.shape_flags[i]) & _C
                  and not (int(scene.shape_flags[i]) & _H)
                  and not any(k in (scene.shape_label[i] or "").lower() for k in _keep_real)]
      if _to_hull and convex_hull_robot:
        _done = scene.approximate_meshes(method="convex_hull", shape_indices=_to_hull,
                                         keep_visual_shapes=True)
        _kept = [scene.shape_label[i].split("/")[-1] for i in range(len(scene.shape_type))
                 if int(scene.shape_type[i]) == int(newton.GeoType.MESH)
                 and int(scene.shape_flags[i]) & _C and i not in _to_hull]
        print(f"[newton-env] convex-hulled {len(_done)} of {len(_to_hull)} robot mesh collider(s); "
              f"kept as real meshes: {_kept}")

    world.replicate(scene, world_count=self.num_envs)
    self.nmodel = world.finalize()

    # Newton drops <sensor> entirely, and 136 of this scene's sensors are the contact sensors the
    # grasp rewards gate on -- multi_tip_surface (weight 5.0), contact_duration (1.0) and
    # object_hard_lift (2.0) read exactly zero without them, for every step of every run. They are
    # added to Newton's own spec just before it compiles, so MuJoCo evaluates its own sensors.
    import mujoco as _mj
    _orig_compile = _mj.MjSpec.compile
    _xml_for_sensors = xml

    def _compile_with_sensors(spec_self, *a, **k):
      try:
        transplant_sensors(spec_self, _xml_for_sensors, verbose=True)
      except Exception as e:
        print(f"[sensors] transplant failed ({type(e).__name__}: {e}); "
              "contact-gated rewards will read zero")
      return _orig_compile(spec_self, *a, **k)

    _mj.MjSpec.compile = _compile_with_sensors
    try:
      with capture_spec() as cap:
      # update_data_interval=0 keeps mjw_data authoritative, so the direct joint/root writes the
      # action term performs during the startup hold are not overwritten from Newton's State.
        # SDF collision has its own solver parameters (sdf_iterations / sdf_initpoints) that
        # default to None; they are passed through here so they can be set for mesh colliders.
        if native_contacts and nconmax <= 512:
          # mjlab's SimulationCfg budget (nconmax 256) is sized for MuJoCo's own narrow phase,
          # which emits a handful of points per pair. With use_mujoco_contacts=False every contact
          # instead comes from Newton's CollisionPipeline -- measured 1486..1542 per step for this
          # scene -- and everything past naconmax is silently dropped. The hydroelastic object-table
          # contacts are generated last, so they were exactly the ones truncated: 46..56 contacts
          # with correct +z normals, max|force| == 0, and the object in textbook free fall
          # (per-step drop increment a constant 3.9 mm = g*dt^2) straight through the table.
          # Per world, and sized for the per-world count AFTER the gap fix: 48-79 contacts, of
          # which 30-37 are the object on the table. The first version of this line used 4096 /
          # 16384, which was right for the 1500 contacts a scene-wide gap produced at one env and
          # OOMed on launch at 2048 worlds -- these are per-world budgets, not totals.
          nconmax, njmax = 512, max(njmax, 2048)
          print(f"[newton-env] native contacts: nconmax -> {nconmax}, njmax -> {njmax} "
                f"(mjlab's 256 is sized for MuJoCo's own narrow phase)")
        _kw = dict(enable_multiccd=True, update_data_interval=0, njmax=njmax, nconmax=nconmax)
        # Newton's own SDF hydroelastic contact, following
        # newton/examples/robot/example_robot_panda_hydro.py: the CollisionPipeline computes the
        # contacts and SolverMuJoCo integrates them, instead of MuJoCo doing its own collision.
        # This is what makes `--sdf-object` mean something: building the SDF alone was never
        # enough, because nothing consumed it.
        self.collision_pipeline = None
        self.contacts = None
        if native_contacts:
          from newton import CollisionPipeline
          from newton.geometry import HydroelasticSDF
          _kw.update(use_mujoco_contacts=False, solver="newton", integrator="implicitfast",
                     cone="elliptic", impratio=1000.0)
        # Caller overrides go LAST. They used to be merged before this block, which silently
        # discarded any attempt to change cone or impratio on the native path -- the very knobs
        # most worth testing, since elliptic/1000 was copied from the hydroelastic example and we
        # no longer run hydroelastic contact.
        _kw.update(solver_kwargs or {})
        self.solver = SolverMuJoCo(self.nmodel, **_kw)
        if native_contacts:
          # The collision-pair table is built by add_mjcf from the ORIGINAL shapes. The SDF
          # colliders are added afterwards, so no pair contains them and the broad phase never
          # tests them -- measured: the object shape appeared in zero contacts and fell straight
          # through the table, while the hydroelastic counter stayed at 0. MuJoCo's own collision
          # path is unaffected because it works from the converted MuJoCo model, which is why this
          # only shows up with use_mujoco_contacts=False.
          # Needed by both branches and by the body-pose sync below, so it is computed here
          # rather than inside the explicit-only pair building.
          _flags_np = wp.to_torch(self.nmodel.shape_flags).cpu().numpy()
          _hyd = [int(i) for i in range(len(_flags_np))
                  if int(_flags_np[i]) & int(newton.ShapeFlags.HYDROELASTIC)]
          # Newton's default of 1e6 mesh-triangle pairs is a global buffer, and this scene
          # overflowed it every few steps -- "Triangle pair buffer overflowed 2177620 > 1000000"
          # for the stapler, 1025759 for the mug. Everything past the cap is dropped, which means
          # missed contacts in exactly the interaction being trained. A pair is a vec3i, so 4 M
          # costs 48 MB.
          _max_tri = 4_000_000
          # grid_size is the hydroelastic working grid and it is charged per step regardless of
          # how many contacts there actually are: profiled at 2048 env, collide() was 234.7 ms of
          # a 253.9 ms substep while solver.step() was 19.2 ms, for 42 contacts per world.
          _hydro_cfg = (HydroelasticSDF.Config(grid_size=int(hydro_grid_size))
                        if hydro_grid_size else HydroelasticSDF.Config())
          # "explicit" does no broad-phase pruning at all: it narrow-phases every listed pair
          # every call. This scene lists 4513 pairs per world -- 9.2 M tests at 2048 worlds --
          # and profiled at 123-235 ms per collide() against 19 ms for the whole solver step,
          # with only 42-83 contacts per world to show for it. The official example gets away
          # with "explicit" because its scene has about twenty shapes; ours has 232, of which
          # 162 are meshes. "sap" prunes by AABB first and takes its pairs from the model's own
          # contact-pair table, which already contains the object/table pair.
          if broad_phase == "explicit":
            import warp as _wp
            _probe = CollisionPipeline(self.nmodel, reduce_contacts=True, broad_phase="explicit",
                                       sdf_hydroelastic_config=HydroelasticSDF.Config())
            _pairs = _wp.to_torch(_probe.shape_pairs_filtered).cpu().numpy().tolist()
            # Per world. The first version paired every hydroelastic shape with every other one
            # across the whole replicated model: at 2048 worlds that is 4096 shapes, 8.4 M pairs,
            # nearly all of them between different worlds, and it OOMed on launch. At one env it
            # added 0 pairs, which is why it survived every probe.
            _shapes_per_world = self.nmodel.shape_count // self.num_envs
            _flags_np = wp.to_torch(self.nmodel.shape_flags).cpu().numpy()
            _hyd = [int(i) for i in range(len(_flags_np))
                    if int(_flags_np[i]) & int(newton.ShapeFlags.HYDROELASTIC)]
            _by_world = {}
            for _i in _hyd:
              _by_world.setdefault(_i // _shapes_per_world, []).append(_i)
            _have = {(min(a, b), max(a, b)) for a, b in _pairs}
            _added = 0
            for _ws in _by_world.values():
              for _i in range(len(_ws)):
                for _j in range(_i + 1, len(_ws)):
                  _k = (min(_ws[_i], _ws[_j]), max(_ws[_i], _ws[_j]))
                  if _k not in _have:
                    _pairs.append([_k[0], _k[1]])
                    _have.add(_k)
                    _added += 1
            print(f"[newton-env] added {_added} hydroelastic collision pair(s) that the MJCF import "
                  f"could not know about ({len(_pairs)} pairs total)")
            _pairs = mujoco_legal_shape_pairs(self.solver, _pairs)
            self.collision_pipeline = CollisionPipeline(
              self.nmodel, reduce_contacts=True, broad_phase="explicit",
              shape_pairs_filtered=_wp.array(_pairs, dtype=_wp.vec2i, device=device),
              sdf_hydroelastic_config=_hydro_cfg,
              max_triangle_pairs=_max_tri)
          else:
            import warp as _wp
            self.collision_pipeline = CollisionPipeline(
              self.nmodel, reduce_contacts=True, broad_phase=broad_phase,
              sdf_hydroelastic_config=_hydro_cfg,
              max_triangle_pairs=_max_tri)
            print(f"[newton-env] broad phase '{broad_phase}': pairs come from the model's own "
                  f"contact-pair table, not the explicit list")
          self.contacts = self.collision_pipeline.contacts()
          _flags = wp.to_torch(self.nmodel.shape_flags)
          _n = int(((_flags & int(newton.ShapeFlags.HYDROELASTIC)) != 0).sum())
          print(f"[newton-env] Newton native contacts: {_n} hydroelastic shape(s) of "
                f"{self.nmodel.shape_count}; MuJoCo collision disabled")
          # SolverMuJoCo derives State.body_q from the joint coordinates by forward kinematics
          # (its own docstring says so), so a body with no joint is NEVER written back. mjlab
          # authors the table as a static body, so the collision pipeline saw it at the world
          # origin: measured SDF boxes were table z -0.070..0.084 against object z 0.770..0.922.
          # The two never overlapped, the hydroelastic broad phase returned 0 blocks for the pair
          # on every combination of gap (0.5/3/10 mm), kh (1e10/1e11/5e11) and broad phase, and
          # the object fell straight through to the floor. The pose has to be copied from MuJoCo.
          _m2n = wp.to_torch(self.solver.mjc_body_to_newton).cpu().numpy()
          _sb_all = wp.to_torch(self.nmodel.shape_body).cpu().numpy()
          _hyd_bodies = {int(_sb_all[i]) for i in _hyd if int(_sb_all[i]) >= 0}
          # ONLY the bodies welded to the world. Those are the ones forward kinematics never
          # writes, which is the whole reason the table was stuck at the origin. Syncing every
          # body instead overwrites the solver's own State output each substep with a pose that
          # is one integration step ahead of it (measured: 0.9 mm apart at rest, 12.7 mm while
          # falling), which breaks the contact matching the pipeline carries between calls.
          _weld = self.solver.mj_model.body_weldid
          _sw, _sbi, _sn = [], [], []
          for _wi in range(_m2n.shape[0]):
            for _bi in range(_m2n.shape[1]):
              _nid = int(_m2n[_wi, _bi])
              if _nid >= 0 and int(_weld[_bi]) == 0:
                _sw.append(_wi); _sbi.append(_bi); _sn.append(_nid)
          _n_hyd_mapped = sum(1 for _nid in _sn if _nid in _hyd_bodies)
          if _n_hyd_mapped == 0 and hydro_object_table:
            raise RuntimeError("no MuJoCo body maps to a hydroelastic shape; the pose sync would "
                               "be a no-op and every SDF contact would be missed")
          if not _sn:
            raise RuntimeError("no world-welded body to sync; the table would sit at the origin")
          self._sync_n = _sn
          self._sync_w_wp = wp.array(_sw, dtype=wp.int32, device=device)
          self._sync_b_wp = wp.array(_sbi, dtype=wp.int32, device=device)
          self._sync_n_wp = wp.array(_sn, dtype=wp.int32, device=device)
          print(f"[newton-env] syncing {len(_sn)} world-welded body pose(s) from MuJoCo every "
                f"substep ({_n_hyd_mapped} of them carry a hydroelastic shape); the solver's own "
                f"forward kinematics owns every other body")
    finally:
      _mj.MjSpec.compile = _orig_compile

    # Reported every run because its absence is silent: with nsensor=0 the contact-gated reward
    # terms return exactly 0.0 forever instead of erroring, which cost 3576 wasted iterations once.
    # mujoco_warp defaults contact_sensor_maxmatch to 64 and mjlab plumbs it through
    # cfg.sim.contact_sensor_maxmatch, which this env never touches because it builds its own
    # solver. Newton's narrow phase emits more points per pair than MuJoCo's, and the native path
    # logged "contact match overflow: please increase Option.contact_sensor_maxmatch to 98" on
    # every step -- silently truncating the very contact sensors the grasp rewards gate on.
    _mm = int(getattr(getattr(cfg, "sim", None), "contact_sensor_maxmatch", 0) or 0)
    _mm = max(_mm, 256 if native_contacts else 64)
    self.solver.mjw_model.opt.contact_sensor_maxmatch = _mm
    print(f"[newton-env] contact_sensor_maxmatch = {_mm}")

    _ns = int(self.solver.mjw_model.nsensor)
    _nc = int((wp.to_torch(self.solver.mjw_model.sensor_type) == 42).sum()) if _ns else 0
    print(f"[newton-env] sensors: nsensor={_ns} contact={_nc} "
          f"nsensordata={int(self.solver.mjw_model.nsensordata)}"
          + ("" if _nc else "  <-- contact-gated rewards will read zero"))
    self._ref_mj = mujoco.MjModel.from_xml_path(xml)
    restore_freejoint_damping(cap.spec, xml, verbose=False)

    # The simple-body fix recompiles Newton's spec and installs put_model/put_data of the result.
    # That is right for an analytic-primitive object, and destructive for a mesh one: Newton attaches
    # the SDF volumes to the warp model when it builds the solver, so replacing that model wholesale
    # leaves the mesh collider with no distance field behind it. Measured: with the fix the first
    # solver step produced 150 NaNs in qpos; without it the same scene steps cleanly.
    #
    # It is also unnecessary here. The fix recovers MuJoCo's compressed mass-matrix layout for a body
    # that qualifies as "simple" -- a free body with isotropic, centred inertia. A real mesh is
    # neither, so nC=1102 is the correct answer for it rather than a defect to repair.
    if sdf_object_stl:
      print(f"[newton-env] simple-body fix skipped: the object is a mesh collider, whose SDF lives "
            f"on the warp model the fix would replace (nC={int(self.solver.mjw_model.nC)})")
    else:
      restore_simple_bodies(self.solver, cap.spec, nworld=self.num_envs,
                            nconmax=nconmax, njmax=njmax, verbose=False)
      nC, nC_ref = int(self.solver.mjw_model.nC), int(self._ref_mj.nC)
      if nC != nC_ref:
        raise RuntimeError(f"nC {nC} != {nC_ref} with a primitive object; the spec repair did not take")

    self.control = self.nmodel.control()
    self.state_in, self.state_out = self.nmodel.state(), self.nmodel.state()

    # Contact stiffness for the object's collider. Measured with the shared defaults
    # (solref 0.02 1.0): mjlab's analytic sphere settles 0.37mm into the table, the stapler's
    # convex-mesh collider settles 1.93mm and transiently 11.6mm -- deep enough to see. The knob
    # is scoped to the object geom so the robot's contacts keep mjlab's parameters exactly.
    if object_solref and sdf_object_stl:
      import mujoco as _mj
      _vals = [float(x) for x in object_solref.replace(" ", "").split(",")]
      _mm = self.solver.mj_model
      _hit = 0
      for _g in range(_mm.ngeom):
        _n = _mj.mj_id2name(_mm, _mj.mjtObj.mjOBJ_GEOM, _g) or ""
        if "apple" not in _n:
          continue
        _mm.geom_solref[_g][:len(_vals)] = _vals
        _hit += 1
      if _hit == 0:
        raise RuntimeError("--object-solref matched no object geom")
      import warp as _wp
      _wp.to_torch(self.solver.mjw_model.geom_solref)[:] = _wp.to_torch(
        _wp.array(_mm.geom_solref, dtype=float))
      print(f"[newton-env] object solref -> {_vals} on {_hit} geom(s) "
            f"(resting penetration 0.04mm stapler / 0.28mm mug, against 1.88mm "
            f"at the shared default and 0.37mm for mjlab's sphere)")

    self._env = NewtonEnv(self.solver.mj_model, self.solver.mjw_data, self.num_envs, device,
                          control=self.control, rename_from=self._ref_mj,
                          physics_dt=self.physics_dt, decimation=self.decimation,
                          solver=self.solver, object_entity=object_entity)
    self._env.forward()
    # Termination terms read these off the env the managers were built with, not off this wrapper.
    self._env.max_episode_length = self.max_episode_length
    self._env.max_episode_length_s = float(cfg.episode_length_s)
    self._env.cfg = cfg
    self._env.common_step_counter = 0

    # --- mjlab's own managers, against Newton state ---------------------------
    from mjlab.tasks.apple_eat import mdp as amdp
    from mjlab.managers.reward_manager import RewardManager
    from mjlab.managers.termination_manager import TerminationManager
    from mjlab.managers.event_manager import EventManager

    sonic_cfg = (cfg.actions.get("sonic_action") if isinstance(cfg.actions, dict)
                 else getattr(cfg.actions, "sonic_action"))
    # The torque law inside Sonic53Action.apply_actions is discarded on this backend --
    # set_joint_effort_target is a verified no-op here -- so it can be cut out along with the host
    # synchronisation it performs on every substep. See src/newton_action.py.
    _action_cls = amdp.Sonic53Action
    if effortless_action:
      from newton_action import make_effortless
      _action_cls = make_effortless(amdp.Sonic53Action)
      print("[newton-env] action term: torque law removed (it reaches no actuator here)")
    # Must be installed before the action term first writes a table pose. See src/newton_table.py:
    # the scene was authored around a 4cm sphere, so a real collider does not rest where the
    # reference puts the object unless the table moves to meet it.
    if table_under_object:
      if not sdf_object_stl:
        raise RuntimeError("--table-under-object needs the object mesh (--sdf-object) to know "
                           "where the real collider bottom is")
      from newton_table import install as _install_table
      _ref_pkl = os.environ.get("APPLE_EAT_PKL")
      if not _ref_pkl:
        raise RuntimeError("--table-under-object needs APPLE_EAT_PKL: the object's resting height "
                           "is read from the reference clip, not guessed")
      _install_table(self.solver.mj_model, sdf_object_stl, _ref_pkl,
                     z_offset=float(os.environ.get("APPLE_SCENE_Z_OFFSET", 0.0)))

    self.action_term = _action_cls(sonic_cfg, self._env)
    self.action_manager = self._env.bind_action_manager(
      self.action_term.action_dim, {"sonic_action": self.action_term})
    self.action_manager.total_action_dim = self.action_term.action_dim

    # Prefer mjlab's real ObservationManager: it owns history buffers, group concatenation and the
    # nan policy. Hand-assembling the groups reproduces none of that, and a history buffer that is
    # not maintained is the exact defect that cost this project a working grasp once already.
    self.observation_manager = None
    try:
      from mjlab.managers.observation_manager import ObservationManager
      self.observation_manager = ObservationManager(cfg.observations, self._env)
      print(f"[newton-env] using mjlab ObservationManager "
            f"({len(self.observation_manager.active_terms)} groups)")
    except Exception as e:
      # Fatal rather than a fallback. The per-group builders below are early-port scaffolding and
      # are not the code mjlab runs; silently switching to them would mean training on a different
      # observation than the one whose parity with mjlab was measured at 8.3e-07.
      raise RuntimeError(
        f"mjlab's ObservationManager failed to construct ({type(e).__name__}: {e}). "
        "Refusing to fall back to the per-group builders -- fix the cause instead.") from e

    obs_cfg = cfg.observations if isinstance(cfg.observations, dict) else vars(cfg.observations)
    self._obs_builders = {}
    if self.observation_manager is None:
      for gname, gcfg in obs_cfg.items():
        terms = getattr(gcfg, "terms", None) or (gcfg if isinstance(gcfg, dict) else None)
        if not terms:
          continue
        t = terms.get("policy") or next(iter(terms.values()))
        try:
          self._obs_builders[gname] = t.func(t, self._env)
        except Exception as e:
          print(f"[newton-env] observation builder {gname!r} unavailable "
                f"({type(e).__name__}: {e})")

    # mjlab's MetricsManager is what produces the task-level numbers -- lift_success, contact,
    # object tracking error. Without it the only things logged are PPO internals, and a run whose
    # physics has failed looks identical to one that is merely early: 3576 iterations were spent
    # that way before the NaN was found by hand.
    self.metrics_manager = None
    try:
      from mjlab.managers.metrics_manager import MetricsManager
      self.metrics_manager = MetricsManager(cfg.metrics, self._env)
      print(f"[newton-env] MetricsManager: "
            f"{len(getattr(self.metrics_manager, 'active_terms', []) or [])} terms")
    except Exception as e:
      # Fatal: the MetricsManager is what produces lift_success, sequence_success and
      # object_mpjpe_mm -- the numbers the whole port is judged on. A run without it looks healthy
      # and reports nothing true.
      raise RuntimeError(
        f"mjlab's MetricsManager failed to construct ({type(e).__name__}: {e})") from e

    # scale_by_dt must come from the config, not the constructor default. mjlab passes
    # cfg.scale_rewards_by_dt here (manager_based_rl_env.py:330) and this task sets it
    # False; taking the default of True multiplies every reward term by dt=0.02, making
    # the whole reward signal 50x smaller than mjlab's while every individual term still
    # measures identical -- which is why this survived a term-by-term parity check.
    # Contact sensors have to reach the MDP through the scene, not just the model. The transplant
    # above puts them in Newton's compiled model, but mjlab reads them as scene entities
    # (`env.scene["hand_apple_contact"].data.found`) and falls back to zeros on a KeyError -- which
    # is why contact_duration (weight 1.0) and object_hard_lift (weight 2.0) never paid out, and
    # why every contact metric read 0.0 while the policy was visibly grasping.
    #
    # A one-environment mjlab env is built purely to obtain its constructed sensor objects; they are
    # then rebound onto Newton's buffers. Constructing them by hand would mean reimplementing
    # mjlab's pattern expansion, which is the kind of second implementation that has drifted before.
    import copy as _copy
    from mjlab.envs import ManagerBasedRlEnv as _MjlabEnv
    from newton_bridge import bind_contact_sensors

    _sensor_cfg = _copy.deepcopy(cfg)
    _sensor_cfg.scene.num_envs = 1
    _probe = _MjlabEnv(cfg=_sensor_cfg, device=device, render_mode=None)
    _bound = bind_contact_sensors(self._env.scene, _probe.scene, self.solver.mj_model,
                                  self.solver.mjw_model, self.solver.mjw_data, device)
    self._contact_sensors = [self._env.scene[n] for n in _bound]
    print(f"[newton-env] contact sensors on the scene: {len(_bound)} "
          f"({', '.join(n for n in _bound if 'contact' in n and n.count('_') < 4)})")

    self.reward_manager = RewardManager(cfg.rewards, self._env,
                                        scale_by_dt=cfg.scale_rewards_by_dt)
    self.termination_manager = TerminationManager(cfg.terminations, self._env)
    self.event_manager = EventManager(cfg.events, self._env)

    # CUDA graph capture of the physics substep. mujoco_warp issues hundreds of short kernels per
    # step, so at these env counts the GPU spends most of its time idle between launches -- 15-20%
    # utilisation at 2048 envs. A captured graph replays the whole launch sequence as one command.
    #
    # Two graphs, not one: the substep swaps state_in/state_out, and a captured graph bakes in the
    # buffer pointers it was recorded with. Decimation is even, so alternating A->B and B->A
    # returns the buffers to their original orientation at the end of each control step.
    # Optional recording of env 0 for video. The table is a mocap body, so its pose lives outside
    # qpos; without it the renderer draws the table at the origin and the object appears to float.
    self._video_path = newton_video
    self._video_steps = int(video_steps)
    self._video_size = video_size
    self._video_cam = video_cam
    self._nonfinite_total = 0
    self._object_collider_view = object_collider_view
    self._viewer_gl = None
    self._video_frames: list = []
    self._cam_eye = self._cam_target = None

    self._dump_qpos_path = dump_qpos
    self._dump_steps = int(dump_steps)
    self._qpos_log: list = []
    self._mocap_log: list = []

    self._use_cuda_graph = bool(cuda_graph)
    self._graphs: list = []
    self._graph_warmup = 4      # uncaptured substeps per parity before capturing
    # The bridge reads the runner's `_residual_*` bookkeeping through this back-reference; the
    # runner writes it on this object, not on the bridge the MDP terms see.
    self._env._owner = self

    self.episode_length_buf = self._env.episode_length_buf
    self._all = torch.arange(self.num_envs, dtype=torch.long, device=device)

    # --- optional live view ---------------------------------------------------
    self.viewer = None
    self._render_every = max(1, int(render_every))
    self._render_tick = 0
    if viser_port:
      import newton.viewer as _nv
      self.viewer = _nv.ViewerViser(port=int(viser_port),
                                    label=f"Newton training ({self.num_envs} envs)")
      # Newton draws visual geometry and hides colliders. The object here has only a collider -- the
      # SDF mesh -- so without this the robot appears to grasp nothing at all.
      _flags = wp.to_torch(self.nmodel.shape_flags)
      _sbody = wp.to_torch(self.nmodel.shape_body)
      _VIS = int(newton.ShapeFlags.VISIBLE)
      for _bid in torch.unique(_sbody):
        _idx = (_sbody == _bid).nonzero(as_tuple=True)[0]
        if int((_flags[_idx] & _VIS).sum()) == 0:
          _flags[_idx] |= _VIS
      self.viewer.set_model(self.nmodel)
      print(f"[newton-env] viser on http://localhost:{viser_port} "
            f"(rendering every {self._render_every} control steps)")
    # mjlab's event manager needs the global step count for reset-mode terms that fire on a schedule
    # (min_step_count_between_reset). It is a counter over env-steps, not control steps.
    self.common_step_counter = 0

  # ---------------------------------------------------------------- interface
  @property
  def unwrapped(self) -> "NewtonVecEnv":
    return self

  @property
  def scene(self):
    return self._env.scene

  @property
  def observation_space(self):
    return None

  @property
  def action_space(self):
    return None

  def get_observations(self, update_history: bool = False):
    if self.observation_manager is not None:
      # update_history=True must happen exactly once per control step: compute() caches, and that
      # cache is what keeps a second call in the same step from pushing the buffers twice.
      return self.observation_manager.compute(update_history=update_history)
    from tensordict import TensorDict
    return TensorDict({g: b(self._env) for g, b in self._obs_builders.items()},
                      batch_size=[self.num_envs])

  def reset(self, env_ids: torch.Tensor | None = None):
    ids = self._all if env_ids is None else env_ids
    self._reset_idx(ids)
    return self.get_observations(), self.extras

  def _nonfinite_worlds(self) -> "torch.Tensor | None":
    """World indices whose MuJoCo state has gone non-finite."""
    import warp as wp
    d = self.solver.mjw_data
    qpos = wp.to_torch(d.qpos)
    qvel = wp.to_torch(d.qvel)
    bad = (~torch.isfinite(qpos)).any(1) | (~torch.isfinite(qvel)).any(1)
    if not bool(bad.any()):
      return None
    return bad.nonzero(as_tuple=False).squeeze(-1)

  def _clear_world_state(self, ids: "torch.Tensor") -> None:
    """Wipe every per-world field a reset does not write, so a dead world can come back.

    The reset events rewrite qpos and qvel. They do not touch qacc, the warm start, or the applied
    forces, and a NaN parked in any of those reproduces itself on the next step -- which is why
    the dead-world count only ever went up.
    """
    import warp as wp
    d = self.solver.mjw_data
    for name in ("qacc", "qacc_warmstart", "qacc_smooth", "qfrc_applied", "xfrc_applied",
                 "qfrc_constraint", "qfrc_smooth", "act", "act_dot", "efc_force", "qvel"):
      arr = getattr(d, name, None)
      if arr is None:
        continue
      try:
        t = wp.to_torch(arr)
      except Exception:
        continue
      if t.shape[0] != self.num_envs:
        continue
      t[ids] = 0.0

  def _reset_idx(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    self.event_manager.apply(mode="reset", env_ids=env_ids,
                             global_env_step_count=self.common_step_counter * self.num_envs)

    # Every manager's reset() has to run, in mjlab's order (it marks the sequence order-sensitive).
    # Skipping them is not merely a logging gap: reward_manager.reset zeroes the per-episode sums,
    # metrics_manager.reset clears episodic accumulators like lift_success, and action_manager.reset
    # drops the previous action the smoothness terms difference against -- so without these, state
    # leaks across episode boundaries for the whole run. The task has no curriculum or command
    # manager (both cfg dicts are empty), so mjlab's calls to those have no counterpart here.
    log = self.extras.setdefault("log", {})
    for mgr in (self.observation_manager, self.action_manager, self.reward_manager,
                self.metrics_manager, self.event_manager, self.termination_manager):
      fn = getattr(mgr, "reset", None)
      if fn is None:      # the action manager here is a bridge view, not mjlab's ActionManager
        continue
      info = fn(env_ids)
      if info:
        log.update(info)

    self._env.episode_length_buf[env_ids] = 0
    # qpos was written; xpos/xquat are stale until forward kinematics runs, and the observation
    # terms read the derived fields rather than qpos.
    self._env.forward()


  def _physics_substep_graphed(self, parity: int) -> None:
    """Run one solver substep, replaying a captured CUDA graph instead of relaunching kernels.

    `parity` selects which of the two buffer orientations this substep uses. Capture happens
    lazily on the first substep of each parity, after the uncaptured warm-up steps have forced
    every lazy kernel compilation and allocation to happen -- capturing an allocation would bake a
    stale pointer into the graph.
    """
    import warp as wp

    while len(self._graphs) <= parity:
      self._graphs.append(None)

    if self._graphs[parity] is None:
      if self._graph_warmup > 0:
        self._graph_warmup -= 1
        self._physics_step()
        return
      try:
        with wp.ScopedCapture() as cap:
          self._physics_step()
        self._graphs[parity] = cap.graph
        print(f"[newton-env] captured CUDA graph for substep parity {parity}")
        # Capture records the launches without running them, so this substep's physics has not
        # happened yet. Replaying the fresh graph performs it, keeping the captured step on the
        # trajectory instead of silently skipping it.
        wp.capture_launch(cap.graph)
        return
      except Exception as e:
        print(f"[newton-env] CUDA graph capture failed ({type(e).__name__}: {e}); "
              "continuing without it")
        self._use_cuda_graph = False
        self._physics_step()
        return

    wp.capture_launch(self._graphs[parity])


  def _record_frame(self) -> None:
    """Append env 0's qpos and mocap pose, and write the npz once enough frames are collected."""
    import warp as wp
    import numpy as np

    d = self.solver.mjw_data
    self._qpos_log.append(wp.to_torch(d.qpos)[0].detach().cpu().numpy().copy())
    self._mocap_log.append((wp.to_torch(d.mocap_pos)[0].detach().cpu().numpy().copy(),
                            wp.to_torch(d.mocap_quat)[0].detach().cpu().numpy().copy()))
    if len(self._qpos_log) < self._dump_steps:
      return
    # Mocap body names travel with the poses. The renderer replays into a different model whose
    # mocap bodies need not be in the same order -- index 2 here is the table, and writing it into
    # index 0 there put the table under the robot's feet in the first video made this way.
    import mujoco as _mj
    m = self.solver.mj_model
    names = [_mj.mj_id2name(m, _mj.mjtObj.mjOBJ_BODY, b) or ""
             for b in range(m.nbody) if m.body_mocapid[b] >= 0]
    order = [b for b in range(m.nbody) if m.body_mocapid[b] >= 0]
    names = [x for _, x in sorted(zip([int(m.body_mocapid[b]) for b in order], names))]
    np.savez_compressed(self._dump_qpos_path,
                        qpos=np.stack(self._qpos_log),
                        mocap_pos=np.stack([q[0] for q in self._mocap_log]),
                        mocap_quat=np.stack([q[1] for q in self._mocap_log]),
                        mocap_names=np.array(names))
    print(f"[newton-env] wrote {self._dump_qpos_path} "
          f"({len(self._qpos_log)} frames of env 0 for rendering)")


  def _setup_newton_viewer(self) -> None:
    """Newton's own offscreen renderer, drawing the scene the physics actually holds.

    Replaying qpos through mjlab's visual model instead -- which is what the first videos did --
    draws the object as the 4cm placeholder sphere the scene authors, because the real mesh only
    ever reached the collider. It also has to re-apply the mocap table by hand, and getting that
    mapping wrong put the table under the robot's feet. Rendering Newton's own model has neither
    failure mode: what is drawn is what was simulated.
    """
    import numpy as np
    import newton
    import torch
    import warp as wp

    # pyglet creates a shadow window even when the viewer is asked for headless, which fails on a
    # box with no display. Its headless path uses EGL and must be selected before pyglet is imported.
    os.environ.setdefault("PYGLET_HEADLESS", "1")
    import pyglet as _pyglet
    _pyglet.options["headless"] = True
    import newton.viewer as _nv

    w, h = (int(x) for x in self._video_size.lower().split("x"))
    self._viewer_gl = _nv.ViewerGL(width=w, height=h, headless=True)

    # Newton draws visual shapes, not colliders. mjlab's robot carries visual meshes, but the object
    # and the table are each a single geom that is both visual and collidable, and arrives here
    # marked collide-but-not-visible -- a video of a robot grasping thin air over an invisible
    # table. Reveal colliders only on bodies that have no visible shape at all, so the robot is not
    # drawn twice.
    flags = wp.to_torch(self.nmodel.shape_flags)
    sbody = wp.to_torch(self.nmodel.shape_body)
    VIS = int(newton.ShapeFlags.VISIBLE)
    revealed = 0
    for bid in torch.unique(sbody):
      idx = (sbody == bid).nonzero(as_tuple=True)[0]
      if int((flags[idx] & VIS).sum()) == 0:
        flags[idx] |= VIS
        revealed += len(idx)

    # Draw the object as the shape the physics actually collides, not the shape it is authored
    # to look like. The two are different objects: the visual is mjlab's authored geom, the
    # collider is the real STL we swapped in and built the SDF from.
    if self._object_collider_view:
      COLL = int(newton.ShapeFlags.COLLIDE_SHAPES)
      HYD = int(newton.ShapeFlags.HYDROELASTIC)
      lbls = list(self.nmodel.shape_label)
      obj_bodies = {int(sbody[i]) for i in range(len(lbls))
                    if int(flags[i]) & HYD and "table" not in (lbls[i] or "").lower()}
      swapped = 0
      for bid in obj_bodies:
        for i in (sbody == bid).nonzero(as_tuple=True)[0].tolist():
          if int(flags[i]) & COLL:
            flags[i] |= VIS
          else:
            flags[i] &= ~VIS
          swapped += 1
      print(f"[newton-env] object drawn as its collider: {swapped} shape(s) on "
            f"{len(obj_bodies)} body/bodies re-flagged")

    self._viewer_gl.set_model(self.nmodel)

    # The viewer spreads parallel worlds apart for display -- _auto_compute_world_offsets runs on
    # set_model -- so with four envs the robot is drawn far from where the physics puts it, and a
    # camera aimed at the true scene sees bare floor with a speck on the horizon. Collapse the
    # offsets so the picture matches the simulation, and draw only world 0 so the video shows one
    # robot at full size instead of four small ones.
    try:
      self._viewer_gl.set_world_offsets((0.0, 0.0, 0.0))
    except Exception as e:
      print(f"[newton-env] could not clear world offsets ({type(e).__name__}: {e})")
    if self.num_envs > 1:
      try:
        self._viewer_gl.set_visible_worlds([0])
      except Exception as e:
        print(f"[newton-env] could not restrict rendering to world 0 "
              f"({type(e).__name__}: {e}); all {self.num_envs} will be drawn")
    cam = getattr(self._viewer_gl, "camera", None)
    if cam is not None:
      # Framed on the hands and the object rather than the whole scene: the earlier eye sat 1.69m
      # back, which fits the robot and the table in but leaves the grasp itself a few pixels wide.
      # This sits 1.07m out in three-quarter view: the whole robot reads large, and the object on the
      # table stays in frame with both hands. Override with --video-cam "ex,ey,ez,tx,ty,tz".
      eye, target = [1.55, -0.55, 1.05], [0.60, -0.10, 0.84]
      if self._video_cam:
        nums = [float(x) for x in self._video_cam.replace(" ", "").split(",")]
        if len(nums) != 6:
          raise ValueError(f"--video-cam wants 6 numbers (eye xyz, target xyz), got {len(nums)}")
        eye, target = nums[:3], nums[3:]

      # Anchor on environment 0. Replicating worlds spreads them across the ground plane, so world
      # coordinates that frame the robot at one env count point at bare floor at another -- checked
      # with one env, the framing was right; the four-env video it was used for showed the robot as
      # a speck on the horizon.
      origin = self._env.scene.env_origins[0].detach().cpu().numpy()
      eye = [eye[i] + float(origin[i]) for i in range(3)]
      target = [target[i] + float(origin[i]) for i in range(3)]
      self._cam_eye, self._cam_target = eye, target
      print(f"[newton-env] env 0 origin {np.round(origin, 3).tolist()}")
      print(f"[newton-env] camera at {eye} looking at {target} "
            f"({np.linalg.norm(np.array(eye) - np.array(target)):.2f} m out)")
    else:
      self._cam_eye = self._cam_target = None

    print(f"[newton-env] Newton ViewerGL {w}x{h} headless, "
          f"{revealed} collider shape(s) revealed, writing {self._video_path}")


  def _apply_camera(self) -> None:
    """Point the viewer's camera at the target, through the API the viewer actually reads.

    ViewerGL has no look-at: `set_camera(pos, pitch, yaw)` is the entry point, and assigning to
    `viewer.camera.pos` does nothing, because set_model replaces the Camera object. For a Z-up
    scene the viewer derives its forward vector as pitch = asin(dz), yaw = atan2(dy, dx).
    """
    import numpy as np
    import warp as wp

    if self._cam_eye is None or self._viewer_gl is None:
      return
    eye = np.asarray(self._cam_eye, dtype=np.float64)
    d = np.asarray(self._cam_target, dtype=np.float64) - eye
    n = np.linalg.norm(d)
    if n < 1e-9:
      return
    d /= n
    pitch = float(np.rad2deg(np.arcsin(np.clip(d[2], -1.0, 1.0))))
    yaw = float(np.rad2deg(np.arctan2(d[1], d[0])))
    try:
      self._viewer_gl.set_camera(wp.vec3(*(float(x) for x in eye)), pitch, yaw)
    except Exception as e:
      if not getattr(self, "_cam_warned", False):
        print(f"[newton-env] camera placement failed ({type(e).__name__}: {e})")
        self._cam_warned = True

  def _capture_video_frame(self) -> None:
    """Render one frame from Newton's own state, after syncing it from the authoritative mjw_data."""
    import numpy as np
    import warp as wp

    if self._viewer_gl is None:
      self._setup_newton_viewer()

    # mjw_data is authoritative here (update_data_interval=0), and the table is a mocap body whose
    # pose never enters State -- without this sync it renders at the origin. Newton body i is
    # MuJoCo body i+1 (MuJoCo body 0 is the world), and Newton stores quaternions xyzw against
    # MuJoCo's wxyz; both conventions were measured, not assumed.
    if getattr(self, "_sync_n", None) is not None:
      # On the native-contacts path State.body_q is live physics input, not just something to draw:
      # the next substep's collide() reads this exact buffer. Writing it from the i+1 assumption
      # below while the collision pipeline's table pose came from mjc_body_to_newton meant two
      # mappings disagreed -- filmed as the stapler resting correctly for 51 steps and then being
      # knocked to the floor. Use the same map the physics uses.
      self._sync_body_q_from_mujoco()
    else:
      bq = wp.to_torch(self.state_in.body_q)
      xp = wp.to_torch(self.solver.mjw_data.xpos)
      xq = wp.to_torch(self.solver.mjw_data.xquat)
      nworld = xp.shape[0]
      per_world = bq.shape[0] // nworld
      bqv = bq.view(nworld, per_world, 7)
      bqv[:, :, 0:3] = xp[:, 1:1 + per_world]
      bqv[:, :, 3:7] = xq[:, 1:1 + per_world][:, :, [1, 2, 3, 0]]

    # Re-applied every frame. set_model rebuilds Camera, and the viewer refits it to the scene on
    # its first frame; either one silently discards a placement made at setup. That is why a
    # one-env check looked right -- the refit lands near the robot when the scene is small -- while
    # the four-env video it stood in for rendered the robot as a speck on the horizon.
    self._apply_camera()

    t = len(self._video_frames) * self.step_dt
    self._viewer_gl.begin_frame(t)
    self._viewer_gl.log_state(self.state_in)
    self._viewer_gl.end_frame()
    img = self._viewer_gl.get_frame()
    self._video_frames.append(
      np.asarray(img.numpy() if hasattr(img, "numpy") else img).copy())

    if len(self._video_frames) < self._video_steps:
      return
    import imageio.v2 as imageio
    wr = imageio.get_writer(self._video_path, fps=int(round(1.0 / self.step_dt)),
                            codec="libx264", macro_block_size=None, quality=8)
    for f in self._video_frames:
      wr.append_data(f)
    wr.close()
    self._viewer_gl.close()
    self._viewer_gl = None
    print(f"[newton-env] wrote {self._video_path} "
          f"({len(self._video_frames)} frames from Newton's own renderer)")
    self._video_path = None


  def _sync_body_q_from_mujoco(self) -> None:
    """Copy every mapped body's world pose from MuJoCo into State.body_q.

    SolverMuJoCo derives State.body_q from the joint coordinates by forward kinematics, so a body
    with no joint is never written back: the table is static, and the collision pipeline saw it at
    the world origin. This is also the single writer of State.body_q outside the solver -- the
    renderer used to write its own, from a different body mapping. MuJoCo stores the quaternion
    w-first; Newton's transform stores it w-last.
    """
    d = self.solver.mjw_data
    wp.launch(_sync_body_q_kernel, dim=len(self._sync_n_wp),
              inputs=[d.xpos, d.xquat, self._sync_w_wp, self._sync_b_wp, self._sync_n_wp],
              outputs=[self.state_in.body_q], device=self.device)

  def _physics_step(self) -> None:
    """One solver substep, with Newton's contacts recomputed first when they are in use.

    Every call site goes through here. The first attempt patched only one of four `solver.step`
    call sites and the other three kept passing None, which surfaces as
    `NoneType has no attribute rigid_contact_max` -- from a site that looked unrelated.
    """
    if self.collision_pipeline is not None:
      self._sync_body_q_from_mujoco()
      self.collision_pipeline.collide(self.state_in, self.contacts)
    self.solver.step(self.state_in, self.state_out, self.control, self.contacts, self.physics_dt)

  def step(self, action: torch.Tensor):
    self.extras["log"] = dict()
    self.action_manager.advance(action)
    self.action_term.process_actions(action)

    for i in range(self.decimation):
      self.action_term.apply_actions()
      if self._use_cuda_graph:
        self._physics_substep_graphed(i % 2)
      else:
        self._physics_step()
      self.state_in, self.state_out = self.state_out, self.state_in

    self._env.episode_length_buf += 1
    self.common_step_counter += 1
    self._env.common_step_counter = self.common_step_counter

    if self.viewer is not None:
      self._render_tick += 1
      if self._render_tick % self._render_every == 0:
        self._render()

    # Order follows mjlab's step (manager_based_rl_env.py:451): terminations, then the reward,
    # then the metrics -- and the reward has to be published on the env as `reward_buf` before the
    # metrics run, because reward_total_metric reads it.
    for _sensor in getattr(self, "_contact_sensors", ()):
      _sensor.update(self.step_dt)

    terminated = self.termination_manager.compute()
    time_out = getattr(self.termination_manager, "time_outs",
                       torch.zeros_like(terminated, dtype=torch.bool))

    reward = self.reward_manager.compute(dt=self.step_dt)
    self._env.reward_buf = reward

    if self.metrics_manager is not None:
      # compute() takes no dt. The earlier `compute(dt=...)` wrapped in `except Exception: pass`
      # raised TypeError on every step of every run and was swallowed, so the manager never
      # produced a metric -- the Episode_Metrics/* values that reached tensorboard were
      # zero-initialised accumulators being flushed at reset.
      self.metrics_manager.compute()

    # Non-finite worlds do not heal. Measured on the native path, 512 envs, 300 steps: the first
    # env goes non-finite somewhere before step 101 and the count climbs 1 -> 4 -> 8 and never
    # falls, even though those envs are terminating and being reset the whole time -- so the NaN
    # lives in state the reset events do not write (qacc and the solver's warm start). The
    # MuJoCo-contact path shows 0 of 512 over the same run, so this is native-only.
    #
    # Two costs if it is left alone: over a 4500-iteration run (the baseline needs 4447 before it
    # lifts anything) the whole batch dies, and every mean-based metric is poisoned long before
    # that -- _safe_log takes mean() and only then nan_to_num(..., 0.0), so one dead world makes a
    # metric read a clean-looking 0.0000.
    _bad = self._nonfinite_worlds()
    if _bad is not None and len(_bad) > 0:
      self._nonfinite_total += int(len(_bad))
      self.extras.setdefault("log", {})["Health/nonfinite_worlds"] = float(len(_bad))
      self._clear_world_state(_bad)
      done_ids = torch.unique(torch.cat([
        (terminated | time_out).nonzero(as_tuple=False).squeeze(-1), _bad]))
      terminated[_bad] = True
    else:
      done_ids = (terminated | time_out).nonzero(as_tuple=False).squeeze(-1)
    if len(done_ids) > 0:
      self._reset_idx(done_ids)

    obs = self.get_observations(update_history=True)
    # surface the task metrics so the runner logs them alongside the PPO curves
    self.extras.setdefault("log", {}).update(self._env.extras.get("log", {}))
    if self._dump_qpos_path is not None and len(self._qpos_log) < self._dump_steps:
      self._record_frame()

    if self._video_path is not None:
      self._capture_video_frame()

    self.extras["time_outs"] = time_out
    # mjlab's wrapper unpacks five values: terminated and truncated are separate, because a timeout
    # must bootstrap the value function while a real termination must not.
    return obs, reward, terminated, time_out, self.extras

  def _render(self) -> None:
    """Draw from mjw_data rather than Newton's State.

    Two reasons. The table is a mocap body: its pose is written to mocap_pos and never enters State,
    so it would draw at the origin with the object apparently floating. And with N replicated worlds
    Newton's bodies are laid out world-major (world w, local body b -> w*B + b) against mujoco's
    per-world (w, b+1), so the two have to be related explicitly rather than copied straight across.
    """
    bq = wp.to_torch(self.state_in.body_q)
    xp = wp.to_torch(self.solver.mjw_data.xpos)
    xq = wp.to_torch(self.solver.mjw_data.xquat)
    nworld, nb_mj = xp.shape[0], xp.shape[1]
    nb = nb_mj - 1                                   # mujoco body 0 is the world
    if bq.shape[0] == nworld * nb:
      bq[:, 0:3] = xp[:, 1:1 + nb].reshape(-1, 3)
      bq[:, 3:7] = xq[:, 1:1 + nb].reshape(-1, 4)[:, [1, 2, 3, 0]]   # wxyz -> xyzw
    self.viewer.begin_frame(self.common_step_counter * self.step_dt)
    self.viewer.log_state(self.state_in)
    self.viewer.end_frame()

  def close(self) -> None:
    if self.viewer is not None:
      try:
        self.viewer.close()
      except Exception:
        pass
    return None
