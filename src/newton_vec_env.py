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
import re
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
               sdf_object_stls: list[str] | None = None,
               clip_env_counts: list[int] | None = None,
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
    _sts = os.environ.get("SIM_TIMESTEP", "").strip()
    if _sts:
      _new_dt = float(_sts)
      _ratio = self.physics_dt / _new_dt
      if abs(_ratio - round(_ratio)) > 1e-6:
        raise RuntimeError(f"SIM_TIMESTEP {_new_dt} does not divide the configured "
                           f"{self.physics_dt}; decimation could not stay integral and the control "
                           f"rate would change along with the contact")
      _old_dt, _old_dec = self.physics_dt, self.decimation
      self.decimation = int(round(self.decimation * _ratio))
      self.physics_dt = _new_dt
      if hasattr(cfg.sim, "mujoco"):
        cfg.sim.mujoco.timestep = _new_dt
      cfg.decimation = self.decimation
      print(f"[newton-env] SIM_TIMESTEP {1000*_new_dt:.3f} ms (was {1000*_old_dt:.3f}); decimation "
            f"{_old_dec} -> {self.decimation}, control rate unchanged at "
            f"{1.0/(self.physics_dt*self.decimation):.1f} Hz", flush=True)
    self.step_dt = self.physics_dt * self.decimation
    self.max_episode_length = int(math.ceil(float(cfg.episode_length_s) / self.step_dt))

    # --- Newton model: one authored scene, replicated into parallel worlds -----
    # Each clip trains on its own object, so the scene is authored once per distinct mesh and
    # replicated into that clip's block of worlds. Everything below is identical between clips
    # except the collider swapped in for `sdf_object_stl`.
    def _author_scene(sdf_object_stl):
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

      # ON BY DEFAULT. Newton does not inherit MuJoCo's "same weld group never collides" rule,
      # so wrist_yaw and palm -- one rigid body, meshes overlapping 9 mm by design -- produced
      # 9 contacts per step per env, 12-14% of every contact in the scene, carrying no force and
      # no information. Measured A/B at 64 envs x 3 seeds on frozen_S8: total contacts
      # 825,455 -> 713,912 (-13.5%, exactly the 115,200 removed), lift_success 0.1641 -> 0.1617
      # (inside the 3% noise floor), penetration and wall-clock unchanged. Set to 0 to disable.
      if os.environ.get("HAND_COLLISION_FIX", "1").strip() not in ("0", "off", "false"):
        # Applied to the MuJoCo spec before the scene is replicated, so every world inherits it.
        import mujoco as _hcf
        _hm = mujoco.MjModel.from_xml_path(xml)
        _hbn = lambda _b: (_hcf.mj_id2name(_hm, _hcf.mjtObj.mjOBJ_BODY, _b) or "")
        _fing = re.compile(r"(left|right)_finger(\d)_link(\d)")
        _hand = re.compile(r"(finger|palm|wrist|mount)")
        _pairs, _reasons = [], {}
        for _b1 in range(_hm.nbody):
          _n1 = _hbn(_b1)
          if not _hand.search(_n1):
            continue
          for _b2 in range(_b1 + 1, _hm.nbody):
            _n2 = _hbn(_b2)
            if not _hand.search(_n2):
              continue
            _why = None
            if int(_hm.body_weldid[_b1]) == int(_hm.body_weldid[_b2]):
              _why = "same weld group"
            elif int(_hm.body_parentid[_b1]) == _b2 or int(_hm.body_parentid[_b2]) == _b1:
              _why = "one joint apart"
            else:
              # one joint apart THROUGH a fixed adapter: walk up past welded ancestors
              def _weld_root(_b):
                while _b > 0 and int(_hm.body_jntnum[_b]) == 0:
                  _b = int(_hm.body_parentid[_b])
                return _b
              _r1, _r2 = _weld_root(_b1), _weld_root(_b2)
              if _r1 != _r2 and (int(_hm.body_parentid[_r1]) == _r2
                                 or int(_hm.body_parentid[_r2]) == _r1):
                _why = "one joint apart via a fixed adapter"
              else:
                _m1, _m2 = _fing.search(_n1), _fing.search(_n2)
                if _m1 and _m2 and _m1.group(1) == _m2.group(1) and _m1.group(2) == _m2.group(2):
                  _why = "same finger"
            if _why:
              _pairs.append((_n1, _n2))
              _reasons[_why] = _reasons.get(_why, 0) + 1
        print(f"[collfix] excluding {len(_pairs)} hand body pairs: "
              + ", ".join(f"{k}={v}" for k, v in sorted(_reasons.items())), flush=True)
        # Map body name -> the scene's shape indices, so the body pairs become shape pairs.
        # Clear the link2 isolation: uniform contype on every hand geom, so what may collide is
        # decided by the exclusion list above and nothing else.
        _nct = 0
        for _g in range(_hm.ngeom):
          _gb = _hbn(int(_hm.geom_bodyid[_g]))
          if not _hand.search(_gb):
            continue
          if int(_hm.geom_contype[_g]) or int(_hm.geom_conaffinity[_g]):
            if int(_hm.geom_contype[_g]) != 1 or int(_hm.geom_conaffinity[_g]) != 1:
              _nct += 1
        print(f"[collfix] hand geoms whose contype/conaffinity is not 1/1: {_nct} "
              f"(the XML isolates every link2 with 2/2)", flush=True)

        _bn2shapes = {}
        for _si in range(len(scene.shape_body)):
          _bi = int(scene.shape_body[_si])
          if _bi < 0:
            continue
          _lbl = scene.body_label[_bi] if hasattr(scene, "body_label") else ""
          _bn2shapes.setdefault(str(_lbl).split("/")[-1], []).append(_si)
        _nfilt = 0
        for _n1, _n2 in _pairs:
          for _s1 in _bn2shapes.get(_n1.split("/")[-1], []):
            for _s2 in _bn2shapes.get(_n2.split("/")[-1], []):
              scene.add_shape_collision_filter_pair(_s1, _s2)
              _nfilt += 1
        print(f"[collfix] {_nfilt} shape-pair filters added from {len(_pairs)} body pairs",
              flush=True)
        _wp = [(a, b) for a, b in _pairs if "wrist" in a and "palm" in b
               or "wrist" in b and "palm" in a]
        print(f"[collfix] wrist/palm body pairs in the rule: {len(_wp)} -> {_wp[:4]}", flush=True)
        _lbls = sorted(_bn2shapes.keys())
        print(f"[collfix] sample of Newton body labels seen: {_lbls[:6]}", flush=True)
        _miss = [p for p in _pairs
                 if not _bn2shapes.get(p[0].split("/")[-1]) or not _bn2shapes.get(p[1].split("/")[-1])]
        print(f"[collfix] body pairs that matched NO shapes: {len(_miss)} of {len(_pairs)}"
              + (f"  e.g. {_miss[:3]}" if _miss else ""), flush=True)
        if _nfilt == 0:
          raise RuntimeError("HAND_COLLISION_FIX matched no shapes; the body-name lookup is wrong, "
                             "and running on would silently leave the model unchanged")



      # Swap the object's collider before replicating, so every world gets it. mjlab authors the object
      # as a 4 cm sphere and its own mesh path uses the *_cir160 convex hull instead of the real shape.
      self._ref_mj_pre = mujoco.MjModel.from_xml_path(xml)
      if sdf_object_stl:
        from grab_objects import swap_collider_to_sdf
        swap_collider_to_sdf(scene, self._ref_mj_pre, f"{object_entity}/{object_entity}",
                             sdf_object_stl, resolution=sdf_resolution,
                             hydroelastic=((sdf_hydroelastic or native_contacts) and hydro_object_table))

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
      return scene

    world = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(world)
    world.default_shape_cfg.gap = 0.0

    # One entry per clip. --sdf-objects wins; --sdf-object stays the single-clip spelling.
    _stls = list(sdf_object_stls) if sdf_object_stls else [sdf_object_stl]
    self.clip_object_stls = _stls
    self.n_clips = len(_stls)
    if clip_env_counts is None and self.num_envs % self.n_clips != 0:
      raise ValueError(
        f"num_envs {self.num_envs} must divide by the clip count {self.n_clips} for the default "
        "equal split; pass --clip-env-counts to allocate unevenly on purpose")
    _per_clip = self.num_envs // self.n_clips

    # replicate() appends worlds, so clip k owns worlds [k*_per_clip, (k+1)*_per_clip). That is a
    # BLOCK layout; mjlab's own default is round-robin (arange % n_clips), so the mapping is handed
    # to it explicitly further down rather than left to agree by luck.
    # Group clips by object first: worlds are laid out one object-block at a time, so a clip can
    # later be moved to any env whose world already carries its mesh.
    self.clip_stls = list(_stls)
    _order: list[str] = []
    for _stl in _stls:
      if (_stl or "") not in _order:
        _order.append(_stl or "")
    self.object_order = _order
    self.clip_object = np.array([_order.index(x or "") for x in _stls], dtype=np.int64)
    self.object_clips = {oi: [c for c in range(self.n_clips) if self.clip_object[c] == oi]
                         for oi in range(len(_order))}

    # Envs per object = its clips' share. Equal split to start with; the quota and the
    # failure-weighted rule below only ever move envs BETWEEN clips of the same object, so these
    # per-object totals stay fixed for the life of the run.
    if clip_env_counts is not None:
      _counts = np.asarray(clip_env_counts, dtype=np.int64)
      if _counts.shape != (self.n_clips,):
        raise ValueError(f"--clip-env-counts has {_counts.size} entries for {self.n_clips} clip(s)")
      if int(_counts.sum()) != self.num_envs:
        raise ValueError(f"--clip-env-counts sums to {int(_counts.sum())}, not --num-envs "
                         f"{self.num_envs}")
      if (_counts <= 0).any():
        raise ValueError("--clip-env-counts must give every clip at least one env; a clip with "
                         "zero envs produces no gradient and no metric, which reads as 'not "
                         "learning' rather than 'not trained'")
    else:
      _counts = np.full(self.n_clips, _per_clip, dtype=np.int64)

    self.object_env_count = np.zeros(len(_order), dtype=np.int64)
    for _c in range(self.n_clips):
      self.object_env_count[self.clip_object[_c]] += int(_counts[_c])

    _scene_cache: dict = {}
    for _oi, _key in enumerate(_order):
      if _key not in _scene_cache:
        _scene_cache[_key] = _author_scene(_key or None)
      world.replicate(_scene_cache[_key], world_count=int(self.object_env_count[_oi]))
    if self.n_clips > 1:
      print(f"[newton-env] MIX: {self.n_clips} clips over {len(_order)} object(s); "
            f"world blocks {self.object_env_count.tolist()} for "
            f"{[os.path.basename(x or 'sphere') for x in _order]}")

    # First env index of each object's block, so a clip can only ever be placed on a world whose
    # mesh matches it.
    self.object_env_start = np.concatenate([[0], np.cumsum(self.object_env_count)[:-1]])

    # Equal counts to start; _apply_clip_counts writes the per-env assignment.
    self.clip_env_count = _counts.copy()
    self.clip_id_np = np.zeros(self.num_envs, dtype=np.int64)
    self._write_clip_layout()

    # PMCP: env-per-clip is re-derived from each clip's own success, on a rollout boundary.
    self.pmcp_every = int(os.environ.get("MIX_PMCP_EVERY", "0"))      # control steps; 0 = off
    self.pmcp_quota = int(os.environ.get("MIX_PMCP_QUOTA", "0"))      # 0 = equal-share floor
    self.pmcp_tau = float(os.environ.get("MIX_PMCP_TAU", "8.0"))
    self.pmcp_ema = float(os.environ.get("MIX_PMCP_EMA", "0.1"))
    self.pmcp_metric = os.environ.get("MIX_PMCP_METRIC", "PhaseA/lift_success")
    # Graduation bars. A clip releases environments once it holds a bar for `hold` windows.
    self.grad_contact_bar = float(os.environ.get("MIX_GRAD_CONTACT", "0.2"))
    self.grad_lift_bar = float(os.environ.get("MIX_GRAD_LIFT", "0.1"))
    self.grad_hold = int(os.environ.get("MIX_GRAD_HOLD", "3"))
    # Windows to let the per-clip averages settle before any promotion is allowed. Without it a
    # transient in the opening windows promotes a clip permanently: the very first mixed run
    # promoted two clips at contact 0.0988 and 0.0249 against a 0.2 bar, and because a stage never
    # falls back the hammer clip trained on half its environments for the rest of the run.
    self.grad_warmup = int(os.environ.get("MIX_GRAD_WARMUP", "3"))
    self.grad_release = float(os.environ.get("MIX_GRAD_RELEASE", "0.5"))  # fraction given up per stage
    self._grad_contact = np.zeros(self.n_clips, dtype=np.float64)
    self._grad_lift = np.zeros(self.n_clips, dtype=np.float64)
    self._grad_hold_c = np.zeros(self.n_clips, dtype=np.int64)
    self._grad_hold_l = np.zeros(self.n_clips, dtype=np.int64)
    self._grad_stage = np.zeros(self.n_clips, dtype=np.int64)   # 0 none, 1 contact, 2 lift
    self._grad_windows = 0
    self._pmcp_success = np.zeros(self.n_clips, dtype=np.float64)
    self._pmcp_seen = False
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
          _max_tri = int(os.environ.get("MAX_TRI_PAIRS", "12000000"))
          # was 4_000_000; see the stall described above
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
    if any(_stls):
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
    if object_solref and any(_stls):
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

    # --- eval-only contact overrides -------------------------------------------------------
    # Both are written the same way the object solref override is: into the compiled mj_model AND
    # pushed into the warp model the solver actually integrates. Training sets neither; they exist
    # to test what a fixed checkpoint's grasp is standing on.
    def _push(_field, _mm_arr):
      import warp as _w
      _t = _w.to_torch(getattr(self.solver.mjw_model, _field))
      _src = _w.to_torch(_w.array(_mm_arr, dtype=float))
      if _t.shape == _src.shape:
        _t[:] = _src
      elif _t.dim() == _src.dim() + 1 and _t.shape[1:] == _src.shape:
        _t[:] = _src.unsqueeze(0)
      else:
        raise RuntimeError(f"cannot push {_field}: mjw {tuple(_t.shape)} vs mj {tuple(_src.shape)}")
      return tuple(_t.shape)

    _ofr = os.environ.get("OBJECT_FRICTION", "").strip()
    if _ofr:
      import mujoco as _mjo
      _mmo = self.solver.mj_model
      _f = float(_ofr); _n = 0
      for _g in range(_mmo.ngeom):
        if "apple" not in (_mjo.mj_id2name(_mmo, _mjo.mjtObj.mjOBJ_GEOM, _g) or ""):
          continue
        _mmo.geom_friction[_g][0] = _f; _n += 1
      if _n == 0:
        raise RuntimeError("OBJECT_FRICTION matched no object geom")
      _sh = _push("geom_friction", _mmo.geom_friction)
      print(f"[newton-env] OBJECT_FRICTION -> slide={_f} on {_n} geom(s), mjw shape {_sh}", flush=True)

    # --- eliminating penetration outright, rather than making the contact harder ------------
    # Stiffening solref alone was measured to change mean penetration by 0.05 mm while tripling the
    # normal force: a soft-constraint solver at a fixed step will always let the bodies overlap,
    # because the restoring force is a FUNCTION of that overlap. The three knobs that actually
    # bound it are: solimp (how fast impedance rises to 1 near contact), priority (whether the
    # object's tuned parameters win outright instead of being averaged with the hand's defaults),
    # and margin (a standoff: the constraint switches on this far BEFORE the surfaces meet, so the
    # visible geometry never interpenetrates even if the constraint surface does).
    _hsi = os.environ.get("HAND_SOLIMP", "").strip()
    if _hsi:
      import mujoco as _mjs
      _vs = [float(x) for x in _hsi.replace(" ", "").split(",")]
      _mms = self.solver.mj_model
      _n4 = 0
      for _g in range(_mms.ngeom):
        _bn = _mjs.mj_id2name(_mms, _mjs.mjtObj.mjOBJ_BODY, int(_mms.geom_bodyid[_g])) or ""
        if ("finger" not in _bn) and ("palm" not in _bn) and ("hand" not in _bn):
          continue
        _mms.geom_solimp[_g][:len(_vs)] = _vs; _n4 += 1
      if _n4 == 0:
        raise RuntimeError("HAND_SOLIMP matched no hand geom")
      print(f"[newton-env] HAND_SOLIMP -> {_vs} on {_n4} hand geom(s), "
            f"mjw shape {_push('geom_solimp', _mms.geom_solimp)}", flush=True)

    _omg = os.environ.get("OBJECT_MARGIN", "").strip()
    if _omg:
      import mujoco as _mjm
      _mv = float(_omg)
      _mmm = self.solver.mj_model
      _n5 = 0
      for _g in range(_mmm.ngeom):
        if "apple" not in (_mjm.mj_id2name(_mmm, _mjm.mjtObj.mjOBJ_GEOM, _g) or ""):
          continue
        _mmm.geom_margin[_g] = _mv; _n5 += 1
      if _n5 == 0:
        raise RuntimeError("OBJECT_MARGIN matched no object geom")
      print(f"[newton-env] OBJECT_MARGIN -> {_mv*1000:.1f} mm standoff on {_n5} geom(s), "
            f"mjw shape {_push('geom_margin', _mmm.geom_margin)}", flush=True)

    _ffl = os.environ.get("FINGER_FORCE_LIMIT", "").strip()
    if _ffl:
      import mujoco as _mjf2
      _lim = float(_ffl)
      _mmf2 = self.solver.mj_model
      _n_act = 0
      _before = 0.0
      _fing_ids = []
      for _a in range(_mmf2.nu):
        _jn = _mjf2.mj_id2name(_mmf2, _mjf2.mjtObj.mjOBJ_JOINT,
                               int(_mmf2.actuator_trnid[_a][0])) or ""
        if ("finger" not in _jn) and ("thumb" not in _jn):
          continue
        _an = _mjf2.mj_id2name(_mmf2, _mjf2.mjtObj.mjOBJ_ACTUATOR, _a) or ""
        _kp_a = float(_mmf2.actuator_gainprm[_a][0])
        if ("unused" in _an) or (_kp_a < 10.0):
          continue                      # the disabled XML motor; leave it inert
        _before = max(_before, float(abs(_mmf2.actuator_forcerange[_a][1])))
        _mmf2.actuator_forcelimited[_a] = 1
        _mmf2.actuator_forcerange[_a][0] = -_lim
        _mmf2.actuator_forcerange[_a][1] = _lim
        _fing_ids.append(_a)
        _n_act += 1
      if _n_act == 0:
        raise RuntimeError("FINGER_FORCE_LIMIT matched no finger actuator")
      import warp as _wfl, torch as _tfl
      _idx = _tfl.tensor(_fing_ids, dtype=_tfl.long)
      _mw0 = self.solver.mjw_model
      _tr0 = _wfl.to_torch(_mw0.actuator_forcerange)
      _pick = _fing_ids[0]
      _b4 = _tr0[0, _pick] if _tr0.dim() == 3 else _tr0[_pick]
      print(f"[newton-env] finger actuator {_pick} forcerange BEFORE: "
            f"{[round(float(x), 4) for x in _b4]}", flush=True)
      _mw = self.solver.mjw_model
      _tr = _wfl.to_torch(_mw.actuator_forcerange)
      _tl = _wfl.to_torch(_mw.actuator_forcelimited) if hasattr(_mw, "actuator_forcelimited") \
          else None
      _idx = _idx.to(_tr.device)
      if _tr.dim() == 3:            # (nworld, nu, 2)
        _tr[:, _idx, 0] = -_lim
        _tr[:, _idx, 1] = _lim
      else:                          # (nu, 2)
        _tr[_idx, 0] = -_lim
        _tr[_idx, 1] = _lim
      # forcelimited is left alone: the fingers already carry it, and the disabled
      # xml_motor_unused_* actuators in the same name match are meant to stay inert.
      _sh_a = tuple(_tr.shape)
      _af = _tr[0, _pick] if _tr.dim() == 3 else _tr[_pick]
      print(f"[newton-env] finger actuator {_pick} forcerange AFTER : "
            f"{[round(float(x), 4) for x in _af]}", flush=True)
      # An arm actuator, printed as the canary: it must keep whatever range it already had.
      _arm = [a for a in range(_mmf2.nu)
              if "elbow" in (_mjf2.mj_id2name(_mmf2, _mjf2.mjtObj.mjOBJ_JOINT,
                                              int(_mmf2.actuator_trnid[a][0])) or "")]
      if _arm:
        _av = _tr[0, _arm[0]] if _tr.dim() == 3 else _tr[_arm[0]]
        print(f"[newton-env] arm actuator forcerange after the write: "
              f"{[round(float(x), 3) for x in _av]} (must not be [0, 0])", flush=True)
      print(f"[newton-env] FINGER_FORCE_LIMIT {_lim} N*m on {_n_act} finger actuator(s) "
            f"(was up to {_before} N*m); at a 30 mm lever that is {_lim/0.03:.0f} N, against "
            f"4.2 N to hold the object; mjw shape {_sh_a}", flush=True)

    _hfr = os.environ.get("HAND_FRICTION", "").strip()
    if _hfr:
      # MuJoCo combines the two geoms' friction with an element-wise MAXIMUM when their priorities
      # are equal, so lowering the object's friction alone cannot lower the contact's friction --
      # the hand's 1.0 still wins. An earlier sweep set only OBJECT_FRICTION and read "friction
      # does not matter" out of a sweep that never changed friction. Both sides have to move.
      import mujoco as _mjhf
      _fh = float(_hfr)
      _mmhf = self.solver.mj_model
      _n3 = 0
      for _g in range(_mmhf.ngeom):
        _bn = _mjhf.mj_id2name(_mmhf, _mjhf.mjtObj.mjOBJ_BODY, int(_mmhf.geom_bodyid[_g])) or ""
        if ("finger" not in _bn) and ("palm" not in _bn) and ("hand" not in _bn):
          continue
        _mmhf.geom_friction[_g][0] = _fh; _n3 += 1
      if _n3 == 0:
        raise RuntimeError("HAND_FRICTION matched no hand geom")
      _sh3 = _push("geom_friction", _mmhf.geom_friction)
      print(f"[newton-env] HAND_FRICTION -> slide={_fh} on {_n3} hand geom(s), mjw shape {_sh3}",
            flush=True)

    _hsr = os.environ.get("HAND_SOLREF", "").strip()
    if _hsr:
      # The object's solref was tuned so it rests on the table with 0.04mm penetration, but the
      # FINGER geoms kept the scene default. MuJoCo mixes the two sides, so the softer finger wins
      # and the hand sinks into the object -- measured 1.2mm mean, 8.7mm worst on the hammer.
      import mujoco as _mjh
      _vals = [float(x) for x in _hsr.replace(" ", "").split(",")]
      _mmh = self.solver.mj_model
      _n2 = 0
      for _g in range(_mmh.ngeom):
        _bn = _mjh.mj_id2name(_mmh, _mjh.mjtObj.mjOBJ_BODY, int(_mmh.geom_bodyid[_g])) or ""
        if ("finger" not in _bn) and ("palm" not in _bn) and ("hand" not in _bn):
          continue
        _mmh.geom_solref[_g][:len(_vals)] = _vals; _n2 += 1
      if _n2 == 0:
        raise RuntimeError("HAND_SOLREF matched no hand geom")
      _sh2 = _push("geom_solref", _mmh.geom_solref)
      print(f"[newton-env] HAND_SOLREF -> {_vals} on {_n2} hand geom(s), mjw shape {_sh2}", flush=True)

    # Geom masks for the penetration metric, built once: mj_id2name over ngeom every step would
    # cost more than the statistic is worth.
    self._pen_log_every = int(os.environ.get("PEN_LOG_EVERY", "4"))
    self._pen_on = os.environ.get("PEN_LOG", "1").strip() not in ("", "0")
    self._pen_tick = 0
    if self._pen_on:
      import mujoco as _mjpl, torch as _tpl
      _mmpl = self.solver.mj_model
      _og, _rg = [], []
      for _g in range(_mmpl.ngeom):
        _gn = _mjpl.mj_id2name(_mmpl, _mjpl.mjtObj.mjOBJ_GEOM, _g) or ""
        _bn = _mjpl.mj_id2name(_mmpl, _mjpl.mjtObj.mjOBJ_BODY, int(_mmpl.geom_bodyid[_g])) or ""
        if "apple" in _gn:
          _og.append(_g)
        # Whitelist on the robot. A blacklist on "world" matches "scene_worldbody", which every
        # compiled body name contains, and silently discards every finger.
        elif "robot" in _bn:
          _rg.append(_g)
      if not _og or not _rg:
        print(f"[pen] disabled: {len(_og)} object geom(s), {len(_rg)} robot geom(s)", flush=True)
        self._pen_on = False
      else:
        _ng = int(_mmpl.ngeom)
        self._pen_obj = _tpl.zeros(_ng, dtype=_tpl.bool, device=device)
        self._pen_rob = _tpl.zeros(_ng, dtype=_tpl.bool, device=device)
        self._pen_obj[_tpl.tensor(_og, device=device)] = True
        self._pen_rob[_tpl.tensor(_rg, device=device)] = True
        print(f"[pen] penetration logging on: {len(_og)} object geom(s) vs {len(_rg)} robot "
              f"geom(s), sampled every {self._pen_log_every} steps", flush=True)

    self._env = NewtonEnv(self.solver.mj_model, self.solver.mjw_data, self.num_envs, device,
                          control=self.control, rename_from=self._ref_mj,
                          physics_dt=self.physics_dt, decimation=self.decimation,
                          solver=self.solver, object_entity=object_entity)
    self._env.forward()

    # Hand mjlab the block layout replicate() actually produced. Its own default is round-robin, so
    # leaving this unset would pair each env's object with a different clip's reference silently.
    # _clip_id() reads this attribute and only computes its own when it is missing.
    import torch as _torch
    self.clip_id = _torch.as_tensor(self.clip_id_np, device=device, dtype=_torch.long)
    self._env._reference_clip_id = self.clip_id
    if self.n_clips > 1:
      from mjlab.tasks.apple_eat import mdp as _amdp
      _got = _amdp._clip_id(self._env)
      if not _torch.equal(_got, self.clip_id):
        raise RuntimeError(
          "mjlab did not take the clip assignment handed to it: "
          f"{_got[:8].tolist()} vs {self.clip_id[:8].tolist()}")
      _ref = _amdp._ref(str(device))
      _n = int(_ref.get("n_clips", 1))
      if _n != self.n_clips:
        raise RuntimeError(
          f"{self.n_clips} object mesh(es) but the reference carries {_n} clip(s); "
          "--sdf-objects and APPLE_EAT_PKL_MIX must list the same clips in the same order")
      _counts = [int((self.clip_id == c).sum()) for c in range(self.n_clips)]
      print(f"[newton-env] MIX: clip->env assignment verified against mjlab, blocks {_counts}")
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
      # The guard is about having a real collider at all, which in mixed training comes from
      # --sdf-objects rather than the singular flag.
      if not all(_stls):
        raise RuntimeError("--table-under-object needs every clip's object mesh (--sdf-object or "
                           "--sdf-objects) to know where the real collider bottom is")
      from newton_table import install as _install_table
      _ref_pkl = os.environ.get("APPLE_EAT_PKL")
      if not _ref_pkl:
        raise RuntimeError("--table-under-object needs APPLE_EAT_PKL: the object's resting height "
                           "is read from the reference clip, not guessed")
      # One resting height per object. A single shift would leave every clip but one either
      # hovering above the table or starting inside it -- silently.
      _mix_env = os.environ.get("APPLE_EAT_PKL_MIX", "").strip()
      _pkls = [x.strip() for x in _mix_env.split(",") if x.strip()] if _mix_env else [_ref_pkl]
      if len(_pkls) != self.n_clips:
        raise RuntimeError(f"{self.n_clips} object mesh(es) but {len(_pkls)} clip(s) in "
                           "APPLE_EAT_PKL_MIX")
      _install_table(self.solver.mj_model, _stls, _pkls,
                     z_offset=float(os.environ.get("APPLE_SCENE_Z_OFFSET", 0.0)),
                     clip_id=(self.clip_id_np if self.n_clips > 1 else None))

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

  def _write_clip_layout(self) -> None:
    """Lay `clip_env_count` out over the env indices, inside each object's own block."""
    for oi, clips in self.object_clips.items():
      cursor = int(self.object_env_start[oi])
      for c in clips:
        n = int(self.clip_env_count[c])
        self.clip_id_np[cursor:cursor + n] = c
        cursor += n
      expected = int(self.object_env_start[oi] + self.object_env_count[oi])
      if cursor != expected:
        raise RuntimeError(
          f"object {oi} block holds {self.object_env_count[oi]} envs but its clips sum to "
          f"{cursor - int(self.object_env_start[oi])}; the two must agree or a clip would land on "
          "a world carrying a different mesh")

  def set_clip_counts(self, counts) -> None:
    """Reassign envs between clips of the SAME object. No rebuild: the mesh in each world is
    untouched, only which reference row an env reads."""
    import numpy as _np
    import torch as _torch

    counts = _np.asarray(counts, dtype=_np.int64)
    if counts.shape != (self.n_clips,):
      raise ValueError(f"expected {self.n_clips} counts, got {counts.shape}")
    for oi, clips in self.object_clips.items():
      if int(counts[clips].sum()) != int(self.object_env_count[oi]):
        raise ValueError(
          f"clips of object {oi} were given {int(counts[clips].sum())} envs but that object's "
          f"world block holds {int(self.object_env_count[oi])}; envs cannot move between objects "
          "without rebuilding the scene")
    self.clip_env_count = counts
    self._write_clip_layout()
    self.clip_id = _torch.as_tensor(self.clip_id_np, device=self.device, dtype=_torch.long)
    self._env._reference_clip_id = self.clip_id
    # The start frame is derived from the clip and cached on the env; drop it so it is rebuilt.
    self._env._reference_start_frame = None
    self._reset_idx(self._all)

  def pmcp_reallocate(self, fail_rate, quota: int = 1, tau: float = 8.0) -> "np.ndarray":
    """Failure-weighted counts, inside each object group, with a floor of `quota` envs per clip.

    The floor is the part that matters: a pure argmax rule hands every spare env to the hardest
    clip and the rest stop being trained at all.
    """
    import numpy as _np

    fail = _np.asarray(fail_rate, dtype=_np.float64).reshape(self.n_clips)
    counts = _np.zeros(self.n_clips, dtype=_np.int64)
    for oi, clips in self.object_clips.items():
      total = int(self.object_env_count[oi])
      if quota * len(clips) > total:
        raise ValueError(f"quota {quota} x {len(clips)} clips exceeds object {oi}'s {total} envs")
      counts[clips] = quota
      spare = total - quota * len(clips)
      if spare <= 0:
        continue
      w = _np.exp(tau * (fail[clips] - fail[clips].max()))
      w = w / w.sum() if w.sum() > 0 else _np.ones(len(clips)) / len(clips)
      extra = _np.floor(w * spare).astype(_np.int64)
      # hand the rounding remainder to the clips with the largest fractional part
      rem = spare - int(extra.sum())
      if rem > 0:
        frac = w * spare - _np.floor(w * spare)
        for idx in _np.argsort(-frac)[:rem]:
          extra[idx] += 1
      counts[clips] += extra
    return counts

  def graduation_counts(self) -> "np.ndarray":
    """Per-clip env counts from the graduation rule, inside each object block.

    A clip at stage k has given up `release^k` of its equal share; whatever is freed inside an
    object block is split evenly among that block's clips that have not graduated. If every clip in
    a block has graduated, the block goes back to an even split -- there is nobody left to help.
    """
    import numpy as _np

    counts = _np.zeros(self.n_clips, dtype=_np.int64)
    for oi, clips in self.object_clips.items():
      total = int(self.object_env_count[oi])
      quota = max(1, int(self.pmcp_quota) if self.pmcp_quota > 0 else total // (4 * len(clips)))
      quota = min(quota, total // len(clips))
      share = total // len(clips)
      held, freed = {}, 0
      for c in clips:
        keep = int(round(share * (self.grad_release ** int(self._grad_stage[c]))))
        keep = max(keep, quota)
        held[c] = keep
        freed += share - keep
      needy = [c for c in clips if self._grad_stage[c] == 0]
      if needy and freed > 0:
        per = freed // len(needy)
        for c in needy:
          held[c] += per
        held[needy[0]] += freed - per * len(needy)
      else:
        # Nobody to give to, because every clip in this object block has already graduated. Hand
        # the freed environments back evenly across the block: piling them all on clips[0] left a
        # two-clip block at 96/32 for no reason anyone chose, which is a curriculum decision made
        # by an accident of ordering.
        per_back = freed // len(clips)
        for c in clips:
          held[c] += per_back
        held[clips[0]] += freed - per_back * len(clips)
      for c in clips:
        counts[c] = held[c]
      drift = total - int(sum(counts[c] for c in clips))
      counts[clips[0]] += drift
    return counts

  def _graduation_update(self) -> bool:
    """Advance each clip's stage from the metrics. Returns True if any stage changed."""
    import numpy as _np

    log = self.extras.get("log") or {}
    changed = False
    a = self.pmcp_ema
    self._grad_windows += 1
    for c in range(self.n_clips):
      cv = log.get(f"Stage/physical_contact/clip{c}")
      lv = log.get(f"PhaseA/lift_success/clip{c}")
      if cv is not None:
        self._grad_contact[c] = (1 - a) * self._grad_contact[c] + a * float(cv)
      if lv is not None:
        self._grad_lift[c] = (1 - a) * self._grad_lift[c] + a * float(lv)
      # hold counters -- a bar has to be met on consecutive windows, and a clip never falls back:
      # returning environments to a clip that dipped is what makes the allocation oscillate.
      # Because a stage is permanent, the counters stay at zero through the warm-up: an average
      # that has not settled yet must not be able to spend a decision that cannot be taken back.
      if self._grad_windows <= self.grad_warmup:
        self._grad_hold_c[c] = 0
        self._grad_hold_l[c] = 0
        continue
      self._grad_hold_c[c] = self._grad_hold_c[c] + 1 if self._grad_contact[c] >= self.grad_contact_bar else 0
      self._grad_hold_l[c] = self._grad_hold_l[c] + 1 if self._grad_lift[c] >= self.grad_lift_bar else 0
      stage = int(self._grad_stage[c])
      if stage < 1 and self._grad_hold_c[c] >= self.grad_hold:
        self._grad_stage[c] = 1; changed = True
      if stage < 2 and self._grad_hold_l[c] >= self.grad_hold:
        self._grad_stage[c] = 2; changed = True
    return changed

  def _pmcp_update(self) -> None:
    """Track per-clip success, and on a rollout boundary re-derive the env split from it."""
    if os.environ.get("MIX_PMCP_RULE", "graduation") == "graduation":
      # The window boundary gates the UPDATE, not just the reallocation. Calling it every step
      # made `grad_hold` count steps, so three consecutive steps over the bar -- a transient the
      # averages have not even absorbed yet -- promoted a clip for good.
      if self.common_step_counter % self.pmcp_every != 0:
        return
      changed = self._graduation_update()
      counts = self.graduation_counts()
      moved = not np.array_equal(counts, self.clip_env_count)
      if not (changed or moved):
        return
      before = self.clip_env_count.tolist()
      if moved:
        self.set_clip_counts(counts)
      # A stage change that frees no environments -- a clip whose object block holds only itself,
      # so there is nobody to hand them to -- still gets printed. Returning silently here is what
      # let a promotion look like "the rule never fired".
      tail = "" if moved else "  (stage only; no environment could be moved inside the object block)"
      print(f"[grad] step {self.common_step_counter} stage={self._grad_stage.tolist()} "
            f"contact={[round(float(x),4) for x in self._grad_contact]} "
            f"lift={[round(float(x),4) for x in self._grad_lift]} "
            f"envs {before} -> {counts.tolist()}{tail}", flush=True)
      return

    forced = os.environ.get("MIX_PMCP_FORCE", "").strip()
    if forced:
      # e.g. MIX_PMCP_FORCE="0.9,0.1" -- a known signal, so a test can tell "the rule ran and
      # decided not to move anything" apart from "the rule never ran".
      self._pmcp_success[:] = np.asarray([float(x) for x in forced.split(",")], dtype=np.float64)
      self._pmcp_seen = True
      log = {}
      vals = [None] * self.n_clips
    else:
      log = self.extras.get("log") or {}
      vals = []
      for c in range(self.n_clips):
        key = f"{self.pmcp_metric}/clip{c}"
        v = log.get(key)
        vals.append(float(v) if v is not None else None)
    if all(v is not None for v in vals):
      cur = np.asarray(vals, dtype=np.float64)
      if not self._pmcp_seen:
        self._pmcp_success[:] = cur
        self._pmcp_seen = True
      else:
        a = self.pmcp_ema
        self._pmcp_success[:] = (1.0 - a) * self._pmcp_success + a * cur

    if not self._pmcp_seen or self.common_step_counter % self.pmcp_every != 0:
      return

    quota = self.pmcp_quota
    if quota <= 0:
      # Default floor: half of an equal share. Enough that a clip nobody is failing on keeps
      # producing gradient, small enough to leave most of the budget to reallocate.
      quota = max(1, int(self.num_envs // self.n_clips // 2))
    fail = 1.0 - self._pmcp_success
    counts = self.pmcp_reallocate(fail, quota=quota, tau=self.pmcp_tau)
    if np.array_equal(counts, self.clip_env_count):
      return
    before = self.clip_env_count.tolist()
    self.set_clip_counts(counts)
    print(f"[pmcp] step {self.common_step_counter}  success="
          f"{[round(float(x), 4) for x in self._pmcp_success]}  "
          f"envs {before} -> {counts.tolist()}", flush=True)

  def get_observations(self, update_history: bool = False):
    if self.observation_manager is not None:
      # update_history=True must happen exactly once per control step: compute() caches, and the
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

    if getattr(self, "_pen_on", False):
      self._pen_tick += 1
      if self._pen_tick % self._pen_log_every == 0:
        import warp as _wpl, torch as _tpl2
        _c = self.solver.mjw_data.contact
        _gm = _wpl.to_torch(_c.geom)
        _ds = _wpl.to_torch(_c.dist)
        _a = _gm[:, 0].long().clamp(min=0)
        _b = _gm[:, 1].long().clamp(min=0)
        _sel = ((self._pen_obj[_a] & self._pen_rob[_b])
                | (self._pen_rob[_a] & self._pen_obj[_b]))
        _dep = (-_ds).clamp(min=0.0) * _sel.float()
        _n = int(_sel.sum())
        # Published every sampled step, empty or not: it is the difference between "the hand did
        # not touch the object" and "this code did not run".
        _log0 = self._env.extras.setdefault("log", {})
        _log0["Penetration/sampled_contacts"] = float(_n)
        if _n:
          _pos = _dep[_sel]
          _log = _log0
          _log["Penetration/mean_mm"] = float(_pos.mean()) * 1000.0
          _log["Penetration/max_mm"] = float(_pos.max()) * 1000.0
          _log["Penetration/frac_over_1mm"] = float((_pos > 0.001).float().mean())
          _log["Penetration/frac_over_3mm"] = float((_pos > 0.003).float().mean())
          _log["Penetration/frac_over_4mm"] = float((_pos > 0.004).float().mean())
          _log["Penetration/contacts"] = float(_n)
          # Accumulate across the sampled steps and emit one line per ~200 samples, so the number
          # the monitor reads is an average over a stretch of training rather than one step.
          if not hasattr(self, "_pen_acc"):
            self._pen_acc = [0.0, 0.0, 0.0, 0.0, 0.0, 0]
          self._pen_acc[0] += float(_pos.mean()) * 1000.0
          self._pen_acc[1] = max(self._pen_acc[1], float(_pos.max()) * 1000.0)
          self._pen_acc[2] += float((_pos > 0.001).float().mean())
          self._pen_acc[3] += float((_pos > 0.003).float().mean())
          self._pen_acc[4] += float((_pos > 0.004).float().mean())
          self._pen_acc[5] += 1
          # per-world age, broadcast to the contacts of that world
          _wid_c = _wpl.to_torch(_c.worldid).long()[_sel]
          _age = self._env.episode_length_buf[_wid_c.clamp(0, self.num_envs - 1)]
          _fresh = _age <= 3
          if not hasattr(self, "_pen_split"):
            self._pen_split = [0.0, 0, 0.0, 0, 0.0, 0.0]
          if int(_fresh.sum()):
            self._pen_split[0] += float(_pos[_fresh].mean()) * 1000.0
            self._pen_split[1] += 1
            self._pen_split[4] = max(self._pen_split[4], float(_pos[_fresh].max()) * 1000.0)
          _old_m = ~_fresh
          if int(_old_m.sum()):
            self._pen_split[2] += float(_pos[_old_m].mean()) * 1000.0
            self._pen_split[3] += 1
            self._pen_split[5] = max(self._pen_split[5], float(_pos[_old_m].max()) * 1000.0)
          if self._pen_acc[5] >= 200:
            _k = self._pen_acc[5]
            print(f"[pen-stat] mean={self._pen_acc[0]/_k:.4f}mm max={self._pen_acc[1]:.4f}mm "
                  f">1mm={self._pen_acc[2]/_k:.5f} >3mm={self._pen_acc[3]/_k:.5f} "
                  f">4mm={self._pen_acc[4]/_k:.5f} n={_n}", flush=True)
            _ps = self._pen_split
            print(f"[pen-split] within 3 steps of reset: mean="
                  f"{(_ps[0]/_ps[1] if _ps[1] else float('nan')):.4f}mm max={_ps[4]:.4f}mm | "
                  f"settled: mean={(_ps[2]/_ps[3] if _ps[3] else float('nan')):.4f}mm "
                  f"max={_ps[5]:.4f}mm", flush=True)
            self._pen_split = [0.0, 0, 0.0, 0, 0.0, 0.0]
            self._pen_acc = [0.0, 0.0, 0.0, 0.0, 0.0, 0]

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
    if getattr(self, "pmcp_every", 0) > 0 and self.n_clips > 1:
      self._pmcp_update()

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
