"""Does Newton give us N parallel worlds, or one model with N robots in it?

Training needs thousands of environments stepping together. Newton replicates a scene by adding the
same builder N times; the question is whether SolverMuJoCo then recognises the copies as identical
worlds and uses mujoco_warp's nworld batching -- data shaped (N, nq) -- or builds one giant model
with N robots in it, shaped (1, N*nq). The first scales; the second does not, and it also breaks
every index map the bridge relies on.
"""
import os, sys, time, numpy as np, mujoco
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
import mjw_compat; mjw_compat.apply()
import newton, warp as wp
from newton.solvers import SolverMuJoCo
from newton_simple_fix import capture_spec, restore_simple_bodies, restore_freejoint_damping

XML = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets/mjlab_scene/scene.xml")
ref = mujoco.MjModel.from_xml_path(XML)
print(f"single-scene reference: nq={ref.nq} nv={ref.nv} nbody={ref.nbody}")

for N in (1, 4, 16):
    t0 = time.time()
    scene = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(scene)
    scene.default_shape_cfg.gap = 0.0
    scene.add_mjcf(XML, collapse_fixed_joints=False, parse_mujoco_options=True)

    if N == 1:
        world = scene
    else:
        world = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(world)
        world.default_shape_cfg.gap = 0.0
        # replicate() is Newton's parallel-environment API: one authored scene, world_count copies
        world.replicate(scene, world_count=N)

    m = world.finalize()
    with capture_spec() as cap:
        sv = SolverMuJoCo(m, enable_multiccd=True, update_data_interval=0,
                          njmax=2048, nconmax=256)
    qpos = wp.to_torch(sv.mjw_data.qpos)
    print(f"\nN={N}: builder bodies={world.body_count}  model nq={sv.mj_model.nq} "
          f"nbody={sv.mj_model.nbody}")
    print(f"   mjw_data.qpos shape = {tuple(qpos.shape)}   built in {time.time()-t0:.1f}s")
    batched = qpos.shape[0] == N and qpos.shape[1] == ref.nq
    giant = qpos.shape[0] == 1 and qpos.shape[1] >= N * ref.nq
    print(f"   -> {'BATCHED worlds (scales)' if batched else 'ONE GIANT WORLD (does not scale)' if giant else 'unexpected layout'}")
    if N > 1 and batched:
        restore_freejoint_damping(cap.spec, XML, verbose=False)
        r = restore_simple_bodies(sv, cap.spec, nworld=N, nconmax=256, njmax=2048, verbose=False)
        print(f"   simple-fix under nworld={N}: nC {r['nC_before']} -> {r['nC_after']} "
              f"(target {ref.nC})   qpos {tuple(wp.to_torch(sv.mjw_data.qpos).shape)}")
