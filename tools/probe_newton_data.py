"""What state surface does Newton's SolverMuJoCo expose, and can it be read as torch without copying?

mjlab's observation builders read entity .data fields that are ultimately views on mujoco_warp's Data
(qpos, qvel, xpos, xquat). Newton's MuJoCo solver runs the same mujoco_warp underneath, and the two
models are now verified to have identical joint/body/actuator ordering -- so if mjw_data is reachable
and its arrays convert to torch, the adapter is a renaming layer rather than a reimplementation.
"""
import numpy as np, newton, warp as wp
from newton.solvers import SolverMuJoCo

b = newton.ModelBuilder(); SolverMuJoCo.register_custom_attributes(b)
b.default_shape_cfg.gap = 0.0
b.add_mjcf("assets/mjlab_scene/scene.xml", collapse_fixed_joints=False, parse_mujoco_options=True)
m = b.finalize()
s = SolverMuJoCo(m, enable_multiccd=True)

print("solver data-ish attrs:", [a for a in dir(s) if "data" in a.lower() and not a.startswith("__")])
d = getattr(s, "mjw_data", None)
print("mjw_data present:", d is not None)
if d is not None:
    for f in ("qpos", "qvel", "ctrl", "xpos", "xquat", "cvel", "act", "time", "qfrc_actuator"):
        a = getattr(d, f, None)
        if a is None:
            print(f"  {f:14s} ABSENT"); continue
        shp = getattr(a, "shape", None)
        dt = getattr(a, "dtype", None)
        print(f"  {f:14s} shape={shp} dtype={dt}")
    # zero-copy to torch?
    try:
        import torch
        t = wp.to_torch(d.qpos)
        print(f"\n  wp.to_torch(qpos): shape={tuple(t.shape)} dtype={t.dtype} device={t.device}")
        print(f"  shares memory (no copy): {t.data_ptr() == d.qpos.ptr}")
    except Exception as e:
        print(f"  wp.to_torch failed: {type(e).__name__}: {e}")

print("\nsolver step signature:")
import inspect
print(" ", inspect.signature(s.step))
print("\nstate objects:")
print(" ", [a for a in dir(m) if a.startswith("state") or a == "control"][:6])
