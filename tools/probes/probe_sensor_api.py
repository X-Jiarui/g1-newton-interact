"""Can sensors be added to an MjSpec before it is compiled?

Newton has no sensor support -- neither its MJCF importer nor its spec construction mentions them --
but it does build an MjSpec and compile it, and that compile is already wrapped to capture the spec.
Adding the sensors there means MuJoCo's own sensors, compiled by MuJoCo, running in mujoco_warp: no
reimplementation of contact semantics, which is the part that would be easy to get subtly wrong.
"""
import os, inspect
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
XML = os.path.join(HERE, "assets/scene_stapler/scene.xml")

spec = mujoco.MjSpec.from_file(XML)
print("MjSpec sensor API:")
for n in ("add_sensor", "sensors", "sensor"):
    a = getattr(spec, n, None)
    if a is None:
        print(f"  spec.{n}: absent"); continue
    if callable(a):
        try: print(f"  spec.{n}{inspect.signature(a)}")
        except Exception: print(f"  spec.{n}: callable")
    else:
        print(f"  spec.{n}: {type(a).__name__}, len={len(a) if hasattr(a,'__len__') else '?'}")

sensors = list(spec.sensors)
print(f"\nsource spec carries {len(sensors)} sensors")
if sensors:
    s0 = sensors[0]
    fields = [f for f in dir(s0) if not f.startswith("_")]
    print(f"  MjsSensor fields: {fields[:22]}")
    for s in sensors[:2] + sensors[4:7]:
        vals = {f: getattr(s, f, None) for f in ("name", "type", "objtype", "objname",
                                                 "reftype", "refname", "intprm", "datatype",
                                                 "needstage", "cutoff", "dim")}
        vals = {k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in vals.items()}
        print(f"   {vals}")

print(f"\nmjSENS_CONTACT value: {int(mujoco.mjtSensor.mjSENS_CONTACT)}")
