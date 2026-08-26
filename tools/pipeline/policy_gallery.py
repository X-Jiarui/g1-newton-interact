"""Viser gallery of policy rollouts: one dropdown entry per training run on this box.

Each entry is a trace recorded by `train_newton.py --rollout-steps ... --dump-qpos ...`, which
rolls the run's own checkpoint out through the env the run was TRAINED with -- same contact
recipe, same object mesh, same table, same reference. That matters: an eval that re-derives those
flags scores the policy in a scene it never saw.

Playback replays the recorded qpos through mjlab's FULL model (visual meshes included) rather than
Newton's converted one, which keeps only colliding geoms and would draw a robot of bare primitives.
The physics on show is still whichever simulator produced the qpos -- this only borrows appearance.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import mujoco, trimesh, viser

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True, help="JSON list of {name,npz,stl,info}")
ap.add_argument("--port", type=int, default=8080)
ap.add_argument("--label", default="policy rollouts")
ap.add_argument("--fps", type=float, default=50.0)
A = ap.parse_args()

ENTRIES = json.load(open(A.manifest))
ENTRIES = [e for e in ENTRIES if os.path.exists(e["npz"])]
if not ENTRIES:
    raise SystemExit("no traces exist yet")
by_name = {e["name"]: e for e in ENTRIES}
names = [e["name"] for e in ENTRIES]


def model_with_object_mesh(xml: str, stl: str | None):
    """Compile the render scene, drawing the object as the mesh the physics actually collided with.

    The scene authors the object as a sphere placeholder; the true mesh only ever reached the
    collider. Rendering the placeholder shows a ball where there was a hammer.
    """
    if not stl:
        return mujoco.MjModel.from_xml_path(xml)
    spec = mujoco.MjSpec.from_file(xml)
    target = None
    for body in spec.bodies:
        for g in body.geoms:
            if g.name.endswith("apple_geom"):
                target = g
                break
    if target is None:
        raise SystemExit("no object geom named *apple_geom in the render scene")
    mesh = spec.add_mesh()
    mesh.name = "render_object_mesh"
    mesh.file = os.path.abspath(stl)
    target.type = mujoco.mjtGeom.mjGEOM_MESH
    target.meshname = "render_object_mesh"
    target.size[:] = (0.0, 0.0, 0.0)
    return spec.compile()


def geom_mesh(model, gid: int):
    t, s = model.geom_type[gid], model.geom_size[gid]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = model.geom_dataid[gid]
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        return (model.mesh_vert[va:va + vn].astype(np.float32),
                model.mesh_face[fa:fa + fn].astype(np.int32))
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        m = trimesh.creation.icosphere(subdivisions=2, radius=float(s[0]))
    elif t == mujoco.mjtGeom.mjGEOM_BOX:
        m = trimesh.creation.box(extents=2.0 * s[:3])
    elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        m = trimesh.creation.capsule(radius=float(s[0]), height=2.0 * float(s[1]))
    elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        m = trimesh.creation.cylinder(radius=float(s[0]), height=2.0 * float(s[1]))
    else:
        return None
    return np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int32)


def mat2wxyz(R: np.ndarray) -> np.ndarray:
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R.reshape(9)))
    return q


def mocap_slot(model, name: str) -> int:
    """Mocap index in the render model for a body recorded under `name`.

    Newton rewrites body names on import, so the match is on the flattened suffix and must be
    unique: guessing an index is how the table once ended up under the robot's feet.
    """
    want = name.replace("/", "_")
    hits = []
    for b in range(model.nbody):
        if model.body_mocapid[b] < 0:
            continue
        n = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").replace("/", "_")
        if n.endswith(want) or want.endswith(n):
            hits.append(int(model.body_mocapid[b]))
    if len(hits) != 1:
        raise SystemExit(f"mocap body {name!r} matched {len(hits)} bodies; trace and model disagree")
    return hits[0]


CACHE: dict = {}


def load(name: str):
    """Precompute every drawable geom's pose for every frame of one run's trace."""
    if name in CACHE:
        return CACHE[name]
    e = by_name[name]
    model = model_with_object_mesh(e["xml"], e.get("stl"))
    data = mujoco.MjData(model)
    z = np.load(e["npz"], allow_pickle=True)
    qpos = z["qpos"]
    mpos = z["mocap_pos"] if "mocap_pos" in z else None
    mquat = z["mocap_quat"] if "mocap_quat" in z else None
    mnames = [str(x) for x in z["mocap_names"]] if "mocap_names" in z else None
    slots = [mocap_slot(model, n) for n in mnames] if mnames else None

    geoms = []
    for g in range(model.ngeom):
        vf = geom_mesh(model, g)
        if vf is None:
            continue
        rgba = model.geom_rgba[g]
        col = tuple(int(255 * x) for x in rgba[:3]) if rgba[3] > 0 else (200, 200, 205)
        geoms.append({"gid": g, "v": vf[0], "f": vf[1], "color": col})

    n = len(qpos)
    xp = np.zeros((n, len(geoms), 3), np.float32)
    xq = np.zeros((n, len(geoms), 4), np.float32)
    for i in range(n):
        data.qpos[:] = qpos[i]
        if mpos is not None and model.nmocap:
            if slots is not None:
                for src, dst in enumerate(slots):
                    data.mocap_pos[dst] = mpos[i][src]
                    data.mocap_quat[dst] = mquat[i][src]
            else:
                data.mocap_pos[:] = mpos[i]
                data.mocap_quat[:] = mquat[i]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        for k, gg in enumerate(geoms):
            xp[i, k] = data.geom_xpos[gg["gid"]]
            xq[i, k] = mat2wxyz(data.geom_xmat[gg["gid"]])
    CACHE[name] = {"geoms": geoms, "xp": xp, "xq": xq, "n": n, "info": e.get("info", "")}
    print(f"[gallery] loaded {name}: {n} frames, {len(geoms)} geoms", flush=True)
    return CACHE[name]


server = viser.ViserServer(port=A.port, label=A.label)
# Without this the robot is drawn on its side: viser's default up is +y, mujoco's is +z.
server.scene.set_up_direction("+z")

gui_run = server.gui.add_dropdown("run", tuple(names), initial_value=names[0])
gui_play = server.gui.add_checkbox("play", True)
gui_frame = server.gui.add_slider("frame", 0, 1, 1, 0)
gui_speed = server.gui.add_slider("speed", 0.1, 2.0, 0.1, 1.0)
gui_info = server.gui.add_text("info", "")

state = {"cur": None, "frame": 0, "pending": names[0], "handles": []}


def select(name: str):
    c = load(name)
    for h in state["handles"]:
        try:
            h.remove()
        except Exception:
            pass
    state["handles"] = [
        server.scene.add_mesh_simple(f"/g{k}", g["v"], g["f"], color=g["color"])
        for k, g in enumerate(c["geoms"])
    ]
    state["cur"] = c
    state["frame"] = 0
    gui_frame.max = c["n"] - 1
    gui_frame.value = 0
    gui_info.value = c["info"]


def draw(i: int):
    c = state["cur"]
    if c is None:
        return
    xp, xq = c["xp"], c["xq"]
    for k, h in enumerate(state["handles"]):
        try:
            h.position = tuple(float(x) for x in xp[i, k])
            h.wxyz = tuple(float(x) for x in xq[i, k])
        except RuntimeError:
            # the handle was removed by a run switch between iterations; the next draw re-adds it
            return


gui_run.on_update(lambda _: state.__setitem__("pending", gui_run.value))
gui_frame.on_update(lambda _: state.__setitem__("frame", int(gui_frame.value)))

print(f"[gallery] viser on http://localhost:{A.port}  ({len(names)} runs)", flush=True)

while True:
    # Run switches happen on the drawing thread, so a handle is never removed while the animation
    # loop still holds it -- doing it in the GUI callback raced and killed the server.
    if state["pending"] is not None:
        want = state["pending"]
        state["pending"] = None
        select(want)
    if state["cur"] is not None:
        draw(state["frame"])
        if gui_play.value:
            state["frame"] = (state["frame"] + 1) % state["cur"]["n"]
            gui_frame.value = state["frame"]
    time.sleep(1.0 / max(A.fps * float(gui_speed.value), 1.0))
