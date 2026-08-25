"""Viser gallery of retargeted GRAB reference clips.

Plays the *reference* motion (what the policy is asked to track), not a physics rollout:
robot pose comes from mujoco FK on the scene model, object and table are drawn from the
STL paths recorded inside each pkl, so one scene xml serves every object.
"""
from __future__ import annotations
import argparse, json, os, pickle, threading, time
import numpy as np
import mujoco, trimesh, viser

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True, help="top-N manifest from rank_sequences_by_travel.py")
ap.add_argument("--top", type=int, default=15)
ap.add_argument("--xml", required=True)
ap.add_argument("--port", type=int, default=8110)
A = ap.parse_args()

man = json.load(open(A.manifest))
clips = man if isinstance(man, list) else (man.get("clips") or man.get("top") or list(man.values())[0])
clips = clips[: A.top]

model = mujoco.MjModel.from_xml_path(A.xml)
data = mujoco.MjData(model)
NHINGE = model.nq - 14  # robot free joint (7) + object free joint (7)

# ---- per-geom render meshes in the geom's own frame, robot only -------------------------
def geom_mesh(gid: int):
    t, s = model.geom_type[gid], model.geom_size[gid]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = model.geom_dataid[gid]
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        return model.mesh_vert[va:va+vn].astype(np.float32), model.mesh_face[fa:fa+fn].astype(np.int32)
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

ROBOT_GEOMS = []
for g in range(model.ngeom):
    bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or ""
    if not bname.startswith("robot/"):
        continue
    vf = geom_mesh(g)
    if vf is not None:
        ROBOT_GEOMS.append((g, vf[0], vf[1]))
print(f"robot render geoms: {len(ROBOT_GEOMS)}   hinges: {NHINGE}")

def mat2wxyz(R: np.ndarray) -> np.ndarray:
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R.reshape(9)))
    return q

# ---- load clips -------------------------------------------------------------------------
LOADED = {}
def load(c):
    key = f'{c["subject"]}/{c["sequence"]}'
    if key in LOADED:
        return LOADED[key]
    d = pickle.load(open(c["path"], "rb"))
    r53 = d["robot_53dof"]
    n = int(d["n_frames"])
    qpos = np.zeros((n, model.nq))
    qpos[:, 0:3] = r53["root_pos"]
    # root_rot is stored xyzw (scipy order); mujoco qpos wants wxyz. Reading it straight through
    # lays the robot on its side -- and pelvis HEIGHT does not catch it, since that comes from
    # root_pos alone. Check the pelvis local z axis instead.
    rr = np.asarray(r53["root_rot"])
    qpos[:, 3:7] = np.column_stack([rr[:, 3], rr[:, 0], rr[:, 1], rr[:, 2]])
    dof = np.asarray(r53["dof_pos"])
    qpos[:, 7:7 + min(NHINGE, dof.shape[1])] = dof[:, :NHINGE]
    obj = d["object"]; tab = d["table"]
    LOADED[key] = dict(qpos=qpos, n=n, fps=float(d.get("fps", 30.0)),
                       obj_pos=np.asarray(obj["pos_mj"]), obj_quat=np.asarray(obj["quat_wxyz_mj"]),
                       obj_stl=obj["stl_path"],
                       tab_pos=np.asarray(tab["pos_mj"]), tab_quat=np.asarray(tab["quat_wxyz_mj"]),
                       tab_stl=tab["stl_path"])
    return LOADED[key]

server = viser.ViserServer(port=A.port, label="GRAB reference clips - top travel")
# The scene is Z-up (mujoco/GRAB convention). Viser does not assume that, so without this the
# whole scene renders rotated 90 degrees and the robot reads as lying on its back.
server.scene.set_up_direction("+z")
server.scene.world_axes.visible = True
names = [f'{i+1:02d}. {c["subject"]}/{c["sequence"]}  ({c["object"]}, {c["path_len_m"]:.2f} m)'
         for i, c in enumerate(clips)]
by_name = dict(zip(names, clips))

robot_handles = []
for k, (g, v, f) in enumerate(ROBOT_GEOMS):
    rgba = model.geom_rgba[g]
    col = tuple(int(255 * x) for x in rgba[:3]) if rgba[3] > 0 else (200, 200, 205)
    robot_handles.append(server.scene.add_mesh_simple(f"/robot/g{k}", v, f, color=col))

obj_handle = {"h": None}
tab_handle = {"h": None}

def set_static(clip):
    for slot, stl, pos, quat, col, name in (
        (obj_handle, clip["obj_stl"], clip["obj_pos"][0], clip["obj_quat"][0], (240, 120, 60), "/object"),
        (tab_handle, clip["tab_stl"], clip["tab_pos"][0], clip["tab_quat"][0], (70, 190, 180), "/table"),
    ):
        if slot["h"] is not None:
            slot["h"].remove()
        m = trimesh.load(stl, force="mesh")
        slot["h"] = server.scene.add_mesh_simple(
            name, np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int32),
            color=col, position=tuple(pos), wxyz=tuple(quat))

gui_clip = server.gui.add_dropdown("clip", tuple(names), initial_value=names[0])
gui_play = server.gui.add_checkbox("play", True)
gui_frame = server.gui.add_slider("frame", 0, 1, 1, 0)
gui_speed = server.gui.add_slider("speed", 0.1, 2.0, 0.1, 1.0)
gui_info = server.gui.add_text("info", "")

state = {"clip": None, "frame": 0, "pending": None}

def select(name):
    c = load(by_name[name])
    state["clip"] = c
    state["frame"] = 0
    gui_frame.max = c["n"] - 1
    gui_frame.value = 0
    set_static(c)
    m = by_name[name]
    gui_info.value = (f'{m["object"]}  |  {m["frames"]} frames @ {m["fps"]:.0f}fps  |  '
                      f'path {m["path_len_m"]:.2f} m  net {m["net_disp_m"]:.3f} m  z {m["z_range_m"]:.2f} m')

def draw(i: int):
    c = state["clip"]
    data.qpos[:] = c["qpos"][i]
    mujoco.mj_forward(model, data)
    for k, (g, _, _) in enumerate(ROBOT_GEOMS):
        robot_handles[k].position = tuple(data.geom_xpos[g])
        robot_handles[k].wxyz = tuple(mat2wxyz(data.geom_xmat[g]))
    if obj_handle["h"] is not None:
        obj_handle["h"].position = tuple(c["obj_pos"][i])
        obj_handle["h"].wxyz = tuple(c["obj_quat"][i])

gui_clip.on_update(lambda _: state.__setitem__("pending", gui_clip.value))
gui_frame.on_update(lambda _: state.__setitem__("frame", int(gui_frame.value)))
select(names[0]); draw(0)
print(f"viser on http://localhost:{A.port}")

while True:
    # Clip switches happen here, on the same thread that draws, so a handle is never removed while
    # the animation loop still holds it.
    if state["pending"] is not None:
        want, state["pending"] = state["pending"], None
        select(want)
        draw(0)
    c = state["clip"]
    if gui_play.value and c is not None:
        state["frame"] = (state["frame"] + 1) % c["n"]
        gui_frame.value = state["frame"]
    if c is not None:
        try:
            draw(state["frame"])
        except RuntimeError:
            pass  # handle swapped out from under us; the next tick redraws
    time.sleep(1.0 / (c["fps"] * max(gui_speed.value, 0.05)) if (c and gui_play.value) else 0.05)
