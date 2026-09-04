"""Step 5a -- record the human's hand-object grasp geometry, in the OBJECT's own frame.

The object is the one thing the retarget does not distort: the mesh is the real object at real size,
and the pipeline changes only where it sits. So the human's hand pose *relative to the object* is
exactly transferable -- express it in the object frame here, and step 5b maps it back out through
whatever pose the object ends up with on the robot. Nothing about the robot enters this file.

The operating hand is read from the HUMAN data (whichever hand is nearer the object), not from the
retargeted robot, so a bad retarget cannot pick the wrong side.

Saved per sequence, in object-local metres:
    tips[5]   thumb, index, middle, ring, pinky fingertips   (SMPL-X 71-75 right / 66-70 left)
    wrist     the wrist joint                                (SMPL-X 21 right / 20 left)
    knuck[5]  the distal joint of each finger, for a shape reference
"""

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from solve_arm_ik import contact_frame  # noqa: E402

# SMPL-X joint indices. Verified by distance: every tip lands within ~3 cm of its own distal joint.
TIPS = {"right": [71, 72, 73, 74, 75], "left": [66, 67, 68, 69, 70]}
WRIST = {"right": 21, "left": 20}
DISTAL = {"right": [54, 42, 45, 51, 48], "left": [39, 27, 30, 36, 33]}


def smplx_joints(npz, smplx_dir, grab_dir, cache={}):
    """GRAB's own build, verified against its own contact annotation.

    Two settings decide whether the reconstructed hand is anywhere near the truth, and the default
    for both is wrong here. Measured as the distance from an annotated contact patch to the nearest
    body vertex -- which must be ~0, since the annotation threshold is 2e-5 m:

        mean shape, flat_hand_mean=True   3.36 cm      <- silently wrong
        v_template, flat_hand_mean=False  0.42 cm

    GRAB does not personalise with betas at all: `body["vtemp"]` is a path to that subject's scanned
    .ply, and it has to be passed as v_template. With the right build the distal joint lands 1.24 cm
    behind the patch it touched, which is where a fingertip joint anatomically belongs.
    """
    import smplx
    import torch
    import trimesh

    z = np.load(npz, allow_pickle=True)
    body = z["body"].item()["params"]
    T = len(np.asarray(body["transl"]))
    nc = body["right_hand_pose"].shape[1]
    vtemp = str(z["body"].item()["vtemp"])
    key = (str(z["gender"]), nc, T, vtemp)
    m = cache.get(key)
    if m is None:
        vt = np.asarray(trimesh.load(str(pathlib.Path(grab_dir) / vtemp),
                                     process=False).vertices, dtype=np.float32)
        m = smplx.create(smplx_dir, model_type="smplx", gender=str(z["gender"]),
                         use_pca=(nc != 45), num_pca_comps=nc, flat_hand_mean=False,
                         batch_size=T, v_template=vt)
        cache[key] = m
    with torch.no_grad():
        out = m(**{k: torch.tensor(body[k]) for k in
                   ("transl", "global_orient", "body_pose", "left_hand_pose", "right_hand_pose",
                    "jaw_pose", "leye_pose", "reye_pose", "expression")})
    return z, out.joints.numpy(), out.vertices.numpy()


def main() -> None:
    from scipy.spatial.transform import Rotation as R

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grab-dir", required=True)
    ap.add_argument("--smplx-dir", required=True)
    ap.add_argument("--dataset-root", required=True, help="only to enumerate the sequences")
    ap.add_argument("--out", required=True)
    ap.add_argument("--back", type=int, default=0,
                    help="frames before first contact, in GRAB's own 120 fps timeline")
    args = ap.parse_args()

    out = {}
    for p in sorted(pathlib.Path(args.dataset_root).glob("*/*.pkl")):
        seq = f"{p.parent.name}/{p.stem}"
        npz = pathlib.Path(args.grab_dir) / "grab" / p.parent.name / f"{p.stem}.npz"
        if not npz.exists():
            continue
        try:
            z, J, Vh = smplx_joints(str(npz), args.smplx_dir, args.grab_dir)
            co = np.asarray(z["contact"].item()["object"])
            hit = np.flatnonzero((co > 0).any(axis=1))
            if not len(hit):
                continue
            # The SAME frame the solver will apply this at. Picking it independently on each side
            # -- first contact here, most-fingers-touching there -- silently transplants the hand
            # geometry of one instant onto the body and object pose of another.
            tg, gside = contact_frame(co)
            if tg is None:
                continue
            t = int(max(0, tg - args.back))
            obj = z["object"].item()["params"]
            ot = np.asarray(obj["transl"], np.float64)[t]
            # GRAB's global_orient is the WORLD -> LOCAL rotation, so object-to-world is its
            # inverse. Two independent checks: Omnigrasp's own process_grab_raw.py applies .inv()
            # before storing the object quaternion, and measured on the human data alone the
            # annotated contact vertices land closer to the fingers under the inverse on 11 of 12
            # sequences. Our pipeline stores that inverse, so `local = R_g @ (world - t)` here and
            # `world = R_pkl @ local + t_pkl` on the way back out.
            oR = R.from_rotvec(np.asarray(obj["global_orient"], np.float64)[t]).as_matrix()

            # the operating hand, decided on the human, not on the robot
            # Geometric, not by label count: at first contact only a finger or two is touching
            # and the label counts are tiny, which picked the wrong hand on s1/apple_lift and
            # s1/cubesmall_lift -- both right-handed sequences read as left.
            side = min(("left", "right"),
                       key=lambda sd: np.linalg.norm(J[t, TIPS[sd]] - ot, axis=1).min())
            to_local = lambda P: (oR @ (P - ot).T).T

            # The SKIN point at each fingertip, not a joint. GRAB's annotated contact patch sits a
            # median 0.42 cm from the nearest body VERTEX but 1.24 cm from the distal JOINT and
            # 3.30 cm from the tip landmark -- the mesh is the only one of the three that is
            # actually where the finger touched. Taken as the vertex near the distal joint that
            # reaches furthest along the last phalanx, so no canonical vertex list is needed.
            skin = []
            for k in range(5):
                dj_, tl = J[t, DISTAL[side][k]], J[t, TIPS[side][k]]
                d_ = tl - dj_
                d_ = d_ / max(np.linalg.norm(d_), 1e-9)
                near = np.flatnonzero(np.linalg.norm(Vh[t] - dj_, axis=1) < 0.035)
                if len(near) == 0:
                    skin.append(tl)
                    continue
                skin.append(Vh[t][near[int(np.argmax((Vh[t][near] - dj_) @ d_))]])
            skin = np.stack(skin)
            out[seq] = dict(side=side, grab_frame=t, contact_frame=int(hit[0]),
                            tips=to_local(J[t, TIPS[side]]),
                            wrist=to_local(J[t, [WRIST[side]]])[0],
                            knuck=to_local(J[t, DISTAL[side]]),
                            tips_skin=to_local(skin))
        except Exception as e:
            print(f"  skip {seq}: {type(e).__name__}: {e}", file=sys.stderr)

    np.savez_compressed(args.out,
                        seqs=np.array(list(out)),
                        side=np.array([v["side"] for v in out.values()]),
                        grab_frame=np.array([v["grab_frame"] for v in out.values()]),
                        contact_frame=np.array([v["contact_frame"] for v in out.values()]),
                        tips=np.array([v["tips"] for v in out.values()]),
                        wrist=np.array([v["wrist"] for v in out.values()]),
                        knuck=np.array([v["knuck"] for v in out.values()]),
                        tips_skin=np.array([v["tips_skin"] for v in out.values()]))
    sp = np.array([np.linalg.norm(v["tips"], axis=1) for v in out.values()])
    print(f"extracted {len(out)} grasps at contact - {args.back} GRAB frames")
    print(f"  fingertip distance to the object ORIGIN: median {100*np.median(sp):.2f} cm "
          f"(this is the geometry being transferred)")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
