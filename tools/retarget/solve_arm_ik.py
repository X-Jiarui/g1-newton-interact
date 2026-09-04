"""Step 5 -- re-solve the arm so the hand meets the object where the human's hand met it.

GMR gives the body: pelvis, torso, shoulder. Everything from the shoulder down the arm is re-solved
here, against targets taken from GRAB's own contact annotation. The hand's 40 finger joints are
frozen at whatever step 3 produced -- the hand is a rigid shape being placed, not a hand being posed.

**The target is a point on the OBJECT, not on the human's hand.** GRAB labels, per frame, which
object vertices each finger touches; those vertices live in the object's canonical mesh frame, so the
target carries no information about how long the human's fingers were and needs no frame convention
to interpret. Targeting the human's fingertip POSITIONS instead was tried first and measured worse
than doing nothing: the Wuji finger is 1.64x a human finger and the hand is rigid, so putting its
tips where the human's tips were drags the palm off the object. Measured on 25 clips, that
formulation moved the hand-object distance from 5.88 to 8.78 cm and four-finger contact from 55 % to
15 %, while hitting its own targets to 4.4 cm. Optimising a target that cannot be met by this hand.

The frame is chosen inside the annotated contact window as the one where the MOST distinct fingers
are touching. First contact is the wrong choice and was the first thing tried: one finger has landed
and the grasp has not formed, so most sequences yielded fewer than two usable targets.

Solver: Newton's IK (Levenberg-Marquardt). Free variables are the seven arm joints of the operating
side; every other DoF is pinned by an IKObjectiveJointLimit whose lower and upper bounds are both the
current value. Two details that cost a debugging round each:

  * Newton maps DoF -> coordinate as free-joint DoFs 0-5 -> coords 0-5, then DoF i -> coord i+1.
    Pinning the free joint with the naive `coord - 1` writes into the root POSITION and commands the
    robot to the world origin; the frozen joints then drifted 46 degrees.
  * Newton's free-joint quaternion is xyzw, matching what our pkl stores, not MuJoCo's wxyz.

Verified after the fix: frozen joints move at most 0.08 degrees, and no arm joint ends within 2 % of
a stop -- the failure mode that ruined eleven s7 sequences in the GMR solve.
"""

import argparse
import csv
import pathlib
import pickle
import sys

import numpy as np

# The DISTAL joint of each finger only -- thumb, index, middle, ring, pinky.
#
# Using all three joints of a finger is wrong and was the first thing tried: vertices labelled with
# the proximal joint are touched by the base of the finger, so the centroid of the whole finger's
# contact band lands mid-finger. Measured, that target sits a median 8.96 cm from the human's own
# fingertip joint -- so the IK was pulling the Wuji FINGERTIP onto a point the human touched with the
# middle of its finger, which drives the tip straight through the surface. Penetration went from
# 0.85 cm to 1.43 cm under that version.
SMPLX_FINGER = {
    "right": {1: (54,), 2: (42,), 3: (45,), 4: (51,), 5: (48,)},
    "left": {1: (39,), 2: (27,), 3: (30,), 4: (36,), 5: (33,)},
}
# All three joints, kept for deciding WHICH side and WHICH frame is the grasp: for that question the
# whole finger counts, and requiring distal-only contact would reject valid grasp frames.
SMPLX_FINGER_ALL = {
    "right": {1: (52, 53, 54), 2: (40, 41, 42), 3: (43, 44, 45), 4: (49, 50, 51), 5: (46, 47, 48)},
    "left": {1: (37, 38, 39), 2: (25, 26, 27), 3: (28, 29, 30), 4: (34, 35, 36), 5: (31, 32, 33)},
}
ARM = {s: [f"{s}_shoulder_pitch_joint", f"{s}_shoulder_roll_joint", f"{s}_shoulder_yaw_joint",
           f"{s}_elbow_joint", f"{s}_wrist_roll_joint", f"{s}_wrist_pitch_joint",
           f"{s}_wrist_yaw_joint"] for s in ("left", "right")}


def hand_frame(wrist, tips):
    """Orthonormal frame from wrist->fingertip-centroid and wrist->thumb-tip.

    Built the same way for the human and for the robot, so no shared frame convention between
    SMPL-X and the MJCF is assumed -- only two directions that exist physically on both hands.
    """
    a = tips.mean(axis=0) - wrist
    na = np.linalg.norm(a)
    if na < 1e-9:
        return None
    a = a / na
    b = tips[0] - wrist
    b = b - (b @ a) * a
    nb = np.linalg.norm(b)
    if nb < 1e-9:
        return None
    b = b / nb
    return np.stack([a, b, np.cross(a, b)], axis=1)


def mat_to_quat_xyzw(R):
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w < 1e-8:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        t = np.sqrt(max(1e-12, 1.0 + R[i, i] - R[j, j] - R[k, k]))
        q = np.zeros(4)
        q[i] = 0.5 * t
        q[3] = (R[k, j] - R[j, k]) / (2 * t)
        q[j] = (R[j, i] + R[i, j]) / (2 * t)
        q[k] = (R[k, i] + R[i, k]) / (2 * t)
        return q
    return np.array([(R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w),
                     (R[1, 0] - R[0, 1]) / (4 * w), w])


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def contact_frame(co):
    """cf: the first frame GRAB annotates as contact, and the hand that is touching there.

    One frame, defined by the dataset, used by every stage. An earlier version picked "the frame in
    the window where the most fingers touch", which is a better-formed grasp but makes the frame a
    function of a heuristic -- and the extractor and the solver then each chose their own, so the
    hand geometry of one instant was transplanted onto the body and object pose of another.
    """
    hit = np.flatnonzero((co > 0).any(axis=1))
    if not len(hit):
        return None, None
    t = int(hit[0])
    lab = co[t]
    side = max(("left", "right"),
               key=lambda sd: sum(np.isin(lab, SMPLX_FINGER_ALL[sd][k]).sum() for k in range(1, 6)))
    return t, side


def main() -> None:
    import mujoco
    import newton
    import newton.ik as ik
    import trimesh
    import warp as wp

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--grab-dir", required=True)
    ap.add_argument("--robot-xml", required=True)
    ap.add_argument("--out-root", default="")
    ap.add_argument("--iterations", type=int, default=150)
    ap.add_argument("--weights", default="3,3,3,1,1",
                    help="per-finger objective weight, thumb..pinky")
    ap.add_argument("--target", choices=("contact", "wrist"), default="contact",
                    help="contact: drive each fingertip to the object point that finger touched. "
                         "wrist: drive the WRIST to the pose the human's wrist held relative to the "
                         "object, taken a few frames BEFORE contact where nothing interpenetrates. "
                         "The wrist route is immune to the 1.64x finger-length difference because it "
                         "never asks a Wuji finger to be where a human finger was.")
    ap.add_argument("--grasp-targets", default="",
                    help="wrist mode: the npz written by extract_grasp_targets.py")
    ap.add_argument("--targets", choices=("contact", "human", "mesh"), default="contact",
                    help="contact: the object-surface point each finger touched, lifted off along "
                         "the normal. human: the human's own fingertip POSITIONS, transferred "
                         "through the object frame -- keeps the object still and moves the finger to "
                         "where the human's finger actually was.")
    ap.add_argument("--rot-weight", type=float, default=0.0,
                    help="weight on a palm ORIENTATION objective. The position-only formulation "
                         "leaves the palm free to roll about the approach direction: measured, the "
                         "hand frame is a median 27.6 deg off the human's, and orientation is the "
                         "one thing the IK never optimised. The target is the human's own hand "
                         "frame at cf, carried into the object frame, so it needs no shared "
                         "convention between SMPL-X and the MJCF -- both frames are built from the "
                         "same two physical directions, wrist->fingertip-centroid and wrist->thumb.")
    ap.add_argument("--tip-extend", type=float, default=0.01,
                    help="targets=human: metres to push the target past the human's last finger "
                         "JOINT, along the last phalanx. A joint is inside the finger, not on its "
                         "pad -- measured, the human's distal joint sits 1.24 cm behind the object "
                         "surface it touched. One centimetre past the joint is that pad.")
    ap.add_argument("--grasp-npz", default="",
                    help="targets=human: output of extract_grasp_targets.py")
    ap.add_argument("--free-waist", action="store_true",
                    help="also solve waist yaw/roll/pitch. A human at full arm extension leans in "
                         "rather than stretching further, and the G1 has the same option: waist "
                         "pitch is +-29.8 deg and the shoulder sits ~0.45 m above the joint, so a "
                         "full lean carries the shoulder ~22 cm forward. Measured, the arm is "
                         "already at 93 %% extension at the contact frame and short by 2.64 cm.")
    ap.add_argument("--free-fingers", action="store_true",
                    help="also solve the 20 finger joints of the operating hand, instead of holding "
                         "the shape step 3 produced. The rigid hand has a floor: the best possible "
                         "rigid placement still leaves a median 3.09 cm of the 4.67 cm residual, so "
                         "two thirds of what is left is shape rather than placement.")
    ap.add_argument("--standoff", type=float, default=0.012,
                    help="metres to lift each target off the object along its outward normal. The "
                         "objective drives the fingertip FRAME, and a frame on the surface means the "
                         "finger's shell is inside it: measured, tips were the deepest part of the "
                         "hand in 69 %% of clips at a median 1.20 cm while the palm never touched. "
                         "The human has the same offset in reverse -- its distal joint sits 1.24 cm "
                         "behind the skin that did the touching.")
    ap.add_argument("--summary", default="")
    args = ap.parse_args()

    W = [float(x) for x in args.weights.split(",")]
    X = args.robot_xml
    mj = mujoco.MjModel.from_xml_path(X)
    dj = mujoco.MjData(mj)
    MB = lambda n: mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, n)
    NB = lambda n: MB(n) - 1                       # Newton drops the world body
    n2j = {mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(mj.njnt)}
    COORD = {n: int(mj.jnt_qposadr[j]) for n, j in n2j.items() if n}
    RANGE = {n: mj.jnt_range[j].copy() for n, j in n2j.items() if n}
    NDOF, NCOORD = mj.nv, mj.nq

    builder = newton.ModelBuilder()
    builder.add_mjcf(X)
    model = builder.finalize()

    def build(side):
        if args.target == "wrist":
            tgt = [wp.zeros(1, dtype=wp.vec3)]
            objs = [ik.IKObjectivePosition(NB(f"{side}_wrist_yaw_link"), wp.vec3(0.0, 0.0, 0.0),
                                           tgt[0], weight=5.0)]
        else:
            tgt = [wp.zeros(1, dtype=wp.vec3) for _ in range(5)]
            objs = [ik.IKObjectivePosition(NB(f"{side}_finger{i+1}_tip"), wp.vec3(0.0, 0.0, 0.0),
                                           tgt[i], weight=W[i]) for i in range(5)]
        rq = wp.zeros(1, dtype=wp.vec4)
        if args.rot_weight > 0:
            objs.append(ik.IKObjectiveRotation(NB(f"{side}_wrist_yaw_link"),
                                               wp.quat(0.0, 0.0, 0.0, 1.0), rq,
                                               weight=args.rot_weight))
        lo = wp.zeros(NDOF, dtype=float); hi = wp.zeros(NDOF, dtype=float)
        objs.append(ik.IKObjectiveJointLimit(lo, hi, weight=50.0))
        return (ik.IKSolver(model, 1, objs, optimizer="lm", sampler="none", n_seeds=1),
                tgt, lo, hi, rq)

    HUMAN_IDX, HUMAN_TIPS, HUMAN_KNUCK, HUMAN_SIDE = {}, None, None, {}
    if args.targets in ("human", "mesh"):
        g = np.load(args.grasp_npz, allow_pickle=True)
        HUMAN_IDX = {str(x): i for i, x in enumerate(g["seqs"])}
        HUMAN_TIPS = g["tips"]
        HUMAN_FRAME = {str(x): int(g["grab_frame"][i]) for i, x in enumerate(g["seqs"])}
        HUMAN_KNUCK = g["knuck"]
        HUMAN_SIDE = {str(x): str(g["side"][i]) for i, x in enumerate(g["seqs"])}
        G_WRIST = g["wrist"]
        HUMAN_SKIN = g["tips_skin"] if "tips_skin" in g else None

    S = {sd: build(sd) for sd in ("left", "right")}
    GD = pathlib.Path(args.grab_dir)
    GT = None
    if args.target == "wrist":
        z_ = np.load(args.grasp_targets, allow_pickle=True)
        GT = {str(k): dict(wrist=z_["wrist"][i], side=str(z_["side"][i]),
                           grab_frame=int(z_["grab_frame"][i]))
              for i, k in enumerate(z_["seqs"])}
    rows = []
    for p in sorted(pathlib.Path(args.dataset_root).glob("*/*.pkl")):
        seq = f"{p.parent.name}/{p.stem}"
        try:
            z = np.load(GD / "grab" / f"{seq}.npz", allow_pickle=True)
            co = np.asarray(z["contact"].item()["object"])
            t, side = contact_frame(co)
            if t is None:
                continue
            mesh = trimesh.load(str(GD / z["object"].item()["object_mesh"]), process=False)
            V = np.asarray(mesh.vertices)
            N = np.asarray(mesh.vertex_normals)
            if len(V) != co.shape[1]:
                continue
            d = pickle.load(open(p, "rb"))
            b = d["robot_53dof"]
            nf = len(np.asarray(b["dof_pos"]))
            # targets=human: use the GRAB frame the hand geometry was recorded at, so the body,
            # the object and the hand all come from one instant.
            f = int(np.clip(round(t * nf / co.shape[0]), 0, nf - 1))
            q0 = np.concatenate([np.asarray(b["root_pos"])[f], np.asarray(b["root_rot"])[f],
                                 np.asarray(b["dof_pos"])[f]]).astype(np.float32)
            op = np.asarray(d["object"]["pos_mj"], np.float64)[f]
            oR = quat_to_mat(np.asarray(d["object"]["quat_wxyz_mj"], np.float64)[f])

            lab = co[t]
            local = {}
            if args.targets == "mesh":
                i = HUMAN_IDX.get(seq)
                if i is None:
                    continue
                side = HUMAN_SIDE[seq]
                solver, tgt, lo_a, hi_a, rq_a = S[side]
                for k in range(1, 6):
                    local[k] = HUMAN_SKIN[i][k - 1]
            elif args.targets == "human":
                i = HUMAN_IDX.get(seq)
                if i is None:
                    continue
                side = HUMAN_SIDE[seq]          # decided geometrically by the extractor
                solver, tgt, lo_a, hi_a, rq_a = S[side]
                # The human's fingertips, expressed in the object frame. The object keeps its size
                # and orientation between the two worlds, so this is the human's finger placed
                # against the same object -- no standoff, the human was not inside it.
                for k in range(1, 6):
                    knuck = HUMAN_KNUCK[i][k - 1]
                    d_ = HUMAN_TIPS[i][k - 1] - knuck
                    n_ = d_ / max(np.linalg.norm(d_), 1e-9)
                    local[k] = knuck + args.tip_extend * n_
            else:
              for k in range(1, 6):
                  m_ = np.isin(lab, SMPLX_FINGER[side][k])
                  if m_.sum() < 3:
                      continue
                  n_ = N[m_].mean(axis=0)
                  n_ = n_ / max(np.linalg.norm(n_), 1e-9)
                  local[k] = V[m_].mean(axis=0) + args.standoff * n_

            solver, tgt, lo_a, hi_a, rq_a = S[side]
            dj.qpos[:] = np.concatenate([q0[:3], q0[[6, 3, 4, 5]], q0[7:]])
            mujoco.mj_forward(mj, dj)
            world = {}
            if args.target == "wrist":
                g = GT.get(seq)
                if g is None:
                    continue
                side = g["side"]
                solver, tgt, lo_a, hi_a, rq_a = S[side]
                # the object pose at the frame the human's wrist pose was taken from
                fw = int(np.clip(round(g["grab_frame"] * nf / co.shape[0]), 0, nf - 1))
                opw = np.asarray(d["object"]["pos_mj"], np.float64)[fw]
                oRw = quat_to_mat(np.asarray(d["object"]["quat_wxyz_mj"], np.float64)[fw])
                wt = oRw @ g["wrist"] + opw
                tgt[0].assign(np.array([wt], dtype=np.float32))
                f = fw
                q0 = np.concatenate([np.asarray(b["root_pos"])[f], np.asarray(b["root_rot"])[f],
                                     np.asarray(b["dof_pos"])[f]]).astype(np.float32)
                op = np.asarray(d["object"]["pos_mj"], np.float64)[f]
                oR = quat_to_mat(np.asarray(d["object"]["quat_wxyz_mj"], np.float64)[f])
                dj.qpos[:] = np.concatenate([q0[:3], q0[[6, 3, 4, 5]], q0[7:]])
                mujoco.mj_forward(mj, dj)
                world = {0: wt}
            else:
                for k in range(5):
                    if (k + 1) in local:
                        world[k] = oR @ local[k + 1] + op
                    else:                              # no annotated contact: leave it where it is
                        world[k] = dj.xpos[MB(f"{side}_finger{k+1}_tip")].copy()
                    tgt[k].assign(np.array([world[k]], dtype=np.float32))

            if args.rot_weight > 0 and args.targets in ("human", "mesh"):
                # The human's hand frame at cf, carried out through the object pose. The wrist BODY
                # frame is what the objective drives, so compose in the constant offset between the
                # wrist body and the hand frame, read off the current pose.
                Fh = hand_frame(G_WRIST[i], HUMAN_TIPS[i])
                r_tips_w = np.stack([dj.xpos[MB(f"{side}_finger{k}_tip")] for k in range(1, 6)])
                Fr = hand_frame(dj.xpos[MB(f"{side}_wrist_yaw_link")], r_tips_w)
                if Fh is not None and Fr is not None:
                    Rw = dj.xmat[MB(f"{side}_wrist_yaw_link")].reshape(3, 3)
                    C = Rw.T @ Fr
                    Rt = (oR @ Fh) @ C.T
                    rq_a.assign(np.array([mat_to_quat_xyzw(Rt)], dtype=np.float32))

            lo = np.empty(NDOF, np.float32); hi = np.empty(NDOF, np.float32)
            for dof in range(NDOF):
                c = dof if dof < 6 else dof + 1
                lo[dof] = q0[c]; hi[dof] = q0[c]
            free = list(ARM[side])
            if args.free_waist:
                free += ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
            if args.free_fingers:
                free += sorted(n for n in COORD if n.startswith(f"{side}_finger"))
            for n in free:
                dof = COORD[n] - 1
                lo[dof] = RANGE[n][0]; hi[dof] = RANGE[n][1]
            lo_a.assign(lo); hi_a.assign(hi)

            qo = wp.zeros((1, NCOORD), dtype=float)
            solver.step(wp.array(q0.reshape(1, -1), dtype=float), qo, iterations=args.iterations)
            qs = qo.numpy()[0]

            def residuals(q):
                """Always reported against the CONTACT points, in both modes -- the wrist route has
                to be judged on the grasp it produces, not on the target it was given."""
                dj.qpos[:] = np.concatenate([q[:3], q[[6, 3, 4, 5]], q[7:]])
                mujoco.mj_forward(mj, dj)
                return np.array([np.linalg.norm(dj.xpos[MB(f"{side}_finger{k}_tip")]
                                                - (oR @ local[k] + op))
                                 for k in range(1, 6) if k in local]) * 100

            r0, r1 = residuals(q0), residuals(qs)
            if r1.size == 0:      # no finger has a distal-contact patch: nothing to score
                continue
            sat = max(min(abs(qs[COORD[n]] - RANGE[n][0]), abs(qs[COORD[n]] - RANGE[n][1]))
                      / (RANGE[n][1] - RANGE[n][0]) for n in free)
            rows.append({"seq": seq, "side": side, "frame": f, "n_targets": len(local),
                         "before_cm": round(float(r0.mean()), 2),
                         "after_cm": round(float(r1.mean()), 2),
                         "worst_after_cm": round(float(r1.max()), 2),
                         "sat": round(float(sat), 4)})
            if args.out_root:
                cols = [COORD[n] - 7 for n in free]
                dp = np.asarray(b["dof_pos"]).copy()
                dp[f, cols] = qs[[COORD[n] for n in free]]
                d["robot_53dof"]["dof_pos"] = dp
                d["arm_ik"] = {"tool": "solve_arm_ik", "frame": f, "side": side,
                               "n_targets": len(local), "residual_cm": float(r1.mean())}
                o = pathlib.Path(args.out_root) / p.parent.name / p.name
                o.parent.mkdir(parents=True, exist_ok=True)
                pickle.dump(d, open(o, "wb"))
        except Exception as e:
            print(f"  skip {seq}: {type(e).__name__}: {e}", file=sys.stderr)

    B = np.array([r["before_cm"] for r in rows]); A = np.array([r["after_cm"] for r in rows])
    Wt = np.array([r["worst_after_cm"] for r in rows]); St = np.array([r["sat"] for r in rows])
    print(f"\n{len(rows)} sequences solved")
    print(f"  fingertip -> its own contact point on the object")
    print(f"    mean over the touching fingers: median {np.median(B):.2f} -> {np.median(A):.2f} cm")
    print(f"    worst finger in the clip      : median {np.median(Wt):.2f} cm")
    print(f"    within 2 cm: {100*(A<2).mean():.1f}%    within 3 cm: {100*(A<3).mean():.1f}%")
    print(f"    improved on {int((A<B).sum())}/{len(rows)}")
    print(f"  arm joints within 2 % of a stop: {int((St<0.02).sum())}/{len(rows)}")
    if args.summary:
        with open(args.summary, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"  wrote {args.summary}")


if __name__ == "__main__":
    main()
