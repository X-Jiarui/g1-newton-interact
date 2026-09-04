"""Object retargeting by radial shrink about the pelvis: bring the scene into the robot's workspace.

The G1 is not a small human. Its arm is 0.369 m shoulder-to-wrist against a human's 0.524 m -- 70 % --
while the Wuji hand is 1.64x a human hand, so the whole reach chain is 0.91. The legacy placement
scales the object by a HEIGHT ratio (0.6355), which by accident lands it at almost exactly the right
distance from the pelvis: measured over 1335 clips the hand and the object sit 53.95 cm and 54.14 cm
out, and the ratio the two imply is 0.992.

That near-agreement is why an earlier sweep scored on C -- the hand-to-object distance at the contact
frame -- found the optimal shrink to be 1.000, i.e. do nothing. **That was the wrong objective.** The
hand can put its FINGERS near the object while being unable to put its PALM on it, and a grasp needs
the second. Scored on whether the hand centroid is reachable at all:

    d(shoulder, object)  <=  L_arm + r_wrist_to_centroid

only 28.2 % of sequences pass as retargeted today. The arm is already at 93 % extension at the contact
frame and needs 2.64 cm more than it has. Shrinking the radius fixes exactly this:

    alpha    reachable    object moved
    1.00        28.2 %       0.00 cm
    0.90        68.4 %       5.78 cm
    0.86        82.7 %       8.09 cm
    0.84        88.0 %       9.24 cm
    0.82        92.0 %      10.40 cm

The two anatomical ratios bracket that range -- 0.704 (arm alone) reaches 99.9 % but displaces the
scene 17 cm; 0.910 (full reach chain) only reaches 64.3 %.

The map is applied to the object AND the table, per frame, about the robot's own pelvis:

    p' = pelvis[t] + alpha * (p[t] - pelvis[t])

A uniform scaling about a point preserves incidence, so an object resting on its table still rests on
it; only the table's own thickness is not scaled, which leaves a sub-millimetre gap error. Being
per-frame also removes the legacy placement's frozen frame-0 world offset, worth 4-7 cm of drift.

Shrinking about the SHOULDER scores better at every alpha (93.8 % vs 88.0 % at 0.84) because the
constraint is a shoulder constraint. The pelvis is used anyway: the shoulder centre is undefined for
a sequence that passes the object from one hand to the other, and it breaks left-right symmetry.
"""

import argparse
import csv
import pathlib
import pickle
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "viz"))
import _kinematic_scene as ks  # noqa: E402


def pelvis_track(pkl, robot_xml):
    """Pelvis world position for every frame."""
    scene = ks.build(pkl, robot_xml, object_mesh="auto")
    pid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    out = np.empty((scene.n_frames, 3))
    for f in range(scene.n_frames):
        scene.set_frame(f)
        out[f] = scene.data.xpos[pid]
    return out, scene


SIDES = ("left", "right")


def reach_alpha(scene, pel, cf, margin):
    """The largest alpha <= 1 that brings the object inside the arm's reach at the contact frame.

    A grasp puts the hand's six-point centroid on the object, and the wrist sits a fixed distance r
    from that centroid because the hand shape is frozen. So the wrist must lie on a sphere of radius
    r about the object, and the arm reaches it iff |shoulder - object| <= L + r.

    Along the shrink ray p(a) = pelvis + a*(object - pelvis) that is a quadratic in a. The feasible
    set lies BETWEEN its two roots -- the parabola opens upward -- so the answer is the LARGER root,
    clipped to 1. Taking the smaller one drags the object past the shoulder and collapses alpha to
    zero, which is what a first version of this did.
    """
    scene.set_frame(cf)
    bid = lambda n: mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, n)
    obj = scene.object_pos[cf]
    best = None
    for sd in SIDES:
        six = np.vstack([np.stack([scene.data.xpos[bid(f"{sd}_finger{i}_tip")] for i in range(1, 6)]),
                         scene.data.xpos[bid(f"{sd}_wrist_yaw_link")][None]])
        dist = float(np.linalg.norm(six.mean(axis=0) - obj))
        if best is None or dist < best[0]:
            best = (dist, six, sd)
    _, six, sd = best
    wrist = scene.data.xpos[bid(f"{sd}_wrist_yaw_link")]
    shoulder = scene.data.xpos[bid(f"{sd}_shoulder_roll_link")]
    elbow = scene.data.xpos[bid(f"{sd}_elbow_link")]
    r = float(np.linalg.norm(wrist - six.mean(axis=0)))
    L = float(np.linalg.norm(shoulder - elbow) + np.linalg.norm(elbow - wrist))
    R = L + r - margin

    s_vec = shoulder - pel[cf]
    v = obj - pel[cf]
    A = float(v @ v); B = float(-2.0 * (s_vec @ v)); C = float(s_vec @ s_vec - R * R)
    disc = B * B - 4 * A * C
    if disc >= 0.0:
        a = (-B + np.sqrt(disc)) / (2 * A)
    else:
        a = -B / (2 * A)          # never reachable on this ray: get as close as the ray allows
    return float(np.clip(min(1.0, a), 0.0, 1.0)), sd, r, L


def contact_frame(pkl, grab_dir, n_frames):
    z = np.load(pathlib.Path(grab_dir) / "grab" / pathlib.Path(pkl).parent.name
                / f"{pathlib.Path(pkl).stem}.npz", allow_pickle=True)
    co = np.asarray(z["contact"].item()["object"])
    hit = np.flatnonzero((co > 0).any(axis=1))
    if not len(hit):
        return None
    return int(np.clip(round(hit[0] * n_frames / co.shape[0]), 0, n_frames - 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--robot-xml", required=True)
    ap.add_argument("--mode", choices=("adaptive", "global"), default="adaptive",
                    help="adaptive shrinks each clip only as far as reachability demands; "
                         "global applies one alpha everywhere")
    ap.add_argument("--alpha", type=float, default=0.85,
                    help="global mode only: radial scale, 1.0 leaves the scene alone")
    ap.add_argument("--margin", type=float, default=0.03,
                    help="adaptive mode only: metres of reach to keep in hand, so the arm is not "
                         "solved at full extension")
    ap.add_argument("--grab-dir", default="", help="adaptive mode: needed to find the contact frame")
    ap.add_argument("--summary", default="")
    args = ap.parse_args()

    rows = []
    for p in sorted(pathlib.Path(args.dataset_root).glob("*/*.pkl")):
        try:
            pel, scene = pelvis_track(str(p), args.robot_xml)
            alpha = args.alpha
            if args.mode == "adaptive":
                cf = contact_frame(str(p), args.grab_dir, scene.n_frames)
                if cf is None:
                    continue
                alpha, side, r_cm, L_cm = reach_alpha(scene, pel, cf, args.margin)
            d = pickle.load(open(p, "rb"))
            moved = {}
            for key in ("object", "table"):
                if key not in d:
                    continue
                q = np.asarray(d[key]["pos_mj"], np.float64)
                n = min(len(q), len(pel))
                new = q.copy()
                new[:n] = pel[:n] + alpha * (q[:n] - pel[:n])
                moved[key] = float(np.linalg.norm(new[:n] - q[:n], axis=1).mean())
                d[key]["pos_mj"] = new.astype(np.float32)
            d["object_radial_shrink"] = {"tool": "shrink_object_radius", "alpha": alpha,
                                         "mode": args.mode, "margin": args.margin,
                                         "centre": "pelvis", "per_frame": True}
            out = pathlib.Path(args.out_root) / p.parent.name / p.name
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "wb") as fh:
                pickle.dump(d, fh)
            rows.append({"seq": f"{p.parent.name}/{p.stem}", "alpha": round(alpha, 4),
                         "obj_moved_cm": round(moved.get("object", 0.0) * 100, 2),
                         "table_moved_cm": round(moved.get("table", 0.0) * 100, 2)})
        except Exception as e:
            print(f"  skip {p.parent.name}/{p.stem}: {type(e).__name__}: {e}", file=sys.stderr)

    M = np.array([r["obj_moved_cm"] for r in rows])
    A = np.array([r["alpha"] for r in rows])
    print(f"shrunk {len(rows)} sequences, mode = {args.mode}")
    print(f"  alpha: median {np.median(A):.3f}  p10 {np.percentile(A, 10):.3f}  "
          f"untouched (alpha = 1) {int((A >= 0.999).sum())}/{len(A)}")
    print(f"  object displaced: median {np.median(M):.2f} cm   p90 {np.percentile(M, 90):.2f} cm")
    if args.summary:
        with open(args.summary, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
