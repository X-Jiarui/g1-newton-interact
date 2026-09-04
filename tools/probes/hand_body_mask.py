"""Offline check of the ported pregrasp target: do the hand bodies resolve, and does the mask work?

`staged_hand_cf_reward` scores the whole grasping hand -- palm plus `finger{1..5}_link{2,3,4}`,
16 bodies a side -- against that body's pose at cf, and picks WHICH bodies count by Omnigrasp's own
rule: the reference bodies within `near_threshold` of the reference object at cf. That rule has no
left/right flag in it, so the first thing to verify is that it selects one hand and the right one.

CPU only, no mjlab, no GPU, so it is safe to run beside live training.

    python tools/probes/hand_body_mask.py <robot.xml> <step6>/<subject>/<stem>.pkl ...

TRAP, cost an hour once: the pkl stores `root_rot` as XYZW and MuJoCo's free joint wants WXYZ.
Read as-is, forward kinematics puts the palm 81 cm from the object instead of 8 cm and the mask
selects nothing. The production path is unaffected -- the reference loader normalises to WXYZ --
but any probe that reads the pkl directly has to reorder.
"""
import pickle
import sys

import mujoco
import numpy as np

HAND_BODIES = tuple(
    f"{side}_{part}"
    for side in ("left", "right")
    for part in ("palm_link", *(f"finger{i}_link{j}" for i in range(1, 6) for j in (2, 3, 4)))
)
REF_FPS, TARGET_FPS = 30.0, 50.0   # the trainer resamples the reference; cf is reported at 50


def main() -> None:
    xml, paths = sys.argv[1], sys.argv[2:]
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    ids, kept = [], []
    for name in HAND_BODIES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            ids.append(int(bid))
            kept.append(name)
    print(f"resolved {len(kept)}/{len(HAND_BODIES)} hand bodies")
    missing = [n for n in HAND_BODIES if n not in kept]
    if missing:
        print("  MISSING:", missing)
    free = next(i for i, t in enumerate(model.jnt_type) if t == mujoco.mjtJoint.mjJNT_FREE)
    fq = int(model.jnt_qposadr[free])

    print(f"\n{'clip':<20}{'cf50':>6}{'used@0.20':>11}{'used@0.10':>11}{'side':>8}{'mean_cm':>9}")
    print("-" * 65)
    for path in paths:
        raw = pickle.load(open(path, "rb"))
        d = raw[list(raw.keys())[0]] if isinstance(list(raw.values())[0], dict) else raw
        cf = int(round(int(d["grasp_blend"]["cf"])))          # the pkl's own frame index
        names = list(d["dof_names"])
        r = d["robot_29dof"]
        q = np.asarray(r["dof_pos"])[cf]
        rp = np.asarray(r["root_pos"])[cf]
        rr = np.asarray(r["root_rot"])[cf]
        obj = np.asarray(d["object"]["pos_mj"])[cf]
        qadr = [
            int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
            for n in names
        ]
        data.qpos[:] = model.qpos0
        data.qpos[fq : fq + 3] = rp
        data.qpos[fq + 3 : fq + 7] = np.array([rr[3], rr[0], rr[1], rr[2]])   # XYZW -> WXYZ
        for adr, val in zip(qadr, q, strict=True):
            data.qpos[adr] = val
        mujoco.mj_forward(model, data)
        dist = np.linalg.norm(data.xpos[ids] - obj, axis=1)
        u20, u10 = dist < 0.20, dist < 0.10
        side = "right" if u20[16:].sum() >= u20[:16].sum() else "left"
        mean = dist[u20].mean() * 100 if u20.any() else float("nan")
        stem = path.split("/")[-1][:-4]
        print(f"{stem:<20}{cf * TARGET_FPS / REF_FPS:>6.0f}{int(u20.sum()):>11}"
              f"{int(u10.sum()):>11}{side:>8}{mean:>9.2f}")
    print("\nOnly the 29 body DOF are applied here, so the fingers sit at qpos0 and the distances "
          "are an upper bound; what is being checked is the SELECTION, not the error.")


if __name__ == "__main__":
    main()
