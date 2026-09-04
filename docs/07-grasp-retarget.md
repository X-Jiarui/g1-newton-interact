# 07 · Grasp retargeting: putting the hand where the human put it

GMR retargets the body well and the hand badly. At the frame GRAB annotates as first contact (`cf`)
the retargeted fingertips sit a median **9.1 cm** from the object points the human touched with them,
and the hand frame is **41.9°** away from the human's own orientation. A reference like that teaches a
policy to miss.

This pipeline re-solves the arm and hand at `cf` against targets taken from GRAB itself. Everything
outside `cf` is still GMR's.

```
GMR output ──1──▶ object pulled into reach ──2──▶ human fingertips, in the object frame ──3──▶ IK
                  shrink_object_radius            extract_grasp_targets                    solve_arm_ik
```

Every parameter, every sweep, and every rejected alternative is in
[`configs/retarget/grasp_ik.yaml`](../configs/retarget/grasp_ik.yaml). This file is the why.

## The adopted route: mesh fingertips + a palm-orientation objective

| | orient | shape | resid | pen | 4+touch |
|---|---|---|---|---|---|
| GMR, no IK | 41.9° | 16.8° | — | 0.86 cm | 58.6 % |
| joint + 2 cm, no rotation | 27.5° | 19.1° | 1.65 cm | 0.34 cm | 67.2 % |
| joint + 2 cm, rotation 0.3 | 12.8° | 12.8° | 1.49 cm | 0.40 cm | 59.1 % |
| mesh tip, no rotation | 25.6° | 18.6° | 1.60 cm | 0.69 cm | 76.8 % |
| **mesh tip + rotation 0.3** | **12.3°** | **12.4°** | **1.60 cm** | **0.55 cm** | **71.7 %** |

198 s1 sequences, 198/198 solved, no arm joint within 2 % of a limit on any row.

`orient` and `shape` are measured in the **object's** frame from two physical directions —
wrist→fingertip-centroid and wrist→thumb-tip — so no frame convention is assumed to be shared
between SMPL-X and the MJCF. `resid` is the mean fingertip-to-its-own-contact-point distance.

## Why the target is a mesh vertex

The obvious target is the human's fingertip joint. It is 3.3 cm from where the finger actually
touched. Measured as the distance from GRAB's annotated contact patch to each candidate:

```
SMPL-X tip landmark (71-75)   3.30 cm
SMPL-X distal joint           1.24 cm
nearest MESH vertex           0.42 cm     <- the annotation threshold is 2e-5 m
```

Only the mesh is where the skin was. The earlier route reached the same place by pushing the distal
joint 2 cm along the phalanx — a hand-tuned constant absorbing three separate offsets. The mesh
target needs none of it and buys 12.6 points of four-finger contact at the same orientation.

Getting the mesh right needs GRAB's own body build: **`v_template` from the subject's scanned .ply,
`flat_hand_mean=False`**. GRAB does not use `betas` at all. Building the mean-shape model instead
puts the body surface 3.36 cm from the contact patch instead of 0.42 cm, and every finger number
derived from it is wrong.

## Why the orientation objective

Position-only IK leaves the palm free to roll about the approach direction — nothing in five point
targets pins it. Measured 27.5° off the human's. One `IKObjectiveRotation` on the wrist, targeting the
human's own hand frame carried out through the object pose, halves both the orientation and the shape
error **and lowers the residual**: the position-only solution was sitting in a worse branch.

Weight 0.3 is the knee. Above it the orientation keeps improving while the residual climbs sharply —
6.82 cm at weight 3.

## What this does not fix

* **No collision term.** `newton.ik` offers Position, Rotation and JointLimit objectives and nothing
  else, so fingers pass through the object and through each other. Median penetration is 0.55 cm,
  which is *lower* than the 0.86 cm the retarget already carried, but the mechanism is absent rather
  than tuned.
* **One frame.** The approach and the carry are untouched, so the reference still jumps at `cf`.
* **Reachability is guaranteed at `cf` only**, and only because the object is moved. With the torso
  locked and no object shift, 52.0 % of clips are reachable. Opening the waist and torso to the IK is
  untested and could remove the need to move the object at all.

## Running it

```bash
X=$GMR_ROOT/assets/g1_wuji/g1_mocap_29dof_with_wuji_hands.xml

python tools/retarget/shrink_object_radius.py \
  --dataset-root $SRC --out-root $WORK/shifted --robot-xml $X \
  --mode adaptive --grab-dir $GRAB_DIR --margin 0.03

python tools/retarget/extract_grasp_targets.py \
  --grab-dir $GRAB_DIR --smplx-dir $GMR_ROOT/assets/body_models \
  --dataset-root $WORK/shifted --out $WORK/targets.npz --back 0

python tools/retarget/solve_arm_ik.py \
  --dataset-root $WORK/shifted --grab-dir $GRAB_DIR --robot-xml $X \
  --targets mesh --grasp-npz $WORK/targets.npz \
  --free-fingers --rot-weight 0.3 \
  --out-root $WORK/solved --summary $WORK/solved.csv
```

`solve_arm_ik.py` needs the `newton` package; the other two need only `mujoco`, `numpy`, `trimesh`,
and `smplx` for the target extraction.

Swap `--targets mesh` for `--targets human --tip-extend 0.020` to run the joint-based route instead.

## Pictures

[`media/grasp_retarget/`](../media/grasp_retarget/) — left column is the joint + 2 cm route, right
column is the adopted mesh route, both with the orientation objective, two camera angles each.
