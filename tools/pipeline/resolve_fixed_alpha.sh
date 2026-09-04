#!/bin/bash
# Re-solve a handful of clips through pipeline steps 4-6 with a FIXED object shrink alpha.
#
# Why not the pipeline's own `--mode adaptive`: adaptive stops as soon as a SHOULDER-REACH
# constraint is satisfied with a 3 cm margin, and on the eight R26 clips it left three at
# alpha = 1.000 -- no correction at all. `shrink_object_radius.py`'s own table says alpha 1.00
# leaves only 28.2 % of sequences geometrically reachable at the contact frame, with the arm at
# 93 % extension and 2.64 cm short; 0.85 takes that to ~88 %.
#
# The verdict on 0.85 is NOT that it makes a better grasp. Scored against the HUMAN's own C and D
# (tools/eval/human_cd.py), 0.85 is better on |D - D_human| on 4/4 clips and a wash on C -- but
# read as "smaller C and D is better", the same numbers say it is worse on 4/4. C and D are targets
# to match, not to minimise; the human's own D is 1.2-4.3 cm on these clips.
#
# Interpreters: the step-3 pkls are written by a numpy-2 process, so a numpy-1 environment cannot
# unpickle them and the per-clip `except: continue` swallows it into "0 sequences solved". Both PY
# and PY_IK must be numpy 2; PY additionally needs `smplx` (pip install --no-deps smplx).
#
# Required: PIPE GMR_ROOT GRAB_DIR SRC OUT PY PY_IK CLIPS
# Optional: ALPHA (0.85)
set -euo pipefail
PIPE=${PIPE:?set PIPE to the grab-g1-wuji-pipeline checkout}
GMR_ROOT=${GMR_ROOT:?set GMR_ROOT}
GRAB_DIR=${GRAB_DIR:?set GRAB_DIR to the dir holding grab/<subject>/<seq>.npz}
SRC=${SRC:?set SRC to the step-3 output (hand-openness-scaled) dataset root}
OUT=${OUT:?set OUT to a fresh output dir}
PY=${PY:?set PY to a numpy-2 python that has smplx}
PY_IK=${PY_IK:?set PY_IK to a numpy-2 python that has warp/CUDA}
CLIPS=${CLIPS:?set CLIPS to whitespace-separated <subject>/<stem> entries}
ALPHA=${ALPHA:-0.85}
export GMR_ROOT GRAB_DIR

ROBOT_XML=${ROBOT_XML:-$GMR_ROOT/assets/g1_wuji/g1_mocap_29dof_with_wuji_hands.xml}
SMPLX_DIR=${SMPLX_DIR:-$GMR_ROOT/assets/body_models}

rm -rf "$OUT"; mkdir -p "$OUT/in"
for c in $CLIPS; do
  mkdir -p "$OUT/in/$(dirname "$c")"
  cp "$SRC/$c.pkl" "$OUT/in/$c.pkl"
done
echo "input: $(find "$OUT/in" -name '*.pkl' | wc -l) clip(s)"

echo "== step 4: fixed alpha=$ALPHA (global), not adaptive =="
$PY "$PIPE/tools/retarget/shrink_object_radius.py" \
  --dataset-root "$OUT/in" --out-root "$OUT/step4_object_in_reach" --robot-xml "$ROBOT_XML" \
  --mode global --alpha "$ALPHA" --grab-dir "$GRAB_DIR" --summary "$OUT/step4_summary.csv"

echo "== step 5a: grasp targets from the SMPL-X mesh =="
$PY "$PIPE/tools/retarget/extract_grasp_targets.py" \
  --grab-dir "$GRAB_DIR" --smplx-dir "$SMPLX_DIR" \
  --dataset-root "$OUT/step4_object_in_reach" --out "$OUT/grasp_targets.npz" --back 0

echo "== step 5b: solve arm + fingers at cf =="
$PY_IK "$PIPE/tools/retarget/solve_arm_ik.py" \
  --dataset-root "$OUT/step4_object_in_reach" --grab-dir "$GRAB_DIR" --robot-xml "$ROBOT_XML" \
  --targets mesh --grasp-npz "$OUT/grasp_targets.npz" --free-fingers --rot-weight 0.3 \
  --out-root "$OUT/step5_grasp_at_cf" --summary "$OUT/step5_summary.csv"

echo "== step 6: carry the grasp into the clip =="
$PY "$PIPE/tools/retarget/blend_grasp_into_clip.py" \
  --gmr-root "$OUT/step4_object_in_reach" --solved-root "$OUT/step5_grasp_at_cf" \
  --summary-in "$OUT/step5_summary.csv" --out-root "$OUT/step6_grasp_blended" \
  --robot-xml "$ROBOT_XML" --window 30 --summary "$OUT/step6_summary.csv"

echo
echo "RESULT: $(find "$OUT/step6_grasp_blended" -name '*.pkl' | wc -l) clips"
find "$OUT/step6_grasp_blended" -name '*.pkl' | sort
