#!/bin/bash
# R28 -> R30: move the task toward Omnigrasp's shape, one switch at a time, on four fresh clips.
#
# Every switch below is an environment variable that defaults to OFF, so a run that does not set it
# is byte-identical to R26. They were turned on in three rounds and each round resumed from the
# previous round's own checkpoint, so clip, data and weights are held fixed and only the switch
# differs. The previous round's logs are the control.
#
#   R28  data re-solved at a FIXED object shrink alpha=0.85 (see tools/pipeline/resolve_fixed_alpha.sh)
#        TABLE_REMOVE_AFTER_CF=0    withdraw the table the moment the clip reaches cf
#   R29  OBJ_REF_WINDOW_ENABLE=0    og_object_far at 0.12 m becomes the ONLY object termination
#        OBJ_REF_STILL_UNTIL_CF=1   hold the object reference at rest until cf, then rebase
#        TIP_CF_MISS_ENABLE=0       Omnigrasp has no such termination
#        RSI_ANCHOR_CF=1            start episodes 10-20 frames before each clip's own cf
#   R30  TABLE_REMOVE_AFTER_CF=50   1.000 s at our 50 Hz, which is Omnigrasp's own margin: it
#                                   removes the table at a fixed 45 frames (1.5 s at its 30 Hz)
#                                   while training starts are clamped to contact - 0.5 s.
#
# Measured, in order:
#   R28   object_reference_window at 0.05 m took 70-100 % of every reset, and fired BEFORE the grasp
#         was due because the reference lifts the object 20-40 frames ahead of cf.
#   R29   with it off, og_object_far became the only terminator and ep_len pinned at 36 startup
#         + 15 approach + 7.8 free-fall = ~59 steps. contact rose for eleven straight cycles while
#         ep_len never moved: the hand touched more and held nothing.
#   R30   ep_len 51-60 -> 81-106, contact 0.16-0.25 -> 0.30-0.42, and three of four runs reached a
#         sustained non-zero lift_success for the first time (binoculars 1.2-1.9 % over fourteen
#         consecutive records). The lifts happen while the table is still there.
#
# Required: ROOT DATA MESH LOGS PY CFG CLIPS
# Optional: ENVS (2048) ITERS (12000) SEED (1) RESUME_FROM (a log dir to warm-start each run from)
#
# CLIPS entries are `gpu:clip_stem:mesh_stem:tag`, whitespace separated:
#   CLIPS="0:apple_lift:apple:R30_APPLE 1:camera_lift:camera:R30_CAMERA"
#
# From scratch is safe with this switch set, which was NOT true before it. R27 collapsed because
# wrist_target_far killed 60-100 % of episodes during the approach; it activates at control step 56
# and is pre_cf_only, and RSI_ANCHOR_CF puts cf at step ~51, so it can no longer fire -- R30 has
# read wfar 0.0000 on every cycle, and two from-scratch cube runs reached contact 0.23 by iteration
# 44 with no collapse. Leave RESUME_FROM unset for a clean round; set it to warm-start instead.
set -u
ROOT=${ROOT:?set ROOT to the repo path}
DATA=${DATA:?set DATA to the step6 reference dir holding <clip_stem>.pkl}
MESH=${MESH:?set MESH to the mesh dir}
LOGS=${LOGS:?set LOGS to the stdout log dir}
PY=${PY:?set PY to the python}
CFG=${CFG:?set CFG to the reward yaml}
CLIPS=${CLIPS:?set CLIPS}
ENVS=${ENVS:-2048}
ITERS=${ITERS:-12000}
SEED=${SEED:-1}
RESUME_FROM=${RESUME_FROM:-}

TABLE_REMOVE_AFTER_CF=${TABLE_REMOVE_AFTER_CF:-50}
TABLE_REMOVE_DROP=${TABLE_REMOVE_DROP:-0.30}
OBJ_REF_STILL_UNTIL_CF=${OBJ_REF_STILL_UNTIL_CF:-1}
OBJ_REF_WINDOW_ENABLE=${OBJ_REF_WINDOW_ENABLE:-0}
TIP_CF_MISS_ENABLE=${TIP_CF_MISS_ENABLE:-0}
RSI_ANCHOR_CF=${RSI_ANCHOR_CF:-1}
RSI_CF_OFFSET_START=${RSI_CF_OFFSET_START:--20}
RSI_CF_OFFSET_END=${RSI_CF_OFFSET_END:--10}

mkdir -p "$LOGS"
cd "$ROOT" || exit 1

for item in $CLIPS; do
  gpu="${item%%:*}"; rest="${item#*:}"
  clip="${rest%%:*}"; rest="${rest#*:}"
  mesh="${rest%%:*}"; tag="${rest#*:}"

  resume=""
  if [ -n "$RESUME_FROM" ]; then
    src="$RESUME_FROM/$(echo "$tag" | sed 's/^R[0-9]*_/R29_/')"
    it=$(ls "$src"/model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
    [ -n "$it" ] && resume="--resume $src/model_$it.pt"
    [ -z "$it" ] && echo "no checkpoint under $src -- $tag starts from scratch"
  fi

  CUDA_VISIBLE_DEVICES="$gpu" \
  APPLE_HAND_KIND=wuji APPLE_SCENE_Z_OFFSET=-0.03 ASTRA_BASE_BACKEND=torch \
  APPLE_OBJECT_PER_WORLD=1 MIX_PMCP_EVERY=0 PYTHONUNBUFFERED=1 \
  TABLE_REMOVE_AFTER_CF="$TABLE_REMOVE_AFTER_CF" TABLE_REMOVE_DROP="$TABLE_REMOVE_DROP" \
  OBJ_REF_STILL_UNTIL_CF="$OBJ_REF_STILL_UNTIL_CF" OBJ_REF_WINDOW_ENABLE="$OBJ_REF_WINDOW_ENABLE" \
  TIP_CF_MISS_ENABLE="$TIP_CF_MISS_ENABLE" \
  RSI_ANCHOR_CF="$RSI_ANCHOR_CF" RSI_CF_OFFSET_START="$RSI_CF_OFFSET_START" \
  RSI_CF_OFFSET_END="$RSI_CF_OFFSET_END" \
  setsid nohup "$PY" tools/run/train_newton.py \
    --xml assets/scene_stapler/scene.xml \
    --reference-pkl "$DATA/${clip}.pkl" \
    --sdf-object "$MESH/${mesh}.stl" \
    --native-contacts --rigid-object-table --table-under-object --cuda-graph \
    --num-envs "$ENVS" --iterations "$ITERS" --seed "$SEED" \
    --reward-cfg "$CFG" \
    --solver-kwargs '{"impratio": 20.0, "cone": "pyramidal", "iterations": 100, "ls_iterations": 50}' \
    --log-root logs/rsl_rl --run-name "$tag" $resume \
    > "$LOGS/$tag.log" 2>&1 < /dev/null &
  disown
  echo "launched $tag on GPU $gpu  ${resume:-（from scratch）}"
  sleep 3
done
sleep 30
echo "alive: $(pgrep -fc train_newton.py)"
