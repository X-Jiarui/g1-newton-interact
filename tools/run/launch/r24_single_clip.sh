#!/bin/bash
# R24, single-clip arm. One clip per GPU, trained FROM SCRATCH on the rewritten approach reward.
#
# Every path is an environment variable because the three boxes disagree about all of them; the
# clip list is the only thing that changes between boxes, and it is passed in as CLIPS.
#
# CLIPS entries are `gpu:clip_path:mesh_stem:tag`, whitespace separated. Example:
#   CLIPS="0:s8/stapler_pass_1:stapler:R24_S8_STAPLER_PASS_1"
#
# Required: ROOT DATA MESH LOGS PY CFG CLIPS
# Optional: ENVS (2048) ITERS (12000) SEED (1)
#
# ITERS is 12000, not the 6000 R20/R21 used. Those were `--resume` continuations sitting on top of
# earlier rounds; this round starts at iteration 0, and the historical first-lift iterations are
# 542/617/760 with the run needing to keep going well past them. 6000 from scratch would end the
# round before the question is answered.
set -u
ROOT=${ROOT:?set ROOT to the repo path}
DATA=${DATA:?set DATA to the step6 reference dir}
MESH=${MESH:?set MESH to the mesh dir}
LOGS=${LOGS:?set LOGS}
PY=${PY:?set PY}
CFG=${CFG:?set CFG to the reward yaml}
CLIPS=${CLIPS:?set CLIPS}
ENVS=${ENVS:-2048}
ITERS=${ITERS:-12000}
SEED=${SEED:-1}
cd "$ROOT" || exit 1
mkdir -p "$LOGS"

# Record what this run is actually made of. Runs have drifted before because a box was on older
# code than the log implied, and nothing in the log said so.
echo "R24 launch  repo=$(git rev-parse --short HEAD 2>/dev/null || echo UNKNOWN)  cfg=$CFG  envs=$ENVS  iters=$ITERS" | tee "$LOGS/_launch_manifest.txt"

go() {  # gpu clip object tag
  CUDA_VISIBLE_DEVICES=$1 \
  APPLE_HAND_KIND=wuji APPLE_SCENE_Z_OFFSET=-0.03 ASTRA_BASE_BACKEND=torch \
  SIM_TIMESTEP=0.0025 FINGER_FORCE_LIMIT=0.6 OBJECT_FRICTION=0.6 HAND_FRICTION=0.6 \
  HAND_COLLISION_FIX=1 TIP_CF_MISS_ENABLE=0 \
  nohup $PY tools/run/train_newton.py \
    --xml assets/scene_stapler/scene.xml \
    --reference-pkl "$DATA/$2.pkl" --sdf-object "$MESH/$3.stl" \
    --native-contacts --rigid-object-table --table-under-object --cuda-graph \
    --num-envs "$ENVS" --iterations "$ITERS" --seed "$SEED" \
    --reward-cfg "$CFG" \
    --solver-kwargs '{"impratio": 20.0, "cone": "pyramidal", "iterations": 100, "ls_iterations": 50}' \
    --log-root logs/rsl_rl --run-name "$4" \
    > "$LOGS/$4.log" 2>&1 &
  echo "GPU$1  $4  <- $2 / $3   (from scratch, $ENVS envs, $ITERS iters)" | tee -a "$LOGS/_launch_manifest.txt"
}

for spec in $CLIPS; do
  IFS=: read -r g c o t <<< "$spec"
  go "$g" "$c" "$o" "$t"
done
