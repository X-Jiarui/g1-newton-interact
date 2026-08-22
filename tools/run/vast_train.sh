#!/usr/bin/env bash
# Launch one Newton training run on the vast box.
#   usage: vast_train.sh <stapler|mug> <gpu> <num_envs> <iterations> <seed> <run_name> [extra args...]
# Kept as a file rather than an inline ssh command because the path to this box is two SSH hops
# deep, and every layer of nesting eats another round of shell quoting.
set -euo pipefail
OBJ=$1; GPU=$2; NENV=$3; ITERS=$4; SEED=$5; NAME=$6; shift 6
EXTRA=("$@")

DATA=/workspace/data
REPO=/root/g1-newton-interact
case "$OBJ" in
  stapler) SCENE=$REPO/assets/scene_stapler/scene.xml
           CLIP=$DATA/grab_g1_wuji_aligned/s8/stapler_pass_1.pkl ;;
  mug)     SCENE=$REPO/assets/scene_mug/scene.xml
           CLIP=$DATA/grab_g1_wuji_aligned/s1/mug_drink_4.pkl ;;
  *) echo "unknown object: $OBJ" >&2; exit 2 ;;
esac
MESH=$DATA/scaled_grab_wuji_all_o70/meshes/$OBJ.stl

for f in "$SCENE" "$CLIP" "$MESH"; do
  [ -f "$f" ] || { echo "missing input: $f" >&2; exit 3; }
done

cd "$REPO"
LOG=/root/logs/newton_${NAME}.log
mkdir -p /root/logs
nohup env APPLE_HAND_KIND=wuji APPLE_SCENE_Z_OFFSET=-0.03 ASTRA_BASE_BACKEND=torch \
  CUDA_VISIBLE_DEVICES="$GPU" \
  /opt/mjlab_venv/bin/python tools/run/train_newton.py \
    --xml "$SCENE" --reference-pkl "$CLIP" --sdf-object "$MESH" \
    --num-envs "$NENV" --iterations "$ITERS" --seed "$SEED" --run-name "$NAME" \
    "${EXTRA[@]}" \
    > "$LOG" 2>&1 &
echo "launched $NAME (obj=$OBJ gpu=$GPU envs=$NENV iters=$ITERS seed=$SEED extra=${EXTRA[*]:-none}) pid=$! log=$LOG"
