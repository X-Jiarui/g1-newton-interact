#!/usr/bin/env bash
# Launch one single-clip from-scratch run per GPU, taking the clips off a travel-ranked manifest.
#
#   ./tools/run/launch_clip_sweep.sh --manifest data/top100_travel.json --top 8 --iterations 6000
#
# Every run uses the recipe in docs/RECIPE.md. One GPU per run, GPU i gets clip i.
set -euo pipefail

MANIFEST=data/top100_travel.json
TOP=8
ITERS=6000
ENVS=2048
SEED=1
LOGROOT=logs/rsl_rl
DATA=${DATA:-/workspace/data}
PY=${PY:-/opt/mjlab_venv/bin/python}

while [ $# -gt 0 ]; do
  case "$1" in
    --manifest) MANIFEST="$2"; shift 2 ;;
    --top) TOP="$2"; shift 2 ;;
    --iterations) ITERS="$2"; shift 2 ;;
    --num-envs) ENVS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --log-root) LOGROOT="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

export APPLE_HAND_KIND=wuji APPLE_SCENE_Z_OFFSET=-0.03 ASTRA_BASE_BACKEND=torch
mkdir -p logs

# The manifest records each clip's own pkl path from wherever it was ranked; rebuild it against
# $DATA here so a manifest made on another box still resolves.
mapfile -t ROWS < <("$PY" - "$MANIFEST" "$TOP" "$DATA" <<'PY'
import json, sys
man, top, data = sys.argv[1], int(sys.argv[2]), sys.argv[3]
d = json.load(open(man))
clips = d["top"] if isinstance(d, dict) else d
for c in clips[:top]:
    pkl = f'{data}/grab_g1_wuji_aligned/{c["subject"]}/{c["sequence"]}.pkl'
    stl = f'{data}/scaled_grab_dataset_wuji/meshes/{c["object"]}.stl'
    name = f'{c["subject"]}_{c["sequence"]}'.upper()
    print(f'{name}\t{pkl}\t{stl}')
PY
)

n=0
for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r NAME PKL STL <<<"$row"
  [ -f "$PKL" ] || { echo "MISSING clip $PKL -- skipping $NAME" >&2; continue; }
  [ -f "$STL" ] || { echo "MISSING mesh $STL -- skipping $NAME" >&2; continue; }
  LOG=logs/run_${NAME}.log
  echo "GPU $n  $NAME"
  CUDA_VISIBLE_DEVICES=$n setsid nohup "$PY" tools/run/train_newton.py \
    --xml assets/scene_stapler/scene.xml \
    --reference-pkl "$PKL" \
    --sdf-object "$STL" \
    --native-contacts --rigid-object-table --table-under-object --cuda-graph \
    --solver-kwargs '{"impratio": 1.0, "cone": "pyramidal"}' \
    --num-envs "$ENVS" --iterations "$ITERS" --seed "$SEED" \
    --log-root "$LOGROOT" --run-name "$NAME" \
    </dev/null > "$LOG" 2>&1 &
  n=$((n+1))
  sleep 3
done
echo "launched $n run(s)"
