#!/usr/bin/env bash
# R25 treatment arm: the SAME cube clip at eight different placements, mixed.
#
# Each variant is the object translated 3 cm in world space -- six axis directions plus the two
# horizontal diagonals -- with the table carried by the same vector, and the whole grasp pipeline
# re-solved for it. The IK targets are the human fingertips expressed in the OBJECT frame, so a
# translation leaves them unchanged and the re-solved cf pose holds the same hand-object relation.
# Measured: alpha 1.0 and 0.0 cm of pull-back on all eight, so every requested offset survives.
#
# Control is R25_CTRL_CUBE on the office box: the same clip at zero offset, regenerated through
# this same pipeline invocation so the arms differ by the offset and nothing else.
set -u
D=/workspace/data/cube8/step6/s1; M=/workspace/data/cube8/meshes
PK=""; ST=""
for t in xp xm yp ym zp zm dpp dmm; do
  PK="$PK,$D/cubesmall_$t.pkl"; ST="$ST,$M/cubesmall.stl"
done
PK=${PK#,}; ST=${ST#,}
LOGS=/workspace/logs_r25; mkdir -p "$LOGS"
echo "R25 launch  repo=$(git rev-parse --short HEAD)" | tee "$LOGS/_launch_manifest.txt"
go() {  # gpu seed
  env CUDA_VISIBLE_DEVICES=$1 \
  APPLE_HAND_KIND=wuji APPLE_SCENE_Z_OFFSET=-0.03 ASTRA_BASE_BACKEND=torch \
  SIM_TIMESTEP=0.0025 FINGER_FORCE_LIMIT=0.6 OBJECT_FRICTION=0.6 HAND_FRICTION=0.6 \
  HAND_COLLISION_FIX=1 TIP_CF_MISS_ENABLE=0 APPLE_OBJECT_PER_WORLD=1 MIX_PMCP_EVERY=0 \
  nohup /opt/mjlab_venv/bin/python tools/run/train_newton.py \
    --xml assets/scene_stapler/scene.xml --reference-pkls "$PK" --sdf-objects "$ST" \
    --native-contacts --rigid-object-table --table-under-object --cuda-graph \
    --num-envs 2048 --iterations 12000 --seed $2 \
    --reward-cfg /workspace/g1-newton-interact/configs/rewards/staged_cf_r24.yaml \
    --solver-kwargs "{\"impratio\": 20.0, \"cone\": \"pyramidal\", \"iterations\": 100, \"ls_iterations\": 50}" \
    --log-root logs/rsl_rl --run-name "R25_RAND_CUBE8_SEED$2" \
    > "$LOGS/R25_RAND_CUBE8_SEED$2.log" 2>&1 &
  echo "GPU$1  R25_RAND_CUBE8_SEED$2  (8 placements, from scratch)" | tee -a "$LOGS/_launch_manifest.txt"
}
go 2 1
go 3 2
