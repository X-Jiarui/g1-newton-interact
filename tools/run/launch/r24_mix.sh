#!/bin/bash
# R24, mixed-clip arm. 8 clips, 4 objects x 2 clips so the scheduler has somewhere to move envs.
#   ARM=A  uniform split, PMCP off        -- the baseline
#   ARM=B  Omnigrasp's cumulative-failure histogram over PhaseA/lift_success
#
# Successor to r22.sh: same clip set, same arms, same everything, on the rewritten approach reward
# and from scratch rather than resumed.
#
# Why lift_success as the PMCP metric even though it is 0 everywhere today: with a cumulative
# histogram an all-zero signal gives every clip the same count, so the split is exactly uniform and
# B == A until the first lift appears. That is intended, not degenerate -- before any lift every
# clip should be pushed equally hard toward one; after one clip starts lifting its share decays on
# its own and the budget moves to the clips still stuck.
set -u
ARM=${ARM:?set ARM to A or B}
ROOT=${ROOT:?set ROOT to the repo path}
DATA=${DATA:?set DATA to the hf8 dir}
LOGS=${LOGS:?set LOGS}
PY=${PY:?set PY}
CFG=${CFG:?set CFG to the reward yaml}
ENVS=${ENVS:-2048}
ITERS=${ITERS:-12000}
cd "$ROOT" || exit 1
mkdir -p "$LOGS"
D=$DATA/step6; M=$DATA/meshes
PK=$D/s1/hammer_use_2.pkl,$D/s2/hammer_lift.pkl,$D/s8/stapler_pass_1.pkl,$D/s3/stapler_lift.pkl,$D/s6/stanfordbunny_pass_1.pkl,$D/s2/stanfordbunny_lift.pkl,$D/s6/waterbottle_shake_1.pkl,$D/s3/waterbottle_lift.pkl
ST=$M/hammer.stl,$M/hammer.stl,$M/stapler.stl,$M/stapler.stl,$M/stanfordbunny.stl,$M/stanfordbunny.stl,$M/waterbottle.stl,$M/waterbottle.stl

if [ "$ARM" = "B" ]; then
  PMCP_ENV="MIX_PMCP_RULE=histogram MIX_PMCP_EVERY=2400 MIX_PMCP_METRIC=PhaseA/lift_success MIX_PMCP_QUOTA=64"
else
  PMCP_ENV="MIX_PMCP_EVERY=0"
fi

echo "R24 launch  repo=$(git rev-parse --short HEAD 2>/dev/null || echo UNKNOWN)  cfg=$CFG  arm=$ARM  envs=$ENVS  iters=$ITERS" | tee "$LOGS/_launch_manifest.txt"

go() {  # gpu seed
  env CUDA_VISIBLE_DEVICES=$1 \
  APPLE_HAND_KIND=wuji APPLE_SCENE_Z_OFFSET=-0.03 ASTRA_BASE_BACKEND=torch \
  SIM_TIMESTEP=0.0025 FINGER_FORCE_LIMIT=0.6 OBJECT_FRICTION=0.6 HAND_FRICTION=0.6 \
  HAND_COLLISION_FIX=1 TIP_CF_MISS_ENABLE=0 APPLE_OBJECT_PER_WORLD=1 $PMCP_ENV \
  nohup $PY tools/run/train_newton.py \
    --xml assets/scene_stapler/scene.xml --reference-pkls $PK --sdf-objects $ST \
    --native-contacts --rigid-object-table --table-under-object --cuda-graph \
    --num-envs "$ENVS" --iterations "$ITERS" --seed "$2" --reward-cfg "$CFG" \
    --solver-kwargs '{"impratio": 20.0, "cone": "pyramidal", "iterations": 100, "ls_iterations": 50}' \
    --log-root logs/rsl_rl --run-name "R24_${ARM}_SEED$2" \
    > "$LOGS/R24_${ARM}_SEED$2.log" 2>&1 &
  echo "GPU$1  R24_${ARM}_SEED$2   [$PMCP_ENV]" | tee -a "$LOGS/_launch_manifest.txt"
}
go 0 1
go 1 2
