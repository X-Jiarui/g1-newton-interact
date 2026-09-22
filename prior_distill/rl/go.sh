#!/usr/bin/env bash
# gpu1 mirror of the H200's wide_trio.sh `go` (PD10_WIDE_S recipe), hand held fixed (fixed_hand_action_frame=0).
#   go.sh <run-name> <gpu> <seed> [extra env VAR=val ...]
# NENV / ITERS override the 4096 / 12000 defaults (smoke). Arm 2 = ZSPACE_STUDENT=... ZSPACE_ANCHOR=...
cd /home/jrxu/zs/h200/g1-newton-interact || exit 1
export SEED_CUBE=/home/jrxu/seed_cube
C=/home/jrxu/mix8/clips; M=/home/jrxu/mix8/meshes
PKLS="$C/s1/cubesmall_inspect_1.pkl,$C/s1/phone_call_1.pkl,$C/s1/gamecontroller_play_1.pkl,$C/s1/binoculars_see_1.pkl,$C/s1/hammer_use_3.pkl,$C/s1/camera_takepicture_2.pkl,$C/s10/banana_eat_1.pkl,$C/s1/flashlight_on_2.pkl"
STLS="$M/cubesmall.stl,$M/phone.stl,$M/gamecontroller.stl,$M/binoculars.stl,$M/hammer.stl,$M/camera.stl,$M/banana.stl,$M/flashlight.stl"
n=$1 g=$2 seed=$3; shift 3
mkdir -p /home/jrxu/zs/logs
pgrep -u jrxu -f "run-name $n\$" >/dev/null && { echo "$n already running"; exit 0; }
nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$g GMR_ROOT=/home/jrxu/GMR METRICS_EVERY=4 \
  PYTHONPATH=/home/jrxu/zs/h200/mjlab-run/src \
  HAND_PD="10,0.2" RSI_ANCHOR_CF=1 RSI_CF_OFFSET_START=-1000 RSI_CF_OFFSET_END=-20 WRIST_TARGET_FAR=0 "$@" \
  /home/jrxu/zs/venv/bin/python tools/run/train_newton.py --config configs/train/best.yaml \
    --agent-cfg-from ${AGENT:-/home/jrxu/zs/agent_handoff/model_7310.pt} \
    --reference-pkls "$PKLS" --sdf-objects "$STLS" \
    --num-envs ${NENV:-4096} --iterations ${ITERS:-12000} --resume "" --seed $seed \
    --run-name $n > /home/jrxu/zs/logs/$n.log 2>&1 &
echo "launched $n on gpu$g seed=$seed $*"
