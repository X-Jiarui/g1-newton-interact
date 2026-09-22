#!/usr/bin/env bash
# Film one arm of the zspace A/B on one clip, 4x5090 box.  vid_zs.sh <arm> <clip k> <gpu> [steps]
#   arm: ACT | Z | Z210 | ZP | ZP390 | NOREF220 | OURLR290 | OURLR330   (checkpoints /workspace/h200/<CK>.pt)
# Same recipe as the H200's vid_h200.sh (rollout-video-recipe): the run's own config, the 8 clips in
# training order, 64 envs, ROLLOUT_START_FRAME=rsi, ROLLOUT_NO_TERM=1, VIDEO_CLIP=k, camera follows.
# ZN arms (OURLR/NOREF) trained with ZSPACE_SAMPLE_Z=1 (policy Gaussian over z); NOREF used the
# agent_noref config (no reference_phase / reference_preview / tracking_error in the actor input).
cd /workspace/h200/g1-newton-interact || exit 1
arm=$1; k=$2; g=$3; steps=${4:-300}
declare -A CK=([ACT]=ZS_ACT_220 [Z]=ZS_Z_210 [ZP]=ZS_ZP_140 [ZP390]=ZS_ZP_390 [Z210]=ZS_Z_210 [NOREF220]=ZS_NOREF_220 [OURLR290]=ZS_OURLR_290 [OURLR330]=ZS_OURLR_330)
declare -A N=([0]=cubesmall [1]=phone [2]=gamecontroller [3]=binoculars [4]=hammer [5]=camera [6]=banana [7]=flashlight)
ZS="ZSPACE_STUDENT=/workspace/student_004000.pt ZSPACE_REF_DEFAULT=/workspace/ref_default_1324.npz"
AG=/workspace/h200/agent_handoff
case $arm in
  ACT) X="";;
  Z|Z210) X="$ZS ZSPACE_ANCHOR=encoder";;
  ZP|ZP390) X="$ZS ZSPACE_ANCHOR=prior";;
  OURLR290|OURLR330) X="$ZS ZSPACE_ANCHOR=prior ZSPACE_SAMPLE_Z=1";;
  NOREF220) X="$ZS ZSPACE_ANCHOR=prior ZSPACE_SAMPLE_Z=1"; AG=/workspace/h200/agent_noref;;
  *) echo "bad arm"; exit 2;;
esac
C=/home/jrxu/mix8/clips; M=/home/jrxu/mix8/meshes
P="$C/s1/cubesmall_inspect_1.pkl,$C/s1/phone_call_1.pkl,$C/s1/gamecontroller_play_1.pkl,$C/s1/binoculars_see_1.pkl,$C/s1/hammer_use_3.pkl,$C/s1/camera_takepicture_2.pkl,$C/s10/banana_eat_1.pkl,$C/s1/flashlight_on_2.pkl"
ST="$M/cubesmall.stl,$M/phone.stl,$M/gamecontroller.stl,$M/binoculars.stl,$M/hammer.stl,$M/camera.stl,$M/banana.stl,$M/flashlight.stl"
mkdir -p /workspace/vidshot
out=/workspace/vidshot/${CK[$arm]}_${N[$k]}
env -i PATH=/opt/mjlab_venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64 HOME=/home/jrxu LANG=en_US.UTF-8 \
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$g PYTHONPATH=/workspace/h200/mjlab-run/src \
  SEED_CUBE=/home/jrxu/seed_cube GMR_ROOT=/home/jrxu/GMR HAND_PD="10,0.2" \
  RSI_ANCHOR_CF=1 RSI_CF_OFFSET_START=-1000 RSI_CF_OFFSET_END=-20 WRIST_TARGET_FAR=0 FIXED_HAND_ZERO=1 \
  ROLLOUT_START_FRAME=rsi ROLLOUT_NO_TERM=1 ROLLOUT_TIMELINE=$out.csv VIDEO_CLIP=$k VIDEO_FOLLOW=1 $X \
  /opt/mjlab_venv/bin/python tools/run/train_newton.py --config configs/train/best.yaml \
    --agent-cfg-from $AG/model_7310.pt \
    --reference-pkls "$P" --sdf-objects "$ST" --num-envs 64 --iterations 0 \
    --rollout-steps $steps --video-steps $steps --seed 1 \
    --resume /workspace/h200/${CK[$arm]}.pt \
    --newton-video $out.mp4 --video-size 1280x960 \
    --run-name VID_${CK[$arm]}_${N[$k]} > $out.log 2>&1
echo "done $out.mp4: $(ls -la $out.mp4 2>/dev/null | awk '{print $5}') bytes"
