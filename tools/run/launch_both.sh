#!/usr/bin/env bash
# Two single-sequence runs, one per GPU: a non-convex object collided through its own SDF, with
# mjlab's MDP unchanged. stapler and mug are different kinds of concavity -- an open gap (73.7% hull
# excess) and a handle hole (252%) -- so they test the representation from two directions.
set -uo pipefail
cd ~/projects/g1-newton-interact
export APPLE_HAND_KIND=wuji APPLE_SCENE_Z_OFFSET=-0.03 ASTRA_BASE_BACKEND=torch
export GMR_ROOT=$HOME/jiarui/GMR
P=$HOME/miniconda3/envs/newton/bin/python
D=$HOME/jiarui/grab_g1_wuji_aligned
M=$HOME/jiarui/scaled_grab_wuji_all_o70/meshes
ENVS=${ENVS:-256}
ITERS=${ITERS:-2000}

launch() {
  local gpu=$1 name=$2 scene=$3 clip=$4 stl=$5
  CUDA_VISIBLE_DEVICES=$gpu nohup $P tools/run/train_newton.py \
    --xml assets/scene_${scene}/scene.xml \
    --reference-pkl "$D/$clip.pkl" \
    --sdf-object "$M/$stl" \
    --num-envs $ENVS --iterations $ITERS \
    --run-name "NEWTON_SDF_${name}" > /tmp/train_${name}.log 2>&1 &
  echo "  gpu$gpu  $name  <- $clip  ($stl)  pid $!"
}

launch 0 STAPLER stapler s8/stapler_pass_1 stapler.stl
sleep 20
launch 1 MUG     mug     s1/mug_drink_4    mug.stl
echo "both launched: $ENVS envs, $ITERS iterations"
