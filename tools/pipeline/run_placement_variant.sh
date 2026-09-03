#!/usr/bin/env bash
# Steps 4-6 for ONE placement variant. Its own DATA_ROOT, and the clip keeps its original
# relative path s1/cubesmall_lift.pkl so extract_grasp_targets can still find the GRAB source.
set -euo pipefail
# TWO interpreters, and they are not interchangeable.
#   PY    steps 4, 5a and 6  -- needs smplx + trimesh + mujoco
#   PY_IK step 5b            -- needs newton (CUDA/Warp), which is a different env here
# The env running steps 4-6 must also be able to UNPICKLE numpy-2 arrays: the reference set was
# written under numpy 2, and numpy 1.23 fails on `numpy._core` while numpy >= 1.26.1 ships the
# compat shim and reads it. Getting this wrong is silent -- extract_grasp_targets catches every
# exception per clip and continues, so a missing module reports as "0 sequences solved" three
# steps later, with no mention of the import that actually failed.
TAG=${1:?tag}
export GMR_ROOT=/home/jiarui/jiarui/GMR
export GRAB_DIR=/home/jiarui/jiarui/Omnigrasp/GRAB
export DATA_ROOT=/home/jiarui/cube_rand/v/$TAG
export IN=$DATA_ROOT/in
export PY=/home/jiarui/miniconda3/envs/wilor/bin/python
export PY_IK=/home/jiarui/miniconda3/envs/newton/bin/python
bash /home/jiarui/grab-g1-wuji-pipeline/run_grasp_pipeline.sh
