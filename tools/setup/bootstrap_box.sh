#!/usr/bin/env bash
# Provision a fresh CUDA box for this project.
#
#   ./tools/setup/bootstrap_box.sh --from user@host:/path/to/source-box
#
# Code comes from this repo (git clone). Two things cannot: the mjlab checkout that defines the
# task, and the retargeted GRAB data. Both are pulled from --from over rsync.
#
# Idempotent: re-running skips whatever is already in place.
set -euo pipefail

FROM=""
VENV=/opt/mjlab_venv
DATA=/workspace/data
MJLAB=/workspace/mjlab-run
PY=3.13

while [ $# -gt 0 ]; do
  case "$1" in
    --from)  FROM="$2"; shift 2 ;;
    --venv)  VENV="$2"; shift 2 ;;
    --data)  DATA="$2"; shift 2 ;;
    --mjlab) MJLAB="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[ -n "$FROM" ] || { echo "--from user@host:/path is required" >&2; exit 2; }
SRC_HOST="${FROM%%:*}"
SRC_PATH="${FROM#*:}"

say () { printf '\n=== %s\n' "$*"; }

say "1/5 uv"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
export PATH="$HOME/.local/bin:$PATH"

say "2/5 venv at $VENV (python $PY)"
[ -x "$VENV/bin/python" ] || uv venv --python "$PY" "$VENV"

say "3/5 pinned deps"
# pyglet backs Newton's headless ViewerGL, so record_runs.sh needs it; without it every
# recording on a box dies at import with ModuleNotFoundError and training is unaffected, which
# makes it easy to miss until you go looking for videos.
# These versions are what every number in docs/RECIPE.md was measured on. newton and mujoco_warp
# in particular are not interchangeable across minor versions -- the contact pipeline changes.
VIRTUAL_ENV="$VENV" uv pip install \
  "torch==2.9.0" \
  "warp-lang==1.16.0" \
  "newton==1.5.0" \
  "mujoco==3.11.0" \
  "mujoco-warp==3.11.0" \
  "rsl-rl-lib==5.2.0" \
  "trimesh==4.8.3" "viser==1.0.27" "numpy==2.3.4" "scipy==1.16.2" \
  "pyglet==2.1.16" \
  "tensorboard" "pyyaml" "tqdm"

say "4/5 mjlab source from $FROM"
mkdir -p "$MJLAB"
rsync -a --info=stats1 "$SRC_HOST:$SRC_PATH/mjlab-src/" "$MJLAB/src/"
SITE=$("$VENV/bin/python" -c 'import site;print(site.getsitepackages()[0])')
echo "$MJLAB/src" > "$SITE/mjlab.pth"
"$VENV/bin/python" -c "import mjlab,os;print('mjlab ->',os.path.dirname(mjlab.__file__))"
# mjlab is rsynced, not versioned, so a fix that lives only in one box's copy is lost on the next
# bootstrap. Mixed-clip training is wrong without this one; it is idempotent and a no-op for
# single-clip work.
"$VENV/bin/python" "$(dirname "$0")/patch_mjlab.py" "$MJLAB/src"

say "5/5 data from $FROM"
mkdir -p "$DATA"
for d in grab_g1_wuji_aligned scaled_grab_dataset_wuji scaled_grab_wuji_all_o70; do
  [ -d "$DATA/$d" ] && { echo "  have $d"; continue; }
  rsync -a --info=stats1 "$SRC_HOST:$SRC_PATH/$d/" "$DATA/$d/"
done

cat <<TXT

done. before running anything:

  export APPLE_HAND_KIND=wuji APPLE_SCENE_Z_OFFSET=-0.03 ASTRA_BASE_BACKEND=torch
  export DATA=$DATA
  export PATH="$VENV/bin:\$PATH"

verify the mjlab you import matches the box the checkpoints came from (README 1.3).
TXT
