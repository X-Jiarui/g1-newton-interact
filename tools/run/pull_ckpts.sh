#!/bin/bash
# Pull the newest checkpoint of every recently-active run off a rented box, onto this Mac.
#
# A vast.ai instance can disappear with a few hours' notice and takes its disk with it. Code is
# already safe -- it lives in git and is pushed -- so this only has to rescue what git does not
# hold: the .pt weights and the training logs that carry the metric history.
#
# Runs are DISCOVERED, never listed: any directory under the checkpoint root whose newest
# `model_*.pt` was written in the last $MAX_AGE_H hours is pulled. A run started by hand after this
# script was written is therefore covered, and a finished round stops being copied on its own.
#
# Only the newest checkpoint per run is fetched. The intermediate ones are reproducible from it in
# a way the round's final weights are not, and pulling all of them is ~10 GB per sweep.
#
#   tools/run/pull_ckpts.sh                 # default box, default destination
#   DEST=/Volumes/ext/ckpt tools/run/pull_ckpts.sh
#
# Safe to run while training: rsync reads, and a checkpoint is written atomically by torch.save to
# a temp path first, so a half-written file is never named model_*.pt.
#
# Measured: 60 MB in 7m39s (~130 KB/s) off this box, so one sweep of four live runs is ~30 min.
# Keep the schedule interval comfortably above that -- overlapping sweeps just fight each other.
#
# A stale lock left by a killed sweep is ignored after 2 h rather than blocking every later run.
set -u

SSH_PORT=${SSH_PORT:-45219}
SSH_HOST=${SSH_HOST:-root@211.72.13.202}
CKPT_ROOT=${CKPT_ROOT:-/workspace/g1-newton-interact/logs/rsl_rl/g1_residual_interact}
LOG_DIRS=${LOG_DIRS:-/workspace/logs_r32}
DEST=${DEST:-$HOME/Documents/g1-newton-interact/ckpt_backup}
# 6 h, not 24: the box clock ran ~20 h behind the Mac's at the time of writing, so a 24 h window
# swept in four dead rounds (R26/R28/R29/R30) alongside the live one. The link is ~130 KB/s, so
# each extra run costs ~8 minutes of a sweep that has to finish inside its own interval.
MAX_AGE_H=${MAX_AGE_H:-6}
KEEP=${KEEP:-2}                # newest N checkpoints to retain locally, per run
SSH="ssh -o ConnectTimeout=45 -n -p $SSH_PORT"

mkdir -p "$DEST"
LOCK="$DEST/.pull.lock"
if [ -d "$LOCK" ] && [ -z "$(find "$LOCK" -maxdepth 0 -mmin +120 2>/dev/null)" ]; then
  echo "[$(date "+%Y-%m-%d %H:%M:%S")] a sweep is still running ($LOCK) -- skipping"
  exit 0
fi
rm -rf "$LOCK"; mkdir -p "$LOCK"
trap 'rm -rf "$LOCK"' EXIT
stamp() { date "+%Y-%m-%d %H:%M:%S"; }
echo "[$(stamp)] pulling from $SSH_HOST:$CKPT_ROOT -> $DEST"

# One remote call decides the whole work list: newest model_*.pt per run, recent ones only.
# `-mmin` rather than a parsed mtime so the filtering happens where the files are.
list=$($SSH "$SSH_HOST" "
  for d in $CKPT_ROOT/*/; do
    f=\$(ls -t \"\$d\"model_*.pt 2>/dev/null | head -1)
    [ -n \"\$f\" ] || continue
    find \"\$f\" -mmin -$((MAX_AGE_H * 60)) -print 2>/dev/null
  done
" 2>/dev/null | grep -avE "^Welcome|^Have fun|AI agents:")

if [ -z "$list" ]; then
  echo "[$(stamp)] no run has written a checkpoint in the last ${MAX_AGE_H}h -- nothing to pull"
  exit 0
fi

n=0
while IFS= read -r remote; do
  [ -n "$remote" ] || continue
  run=$(basename "$(dirname "$remote")")
  mkdir -p "$DEST/$run"
  # --ignore-existing: a checkpoint file never changes once named, so a re-run is a cheap no-op.
  if rsync -a --ignore-existing --partial \
      -e "ssh -o ConnectTimeout=45 -p $SSH_PORT" \
      "$SSH_HOST:$remote" "$DEST/$run/" 2>/dev/null; then
    echo "  $run  <- $(basename "$remote")"
    n=$((n + 1))
  else
    echo "  $run  FAILED $(basename "$remote")"
  fi
  # Keep the round's history bounded; the newest weights are what a lost box actually costs.
  ls -t "$DEST/$run"/model_*.pt 2>/dev/null | tail -n +$((KEEP + 1)) | while read -r old; do
    rm -f "$old" && echo "      pruned $(basename "$old")"
  done
done <<< "$list"

# The logs are small and hold the whole metric history, which no checkpoint contains.
for dir in $LOG_DIRS; do
  mkdir -p "$DEST/_logs"
  rsync -a -e "ssh -o ConnectTimeout=45 -p $SSH_PORT" \
    "$SSH_HOST:$dir/" "$DEST/_logs/$(basename "$dir")/" 2>/dev/null \
    && echo "  logs <- $dir" || echo "  logs FAILED $dir"
done

echo "[$(stamp)] $n checkpoint(s) current; $(du -sh "$DEST" 2>/dev/null | cut -f1) on disk"
