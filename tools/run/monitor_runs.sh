#!/usr/bin/env bash
# Append one line per run every INTERVAL seconds. Reads the stdout logs, not tensorboard.
#   ./tools/run/monitor_runs.sh logs/monitor.txt 600 NAME1 NAME2 ...
OUT=${1:-logs/monitor.txt}; INT=${2:-600}; shift 2 || true
RUNS=("$@")
row () {
  local log=logs/run_$1.log it rew
  [ -f "$log" ] || { printf "%-28s (no log)\n" "$1"; return; }
  it=$(grep -a "Learning iteration" "$log" | tail -1 | grep -oE "[0-9]+/[0-9]+")
  rew=$(grep -a "Mean reward:" "$log" | tail -1 | grep -oE "[-0-9.]+$")
  printf "%-28s iter %-11s rew %-10s" "$1" "${it:-?}" "${rew:-?}"
  for k in contact:Stage/physical_contact lift:Episode_Metrics/lift_success \
           liftA:PhaseA/lift_success seq:PhaseA/sequence_success \
           stand:Stage/stable_not_fallen ep_len:Episode_Metrics/ep_len \
           objfar:Episode_Termination/og_object_far nonfin:Health/nonfinite_worlds; do
    local v; v=$(grep -a "${k#*:}:" "$log" | tail -1 | grep -oE "[-0-9.]+$")
    printf "%s=%-8s" "${k%%:*}" "${v:-0}"
  done
  # count>1 because the grep pattern itself shows up in the ps output
  [ "$(ps -eo cmd | grep -ac -- "run-name $1")" -gt 1 ] || printf " [DEAD]"
  printf "\n"
}
while true; do
  { echo "=== $(date '+%m-%d %H:%M') ==="; for r in "${RUNS[@]}"; do row "$r"; done; } >> "$OUT" 2>&1
  sleep "$INT"
done
