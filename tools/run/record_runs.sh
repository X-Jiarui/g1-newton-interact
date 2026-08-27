#!/usr/bin/env bash
# Record one mp4 per training run with NEWTON'S OWN renderer.
#
# Why this exists: the obvious way to make a video -- dump qpos and replay it through mjlab's
# MuJoCo model -- shows the right trajectory in the wrong scene. Newton's model is what the physics
# holds (real object mesh, real table, SDF colliders); mjlab's carries the visual meshes. Replaying
# into mjlab borrows the appearance and quietly changes the object, the table and the collision
# geometry, so the video stops being evidence about the run. `--newton-video` draws the model the
# physics actually integrates, which is the only version worth looking at.
#
# The rollout itself reuses train_newton.py, so the contact recipe, object mesh, table height and
# reference are the ones the run was TRAINED with -- see tools/setup/patch_mjlab.py for what
# happens when an eval re-derives those flags instead.
#
# Three modes:
# Recording is always --rollout-free-run: the terminations and the episode time limit are off, so
# the clip plays out as one continuous attempt instead of a sequence of resets mid-grasp.
#
#   auto      every train_newton.py process alive on this box       (default)
#   --runs    explicit names, for runs that have already finished   (needs --log-root)
#   --mix     one mixed checkpoint rolled out on each of its clips
#
# Examples:
#   tools/run/record_runs.sh --out /root/nvideos
#   tools/run/record_runs.sh --out ~/nvideos --log-root logs/rsl_rl --runs FRICTION_PYRAMIDAL,FRICTION_RESUME2000 \
#                            --pkl $D/s8/stapler_pass_1.pkl --stl $M/stapler.stl
#   tools/run/record_runs.sh --out /workspace/nvideos --mix MIX8_BIG --data $D --mesh $M \
#                            --clips s1/hammer_use_2:hammer,s8/hammer_use_2:hammer
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-/opt/mjlab_venv/bin/python}"
OUT=""; XML=""; STEPS=500; RUNS=""; LOGROOT=""; PKL=""; STL=""
MIX=""; DATA=""; MESH=""; CLIPS=""

while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --python) PY="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --xml) XML="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --log-root) LOGROOT="$2"; shift 2 ;;
    --pkl) PKL="$2"; shift 2 ;;
    --stl) STL="$2"; shift 2 ;;
    --mix) MIX="$2"; shift 2 ;;
    --data) DATA="$2"; shift 2 ;;
    --mesh) MESH="$2"; shift 2 ;;
    --clips) CLIPS="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[ -n "$OUT" ] || { echo "--out is required" >&2; exit 2; }
[ -n "$XML" ] || XML="$REPO/assets/scene_stapler/scene.xml"
mkdir -p "$OUT"

export APPLE_HAND_KIND=${APPLE_HAND_KIND:-wuji}
export APPLE_SCENE_Z_OFFSET=${APPLE_SCENE_Z_OFFSET:--0.03}
export ASTRA_BASE_BACKEND=${ASTRA_BASE_BACKEND:-torch}
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# One rollout -> one mp4. Every caller goes through here so the flags cannot drift between modes.
record() {  # name, checkpoint, pkl, stl, extra_env
  local name="$1" ck="$2" pkl="$3" stl="$4" extra="${5:-}"
  local vid="$OUT/${name}_newton.mp4"
  if [ -f "$vid" ]; then echo "have $name"; return 0; fi
  echo "--- $name"
  ( cd "$REPO" && env $extra "$PY" tools/run/train_newton.py \
      --xml "$XML" --reference-pkl "$pkl" --sdf-object "$stl" \
      --native-contacts --rigid-object-table --table-under-object \
      --solver-kwargs '{"impratio": 1.0, "cone": "pyramidal"}' \
      --num-envs 1 --iterations 0 --seed 1 --resume "$ck" \
      --rollout-free-run --rollout-steps "$STEPS" \
      --newton-video "$vid" --video-steps "$STEPS" \
      --log-root /tmp/record_runs --run-name REC ) > "$OUT/$name.log" 2>&1 || true
  # The rollout prints the run's own contact/lift/stand. Compare them with the training row before
  # trusting the video: a picture is an opinion, these three numbers are a measurement.
  grep -aE "Stage/physical_contact|PhaseA/lift_success|stable_not_fallen" "$OUT/$name.log" | tail -3 || true
  if [ -f "$vid" ]; then echo "  ok  $vid"; else echo "  FAILED"; tail -4 "$OUT/$name.log"; fi
}

latest_ck() {  # log-root, run name -> path
  ls -1 "$1/g1_residual_interact/$2"/model_*.pt 2>/dev/null \
    | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1
}

if [ -n "$MIX" ]; then
  [ -n "$DATA" ] && [ -n "$MESH" ] && [ -n "$CLIPS" ] || {
    echo "--mix needs --data, --mesh and --clips" >&2; exit 2; }
  LOGROOT="${LOGROOT:-logs/rsl_rl}"
  it=$(cd "$REPO" && latest_ck "$LOGROOT" "$MIX")
  [ -n "$it" ] || { echo "no checkpoint for $MIX under $LOGROOT" >&2; exit 1; }
  echo "=== $MIX iter=$it, one video per clip"
  IFS=',' read -ra items <<< "$CLIPS"
  for it_pair in "${items[@]}"; do
    clip="${it_pair%%:*}"; mesh="${it_pair##*:}"
    record "${MIX}_iter${it}_$(echo "$clip" | tr '/' '_')" \
           "$REPO/$LOGROOT/g1_residual_interact/$MIX/model_$it.pt" \
           "$DATA/$clip.pkl" "$MESH/$mesh.stl" \
           "APPLE_OBJECT_ENTITIES=1 MIX_PMCP_RULE=none MIX_PMCP_EVERY=0"
  done
elif [ -n "$RUNS" ]; then
  [ -n "$LOGROOT" ] && [ -n "$PKL" ] && [ -n "$STL" ] || {
    echo "--runs needs --log-root, --pkl and --stl" >&2; exit 2; }
  IFS=',' read -ra names <<< "$RUNS"
  for n in "${names[@]}"; do
    it=$(cd "$REPO" && latest_ck "$LOGROOT" "$n")
    [ -n "$it" ] || { echo "no checkpoint for $n"; continue; }
    record "${n}_iter${it}" "$REPO/$LOGROOT/g1_residual_interact/$n/model_$it.pt" "$PKL" "$STL"
  done
else
  # Read the live processes rather than a launcher script, so a run started by hand is covered too.
  "$PY" - > /tmp/record_runs.json <<'PYEOF'
import json, os, re, sys, glob
def argv(pid):
    try:
        return open(f"/proc/{pid}/cmdline","rb").read().decode(errors="replace").split("\0")
    except OSError:
        return []
def flag(a, n, d=None):
    for i, x in enumerate(a):
        if x == n and i + 1 < len(a): return a[i+1]
        if x.startswith(n + "="): return x.split("=",1)[1]
    return d
out = []
for p in (q for q in os.listdir("/proc") if q.isdigit()):
    a = argv(p)
    if not any("train_newton.py" in x for x in a): continue
    name = flag(a, "--run-name")
    pkl = flag(a, "--reference-pkl")
    if not name or not pkl: continue          # mixed runs need --mix, they have no single clip
    d = os.path.join(os.path.realpath(f"/proc/{p}/cwd"),
                     flag(a, "--log-root", "logs/rsl_rl"), "g1_residual_interact", name)
    cks = glob.glob(os.path.join(d, "model_*.pt"))
    if not cks: continue
    ck = max(cks, key=lambda q: int(re.search(r"model_(\d+)\.pt", q).group(1)))
    out.append({"name": name, "iter": int(re.search(r"model_(\d+)\.pt", ck).group(1)),
                "ck": ck, "pkl": pkl, "stl": flag(a, "--sdf-object")})
json.dump(sorted(out, key=lambda r: r["name"]), sys.stdout)
PYEOF
  n=$("$PY" -c "import json;print(len(json.load(open('/tmp/record_runs.json'))))")
  echo "=== $n live run(s) -> $OUT"
  for i in $(seq 0 $((n-1))); do
    eval "$("$PY" - "$i" <<'PYEOF'
import json, sys
r = json.load(open('/tmp/record_runs.json'))[int(sys.argv[1])]
for k in ("name","iter","ck","pkl","stl"):
    print(f'R_{k.upper()}={r[k]}')
PYEOF
)"
    record "${R_NAME}_iter${R_ITER}" "$R_CK" "$R_PKL" "$R_STL"
  done
fi
echo ALLDONE
