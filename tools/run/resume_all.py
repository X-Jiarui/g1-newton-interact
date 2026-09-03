#!/usr/bin/env python3
"""Restart every training process on this box from its own latest checkpoint, unchanged.

Used to pick up a code fix without discarding the weights. The runs are reconstructed from the
processes themselves -- argv, environment and working directory are read from /proc and replayed
verbatim -- so nothing depends on remembering how each one was launched, and a run launched with
options this script has never heard of comes back with them intact.

Two phases, because the overlay has to be installed while nothing is running:

    capture <spec.json>     record every live run, then kill it
    relaunch <spec.json>    start each one again with --resume <its latest checkpoint>

argv is replayed as a list, never through a shell: one of the arguments is a JSON object with
spaces in it and any quoting round-trip would corrupt it.

stdout is appended to the run's original log rather than truncating it, so the metric history
across the restart stays in one file and the reporters keep working.
"""

import json
import os
import pathlib
import signal
import subprocess
import sys
import time

PROC = pathlib.Path("/proc")
MATCH = "train_newton.py"
# The environment is replayed wholesale, minus the entries that describe the OLD process rather
# than the run: keeping these makes the child think it is its parent.
DROP_ENV = {"_", "SHLVL", "PWD", "OLDPWD"}


def live():
    out = []
    for d in PROC.iterdir():
        if not d.name.isdigit():
            continue
        try:
            raw = (d / "cmdline").read_bytes()
        except OSError:
            continue
        if MATCH.encode() not in raw:
            continue
        argv = [a for a in raw.decode("utf-8", "ignore").split("\0") if a]
        name = None
        for i, a in enumerate(argv):
            if a == "--run-name" and i + 1 < len(argv):
                name = argv[i + 1]
        try:
            env = {}
            for kv in (d / "environ").read_bytes().decode("utf-8", "ignore").split("\0"):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    if k not in DROP_ENV:
                        env[k] = v
            cwd = os.readlink(d / "cwd")
            log = os.readlink(d / "fd" / "1")
        except OSError:
            continue
        out.append({"pid": int(d.name), "name": name, "argv": argv,
                    "env": env, "cwd": cwd, "log": log})
    return sorted(out, key=lambda r: r["name"] or "")


def latest_ckpt(cwd, name):
    d = pathlib.Path(cwd) / "logs" / "rsl_rl" / "g1_residual_interact" / name
    best, best_it = None, -1
    for p in d.glob("model_*.pt"):
        try:
            it = int(p.stem.split("_")[-1])
        except ValueError:
            continue
        if it > best_it:
            best, best_it = p, it
    return (str(best), best_it) if best else (None, -1)


def capture(path):
    runs = live()
    if not runs:
        print("no live training processes; nothing to do")
        return 1
    for r in runs:
        ck, it = latest_ckpt(r["cwd"], r["name"] or "")
        r["resume"], r["resume_iter"] = ck, it
        print(f"  {r['name']:<28s} pid {r['pid']:<8d} resume from model_{it}.pt"
              if ck else f"  {r['name']:<28s} pid {r['pid']:<8d} NO CHECKPOINT -- will restart from scratch")
    pathlib.Path(path).write_text(json.dumps(runs, indent=1))
    for r in runs:
        try:
            os.kill(r["pid"], signal.SIGTERM)
        except OSError:
            pass
    for _ in range(30):
        time.sleep(1)
        if not live():
            break
    for r in live():
        try:
            os.kill(r["pid"], signal.SIGKILL)
        except OSError:
            pass
    time.sleep(2)
    print(f"captured {len(runs)} run(s) to {path}; {len(live())} still alive")
    return 0


def relaunch(path):
    runs = json.loads(pathlib.Path(path).read_text())
    for r in runs:
        argv = list(r["argv"])
        if "--resume" in argv:
            print(f"  {r['name']}: already has --resume; left as recorded")
        elif r.get("resume"):
            argv += ["--resume", r["resume"]]
        else:
            print(f"  {r['name']}: no checkpoint, restarting from scratch")
        log = open(r["log"], "ab")          # append: keep the pre-restart history in one file
        p = subprocess.Popen(argv, cwd=r["cwd"], env=r["env"], stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True)
        gpu = r["env"].get("CUDA_VISIBLE_DEVICES", "?")
        print(f"  GPU{gpu}  {r['name']:<28s} pid {p.pid}  resume={r.get('resume_iter', -1)}")
    print(f"relaunched {len(runs)} run(s)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("capture", "relaunch"):
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(capture(sys.argv[2]) if sys.argv[1] == "capture" else relaunch(sys.argv[2]))
