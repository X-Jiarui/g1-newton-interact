#!/usr/bin/env python3
"""Report every training run this box is ACTUALLY running, discovered from the processes.

Why not a glob over a log directory. Every previous monitor named its log directories by hand,
which means it reports whatever was running when the monitor was written. Runs launched into a
new directory are then invisible, and the report still looks complete -- it has rows, they are
just the wrong set. This walks /proc instead: a run exists because a process exists.

The log is taken from the process's own stdout (`/proc/<pid>/fd/1`), which is where the launch
scripts redirect it, so the row and the process cannot disagree about which file is whose.

Also reports the opposite mismatch, which is the one that matters operationally: a log that was
being written recently but has no live process behind it any more. That is a run that DIED, and
a monitor keyed on directories reports it as a healthy row until the numbers go stale enough to
notice.

Usage:
    python npscan.py                      # live runs, plus recently-dead ones it can find
    python npscan.py /workspace/logs_*    # also sweep these dirs for orphaned logs
"""

import os
import pathlib
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import nprow  # noqa: E402  -- same directory; provides KEYS and row()

PROC = pathlib.Path("/proc")
MATCH = "train_newton.py"


def live_runs():
    """(run_name, log_path, pid) for every training process, newest pid last."""
    out = []
    for d in PROC.iterdir():
        if not d.name.isdigit():
            continue
        try:
            cmd = (d / "cmdline").read_bytes().decode("utf-8", "ignore")
        except OSError:
            continue
        if MATCH not in cmd:
            continue
        argv = [a for a in cmd.split("\0") if a]
        name = None
        for i, a in enumerate(argv):
            if a == "--run-name" and i + 1 < len(argv):
                name = argv[i + 1]
            elif a.startswith("--run-name="):
                name = a.split("=", 1)[1]
        try:
            log = os.readlink(d / "fd" / "1")
        except OSError:
            log = ""
        out.append((name or f"pid{d.name}", log, int(d.name)))
    return sorted(out)


def main():
    runs = live_runs()
    seen = set()
    print(f"live training processes: {len(runs)}")
    for name, log, pid in runs:
        seen.add(os.path.realpath(log) if log else "")
        if log and os.path.exists(log):
            # alive is known from the process, so do not fall back to the mtime guess
            print(nprow.row(log, name=name, alive=""))
        else:
            print(f"{name:<22s}  (running as pid {pid}, but its stdout is not a readable file: {log!r})")

    # Anything written recently with nothing behind it is a run that stopped.
    dirs = sys.argv[1:]
    if dirs:
        dead = []
        for d in dirs:
            for p in sorted(pathlib.Path(d).glob("*.log")) if pathlib.Path(d).is_dir() else []:
                rp = os.path.realpath(p)
                if rp in seen:
                    continue
                try:
                    age = time.time() - p.stat().st_mtime
                except OSError:
                    continue
                if age < 6 * 3600:          # touched in the last six hours, so recently relevant
                    dead.append((age, p))
        if dead:
            print(f"\nlogs written in the last 6 h with NO live process ({len(dead)}):")
            for age, p in sorted(dead):
                print(f"  {p}   last write {age/60:.0f} min ago")
                print("  " + nprow.row(str(p), alive="  DEAD"))


if __name__ == "__main__":
    main()
