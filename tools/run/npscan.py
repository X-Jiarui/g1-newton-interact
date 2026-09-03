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

The dead-log window is deliberately short. A long one reports every run that was stopped on
purpose today, which buries the live rows in exactly the report meant to show them; 30 minutes
answers "did something fall over just now" and nothing else.

Usage:
    python npscan.py                          # live runs only
    python npscan.py /workspace/logs_*        # also flag logs orphaned in the last 30 min
    DEAD_WINDOW_MIN=180 python npscan.py ...  # widen that window
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
    window = float(os.environ.get("DEAD_WINDOW_MIN", "30")) * 60.0
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
                if age < window:
                    dead.append((age, p))
        if dead:
            print(f"\nDIED? logs written in the last {window/60:.0f} min with no live process "
                  f"({len(dead)}):")
            for age, p in sorted(dead):
                got = nprow.parse(str(p))
                it = f"it {got[1]}/{got[2]}" if got else "unreadable"
                print(f"  {p.name:<32s} {it:<16s} last write {age/60:.0f} min ago")


if __name__ == "__main__":
    main()
