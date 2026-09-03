#!/usr/bin/env python3
"""Copy the canonical mjlab task files over whatever a box has installed.

Why a whole-file overlay rather than more entries in `patch_mjlab.py`.

`patch_mjlab.py` rewrites a PRISTINE mjlab by matching exact source strings. That works exactly
once. Four training boxes later, the three task files had drifted into three different copies --
the same fixes written twice in different places (`import os` vs `import os as _os`), and two
boxes each carrying allowlist entries the others lacked, so the SAME run produced different
tensorboard panels depending on where it ran. A string patch has no defined behaviour against
three bases; it either fails to match or matches the wrong thing.

These five files are ours end to end -- every line that differs from upstream mjlab was written
for this task -- so shipping them whole is both simpler and the only thing that guarantees the
property we actually want: byte-identical task code on every box.

`patch_mjlab.py` is still required. It covers files this overlay does not carry, and on a fresh
install it is harmless to run both: the overlay files already contain its edits, so it reports
"already applied" for the overlapping ones. Run patch first, overlay second.

Usage:  python tools/setup/apply_mjlab_overlay.py /path/to/mjlab-run
"""

import filecmp
import shutil
import sys
from pathlib import Path

FILES = [
    ("residual_interact", "mdp.py"),
    ("residual_interact", "staged_mdp.py"),
    ("residual_interact", "env_cfgs.py"),
    ("residual_interact", "rl.py"),
    ("apple_eat", "mdp.py"),
    ("apple_eat", "object_pool.py"),
]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    root = Path(sys.argv[1]).resolve()
    src_root = Path(__file__).resolve().parent / "mjlab_overlay"
    tasks = root / "src" / "mjlab" / "tasks"
    if not tasks.is_dir():
        print(f"not an mjlab checkout: {tasks} does not exist")
        return 1

    changed = 0
    for pkg, name in FILES:
        src = src_root / pkg / name
        dst = tasks / pkg / name
        if not src.is_file():
            print(f"MISSING in overlay: {src}")
            return 1
        if not dst.is_file():
            print(f"MISSING on box:     {dst}")
            return 1
        if filecmp.cmp(src, dst, shallow=False):
            print(f"same      {pkg}/{name}")
            continue
        # Keep exactly one backup: the state before the first overlay. Later runs must not
        # overwrite it with already-overlaid content, or the original is lost.
        bak = dst.with_suffix(dst.suffix + ".pre_overlay")
        if not bak.exists():
            shutil.copy2(dst, bak)
        shutil.copy2(src, dst)
        print(f"OVERLAID  {pkg}/{name}")
        changed += 1

    print(f"{changed} file(s) changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
