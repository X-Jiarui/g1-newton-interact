#!/usr/bin/env python3
"""Catch undefined names in the overlay before a box does.

`ast.parse` accepts `out_q.append(...)` with no `out_q = []` anywhere -- it is a NameError, not a
SyntaxError, and it only surfaces when the box builds the environment. That cost two dead launches
once, from a scripted string replacement that silently matched the wrong function.

This walks each function and reports a name that is read before any binding in the same scope and
is not a module-level, builtin, argument or comprehension name. It is deliberately conservative:
it reports only the shape that bug had, so a clean run means something.

    python tools/setup/check_overlay_names.py            # every overlay file
    python tools/setup/check_overlay_names.py <file>...
"""
from __future__ import annotations

import ast
import builtins
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
OVERLAY = ROOT / "tools" / "setup" / "mjlab_overlay"


# module dunders exist at runtime but are neither builtins nor assigned anywhere
_MODULE_DUNDERS = {"__file__", "__name__", "__doc__", "__package__", "__spec__", "__loader__"}


def module_names(tree: ast.Module) -> set[str]:
    out: set[str] = set(_MODULE_DUNDERS)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            out.add(node.id)
        elif isinstance(node, ast.alias):
            out.add((node.asname or node.name).split(".")[0])
        elif isinstance(node, ast.arg):
            out.add(node.arg)
        elif isinstance(node, ast.Global):
            out.update(node.names)
    return out


def check_function(fn: ast.AST, known: set[str], path: pathlib.Path) -> list[str]:
    bound: set[str] = set(known)
    problems: list[str] = []
    for node in ast.walk(fn):
        if isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                bound.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in bound or hasattr(builtins, node.id):
                continue
            problems.append(f"{path.name}:{node.lineno}: undefined name {node.id!r} "
                            f"in {getattr(fn, 'name', '<lambda>')}")
    return problems


def main() -> int:
    files = [pathlib.Path(a) for a in sys.argv[1:]] or sorted(OVERLAY.rglob("*.py"))
    bad: list[str] = []
    for path in files:
        tree = ast.parse(path.read_text())
        known = module_names(tree)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                bad += check_function(node, known, path)
    for line in sorted(set(bad)):
        print(line)
    print(f"[check] {len(files)} file(s), {len(set(bad))} undefined name(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
