"""Let mjlab's 3.8-era code run on mujoco_warp 3.11.

mjlab sets `wp_model.opt.ls_parallel`, which mujoco_warp 3.9.1 removed and which now raises
AttributeError from a property that exists only to produce that error. Newton 1.5 pins mujoco_warp
3.11, so any process that runs both hits this the moment an mjlab Simulation is built.

The shim patches the NEW library to tolerate the OLD attribute rather than editing mjlab. mjlab is the
baseline this whole migration is measured against; a source change there -- however harmless-looking --
would mean the reference and the port no longer run identical code, and that is precisely the kind of
quiet divergence this project keeps getting bitten by.

What it does NOT do is pretend the option still works. `ls_parallel` selected a parallel line search;
3.9.1 removed the choice. Assignments are swallowed and *recorded*, so `removed_option_writes()` can
report what mjlab asked for and what the version bump silently ignored. That difference belongs in the
version noise floor, not hidden inside a compatibility layer.

Import this before constructing any mjlab Simulation.
"""

from __future__ import annotations

_RECORDED: list[tuple[str, object]] = []
_PATCHED: list[str] = []


def _is_removal_property(obj) -> bool:
  """A property whose only job is to raise 'was removed in MuJoCo Warp'."""
  if not isinstance(obj, property) or obj.fget is None:
    return False
  doc = (obj.fget.__doc__ or "")
  code = getattr(obj.fget, "__code__", None)
  consts = code.co_consts if code is not None else ()
  return any(isinstance(c, str) and "was removed in MuJoCo Warp" in c for c in consts) or \
         "was removed in MuJoCo Warp" in doc


def apply() -> list[str]:
  """Replace removal-stub properties with recording no-ops. Returns the names patched."""
  if _PATCHED:
    return list(_PATCHED)
  import mujoco_warp
  from mujoco_warp._src import types as _types

  for cls_name in dir(_types):
    cls = getattr(_types, cls_name)
    if not isinstance(cls, type):
      continue
    for attr in list(vars(cls)):
      obj = vars(cls).get(attr)
      if not _is_removal_property(obj):
        continue

      def _make(name):
        def _get(self):
          return None

        def _set(self, value, _n=name):
          _RECORDED.append((_n, value))

        return property(_get, _set)

      setattr(cls, attr, _make(f"{cls_name}.{attr}"))
      _PATCHED.append(f"{cls_name}.{attr}")
  return list(_PATCHED)


def removed_option_writes() -> list[tuple[str, object]]:
  """Every write mjlab made to an option this mujoco_warp no longer has.

  Non-empty means the two versions were asked to solve with different settings, and the difference
  is real even though nothing raised.
  """
  return list(_RECORDED)
