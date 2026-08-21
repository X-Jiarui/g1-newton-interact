"""Dump one clip's full structure. The previous probe truncated the key list to 8 and reported
'no object positions' for every file, which was my own error rather than a dataset problem."""
import os, pickle
import numpy as np

p = os.path.expanduser("~/jiarui/grab_g1_wuji_aligned/s1/stapler_lift.pkl")
d = pickle.load(open(p, "rb"))
print(f"{os.path.basename(p)}: {len(d)} keys\n")
for k, v in d.items():
  if isinstance(v, np.ndarray):
    print(f"  {k:22s} ndarray {str(v.shape):16s} {v.dtype}")
  elif isinstance(v, dict):
    inner = {kk: (vv.shape if isinstance(vv, np.ndarray) else type(vv).__name__)
             for kk, vv in list(v.items())[:8]}
    print(f"  {k:22s} dict     {inner}")
  elif isinstance(v, (list, tuple)):
    print(f"  {k:22s} {type(v).__name__} len={len(v)}")
  else:
    print(f"  {k:22s} {type(v).__name__} = {str(v)[:60]}")
