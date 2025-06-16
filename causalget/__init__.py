import sys
import struct

import numpy as np
import pandas as pd

from .c_backend import (
  boss_from_cov,
  boss_from_data,
)


# ignoring knowledge and seed
def boss(data, n=None, discount=1.0, restarts=1, seed=None):
  '''
  runs the boss algorithm...
  '''

  byte_order = "<" if sys.byteorder == "little" else ">"

  knwl_buf = struct.pack(byte_order + "III", 0, 0, 0)

  if isinstance(n, int) and isinstance(data, np.ndarray):
    print("boss from cov")
    _, p = data.shape
    R = data.astype(np.float32) # float32
    cov_buf = struct.pack(byte_order + "II", n, p)
    cov_buf += R.tobytes()
    blob = boss_from_cov(cov_buf, knwl_buf, float(discount), int(restarts)) 

  elif isinstance(data, np.ndarray):
    print("boss from data")
    n, p = data.shape
    X = data.astype(np.float32).T # float32 transposed 
    data_buf = struct.pack(byte_order + "II", n, p)
    data_buf += X.tobytes()
    blob = boss_from_data(data_buf, knwl_buf, float(discount), int(restarts))

  elif isinstance(data, pd.DataFrame):
    print("boss from data")
    n, p = data.shape
    X = data.values.astype(np.float32).T # float32 transposed
    data_buf = struct.pack(byte_order + "II", n, p)
    data_buf += X.tobytes()
    blob = boss_from_data(data_buf, knwl_buf, float(discount), int(restarts))

  else:
    print("ERROR: invalid input")
    quit()

  STRUCT_FMT = byte_order + "iii"
  STRUCT_SIZE = struct.calcsize(STRUCT_FMT)
  edges = [struct.unpack_from(STRUCT_FMT, blob, offset) for offset in range(0, len(blob), STRUCT_SIZE)]

  dag = np.zeros([p, p], dtype=np.uint8)
  for i, j, e in edges:
    if e == 2: dag[i, j] = 1
    if e == 1: dag[j, i] = 1

  return dag
