import sys
import struct

import numpy as np
import pandas as pd
import daosim as ds
import causalget as cg


# check endianess of the current machine
byte_order = "<" if sys.byteorder == "little" else ">"


p = 6
ad = 2
n = 100


g = ds.er_dag(p, ad=ad)
print(g)
_, B, O = ds.corr(g)
X = ds.simulate(B, O, n).astype(np.float32)

cols = [i for i in range(p)]
df = pd.DataFrame(X, columns=cols)
R = df.corr().values.astype(np.float32)

# cov buffer
cov_buf = struct.pack(byte_order + "II", n, p)
cov_buf += R.astype(np.float32).tobytes() # float32

# data buffer
data_buf = struct.pack(byte_order + "II", n, p)
data_buf += X.astype(np.float32).tobytes() # float32


# uint32
num_groups = 3

# uint32
group_members = [x for x in cols]

# uint32 (in this example all groups have size 10)
# group index i with have size equal to the ith member of this list
group_sizes = [2 for i in range(num_groups)]

# (uint32, uint32, uint32) -> (group a, group b, forbidden edge type)
forbidden = []
for i in range(num_groups):
  for j in range(i):
    forbidden += [i, j, 2]

# the components of knowledge
# print(num_groups)
# print(group_sizes)
# print(group_members)
# print(forbidden)

knwl_buf = struct.pack(byte_order + "I", num_groups)
knwl_buf += struct.pack(byte_order + f"{num_groups}I", *group_sizes)
knwl_buf += struct.pack(byte_order + f"{sum([i for i in group_sizes])}I", *group_members)
knwl_buf += struct.pack(byte_order + "I", len(forbidden) // 3)
knwl_buf += struct.pack(byte_order + f"{len(forbidden)}I", *forbidden)


if 1:
  blob = cg.boss_from_cov(cov_buf, knwl_buf)

  STRUCT_FMT = byte_order + "iii"
  STRUCT_SIZE = struct.calcsize(STRUCT_FMT)
  edges = [struct.unpack_from(STRUCT_FMT, blob, offset) for offset in range(0, len(blob), STRUCT_SIZE)]

  print(edges)

  for i, j, e in edges:
    if e == 1: print(i, "<--", j)
    if e == 2: print(i, "-->", j)


if 1:
  blob = cg.boss_from_data(data_buf, knwl_buf)

  STRUCT_FMT = byte_order + "iii"
  STRUCT_SIZE = struct.calcsize(STRUCT_FMT)
  edges = [struct.unpack_from(STRUCT_FMT, blob, offset) for offset in range(0, len(blob), STRUCT_SIZE)]

  print(edges)

  for i, j, e in edges:
    if e == 1: print(i, "<--", j)
    if e == 2: print(i, "-->", j)
