import numpy as np
import pandas as pd

import daosim as ds
import causalget as cg


if __name__ == "__main__":
  n = 1000
  p = 100
  ad = 6

  g = ds.er_dag(p, ad=ad)
  # _, B, O = ds.corr(g)
  _, B, O = ds.cov(g)
  X = ds.simulate(B, O, n)
  df = pd.DataFrame(X)
  R = df.corr().values

  ## TESTING FROM COV $$

  print("from ndarray (corr):")
  dag = cg.boss(R, n=n, discount=2)
  print("SHD:", np.sum(dag != g))
  print()

  ## TESTING FROM DATA ##

  # print("from ndarray (data):")
  # dag = cg.boss(X)
  # print("SHD:", np.sum(dag != g))
  # print()

  # print("from dataframe:")
  # dag = cg.boss(df)
  # print("SHD:", np.sum(dag != g))
  # print()

  ## TESTING SEEDS ##

  # dag1 = cg.boss(R, n=n, discount=2, seed=32)
  # dag2 = cg.boss(R, n=n, discount=2, seed=32)
  # dag3 = cg.boss(R, n=n, discount=2, seed=23)
 
  # print(f"seed test: {int(np.sum(dag1 == dag2))} / {p * p}")
  # print(f"seed test: {int(np.sum(dag2 == dag3))} / {p * p}")
