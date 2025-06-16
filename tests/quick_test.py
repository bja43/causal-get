import numpy as np
import pandas as pd

import daosim as ds
import causalget as cg


n = 1000
p = 400
ad = 20

g = ds.er_dag(p, ad=ad)
_, B, O = ds.corr(g)
X = ds.simulate(B, O, n)
df = pd.DataFrame(X)
R = df.corr().values

print("from ndarray (corr):")
dag = cg.boss(R, n=n, discount=2)
print(dag)
print()

# print("from ndarray (data):")
# dag = cg.boss(X)
# print(dag)
# print()
# 
# print("from dataframe:")
# dag = cg.boss(df)
# print(dag)
# print()
