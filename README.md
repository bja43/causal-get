# Causal-GET
Causal Graph Estimation Toolbox

---

## How to build and install:

```
python -m build
pip install dist/*.whl
```

---

## Example usage:

```python
import numpy as np
import pandas as pd

import daosim as ds
import causalget as cg


## SIMULATION PARAMETERS ##

n = 1000  # NUMBER OF SAMPLES
p = 100   # NUMBER OF VARIABLES
ad = 10   # AVERAGE DEGREE

## SIMULATION DATA VIA DAO ##

g = ds.er_dag(p, ad=ad)
_, B, O = ds.corr(g)
X = ds.simulate(B, O, n)
df = pd.DataFrame(X)
R = df.corr().values

## TESTING FROM COV ##

print("from ndarray (corr):")
dag = cg.boss(R, n=n, discount=2.0, restarts=10)
print(SHD: np.sum(dag == g))
print()

## TESTING FROM DATA ##

print("from datafrom (data):")
dag = cg.boss(df, discount=2.0, restarts=1)
print(SHD: np.sum(dag == g))
print()
```
