import os
# IF USING PYTHON >3.11 REQUIRED?
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import struct

import numpy as np
import pandas as pd
import daosim as ds
import causalget as cg

from timeit import default_timer as timer

import jpype
import jpype.imports

# jpype.startJVM("-Xmx4g", jvmpath="/usr/lib/jvm/java-21-openjdk-amd64/lib/server/libjvm.so", classpath="tetrad-current.jar")
jpype.startJVM("-Xmx4g", jvmpath="/opt/homebrew/opt/openjdk@23/libexec/openjdk.jdk/Contents/MacOS/libjli.dylib", classpath="tetrad-current.jar")

import java.util as util
import edu.cmu.tetrad.data as td
import edu.cmu.tetrad.graph as tg
import edu.cmu.tetrad.search as ts
import edu.cmu.tetrad.algcomparison as ta


# check endianess of the current machine
byte_order = "<" if sys.byteorder == "little" else ">"
STRUCT_FMT = byte_order + "iii"
STRUCT_SIZE = struct.calcsize(STRUCT_FMT)


def df_to_data(df):
  cols = df.columns
  values = df.values
  n, p = df.shape

  variables = util.ArrayList()
  for col in cols:
    variables.add(td.ContinuousVariable(str(col)))

  databox = td.DoubleDataBox(n, p)
  for col, var in enumerate(values.T):
    for row, val in enumerate(var):
      databox.set(row, col, val)

  return td.BoxDataSet(databox, variables)


def construct_graph(g, nodes, cpdag=True):
  graph = tg.EdgeListGraph(nodes)

  for i, a in enumerate(nodes):
    for j, b in enumerate(nodes):
      if g[i, j]: graph.addDirectedEdge(b, a)

  if cpdag: graph = tg.GraphTransforms.cpdagForDag(graph)

  return graph


def run_sim(n, p, ad, sf):
  
  g = ds.er_dag(p, ad=ad)
  if sf[0]: g = ds.sf_out(g)
  if sf[1]: g = ds.sf_in(g)
  g = ds.randomize_graph(g)

  _, B, O = ds.corr(g)
  # _, B, O = ds.cov(g)
  X = ds.simulate(B, O, n)
  X = ds.standardize(X)

  R = np.corrcoef(X.T).astype(np.float32)

  df = pd.DataFrame(X, columns=[f'X{i}' for i in range(p)])
  data = df_to_data(df)

  cov_buf = struct.pack(byte_order + "II", n, p)
  cov_buf += R.astype(np.float32).tobytes() # float32
  data_buf = struct.pack(byte_order + "II", n, p)
  data_buf += X.astype(np.float32).T.tobytes() # float32 transposed
  knwl_buf = struct.pack(byte_order + "III", 0, 0, 0)

  nodes = data.getVariables()
  cpdag = construct_graph(g, nodes)

  algs = []
  graphs = []
  times = [timer()]


  if 0:
    algs.append("boss-tetrad")

    score = ts.score.SemBicScore(data, True)
    score.setPenaltyDiscount(2)
    score.setStructurePrior(0)

    search = ts.Boss(score)
    search.setUseBes(False)
    search.setNumStarts(1)
    search.setNumThreads(1)
    search.setUseDataOrder(False)
    search.setResetAfterBM(False)
    search.setResetAfterRS(False)
    search.setVerbose(False)
    search = ts.PermutationSearch(search)
    graphs.append(search.search())
    times.append(timer())


  if 1:
    algs.append("boss-from-cov")

    blob = cg.boss_from_cov(cov_buf, knwl_buf, 2.0, 10) 
    edges = [struct.unpack_from(STRUCT_FMT, blob, offset) for offset in range(0, len(blob), STRUCT_SIZE)]

    dag = np.zeros([p, p], dtype=np.uint8)
    for i, j, e in edges:
      if e == 2: dag[i, j] = 1
      if e == 1: dag[j, i] = 1

    graphs.append(construct_graph(dag, nodes))
    times.append(timer())


  if 0:
    algs.append("boss-from-data")

    blob = cg.boss_from_data(data_buf, knwl_buf, 2.0, 1)
    edges = [struct.unpack_from(STRUCT_FMT, blob, offset) for offset in range(0, len(blob), STRUCT_SIZE)]

    dag = np.zeros([p, p], dtype=np.uint8)
    for i, j, e in edges:
      if e == 2: dag[i, j] = 1
      if e == 1: dag[j, i] = 1

    graphs.append(construct_graph(dag, nodes))
    times.append(timer())

  return (tg.GraphUtils.replaceNodes(cpdag, nodes), data, [(alg, tg.GraphUtils.replaceNodes(graphs[i], nodes), times[i + 1] - times[i]) for i, alg in enumerate(algs)])


reps = 10

unique_sims = [(n, p, ad, sf)
               for n in [1000]
               for p in [100]
               for ad in [10]
               for sf in [(1, 0)]]



stats = [ta.statistic.AdjacencyPrecision(),
         ta.statistic.AdjacencyRecall(),
         ta.statistic.OrientationPrecision(),
         ta.statistic.OrientationRecall()]

results = []

for n, p, ad, sf, rep in [(*sim, rep) for sim in unique_sims for rep in range(reps)]:
  print(f"samples: {n} | variables: {p} | avg_deg: {ad} | scale-free: {sf} | rep: {rep} {10 * ' '}")
  true_cpdag, data, algs = run_sim(n, p, ad, sf)
  for alg, est_cpdag, seconds in algs:
    results.append([n, p, ad, str(sf), rep, alg,
                    stats[0].getValue(true_cpdag, est_cpdag, data),
                    stats[1].getValue(true_cpdag, est_cpdag, data),
                    stats[2].getValue(true_cpdag, est_cpdag, data),
                    stats[3].getValue(true_cpdag, est_cpdag, data),
                    seconds])

stat_cols = ["adj_pre", "adj_rec", "ori_pre", "ori_rec", "seconds"]
param_cols = ["samples", "variables", "avg_deg", "scale-free", "run", "algorithm"]
df = pd.DataFrame(np.array(results), columns=param_cols+stat_cols)
for col in stat_cols: df[col] = df[col].astype(float)

param_cols.remove("run")
print(f"\n\nreps: {reps}\n")
print(df.groupby(param_cols)[stat_cols].agg("mean").round(2).to_string())
