import os
# IF USING PYTHON >3.11 REQUIRED?
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import daosim as ds

from timeit import default_timer as timer

from boss import boss

import jpype
import jpype.imports

jpype.startJVM("-Xmx4g", jvmpath="/usr/lib/jvm/java-21-openjdk-amd64/lib/server/libjvm.so", classpath="tetrad-current.jar")

import java.util as util
import edu.cmu.tetrad.data as td
import edu.cmu.tetrad.graph as tg
import edu.cmu.tetrad.search as ts
import edu.cmu.tetrad.algcomparison as ta


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

  nodes = data.getVariables()
  cpdag = construct_graph(g, nodes)

  algs = []
  graphs = []
  times = [timer()]

  # SET IF TETRAD ALG WILL RUN AS COMPARISON
  if 1:

    algs.append("tetrad")

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

  algs.append("c")

  dag = boss(R, n, 2.0, 10)
  # dag = boss(X.astype(np.float32).T, 2.0, 1)
  graphs.append(construct_graph(dag, nodes))
  times.append(timer())

  return (tg.GraphUtils.replaceNodes(cpdag, nodes), data, [(alg, tg.GraphUtils.replaceNodes(graphs[i], nodes), times[i + 1] - times[i]) for i, alg in enumerate(algs)])


reps = 10

unique_sims = [(n, p, ad, sf)
               for n in [1000]
               for p in [60]
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
