import os
import sys

import numpy as np
import pandas as pd


def process_edges(edges, pa, ch, ne):

  for a, e, b in edges:
    if a not in pa: pa[a] = []
    if a not in ch: ch[a] = []
    if a not in ne: ne[a] = []
    if b not in pa: pa[b] = []
    if b not in ch: ch[b] = []
    if b not in ne: ne[b] = []

    if e == "<--":
      pa[a].append(b)
      ch[b].append(a)
    elif e == "-->":
      pa[b].append(a)
      ch[a].append(b)
    elif e == "---":
      ne[a].append(b)
      ne[b].append(a)
    else: print(f"UKNOWN EDGE: {e}")


def invalid_sink(V, v, ch, ne):

  # check for children
  if [w for w in ch[v] if w in V]: return True
  
  # check if neighbors form a clique
  W = [w for w in ne[v] if w in V]
  while W:
    v = W.pop()
    if [w for w in W if w not in ne[v]]: return True

  return False


def get_order(ch, ne):

  V = [v for v in ch]
  for v in ne:
    if v not in V: V.append(v)

  order = []
  
  while V:
    i = 0
    while invalid_sink(V, V[i], ch, ne): i += 1
    v = V.pop(i)
    order.append(v)
  order.reverse()

  return order


def get_sets(pa, ch, ne, dyads, triads):

  V = get_order(ch, ne)
  while V:
    v = V.pop()
    W = [w for w in pa[v] + ne[v] if w in V]
    while W:
      a = W.pop()
      dyads.append(tuple(sorted([a, v])))
      for b in W:
        triads.append(tuple(sorted([a, b, v])))


path = sys.argv[1]  # path to graphs
fnames = os.listdir(path)

dyad_t = float(sys.argv[2]) * len(fnames)
triad_t = float(sys.argv[3]) * len(fnames)

dyad_freqs = {}
triad_freqs = {}

for fname in fnames:
  with open(path + fname) as f:
    edges = [edge.split()[1:] for edge in f.read().split("\n\n")[1].strip().split("\n")[1:]]

  pa = {}
  ch = {}
  ne = {}
  dyads = []
  triads = []

  process_edges(edges, pa, ch, ne)
  get_sets(pa, ch, ne, dyads, triads)

  for dyad in dyads:
    if dyad not in dyad_freqs: dyad_freqs[dyad] = 0
    dyad_freqs[dyad] += 1

  for triad in triads:
    if triad not in triad_freqs: triad_freqs[triad] = 0
    triad_freqs[triad] += 1


dyads = [dyad for dyad in dyad_freqs if dyad_freqs[dyad] >= dyad_t]
triads = [triad for triad in triad_freqs if triad_freqs[triad] >= triad_t]

graph = {dyad: ["-","-"] for dyad in dyads}

for a in dyads:
  for b in dyads:
    c = tuple(sorted(set(a) ^ set(b)))
    if len(c) != 2: continue
    if c in dyads: continue
    if tuple(sorted(set(a) | set(b))) not in triads: continue
    inter = tuple(set(a) & set(b))[0]
    if a.index(inter): graph[a][1] = ">"
    else: graph[a][0] = "<"
    if b.index(inter): graph[b][1] = ">"
    else: graph[b][0] = "<"


print("Graph Nodes:")
print(";".join([v for v in get_order(ch, ne)]))
print()
print("Graph Edges:")
for i, edge in enumerate(graph): print(f"{i}.", edge[0], "-".join(graph[edge]), edge[1])
