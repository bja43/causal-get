#include <stdio.h>
#include <stdint.h>
#include <time.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define BOSS_IMPLEMENTATION

#ifndef BOSS_H_
#include "boss.h"
#endif // BOSS_H_

#ifndef BIC_H_
#include "bic.h"
#endif // BIC_H_

// MOVE THIS SOMEWHERE ELSE
typedef struct {
  uint32_t i;
  uint32_t j;
  uint32_t edge;
} Edge;

// MOVE THIS SOMEWHERE ELSE
typedef struct {
  uint32_t num_edges;
  Edge *edges;
} EdgeList;

// MOVE THIS SOMEWHERE ELSE
typedef struct {
  uint32_t num_groups;
  uint32_t *group_sizes;
  uint32_t *group_members;
  EdgeList forbidden;
} Knowledge;


static PyObject *boss_from_cov(PyObject *self, PyObject *args, PyObject *kw)
{
  (void)self;   // mark 'self' as unused to suppress warning

  Py_buffer cov_view;
  Py_buffer knwl_view;

  float discount = 1.0;
  uint32_t restarts = 1;
  uint32_t seed = 0;

  static char *kwlist[] = {"cov", "knowledge", "discount", "restarts", "seed", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kw, "y*y*|fII", kwlist, &cov_view, &knwl_view, &discount, &restarts, &seed)) {
    return NULL;
  }

  // printf("discount: %4.2f, restarts: %u, seed: %u\n", discount, restarts, seed);

  if (seed) srand(seed);
  else srand(time(NULL));

  void *itr;
  
  // HOW DOES THIS PTR ARITHMETIC KNOW HOW FAR TO ADVANCE???
  itr = cov_view.buf;
  uint32_t n = *((uint32_t *)itr);
  itr += sizeof(uint32_t);
  uint32_t p = *((uint32_t *)itr);
  itr += sizeof(uint32_t);
  float *cov = (float *)itr;

  // printf("%u %u\n", n, p);

  itr = knwl_view.buf;
  Knowledge knwl = {0};

  knwl.num_groups = *((uint32_t *)itr);
  itr += sizeof(uint32_t);
  knwl.group_sizes = (uint32_t *)itr;
  itr += sizeof(uint32_t) * knwl.num_groups;
  knwl.group_members = (uint32_t *)itr;
  for (size_t i = 0; i < knwl.num_groups; i++)
    itr += sizeof(uint32_t) * knwl.group_sizes[i];

  EdgeList knwl_graph = knwl.forbidden;
  knwl_graph.num_edges = *((uint32_t *)itr);
  itr += sizeof(uint32_t);
  knwl_graph.edges = (Edge *)itr;


  // ADD KNOWLEDGE TO THIS CALL!
  double *L = malloc(sizeof(double) * TNU(p));
  double *D = malloc(sizeof(double) * p);
  uint32_t *z = malloc(sizeof(uint32_t) * p);

  // TEMPORARY SOLUTION!
  uint8_t *tmp = malloc(sizeof(uint8_t) * p * p);

  BIC bic = { discount, cov, n, p, get_cov_precomp, L, D, 0, 0, z };

  boss_search(&bic, restarts, tmp);

  free(L);
  free(D);
  free(z);

  EdgeList graph = {0};
  graph.edges = malloc(sizeof(Edge) * p * p); // overkill for now

  // for (size_t i = 0; i < p; i++) {
  //   for (size_t j = 0; j < p; j++) {
  //     printf(" %hhu", tmp[i * p + j]);
  //   }
  //   printf("\n");
  // }

  for (uint32_t i = 0; i < p; i++) {
    for (uint32_t j = 0; j < p; j++) {
      if (tmp[i * p + j]) {
        Edge edge = {j, i, 1};
        graph.edges[graph.num_edges++] = edge;
      }
    }
  }

  PyObject *edges = PyBytes_FromStringAndSize((const char *)graph.edges, graph.num_edges * sizeof(Edge));

  free(tmp);
  free(graph.edges);

  PyBuffer_Release(&cov_view);
  PyBuffer_Release(&knwl_view);

  return edges;
}


static PyObject *boss_from_data(PyObject *self, PyObject *args, PyObject *kw)
{
  (void)self;   // mark 'self' as unused to suppress warning

  Py_buffer data_view;
  Py_buffer knwl_view;

  float discount = 1.0;
  uint32_t restarts = 1;
  uint32_t seed = 0;

  static char *kwlist[] = {"data", "knowledge", "discount", "restarts", "seed", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kw, "y*y*|fII", kwlist, &data_view, &knwl_view, &discount, &restarts, &seed)) {
    return NULL;
  }

  // printf("discount: %4.2f, restarts: %u, seed: %u\n", discount, restarts, seed);

  if (seed) srand(seed);
  else srand(time(NULL));

  void *itr;
  
  itr = data_view.buf;
  uint32_t n = *((uint32_t *)itr);
  itr += sizeof(uint32_t);
  uint32_t p = *((uint32_t *)itr);
  itr += sizeof(uint32_t);
  float *data = (float *)itr;

  // printf("%u %u\n", n, p);

  itr = knwl_view.buf;
  Knowledge knwl = {0};

  knwl.num_groups = *((uint32_t *)itr);
  itr += sizeof(uint32_t);
  knwl.group_sizes = (uint32_t *)itr;
  itr += sizeof(uint32_t) * knwl.num_groups;
  knwl.group_members = (uint32_t *)itr;
  for (size_t i = 0; i < knwl.num_groups; i++)
    itr += sizeof(uint32_t) * knwl.group_sizes[i];

  EdgeList knwl_graph = knwl.forbidden;
  knwl_graph.num_edges = *((uint32_t *)itr);
  itr += sizeof(uint32_t);
  knwl_graph.edges = (Edge *)itr;


  // ADD KNOWLEDGE TO THIS CALL!
  double *L = malloc(sizeof(double) * TNU(p));
  double *D = malloc(sizeof(double) * p);
  uint32_t *z = malloc(sizeof(uint32_t) * p);

  // TEMPORARY SOLUTION!
  uint8_t *tmp = malloc(sizeof(uint8_t) * p * p);

  BIC bic = { discount, data, n, p, get_cov_onfly, L, D, 0, 0, z };

  boss_search(&bic, restarts, tmp);

  free(L);
  free(D);
  free(z);

  EdgeList graph = {0};
  graph.edges = malloc(sizeof(Edge) * p * p); // overkill for now

  // for (size_t i = 0; i < p; i++) {
  //   for (size_t j = 0; j < p; j++) {
  //     printf(" %hhu", tmp[i * p + j]);
  //   }
  //   printf("\n");
  // }

  for (uint32_t i = 0; i < p; i++) {
    for (uint32_t j = 0; j < p; j++) {
      if (tmp[i * p + j]) {
        Edge edge = {j, i, 1};
        graph.edges[graph.num_edges++] = edge;
      }
    }
  }

  PyObject *edges = PyBytes_FromStringAndSize((const char *)graph.edges, graph.num_edges * sizeof(Edge));

  free(tmp);
  free(graph.edges);

  PyBuffer_Release(&data_view);
  PyBuffer_Release(&knwl_view);

  return edges;
}


static PyMethodDef methods[] = {
  { "boss_from_cov", (PyCFunction)boss_from_cov, METH_VARARGS | METH_KEYWORDS, "runs boss from cov..." },
  { "boss_from_data", (PyCFunction)boss_from_data, METH_VARARGS | METH_KEYWORDS, "runs boss from data..." },
  { NULL, NULL, 0, NULL }
};


static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "causalget",
  "Causal Graph Estimation Toolbox",
  -1,
  methods,
  NULL,
  NULL,
  NULL,
  NULL
};


PyMODINIT_FUNC PyInit_causalget(void)
{
  return PyModule_Create(&moduledef);
}




//  // print the knwl groups
//  size_t offset = 0;
//  for (size_t i = 0; i < knwl.num_groups; i++) {
//    printf("Group %zu:", i);
//    for (size_t j = 0; j < knwl.group_sizes[i]; j++)
//      printf(" %u", knwl.group_members[offset + j]);
//    offset += knwl.group_sizes[i];
//    printf("\n");
//  }
//
//  // print forbidden knwl knwl_graph (on groups)
//  for (size_t i = 0; i < knwl_graph.num_edges; i++) {
//    if (knwl_graph.edges[i].edge) {
//      printf("%zu. %u <-- %u\n", i, knwl_graph.edges[i].i, knwl_graph.edges[i].j);
//    } else if (knwl_graph.edges[i].edge == 2) {
//      printf("%zu. %u --> %u\n", i, knwl_graph.edges[i].i, knwl_graph.edges[i].j);
//    }
//  }
