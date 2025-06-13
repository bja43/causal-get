#include <stdio.h>
#include <stdint.h>
#include <time.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define BOSS_IMPLEMENTATION

#ifndef BOSS_H_
#include "boss.h"
#endif // BOSS_H_


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
  Py_buffer cov_view;
  Py_buffer knwl_view;

  float discount = 1.0;
  uint32_t restarts = 1;
  uint32_t seed = 0;

  static char *kwlist[] = {"cov", "knowledge", "discount", "restarts", "seed", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kw, "y*y*|fII", kwlist, &cov_view, &knwl_view, &discount, &restarts, &seed)) {
    return NULL;
  }

  printf("discount: %f, restarts: %u, seed: %u\n", discount, restarts, seed);

  // PASS SEED HERE!
  srand(time(NULL));

  void *itr;
  
  itr = cov_view.buf;
  uint32_t n = *((uint32_t *)itr);
  itr += sizeof(uint32_t);
  uint32_t p = *((uint32_t *)itr);
  itr += sizeof(uint32_t);
  float *cov = (float *)itr;

  for (size_t i = 0; i < p; i++) {
    for (size_t j = 0; j < p; j++) {
      printf(" %6.3f", cov[i * p + j]);
    }
    printf("\n");
  }
  printf("%u %u\n", n, p);

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


  // TEMPORARY SOLUTION!
  uint8_t *tmp = malloc(sizeof(uint8_t) * p * p);

  // ADD KNOWLEDGE TO THIS CALL!
  boss_search(cov, n, p, discount, restarts, tmp);

  EdgeList graph = {0};
  graph.edges = malloc(sizeof(Edge) * p * p); // overkill for now

  for (size_t i = 0; i < p; i++) {
    for (size_t j = 0; j < p; j++) {
      printf(" %hhu", tmp[i * p + j]);
    }
    printf("\n");
  }

  for (uint32_t i = 0; i < p; i++) {
    for (uint32_t j = 0; j < p; j++) {
      if (tmp[i * p + j] == 1) {
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

  printf("THIS IS NOT HOOKED UP TO BOSS YET\n");
  printf("THE CURRENT OUTPUT IS FOR TESTING PURPOSES ONLY\n");

  Py_buffer data_view;
  Py_buffer knwl_view;

  static char *kwlist[] = {"data", "knwl", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kw, "y*y*", kwlist, &data_view, &knwl_view)) {
    return NULL;
  }

  void *itr;
  
  itr = data_view.buf;
  uint32_t n = *((uint32_t *)itr);
  itr += sizeof(uint32_t);
  uint32_t p = *((uint32_t *)itr);
  itr += sizeof(uint32_t);
  float *data = (float *)itr;

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < p; j++) {
      printf(" %6.3f", data[i * p + j]);
    }
    printf("\n");
  }
  printf("%u %u\n", n, p);

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

  // print the knwl groups
  size_t offset = 0;
  for (size_t i = 0; i < knwl.num_groups; i++) {
    printf("Group %zu:", i);
    for (size_t j = 0; j < knwl.group_sizes[i]; j++)
      printf(" %u", knwl.group_members[offset + j]);
    offset += knwl.group_sizes[i];
    printf("\n");
  }

  // print forbidden knwl knwl_graph (on groups)
  for (size_t i = 0; i < knwl_graph.num_edges; i++) {
    if (knwl_graph.edges[i].edge == 1) {
      printf("%zu. %u <-- %u\n", i, knwl_graph.edges[i].i, knwl_graph.edges[i].j);
    } else if (knwl_graph.edges[i].edge == 2) {
      printf("%zu. %u --> %u\n", i, knwl_graph.edges[i].i, knwl_graph.edges[i].j);
    }
  }

  PyObject *edges = PyBytes_FromStringAndSize((const char *)knwl_graph.edges, knwl_graph.num_edges * sizeof(Edge));

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
