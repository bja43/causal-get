#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define BTA_IMPLEMENTATION
#define PQ_IMPLEMENTATION
#define BIC_IMPLEMENTATION
#define GST_IMPLEMENTATION
#define BOSS_IMPLEMENTATION
#define SP_IMPLEMENTATION

#ifndef BTA_H_
#include "bta.h"
#endif // BTA_H_

#ifndef PQ_H_
#include "pq.h"
#endif // PQ_H_

#ifndef BIC_H_
#include "bic.h"
#endif // BIC_H_

#ifndef GST_H_
#include "gst.h"
#endif // GST_H_

#ifndef BOSS_H_
#include "boss.h"
#endif // BOSS_H_

#ifndef SP_H_
#include "sp.h"
#endif // SP_H_


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




// MOVE THIS SOMEWHERE ELSE
void parse_knowledge(Knowledge *knwl, u_int32_t *itr)
{
  knwl->num_groups = *itr++;
  knwl->group_sizes = itr;
  itr += knwl->num_groups;
  knwl->group_members = itr;
  for (size_t i = 0; i < knwl->num_groups; i++)
    itr += knwl->group_sizes[i];
  knwl->forbidden.num_edges = *itr++;
  knwl->forbidden.edges = (Edge *)itr;
}




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

  uint32_t *itr;
  
  itr = cov_view.buf;
  uint32_t n = *itr++;
  uint32_t p = *itr++;
  float *cov = (float *)itr;

  // printf("%u %u\n", n, p);

  itr = knwl_view.buf;
  Knowledge knwl = {0};
  parse_knowledge(&knwl, itr);

  // print the knwl groups
  // EdgeList knwl_graph = knwl.forbidden;
  // size_t offset = 0;
  // for (size_t i = 0; i < knwl.num_groups; i++) {
  //   printf("Group %zu:", i);
  //   for (size_t j = 0; j < knwl.group_sizes[i]; j++)
  //     printf(" %u", knwl.group_members[offset + j]);
  //   offset += knwl.group_sizes[i];
  //   printf("\n");
  // }

  // print forbidden knwl knwl_graph (on groups)
  // for (size_t i = 0; i < knwl_graph.num_edges; i++) {
  //   if (knwl_graph.edges[i].edge) {
  //     printf("%zu. %u <-- %u\n", i, knwl_graph.edges[i].i, knwl_graph.edges[i].j);
  //   } else if (knwl_graph.edges[i].edge == 2) {
  //     printf("%zu. %u --> %u\n", i, knwl_graph.edges[i].i, knwl_graph.edges[i].j);
  //   }
  // }


  // MAKE BIC INIT / ALLOC AND FREE FUNCTIONS?
  double *L = malloc(sizeof(double) * TNU(p));
  double *D = malloc(sizeof(double) * p);
  uint32_t *z = malloc(sizeof(uint32_t) * p);
  BIC bic = { discount, cov, n, p, get_cov_precomp, L, D, 0, 0, z };

  Bit_Array prefix = bta_alloc(p);
  Bit_Array skip = bta_alloc(p);
  Priority_Queue pq = pq_alloc(p);

  GST *gsts = malloc(sizeof(GST) * p);
  for (size_t i = 0; i < p; i++) gst_init(gsts + i, i, &bic);

  // MOVED HERE FROM THE BOSS CALL
  uint32_t *order = malloc(sizeof(uint32_t) * p);






  // TEMPORARY SOLUTION!
  uint8_t *tmp = malloc(sizeof(uint8_t) * p * p);

  // STILL TEMP --- ZERO OUT THE GRAPH
  for (size_t i = 0; i < p; i++) {
    for (size_t j = 0; j < p; j++) {
      tmp[i * p + j] = 0;
    }
  }






  // RANDOM RESTARTS ARE BEING IGNORED!
 

  // graphs are being passed and calculated in boss/sp
  // but not used in this function?

  if (!knwl.num_groups) {
    printf("HERHEHREHRHEHREHRHE\n");    
    // IF NO KNOWLEDGE
    for (size_t i = 0; i < p; i++) order[i] = i;
    shuffle(order, p);
    Py_BEGIN_ALLOW_THREADS
    // boss_search_alt(&bic, order, p, gsts, prefix, skip, &pq, tmp);
    sp_search(&bic, order, p, gsts, prefix, skip, &pq, tmp);
    Py_END_ALLOW_THREADS
  } else {
    // IF KNOWLEDGE
    // ASSUME GROUP PARTITION AND ARE IN ORDER
    for (size_t i = 0; i < p; i++) order[i] = knwl.group_members[i];
    uint32_t *suborder = order;
    for (size_t i = 0; i < knwl.num_groups; i++) {
      size_t sub_p = knwl.group_sizes[i];
      shuffle(suborder, sub_p);
      Py_BEGIN_ALLOW_THREADS
      // boss_search_alt(&bic, suborder, sub_p, gsts, prefix, skip, &pq, tmp);
      sp_search(&bic, suborder, sub_p, gsts, prefix, skip, &pq, tmp);
      Py_END_ALLOW_THREADS

      // current suborder is added to prefix
      for (size_t i = 0; i < sub_p; i++) bta_set(prefix, order[i]);

      suborder += sub_p;
    }
  }

  printf("\n\n\n");
  for (size_t i = 0; i <p; i++) printf(" %u", order[i]);
  printf("\n\n\n");

  // WE SHOULD NOT HAVE TO RECALCULATE THE PARENTS
  bta_reset(prefix);
  for (size_t i = 0; i < p; i++) {
    gst_trace(gsts + order[i], prefix, skip, &pq, &bic);
    bic_shrink(&bic);
    bta_set(prefix, order[i]);
    for (size_t j = 0; j < bic.q; j++) {
      tmp[order[i] * p + bic.z[j]] = 1;
    }
  }




  // MOVED HERE FROM THE BOSS CALL
  free(order);



  // freeing GST
  for (size_t i = 0; i < p; i++) gst_free(gsts + i);
  free(gsts);

  
  bta_free(prefix); 
  bta_free(skip);
  pq_free(pq);


  // freeing components of BIC
  free(L);
  free(D);
  free(z);

  EdgeList graph = {0};
  graph.edges = malloc(sizeof(Edge) * p * p); // overkill for now

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

  uint32_t *itr;
  
  itr = data_view.buf;
  uint32_t n = *itr++;
  uint32_t p = *itr++;
  float *data = (float *)itr;

  // printf("%u %u\n", n, p);

  itr = knwl_view.buf;
  Knowledge knwl = {0};
  parse_knowledge(&knwl, itr);

  double *L = malloc(sizeof(double) * TNU(p));
  double *D = malloc(sizeof(double) * p);
  uint32_t *z = malloc(sizeof(uint32_t) * p);

  // TEMPORARY SOLUTION!
  uint8_t *tmp = malloc(sizeof(uint8_t) * p * p);

  BIC bic = { discount, data, n, p, get_cov_onfly, L, D, 0, 0, z };

  // ADD KNOWLEDGE TO THIS CALL!
  Py_BEGIN_ALLOW_THREADS
  boss_search(&bic, restarts, tmp);
  Py_END_ALLOW_THREADS

  free(L);
  free(D);
  free(z);

  // THE CURRENTLY RETURNED GRAPH OBJECT IS A TMP SOLUTION
  EdgeList graph = {0};
  graph.edges = malloc(sizeof(Edge) * p * p); // overkill for now

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
  { "boss_from_cov", (PyCFunction)(void(*)(void))boss_from_cov, METH_VARARGS | METH_KEYWORDS, "runs boss from cov..." },
  { "boss_from_data", (PyCFunction)(void(*)(void))boss_from_data, METH_VARARGS | METH_KEYWORDS, "runs boss from data..." },
  { NULL, NULL, 0, NULL }
};


static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "c_backend",
  "C Backend for Causal Graph Estimation Toolbox",
  -1,
  methods,
  NULL,
  NULL,
  NULL,
  NULL
};


PyMODINIT_FUNC PyInit_c_backend(void)
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
