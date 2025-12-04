#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>

#define ORD_IMPLEMENTAION

#ifndef ORD_H_
#include "ord.h"
#endif // ORD_H_

// represent un and bi with triangular matrices
typedef struct {
  size_t p;
  uint8_t *di;
  uint8_t *un;
} Graph_Matrix;

Graph_Matrix gmat_alloc(size_t p);
Graph_Matrix gmat_copy(Graph_Matrix gmat);
void gmat_free(Graph_Matrix gmat);

void gmat_add_di(Graph_Matrix gmat, size_t i, size_t j);
void gmat_add_un(Graph_Matrix gmat, size_t i, size_t j);

void gmat_rmv_di(Graph_Matrix gmat, size_t i, size_t j);
void gmat_rmv_un(Graph_Matrix gmat, size_t i, size_t j);

bool gmat_is_di(Graph_Matrix gmat, size_t i, size_t j);
bool gmat_is_un(Graph_Matrix gmat, size_t i, size_t j);
bool gmat_is_adj(Graph_Matrix gmat, size_t i, size_t j);

void gmat_print_di(Graph_Matrix gmat);
void gmat_print_un(Graph_Matrix gmat);


// currently being developed
void gmat_consistent_order(Graph_Matrix gmat, Order ord);
bool is_source(Graph_Matrix gmat, Order ord, size_t i);
void gmat_extend_pdag(Graph_Matrix gmat);


// there is a lot of utility in this lists
// perhaps rename them since they can be much more than lists
// maybe make a macro or alias for uint32_t that are to be used as vertices


// add ancester and decendant functions
// Start from the child and add paremts recursively
// this approach can be used to get all ancestors
// or to simply check if if another vertex is a parent


#ifndef ORD_H_
#include "ord.h"
#endif // ORD_H_


bool is_source(Graph_Matrix gmat, Order ord, size_t i)
{
  for (size_t j = 0; j < gmat.p; j++) {
    if (i == j) continue;
    if (ord_contains(ord, j)) continue;
    if (gmat_is_di(gmat, i, j)) return false;
  }
  return true;
}


void gmat_consistent_order(Graph_Matrix gmat, Order ord)
{
  assert(gmat.p == ord.p);
  ord.p = 0;
  bool acyclic;
  while (ord.p < gmat.p) {
    acyclic = false;
    for (size_t i = 0; i < gmat.p; i++) {
      if (ord_contains(ord, i)) continue;
      if (is_source(gmat, ord, i)) {
        ord.v[ord.p++] = i;
        acyclic = true;
      }
    }
    assert(acyclic);
  }
}


void gmat_extend_pdag(Graph_Matrix gmat)
{
  Order oriented = ord_alloc(g.p);
  bool sink, clique;

  for (size_t i = 0; oriented.p < gmat.p; i++) {
    if (ord.contains(oriented, i)) continue;

    sink = true;
    for (size_t j = 0; sink && j < gmat.p; j++) {
      if (!gmat_is_di(j, i)) continue;
      if (!ord_containts(oriented, j)) sink = false;
    }
    if (!sink) continue;

    clique = true;
    for (size_t j = 0; clique && j < i; j++) { // check the lt-neighbors
      if (!gmat_is_un(i, j)) continue;
      for (size_t k = 0; clique && k < gmat.p; k++) {
        if (i == k) continue;
        if (j == k) continue;
        if (!gmat_is_adj(i, j)) clique = false;
      }
    }
    if (!clique) continue;

    for (size_t j = 0; j < gmat.p; j++) {
      if (!gmat_is_un(i, j)) continue;
      gmat_rvm_un(i, j);
      gmat-add_di(i, j;)
    }

    oriented.v[oriented.p++] = i;
    i = 0;    
  }

  ord_free(oriented);
}


// add function to dag to cpdag
// order edges
// find compelled


// move this somewhere else
#define append(xs,x) \
  do{ \
    if (xs.size >= xs.capacity) { \
      if (xs.capacity == 0) xs.capacity = 256; \
      else xs.capacity *= 2; \
      xs.items = realloc(xs.items, sizeof(*xs.items) * xs.capacity); \
    } \
  while(0)


typedef struct {
  uint32_t a;
  uint32_t b;
} Edge;

typedef struct {
  Edge *items;
  size_t size;
  size_t capacity;
} Edges;


// this is not quite right but it is getting close
void order_edges(Graph_Matrix gmat, Order, ord)
{
  gmat_consistent_order(Graph_Matrix gmat, Order ord);
  Edges ordered = {0};
  Edges unordered = {0};
  
  for (size_t i = 0; i < ord.v; i++) {
    for (size_t j = 0; j < ord.v; j ++) {
      if (gmat_is_di(i, j)) {
        Edge edge = {i, j};
        append(unordered, edge);
      }
    }
  }

//while 
// set i = 0
// while there are unordered edges in G
  // let y be the lowest ordered node that has an unordered edge incident to it
  // let x be the highest ordered node for which x -> y is unordered
  // label x -> y with order i
  // i += 1
}


void find_compelled()
{
// order the edges in G using order_edges
// lebel every edges in G as unknown
// while there are edges labelled unknown in G
  // let x -> y be the lowest ordered edge that is labelled unknown
  // for every edge w -> y labelled compelled
    // if w is not a parent of y, then label x -> y and every edge incident into y with compelled and goto start of while
    // else label w -> y with compelled
  // if there exists an edge z -> y such that z != x and z is not a parent of x, then label x -> y with compelled
  // else label x -> y and all unknown edges incident into y with reversible
}








Graph_Matrix gmat_alloc(size_t p)
{
  size_t size = (p * p + 7u) >> 3;

  // use this trick elsewhere
  Graph_Matrix gmat;
  gmat.p = p;
  // maybe do two mallocs ?
  gmat.di = malloc(sizeof(*gmat.di) * 2 * size);
  assert(gmat.di != NULL);
  gmat.un = gmat.di + size;
  

  for (size_t i = 0; i < size; i++) {
    gmat.di[i] = 0;
    gmat.un[i] = 0;
  }

  return gmat;
}

Graph_Matrix gmat_copy(Graph_Matrix gmat)
{
  size_t size = (gmat.p * gmat.p + 7u) >> 3;

  Graph_Matrix copy;
  copy.p = gmat.p,
  copy.di = malloc(sizeof(*gmat.di) * 2 * size);
  assert(copy.di!= NULL);
  copy.un = copy.di + size;

  for (size_t i = 0; i < size; i++) {
    copy.di[i] = gmat.di[i];
    copy.un[i] = gmat.un[i];
  }

  return copy;
}

void gmat_free(Graph_Matrix gmat)
{
  free(gmat.di);
}

void gmat_add_di(Graph_Matrix gmat, size_t i, size_t j)
{
  assert(i < gmat.p);
  assert(j < gmat.p);
  size_t idx = i + j * gmat.p;
  gmat.di[idx >> 3] |= (1u << (idx & 7u));
}

// perhaps store as triangular matrix and just sort the idxs
void gmat_add_un(Graph_Matrix gmat, size_t i, size_t j)
{
  assert(i < gmat.p);
  assert(j < gmat.p);
  size_t idx;
  idx = i + j * gmat.p;
  gmat.un[idx >> 3] |= (1u << (idx & 7u));
  idx = j + i * gmat.p;
  gmat.un[idx >> 3] |= (1u << (idx & 7u));
}

void gmat_rmv_di(Graph_Matrix gmat, size_t i, size_t j)
{
  assert(i < gmat.p);
  assert(j < gmat.p);
  size_t idx = i + j * gmat.p;
  gmat.di[idx >> 3] &= ~(1u << (idx & 7u));
}

void gmat_rmv_un(Graph_Matrix gmat, size_t i, size_t j)
{
  assert(i < gmat.p);
  assert(j < gmat.p);
  size_t idx;
  idx = i + j * gmat.p;
  gmat.un[idx >> 3] &= ~(1u << (idx & 7u));
  idx = j + i * gmat.p;
  gmat.un[idx >> 3] &= ~(1u << (idx & 7u));
}

bool gmat_is_di(Graph_Matrix gmat, size_t i, size_t j)
{
  assert(i < gmat.p);
  assert(j < gmat.p);
  size_t idx = i + j * gmat.p;
  return gmat.di[idx >> 3] & (1u << (idx & 7u));
}

bool gmat_is_un(Graph_Matrix gmat, size_t i, size_t j)
{
  assert(i < gmat.p);
  assert(j < gmat.p);
  size_t idx = i + j * gmat.p;
  return gmat.un[idx >> 3] & (1u << (idx & 7u));
}

bool gmat_is_adj(Graph_Matrix gmat, size_t i, size_t j)
{
  return gmat_is_di(gmat, i, j) || gmat_is_un(gmat, i, j);
}

void gmat_print_di(Graph_Matrix gmat)
{
  for (size_t i = 0; i < gmat.p; i++) {
    for (size_t j = 0; j < gmat.p; j++) {
      printf(" %d", gmat_is_di(gmat, i, j));
    }
    printf("\n");
  }
}

void gmat_print_un(Graph_Matrix gmat)
{
  for (size_t i = 0; i < gmat.p; i++) {
    for (size_t j = 0; j < gmat.p; j++) {
      printf(" %d", gmat_is_un(gmat, i, j));
    }
    printf("\n");
  }
}


int main()
{
  size_t p = 4;
  Graph_Matrix g = gmat_alloc(p);

  gmat_add_di(g, 2, 3);
  gmat_add_di(g, 1, 2);
  gmat_add_di(g, 0, 1);

  gmat_print_di(g);
  printf("\n");
  gmat_print_un(g);
  printf("\n");

  Graph_Matrix h = gmat_copy(g);

  gmat_print_di(h);
  printf("\n");
  gmat_print_un(h);
  printf("\n");

  Order ord = ord_alloc(p);
  gmat_consistent_order(g, ord);
  ord_print(ord);

  gmat_free(g);
  gmat_free(h);
  ord_free(ord);

  return 0;
}
