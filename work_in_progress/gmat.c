#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>

#define ORD_IMPLEMENTAION

#ifndef ORD_H_
#include "ord.h"
#endif // ORD_H_

// typedef struct {
//   size_t p;
//   uint8_t *di;
//   uint8_t *un;
// } Graph_Matrix;

typedef struct {
  size_t p;
  uint8_t *bits;
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



void gmat_consistent_order(Graph_Matrix gmat, Order ord);
bool is_source(Graph_Matrix gmat, Order ord, size_t i);



// add function to dag to cpdag

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



Graph_Matrix gmat_alloc(size_t p)
{
  size_t size = (p * p + 7u) >> 3;

  // use this trick elsewhere
  Graph_Matrix gmat;
  gmat.p = p;
  // gmat.di = malloc(sizeof(*gmat.di) * size);
  // gmat.un = malloc(sizeof(*gmat.un) * size);
  // assert(gmat.di != NULL);
  // assert(gmat.un != NULL);
  gmat.bits = malloc(sizeof(*gmat.bits) * 2 * size);
  assert(gmat.bits != NULL);
  gmat.di = gmat.bits;
  gmat.un = gmat.bits + size;
  

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
  // copy.di = malloc(sizeof(*copy.di) * size),
  // copy.un = malloc(sizeof(*copy.un) * size),
  // assert(copy.di != NULL);
  // assert(copy.un != NULL);
  copy.bits = malloc(sizeof(*gmat.bits) * 2 * size);
  assert(copy.bits != NULL);
  copy.di = copy.bits;
  copy.un = copy.bits + size;

  for (size_t i = 0; i < size; i++) {
    copy.di[i] = gmat.di[i];
    copy.un[i] = gmat.un[i];
  }

  return copy;
}

void gmat_free(Graph_Matrix gmat)
{
  // free(gmat.di);
  // free(gmat.un);
  free(gmat.bits);
}

void gmat_add_di(Graph_Matrix gmat, size_t i, size_t j)
{
  assert(i < gmat.p);
  assert(j < gmat.p);
  size_t idx = i + j * gmat.p;
  gmat.di[idx >> 3] |= (1u << (idx & 7u));
}

void gmat_add_un(Graph_Matrix gmat, size_t i, size_t j)
{
  assert(i < gmat.p);
  assert(j < gmat.p);
  size_t idx = i + j * gmat.p;
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
  size_t idx = i + j * gmat.p;
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

// this could be better optimized, but does it really matter?
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
