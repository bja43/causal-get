#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>

typedef struct {
  size_t p;
  uint8_t *di;
  uint8_t *un;
} Graph_Matrix;

Graph_Matrix gmat_alloc(size_t p);
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


// add function for getting a consistent order

// add function to dag to cpdag

// add ancester and decendant functions


Graph_Matrix gmat_alloc(size_t p)
{
  size_t size = (p * p + 7u) >> 3;

  Graph_Matrix gmat = {
    .p = p,
    .di = malloc(sizeof(uint8_t) * size),
    .un = malloc(sizeof(uint8_t) * size),
  };

  for (size_t i = 0; i < size; i++) {
    gmat.di[i] = 0;
    gmat.un[i] = 0;
  }

  return gmat;
}

void gmat_free(Graph_Matrix gmat)
{
  free(gmat.di);
  free(gmat.un);
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
      printf(" %d", gmat_is_di(gmat, j, i));
    }
    printf("\n");
  }
}

void gmat_print_un(Graph_Matrix gmat)
{
  for (size_t i = 0; i < gmat.p; i++) {
    for (size_t j = 0; j < gmat.p; j++) {
      printf(" %d", gmat_is_un(gmat, j, i));
    }
    printf("\n");
  }
}


int main()
{
  size_t p = 4;
  Graph_Matrix g = gmat_alloc(p);

  gmat_add_di(g, 1, 2);
  gmat_add_di(g, 2, 3);

  gmat_print_di(g);
  printf("\n");
  gmat_print_un(g);

  gmat_free(g);
  return 0;
}
