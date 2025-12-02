#ifndef ORD_H_
#define ORD_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>

typedef struct {
  size_t p;
  uint32_t *v;
} Order;

Order ord_alloc(size_t p);
void ord_free(Order ord);

void ord_init(Order ord);
void ord_copy(Order ord, Order copy);

void ord_print(Order ord);
void ord_shuffle(Order ord);

bool ord_contains(Order ord, uint32_t x);

#endif // ORD_H_

#ifdef ORD_IMPLEMENTAION

Order ord_alloc(size_t p)
{
  Order ord;
  ord.p = p,
  ord.v = malloc(sizeof(*ord.v) * p),
  assert(ord.v != NULL);
  return ord;
}

void ord_free(Order ord)
{
  free(ord.v);
}

void ord_init(Order ord)
{
  for (size_t i = 0; i < ord.p; i++) {
    ord.v[i] = i;
  }
}

void ord_copy(Order ord, Order copy)
{
  assert(ord.p == copy.p);
  for (size_t i = 0; i < ord.p; i++) {
    copy.v[i] = ord.v[i];
  }
}

void ord_print(Order ord)
{
  for (size_t i = 0; i < ord.p; i++) {
    printf(" %u", ord.v[i]);
  }
  printf("\n");
}

void ord_shuffle(Order ord)
{
  if (ord.p < 2) return; 
  for (size_t i = ord.p - 1; i > 0; i--) {
    size_t j = rand() % (i + 1);
    uint32_t tmp = ord.v[i];
    ord.v[i] = ord.v[j];
    ord.v[j] = tmp;
  }
}

bool ord_contains(Order ord, uint32_t i)
{
  for (size_t j = 0; j < ord.p; j++) {
    if (i == ord.v[j]) return true;
  }
  return false;
}

#endif // ORD_IMPLEMENTAION
