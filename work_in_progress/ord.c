#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

typedef struct {
  size_t p;
  uint32_t *v;
} Order;

Order ord_alloc(size_t p);
Order ord_copy(Order ord);
void ord_free(Order ord);
void ord_print(Order ord);

Order ord_alloc(size_t p)
{
  Order ord = {
    .p = p,
    .v = malloc(sizeof(uint32_t) * p),
  };

  for (size_t i = 0; i < p; i++) {
    ord.v[i] = (uint32_t) i;
  }

  return ord;
}

Order ord_copy(Order ord)
{
  Order copy = {
    .p = ord.p,
    .v = malloc(sizeof(uint32_t) * p),
  };

  for (size_t i = 0; i < ord.p; i++) {
    copy.v[i] = ord.v[i];
  }

  return copy;
}

void ord_free(Order ord)
{
  free(ord.v);
}

void ord_print(Order ord)
{
  for (size_t i = 0; i < ord.p; i++) {
    printf(" %u", ord.v[i]);
  }
  printf("\n");
}


int main()
{
  size_t p = 4;
  Order ord = ord_alloc(p);

  ord_print(ord);

  ord_free(ord);
  return 0;
}
