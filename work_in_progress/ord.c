#include <time.h>

#define ORD_IMPLEMENTAION

#ifndef ORD_H_
#include "ord.h"
#endif // ORD_H_

int main()
{
  srand(time(NULL));

  size_t p = 4;
  Order ord = ord_alloc(p);
  ord_init(ord);
  ord_print(ord);

  Order copy = ord_alloc(p);
  ord_copy(ord, copy);

  ord_shuffle(ord);

  ord_print(ord);
  ord_print(copy);

  ord_free(ord);
  ord_free(copy);

  return 0;
}
