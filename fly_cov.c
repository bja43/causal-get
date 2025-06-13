#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define PRB_IMPLEMENTATION

#ifndef PRB_H_
#include "prb.h"
#endif // PRB_H_


int main() {

  srand(time(NULL));

  size_t n = 1000;
  float A[2][n];

  for (size_t i = 0; i < 2; i++) { 
    for (size_t j = 0; j < n; j++) { 
      A[i][j] = (float) prb_randn();
    }
  }

  // for (size_t i = 0; i < n; i++) { 
  //   printf("%f  %f\n", A[0][i], A[1][i]);
  // }

  double cov = 0;
  for (size_t i = 0; i < n; i++) { 
    cov += A[0][i] * A[1][i];
  }
  cov /= n;

  printf("\n%f\n", cov);

}
