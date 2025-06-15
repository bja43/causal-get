#ifndef BIC_H_
#define BIC_H_

#include <stdlib.h>
#include <math.h>

#define TNU(x) (x * (x + 1) >> 1)
#define GET(X, r, c) X[TNU(r) + c]

#define BTA_IMPLEMENTATION

#ifndef BTA_H_
#include "bta.h"
#endif // BTA_H_

typedef struct BIC BIC;

struct BIC {
  // parameters
  float discount;

  // data
  float *X;
  size_t n;
  size_t p;

  // covariance
  float (*get_cov) (BIC *bic, uint32_t a, uint32_t b);

  // LDL
  double *L;
  double *D;
  size_t q;

  // indices
  uint32_t y;
  uint32_t *z;
};


float get_cov_precomp(BIC *bic, uint32_t a, uint32_t b);
float get_cov_onfly(BIC *bic, uint32_t a, uint32_t b);

void bic_update(BIC *bic, uint32_t x);
float bic_score(BIC *bic);

int bic_contains(uint32_t *z, size_t size, uint32_t x);
int bic_find(uint32_t *z, size_t size, uint32_t x);

void bic_grow(BIC *bic, Bit_Array prefix);
void bic_shrink(BIC *bic);

#endif // BIC_H_

#ifdef BIC_IMPLEMENTATION


float get_cov_precomp(BIC *bic, uint32_t a, uint32_t b)
{
  return bic->X[a * bic->p + b];
}


float get_cov_onfly(BIC *bic, uint32_t a, uint32_t b)
{
  size_t n = bic->n;
  if (n < 2) return 0.0f;

  float *x = &bic->X[a * n];
  float *y = &bic->X[b * n];

  float mu_x = 0.0f;
  float mu_y = 0.0f;
  for (size_t i = 0; i < n; i++) {
    mu_x += x[i];
    mu_y += y[i];
  }
  mu_x /= n;
  mu_y /= n;

  float cov_xy = 0.0f;
  for (size_t i = 0; i < n; i++) {
    cov_xy += (x[i] - mu_x) * (y[i] - mu_y);
  }
  cov_xy /= n;

  return cov_xy;
}


void bic_update(BIC *bic, uint32_t x)
{
  double *L = bic->L;
  double *D = bic->D;
  size_t i = bic->q;

  uint32_t *z = bic->z;

  for (size_t j = 0; j < i; j++) {
    //double acc = X[x * p + z[j]];
    double acc = bic->get_cov(bic, x, z[j]);
    for (size_t k = 0; k < j; k++) {
      acc -= GET(L, i, k) * GET(L, j, k);
    }
    GET(L, i, j) = acc * D[j];
  }

  // D[i] = X[p * x + x];
  D[i] = bic->get_cov(bic, x, x);

  // THE SUM OF SQUARES MUST BE LESS THAN ONE
  for (size_t k = 0; k < i; k++) {
    D[i] -= GET(L, i, k) * GET(L, i, k);
  }
  // CHECK HERE FOR SIGULARITY
  // CLIP THIS?
  D[i] = 1.0 / sqrt(D[i]);
}


float bic_score(BIC *bic)
{
  bic_update(bic, bic->y);

  float c = bic->discount;                  // discount
  size_t k = bic->q;                        // num parents
  float ll = log(bic->D[k]);                // (nll / n) + const
  float logn = log(bic->n) / (2 * bic->n);  // logn / 2n

  return ll - c * k * logn;
}


// change to bool?
int bic_contains(uint32_t *z, size_t size, uint32_t x)
{
  for (size_t i = 0; i < size; i++) {
    if (z[i] == x) return 1;
  }

  return 0;
}


// change to uint32_t or size_t?
int bic_find(uint32_t *z, size_t size, uint32_t x)
{
  for (size_t i = 0; i < size; i++) {
    if (z[i] == x) return i;
  }

  // if not found return the last element 
  // (assumes the item is in the list)
  return size;
}


void bic_grow(BIC *bic, Bit_Array prefix)
{
  size_t p = bic->p;
  uint32_t y = bic->y;
  uint32_t *z = bic->z;

  int add = -1;
  float score;
  float best = bic_score(bic);

  while(bic->q < p) {
    for (uint32_t x = 0; x < p; x++) {
      if (x == y) continue;
      if (bta_check(prefix, x)) continue; // backward? continue if not in prefix?
      if (bic_contains(z, bic->q, x)) continue;

      bic_update(bic, x);
      z[bic->q++] = x;
      score = bic_score(bic);
      bic->q--;

      if (score > best) { 
        best = score;
        add = x;
      }
    }

    if (add != -1) {
      bic_update(bic, add);
      z[bic->q++] = add;
      add = -1;
    } else break;
  }
}


void bic_shrink(BIC *bic)
{
  if (bic->q == 0) return;
  uint32_t *z = bic->z;

  int del = -1;
  float score;
  float best = bic_score(bic);

  for (size_t size = bic->q - 1; size > 0; size--) {
    for (size_t i = 0; i < size + 1; i++) {

      uint32_t x = z[size - i];
      z[size - i] = z[size];
      z[size] = x;

      for (bic->q = size - i; bic->q < size; bic->q++) {
        bic_update(bic, z[bic->q]);
      }
      score = bic_score(bic);

      if (score > best) { 
        best = score;
        del = x;
      }
    }

    if (del != -1) {
      int i = bic_find(z, size, del);
      z[i] = z[size];
      for (bic->q = i; bic->q < size; bic->q++) {
        bic_update(bic, z[bic->q]);
      }
      del = -1;
    } else {
      bic_update(bic, z[bic->q]);
      bic->q++;
      break;
    }
  }
}

#endif // BIC_IMPLEMENTATION
