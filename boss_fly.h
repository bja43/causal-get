#ifndef BOSS_H_ 
#define BOSS_H_

#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>

#define BTA_IMPLEMENTATION
#define PQ_IMPLEMENTATION
#define BIC_IMPLEMENTATION
#define GST_IMPLEMENTATION

#ifndef BTA_H_
#include "bta.h"
#endif // BTA_H_

#ifndef PQ_H_
#include "pq.h"
#endif // PQ_H_

#ifndef BIC_H_
#include "fly.h"
#endif // BIC_H_

#ifndef GST_H_
#include "gst.h"
#endif // GST_H_

void shuffle(uint32_t *arr, size_t size);

bool better_mutation(uint32_t *order, uint32_t* ptr, GST *gsts, Bit_Array prefix, Bit_Array skip, Priority_Queue *pq, BIC *bic);
void boss_search(float *X, size_t n, size_t p, float lambda, size_t restarts, uint8_t *graph);

#endif // BOSS_H_

#ifdef BOSS_IMPLEMENTATION

void shuffle(uint32_t *arr, size_t size)
{
  if (size < 2) return;
  
  for (size_t i = size - 1; i > 0; i--) {
    size_t j = rand() % (i + 1);   // randint(i + 1)
    uint32_t tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
}


bool better_mutation(uint32_t *order, uint32_t* ptr, GST *gsts, Bit_Array prefix, Bit_Array skip, Priority_Queue *pq, BIC *bic)
{
  uint32_t *first = order;
  uint32_t *last = order + bic->p - 1;

  float *scores = malloc(sizeof(float) * bic->p);
  float *best = scores + bic->p - 1;

  float score = 0;
  bta_reset(prefix);

  while (1) {
    *scores = gst_trace(gsts + *ptr, prefix, skip, pq, bic) + score;
    if (order == ptr) break;
    score += gst_trace(gsts + *order, prefix, skip, pq, bic);
    bta_set(prefix, *order);
    order++;
    scores++;
  }

  while (order != last) {
    order++;
    scores++;
    score += gst_trace(gsts + *order, prefix, skip, pq, bic);
    bta_set(prefix, *order);
    *scores = gst_trace(gsts + *ptr, prefix, skip, pq, bic) + score;
  }

  score = 0;
  bta_set(prefix, *ptr);

  while (1) {
    *scores += score;
    if (*scores > *best) best = scores;
    if (order == ptr) break;
    bta_clear(prefix, *order);
    score += gst_trace(gsts + *order, prefix, skip, pq, bic);
    order--;
    scores--;
  }

  while (order != first) {
    order--;
    scores--;
    score += gst_trace(gsts + *order, prefix, skip, pq, bic);
    bta_clear(prefix, *order);
    *scores += score;
    if (*scores > *best) best = scores;
  }

  size_t i = ptr - order;
  size_t j = best - scores;

  if (scores[i] + 1e-3 > *best) {
    free(scores);
    return false;
  }
  free(scores);

  uint32_t value = order[i];

  if (i < j) for (size_t k = i; k < j; k++) order[k] = order[k + 1];
  else for (size_t k = i; k > j; k--) order[k] = order[k - 1];
  order[j] = value;

  return true;
}


void boss_search(float *X, size_t n, size_t p, float lambda, size_t restarts, uint8_t *graph)
{
  double *L = malloc(sizeof(double) * TNU(p));
  double *D = malloc(sizeof(double) * p);
  uint32_t *z = malloc(sizeof(uint32_t) * p);

  BIC bic = { lambda, X, log(n) / (2 * n), n, p, L, D, 0, 0, z };

  Priority_Queue pq = pq_alloc(p);
  Bit_Array prefix = bta_alloc(p);
  Bit_Array skip = bta_alloc(p);

  uint32_t* order = malloc(sizeof(uint32_t) * p);
  uint32_t* best = malloc(sizeof(uint32_t) * p);
  uint32_t* itr = malloc(sizeof(uint32_t) * p);
  for (size_t i = 0; i < p; i++) {
    order[i] = i;
    best[i] = i;
    itr[i] = i;
  }

  uint32_t *ptr;

  GST *gsts = malloc(sizeof(GST) * p);
  for (size_t i = 0; i < p; i++) gst_init(gsts + i, i, &bic);

  float best_score;
  bool improved;

  for (size_t i = 0; i < restarts; i++) {
    shuffle(order, p);

    printf("%zu\n", i);

    do {
      for (size_t j = 0; j < p; j++) itr[j] = order[j];

      printf("bm\n");

      improved = false;
      for (size_t j = 0; j < p; j++) {
        ptr = order;
        while (*ptr != itr[j]) ptr++;
        improved |= better_mutation(order, ptr, gsts, prefix, skip, &pq, &bic);
      }
    } while(improved);
   
    // GET THE SCORE OF THE CURRENT ORDER
    float score = 0;
    bta_reset(prefix);
    for (size_t j = 0; j < p; j++) {
      score += gst_trace(gsts + order[j], prefix, skip, &pq, &bic);
      bta_set(prefix, order[j]);
    }

    printf("%f\n", score);

    if (i == 0 || score > best_score) {
      best_score = score;
      for (size_t j = 0; j < p; j++) best[j] = order[j];
    }
  }

  printf("%f\n", best_score);

  for (size_t i = 0; i < p; i++) {
    for (size_t j = 0; j < p; j++) {
      graph[i * p + j] = 0;
    }
  }

  bta_reset(prefix);
  for (size_t i = 0; i < p; i++) {
    gst_trace(gsts + best[i], prefix, skip, &pq, &bic);
    bic_shrink(&bic);
    bta_set(prefix, best[i]);
    for (size_t j = 0; j < bic.q; j++) {
      graph[best[i] * p + bic.z[j]] = 1;
    }
  }

  for (size_t i = 0; i < p; i++) printf("%u ", best[i]);
  printf("\n");

  for (size_t i = 0; i < p; i++) gst_free(gsts + i);
  free(gsts);

  free(order);
  free(best);
  free(itr);

  pq_free(pq);
  bta_free(prefix); 
  bta_free(skip);

  free(L);
  free(D);
  free(z);
}

#endif // BOSS_IMPLEMENTATION
