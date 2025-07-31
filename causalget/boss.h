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
#include "bic.h"
#endif // BIC_H_

#ifndef GST_H_
#include "gst.h"
#endif // GST_H_


// should order and score be included in the search state?
typedef struct {
  Bit_Array prefix; // maybe make this a pointer to be consistent
  Bit_Array skip; // maybe make this a pointer to be consistent
  Priority_Queue *pq; // the size changes so this has to be a pointer
  BIC *bic; // should this be included?
  GST *gsts;
  // uint32_t *order; // include this? should this be a suborder?
  // if so, we need the number of sample in it.
  // float scores; // include this?
} Search_State;


bool better_mutation(uint32_t *order, size_t p, uint32_t *ptr, GST *gsts, Bit_Array prefix, Bit_Array skip, Priority_Queue *pq, BIC *bic, float *scores);
void shuffle(uint32_t *arr, size_t size);

void boss_search_alt(BIC *bic, uint32_t *order, size_t p, GST *gsts, Bit_Array prefix, Bit_Array skip, Priority_Queue *pq, uint8_t *graph);

void boss_search(BIC *bic, size_t restarts, uint8_t *graph);



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








// *order is a pointer to the start of the current suborder
// p is the length of the current suborder
// *ptr is a pointer to the position

// the difference here is that we take a point to the starting position in the suborder and a length of the suborder
bool better_mutation(uint32_t *order, size_t p, uint32_t *ptr, GST *gsts, Bit_Array prefix, Bit_Array skip, Priority_Queue *pq, BIC *bic, float *scores)
{
  uint32_t *first = order;
  uint32_t *last = order + p - 1;

  // scores and order become iterators

  float *best = scores + p - 1;                // best is set to be the final score?

  // THE PREFIX NEEDS TO BE RESET TO A PARTICULAR STATE FOR SUBORDERS TO WORK
  // IF ITS JUST HANDLED OUTSIDE THEN NO WORRIES
  // prefix is now properly reset at the end of the function so it can be moved outside
  // bta_reset(prefix); // do I really want to handle resetting here?
  double score = 0; // must be a double to prevent catestrophic cancellation

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

  bta_clear(prefix, *ptr); // this resets the prefix to what it was prior to entering the function

  size_t i = ptr - order; // current location (in memory)
  size_t j = best - scores; // best location (in memory)

  // THIS IS A BIT OF A HACK (NEEDS TO BE 1e-3 BETTER THAN CURRENT)
  if (scores[i] + 1e-3 > *best) return false;

  // printf("   score: %.4f -> %.4f\n", scores[i], scores[j]); // added for debugging

  uint32_t value = order[i];

  if (i < j) for (size_t k = i; k < j; k++) order[k] = order[k + 1];
  else for (size_t k = i; k > j; k--) order[k] = order[k - 1];
  order[j] = value;

  // check the contents of the prefix
  // for (size_t k = 0; k < p; k++) {
  //   printf("%d ", (int)bta_check(prefix, k));
  // }
  // printf("\n");

  return true;
}


// maybe an order should be passed in?
// what about random restarts?
// handle them outside
// what about passing a graph in?


void boss_search_alt(BIC *bic, uint32_t *order, size_t p, GST *gsts, Bit_Array prefix, Bit_Array skip, Priority_Queue *pq, uint8_t *graph)
{

  // there should be a search struct that contains these three things + a bic score
  // these are the data requires to use a GST in search (again + a bic score)
  // perhaps the array of floats for score should be initialized here as well?


  uint32_t *itr = malloc(sizeof(uint32_t) * p);

  // will need one per suborder
  float *scores = malloc(sizeof(float) * p);
  
  uint32_t *ptr;


  bool improved;


  // printf("%zu\n", i);

  // itr is a copy of the entire order so that we can iterate over the order while it is being modified
  // we are iterating over the frozen copy
  // itr only need to be a copy of the suborder 
  do {
    for (size_t i = 0; i < p; i++) itr[i] = order[i];

    printf("better mutation...\n");

    // this sets prefix to be empty, but it should include all members of prior suborders
    // bta_reset(prefix); 

    // fixes but maybe slow?
    for(size_t i = 0; i < p; i++) bta_clear(prefix, order[i]);

    improved = false;
    for (size_t i = 0; i < p; i++) {
      ptr = order;
      while (*ptr != itr[i]) ptr++; // make sure ptr points at the (ith variable in the frozen order)'s location in order 
      improved |= better_mutation(order, p, ptr, gsts, prefix, skip, pq, bic, scores);
      // scores should be a pointer to the same offset as ptr (I think) so that things can run in parallel
    }
  } while(improved);
  
  free(scores);

  free(itr);

}


















void boss_search(BIC *bic, size_t restarts, uint8_t *graph)
{
  size_t p = bic->p;

  // there should be a search struct that contains these three things + a bic score
  // these are the data requires to use a GST in search (again + a bic score)
  // perhaps the array of floats for score should be initialized here as well?

  Priority_Queue pq = pq_alloc(p);
  Bit_Array prefix = bta_alloc(p);
  Bit_Array skip = bta_alloc(p);

  uint32_t *order = malloc(sizeof(uint32_t) * p);
  uint32_t *best = malloc(sizeof(uint32_t) * p);
  uint32_t *itr = malloc(sizeof(uint32_t) * p);
  for (size_t i = 0; i < p; i++) { // Maybe need to assign first shuffle?
    order[i] = i;
    best[i] = i;
    itr[i] = i;
  }


  // will need one per suborder
  float *scores = malloc(sizeof(float) * p);
  
  uint32_t *ptr;

  GST *gsts = malloc(sizeof(GST) * p);
  for (size_t i = 0; i < p; i++) gst_init(gsts + i, i, bic);

  float best_score;
  bool improved;

  for (size_t i = 0; i < restarts; i++) {
    shuffle(order, p);        // THIS SHUFFLE SHOULD ONLY SHUFFLE WITHIN SUBORDERS

    // printf("%zu\n", i);

    // itr is a copy of the entire order so that we can iterate over the order while it is being modified
    // we are iterating over the frozen copy
    // itr only need to be a copy of the suborder 
    do {
      for (size_t j = 0; j < p; j++) itr[j] = order[j];

      printf("better mutation...\n");

      // this sets prefix to be empty, but it should include all members of prior suborders
      bta_reset(prefix); 

      improved = false;
      for (size_t j = 0; j < p; j++) {
        ptr = order;
        while (*ptr != itr[j]) ptr++; // make sure ptr points at the (jth variable in the frozen order)'s location in order 
        improved |= better_mutation(order, p, ptr, gsts, prefix, skip, &pq, bic, scores);
        // scores should be a pointer to the same offset as ptr (I think) so that things can run in parallel
      }
    } while(improved);
   
    // GET THE SCORE OF THE CURRENT ORDER
    // THIS SHOULD BE DONE BETTER
    float score = 0;
    bta_reset(prefix);
    for (size_t j = 0; j < p; j++) {
      score += gst_trace(gsts + order[j], prefix, skip, &pq, bic);
      bta_set(prefix, order[j]);
    }

    // printf("%f\n", score);

    if (i == 0 || score > best_score) {
      best_score = score;
      for (size_t j = 0; j < p; j++) best[j] = order[j];
    }
  }

  // printf("%f\n", best_score);

  // ZERO OUT THE GRAPH
  for (size_t i = 0; i < p; i++) {
    for (size_t j = 0; j < p; j++) {
      graph[i * p + j] = 0;
    }
  }

  // WE SHOULD NOT HAVE TO RECALCULATE THE PARENTS
  bta_reset(prefix);
  for (size_t i = 0; i < p; i++) {
    gst_trace(gsts + best[i], prefix, skip, &pq, bic);
    bic_shrink(bic);
    bta_set(prefix, best[i]);
    for (size_t j = 0; j < bic->q; j++) {
      graph[best[i] * p + bic->z[j]] = 1;
    }
  }

  free(scores);

  for (size_t i = 0; i < p; i++) gst_free(gsts + i);
  free(gsts);

  free(order);
  free(best);
  free(itr);

  pq_free(pq);
  bta_free(prefix); 
  bta_free(skip);
}

#endif // BOSS_IMPLEMENTATION
