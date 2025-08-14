
#ifndef SP_H_ 
#define SP_H_

#include <stdio.h>
#include <stdlib.h>

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


void print_perm(uint32_t *arr, size_t n);
void sjt_perm(uint32_t *arr, size_t n);
void heap_perm(uint32_t *arr, size_t n);

static inline void swap_uint32(uint32_t *a, uint32_t *b);
static inline void swap_int(int *a, int *b);

void sp_search(BIC *bic, uint32_t *order, size_t p, GST *gsts, Bit_Array prefix, Bit_Array skip, Priority_Queue *pq, uint8_t *graph);
void sp_search_old(BIC *bic, uint32_t *order, size_t p, GST *gsts, Bit_Array prefix, Bit_Array skip, Priority_Queue *pq, uint8_t *graph);



#endif // SP_H_

#ifdef SP_IMPLEMENTATION






// int main() {
//   size_t n = 12;
//   uint32_t *arr = malloc(n * sizeof(uint32_t));

//   for (size_t i = 0; i < n; i++) arr[i] = i;

//   print_perm(arr, n);

//   printf("start\n");
//   sjt_perm(arr, n);
//   printf("end\n");

//   putchar('\n');

//   printf("start\n");
//   heap_perm(arr, n);
//   printf("end\n");

//   free(arr);
// }


// FROM CHATGPT
void print_perm(uint32_t *arr, size_t n) {
  for (size_t i = 0; i < n; i++) printf("%u ", arr[i]);
  putchar('\n');
}


// FROM CHATGPT
void sjt_perm(uint32_t *arr, size_t n) {
  if (n == 0) return;

  // Direction: -1 = left, 1 = right, 0 = stationary
  int8_t *dirs = malloc(n * sizeof(int8_t));
  if (!dirs) return;

  // Initialization
  // All initially move left
  // 0 is immobile at start
  for (uint32_t i = 0; i < n; i++) dirs[i] = -1;
  dirs[0] = 0; 

  // print_perm(arr, n);

  while (1) {
    // Step 1: Find largest mobile integer
    int32_t i = -1;
    uint32_t x = 0;
    for (size_t j = 0; j < n; j++) {
      int8_t d = dirs[arr[j]];
      int32_t k = j + d;
      if (d != 0 && k >= 0 && k < n && arr[j] > arr[k]) {
        if (arr[j] > x) {
          x = arr[j];
          i = j;
        }
      }
    }

    if (i == -1) break; // Done

    int8_t d = dirs[x];
    int32_t k = i + d;

    // Step 2: Swap arr[i] and arr[k]
    uint32_t tmp = arr[i];
    arr[i] = arr[k];
    arr[k] = tmp;

    // Step 3: Update direction of x (now at position k)
    if (k == 0 || k == n - 1 || arr[k + dirs[arr[k]]] > x) {
      dirs[x] = 0;
    }

    // Step 4: Reverse directions of elements larger than x
    for (size_t j = 0; j < n; j++) {
      if (arr[j] > x) {
        dirs[arr[j]] = (j < k) ? 1 : -1;
      }
    }

    // print_perm(arr, n);
  }

  free(dirs);
}


// FROM CHATGPT
void heap_perm(uint32_t *arr, size_t n) {
  if (n == 0) return;

  // Create a count array (like a stack frame counter)
  uint32_t *c = malloc(n * sizeof(uint32_t));
  if (!c) return;

  for (size_t i = 0; i < n; i++) c[i] = 0;

  // print_perm(arr, n); // First permutation

  size_t i = 0;
  while (i < n) {
    if (c[i] < i) {
      if (i % 2 == 0) {
        // swap(0, i)
        uint32_t tmp = arr[0];
        arr[0] = arr[i];
        arr[i] = tmp;
      } else {
        // swap(c[i], i)
        uint32_t tmp = arr[c[i]];
        arr[c[i]] = arr[i];
        arr[i] = tmp;
      }
      // print_perm(arr, n);
      c[i] += 1;
      i = 0;
    } else {
      c[i] = 0;
      i += 1;
    }
  }

  free(c);
}














#define LEFT  -1
#define RIGHT  1

static inline void swap_uint32(uint32_t *a, uint32_t *b) {
    uint32_t tmp = *a;
    *a = *b;
    *b = tmp;
}

static inline void swap_int(int *a, int *b) {
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

void sp_search(BIC *bic, uint32_t *order, size_t p, GST *gsts, Bit_Array prefix, Bit_Array skip, Priority_Queue *pq, uint8_t *graph)
{
  uint32_t *perm = malloc(sizeof(uint32_t) * p);
  for (size_t i = 0; i < p; i++) perm[i] = order[i];
  for (size_t i = 0; i < p; i++) bta_clear(prefix, perm[i]); // perhaps not required

  float *scores = malloc(sizeof(float) * p);
  double best = 0;

  for (size_t i = 0; i < p; i++) {
    scores[i] = gst_trace(gsts + i, prefix, skip, pq, bic);
    best += scores[i];
    bta_set(prefix, perm[i]);
  }  

  int *dir = malloc(sizeof(int) * p);
  for (size_t i = 0; i < p; i++) dir[i] = LEFT;

  bool more = true;
  while (more) {

    // Print current permutation
    // for (size_t i = 0; i < p; i++) {
    //   printf("%u ", perm[i]);
    // }
    // printf("\n");

    // Step 1: Find largest mobile element
    int largest_idx = -1;
    uint32_t largest_val = 0;
    for (size_t i = 0; i < p; i++) {
      int neighbor = i + dir[i];
      if (neighbor >= 0 && neighbor < (int)p && perm[i] > perm[neighbor]) {
        if (largest_idx == -1 || perm[i] > largest_val) {
          largest_val = perm[i];
          largest_idx = (int)i;
        }
      }
    }

    if (largest_idx == -1) {
      more = false; // No mobile element → done
      continue;
    }

    // Step 2: Swap with neighbor in direction
    int neighbor = largest_idx + dir[largest_idx];
    swap_uint32(&perm[largest_idx], &perm[neighbor]);
    swap_int(&dir[largest_idx], &dir[neighbor]);





    // score after the swap

    for (size_t i = 0; i < largest_idx; i++) bta_set(prefix, perm[i]);
    for (size_t i = largest_idx; i < p; i++) bta_clear(prefix, perm[i]);
    scores[largest_idx] = gst_trace(gsts + perm[largest_idx], prefix, skip, pq, bic);
    
    if (largest_idx < neighbor) bta_set(prefix, perm[largest_idx]);
    else bta_clear(prefix, perm[neighbor]);
    scores[neighbor] = gst_trace(gsts + perm[neighbor], prefix, skip, pq, bic);

    double score = 0;
    for (size_t i = 0; i < p; i++) score += scores[i];

    // if better score then update
    if (score > best + 1e-3) {
      for (size_t i = 0; i < p; i++) order[i] = perm[i];
      best = score;
    }


    // Step 2.1: Update the largest index
    largest_idx = neighbor;

    // Step 3: Reverse directions of all elements larger than moved element
    for (size_t i = 0; i < p; i++) {
      if (perm[i] > largest_val) {
        dir[i] = -dir[i];
      }
    }
  }

  free(dir);
  free(scores);
  free(perm);
}






















// steinhaus johnson trotter algorithm

void sp_search_old(BIC *bic, uint32_t *order, size_t p, GST *gsts, Bit_Array prefix, Bit_Array skip, Priority_Queue *pq, uint8_t *graph)
{

  printf("%zu\n", p);

  if (p == 0) return; // if order is empty?



  // setup scoring stuff

  uint32_t *tmp= malloc(sizeof(uint32_t) * p);

  float *scores = malloc(sizeof(float) * p);
  
  for (size_t i = 0; i < p; i++) tmp[i] = order[i];
  for (size_t i = 0; i < p; i++) bta_clear(prefix, tmp[i]); // perhaps not required

  for (size_t i = 0; i < p; i++) {
    scores[i] = gst_trace(gsts + i, prefix, skip, pq, bic);
    bta_set(prefix, tmp[i]);
  }
  
  double best = 0;
  for (size_t i = 0; i < p; i++) best += scores[i];



  // permutation stuff

  // Direction: -1 = left, 1 = right, 0 = stationary
  int8_t *dirs = malloc(p * sizeof(int8_t));
  if (!dirs) return; // failed to allocate memmory...

  // Initialization
  // All initially move left
  // 0 is immobile at start
  for (size_t i = 0; i < p; i++) dirs[i] = -1;
  dirs[0] = 0; 

  // everything points left at the start
  // except for the first position which is immobile

  while (1) {
    // Step 1: Find largest mobile integer
    size_t i = p; // index of the largest mobile number
    uint32_t x = 0; // value of the largest mobile number
    for (size_t j = 0; j < p; j++) {
      int8_t d = dirs[tmp[j]];
      size_t k = j + d;
      if (d != 0 && k >= 0 && k < p && tmp[j] > tmp[k]) { // check bounds and if mobile number
        if (tmp[j] > x) {
          x = tmp[j];
          i = j;
        }
      }
    }

    // no highest mobile number so we are done
    if (i == p) break;

    int8_t d = dirs[x];
    size_t k = i + d;

    // Step 2: Swap tmp[i] and tmp[k]
    uint32_t swp = tmp[i];
    tmp[i] = tmp[k];
    tmp[k] = swp;



    // score after the swap

    for (size_t j = 0; j < i; j++) bta_set(prefix, tmp[j]);
    for (size_t j = i; j < p; j++) bta_clear(prefix, tmp[j]);
    scores[i] = gst_trace(gsts + tmp[i], prefix, skip, pq, bic);
    
    if (i < k) bta_set(prefix, tmp[i]);
    else bta_clear(prefix, tmp[k]);
    scores[k] = gst_trace(gsts + tmp[k], prefix, skip, pq, bic);

    double cur = 0;
    for (size_t j = 0; j < p; j++) cur += scores[j];

    // if better score then update
    if (cur > best + 1e-3) {
      for (size_t j = 0; j < p; j++) order[i] = tmp[i];
      best = cur;
    }




    // continue the permutation stuff

    // Step 3: Update direction of x (now at position k)
    if (k == 0 || k == p - 1 || tmp[k + dirs[tmp[k]]] > x) {
      dirs[x] = 0;
    }

    // Step 4: Reverse directions of elements larger than x
    for (size_t j = 0; j < p; j++) {
      if (tmp[j] > x) {
        dirs[tmp[j]] = (j < k) ? 1 : -1;
      }
    }
  }

  free(dirs);

  free(scores);

  free(tmp);

  for (size_t i = 0; i < p; i++) printf("%u ", order[i]);
  printf("\n");

}







#endif // SP_IMPLEMENTATION
