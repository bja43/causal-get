#ifndef PQ_H_
#define PQ_H_

#include <stdlib.h>
#include <assert.h>

typedef struct {
  uint32_t idx;
  float val;
} PQ_Node;

typedef struct {
  size_t size;
  size_t cap;
  PQ_Node *nodes;
} Priority_Queue;

Priority_Queue pq_alloc(size_t cap);
void pq_free(Priority_Queue pq);

void pq_push(Priority_Queue *pq, uint32_t idx, float val);
PQ_Node pq_pop(Priority_Queue *pq);

#endif // PQ_H_

#ifdef PQ_IMPLEMENTATION

Priority_Queue pq_alloc(size_t cap)
{
  Priority_Queue pq = {
    .size = 0,
    .cap = cap,
    .nodes = malloc(sizeof(PQ_Node) * cap),
  };

  return pq;
}


void pq_free(Priority_Queue pq)
{
  free(pq.nodes);
}


void pq_push(Priority_Queue *pq, uint32_t idx, float val)
{
  assert(pq->size < pq->cap);

  size_t i = pq->size++;
  size_t j;

  while (i > 0) {
    j = (i - 1) / 2; // PARENT(i)
    if (val <= pq->nodes[j].val) break;
    pq->nodes[i] = pq->nodes[j];
    i = j;
  }

  pq->nodes[i].idx = idx;
  pq->nodes[i].val = val;
}


PQ_Node pq_pop(Priority_Queue *pq)
{
  assert(pq->size > 0);
  PQ_Node node = *pq->nodes;

  float val = pq->nodes[--pq->size].val;
  size_t i = 0;
  size_t j = 1;

  while (j < pq->size) {
    if (j + 1 < pq->size && pq->nodes[j + 1].val > pq->nodes[j].val) j++;
    if (val >= pq->nodes[j].val) break;
    pq->nodes[i] = pq->nodes[j];
    i = j;
    j = 2 * i + 1; // LEFT_CHILD(i)
  }

  pq->nodes[i] = pq->nodes[pq->size];

  return node;
}

#endif // PQ_IMPLEMENTATION
