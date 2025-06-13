#ifndef GST_H_
#define GST_H_

#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>

#ifndef BIC_H_
#include "bic.h"
#endif // BIC_H_

#ifndef BTA_H_
#include "bta.h"
#endif // BTA_H_

#ifndef PQ_H_
#include "pq.h"
#endif // PQ_H_

typedef struct {
  float grow_score;
  float shrink_score;
  uint32_t idx;
  uint32_t children;
  uint32_t offset;
} GST_Node;

typedef struct {
  size_t size;
  size_t cap;
  GST_Node *root;
} GST;

void gst_init(GST *gst, uint32_t root_idx, BIC *bic);
void gst_grow(GST *gst, GST_Node *node, Bit_Array skip, Priority_Queue *pq, BIC *bic);

void gst_alloc_nodes(GST *gst, size_t size);
void gst_node_init(GST_Node *node, uint32_t idx, float score);
void gst_free(GST *gst);

float gst_next_node(GST *gst, size_t offset, Bit_Array prefix, Bit_Array skip, Priority_Queue *pq, BIC *bic);
float gst_trace(GST *gst, Bit_Array prefix, Bit_Array skip, Priority_Queue *pq, BIC *bic);

#endif // GST_H_

#ifdef GST_IMPLEMENTATION

void gst_init(GST *gst, uint32_t idx, BIC *bic)
{
  gst->size = 0;
  gst->cap = 0;
  gst_alloc_nodes(gst, 1);

  bic->y = idx;
  bic->q = 0;
  gst_node_init(gst->root, idx, bic_score(bic));
}


void gst_alloc_nodes(GST *gst, size_t size)
{
  if (gst->cap == 0) {
    gst->cap = 256;
    gst->root = malloc(sizeof(GST_Node) * gst->cap);
  }

  gst->size += size;
  if (gst->cap < gst->size) { 
    while (gst->cap < gst->size) gst->cap *= 2;
    // TODO: ADD ERRORS FOR (M/RE)ALLOC
    gst->root = realloc(gst->root, sizeof(GST_Node) * gst->cap);
  }
}


void gst_free(GST *gst)
{
  gst->size = 0;
  gst->cap = 0;
  free(gst->root);
}


void gst_node_init(GST_Node *node, uint32_t idx, float score)
{
  node->grow_score = score;
  node->shrink_score = NAN;
  node->idx = idx;
  node->children = 0;
  node->offset = 0;
}


void gst_grow(GST *gst, GST_Node *node, Bit_Array skip, Priority_Queue *pq, BIC *bic)
{
  pq->size = 0;
  for (uint32_t idx = 0; idx < bic->p; idx++) {
    if (bta_check(skip, idx)) continue;

    // TODO: FACTORED OUT INTO A FUNCTION CALL IN BIC?
    bic_update(bic, idx);
    bic->z[bic->q++] = idx;
    float score = bic_score(bic);
    bic->q--;

    if (node->grow_score < score) {
      pq_push(pq, idx, score);
    }
  }

  node->offset = gst->size;
  size_t child_offset = gst->size;
  node->children = pq->size;
  gst_alloc_nodes(gst, pq->size);

  while (0 < pq->size) {
    GST_Node *child = gst->root + child_offset++;
    PQ_Node pq_node = pq_pop(pq);
    gst_node_init(child, pq_node.idx, pq_node.val);
  }
}


float gst_trace(GST *gst, Bit_Array prefix, Bit_Array skip, Priority_Queue *pq, BIC *bic)
{
  // WHY IS THIS BEING RESET HERE INSTEAD OF OUTSIDE?
  // THIS SHOULD BE PART OF THE SEARCH-STATE STRUCT
  // (ALONG WITH PQ AND BIC)
  // KNOWLEDGE IMPLEMENTED HERE... ADD FORBIDDEN TO SKIP
  bta_reset(skip);
  bta_set(skip, gst->root->idx);

  bic->y = gst->root->idx;
  bic->q = 0;

  return gst_next_node(gst, 0, prefix, skip, pq, bic);
}


float gst_next_node(GST *gst, size_t offset, Bit_Array prefix, Bit_Array skip, Priority_Queue *pq, BIC *bic)
{
  GST_Node *node = gst->root + offset;
  // WHAT IF THE ROOT NODE DOES NOT ADD ANY PARENTS?
  if (node->offset == 0 && node->children == 0) {
    gst_grow(gst, node, skip, pq, bic);
    node = gst->root + offset; // GST->ROOT MAY CHANGE IN GROW CALL
  }

  for (size_t i = 0; i < node->children; i++) {
    GST_Node *child = gst->root + node->offset + i;

    bta_set(skip, child->idx);
    if (!bta_check(prefix, child->idx)) continue;

    bic_update(bic, child->idx);
    bic->z[bic->q++] = child->idx;

    return gst_next_node(gst, node->offset + i, prefix, skip, pq, bic);
  }

  if (isnan(node->shrink_score)) {
    bic_shrink(bic);
    node->shrink_score = bic_score(bic);
  }

  // CHECK INF --- THIS CHECK COULD/SHOULD BE BETTER
  if (isinf(node->shrink_score)) {
    printf("%f\n", node->shrink_score);
  }

  // THIS RETURNS THE CORRECT SCORE VALUE BUT THE BIC 
  // PARENTS ARE THE RESULTS OF GROW AND NOT GROW SHRINK
  return node->shrink_score;
}

#endif // GST_IMPLEMENTATION
