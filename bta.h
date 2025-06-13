#ifndef BTA_H_
#define BTA_H_

#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>

typedef struct {
  size_t size;
  uint8_t *bits;
} Bit_Array;

Bit_Array bta_alloc(size_t num_bits);
void bta_free(Bit_Array bta);
void bta_reset(Bit_Array bta);
void bta_set(Bit_Array bta, size_t idx);
void bta_clear(Bit_Array bta, size_t idx);
bool bta_check(Bit_Array bta, size_t idx);

#endif // BTA_H_

#ifdef BTA_IMPLEMENTATION

Bit_Array bta_alloc(size_t num_bits)
{
  size_t size = (num_bits + 7u) >> 3;

  Bit_Array bta = {
    .size = size,
    .bits = malloc(sizeof(uint8_t) * size),
  };

  return bta;
}


void bta_free(Bit_Array bta)
{
  free(bta.bits);
}


void bta_reset(Bit_Array bta)
{
  for (size_t i = 0; i < bta.size; i++) bta.bits[i] = 0;
}


void bta_set(Bit_Array bta, size_t idx)
{
  assert((idx + 7u) >> 3 < bta.size);
  bta.bits[idx >> 3] |= (1u << (idx & 7u));
}


void bta_clear(Bit_Array bta, size_t idx)
{
  assert((idx + 7u) >> 3 < bta.size);
  bta.bits[idx >> 3] &= ~(1u << (idx & 7u));
}


bool bta_check(Bit_Array bta, size_t idx)
{
  assert((idx + 7u) >> 3 < bta.size);
  return bta.bits[idx >> 3] & (1u << (idx & 7u));
}

#endif // BTA_IMPLEMENTATION
