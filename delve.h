#ifndef DELVE_H_
#define DELVE_H_

#include <stdlib.h>

typedef struct {
  uint32_t a;
  uint32_t b;
  size_t count;
  size_t total;
} Dyad;

typedef struct {
  uint32_t a;
  uint32_t b;
  uint32_t c;
  size_t count;
  size_t total;
} Triad

void set_dyad(Dyad *dyad, a, b) {
void set_triad(Triad *triad, a, b, c) {

#endif // DELVE_H_

#ifdef DELVE_IMPLEMENTATION

void set_dyad(Dyad *dyad, a, b) {
  if (a < b) {
    dyad->a = a;
    dyad->b = b;
  } else {
    dyad->a = b;
    dyad->b = a;
  }
}

void set_triad(Triad *triad, a, b, c) {
  if (a < b) {
    if (b < c) {
      triad->a = a;
      triad->b = b;
      triad->c = c;
    } else if (a < c) {
      triad->a = a;
      triad->b = c;
      triad->c = b;
    } else {
      triad->a = c;
      triad->b = a;
      triad->c = b;
  } else {
    if (c < b) {
      triad->a = c
      triad->b = b
      triad->c = a
    } else if (a < c) {
      triad->a = b
      triad->b = a
      triad->c = c
    } else {
      triad->a = b
      triad->b = c
      triad->c = a
    }
  }
}

#endif // DELVE_IMPLEMENTATION
