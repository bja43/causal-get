// #ifndef GRAPH_H_
// #define GRAPH_H_


#include <stdlib.h> 
#include <stdio.h> 

#define SIZE 10
#define CAP 1000


typedef struct Vertex{
  size_t pa_size;
  size_t ch_size;
  size_t ne_size;
  size_t sib_size;
  uint32_t *pa;
  uint32_t *ch;
  uint32_t *ne;
  uint32_t *sib;
} Vertex;


typedef struct {
  size_t v_size;
  size_t v_cap;
  Vertex *v;
  size_t e_size;
  size_t e_cap;
  uint32_t *e;
} Graph;


void add_parent(Graph *g, uint32_t i, uint32_t j)
{
  for (size_t k = g->e_size; g->e + k > g->v[i].pa + g->v[i].pa_size; k--) {
    g->e[k] = g->e[k - 1];
  }
  g->e_size++;

  for (size_t k = i + 1; k < g->v_size; k++) {
    g->v[k].pa++;
    g->v[k].ch++;
    g->v[k].ne++;
    g->v[k].sib++;
  }
  g->v[i].ch++;
  g->v[i].ne++;
  g->v[i].sib++;

  g->v[i].pa[g->v[i].pa_size++] = j;
}


void add_child(Graph *g, uint32_t i, uint32_t j)
{
  for (size_t k = g->e_size; g->e + k > g->v[i].ch + g->v[i].ch_size; k--) {
    g->e[k] = g->e[k - 1];
  }
  g->e_size++;

  for (size_t k = i + 1; k < g->v_size; k++) {
    g->v[k].pa++;
    g->v[k].ch++;
    g->v[k].ne++;
    g->v[k].sib++;
  }
  g->v[i].ne++;
  g->v[i].sib++;

  g->v[i].ch[g->v[i].ch_size++] = j;
}


void add_neighbor(Graph *g, uint32_t i, uint32_t j)
{
  for (size_t k = g->e_size; g->e + k > g->v[i].ne + g->v[i].ne_size; k--) {
    g->e[k] = g->e[k - 1];
  }
  g->e_size++;

  for (size_t k = i + 1; k < g->v_size; k++) {
    g->v[k].pa++;
    g->v[k].ch++;
    g->v[k].ne++;
    g->v[k].sib++;
  }
  g->v[i].sib++;

  g->v[i].ne[g->v[i].ne_size++] = j;
}


void add_sibling(Graph *g, uint32_t i, uint32_t j)
{
  for (size_t k = g->e_size; g->e + k > g->v[i].sib + g->v[i].sib_size; k--) {
    g->e[k] = g->e[k - 1];
  }
  g->e_size++;

  for (size_t k = i + 1; k < g->v_size; k++) {
    g->v[k].pa++;
    g->v[k].ch++;
    g->v[k].ne++;
    g->v[k].sib++;
  }

  g->v[i].sib[g->v[i].sib_size++] = j;
}


void remove_parent(Graph *g, uint32_t i, uint32_t j)
{
  for (size_t k = 0; k < g.v[i].ch_size; k++) {
    if (g.v[i].pa[k] == j) {
      
      // shift everything left
      while (k < )
        g->e[k] = g->e[k + 1];
      }
      g->e_size--;

      break;
    }
  }
}



void remove_child(Graph *g, uint32_t i, uint32_t j) {}
void remove_neighbor(Graph *g, uint32_t i, uint32_t j) {}
void remove_sibling(Graph *g, uint32_t i, uint32_t j) {}



int main(void) {

  Graph g = {
    // vertices
    .v_size = 0,
    .v_cap = SIZE,
    .v = malloc(sizeof(Vertex) * SIZE),
    //edges
    .e_size = 0,
    .e_cap = CAP,
    .e = malloc(sizeof(uint32_t) * CAP),
  };

  // add variables
  for (uint32_t i = 0; i < SIZE; i++) {
    g.v_size++;
    g.v[i].pa_size = 0;
    g.v[i].ch_size = 0;
    g.v[i].ne_size = 0;
    g.v[i].sib_size = 0;
    // g.v[i].pa = g.e + g.e_size;
    // g.v[i].ch = g.e + g.e_size;
    // g.v[i].ne = g.e + g.e_size;
    // g.v[i].sib = g.e + g.e_size;
    g.v[i].pa = g.e
    g.v[i].ch = g.e
    g.v[i].ne = g.e
    g.v[i].sib = g.e
  }

  // add edges
  for (uint32_t i = 0; i < SIZE; i++) {

    printf("%u %zu\n", i, g.e_size);

    for (uint32_t j = i + 1; j < SIZE; j += 2) {

      if (j > i + 2) continue;

      // add j to the parents of i
      add_parent(&g, i, j);

      // add i to the children of j
      add_child(&g, j, i);

      // add i and j as neighbors of eachother
      add_neighbor(&g, i, j);
      add_neighbor(&g, j, i);

      // add i and j as siblings of eachother
      add_sibling(&g, i, j);
      add_sibling(&g, j, i);

      // remove parent
      
      // remove child

      // remove neighbor

      // remove sibling

    }
  }

  // print parents
  for (uint32_t i = 0; i < 10; i++) {
    for (size_t j = 0; j < g.v[i].pa_size; j++) {
      printf("%u <-- %u\n", i, g.v[i].pa[j]);
    }
  }

  printf("\n");

  // print children
  for (uint32_t i = 0; i < 10; i++) {
    for (size_t j = 0; j < g.v[i].ch_size; j++) {
      printf("%u --> %u\n", i, g.v[i].ch[j]);
    }
  }

  printf("\n");

  // print neighbors
  for (uint32_t i = 0; i < 10; i++) {
    for (size_t j = 0; j < g.v[i].ne_size; j++) {
      printf("%u --- %u\n", i, g.v[i].ne[j]);
    }
  }

  printf("\n");

  // print siblings 
  for (uint32_t i = 0; i < 10; i++) {
    for (size_t j = 0; j < g.v[i].sib_size; j++) {
      printf("%u <-> %u\n", i, g.v[i].sib[j]);
    }
  }

  free(g.v);
  free(g.e);
}


// #endif // GRAPH_H_


// #ifdef GRAPH_IMPLEMENTATION


// #endif // GRAPH_IMPLEMENTATION
