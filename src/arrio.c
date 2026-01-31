#include "arrio.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void array_print(const float *data, const size_t *shape, size_t ndim,
                 size_t dim, size_t *offset) {
  printf("[");
  for (size_t i = 0; i < shape[dim]; i++) {
    if (dim == ndim - 1) {
      // Last dimension - print the element
      printf("%g", data[*offset]);
      (*offset)++;
    } else {
      // Recursively print inner dimensions
      array_print(data, shape, ndim, dim + 1, offset);
    }

    if (i + 1 < shape[dim]) {
      printf(", ");
    }
  }
  printf("]");
}
