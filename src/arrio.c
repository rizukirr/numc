#include "arrio.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_elem(const void *ptr, size_t elem_size) {
  if (elem_size == sizeof(int)) {
    printf("%d", *(const int *)ptr);
  } else if (elem_size == sizeof(float)) {
    printf("%g", *(const float *)ptr);
  } else if (elem_size == sizeof(double)) {
    printf("%g", *(const double *)ptr);
  } else {
    printf("?");
  }
}

static void array_print_recursive(const Array *array, size_t dim,
                                  size_t *indices) {
  printf("[");

  for (size_t i = 0; i < array->shape[dim]; i++) {
    indices[dim] = i;

    if (dim + 1 == array->ndim) {
      void *ptr = array_get_ptr(array, indices);
      print_elem(ptr, array->elem_size);
    } else {
      array_print_recursive(array, dim + 1, indices);
    }

    if (i + 1 < array->shape[dim])
      printf(", ");
  }

  printf("]");
}

void array_print(const Array *array) {
  if (!array) {
    printf("(null)\n");
    return;
  }

  size_t *indices = calloc(array->ndim, sizeof(size_t));
  array_print_recursive(array, 0, indices);
  printf("\n");
  free(indices);
}
