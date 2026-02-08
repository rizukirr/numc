/**
 * @file array_print.c
 * @brief Array printing for all types, with stride-aware indexing.
 */

#include <numc/array.h>
#include <stdio.h>

/**
 * @brief Print a single element at the given byte offset.
 */
static void print_element(const char *data, size_t byte_offset,
                          NUMC_TYPE numc_type) {
  const void *elem = data + byte_offset;
  switch (numc_type) {
  case NUMC_TYPE_BYTE:
    printf("%d", *(const NUMC_BYTE *)elem);
    break;
  case NUMC_TYPE_UBYTE:
    printf("%u", *(const NUMC_UBYTE *)elem);
    break;
  case NUMC_TYPE_SHORT:
    printf("%d", *(const NUMC_SHORT *)elem);
    break;
  case NUMC_TYPE_USHORT:
    printf("%u", *(const NUMC_USHORT *)elem);
    break;
  case NUMC_TYPE_INT:
    printf("%d", *(const NUMC_INT *)elem);
    break;
  case NUMC_TYPE_UINT:
    printf("%u", *(const NUMC_UINT *)elem);
    break;
  case NUMC_TYPE_LONG:
    printf("%ld", *(const NUMC_LONG *)elem);
    break;
  case NUMC_TYPE_ULONG:
    printf("%lu", *(const NUMC_ULONG *)elem);
    break;
  case NUMC_TYPE_FLOAT:
    printf("%g", *(const NUMC_FLOAT *)elem);
    break;
  case NUMC_TYPE_DOUBLE:
    printf("%g", *(const NUMC_DOUBLE *)elem);
    break;
  }
}

/**
 * @brief Recursive printer that walks dimensions using strides.
 *
 * @param data       Base data pointer (as char* for byte arithmetic).
 * @param shape      Shape array.
 * @param strides    Strides array (bytes per step in each dimension).
 * @param ndim       Total number of dimensions.
 * @param dim        Current dimension being printed.
 * @param offset     Current byte offset from data base.
 * @param numc_type  Element type for formatting.
 */
static void print_recursive(const char *data, const size_t *shape,
                             const size_t *strides, size_t ndim, size_t dim,
                             size_t offset, NUMC_TYPE numc_type) {
  printf("[");
  for (size_t i = 0; i < shape[dim]; i++) {
    size_t elem_offset = offset + i * strides[dim];

    if (dim == ndim - 1) {
      print_element(data, elem_offset, numc_type);
    } else {
      print_recursive(data, shape, strides, ndim, dim + 1, elem_offset,
                      numc_type);
    }

    if (i + 1 < shape[dim])
      printf(", ");
  }
  printf("]");
}

void array_print(const Array *array) {
  if (!array || !array->data) {
    printf("(null)\n");
    return;
  }

  if (array->size == 0) {
    printf("[]\n");
    return;
  }

  print_recursive((const char *)array->data, array->shape, array->strides,
                  array->ndim, 0, 0, array->numc_type);
  printf("\n");
}
