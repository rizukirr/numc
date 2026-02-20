#include "internal.h"
#include <numc/array.h>
#include <stdio.h>

#define NUMC_PRINT_TYPES(X)                                                    \
  X(NUMC_DTYPE_INT8, NUMC_INT8, "d")                                           \
  X(NUMC_DTYPE_UINT8, NUMC_UINT8, "u")                                         \
  X(NUMC_DTYPE_INT16, NUMC_INT16, "d")                                         \
  X(NUMC_DTYPE_UINT16, NUMC_UINT16, "u")                                       \
  X(NUMC_DTYPE_INT32, NUMC_INT32, "d")                                         \
  X(NUMC_DTYPE_UINT32, NUMC_UINT32, "u")                                       \
  X(NUMC_DTYPE_INT64, NUMC_INT64, "ld")                                        \
  X(NUMC_DTYPE_UINT64, NUMC_UINT64, "lu")                                      \
  X(NUMC_DTYPE_FLOAT32, NUMC_FLOAT32, "g")                                     \
  X(NUMC_DTYPE_FLOAT64, NUMC_FLOAT64, "g")

static int elem_width(const char *data, size_t byte_offset, NumcDType dtype) {
  const void *elem = data + byte_offset;
  switch (dtype) {
#define X(DT, CT, FMT) case DT: return snprintf(NULL, 0, "%" FMT, *(const CT *)elem);
    NUMC_PRINT_TYPES(X)
#undef X
  }
  return 0;
}

static void print_element(const char *data, size_t byte_offset, NumcDType dtype,
                          int width) {
  const void *elem = data + byte_offset;
  switch (dtype) {
#define X(DT, CT, FMT) case DT: printf("%*" FMT, width, *(const CT *)elem); break;
    NUMC_PRINT_TYPES(X)
#undef X
  }
}

static int max_elem_width(const char *data, const size_t *shape,
                          const size_t *strides, size_t ndim, size_t dim,
                          size_t offset, NumcDType dtype) {
  int max_w = 0;
  for (size_t i = 0; i < shape[dim]; i++) {
    size_t off = offset + i * strides[dim];
    int w;
    if (dim == ndim - 1)
      w = elem_width(data, off, dtype);
    else
      w = max_elem_width(data, shape, strides, ndim, dim + 1, off, dtype);
    if (w > max_w)
      max_w = w;
  }
  return max_w;
}

static void print_recursive(const char *data, const size_t *shape,
                            const size_t *strides, size_t ndim, size_t dim,
                            size_t offset, NumcDType dtype, int width,
                            size_t indent) {
  printf("[");
  for (size_t i = 0; i < shape[dim]; i++) {
    size_t off = offset + i * strides[dim];

    if (dim == ndim - 1) {
      print_element(data, off, dtype, width);
      if (i + 1 < shape[dim])
        printf(", ");
    } else {
      if (i > 0) {
        printf(",");
        /* blank lines between sub-arrays: ndim - dim - 1 newlines total */
        for (size_t b = 0; b < ndim - dim - 1; b++)
          printf("\n");
        for (size_t s = 0; s < indent + 1; s++)
          putchar(' ');
      }
      print_recursive(data, shape, strides, ndim, dim + 1, off, dtype, width,
                      indent + 1);
    }
  }
  printf("]");
}

void numc_array_print(const NumcArray *array) {
  if (!array) {
    printf("(null)\n");
    return;
  }

  if (numc_array_size(array) == 0) {
    printf("[]\n");
    return;
  }

  void *data = numc_array_data(array);
  size_t dim = numc_array_ndim(array);
  size_t shape[dim];
  numc_array_shape(array, shape);
  size_t strides[dim];
  numc_array_strides(array, strides);
  NumcDType dtype = numc_array_dtype(array);

  int width =
      max_elem_width((const char *)data, shape, strides, dim, 0, 0, dtype);

  print_recursive((const char *)data, shape, strides, dim, 0, 0, dtype, width,
                  0);
  printf("\n");
}
