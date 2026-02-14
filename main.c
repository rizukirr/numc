#include "array/array_core.h"
#include "array/array_dtype.h"
#include "array/array_print.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

int main(void) {
  size_t shape[] = {2, 3, 4};
  size_t dim = 3;
  int32_t data[][3][4] = {
      {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
      {{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}}};

  NumcCtx *ctx = array_create_ctx();
  if (ctx == NULL)
    return 1;

  printf("Demonstrating Numc array core:\n");
  NumcArray *array = array_create(ctx, shape, dim, NUMC_DTYPE_INT32);
  if (array == NULL)
    return 1;
  printf("array_write_data:\n");
  array_write_data(array, data);
  array_print(array);
  printf("is array contiguous: %d\n", is_contiguous(array));

  printf("array_zeros:\n");
  NumcArray *zeros = array_zeros(ctx, shape, dim, NUMC_DTYPE_INT32);
  if (zeros == NULL)
    return 1;
  array_print(zeros);
  printf("is array contiguous: %d\n", is_contiguous(zeros));

  printf("array_fill_with:\n");
  int32_t value = 1;
  NumcArray *ones = array_fill_with(ctx, shape, dim, NUMC_DTYPE_INT32, &value);
  if (zeros == NULL)
    return 1;
  array_print(ones);
  printf("is array contiguous: %d\n", is_contiguous(ones));

  printf("array_copy:\n");
  NumcArray *copy = array_copy(array);
  if (copy == NULL)
    return 1;
  array_print(copy);
  printf("is array contiguous: %d\n", is_contiguous(copy));

  printf("array_reshape_inplace:\n");
  size_t new_shape[] = {2, 12};
  size_t new_dim = 2;
  array_reshape_inplace(array, new_shape, new_dim);
  array_print(array);
  printf("is array contiguous: %d\n", is_contiguous(array));

  printf("array_reshape_copy:\n");
  NumcArray *reshaped = array_reshape_copy(array, shape, dim);
  if (reshaped == NULL)
    return 1;
  array_print(reshaped);
  printf("is array contiguous: %d\n", is_contiguous(reshaped));

  printf("array_transpose_inplace:\n");
  size_t axes[] = {1, 0};
  if (array_transpose_inplace(array, axes) == 0) {
    array_print(array);
  }
  printf("is array contiguous: %d\n", is_contiguous(array));

  size_t axes2[] = {0, 1};
  printf("array_transpose_copy:\n");
  NumcArray *transposed = array_transpose_copy(array, axes2);
  if (transposed == NULL)
    return 1;
  array_print(transposed);
  printf("is array contiguous: %d\n", is_contiguous(transposed));

  printf("array_slice:\n");
  NumcArray *slice =
      array_slice(array, .axis = 1, .start = 1, .stop = 3, .step = 2);
  if (slice == NULL)
    return 1;
  array_print(slice);
  printf("is array contiguous: %d\n", is_contiguous(slice));
  array_free(ctx);

  return 0;
}
