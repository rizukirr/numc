#include "array/array_core.h"
#include "array/array_dtype.h"
#include "array/array_print.h"
#include "math/math.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

int main(void) {
  size_t shape[] = {2, 2, 4};
  size_t dim = 3;
  int32_t data[][2][4] = {{{1, 2, 3, 4}, {5, 6, 7, 8}},
                          {{9, 10, 11, 12}, {13, 14, 15, 16}}};

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

  printf("Array element-wise operations:\n");
  NumcArray *add = array_create(ctx, shape, dim, NUMC_DTYPE_INT32);
  int err = array_add(array, array, add);
  if (err < 0)
    return 1;
  array_print(add);

  array_free(ctx);
}
