#include "../helpers.h"

static void demo_shape(NumcCtx *ctx) {
  section("Shape Manipulation");

  /* Setup: 2x3 array [1..6] */
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  int32_t data[] = {1, 2, 3, 4, 5, 6};
  numc_array_write(a, data);

  label("original (2x3)");
  numc_array_print(a);

  /* numc_array_reshape — in-place */
  label("numc_array_reshape (3x2, in-place)");
  size_t new_shape[] = {3, 2};
  numc_array_reshape(a, new_shape, 2);
  numc_array_print(a);

  /* numc_array_reshape_copy — returns new array */
  label("numc_array_reshape_copy (6x1, new array)");
  size_t flat_shape[] = {6, 1};
  NumcArray *flat = numc_array_reshape_copy(a, flat_shape, 2);
  numc_array_print(flat);

  /* numc_array_transpose — in-place */
  label("numc_array_transpose (3x2 -> 2x3, in-place)");
  size_t axes[] = {1, 0};
  numc_array_transpose(a, axes);
  printf("contiguous after transpose: %s\n",
         numc_array_is_contiguous(a) ? "true" : "false");
  numc_array_print(a);

  /* numc_array_contiguous — make contiguous again */
  label("numc_array_contiguous (re-layout memory)");
  numc_array_contiguous(a);
  printf("contiguous after fix: %s\n",
         numc_array_is_contiguous(a) ? "true" : "false");
  numc_array_print(a);

  /* numc_array_transpose_copy — returns new array */
  label("numc_array_transpose_copy (2x3 -> 3x2, new array)");
  NumcArray *t = numc_array_transpose_copy(a, axes);
  printf("contiguous: %s\n", numc_array_is_contiguous(t) ? "true" : "false");
  numc_array_print(t);

  /* numc_array_slice — view, no data copy */
  label("numc_slice (row 1 of 2x3 = 3 elements)");
  size_t shape2[] = {2, 3};
  NumcArray *b = numc_array_create(ctx, shape2, 2, NUMC_DTYPE_INT32);
  int32_t data2[] = {10, 20, 30, 40, 50, 60};
  numc_array_write(b, data2);
  printf("original:\n");
  numc_array_print(b);

  NumcArray *row = numc_slice(b, .axis = 0, .start = 1, .stop = 2, .step = 1);
  printf("slice [1:2, :] :\n");
  numc_array_print(row);
}

int main(void) {
  NumcCtx *ctx = numc_ctx_create();
  demo_shape(ctx);
  numc_ctx_free(ctx);
  return 0;
}
