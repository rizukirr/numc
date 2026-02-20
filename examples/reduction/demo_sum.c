#include "../helpers.h"

static void demo_sum(NumcCtx *ctx) {
  section("Sum");

  /* Full sum of a 2x3 array */
  label("numc_sum (full reduction, 2x3 float32)");
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  float da[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  numc_array_write(a, da);
  printf("a:\n");
  numc_array_print(a);

  size_t sshape[] = {1};
  NumcArray *scalar = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_FLOAT32);
  numc_sum(a, scalar);
  printf("sum(a) = ");
  numc_array_print(scalar);

  /* sum axis=0: reduce rows → (3,) */
  label("numc_sum_axis (axis=0, 2x3 -> 3)");
  size_t oshape0[] = {3};
  NumcArray *out0 = numc_array_zeros(ctx, oshape0, 1, NUMC_DTYPE_FLOAT32);
  numc_sum_axis(a, 0, 0, out0);
  numc_array_print(out0);

  /* sum axis=1: reduce cols → (2,) */
  label("numc_sum_axis (axis=1, 2x3 -> 2)");
  size_t oshape1[] = {2};
  NumcArray *out1 = numc_array_zeros(ctx, oshape1, 1, NUMC_DTYPE_FLOAT32);
  numc_sum_axis(a, 1, 0, out1);
  numc_array_print(out1);

  /* sum axis=0 keepdim: reduce rows → (1,3) */
  label("numc_sum_axis (axis=0, keepdim=1, 2x3 -> 1x3)");
  size_t oshape_kd[] = {1, 3};
  NumcArray *out_kd = numc_array_zeros(ctx, oshape_kd, 2, NUMC_DTYPE_FLOAT32);
  numc_sum_axis(a, 0, 1, out_kd);
  numc_array_print(out_kd);

  /* 3D array sum along middle axis */
  label("numc_sum_axis (3D int32, axis=1, 2x3x4 -> 2x4)");
  size_t shape3d[] = {2, 3, 4};
  NumcArray *b = numc_array_create(ctx, shape3d, 3, NUMC_DTYPE_INT32);
  int32_t db[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                  13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
  numc_array_write(b, db);
  printf("b:\n");
  numc_array_print(b);

  size_t oshape3d[] = {2, 4};
  NumcArray *out3d = numc_array_zeros(ctx, oshape3d, 2, NUMC_DTYPE_INT32);
  numc_sum_axis(b, 1, 0, out3d);
  printf("sum(b, axis=1):\n");
  numc_array_print(out3d);

  /* Transposed (non-contiguous) full sum */
  label("numc_sum (transposed 2x3 -> non-contiguous)");
  NumcArray *t = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  numc_array_write(t, da);
  size_t axes[] = {1, 0};
  numc_array_transpose(t, axes);
  printf("transposed (3x2, non-contiguous):\n");
  numc_array_print(t);

  NumcArray *scalar2 = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_FLOAT32);
  numc_sum(t, scalar2);
  printf("sum = ");
  numc_array_print(scalar2);
}

int main(void) {
  NumcCtx *ctx = numc_ctx_create();
  demo_sum(ctx);
  numc_ctx_free(ctx);
  return 0;
}
