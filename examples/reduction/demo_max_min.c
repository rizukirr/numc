#include "../helpers.h"

static void demo_max_min(NumcCtx *ctx) {
  section("Max / Min");

  /* Full max/min of a 2x3 array */
  label("numc_max / numc_min (full reduction, 2x3 float32)");
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  float da[] = {3.0f, 1.0f, 5.0f, 4.0f, 6.0f, 2.0f};
  numc_array_write(a, da);
  printf("a:\n");
  numc_array_print(a);

  size_t sshape[] = {1};
  NumcArray *smax = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *smin = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_FLOAT32);
  numc_max(a, smax);
  numc_min(a, smin);
  printf("max(a) = ");
  numc_array_print(smax);
  printf("min(a) = ");
  numc_array_print(smin);

  /* max/min axis=0: reduce rows -> (3,) */
  label("numc_max_axis / numc_min_axis (axis=0, 2x3 -> 3)");
  size_t oshape0[] = {3};
  NumcArray *max0 = numc_array_zeros(ctx, oshape0, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *min0 = numc_array_zeros(ctx, oshape0, 1, NUMC_DTYPE_FLOAT32);
  numc_max_axis(a, 0, 0, max0);
  numc_min_axis(a, 0, 0, min0);
  printf("max(a, axis=0): ");
  numc_array_print(max0);
  printf("min(a, axis=0): ");
  numc_array_print(min0);

  /* max/min axis=1: reduce cols -> (2,) */
  label("numc_max_axis / numc_min_axis (axis=1, 2x3 -> 2)");
  size_t oshape1[] = {2};
  NumcArray *max1 = numc_array_zeros(ctx, oshape1, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *min1 = numc_array_zeros(ctx, oshape1, 1, NUMC_DTYPE_FLOAT32);
  numc_max_axis(a, 1, 0, max1);
  numc_min_axis(a, 1, 0, min1);
  printf("max(a, axis=1): ");
  numc_array_print(max1);
  printf("min(a, axis=1): ");
  numc_array_print(min1);

  /* max axis=0 keepdim */
  label("numc_max_axis (axis=0, keepdim=1, 2x3 -> 1x3)");
  size_t oshape_kd[] = {1, 3};
  NumcArray *max_kd = numc_array_zeros(ctx, oshape_kd, 2, NUMC_DTYPE_FLOAT32);
  numc_max_axis(a, 0, 1, max_kd);
  numc_array_print(max_kd);

  /* 3D int32 */
  label("numc_max_axis (3D int32, axis=1, 2x3x4 -> 2x4)");
  size_t shape3d[] = {2, 3, 4};
  NumcArray *b = numc_array_create(ctx, shape3d, 3, NUMC_DTYPE_INT32);
  int32_t db[] = {12, 1,  8,  3,  5,  14, 7,  2,  9,  6,  11, 4,
                  16, 13, 20, 15, 21, 18, 19, 24, 17, 22, 23, 10};
  numc_array_write(b, db);
  printf("b:\n");
  numc_array_print(b);

  size_t oshape3d[] = {2, 4};
  NumcArray *max3d = numc_array_zeros(ctx, oshape3d, 2, NUMC_DTYPE_INT32);
  numc_max_axis(b, 1, 0, max3d);
  printf("max(b, axis=1):\n");
  numc_array_print(max3d);

  /* Transposed (non-contiguous) full max */
  label("numc_max (transposed 2x3 -> non-contiguous)");
  NumcArray *t = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  numc_array_write(t, da);
  size_t axes[] = {1, 0};
  numc_array_transpose(t, axes);
  printf("transposed (3x2, non-contiguous):\n");
  numc_array_print(t);

  NumcArray *smax2 = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_FLOAT32);
  numc_max(t, smax2);
  printf("max = ");
  numc_array_print(smax2);
}

int main(void) {
  NumcCtx *ctx = numc_ctx_create();
  demo_max_min(ctx);
  numc_ctx_free(ctx);
  return 0;
}
