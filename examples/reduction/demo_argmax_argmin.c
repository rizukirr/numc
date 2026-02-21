#include "../helpers.h"

static void demo_argmax_argmin(NumcCtx *ctx) {
  section("Argmax / Argmin");

  /* Full argmax/argmin of a 2x3 array */
  label("numc_argmax / numc_argmin (full reduction, 2x3 float32)");
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  float da[] = {3.0f, 1.0f, 5.0f, 4.0f, 6.0f, 2.0f};
  numc_array_write(a, da);
  printf("a:\n");
  numc_array_print(a);

  size_t sshape[] = {1};
  NumcArray *smax = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_INT64);
  NumcArray *smin = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_INT64);
  numc_argmax(a, smax);
  numc_argmin(a, smin);
  printf("argmax(a) = ");
  numc_array_print(smax);
  printf("argmin(a) = ");
  numc_array_print(smin);

  /* argmax/argmin axis=0: reduce rows -> (3,) */
  label("numc_argmax_axis / numc_argmin_axis (axis=0, 2x3 -> 3)");
  size_t oshape0[] = {3};
  NumcArray *amax0 = numc_array_zeros(ctx, oshape0, 1, NUMC_DTYPE_INT64);
  NumcArray *amin0 = numc_array_zeros(ctx, oshape0, 1, NUMC_DTYPE_INT64);
  numc_argmax_axis(a, 0, 0, amax0);
  numc_argmin_axis(a, 0, 0, amin0);
  printf("argmax(a, axis=0): ");
  numc_array_print(amax0);
  printf("argmin(a, axis=0): ");
  numc_array_print(amin0);

  /* argmax/argmin axis=1: reduce cols -> (2,) */
  label("numc_argmax_axis / numc_argmin_axis (axis=1, 2x3 -> 2)");
  size_t oshape1[] = {2};
  NumcArray *amax1 = numc_array_zeros(ctx, oshape1, 1, NUMC_DTYPE_INT64);
  NumcArray *amin1 = numc_array_zeros(ctx, oshape1, 1, NUMC_DTYPE_INT64);
  numc_argmax_axis(a, 1, 0, amax1);
  numc_argmin_axis(a, 1, 0, amin1);
  printf("argmax(a, axis=1): ");
  numc_array_print(amax1);
  printf("argmin(a, axis=1): ");
  numc_array_print(amin1);

  /* argmax axis=0 keepdim */
  label("numc_argmax_axis (axis=0, keepdim=1, 2x3 -> 1x3)");
  size_t oshape_kd[] = {1, 3};
  NumcArray *amax_kd = numc_array_zeros(ctx, oshape_kd, 2, NUMC_DTYPE_INT64);
  numc_argmax_axis(a, 0, 1, amax_kd);
  numc_array_print(amax_kd);
}

int main(void) {
  NumcCtx *ctx = numc_ctx_create();
  demo_argmax_argmin(ctx);
  numc_ctx_free(ctx);
  return 0;
}
