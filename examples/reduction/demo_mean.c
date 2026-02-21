#include "../helpers.h"

static void demo_mean(NumcCtx *ctx) {
  section("Mean");

  /* Full mean of a 2x3 array */
  label("numc_mean (full reduction, 2x3 float32)");
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  float da[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  numc_array_write(a, da);
  printf("a:\n");
  numc_array_print(a);

  size_t sshape[] = {1};
  NumcArray *scalar = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_FLOAT32);
  numc_mean(a, scalar);
  printf("mean(a) = ");
  numc_array_print(scalar);

  /* mean axis=0: reduce rows -> (3,) */
  label("numc_mean_axis (axis=0, 2x3 -> 3)");
  size_t oshape0[] = {3};
  NumcArray *out0 = numc_array_zeros(ctx, oshape0, 1, NUMC_DTYPE_FLOAT32);
  numc_mean_axis(a, 0, 0, out0);
  numc_array_print(out0);

  /* mean axis=1: reduce cols -> (2,) */
  label("numc_mean_axis (axis=1, 2x3 -> 2)");
  size_t oshape1[] = {2};
  NumcArray *out1 = numc_array_zeros(ctx, oshape1, 1, NUMC_DTYPE_FLOAT32);
  numc_mean_axis(a, 1, 0, out1);
  numc_array_print(out1);

  /* mean axis=0 keepdim: reduce rows -> (1,3) */
  label("numc_mean_axis (axis=0, keepdim=1, 2x3 -> 1x3)");
  size_t oshape_kd[] = {1, 3};
  NumcArray *out_kd = numc_array_zeros(ctx, oshape_kd, 2, NUMC_DTYPE_FLOAT32);
  numc_mean_axis(a, 0, 1, out_kd);
  numc_array_print(out_kd);

  /* Integer truncation */
  label("numc_mean (int32 truncation, [1..6] / 6 = 3)");
  NumcArray *ai = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  int32_t di[] = {1, 2, 3, 4, 5, 6};
  numc_array_write(ai, di);
  printf("a:\n");
  numc_array_print(ai);

  NumcArray *si = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_INT32);
  numc_mean(ai, si);
  printf("mean(a) = ");
  numc_array_print(si);
}

int main(void) {
  NumcCtx *ctx = numc_ctx_create();
  demo_mean(ctx);
  numc_ctx_free(ctx);
  return 0;
}
