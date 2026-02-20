#include "../helpers.h"

static void demo_clip(NumcCtx *ctx) {
  section("Clip");

  /* float32 — clamp values to [2.0, 5.0] */
  label("float32: clip([1, 2, 3, 4, 5, 6], min=2, max=5)");
  size_t shape1[] = {6};
  NumcArray *f32 = numc_array_create(ctx, shape1, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *f32_out = numc_array_zeros(ctx, shape1, 1, NUMC_DTYPE_FLOAT32);
  float df32[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  numc_array_write(f32, df32);
  printf("in:  ");
  numc_array_print(f32);
  numc_clip(f32, f32_out, 2.0, 5.0);
  printf("out: ");
  numc_array_print(f32_out);

  /* int32 — 2D array */
  label("int32: clip 2x3, min=-10, max=10");
  size_t shape2[] = {2, 3};
  NumcArray *i32 = numc_array_create(ctx, shape2, 2, NUMC_DTYPE_INT32);
  NumcArray *i32_out = numc_array_zeros(ctx, shape2, 2, NUMC_DTYPE_INT32);
  int32_t di32[] = {-50, -5, 0, 5, 50, 100};
  numc_array_write(i32, di32);
  printf("in:\n");
  numc_array_print(i32);
  numc_clip(i32, i32_out, -10.0, 10.0);
  printf("out:\n");
  numc_array_print(i32_out);

  /* inplace variant */
  label("numc_clip_inplace (float32, clamp to [0, 3])");
  size_t shape3[] = {4};
  NumcArray *ip = numc_array_create(ctx, shape3, 1, NUMC_DTYPE_FLOAT32);
  float dip[] = {-1.0f, 1.5f, 3.5f, 10.0f};
  numc_array_write(ip, dip);
  printf("before: ");
  numc_array_print(ip);
  numc_clip_inplace(ip, 0.0, 3.0);
  printf("after:  ");
  numc_array_print(ip);
}

int main(void) {
  NumcCtx *ctx = numc_ctx_create();
  demo_clip(ctx);
  numc_ctx_free(ctx);
  return 0;
}
