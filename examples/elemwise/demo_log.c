#include "../helpers.h"

static void demo_log(NumcCtx *ctx) {
  section("Log");

  /* float32 — bit-manipulation kernel, powers of 2 are exact */
  label("float32: log([1, 2, 4, 8])");
  size_t shape1[] = {4};
  NumcArray *f32 = numc_array_create(ctx, shape1, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *f32_out = numc_array_zeros(ctx, shape1, 1, NUMC_DTYPE_FLOAT32);
  float df32[] = {1.0f, 2.0f, 4.0f, 8.0f};
  numc_array_write(f32, df32);
  printf("in:  ");
  numc_array_print(f32);
  numc_log(f32, f32_out);
  printf("out: ");
  numc_array_print(f32_out);

  /* float64 — same bit-manipulation approach, double precision */
  label("float64: log([1, 2, 4, 8])");
  size_t shape2[] = {4};
  NumcArray *f64 = numc_array_create(ctx, shape2, 1, NUMC_DTYPE_FLOAT64);
  NumcArray *f64_out = numc_array_zeros(ctx, shape2, 1, NUMC_DTYPE_FLOAT64);
  double df64[] = {1.0, 2.0, 4.0, 8.0};
  numc_array_write(f64, df64);
  printf("in:  ");
  numc_array_print(f64);
  numc_log(f64, f64_out);
  printf("out: ");
  numc_array_print(f64_out);

  /* int8 — cast through float32 log, result truncated to int8 */
  label("int8: log([1, 2, 4, 8]) — cast through float, truncated");
  size_t shape3[] = {4};
  NumcArray *i8 = numc_array_create(ctx, shape3, 1, NUMC_DTYPE_INT8);
  NumcArray *i8_out = numc_array_zeros(ctx, shape3, 1, NUMC_DTYPE_INT8);
  int8_t di8[] = {1, 2, 4, 8};
  numc_array_write(i8, di8);
  printf("in:  ");
  numc_array_print(i8);
  numc_log(i8, i8_out);
  printf("out: ");
  numc_array_print(i8_out);

  /* int32 — cast through float64 log, result truncated to int32 */
  label("int32: log([1, 4, 1024]) — cast through double, truncated");
  size_t shape4[] = {3};
  NumcArray *i32 = numc_array_create(ctx, shape4, 1, NUMC_DTYPE_INT32);
  NumcArray *i32_out = numc_array_zeros(ctx, shape4, 1, NUMC_DTYPE_INT32);
  int32_t di32[] = {1, 4, 1024};
  numc_array_write(i32, di32);
  printf("in:  ");
  numc_array_print(i32);
  numc_log(i32, i32_out);
  printf("out: ");
  numc_array_print(i32_out);

  /* inplace variant */
  label("numc_log_inplace (float32, mutates in place)");
  size_t shape5[] = {4};
  NumcArray *ip = numc_array_create(ctx, shape5, 1, NUMC_DTYPE_FLOAT32);
  float dip[] = {1.0f, 2.0f, 4.0f, 8.0f};
  numc_array_write(ip, dip);
  printf("before: ");
  numc_array_print(ip);
  numc_log_inplace(ip);
  printf("after:  ");
  numc_array_print(ip);
}

int main(void) {
  NumcCtx *ctx = numc_ctx_create();
  demo_log(ctx);
  numc_ctx_free(ctx);
  return 0;
}
