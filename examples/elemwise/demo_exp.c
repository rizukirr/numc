#include "../helpers.h"

static void demo_exp(NumcCtx *ctx) {
  section("Exp");

  /* float32 — Cephes polynomial, < 1 ULP error */
  label("float32: exp([0, 1, 2, 3])");
  size_t shape1[] = {4};
  NumcArray *f32 = numc_array_create(ctx, shape1, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *f32_out = numc_array_zeros(ctx, shape1, 1, NUMC_DTYPE_FLOAT32);
  float df32[] = {0.0f, 1.0f, 2.0f, 3.0f};
  numc_array_write(f32, df32);
  printf("in:  ");
  numc_array_print(f32);
  numc_exp(f32, f32_out);
  printf("out: ");
  numc_array_print(f32_out);

  /* float64 — 11-term Taylor polynomial, < 0.23 × 2⁻⁵³ error */
  label("float64: exp([0, 1, 2, 3])");
  size_t shape2[] = {4};
  NumcArray *f64 = numc_array_create(ctx, shape2, 1, NUMC_DTYPE_FLOAT64);
  NumcArray *f64_out = numc_array_zeros(ctx, shape2, 1, NUMC_DTYPE_FLOAT64);
  double df64[] = {0.0, 1.0, 2.0, 3.0};
  numc_array_write(f64, df64);
  printf("in:  ");
  numc_array_print(f64);
  numc_exp(f64, f64_out);
  printf("out: ");
  numc_array_print(f64_out);

  /* overflow / underflow clamping */
  label(
      "float32: overflow (exp(89.0) -> +inf) and underflow (exp(-104.0) -> 0)");
  size_t shape3[] = {2};
  NumcArray *edge = numc_array_create(ctx, shape3, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *edge_out = numc_array_zeros(ctx, shape3, 1, NUMC_DTYPE_FLOAT32);
  float dedge[] = {89.0f, -104.0f};
  numc_array_write(edge, dedge);
  printf("in:  ");
  numc_array_print(edge);
  numc_exp(edge, edge_out);
  printf("out: ");
  numc_array_print(edge_out);

  /* int8 — cast through float32 exp, result truncated to int8 */
  label("int8: exp([0, 1, 2, 3]) — cast through float, truncated");
  size_t shape4[] = {4};
  NumcArray *i8 = numc_array_create(ctx, shape4, 1, NUMC_DTYPE_INT8);
  NumcArray *i8_out = numc_array_zeros(ctx, shape4, 1, NUMC_DTYPE_INT8);
  int8_t di8[] = {0, 1, 2, 3};
  numc_array_write(i8, di8);
  printf("in:  ");
  numc_array_print(i8);
  numc_exp(i8, i8_out);
  printf("out: ");
  numc_array_print(i8_out);

  /* int32 — cast through float64 exp */
  label("int32: exp([0, 1, 10]) — cast through double, truncated");
  size_t shape5[] = {3};
  NumcArray *i32 = numc_array_create(ctx, shape5, 1, NUMC_DTYPE_INT32);
  NumcArray *i32_out = numc_array_zeros(ctx, shape5, 1, NUMC_DTYPE_INT32);
  int32_t di32[] = {0, 1, 10};
  numc_array_write(i32, di32);
  printf("in:  ");
  numc_array_print(i32);
  numc_exp(i32, i32_out);
  printf("out: ");
  numc_array_print(i32_out);

  /* inplace variant */
  label("numc_exp_inplace (float32, mutates in place)");
  size_t shape6[] = {4};
  NumcArray *ip = numc_array_create(ctx, shape6, 1, NUMC_DTYPE_FLOAT32);
  float dip[] = {0.0f, 1.0f, 2.0f, 3.0f};
  numc_array_write(ip, dip);
  printf("before: ");
  numc_array_print(ip);
  numc_exp_inplace(ip);
  printf("after:  ");
  numc_array_print(ip);
}

int main(void) {
  NumcCtx *ctx = numc_ctx_create();
  demo_exp(ctx);
  numc_ctx_free(ctx);
  return 0;
}
