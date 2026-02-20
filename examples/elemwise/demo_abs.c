#include "../helpers.h"

static void demo_abs(NumcCtx *ctx) {
  section("Abs");

  /* abs only applies to signed types (int8/16/32/64, float32/64).
     unsigned types have no negative values, so abs is not needed. */

  /* abs only applies to signed types (int8/16/32/64, float32/64).
     unsigned types have no negative values, so abs is not needed. */

  /* --- signed integers --- */
  label("int8");
  size_t shape1[] = {6};
  NumcArray *i8 = numc_array_create(ctx, shape1, 1, NUMC_DTYPE_INT8);
  NumcArray *i8_out = numc_array_zeros(ctx, shape1, 1, NUMC_DTYPE_INT8);
  int8_t di8[] = {-5, -4, -3, 0, 3, 5};
  numc_array_write(i8, di8);
  printf("in:  ");
  numc_array_print(i8);
  numc_abs(i8, i8_out);
  printf("out: ");
  numc_array_print(i8_out);

  /* INT8_MIN (-128) has no positive counterpart in int8 — overflows back to
   * -128 */
  label("int8: INT8_MIN edge case (abs(-128) wraps to -128)");
  size_t shape_edge[] = {1};
  NumcArray *edge = numc_array_create(ctx, shape_edge, 1, NUMC_DTYPE_INT8);
  NumcArray *edge_out = numc_array_zeros(ctx, shape_edge, 1, NUMC_DTYPE_INT8);
  int8_t d_edge[] = {-128};
  numc_array_write(edge, d_edge);
  printf("in:  ");
  numc_array_print(edge);
  numc_abs(edge, edge_out);
  printf("out: ");
  numc_array_print(edge_out); /* still -128 */

  label("int32");
  size_t shape2[] = {2, 3};
  NumcArray *i32 = numc_array_create(ctx, shape2, 2, NUMC_DTYPE_INT32);
  NumcArray *i32_out = numc_array_zeros(ctx, shape2, 2, NUMC_DTYPE_INT32);
  int32_t di32[] = {-10, -20, -30, 10, 20, 30};
  numc_array_write(i32, di32);
  printf("in:\n");
  numc_array_print(i32);
  numc_abs(i32, i32_out);
  printf("out:\n");
  numc_array_print(i32_out);

  /* --- floats: clears IEEE-754 sign bit, no overflow possible --- */
  label("float32");
  size_t shape3[] = {2, 3};
  NumcArray *f32 = numc_array_create(ctx, shape3, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *f32_out = numc_array_zeros(ctx, shape3, 2, NUMC_DTYPE_FLOAT32);
  float df32[] = {-1.5f, -2.5f, -3.5f, 1.5f, 2.5f, 3.5f};
  numc_array_write(f32, df32);
  printf("in:\n");
  numc_array_print(f32);
  numc_abs(f32, f32_out);
  printf("out:\n");
  numc_array_print(f32_out);

  /* --- inplace variant --- */
  label("numc_abs_inplace (float32, mutates in place)");
  size_t shape4[] = {4};
  NumcArray *ip = numc_array_create(ctx, shape4, 1, NUMC_DTYPE_FLOAT32);
  float dip[] = {-1.0f, -2.0f, 3.0f, -4.0f};
  numc_array_write(ip, dip);
  printf("before: ");
  numc_array_print(ip);
  numc_abs_inplace(ip);
  printf("after:  ");
  numc_array_print(ip);
}

int main(void) {
  NumcCtx *ctx = numc_ctx_create();
  demo_abs(ctx);
  numc_ctx_free(ctx);
  return 0;
}
