#include <numc/numc.h>
#include <stdio.h>

/* ── Helpers ───────────────────────────────────────────────────────── */

static void section(const char *title) {
  printf("\n══════════════════════════════════════════\n");
  printf("  %s\n", title);
  printf("══════════════════════════════════════════\n\n");
}

static void label(const char *name) { printf("--- %s ---\n", name); }

/* ── Array Creation ────────────────────────────────────────────────── */

static void demo_array_creation(NumcCtx *ctx) {
  section("Array Creation");

  /* numc_array_create — uninitialized */
  label("numc_array_create (2x3 float32, then write data)");
  size_t shape1[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape1, 2, NUMC_DTYPE_FLOAT32);
  float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  numc_array_write(a, data);
  numc_array_print(a);

  /* numc_array_zeros */
  label("numc_array_zeros (3x3 int32)");
  size_t shape2[] = {3, 3};
  NumcArray *z = numc_array_zeros(ctx, shape2, 2, NUMC_DTYPE_INT32);
  numc_array_print(z);

  /* numc_array_fill */
  label("numc_array_fill (2x4 float64, filled with 3.14)");
  size_t shape3[] = {2, 4};
  double fill_val = 3.14;
  NumcArray *f = numc_array_fill(ctx, shape3, 2, NUMC_DTYPE_FLOAT64, &fill_val);
  numc_array_print(f);

  /* numc_array_copy */
  label("numc_array_copy (deep copy of the float32 array)");
  NumcArray *c = numc_array_copy(a);
  numc_array_print(c);

  /* numc_array_write — multi-dimensional */
  label("numc_array_write (2x2x4 int32)");
  size_t shape4[] = {2, 2, 4};
  NumcArray *w = numc_array_create(ctx, shape4, 3, NUMC_DTYPE_INT32);
  int32_t data3d[][2][4] = {
      {{1, 2, 3, 4}, {5, 6, 7, 8}},
      {{9, 10, 11, 12}, {13, 14, 15, 16}},
  };
  numc_array_write(w, data3d);
  numc_array_print(w);
}

/* ── Properties ────────────────────────────────────────────────────── */

static void demo_properties(NumcCtx *ctx) {
  section("Properties");

  size_t shape[] = {2, 3, 4};
  float val = 1.0f;
  NumcArray *a = numc_array_fill(ctx, shape, 3, NUMC_DTYPE_FLOAT32, &val);

  printf("ndim:      %zu\n", numc_array_ndim(a));
  printf("size:      %zu\n", numc_array_size(a));
  printf("capacity:  %zu\n", numc_array_capacity(a));
  printf("elem_size: %zu\n", numc_array_elem_size(a));
  printf("dtype:     %d (NUMC_DTYPE_FLOAT32 = %d)\n", numc_array_dtype(a),
         NUMC_DTYPE_FLOAT32);

  size_t ndim = numc_array_ndim(a);
  size_t s[ndim], st[ndim];
  numc_array_shape(a, s);
  numc_array_strides(a, st);

  printf("shape:     [");
  for (size_t i = 0; i < ndim; i++)
    printf("%zu%s", s[i], i + 1 < ndim ? ", " : "");
  printf("]\n");

  printf("strides:   [");
  for (size_t i = 0; i < ndim; i++)
    printf("%zu%s", st[i], i + 1 < ndim ? ", " : "");
  printf("] (bytes)\n");

  printf("data ptr:  %p\n", numc_array_data(a));
  printf("contiguous: %s\n", numc_array_is_contiguous(a) ? "true" : "false");
}

/* ── Shape Manipulation ────────────────────────────────────────────── */

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

/* ── Element-wise Binary Ops ───────────────────────────────────────── */

static void demo_math_binary(NumcCtx *ctx) {
  section("Element-wise Binary Ops");

  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_FLOAT32);

  float da[] = {10, 20, 30, 40, 50, 60};
  float db[] = {1, 2, 3, 4, 5, 6};
  numc_array_write(a, da);
  numc_array_write(b, db);

  printf("a:\n");
  numc_array_print(a);
  printf("b:\n");
  numc_array_print(b);

  label("numc_add (a + b)");
  numc_add(a, b, out);
  numc_array_print(out);

  label("numc_sub (a - b)");
  numc_sub(a, b, out);
  numc_array_print(out);

  label("numc_mul (a * b)");
  numc_mul(a, b, out);
  numc_array_print(out);

  label("numc_div (a / b)");
  numc_div(a, b, out);
  numc_array_print(out);
}

/* ── Element-wise Scalar Ops ───────────────────────────────────────── */

static void demo_math_scalar(NumcCtx *ctx) {
  section("Element-wise Scalar Ops");

  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_FLOAT32);

  float da[] = {10, 20, 30, 40, 50, 60};
  numc_array_write(a, da);

  printf("a:\n");
  numc_array_print(a);

  label("numc_add_scalar (a + 100)");
  numc_add_scalar(a, 100.0, out);
  numc_array_print(out);

  label("numc_sub_scalar (a - 5)");
  numc_sub_scalar(a, 5.0, out);
  numc_array_print(out);

  label("numc_mul_scalar (a * 0.5)");
  numc_mul_scalar(a, 0.5, out);
  numc_array_print(out);

  label("numc_div_scalar (a / 3)");
  numc_div_scalar(a, 3.0, out);
  numc_array_print(out);
}

/* ── Scalar Inplace Ops ────────────────────────────────────────────── */

static void demo_math_scalar_inplace(NumcCtx *ctx) {
  section("Scalar Inplace Ops");

  size_t shape[] = {2, 3};
  float da[] = {10, 20, 30, 40, 50, 60};

  label("numc_add_scalar_inplace (a += 1000)");
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  numc_array_write(a, da);
  numc_add_scalar_inplace(a, 1000.0);
  numc_array_print(a);

  label("numc_sub_scalar_inplace (a -= 5)");
  a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  numc_array_write(a, da);
  numc_sub_scalar_inplace(a, 5.0);
  numc_array_print(a);

  label("numc_mul_scalar_inplace (a *= 2)");
  a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  numc_array_write(a, da);
  numc_mul_scalar_inplace(a, 2.0);
  numc_array_print(a);

  label("numc_div_scalar_inplace (a /= 10)");
  a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  numc_array_write(a, da);
  numc_div_scalar_inplace(a, 10.0);
  numc_array_print(a);
}

/* ── Neg ──────────────────────────────────────────────────────────────── */

static void demo_neg(NumcCtx *ctx) {
  section("Neg");

  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_FLOAT32);

  float da[] = {10, -20, 30, 40, 50, 60};
  numc_array_write(a, da);

  printf("a:\n");
  numc_array_print(a);

  label("numc_neg (a)");
  numc_neg(a, out);
  numc_array_print(out);

  label("numc_neg_inplace (a)");
  numc_neg_inplace(a);
  numc_array_print(a);
}

/* ── Abs ──────────────────────────────────────────────────────────────── */

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

/* ── Log ──────────────────────────────────────────────────────────────── */

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

/* ── Exp ──────────────────────────────────────────────────────────────── */

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

/* ── Clip ─────────────────────────────────────────────────────────── */

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

/* ── Maximum / Minimum ────────────────────────────────────────────── */

static void demo_maximum_minimum(NumcCtx *ctx) {
  section("Maximum / Minimum");

  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_FLOAT32);

  float da[] = {1, 5, 3, 8, 2, 7};
  float db[] = {4, 2, 6, 1, 9, 3};
  numc_array_write(a, da);
  numc_array_write(b, db);

  printf("a:\n");
  numc_array_print(a);
  printf("b:\n");
  numc_array_print(b);

  label("numc_maximum (a, b)");
  numc_maximum(a, b, out);
  numc_array_print(out);

  label("numc_minimum (a, b)");
  numc_minimum(a, b, out);
  numc_array_print(out);

  /* int32 */
  label("int32: maximum and minimum");
  NumcArray *i1 = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  NumcArray *i2 = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  NumcArray *iout = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_INT32);
  int32_t di1[] = {-10, 20, -30, 40, -50, 60};
  int32_t di2[] = {10, -20, 30, -40, 50, -60};
  numc_array_write(i1, di1);
  numc_array_write(i2, di2);
  printf("a:\n");
  numc_array_print(i1);
  printf("b:\n");
  numc_array_print(i2);
  numc_maximum(i1, i2, iout);
  printf("max: ");
  numc_array_print(iout);
  numc_minimum(i1, i2, iout);
  printf("min: ");
  numc_array_print(iout);

  /* Inplace variants */
  label("numc_maximum_inplace (a = max(a, b), int32)");
  NumcArray *ma = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  NumcArray *mb = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  numc_array_write(ma, di1);
  numc_array_write(mb, di2);
  printf("a: ");
  numc_array_print(ma);
  printf("b: ");
  numc_array_print(mb);
  numc_maximum_inplace(ma, mb);
  printf("a: ");
  numc_array_print(ma);

  label("numc_minimum_inplace (a = min(a, b), int32)");
  NumcArray *na = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  NumcArray *nb = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  numc_array_write(na, di1);
  numc_array_write(nb, di2);
  printf("a: ");
  numc_array_print(na);
  printf("b: ");
  numc_array_print(nb);
  numc_minimum_inplace(na, nb);
  printf("a: ");
  numc_array_print(na);
}

/* ── Error Handling ────────────────────────────────────────────────── */

static void demo_error(NumcCtx *ctx) {
  section("Error Handling");

  label("shape mismatch (add 2x3 + 3x2)");
  size_t s1[] = {2, 3}, s2[] = {3, 2};
  NumcArray *a = numc_array_zeros(ctx, s1, 2, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_zeros(ctx, s2, 2, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, s1, 2, NUMC_DTYPE_INT32);
  int err = numc_add(a, b, out);
  printf("numc_add returned: %d (NUMC_ERR_SHAPE = %d)\n", err, NUMC_ERR_SHAPE);

  label("dtype mismatch (add int32 + float32)");
  NumcArray *c = numc_array_zeros(ctx, s1, 2, NUMC_DTYPE_FLOAT32);
  err = numc_add(a, c, out);
  printf("numc_add returned: %d (NUMC_ERR_TYPE = %d)\n", err, NUMC_ERR_TYPE);

  label("null pointer");
  err = numc_add(NULL, b, out);
  printf("numc_add returned: %d (NUMC_ERR_NULL = %d)\n", err, NUMC_ERR_NULL);

  label("numc_set_error / numc_get_error");
  numc_set_error(-99, "custom error message");
  NumcError e = numc_get_error();
  printf("code: %d, msg: \"%s\"\n", e.code, e.msg);
}

/* ── main ──────────────────────────────────────────────────────────── */

int main(void) {
  printf("numc API demo\n");

  NumcCtx *ctx = numc_ctx_create();
  if (!ctx) {
    fprintf(stderr, "Failed to create context\n");
    return 1;
  }

  demo_array_creation(ctx);
  demo_properties(ctx);
  demo_shape(ctx);
  demo_math_binary(ctx);
  demo_math_scalar(ctx);
  demo_math_scalar_inplace(ctx);
  demo_neg(ctx);
  demo_abs(ctx);
  demo_log(ctx);
  demo_exp(ctx);
  demo_clip(ctx);
  demo_maximum_minimum(ctx);
  demo_error(ctx);

  numc_ctx_free(ctx);
  printf("\nnumc_ctx_free — all arrays freed.\n");
  return 0;
}
