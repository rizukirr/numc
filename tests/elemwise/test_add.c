#include "../helpers.h"

static int test_add_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float db[] = {10.0f, 20.0f, 30.0f, 40.0f};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_add(a, b, out);
  ASSERT_MSG(err == 0, "add should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 11.0f && r[1] == 22.0f && r[2] == 33.0f && r[3] == 44.0f,
             "add results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_add_int32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);

  int32_t da[] = {1, 2, 3, 4};
  int32_t db[] = {100, 200, 300, 400};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_add(a, b, out);
  ASSERT_MSG(err == 0, "add int32 should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 101 && r[1] == 202 && r[2] == 303 && r[3] == 404,
             "add int32 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_add_int8(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT8);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT8);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT8);

  int8_t da[] = {1, 2, 3, 4};
  int8_t db[] = {10, 20, 30, 40};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_add(a, b, out);
  ASSERT_MSG(err == 0, "add int8 should succeed");

  int8_t *r = (int8_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 11 && r[1] == 22 && r[2] == 33 && r[3] == 44,
             "add int8 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_add_float64(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT64);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT64);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT64);

  double da[] = {1.5, 2.5, 3.5};
  double db[] = {0.5, 0.5, 0.5};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_add(a, b, out);
  ASSERT_MSG(err == 0, "add float64 should succeed");

  double *r = (double *)numc_array_data(out);
  ASSERT_MSG(r[0] == 2.0 && r[1] == 3.0 && r[2] == 4.0,
             "add float64 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_binary_op_2d(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_INT32);

  int32_t da[] = {1, 2, 3, 4, 5, 6};
  int32_t db[] = {10, 10, 10, 10, 10, 10};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_add(a, b, out);
  ASSERT_MSG(err == 0, "2D add should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 11 && r[5] == 16, "2D add results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_binary_op_strided(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);

  int32_t da[] = {1, 2, 3, 4, 5, 6};
  int32_t db[] = {10, 20, 30, 40, 50, 60};
  numc_array_write(a, da);
  numc_array_write(b, db);

  // Transpose both to make them non-contiguous
  size_t axes[] = {1, 0};
  numc_array_transpose(a, axes);
  numc_array_transpose(b, axes);

  // out shape matches transposed: {3, 2}
  size_t out_shape[] = {3, 2};
  NumcArray *out = numc_array_zeros(ctx, out_shape, 2, NUMC_DTYPE_INT32);

  int err = numc_add(a, b, out);
  ASSERT_MSG(err == 0, "strided add should succeed");

  // Transposed a: [[1,4],[2,5],[3,6]], b: [[10,40],[20,50],[30,60]]
  // sum: [[11,44],[22,55],[33,66]]
  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 11 && r[1] == 44, "strided row 0");
  ASSERT_MSG(r[2] == 22 && r[3] == 55, "strided row 1");
  ASSERT_MSG(r[4] == 33 && r[5] == 66, "strided row 2");

  numc_ctx_free(ctx);
  return 0;
}

static int test_binary_op_null(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  ASSERT_MSG(numc_add(NULL, a, out) != 0, "add with NULL a should fail");
  ASSERT_MSG(numc_add(a, NULL, out) != 0, "add with NULL b should fail");
  ASSERT_MSG(numc_add(a, a, NULL) != 0, "add with NULL out should fail");
  ASSERT_MSG(numc_sub(NULL, a, out) != 0, "sub with NULL should fail");
  ASSERT_MSG(numc_mul(NULL, a, out) != 0, "mul with NULL should fail");
  ASSERT_MSG(numc_div(NULL, a, out) != 0, "div with NULL should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_binary_op_type_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  ASSERT_MSG(numc_add(a, b, out) != 0, "add with dtype mismatch should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_binary_op_shape_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape_a[] = {4};
  size_t shape_b[] = {5};
  NumcArray *a = numc_array_zeros(ctx, shape_a, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_zeros(ctx, shape_b, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape_a, 1, NUMC_DTYPE_FLOAT32);

  ASSERT_MSG(numc_add(a, b, out) != 0, "add with shape mismatch should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_binary_op_dim_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape_1d[] = {6};
  size_t shape_2d[] = {2, 3};
  NumcArray *a = numc_array_zeros(ctx, shape_1d, 1, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_zeros(ctx, shape_2d, 2, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape_1d, 1, NUMC_DTYPE_INT32);

  ASSERT_MSG(numc_add(a, b, out) != 0, "add with dim mismatch should fail");

  numc_ctx_free(ctx);
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== elemwise/test_add ===\n\n");

  printf("numc_add:\n");
  RUN_TEST(test_add_float32);
  RUN_TEST(test_add_int32);
  RUN_TEST(test_add_int8);
  RUN_TEST(test_add_float64);
  RUN_TEST(test_binary_op_2d);
  RUN_TEST(test_binary_op_strided);

  printf("\nBinary op error cases:\n");
  RUN_TEST(test_binary_op_null);
  RUN_TEST(test_binary_op_type_mismatch);
  RUN_TEST(test_binary_op_shape_mismatch);
  RUN_TEST(test_binary_op_dim_mismatch);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
