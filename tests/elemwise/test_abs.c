#include "../helpers.h"

static int test_abs_int8(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {6};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT8);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT8);

  int8_t da[] = {-5, -4, -3, 0, 3, 5};
  numc_array_write(a, da);

  int err = numc_abs(a, out);
  ASSERT_MSG(err == 0, "abs int8 should succeed");

  int8_t *r = (int8_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 5 && r[1] == 4 && r[2] == 3, "abs int8 negatives");
  ASSERT_MSG(r[3] == 0 && r[4] == 3 && r[5] == 5,
             "abs int8 zero and positives unchanged");

  numc_ctx_free(ctx);
  return 0;
}

static int test_abs_int16(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT16);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT16);

  int16_t da[] = {-1000, -1, 1, 1000};
  numc_array_write(a, da);

  int err = numc_abs(a, out);
  ASSERT_MSG(err == 0, "abs int16 should succeed");

  int16_t *r = (int16_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1000 && r[1] == 1 && r[2] == 1 && r[3] == 1000,
             "abs int16 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_abs_int32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);

  int32_t da[] = {-100, -1, 1, 100};
  numc_array_write(a, da);

  int err = numc_abs(a, out);
  ASSERT_MSG(err == 0, "abs int32 should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 100 && r[1] == 1 && r[2] == 1 && r[3] == 100,
             "abs int32 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_abs_int64(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT64);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT64);

  int64_t da[] = {-1000000000LL, -1, 1, 1000000000LL};
  numc_array_write(a, da);

  int err = numc_abs(a, out);
  ASSERT_MSG(err == 0, "abs int64 should succeed");

  int64_t *r = (int64_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1000000000LL && r[1] == 1 && r[2] == 1 &&
                 r[3] == 1000000000LL,
             "abs int64 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_abs_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {-1.5f, -0.5f, 0.5f, 1.5f};
  numc_array_write(a, da);

  int err = numc_abs(a, out);
  ASSERT_MSG(err == 0, "abs float32 should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1.5f && r[1] == 0.5f && r[2] == 0.5f && r[3] == 1.5f,
             "abs float32 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_abs_float64(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT64);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT64);

  double da[] = {-1.5, -0.5, 0.5, 1.5};
  numc_array_write(a, da);

  int err = numc_abs(a, out);
  ASSERT_MSG(err == 0, "abs float64 should succeed");

  double *r = (double *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1.5 && r[1] == 0.5 && r[2] == 0.5 && r[3] == 1.5,
             "abs float64 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_abs_2d(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_INT32);

  int32_t da[] = {-10, -20, -30, 10, 20, 30};
  numc_array_write(a, da);

  int err = numc_abs(a, out);
  ASSERT_MSG(err == 0, "2D abs should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 10 && r[1] == 20 && r[2] == 30, "abs 2D row 0");
  ASSERT_MSG(r[3] == 10 && r[4] == 20 && r[5] == 30, "abs 2D row 1 unchanged");

  numc_ctx_free(ctx);
  return 0;
}

static int test_abs_strided(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);

  int32_t da[] = {-1, -2, -3, 4, 5, 6};
  numc_array_write(a, da);

  // Transpose to make it non-contiguous
  size_t axes[] = {1, 0};
  numc_array_transpose(a, axes);

  // out shape matches transposed: {3, 2}
  size_t out_shape[] = {3, 2};
  NumcArray *out = numc_array_zeros(ctx, out_shape, 2, NUMC_DTYPE_INT32);

  int err = numc_abs(a, out);
  ASSERT_MSG(err == 0, "strided abs should succeed");

  // Transposed a: [[-1,4],[-2,5],[-3,6]]
  // abs:          [[1,4],[2,5],[3,6]]
  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1 && r[1] == 4, "strided row 0");
  ASSERT_MSG(r[2] == 2 && r[3] == 5, "strided row 1");
  ASSERT_MSG(r[4] == 3 && r[5] == 6, "strided row 2");

  numc_ctx_free(ctx);
  return 0;
}

static int test_abs_zeros(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);

  int err = numc_abs(a, out);
  ASSERT_MSG(err == 0, "abs of zeros should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  for (size_t i = 0; i < 4; i++) {
    ASSERT_MSG(r[i] == 0, "abs of zero should be zero");
  }

  numc_ctx_free(ctx);
  return 0;
}

static int test_abs_all_positive(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);

  int32_t da[] = {1, 2, 3, 4};
  numc_array_write(a, da);

  int err = numc_abs(a, out);
  ASSERT_MSG(err == 0, "abs of positives should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1 && r[1] == 2 && r[2] == 3 && r[3] == 4,
             "abs of positives should be unchanged");

  numc_ctx_free(ctx);
  return 0;
}

static int test_abs_int8_min(void) {
  /* INT8_MIN (-128) has no positive counterpart in int8 —
     abs(-128) wraps back to -128 (two's complement overflow) */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {1};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT8);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT8);

  int8_t da[] = {-128};
  numc_array_write(a, da);

  numc_abs(a, out);

  int8_t *r = (int8_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == -128, "abs(INT8_MIN) wraps to -128");

  numc_ctx_free(ctx);
  return 0;
}

static int test_abs_null(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  ASSERT_MSG(numc_abs(NULL, out) != 0, "abs with NULL a should fail");
  ASSERT_MSG(numc_abs(a, NULL) != 0, "abs with NULL out should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_abs_type_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);

  ASSERT_MSG(numc_abs(a, out) != 0, "abs with dtype mismatch should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_abs_shape_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape_a[] = {4};
  size_t shape_o[] = {5};
  NumcArray *a = numc_array_zeros(ctx, shape_a, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape_o, 1, NUMC_DTYPE_FLOAT32);

  ASSERT_MSG(numc_abs(a, out) != 0, "abs with shape mismatch should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_abs_inplace_int8(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT8);

  int8_t da[] = {-5, -10, 15, -20};
  numc_array_write(a, da);

  int err = numc_abs_inplace(a);
  ASSERT_MSG(err == 0, "abs_inplace int8 should succeed");

  int8_t *r = (int8_t *)numc_array_data(a);
  ASSERT_MSG(r[0] == 5 && r[1] == 10 && r[2] == 15 && r[3] == 20,
             "abs_inplace int8 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_abs_inplace_int32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);

  int32_t da[] = {-100, -1, 1, 100};
  numc_array_write(a, da);

  int err = numc_abs_inplace(a);
  ASSERT_MSG(err == 0, "abs_inplace int32 should succeed");

  int32_t *r = (int32_t *)numc_array_data(a);
  ASSERT_MSG(r[0] == 100 && r[1] == 1 && r[2] == 1 && r[3] == 100,
             "abs_inplace int32 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_abs_inplace_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {-1.5f, -0.5f, 0.5f, 1.5f};
  numc_array_write(a, da);

  int err = numc_abs_inplace(a);
  ASSERT_MSG(err == 0, "abs_inplace float32 should succeed");

  float *r = (float *)numc_array_data(a);
  ASSERT_MSG(r[0] == 1.5f && r[1] == 0.5f && r[2] == 0.5f && r[3] == 1.5f,
             "abs_inplace float32 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_abs_inplace_float64(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT64);

  double da[] = {-1.5, -0.5, 0.5, 1.5};
  numc_array_write(a, da);

  int err = numc_abs_inplace(a);
  ASSERT_MSG(err == 0, "abs_inplace float64 should succeed");

  double *r = (double *)numc_array_data(a);
  ASSERT_MSG(r[0] == 1.5 && r[1] == 0.5 && r[2] == 0.5 && r[3] == 1.5,
             "abs_inplace float64 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_abs_inplace_2d(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);

  int32_t da[] = {-10, -20, -30, 10, 20, 30};
  numc_array_write(a, da);

  int err = numc_abs_inplace(a);
  ASSERT_MSG(err == 0, "2D abs_inplace should succeed");

  int32_t *r = (int32_t *)numc_array_data(a);
  ASSERT_MSG(r[0] == 10 && r[1] == 20 && r[2] == 30, "abs_inplace 2D row 0");
  ASSERT_MSG(r[3] == 10 && r[4] == 20 && r[5] == 30,
             "abs_inplace 2D row 1 unchanged");

  numc_ctx_free(ctx);
  return 0;
}

static int test_abs_inplace_zeros(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);

  int err = numc_abs_inplace(a);
  ASSERT_MSG(err == 0, "abs_inplace of zeros should succeed");

  int32_t *r = (int32_t *)numc_array_data(a);
  for (size_t i = 0; i < 4; i++) {
    ASSERT_MSG(r[i] == 0, "abs_inplace of zero should be zero");
  }

  numc_ctx_free(ctx);
  return 0;
}

static int test_abs_inplace_null(void) {
  ASSERT_MSG(numc_abs_inplace(NULL) != 0, "abs_inplace with NULL should fail");
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== elemwise/test_abs ===\n\n");

  printf("numc_abs:\n");
  RUN_TEST(test_abs_int8);
  RUN_TEST(test_abs_int16);
  RUN_TEST(test_abs_int32);
  RUN_TEST(test_abs_int64);
  RUN_TEST(test_abs_float32);
  RUN_TEST(test_abs_float64);
  RUN_TEST(test_abs_2d);
  RUN_TEST(test_abs_strided);
  RUN_TEST(test_abs_zeros);
  RUN_TEST(test_abs_all_positive);
  RUN_TEST(test_abs_int8_min);
  RUN_TEST(test_abs_null);
  RUN_TEST(test_abs_type_mismatch);
  RUN_TEST(test_abs_shape_mismatch);

  printf("\nnumc_abs_inplace:\n");
  RUN_TEST(test_abs_inplace_int8);
  RUN_TEST(test_abs_inplace_int32);
  RUN_TEST(test_abs_inplace_float32);
  RUN_TEST(test_abs_inplace_float64);
  RUN_TEST(test_abs_inplace_2d);
  RUN_TEST(test_abs_inplace_zeros);
  RUN_TEST(test_abs_inplace_null);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
