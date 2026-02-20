#include "../helpers.h"

#define FABSF_T(x) ((x) >= 0.0f ? (x) : -(x))
#define FABS_T(x) ((x) >= 0.0 ? (x) : -(x))

#define POW_EPS32 1e-4f
#define POW_EPS64 1e-10

static int test_pow_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {2.0f, 3.0f, 4.0f, 10.0f};
  float db[] = {3.0f, 2.0f, 0.5f, 2.0f};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_pow(a, b, out);
  ASSERT_MSG(err == 0, "pow float32 should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(FABSF_T(r[0] - 8.0f) <= POW_EPS32, "pow(2,3) ≈ 8");
  ASSERT_MSG(FABSF_T(r[1] - 9.0f) <= POW_EPS32, "pow(3,2) ≈ 9");
  ASSERT_MSG(FABSF_T(r[2] - 2.0f) <= POW_EPS32, "pow(4,0.5) ≈ 2");
  ASSERT_MSG(FABSF_T(r[3] - 100.0f) <= POW_EPS32, "pow(10,2) ≈ 100");

  numc_ctx_free(ctx);
  return 0;
}

static int test_pow_float64(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT64);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT64);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT64);

  double da[] = {2.0, 3.0, 4.0, 10.0};
  double db[] = {3.0, 2.0, 0.5, 2.0};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_pow(a, b, out);
  ASSERT_MSG(err == 0, "pow float64 should succeed");

  double *r = (double *)numc_array_data(out);
  ASSERT_MSG(FABS_T(r[0] - 8.0) <= POW_EPS64, "pow(2,3) ≈ 8");
  ASSERT_MSG(FABS_T(r[1] - 9.0) <= POW_EPS64, "pow(3,2) ≈ 9");
  ASSERT_MSG(FABS_T(r[2] - 2.0) <= POW_EPS64, "pow(4,0.5) ≈ 2");
  ASSERT_MSG(FABS_T(r[3] - 100.0) <= POW_EPS64, "pow(10,2) ≈ 100");

  numc_ctx_free(ctx);
  return 0;
}

static int test_pow_int32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);

  int32_t da[] = {2, 3, 5, 10};
  int32_t db[] = {10, 5, 3, 2};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_pow(a, b, out);
  ASSERT_MSG(err == 0, "pow int32 should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1024, "pow(2,10) == 1024");
  ASSERT_MSG(r[1] == 243, "pow(3,5) == 243");
  ASSERT_MSG(r[2] == 125, "pow(5,3) == 125");
  ASSERT_MSG(r[3] == 100, "pow(10,2) == 100");

  numc_ctx_free(ctx);
  return 0;
}

static int test_pow_int8(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT8);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT8);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT8);

  int8_t da[] = {2, 3, 5, 2};
  int8_t db[] = {6, 4, 2, 3};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_pow(a, b, out);
  ASSERT_MSG(err == 0, "pow int8 should succeed");

  int8_t *r = (int8_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 64, "pow(2,6) == 64");
  ASSERT_MSG(r[1] == 81, "pow(3,4) == 81");
  ASSERT_MSG(r[2] == 25, "pow(5,2) == 25");
  ASSERT_MSG(r[3] == 8, "pow(2,3) == 8");

  numc_ctx_free(ctx);
  return 0;
}

static int test_pow_uint8(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_UINT8);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_UINT8);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_UINT8);

  uint8_t da[] = {2, 3, 5};
  uint8_t db[] = {7, 4, 2};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_pow(a, b, out);
  ASSERT_MSG(err == 0, "pow uint8 should succeed");

  uint8_t *r = (uint8_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 128, "pow(2,7) == 128");
  ASSERT_MSG(r[1] == 81, "pow(3,4) == 81");
  ASSERT_MSG(r[2] == 25, "pow(5,2) == 25");

  numc_ctx_free(ctx);
  return 0;
}

static int test_pow_int32_negative_base(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);

  int32_t da[] = {-3, -3, -2, -1};
  int32_t db[] = {2, 3, 4, 5};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_pow(a, b, out);
  ASSERT_MSG(err == 0, "pow int32 neg base should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 9, "(-3)^2 == 9");
  ASSERT_MSG(r[1] == -27, "(-3)^3 == -27");
  ASSERT_MSG(r[2] == 16, "(-2)^4 == 16");
  ASSERT_MSG(r[3] == -1, "(-1)^5 == -1");

  numc_ctx_free(ctx);
  return 0;
}

static int test_pow_int32_zero_exp(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);

  int32_t da[] = {0, 1, 42, -7};
  int32_t db[] = {0, 0, 0, 0};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_pow(a, b, out);
  ASSERT_MSG(err == 0, "pow int32 zero exp should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1, "0^0 == 1");
  ASSERT_MSG(r[1] == 1, "1^0 == 1");
  ASSERT_MSG(r[2] == 1, "42^0 == 1");
  ASSERT_MSG(r[3] == 1, "(-7)^0 == 1");

  numc_ctx_free(ctx);
  return 0;
}

static int test_pow_int32_exp_one(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);

  int32_t da[] = {0, 42, -7};
  int32_t db[] = {1, 1, 1};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_pow(a, b, out);
  ASSERT_MSG(err == 0, "pow int32 exp one should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 0, "0^1 == 0");
  ASSERT_MSG(r[1] == 42, "42^1 == 42");
  ASSERT_MSG(r[2] == -7, "(-7)^1 == -7");

  numc_ctx_free(ctx);
  return 0;
}

static int test_pow_int32_neg_exp(void) {
  /* Negative exponent: integer truncation → 0 for |base| > 1 */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);

  int32_t da[] = {2, 10, 3};
  int32_t db[] = {-1, -2, -3};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_pow(a, b, out);
  ASSERT_MSG(err == 0, "pow int32 neg exp should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 0, "2^(-1) → 0 (integer truncation)");
  ASSERT_MSG(r[1] == 0, "10^(-2) → 0 (integer truncation)");
  ASSERT_MSG(r[2] == 0, "3^(-3) → 0 (integer truncation)");

  numc_ctx_free(ctx);
  return 0;
}

static int test_pow_2d(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 2};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_INT32);

  int32_t da[] = {2, 3, 4, 5};
  int32_t db[] = {3, 2, 2, 3};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_pow(a, b, out);
  ASSERT_MSG(err == 0, "pow 2d should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 8, "pow(2,3) == 8");
  ASSERT_MSG(r[1] == 9, "pow(3,2) == 9");
  ASSERT_MSG(r[2] == 16, "pow(4,2) == 16");
  ASSERT_MSG(r[3] == 125, "pow(5,3) == 125");

  numc_ctx_free(ctx);
  return 0;
}

static int test_pow_null(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  int err = numc_pow(NULL, a, out);
  ASSERT_MSG(err == NUMC_ERR_NULL, "pow(NULL, a, out) should fail");
  err = numc_pow(a, NULL, out);
  ASSERT_MSG(err == NUMC_ERR_NULL, "pow(a, NULL, out) should fail");
  err = numc_pow(a, a, NULL);
  ASSERT_MSG(err == NUMC_ERR_NULL, "pow(a, a, NULL) should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_pow_type_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  int err = numc_pow(a, b, out);
  ASSERT_MSG(err == NUMC_ERR_TYPE, "pow type mismatch should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_pow_shape_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape_a[] = {4};
  size_t shape_b[] = {3};
  NumcArray *a = numc_array_create(ctx, shape_a, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_create(ctx, shape_b, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape_a, 1, NUMC_DTYPE_FLOAT32);

  int err = numc_pow(a, b, out);
  ASSERT_MSG(err == NUMC_ERR_SHAPE, "pow shape mismatch should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_pow_inplace_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {2.0f, 3.0f, 10.0f};
  float db[] = {3.0f, 2.0f, 2.0f};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_pow_inplace(a, b);
  ASSERT_MSG(err == 0, "pow_inplace float32 should succeed");

  float *r = (float *)numc_array_data(a);
  ASSERT_MSG(FABSF_T(r[0] - 8.0f) <= POW_EPS32, "pow_inplace(2,3) ≈ 8");
  ASSERT_MSG(FABSF_T(r[1] - 9.0f) <= POW_EPS32, "pow_inplace(3,2) ≈ 9");
  ASSERT_MSG(FABSF_T(r[2] - 100.0f) <= POW_EPS32, "pow_inplace(10,2) ≈ 100");

  numc_ctx_free(ctx);
  return 0;
}

static int test_pow_inplace_int32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);

  int32_t da[] = {2, -3, 5, 10};
  int32_t db[] = {10, 3, 2, 2};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_pow_inplace(a, b);
  ASSERT_MSG(err == 0, "pow_inplace int32 should succeed");

  int32_t *r = (int32_t *)numc_array_data(a);
  ASSERT_MSG(r[0] == 1024, "pow_inplace(2,10) == 1024");
  ASSERT_MSG(r[1] == -27, "pow_inplace(-3,3) == -27");
  ASSERT_MSG(r[2] == 25, "pow_inplace(5,2) == 25");
  ASSERT_MSG(r[3] == 100, "pow_inplace(10,2) == 100");

  numc_ctx_free(ctx);
  return 0;
}

static int test_pow_inplace_null(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  int err = numc_pow_inplace(NULL, a);
  ASSERT_MSG(err == NUMC_ERR_NULL, "pow_inplace(NULL, a) should fail");
  err = numc_pow_inplace(a, NULL);
  ASSERT_MSG(err == NUMC_ERR_NULL, "pow_inplace(a, NULL) should fail");

  numc_ctx_free(ctx);
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== elemwise/test_pow ===\n\n");

  printf("numc_pow:\n");
  RUN_TEST(test_pow_float32);
  RUN_TEST(test_pow_float64);
  RUN_TEST(test_pow_int32);
  RUN_TEST(test_pow_int8);
  RUN_TEST(test_pow_uint8);
  RUN_TEST(test_pow_int32_negative_base);
  RUN_TEST(test_pow_int32_zero_exp);
  RUN_TEST(test_pow_int32_exp_one);
  RUN_TEST(test_pow_int32_neg_exp);
  RUN_TEST(test_pow_2d);
  RUN_TEST(test_pow_null);
  RUN_TEST(test_pow_type_mismatch);
  RUN_TEST(test_pow_shape_mismatch);

  printf("\nnumc_pow_inplace:\n");
  RUN_TEST(test_pow_inplace_float32);
  RUN_TEST(test_pow_inplace_int32);
  RUN_TEST(test_pow_inplace_null);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
