#include "../helpers.h"

static int test_mul_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {2.0f, 3.0f, 4.0f, 5.0f};
  float db[] = {10.0f, 10.0f, 10.0f, 10.0f};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_mul(a, b, out);
  ASSERT_MSG(err == 0, "mul float32 should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 20.0f && r[1] == 30.0f && r[2] == 40.0f && r[3] == 50.0f,
             "mul float32 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_mul_int32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);

  int32_t da[] = {2, 3, 4, 5};
  int32_t db[] = {10, 10, 10, 10};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_mul(a, b, out);
  ASSERT_MSG(err == 0, "mul int32 should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 20 && r[1] == 30 && r[2] == 40 && r[3] == 50,
             "mul int32 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_mul_float64(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT64);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT64);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT64);

  double da[] = {1.5, 2.5, 3.5};
  double db[] = {2.0, 2.0, 2.0};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_mul(a, b, out);
  ASSERT_MSG(err == 0, "mul float64 should succeed");

  double *r = (double *)numc_array_data(out);
  ASSERT_MSG(r[0] == 3.0 && r[1] == 5.0 && r[2] == 7.0,
             "mul float64 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_mul_scalar_inplace(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {1.0f, 2.0f, 3.0f, 4.0f};
  numc_array_write(a, da);

  int err = numc_mul_scalar_inplace(a, 3.0);
  ASSERT_MSG(err == 0, "mul_scalar_inplace should succeed");

  float *r = (float *)numc_array_data(a);
  ASSERT_MSG(r[0] == 3.0f && r[1] == 6.0f && r[2] == 9.0f && r[3] == 12.0f,
             "mul_scalar_inplace results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_mul_null(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  ASSERT_MSG_CTX(numc_mul(NULL, a, out) != 0, "mul NULL a should fail", ctx);
  ASSERT_MSG_CTX(numc_mul(a, NULL, out) != 0, "mul NULL b should fail", ctx);
  ASSERT_MSG_CTX(numc_mul(a, a, NULL) != 0,   "mul NULL out should fail", ctx);

  numc_ctx_free(ctx);
  return 0;
}

static int test_mul_dtype_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *b   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT64);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  ASSERT_MSG_CTX(numc_mul(a, b, out) != 0,
                 "mul with dtype mismatch should fail", ctx);

  numc_ctx_free(ctx);
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== elemwise/test_mul ===\n\n");

  printf("numc_mul correctness:\n");
  RUN_TEST(test_mul_float32);
  RUN_TEST(test_mul_int32);
  RUN_TEST(test_mul_float64);

  printf("\nnumc_mul_scalar_inplace:\n");
  RUN_TEST(test_mul_scalar_inplace);

  printf("\nnumc_mul error cases:\n");
  RUN_TEST(test_mul_null);
  RUN_TEST(test_mul_dtype_mismatch);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
