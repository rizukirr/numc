#include "../helpers.h"

static int test_div_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {20.0f, 30.0f, 40.0f, 50.0f};
  float db[] = {10.0f, 10.0f, 10.0f, 10.0f};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_div(a, b, out);
  ASSERT_MSG(err == 0, "div float32 should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 2.0f && r[1] == 3.0f && r[2] == 4.0f && r[3] == 5.0f,
             "div float32 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_div_int32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);

  int32_t da[] = {100, 200, 300, 400};
  int32_t db[] = {10, 20, 30, 40};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_div(a, b, out);
  ASSERT_MSG(err == 0, "div int32 should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 10 && r[1] == 10 && r[2] == 10 && r[3] == 10,
             "div int32 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_div_float64(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT64);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT64);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT64);

  double da[] = {3.0, 6.0, 9.0};
  double db[] = {1.5, 1.5, 1.5};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_div(a, b, out);
  ASSERT_MSG(err == 0, "div float64 should succeed");

  double *r = (double *)numc_array_data(out);
  ASSERT_MSG(r[0] == 2.0 && r[1] == 4.0 && r[2] == 6.0,
             "div float64 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_div_scalar_inplace(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {10.0f, 20.0f, 30.0f, 40.0f};
  numc_array_write(a, da);

  int err = numc_div_scalar_inplace(a, 10.0);
  ASSERT_MSG(err == 0, "div_scalar_inplace should succeed");

  float *r = (float *)numc_array_data(a);
  ASSERT_MSG(r[0] == 1.0f && r[1] == 2.0f && r[2] == 3.0f && r[3] == 4.0f,
             "div_scalar_inplace results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_div_null(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  ASSERT_MSG_CTX(numc_div(NULL, a, out) != 0, "div NULL a should fail", ctx);
  ASSERT_MSG_CTX(numc_div(a, NULL, out) != 0, "div NULL b should fail", ctx);
  ASSERT_MSG_CTX(numc_div(a, a, NULL) != 0,   "div NULL out should fail", ctx);

  numc_ctx_free(ctx);
  return 0;
}

static int test_div_dtype_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *b   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT64);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  ASSERT_MSG_CTX(numc_div(a, b, out) != 0,
                 "div with dtype mismatch should fail", ctx);

  numc_ctx_free(ctx);
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== elemwise/test_div ===\n\n");

  printf("numc_div correctness:\n");
  RUN_TEST(test_div_float32);
  RUN_TEST(test_div_int32);
  RUN_TEST(test_div_float64);

  printf("\nnumc_div_scalar_inplace:\n");
  RUN_TEST(test_div_scalar_inplace);

  printf("\nnumc_div error cases:\n");
  RUN_TEST(test_div_null);
  RUN_TEST(test_div_dtype_mismatch);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
