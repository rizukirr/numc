#include "../helpers.h"

static int test_sub_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {10.0f, 20.0f, 30.0f, 40.0f};
  float db[] = {1.0f, 2.0f, 3.0f, 4.0f};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_sub(a, b, out);
  ASSERT_MSG(err == 0, "sub float32 should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 9.0f && r[1] == 18.0f && r[2] == 27.0f && r[3] == 36.0f,
             "sub float32 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sub_int32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);

  int32_t da[] = {100, 200, 300, 400};
  int32_t db[] = {1, 2, 3, 4};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_sub(a, b, out);
  ASSERT_MSG(err == 0, "sub int32 should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 99 && r[1] == 198 && r[2] == 297 && r[3] == 396,
             "sub int32 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sub_float64(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT64);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT64);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT64);

  double da[] = {1.5, 2.5, 3.5};
  double db[] = {0.5, 0.5, 0.5};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_sub(a, b, out);
  ASSERT_MSG(err == 0, "sub float64 should succeed");

  double *r = (double *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1.0 && r[1] == 2.0 && r[2] == 3.0,
             "sub float64 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sub_scalar_inplace(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {10.0f, 20.0f, 30.0f, 40.0f};
  numc_array_write(a, da);

  int err = numc_sub_scalar_inplace(a, 5.0);
  ASSERT_MSG(err == 0, "sub_scalar_inplace should succeed");

  float *r = (float *)numc_array_data(a);
  ASSERT_MSG(r[0] == 5.0f && r[1] == 15.0f && r[2] == 25.0f && r[3] == 35.0f,
             "sub_scalar_inplace results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sub_null(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  ASSERT_MSG_CTX(numc_sub(NULL, a, out) != 0, "sub NULL a should fail", ctx);
  ASSERT_MSG_CTX(numc_sub(a, NULL, out) != 0, "sub NULL b should fail", ctx);
  ASSERT_MSG_CTX(numc_sub(a, a, NULL) != 0,   "sub NULL out should fail", ctx);

  numc_ctx_free(ctx);
  return 0;
}

static int test_sub_dtype_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *b   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT64);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  ASSERT_MSG_CTX(numc_sub(a, b, out) != 0,
                 "sub with dtype mismatch should fail", ctx);

  numc_ctx_free(ctx);
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== elemwise/test_sub ===\n\n");

  printf("numc_sub correctness:\n");
  RUN_TEST(test_sub_float32);
  RUN_TEST(test_sub_int32);
  RUN_TEST(test_sub_float64);

  printf("\nnumc_sub_scalar_inplace:\n");
  RUN_TEST(test_sub_scalar_inplace);

  printf("\nnumc_sub error cases:\n");
  RUN_TEST(test_sub_null);
  RUN_TEST(test_sub_dtype_mismatch);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
