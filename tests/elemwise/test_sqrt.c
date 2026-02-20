#include "../helpers.h"

static int test_sqrt_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {0.0f, 1.0f, 4.0f, 9.0f};
  numc_array_write(a, da);

  int err = numc_sqrt(a, out);
  ASSERT_MSG(err == 0, "sqrt float32 should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 0.0f, "sqrt(0.0)  == 0.0");
  ASSERT_MSG(r[1] == 1.0f, "sqrt(1.0)  == 1.0");
  ASSERT_MSG(r[2] == 2.0f, "sqrt(4.0)  == 2.0");
  ASSERT_MSG(r[3] == 3.0f, "sqrt(9.0)  == 3.0");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sqrt_float64(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT64);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT64);

  double da[] = {0.0, 1.0, 4.0, 9.0};
  numc_array_write(a, da);

  int err = numc_sqrt(a, out);
  ASSERT_MSG(err == 0, "sqrt float64 should succeed");

  double *r = (double *)numc_array_data(out);
  ASSERT_MSG(r[0] == 0.0, "sqrt(0.0)  == 0.0");
  ASSERT_MSG(r[1] == 1.0, "sqrt(1.0)  == 1.0");
  ASSERT_MSG(r[2] == 2.0, "sqrt(4.0)  == 2.0");
  ASSERT_MSG(r[3] == 3.0, "sqrt(9.0)  == 3.0");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sqrt_int8(void) {
  /* int8 casts through float32; result truncated.
   * sqrt(0)=0, sqrt(1)=1, sqrt(4)=2, sqrt(9)=3 */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT8);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT8);

  int8_t da[] = {0, 1, 4, 9};
  numc_array_write(a, da);

  int err = numc_sqrt(a, out);
  ASSERT_MSG(err == 0, "sqrt int8 should succeed");

  int8_t *r = (int8_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 0, "sqrt(0) -> int8 0");
  ASSERT_MSG(r[1] == 1, "sqrt(1) -> int8 1");
  ASSERT_MSG(r[2] == 2, "sqrt(4) -> int8 2");
  ASSERT_MSG(r[3] == 3, "sqrt(9) -> int8 3");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sqrt_int8_negative(void) {
  /* Negative inputs are clamped to 0 before sqrt. */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT8);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT8);

  int8_t da[] = {-4, -1, -9};
  numc_array_write(a, da);

  int err = numc_sqrt(a, out);
  ASSERT_MSG(err == 0, "sqrt int8 negative should succeed");

  int8_t *r = (int8_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 0, "sqrt(-4) -> int8 0 (clamped)");
  ASSERT_MSG(r[1] == 0, "sqrt(-1) -> int8 0 (clamped)");
  ASSERT_MSG(r[2] == 0, "sqrt(-9) -> int8 0 (clamped)");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sqrt_int32(void) {
  /* int32 casts through float64; result truncated. */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);

  int32_t da[] = {0, 1, 100, 10000};
  numc_array_write(a, da);

  int err = numc_sqrt(a, out);
  ASSERT_MSG(err == 0, "sqrt int32 should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 0,   "sqrt(0)     -> int32 0");
  ASSERT_MSG(r[1] == 1,   "sqrt(1)     -> int32 1");
  ASSERT_MSG(r[2] == 10,  "sqrt(100)   -> int32 10");
  ASSERT_MSG(r[3] == 100, "sqrt(10000) -> int32 100");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sqrt_uint8(void) {
  /* Unsigned — always non-negative, cast through float32. */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_UINT8);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_UINT8);

  uint8_t da[] = {0, 1, 4, 9};
  numc_array_write(a, da);

  int err = numc_sqrt(a, out);
  ASSERT_MSG(err == 0, "sqrt uint8 should succeed");

  uint8_t *r = (uint8_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 0, "sqrt(0) -> uint8 0");
  ASSERT_MSG(r[1] == 1, "sqrt(1) -> uint8 1");
  ASSERT_MSG(r[2] == 2, "sqrt(4) -> uint8 2");
  ASSERT_MSG(r[3] == 3, "sqrt(9) -> uint8 3");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sqrt_2d(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 2};
  NumcArray *a   = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_FLOAT32);

  float da[] = {1.0f, 4.0f, 9.0f, 16.0f};
  numc_array_write(a, da);

  int err = numc_sqrt(a, out);
  ASSERT_MSG(err == 0, "sqrt 2D should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1.0f, "sqrt(1.0)");
  ASSERT_MSG(r[1] == 2.0f, "sqrt(4.0)");
  ASSERT_MSG(r[2] == 3.0f, "sqrt(9.0)");
  ASSERT_MSG(r[3] == 4.0f, "sqrt(16.0)");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sqrt_null(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  int err = numc_sqrt(NULL, out);
  ASSERT_MSG(err == NUMC_ERR_NULL,
             "sqrt(NULL, out) should return NUMC_ERR_NULL");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sqrt_type_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT64);

  int err = numc_sqrt(a, out);
  ASSERT_MSG(err == NUMC_ERR_TYPE,
             "dtype mismatch should return NUMC_ERR_TYPE");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sqrt_shape_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sa[] = {4}, sb[] = {6};
  NumcArray *a   = numc_array_create(ctx, sa, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, sb, 1, NUMC_DTYPE_FLOAT32);

  int err = numc_sqrt(a, out);
  ASSERT_MSG(err == NUMC_ERR_SHAPE,
             "shape mismatch should return NUMC_ERR_SHAPE");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sqrt_inplace_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {0.0f, 1.0f, 4.0f, 9.0f};
  numc_array_write(a, da);

  int err = numc_sqrt_inplace(a);
  ASSERT_MSG(err == 0, "sqrt_inplace float32 should succeed");

  float *r = (float *)numc_array_data(a);
  ASSERT_MSG(r[0] == 0.0f, "sqrt_inplace(0.0) == 0.0");
  ASSERT_MSG(r[1] == 1.0f, "sqrt_inplace(1.0) == 1.0");
  ASSERT_MSG(r[2] == 2.0f, "sqrt_inplace(4.0) == 2.0");
  ASSERT_MSG(r[3] == 3.0f, "sqrt_inplace(9.0) == 3.0");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sqrt_inplace_float64(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT64);

  double da[] = {0.0, 1.0, 4.0, 9.0};
  numc_array_write(a, da);

  int err = numc_sqrt_inplace(a);
  ASSERT_MSG(err == 0, "sqrt_inplace float64 should succeed");

  double *r = (double *)numc_array_data(a);
  ASSERT_MSG(r[0] == 0.0, "sqrt_inplace(0.0) == 0.0");
  ASSERT_MSG(r[1] == 1.0, "sqrt_inplace(1.0) == 1.0");
  ASSERT_MSG(r[2] == 2.0, "sqrt_inplace(4.0) == 2.0");
  ASSERT_MSG(r[3] == 3.0, "sqrt_inplace(9.0) == 3.0");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sqrt_inplace_null(void) {
  int err = numc_sqrt_inplace(NULL);
  ASSERT_MSG(err == NUMC_ERR_NULL,
             "sqrt_inplace(NULL) should return NUMC_ERR_NULL");
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== elemwise/test_sqrt ===\n\n");

  printf("numc_sqrt:\n");
  RUN_TEST(test_sqrt_float32);
  RUN_TEST(test_sqrt_float64);
  RUN_TEST(test_sqrt_int8);
  RUN_TEST(test_sqrt_int8_negative);
  RUN_TEST(test_sqrt_int32);
  RUN_TEST(test_sqrt_uint8);
  RUN_TEST(test_sqrt_2d);
  RUN_TEST(test_sqrt_null);
  RUN_TEST(test_sqrt_type_mismatch);
  RUN_TEST(test_sqrt_shape_mismatch);

  printf("\nnumc_sqrt_inplace:\n");
  RUN_TEST(test_sqrt_inplace_float32);
  RUN_TEST(test_sqrt_inplace_float64);
  RUN_TEST(test_sqrt_inplace_null);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
