#include "../helpers.h"

/* LN2 constants matching the kernel's bit-manipulation coefficients */
#define LN2F 0.69314718f
#define LN2D 0.6931471805599453

static int test_log_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  /* Powers of 2: mantissa == 1.0, polynomial contribution == 0.
   * Result = exponent * LN2F exactly. */
  float da[] = {1.0f, 2.0f, 4.0f};
  numc_array_write(a, da);

  int err = numc_log(a, out);
  ASSERT_MSG(err == 0, "log float32 should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 0.0f, "log(1.0) == 0.0");
  ASSERT_MSG(r[1] == LN2F, "log(2.0) == LN2F");
  ASSERT_MSG(r[2] == 2 * LN2F, "log(4.0) == 2*LN2F");

  numc_ctx_free(ctx);
  return 0;
}

static int test_log_float64(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT64);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT64);

  double da[] = {1.0, 2.0, 4.0};
  numc_array_write(a, da);

  int err = numc_log(a, out);
  ASSERT_MSG(err == 0, "log float64 should succeed");

  double *r = (double *)numc_array_data(out);
  ASSERT_MSG(r[0] == 0.0, "log(1.0) == 0.0");
  ASSERT_MSG(r[1] == LN2D, "log(2.0) == LN2D");
  ASSERT_MSG(r[2] == 2 * LN2D, "log(4.0) == 2*LN2D");

  numc_ctx_free(ctx);
  return 0;
}

static int test_log_int8(void) {
  /* int8 casts through float32 log; result is truncated to int8.
   * log(1)=0.0->0, log(2)=0.69->0, log(4)=1.38->1, log(8)=2.07->2 */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT8);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT8);

  int8_t da[] = {1, 2, 4, 8};
  numc_array_write(a, da);

  int err = numc_log(a, out);
  ASSERT_MSG(err == 0, "log int8 should succeed");

  int8_t *r = (int8_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 0, "log(1) -> int8 0");
  ASSERT_MSG(r[1] == 0, "log(2) -> int8 0 (truncated)");
  ASSERT_MSG(r[2] == 1, "log(4) -> int8 1");
  ASSERT_MSG(r[3] == 2, "log(8) -> int8 2");

  numc_ctx_free(ctx);
  return 0;
}

static int test_log_int32(void) {
  /* int32 casts through float64 log; result truncated to int32.
   * log(1)=0, log(4)=1.38->1, log(1024)=10*LN2=6.93->6 */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);

  int32_t da[] = {1, 4, 1024};
  numc_array_write(a, da);

  int err = numc_log(a, out);
  ASSERT_MSG(err == 0, "log int32 should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 0, "log(1)    -> int32 0");
  ASSERT_MSG(r[1] == 1, "log(4)    -> int32 1");
  ASSERT_MSG(r[2] == 6, "log(1024) -> int32 6");

  numc_ctx_free(ctx);
  return 0;
}

static int test_log_2d(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 2};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_FLOAT32);

  float da[] = {1.0f, 2.0f, 4.0f, 8.0f};
  numc_array_write(a, da);

  int err = numc_log(a, out);
  ASSERT_MSG(err == 0, "log 2D should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 0.0f, "log(1.0)");
  ASSERT_MSG(r[1] == LN2F, "log(2.0)");
  ASSERT_MSG(r[2] == 2 * LN2F, "log(4.0)");
  ASSERT_MSG(r[3] == 3 * LN2F, "log(8.0)");

  numc_ctx_free(ctx);
  return 0;
}

static int test_log_x1(void) {
  /* log(1) == 0 for all supported float types */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {1};

  NumcArray *f32 = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *f32_out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  float vf = 1.0f;
  numc_array_write(f32, &vf);
  numc_log(f32, f32_out);
  ASSERT_MSG(*(float *)numc_array_data(f32_out) == 0.0f, "log(1.0f) == 0");

  NumcArray *f64 = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT64);
  NumcArray *f64_out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT64);
  double vd = 1.0;
  numc_array_write(f64, &vd);
  numc_log(f64, f64_out);
  ASSERT_MSG(*(double *)numc_array_data(f64_out) == 0.0, "log(1.0) == 0");

  numc_ctx_free(ctx);
  return 0;
}

static int test_log_null(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  int err = numc_log(NULL, out);
  ASSERT_MSG(err == NUMC_ERR_NULL,
             "log(NULL, out) should return NUMC_ERR_NULL");

  numc_ctx_free(ctx);
  return 0;
}

static int test_log_type_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT64);

  int err = numc_log(a, out);
  ASSERT_MSG(err == NUMC_ERR_TYPE,
             "dtype mismatch should return NUMC_ERR_TYPE");

  numc_ctx_free(ctx);
  return 0;
}

static int test_log_shape_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sa[] = {4}, sb[] = {6};
  NumcArray *a = numc_array_create(ctx, sa, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, sb, 1, NUMC_DTYPE_FLOAT32);

  int err = numc_log(a, out);
  ASSERT_MSG(err == NUMC_ERR_SHAPE,
             "shape mismatch should return NUMC_ERR_SHAPE");

  numc_ctx_free(ctx);
  return 0;
}

static int test_log_inplace_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {1.0f, 2.0f, 4.0f};
  numc_array_write(a, da);

  int err = numc_log_inplace(a);
  ASSERT_MSG(err == 0, "log_inplace float32 should succeed");

  float *r = (float *)numc_array_data(a);
  ASSERT_MSG(r[0] == 0.0f, "log_inplace(1.0) == 0.0");
  ASSERT_MSG(r[1] == LN2F, "log_inplace(2.0) == LN2F");
  ASSERT_MSG(r[2] == 2 * LN2F, "log_inplace(4.0) == 2*LN2F");

  numc_ctx_free(ctx);
  return 0;
}

static int test_log_inplace_float64(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT64);

  double da[] = {1.0, 2.0, 4.0};
  numc_array_write(a, da);

  int err = numc_log_inplace(a);
  ASSERT_MSG(err == 0, "log_inplace float64 should succeed");

  double *r = (double *)numc_array_data(a);
  ASSERT_MSG(r[0] == 0.0, "log_inplace(1.0) == 0.0");
  ASSERT_MSG(r[1] == LN2D, "log_inplace(2.0) == LN2D");
  ASSERT_MSG(r[2] == 2 * LN2D, "log_inplace(4.0) == 2*LN2D");

  numc_ctx_free(ctx);
  return 0;
}

static int test_log_inplace_null(void) {
  int err = numc_log_inplace(NULL);
  ASSERT_MSG(err == NUMC_ERR_NULL,
             "log_inplace(NULL) should return NUMC_ERR_NULL");
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== elemwise/test_log ===\n\n");

  printf("numc_log:\n");
  RUN_TEST(test_log_float32);
  RUN_TEST(test_log_float64);
  RUN_TEST(test_log_int8);
  RUN_TEST(test_log_int32);
  RUN_TEST(test_log_2d);
  RUN_TEST(test_log_x1);
  RUN_TEST(test_log_null);
  RUN_TEST(test_log_type_mismatch);
  RUN_TEST(test_log_shape_mismatch);

  printf("\nnumc_log_inplace:\n");
  RUN_TEST(test_log_inplace_float32);
  RUN_TEST(test_log_inplace_float64);
  RUN_TEST(test_log_inplace_null);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
