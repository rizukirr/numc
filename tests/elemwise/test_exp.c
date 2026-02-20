#include "../helpers.h"

/* Simple absolute-value helpers — avoid math.h dependency in the test file */
#define FABSF_T(x) ((x) >= 0.0f ? (x) : -(x))
#define FABS_T(x) ((x) >= 0.0 ? (x) : -(x))

/* < 1 ULP for float32 at |x|~1 is ~1.2e-7; 1e-5 gives comfortable headroom */
#define EXP_EPS32 1e-5f
#define EXP_EPS64 1e-12

static int test_exp_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {0.0f, 1.0f, -1.0f};
  numc_array_write(a, da);

  int err = numc_exp(a, out);
  ASSERT_MSG(err == 0, "exp float32 should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1.0f, "exp(0.0f) == 1.0f exactly");
  ASSERT_MSG(FABSF_T(r[1] - 2.7182817f) <= EXP_EPS32, "exp(1.0f) ≈ e");
  ASSERT_MSG(FABSF_T(r[2] - 0.36787944f) <= EXP_EPS32, "exp(-1.0f) ≈ 1/e");

  numc_ctx_free(ctx);
  return 0;
}

static int test_exp_float64(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT64);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT64);

  double da[] = {0.0, 1.0, -1.0};
  numc_array_write(a, da);

  int err = numc_exp(a, out);
  ASSERT_MSG(err == 0, "exp float64 should succeed");

  double *r = (double *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1.0, "exp(0.0) == 1.0 exactly");
  ASSERT_MSG(FABS_T(r[1] - 2.718281828459045) <= EXP_EPS64, "exp(1.0) ≈ e");
  ASSERT_MSG(FABS_T(r[2] - 0.36787944117144233) <= EXP_EPS64,
             "exp(-1.0) ≈ 1/e");

  numc_ctx_free(ctx);
  return 0;
}

static int test_exp_overflow_underflow(void) {
  /* float32 overflows above 88.376..., underflows below -103.972... */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {89.0f, -104.0f};
  numc_array_write(a, da);

  int err = numc_exp(a, out);
  ASSERT_MSG(err == 0, "exp overflow/underflow should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] > 1e38f, "exp(89.0f) should be +inf");
  ASSERT_MSG(r[1] == 0.0f, "exp(-104.0f) should be 0.0f");

  numc_ctx_free(ctx);
  return 0;
}

static int test_exp_int8(void) {
  /* int8 casts through float32; result truncated.
   * exp(0)=1, exp(1)≈2.718->2, exp(2)≈7.389->7, exp(3)≈20.08->20 */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT8);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT8);

  int8_t da[] = {0, 1, 2, 3};
  numc_array_write(a, da);

  int err = numc_exp(a, out);
  ASSERT_MSG(err == 0, "exp int8 should succeed");

  int8_t *r = (int8_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1, "exp(0) -> int8 1");
  ASSERT_MSG(r[1] == 2, "exp(1) -> int8 2 (truncated)");
  ASSERT_MSG(r[2] == 7, "exp(2) -> int8 7 (truncated)");
  ASSERT_MSG(r[3] == 20, "exp(3) -> int8 20 (truncated)");

  numc_ctx_free(ctx);
  return 0;
}

static int test_exp_int32(void) {
  /* int32 casts through float64; result truncated.
   * exp(0)=1, exp(1)≈2.718->2, exp(10)≈22026.46->22026 */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);

  int32_t da[] = {0, 1, 10};
  numc_array_write(a, da);

  int err = numc_exp(a, out);
  ASSERT_MSG(err == 0, "exp int32 should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1, "exp(0)  -> int32 1");
  ASSERT_MSG(r[1] == 2, "exp(1)  -> int32 2");
  ASSERT_MSG(r[2] == 22026, "exp(10) -> int32 22026");

  numc_ctx_free(ctx);
  return 0;
}

static int test_exp_uint8(void) {
  /* uint8 casts through float32; result truncated.
   * exp(0)=1, exp(1)≈2.718->2, exp(2)≈7.389->7, exp(3)≈20.08->20 */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_UINT8);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_UINT8);

  uint8_t da[] = {0, 1, 2, 3};
  numc_array_write(a, da);

  int err = numc_exp(a, out);
  ASSERT_MSG(err == 0, "exp uint8 should succeed");

  uint8_t *r = (uint8_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1, "exp(0) -> uint8 1");
  ASSERT_MSG(r[1] == 2, "exp(1) -> uint8 2 (truncated)");
  ASSERT_MSG(r[2] == 7, "exp(2) -> uint8 7 (truncated)");
  ASSERT_MSG(r[3] == 20, "exp(3) -> uint8 20 (truncated)");

  numc_ctx_free(ctx);
  return 0;
}

static int test_exp_uint16(void) {
  /* uint16 casts through float32; result truncated.
   * exp(0)=1, exp(1)≈2.718->2, exp(10)≈22026.47->22026 */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_UINT16);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_UINT16);

  uint16_t da[] = {0, 1, 10};
  numc_array_write(a, da);

  int err = numc_exp(a, out);
  ASSERT_MSG(err == 0, "exp uint16 should succeed");

  uint16_t *r = (uint16_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1, "exp(0)  -> uint16 1");
  ASSERT_MSG(r[1] == 2, "exp(1)  -> uint16 2 (truncated)");
  ASSERT_MSG(r[2] == 22026, "exp(10) -> uint16 22026 (truncated)");

  numc_ctx_free(ctx);
  return 0;
}

static int test_exp_2d(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 2};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_FLOAT32);

  float da[] = {0.0f, 1.0f, 2.0f, 3.0f};
  numc_array_write(a, da);

  int err = numc_exp(a, out);
  ASSERT_MSG(err == 0, "exp 2D should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1.0f, "exp(0.0)");
  ASSERT_MSG(FABSF_T(r[1] - 2.7182817f) <= EXP_EPS32, "exp(1.0)");
  ASSERT_MSG(FABSF_T(r[2] - 7.389056f) <= EXP_EPS32, "exp(2.0)");
  ASSERT_MSG(FABSF_T(r[3] - 20.085537f) <= EXP_EPS32, "exp(3.0)");

  numc_ctx_free(ctx);
  return 0;
}

static int test_exp_x0(void) {
  /* exp(0) == 1 for both float types */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {1};

  NumcArray *f32 = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *f32_out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  float vf = 0.0f;
  numc_array_write(f32, &vf);
  numc_exp(f32, f32_out);
  ASSERT_MSG(*(float *)numc_array_data(f32_out) == 1.0f, "exp(0.0f) == 1.0f");

  NumcArray *f64 = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT64);
  NumcArray *f64_out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT64);
  double vd = 0.0;
  numc_array_write(f64, &vd);
  numc_exp(f64, f64_out);
  ASSERT_MSG(*(double *)numc_array_data(f64_out) == 1.0, "exp(0.0) == 1.0");

  numc_ctx_free(ctx);
  return 0;
}

static int test_exp_null(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  int err = numc_exp(NULL, out);
  ASSERT_MSG(err == NUMC_ERR_NULL,
             "exp(NULL, out) should return NUMC_ERR_NULL");

  numc_ctx_free(ctx);
  return 0;
}

static int test_exp_type_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT64);

  int err = numc_exp(a, out);
  ASSERT_MSG(err == NUMC_ERR_TYPE,
             "dtype mismatch should return NUMC_ERR_TYPE");

  numc_ctx_free(ctx);
  return 0;
}

static int test_exp_shape_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sa[] = {4}, sb[] = {6};
  NumcArray *a = numc_array_create(ctx, sa, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, sb, 1, NUMC_DTYPE_FLOAT32);

  int err = numc_exp(a, out);
  ASSERT_MSG(err == NUMC_ERR_SHAPE,
             "shape mismatch should return NUMC_ERR_SHAPE");

  numc_ctx_free(ctx);
  return 0;
}

static int test_exp_inplace_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {0.0f, 1.0f, 2.0f};
  numc_array_write(a, da);

  int err = numc_exp_inplace(a);
  ASSERT_MSG(err == 0, "exp_inplace float32 should succeed");

  float *r = (float *)numc_array_data(a);
  ASSERT_MSG(r[0] == 1.0f, "exp_inplace(0.0) == 1.0");
  ASSERT_MSG(FABSF_T(r[1] - 2.7182817f) <= EXP_EPS32, "exp_inplace(1.0) ≈ e");
  ASSERT_MSG(FABSF_T(r[2] - 7.389056f) <= EXP_EPS32, "exp_inplace(2.0) ≈ e^2");

  numc_ctx_free(ctx);
  return 0;
}

static int test_exp_inplace_float64(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT64);

  double da[] = {0.0, 1.0, -1.0};
  numc_array_write(a, da);

  int err = numc_exp_inplace(a);
  ASSERT_MSG(err == 0, "exp_inplace float64 should succeed");

  double *r = (double *)numc_array_data(a);
  ASSERT_MSG(r[0] == 1.0, "exp_inplace(0.0) == 1.0");
  ASSERT_MSG(FABS_T(r[1] - 2.718281828459045) <= EXP_EPS64,
             "exp_inplace(1.0) ≈ e");
  ASSERT_MSG(FABS_T(r[2] - 0.36787944117144233) <= EXP_EPS64,
             "exp_inplace(-1.0) ≈ 1/e");

  numc_ctx_free(ctx);
  return 0;
}

static int test_exp_inplace_uint8(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_UINT8);

  uint8_t da[] = {0, 1, 2};
  numc_array_write(a, da);

  int err = numc_exp_inplace(a);
  ASSERT_MSG(err == 0, "exp_inplace uint8 should succeed");

  uint8_t *r = (uint8_t *)numc_array_data(a);
  ASSERT_MSG(r[0] == 1, "exp_inplace(0) -> uint8 1");
  ASSERT_MSG(r[1] == 2, "exp_inplace(1) -> uint8 2 (truncated)");
  ASSERT_MSG(r[2] == 7, "exp_inplace(2) -> uint8 7 (truncated)");

  numc_ctx_free(ctx);
  return 0;
}

static int test_exp_inplace_uint16(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_UINT16);

  uint16_t da[] = {0, 1, 10};
  numc_array_write(a, da);

  int err = numc_exp_inplace(a);
  ASSERT_MSG(err == 0, "exp_inplace uint16 should succeed");

  uint16_t *r = (uint16_t *)numc_array_data(a);
  ASSERT_MSG(r[0] == 1, "exp_inplace(0)  -> uint16 1");
  ASSERT_MSG(r[1] == 2, "exp_inplace(1)  -> uint16 2 (truncated)");
  ASSERT_MSG(r[2] == 22026, "exp_inplace(10) -> uint16 22026 (truncated)");

  numc_ctx_free(ctx);
  return 0;
}

static int test_exp_inplace_null(void) {
  int err = numc_exp_inplace(NULL);
  ASSERT_MSG(err == NUMC_ERR_NULL,
             "exp_inplace(NULL) should return NUMC_ERR_NULL");
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== elemwise/test_exp ===\n\n");

  printf("numc_exp:\n");
  RUN_TEST(test_exp_float32);
  RUN_TEST(test_exp_float64);
  RUN_TEST(test_exp_overflow_underflow);
  RUN_TEST(test_exp_int8);
  RUN_TEST(test_exp_int32);
  RUN_TEST(test_exp_uint8);
  RUN_TEST(test_exp_uint16);
  RUN_TEST(test_exp_2d);
  RUN_TEST(test_exp_x0);
  RUN_TEST(test_exp_null);
  RUN_TEST(test_exp_type_mismatch);
  RUN_TEST(test_exp_shape_mismatch);

  printf("\nnumc_exp_inplace:\n");
  RUN_TEST(test_exp_inplace_float32);
  RUN_TEST(test_exp_inplace_float64);
  RUN_TEST(test_exp_inplace_uint8);
  RUN_TEST(test_exp_inplace_uint16);
  RUN_TEST(test_exp_inplace_null);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
