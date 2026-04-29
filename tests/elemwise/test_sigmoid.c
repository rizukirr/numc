#include "../helpers.h"

#define FABSF_SIG(x) ((x) >= 0.0f ? (x) : -(x))
#define FABS_SIG(x)  ((x) >= 0.0 ? (x) : -(x))

#define SIGMOID_EPS32 1e-5f
#define SIGMOID_EPS64 1e-12

static int test_sigmoid_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {-1.0f, 0.0f, 1.0f};
  numc_array_write(a, da);

  int err = numc_sigmoid(a, out);
  ASSERT_MSG(err == 0, "sigmoid float32 should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(FABSF_SIG(r[0] - 0.26894143f) < SIGMOID_EPS32, "sigmoid(-1.0f)");
  ASSERT_MSG(FABSF_SIG(r[1] - 0.5f) < SIGMOID_EPS32, "sigmoid(0.0f)");
  ASSERT_MSG(FABSF_SIG(r[2] - 0.7310586f) < SIGMOID_EPS32, "sigmoid(1.0f)");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sigmoid_float64(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT64);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT64);

  double da[] = {-1.0, 0.0, 1.0};
  numc_array_write(a, da);

  int err = numc_sigmoid(a, out);
  ASSERT_MSG(err == 0, "sigmoid float64 should succeed");

  double *r = (double *)numc_array_data(out);
  ASSERT_MSG(FABS_SIG(r[0] - 0.2689414213699951) <= SIGMOID_EPS64,
             "sigmoid(-1.0)");
  ASSERT_MSG(FABS_SIG(r[1] - 0.5) <= SIGMOID_EPS64, "sigmoid(0.0)");
  ASSERT_MSG(FABS_SIG(r[2] - 0.7310585786300049) <= SIGMOID_EPS64,
             "sigmoid(1.0)");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sigmoid_all_integer_dtypes_out(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  {
    NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT8);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT8);
    int8_t da[] = {-2, -1, 0, 2};
    numc_array_write(a, da);
    ASSERT_MSG(numc_sigmoid(a, out) == 0, "sigmoid int8 should succeed");
    int8_t *r = (int8_t *)numc_array_data(out);
    ASSERT_MSG(r[0] == 0 && r[1] == 0 && r[2] == 0 && r[3] == 0,
               "sigmoid int8 cast result should be all zeros");
  }
  {
    NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_UINT8);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_UINT8);
    uint8_t da[] = {0, 1, 2, 3};
    numc_array_write(a, da);
    ASSERT_MSG(numc_sigmoid(a, out) == 0, "sigmoid uint8 should succeed");
    uint8_t *r = (uint8_t *)numc_array_data(out);
    ASSERT_MSG(r[0] == 0 && r[1] == 0 && r[2] == 0 && r[3] == 0,
               "sigmoid uint8 cast result should be all zeros");
  }
  numc_ctx_free(ctx);
  return 0;
}

static int test_sigmoid_float32_bounds_monotonic_extremes(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {5};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  float da[] = {-100.0f, -1.0f, 0.0f, 1.0f, 100.0f};
  numc_array_write(a, da);
  ASSERT_MSG(numc_sigmoid(a, out) == 0, "sigmoid bounds should succeed");
  float *r = (float *)numc_array_data(out);
  printf("sigmoid outputs: %.9g %.9g %.9g %.9g %.9g\n", r[0], r[1], r[2], r[3],
         r[4]);
  for (size_t i = 0; i < 5; i++)
    ASSERT_MSG(r[i] >= 0.0f && r[i] <= 1.0f, "sigmoid output in [0, 1]");
  for (size_t i = 1; i < 5; i++)
    ASSERT_MSG(r[i - 1] <= r[i], "sigmoid should be monotonic");
  ASSERT_MSG(r[0] < 1e-6f, "sigmoid(-100) should be near 0");
  ASSERT_MSG(r[4] > 1.0f - 1e-6f, "sigmoid(100) should be near 1");
  numc_ctx_free(ctx);
  return 0;
}

static int test_sigmoid_inplace_matches_out_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {5};
  NumcArray *a_out = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *a_inp = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  float da[] = {-3.0f, -0.5f, 0.0f, 0.5f, 3.0f};
  numc_array_write(a_out, da);
  numc_array_write(a_inp, da);
  ASSERT_MSG(numc_sigmoid(a_out, out) == 0,
             "sigmoid out-of-place should succeed");
  ASSERT_MSG(numc_sigmoid_inplace(a_inp) == 0,
             "sigmoid inplace should succeed");
  float *ro = (float *)numc_array_data(out);
  float *ri = (float *)numc_array_data(a_inp);
  for (size_t i = 0; i < 5; i++)
    ASSERT_MSG(FABSF_SIG(ro[i] - ri[i]) <= SIGMOID_EPS32,
               "sigmoid inplace must match out-of-place");
  numc_ctx_free(ctx);
  return 0;
}

static int test_sigmoid_null(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2};
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  int err = numc_sigmoid(NULL, out);
  ASSERT_MSG(err == NUMC_ERR_NULL,
             "sigmoid(NULL, out) should return NUMC_ERR_NULL");
  numc_ctx_free(ctx);
  return 0;
}
static int test_sigmoid_type_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT64);
  int err = numc_sigmoid(a, out);
  ASSERT_MSG(err == NUMC_ERR_TYPE,
             "dtype mismatch should return NUMC_ERR_TYPE");
  numc_ctx_free(ctx);
  return 0;
}

static int test_sigmoid_shape_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sa[] = {2}, sb[] = {3};
  NumcArray *a = numc_array_create(ctx, sa, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, sb, 1, NUMC_DTYPE_FLOAT32);
  int err = numc_sigmoid(a, out);
  ASSERT_MSG(err == NUMC_ERR_SHAPE,
             "shape mismatch should return NUMC_ERR_SHAPE");
  numc_ctx_free(ctx);
  return 0;
}

static int test_sigmoid_inplace_null(void) {
  int err = numc_sigmoid_inplace(NULL);
  ASSERT_MSG(err == NUMC_ERR_NULL,
             "sigmoid_inplace(NULL) should return NUMC_ERR_NULL");
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== elemwise/test_sigmoid ===\n\n");

  printf("numc_sigmoid:");
  RUN_TEST(test_sigmoid_float32);
  RUN_TEST(test_sigmoid_float64);
  RUN_TEST(test_sigmoid_all_integer_dtypes_out);
  RUN_TEST(test_sigmoid_float32_bounds_monotonic_extremes);
  RUN_TEST(test_sigmoid_null);
  RUN_TEST(test_sigmoid_type_mismatch);
  RUN_TEST(test_sigmoid_shape_mismatch);
  printf("\nnumc_sigmoid_inplace:\n");
  RUN_TEST(test_sigmoid_inplace_matches_out_float32);
  RUN_TEST(test_sigmoid_inplace_null);

  printf("\nSummary: %d passed, %d failed\n", passes, fails);
  return fails ? 1 : 0;
}
