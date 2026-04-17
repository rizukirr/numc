#include "../helpers.h"

#define FABSF_T(x) ((x) >= 0.0f ? (x) : -(x))
#define FABS_T(x)  ((x) >= 0.0 ? (x) : -(x))

#define TANH_EPS32 1e-5f
#define TANH_EPS64 1e-12

static int test_tanh_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  float da[] = {-1.0f, 0.0f, 1.0f};
  numc_array_write(a, da);
  int err = numc_tanh(a, out);
  ASSERT_MSG(err == 0, "tanh float32 should succeed");
  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(FABSF_T(r[0] - (-0.7615942f)) <= TANH_EPS32, "tanh(-1.0f)");
  ASSERT_MSG(r[1] == 0.0f, "tanh(0.0f) == 0.0f");
  ASSERT_MSG(FABSF_T(r[2] - 0.7615942f) <= TANH_EPS32, "tanh(1.0f)");
  numc_ctx_free(ctx);
  return 0;
}
static int test_tanh_float64(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT64);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT64);
  double da[] = {-1.0, 0.0, 1.0};
  numc_array_write(a, da);
  int err = numc_tanh(a, out);
  ASSERT_MSG(err == 0, "tanh float64 should succeed");
  double *r = (double *)numc_array_data(out);
  ASSERT_MSG(FABS_T(r[0] - (-0.7615941559557649)) <= TANH_EPS64, "tanh(-1.0)");
  ASSERT_MSG(r[1] == 0.0, "tanh(0.0) == 0.0");
  ASSERT_MSG(FABS_T(r[2] - 0.7615941559557649) <= TANH_EPS64, "tanh(1.0)");
  numc_ctx_free(ctx);
  return 0;
}
/* For integer outputs, tanh(x) is in (-1, 1), cast truncates to 0. */
static int test_tanh_all_integer_dtypes_out(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  {
    NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT8);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT8);
    int8_t da[] = {-1, 0, 1, 2};
    numc_array_write(a, da);
    ASSERT_MSG(numc_tanh(a, out) == 0, "tanh int8 should succeed");
    int8_t *r = (int8_t *)numc_array_data(out);
    ASSERT_MSG(r[0] == 0 && r[1] == 0 && r[2] == 0 && r[3] == 0,
               "tanh int8 cast result should be all zeros");
  }
  {
    NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT16);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT16);
    int16_t da[] = {-1, 0, 1, 2};
    numc_array_write(a, da);
    ASSERT_MSG(numc_tanh(a, out) == 0, "tanh int16 should succeed");
    int16_t *r = (int16_t *)numc_array_data(out);
    ASSERT_MSG(r[0] == 0 && r[1] == 0 && r[2] == 0 && r[3] == 0,
               "tanh int16 cast result should be all zeros");
  }
  {
    NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);
    int32_t da[] = {-1, 0, 1, 2};
    numc_array_write(a, da);
    ASSERT_MSG(numc_tanh(a, out) == 0, "tanh int32 should succeed");
    int32_t *r = (int32_t *)numc_array_data(out);
    ASSERT_MSG(r[0] == 0 && r[1] == 0 && r[2] == 0 && r[3] == 0,
               "tanh int32 cast result should be all zeros");
  }
  {
    NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT64);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT64);
    int64_t da[] = {-1, 0, 1, 2};
    numc_array_write(a, da);
    ASSERT_MSG(numc_tanh(a, out) == 0, "tanh int64 should succeed");
    int64_t *r = (int64_t *)numc_array_data(out);
    ASSERT_MSG(r[0] == 0 && r[1] == 0 && r[2] == 0 && r[3] == 0,
               "tanh int64 cast result should be all zeros");
  }
  {
    NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_UINT8);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_UINT8);
    uint8_t da[] = {0, 1, 2, 3};
    numc_array_write(a, da);
    ASSERT_MSG(numc_tanh(a, out) == 0, "tanh uint8 should succeed");
    uint8_t *r = (uint8_t *)numc_array_data(out);
    ASSERT_MSG(r[0] == 0 && r[1] == 0 && r[2] == 0 && r[3] == 0,
               "tanh uint8 cast result should be all zeros");
  }
  {
    NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_UINT16);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_UINT16);
    uint16_t da[] = {0, 1, 2, 3};
    numc_array_write(a, da);
    ASSERT_MSG(numc_tanh(a, out) == 0, "tanh uint16 should succeed");
    uint16_t *r = (uint16_t *)numc_array_data(out);
    ASSERT_MSG(r[0] == 0 && r[1] == 0 && r[2] == 0 && r[3] == 0,
               "tanh uint16 cast result should be all zeros");
  }
  {
    NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_UINT32);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_UINT32);
    uint32_t da[] = {0, 1, 2, 3};
    numc_array_write(a, da);
    ASSERT_MSG(numc_tanh(a, out) == 0, "tanh uint32 should succeed");
    uint32_t *r = (uint32_t *)numc_array_data(out);
    ASSERT_MSG(r[0] == 0 && r[1] == 0 && r[2] == 0 && r[3] == 0,
               "tanh uint32 cast result should be all zeros");
  }
  {
    NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_UINT64);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_UINT64);
    uint64_t da[] = {0, 1, 2, 3};
    numc_array_write(a, da);
    ASSERT_MSG(numc_tanh(a, out) == 0, "tanh uint64 should succeed");
    uint64_t *r = (uint64_t *)numc_array_data(out);
    ASSERT_MSG(r[0] == 0 && r[1] == 0 && r[2] == 0 && r[3] == 0,
               "tanh uint64 cast result should be all zeros");
  }
  numc_ctx_free(ctx);
  return 0;
}
static int test_tanh_inplace_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  float da[] = {-1.0f, 0.0f, 1.0f};
  numc_array_write(a, da);
  int err = numc_tanh_inplace(a);
  ASSERT_MSG(err == 0, "tanh_inplace float32 should succeed");
  float *r = (float *)numc_array_data(a);
  ASSERT_MSG(FABSF_T(r[0] - (-0.7615942f)) <= TANH_EPS32,
             "tanh_inplace(-1.0f)");
  ASSERT_MSG(r[1] == 0.0f, "tanh_inplace(0.0f) == 0.0f");
  ASSERT_MSG(FABSF_T(r[2] - 0.7615942f) <= TANH_EPS32, "tanh_inplace(1.0f)");
  numc_ctx_free(ctx);
  return 0;
}
static int test_tanh_inplace_all_integer_dtypes(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  {
    NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT8);
    int8_t da[] = {-1, 0, 1};
    numc_array_write(a, da);
    ASSERT_MSG(numc_tanh_inplace(a) == 0, "tanh_inplace int8 should succeed");
    int8_t *r = (int8_t *)numc_array_data(a);
    ASSERT_MSG(r[0] == 0 && r[1] == 0 && r[2] == 0,
               "tanh_inplace int8 cast result should be all zeros");
  }
  {
    NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT16);
    int16_t da[] = {-1, 0, 1};
    numc_array_write(a, da);
    ASSERT_MSG(numc_tanh_inplace(a) == 0, "tanh_inplace int16 should succeed");
    int16_t *r = (int16_t *)numc_array_data(a);
    ASSERT_MSG(r[0] == 0 && r[1] == 0 && r[2] == 0,
               "tanh_inplace int16 cast result should be all zeros");
  }
  {
    NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
    int32_t da[] = {-1, 0, 1};
    numc_array_write(a, da);
    ASSERT_MSG(numc_tanh_inplace(a) == 0, "tanh_inplace int32 should succeed");
    int32_t *r = (int32_t *)numc_array_data(a);
    ASSERT_MSG(r[0] == 0 && r[1] == 0 && r[2] == 0,
               "tanh_inplace int32 cast result should be all zeros");
  }
  {
    NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT64);
    int64_t da[] = {-1, 0, 1};
    numc_array_write(a, da);
    ASSERT_MSG(numc_tanh_inplace(a) == 0, "tanh_inplace int64 should succeed");
    int64_t *r = (int64_t *)numc_array_data(a);
    ASSERT_MSG(r[0] == 0 && r[1] == 0 && r[2] == 0,
               "tanh_inplace int64 cast result should be all zeros");
  }
  {
    NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_UINT8);
    uint8_t da[] = {0, 1, 2};
    numc_array_write(a, da);
    ASSERT_MSG(numc_tanh_inplace(a) == 0, "tanh_inplace uint8 should succeed");
    uint8_t *r = (uint8_t *)numc_array_data(a);
    ASSERT_MSG(r[0] == 0 && r[1] == 0 && r[2] == 0,
               "tanh_inplace uint8 cast result should be all zeros");
  }
  {
    NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_UINT16);
    uint16_t da[] = {0, 1, 2};
    numc_array_write(a, da);
    ASSERT_MSG(numc_tanh_inplace(a) == 0, "tanh_inplace uint16 should succeed");
    uint16_t *r = (uint16_t *)numc_array_data(a);
    ASSERT_MSG(r[0] == 0 && r[1] == 0 && r[2] == 0,
               "tanh_inplace uint16 cast result should be all zeros");
  }
  {
    NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_UINT32);
    uint32_t da[] = {0, 1, 2};
    numc_array_write(a, da);
    ASSERT_MSG(numc_tanh_inplace(a) == 0, "tanh_inplace uint32 should succeed");
    uint32_t *r = (uint32_t *)numc_array_data(a);
    ASSERT_MSG(r[0] == 0 && r[1] == 0 && r[2] == 0,
               "tanh_inplace uint32 cast result should be all zeros");
  }
  {
    NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_UINT64);
    uint64_t da[] = {0, 1, 2};
    numc_array_write(a, da);
    ASSERT_MSG(numc_tanh_inplace(a) == 0, "tanh_inplace uint64 should succeed");
    uint64_t *r = (uint64_t *)numc_array_data(a);
    ASSERT_MSG(r[0] == 0 && r[1] == 0 && r[2] == 0,
               "tanh_inplace uint64 cast result should be all zeros");
  }
  numc_ctx_free(ctx);
  return 0;
}
static int test_tanh_null(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2};
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  int err = numc_tanh(NULL, out);
  ASSERT_MSG(err == NUMC_ERR_NULL,
             "tanh(NULL, out) should return NUMC_ERR_NULL");
  numc_ctx_free(ctx);
  return 0;
}
static int test_tanh_float32_symmetry_saturation(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {-10.0f, -1.0f, 1.0f, 10.0f};
  numc_array_write(a, da);

  int err = numc_tanh(a, out);
  ASSERT_MSG(err == 0, "tanh symmetry saturation should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] < -0.99f, "tanh(-10) should saturate near -1");
  ASSERT_MSG(r[3] > 0.99f, "tanh(10) should saturate near +1");
  ASSERT_MSG(FABSF_T(r[1] + r[2]) <= TANH_EPS32,
             "tanh(-x) + tanh(x) ~= 0 (odd symmetry)");
  numc_ctx_free(ctx);
  return 0;
  return 0;
}
static int test_tanh_inplace_matches_out_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {5};

  NumcArray *a_out = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *a_inp = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {-3.0f, -0.5f, 0.0f, 0.5f, 3.0f};
  numc_array_write(a_out, da);

  numc_array_write(a_inp, da);
  ASSERT_MSG(numc_tanh(a_out, out) == 0, "tanh out-of-place should succeed");
  ASSERT_MSG(numc_tanh_inplace(a_inp) == 0, "tanh inplace should succeed");

  float *ro = (float *)numc_array_data(out);
  float *ri = (float *)numc_array_data(a_inp);
  for (size_t i = 0; i < 5; i++) {
    ASSERT_MSG(FABSF_T(ro[i] - ri[i]) <= TANH_EPS32,
               "tanh inplace must match out-of-place");
  }
  numc_ctx_free(ctx);
  return 0;
}
static int test_tanh_type_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT64);
  int err = numc_tanh(a, out);
  ASSERT_MSG(err == NUMC_ERR_TYPE,
             "dtype mismatch should return NUMC_ERR_TYPE");
  numc_ctx_free(ctx);
  return 0;
}
static int test_tanh_shape_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sa[] = {2}, sb[] = {3};
  NumcArray *a = numc_array_create(ctx, sa, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, sb, 1, NUMC_DTYPE_FLOAT32);
  int err = numc_tanh(a, out);
  ASSERT_MSG(err == NUMC_ERR_SHAPE,
             "shape mismatch should return NUMC_ERR_SHAPE");
  numc_ctx_free(ctx);
  return 0;
}
static int test_tanh_inplace_null(void) {
  int err = numc_tanh_inplace(NULL);
  ASSERT_MSG(err == NUMC_ERR_NULL,
             "tanh_inplace(NULL) should return NUMC_ERR_NULL");
  return 0;
}
int main(void) {
  int passes = 0, fails = 0;
  printf("=== elemwise/test_tanh ===\n\n");
  printf("numc_tanh:\n");
  RUN_TEST(test_tanh_float32);
  RUN_TEST(test_tanh_float64);
  RUN_TEST(test_tanh_all_integer_dtypes_out);
  RUN_TEST(test_tanh_float32_symmetry_saturation);
  RUN_TEST(test_tanh_null);
  RUN_TEST(test_tanh_type_mismatch);
  RUN_TEST(test_tanh_shape_mismatch);
  printf("\nnumc_tanh_inplace:\n");
  RUN_TEST(test_tanh_inplace_float32);
  RUN_TEST(test_tanh_inplace_all_integer_dtypes);
  RUN_TEST(test_tanh_inplace_matches_out_float32);
  RUN_TEST(test_tanh_inplace_null);
  printf("\nSummary: %d passed, %d failed\n", passes, fails);
  return fails ? 1 : 0;
}
