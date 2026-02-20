#include "../helpers.h"

static int test_clip_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {6};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {-1.0f, 0.0f, 2.5f, 5.0f, 7.5f, 10.0f};
  numc_array_write(a, da);

  int err = numc_clip(a, out, 0.0, 5.0);
  ASSERT_MSG(err == 0, "clip float32 should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 0.0f, "clip(-1.0, 0, 5) == 0.0");
  ASSERT_MSG(r[1] == 0.0f, "clip(0.0, 0, 5)  == 0.0");
  ASSERT_MSG(r[2] == 2.5f, "clip(2.5, 0, 5)  == 2.5");
  ASSERT_MSG(r[3] == 5.0f, "clip(5.0, 0, 5)  == 5.0");
  ASSERT_MSG(r[4] == 5.0f, "clip(7.5, 0, 5)  == 5.0");
  ASSERT_MSG(r[5] == 5.0f, "clip(10.0, 0, 5) == 5.0");

  numc_ctx_free(ctx);
  return 0;
}

static int test_clip_float64(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT64);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT64);

  double da[] = {-100.0, -0.5, 0.5, 100.0};
  numc_array_write(a, da);

  int err = numc_clip(a, out, -1.0, 1.0);
  ASSERT_MSG(err == 0, "clip float64 should succeed");

  double *r = (double *)numc_array_data(out);
  ASSERT_MSG(r[0] == -1.0, "clip(-100, -1, 1) == -1.0");
  ASSERT_MSG(r[1] == -0.5, "clip(-0.5, -1, 1) == -0.5");
  ASSERT_MSG(r[2] == 0.5,  "clip(0.5, -1, 1)  ==  0.5");
  ASSERT_MSG(r[3] == 1.0,  "clip(100, -1, 1)  ==  1.0");

  numc_ctx_free(ctx);
  return 0;
}

static int test_clip_int32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {5};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);

  int32_t da[] = {-50, -5, 0, 5, 50};
  numc_array_write(a, da);

  int err = numc_clip(a, out, -10.0, 10.0);
  ASSERT_MSG(err == 0, "clip int32 should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == -10, "clip(-50, -10, 10) == -10");
  ASSERT_MSG(r[1] == -5,  "clip(-5, -10, 10)  == -5");
  ASSERT_MSG(r[2] == 0,   "clip(0, -10, 10)   == 0");
  ASSERT_MSG(r[3] == 5,   "clip(5, -10, 10)   == 5");
  ASSERT_MSG(r[4] == 10,  "clip(50, -10, 10)  == 10");

  numc_ctx_free(ctx);
  return 0;
}

static int test_clip_int8(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT8);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT8);

  int8_t da[] = {-100, -1, 1, 100};
  numc_array_write(a, da);

  int err = numc_clip(a, out, -10.0, 10.0);
  ASSERT_MSG(err == 0, "clip int8 should succeed");

  int8_t *r = (int8_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == -10, "clip(-100, -10, 10) == -10");
  ASSERT_MSG(r[1] == -1,  "clip(-1, -10, 10)   == -1");
  ASSERT_MSG(r[2] == 1,   "clip(1, -10, 10)    == 1");
  ASSERT_MSG(r[3] == 10,  "clip(100, -10, 10)  == 10");

  numc_ctx_free(ctx);
  return 0;
}

static int test_clip_2d(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a   = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_INT32);

  int32_t da[] = {-50, -5, 0, 5, 50, 100};
  numc_array_write(a, da);

  int err = numc_clip(a, out, -10.0, 10.0);
  ASSERT_MSG(err == 0, "clip 2d should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == -10, "2d clip [0][0]");
  ASSERT_MSG(r[1] == -5,  "2d clip [0][1]");
  ASSERT_MSG(r[2] == 0,   "2d clip [0][2]");
  ASSERT_MSG(r[3] == 5,   "2d clip [1][0]");
  ASSERT_MSG(r[4] == 10,  "2d clip [1][1]");
  ASSERT_MSG(r[5] == 10,  "2d clip [1][2]");

  numc_ctx_free(ctx);
  return 0;
}

static int test_clip_all_within(void) {
  /* All values already within [min, max] — should pass through unchanged */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {1.0f, 2.0f, 3.0f, 4.0f};
  numc_array_write(a, da);

  int err = numc_clip(a, out, 0.0, 5.0);
  ASSERT_MSG(err == 0, "clip all-within should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1.0f, "passthrough 1.0");
  ASSERT_MSG(r[1] == 2.0f, "passthrough 2.0");
  ASSERT_MSG(r[2] == 3.0f, "passthrough 3.0");
  ASSERT_MSG(r[3] == 4.0f, "passthrough 4.0");

  numc_ctx_free(ctx);
  return 0;
}

static int test_clip_null(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  int err = numc_clip(NULL, out, 0.0, 1.0);
  ASSERT_MSG(err == NUMC_ERR_NULL,
             "clip(NULL, out) should return NUMC_ERR_NULL");

  numc_ctx_free(ctx);
  return 0;
}

static int test_clip_type_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT64);

  int err = numc_clip(a, out, 0.0, 1.0);
  ASSERT_MSG(err == NUMC_ERR_TYPE,
             "dtype mismatch should return NUMC_ERR_TYPE");

  numc_ctx_free(ctx);
  return 0;
}

static int test_clip_shape_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sa[] = {4}, sb[] = {6};
  NumcArray *a   = numc_array_create(ctx, sa, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, sb, 1, NUMC_DTYPE_FLOAT32);

  int err = numc_clip(a, out, 0.0, 1.0);
  ASSERT_MSG(err == NUMC_ERR_SHAPE,
             "shape mismatch should return NUMC_ERR_SHAPE");

  numc_ctx_free(ctx);
  return 0;
}

static int test_clip_inplace_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {-1.0f, 1.5f, 3.5f, 10.0f};
  numc_array_write(a, da);

  int err = numc_clip_inplace(a, 0.0, 3.0);
  ASSERT_MSG(err == 0, "clip_inplace float32 should succeed");

  float *r = (float *)numc_array_data(a);
  ASSERT_MSG(r[0] == 0.0f, "clip_inplace(-1.0, 0, 3) == 0.0");
  ASSERT_MSG(r[1] == 1.5f, "clip_inplace(1.5, 0, 3)  == 1.5");
  ASSERT_MSG(r[2] == 3.0f, "clip_inplace(3.5, 0, 3)  == 3.0");
  ASSERT_MSG(r[3] == 3.0f, "clip_inplace(10.0, 0, 3) == 3.0");

  numc_ctx_free(ctx);
  return 0;
}

static int test_clip_inplace_int32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);

  int32_t da[] = {-50, 0, 5, 50};
  numc_array_write(a, da);

  int err = numc_clip_inplace(a, -10.0, 10.0);
  ASSERT_MSG(err == 0, "clip_inplace int32 should succeed");

  int32_t *r = (int32_t *)numc_array_data(a);
  ASSERT_MSG(r[0] == -10, "clip_inplace(-50, -10, 10) == -10");
  ASSERT_MSG(r[1] == 0,   "clip_inplace(0, -10, 10)   == 0");
  ASSERT_MSG(r[2] == 5,   "clip_inplace(5, -10, 10)   == 5");
  ASSERT_MSG(r[3] == 10,  "clip_inplace(50, -10, 10)  == 10");

  numc_ctx_free(ctx);
  return 0;
}

static int test_clip_inplace_null(void) {
  int err = numc_clip_inplace(NULL, 0.0, 1.0);
  ASSERT_MSG(err == NUMC_ERR_NULL,
             "clip_inplace(NULL) should return NUMC_ERR_NULL");
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== elemwise/test_clip ===\n\n");

  printf("numc_clip:\n");
  RUN_TEST(test_clip_float32);
  RUN_TEST(test_clip_float64);
  RUN_TEST(test_clip_int32);
  RUN_TEST(test_clip_int8);
  RUN_TEST(test_clip_2d);
  RUN_TEST(test_clip_all_within);
  RUN_TEST(test_clip_null);
  RUN_TEST(test_clip_type_mismatch);
  RUN_TEST(test_clip_shape_mismatch);

  printf("\nnumc_clip_inplace:\n");
  RUN_TEST(test_clip_inplace_float32);
  RUN_TEST(test_clip_inplace_int32);
  RUN_TEST(test_clip_inplace_null);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
