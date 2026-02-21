#include "../helpers.h"

/* --- numc_mean (full reduction to scalar) --- */

static int test_mean_1d_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {6};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  size_t sshape[] = {1};
  NumcArray *out = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  numc_array_write(a, da);

  int err = numc_mean(a, out);
  ASSERT_MSG(err == 0, "mean float32 should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 3.5f, "mean([1..6]) == 3.5");

  numc_ctx_free(ctx);
  return 0;
}

static int test_mean_1d_int32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {5};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  size_t sshape[] = {1};
  NumcArray *out = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_INT32);

  int32_t da[] = {10, 20, 30, 40, 50};
  numc_array_write(a, da);

  int err = numc_mean(a, out);
  ASSERT_MSG(err == 0, "mean int32 should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 30, "mean([10,20,30,40,50]) == 30");

  numc_ctx_free(ctx);
  return 0;
}

static int test_mean_1d_int32_truncates(void) {
  /* 150 / 4 = 37.5 -> truncates to 37 */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  size_t sshape[] = {1};
  NumcArray *out = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_INT32);

  int32_t da[] = {10, 20, 50, 70};
  numc_array_write(a, da);

  int err = numc_mean(a, out);
  ASSERT_MSG(err == 0, "mean int32 truncating should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 37, "mean([10,20,50,70]) == 37 (truncated)");

  numc_ctx_free(ctx);
  return 0;
}

static int test_mean_2d_float64(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a   = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT64);
  size_t sshape[] = {1};
  NumcArray *out = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_FLOAT64);

  double da[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  numc_array_write(a, da);

  int err = numc_mean(a, out);
  ASSERT_MSG(err == 0, "mean float64 2d should succeed");

  double *r = (double *)numc_array_data(out);
  ASSERT_MSG(r[0] == 3.5, "mean 2x3 == 3.5");

  numc_ctx_free(ctx);
  return 0;
}

static int test_mean_transposed(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  size_t sshape[] = {1};
  NumcArray *out = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  numc_array_write(a, da);

  size_t axes[] = {1, 0};
  numc_array_transpose(a, axes);
  ASSERT_MSG(!numc_array_is_contiguous(a), "transposed should be non-contiguous");

  int err = numc_mean(a, out);
  ASSERT_MSG(err == 0, "mean transposed should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 3.5f, "mean of transposed == 3.5 (same elements)");

  numc_ctx_free(ctx);
  return 0;
}

static int test_mean_null(void) {
  int err = numc_mean(NULL, NULL);
  ASSERT_MSG(err == NUMC_ERR_NULL, "mean(NULL) should return NUMC_ERR_NULL");
  return 0;
}

static int test_mean_type_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  size_t sshape[] = {1};
  NumcArray *out = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_FLOAT32);

  int err = numc_mean(a, out);
  ASSERT_MSG(err == NUMC_ERR_TYPE, "mean type mismatch should return NUMC_ERR_TYPE");

  numc_ctx_free(ctx);
  return 0;
}

static int test_mean_out_not_scalar(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);

  int err = numc_mean(a, out);
  ASSERT_MSG(err == NUMC_ERR_SHAPE, "mean with non-scalar out should return NUMC_ERR_SHAPE");

  numc_ctx_free(ctx);
  return 0;
}

/* --- numc_mean_axis --- */

static int test_mean_axis0_2d_float32(void) {
  /* [[1, 2, 3], [4, 5, 6]] -> mean axis=0 -> [2.5, 3.5, 4.5] */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  size_t oshape[] = {3};
  NumcArray *out = numc_array_zeros(ctx, oshape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  numc_array_write(a, da);

  int err = numc_mean_axis(a, 0, 0, out);
  ASSERT_MSG(err == 0, "mean_axis(0) should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 2.5f, "mean_axis0[0] == 2.5");
  ASSERT_MSG(r[1] == 3.5f, "mean_axis0[1] == 3.5");
  ASSERT_MSG(r[2] == 4.5f, "mean_axis0[2] == 4.5");

  numc_ctx_free(ctx);
  return 0;
}

static int test_mean_axis1_2d_float32(void) {
  /* [[1, 2, 3], [4, 5, 6]] -> mean axis=1 -> [2, 5] */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  size_t oshape[] = {2};
  NumcArray *out = numc_array_zeros(ctx, oshape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  numc_array_write(a, da);

  int err = numc_mean_axis(a, 1, 0, out);
  ASSERT_MSG(err == 0, "mean_axis(1) should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 2.0f, "mean_axis1[0] == 2");
  ASSERT_MSG(r[1] == 5.0f, "mean_axis1[1] == 5");

  numc_ctx_free(ctx);
  return 0;
}

static int test_mean_axis_1d(void) {
  /* [10, 20, 30] -> mean axis=0 -> scalar 20 */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  size_t sshape[] = {1};
  NumcArray *out = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_INT32);

  int32_t da[] = {10, 20, 30};
  numc_array_write(a, da);

  int err = numc_mean_axis(a, 0, 0, out);
  ASSERT_MSG(err == 0, "mean_axis 1d should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 20, "mean_axis0 of [10,20,30] == 20");

  numc_ctx_free(ctx);
  return 0;
}

static int test_mean_axis0_3d(void) {
  /* shape (2,2,3), mean axis=0 -> (2,3)
   * [[[1,2,3],[4,5,6]],
   *  [[7,8,9],[10,11,12]]]
   * axis=0 -> [[4,5,6],[7,8,9]] */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 3, NUMC_DTYPE_FLOAT32);
  size_t oshape[] = {2, 3};
  NumcArray *out = numc_array_zeros(ctx, oshape, 2, NUMC_DTYPE_FLOAT32);

  float da[] = {1,2,3, 4,5,6, 7,8,9, 10,11,12};
  numc_array_write(a, da);

  int err = numc_mean_axis(a, 0, 0, out);
  ASSERT_MSG(err == 0, "mean_axis(0) 3d should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 4.0f,  "r[0] == (1+7)/2");
  ASSERT_MSG(r[1] == 5.0f,  "r[1] == (2+8)/2");
  ASSERT_MSG(r[2] == 6.0f,  "r[2] == (3+9)/2");
  ASSERT_MSG(r[3] == 7.0f,  "r[3] == (4+10)/2");
  ASSERT_MSG(r[4] == 8.0f,  "r[4] == (5+11)/2");
  ASSERT_MSG(r[5] == 9.0f,  "r[5] == (6+12)/2");

  numc_ctx_free(ctx);
  return 0;
}

static int test_mean_axis2_3d(void) {
  /* shape (2,2,3), mean axis=2 -> (2,2)
   * [[[1,2,3],[4,5,6]],
   *  [[7,8,9],[10,11,12]]]
   * axis=2 -> [[2,5],[8,11]] */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 3, NUMC_DTYPE_FLOAT32);
  size_t oshape[] = {2, 2};
  NumcArray *out = numc_array_zeros(ctx, oshape, 2, NUMC_DTYPE_FLOAT32);

  float da[] = {1,2,3, 4,5,6, 7,8,9, 10,11,12};
  numc_array_write(a, da);

  int err = numc_mean_axis(a, 2, 0, out);
  ASSERT_MSG(err == 0, "mean_axis(2) 3d should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 2.0f,  "r[0] == (1+2+3)/3");
  ASSERT_MSG(r[1] == 5.0f,  "r[1] == (4+5+6)/3");
  ASSERT_MSG(r[2] == 8.0f,  "r[2] == (7+8+9)/3");
  ASSERT_MSG(r[3] == 11.0f, "r[3] == (10+11+12)/3");

  numc_ctx_free(ctx);
  return 0;
}

static int test_mean_axis_keepdim(void) {
  /* [[1, 2, 3], [4, 5, 6]] -> mean axis=0, keepdim=1 -> [[2.5, 3.5, 4.5]] (1x3) */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  size_t oshape[] = {1, 3};
  NumcArray *out = numc_array_zeros(ctx, oshape, 2, NUMC_DTYPE_FLOAT32);

  float da[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  numc_array_write(a, da);

  int err = numc_mean_axis(a, 0, 1, out);
  ASSERT_MSG(err == 0, "mean_axis keepdim should succeed");

  ASSERT_MSG(numc_array_ndim(out) == 2, "keepdim output should be 2d");
  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 2.5f, "keepdim mean[0] == 2.5");
  ASSERT_MSG(r[1] == 3.5f, "keepdim mean[1] == 3.5");
  ASSERT_MSG(r[2] == 4.5f, "keepdim mean[2] == 4.5");

  numc_ctx_free(ctx);
  return 0;
}

static int test_mean_axis_keepdim_axis1(void) {
  /* [[1, 2, 3], [4, 5, 6]] -> mean axis=1, keepdim=1 -> [[2], [5]] (2x1) */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  size_t oshape[] = {2, 1};
  NumcArray *out = numc_array_zeros(ctx, oshape, 2, NUMC_DTYPE_FLOAT32);

  float da[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  numc_array_write(a, da);

  int err = numc_mean_axis(a, 1, 1, out);
  ASSERT_MSG(err == 0, "mean_axis(1) keepdim should succeed");

  ASSERT_MSG(numc_array_ndim(out) == 2, "keepdim output should be 2d");
  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 2.0f, "keepdim mean[0] == 2");
  ASSERT_MSG(r[1] == 5.0f, "keepdim mean[1] == 5");

  numc_ctx_free(ctx);
  return 0;
}

static int test_mean_axis_transposed(void) {
  /* 2x3 transposed to 3x2, mean axis=0 -> [2, 5]
   * Original: [[1,2,3],[4,5,6]]
   * Transposed (3x2): [[1,4],[2,5],[3,6]]
   * mean axis=0: [(1+2+3)/3, (4+5+6)/3] = [2, 5] */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);

  float da[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  numc_array_write(a, da);

  size_t axes[] = {1, 0};
  numc_array_transpose(a, axes);

  size_t oshape[] = {2};
  NumcArray *out = numc_array_zeros(ctx, oshape, 1, NUMC_DTYPE_FLOAT32);

  int err = numc_mean_axis(a, 0, 0, out);
  ASSERT_MSG(err == 0, "mean_axis transposed should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 2.0f, "transposed mean_axis0[0] == 2");
  ASSERT_MSG(r[1] == 5.0f, "transposed mean_axis0[1] == 5");

  numc_ctx_free(ctx);
  return 0;
}

static int test_mean_axis_int32_truncates(void) {
  /* [[1, 2, 3], [4, 5, 6]] -> mean axis=0 -> [2, 3, 4] (truncated)
   * (1+4)/2=2, (2+5)/2=3, (3+6)/2=4 */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  size_t oshape[] = {3};
  NumcArray *out = numc_array_zeros(ctx, oshape, 1, NUMC_DTYPE_INT32);

  int32_t da[] = {1, 2, 3, 4, 5, 6};
  numc_array_write(a, da);

  int err = numc_mean_axis(a, 0, 0, out);
  ASSERT_MSG(err == 0, "mean_axis int32 should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 2, "mean_axis0[0] == (1+4)/2 = 2 (truncated)");
  ASSERT_MSG(r[1] == 3, "mean_axis0[1] == (2+5)/2 = 3 (truncated)");
  ASSERT_MSG(r[2] == 4, "mean_axis0[2] == (3+6)/2 = 4 (truncated)");

  numc_ctx_free(ctx);
  return 0;
}

static int test_mean_axis_null(void) {
  int err = numc_mean_axis(NULL, 0, 0, NULL);
  ASSERT_MSG(err == NUMC_ERR_NULL, "mean_axis(NULL) should return NUMC_ERR_NULL");
  return 0;
}

static int test_mean_axis_type_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a   = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  size_t oshape[] = {3};
  NumcArray *out = numc_array_zeros(ctx, oshape, 1, NUMC_DTYPE_FLOAT32);

  int err = numc_mean_axis(a, 0, 0, out);
  ASSERT_MSG(err == NUMC_ERR_TYPE, "mean_axis type mismatch should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_mean_axis_invalid(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  size_t oshape[] = {2};
  NumcArray *out = numc_array_zeros(ctx, oshape, 1, NUMC_DTYPE_INT32);

  int err = numc_mean_axis(a, 5, 0, out);
  ASSERT_MSG(err == NUMC_ERR_SHAPE, "mean_axis with invalid axis should fail");

  err = numc_mean_axis(a, -1, 0, out);
  ASSERT_MSG(err == NUMC_ERR_SHAPE, "mean_axis with negative axis should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_mean_axis_shape_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  size_t oshape[] = {2};
  NumcArray *out = numc_array_zeros(ctx, oshape, 1, NUMC_DTYPE_INT32);

  int err = numc_mean_axis(a, 0, 0, out);
  ASSERT_MSG(err == NUMC_ERR_SHAPE, "mean_axis with wrong output shape should fail");

  numc_ctx_free(ctx);
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== reduction/test_mean ===\n\n");

  printf("numc_mean:\n");
  RUN_TEST(test_mean_1d_float32);
  RUN_TEST(test_mean_1d_int32);
  RUN_TEST(test_mean_1d_int32_truncates);
  RUN_TEST(test_mean_2d_float64);
  RUN_TEST(test_mean_transposed);
  RUN_TEST(test_mean_null);
  RUN_TEST(test_mean_type_mismatch);
  RUN_TEST(test_mean_out_not_scalar);

  printf("\nnumc_mean_axis:\n");
  RUN_TEST(test_mean_axis0_2d_float32);
  RUN_TEST(test_mean_axis1_2d_float32);
  RUN_TEST(test_mean_axis_1d);
  RUN_TEST(test_mean_axis0_3d);
  RUN_TEST(test_mean_axis2_3d);
  RUN_TEST(test_mean_axis_keepdim);
  RUN_TEST(test_mean_axis_keepdim_axis1);
  RUN_TEST(test_mean_axis_transposed);
  RUN_TEST(test_mean_axis_int32_truncates);
  RUN_TEST(test_mean_axis_null);
  RUN_TEST(test_mean_axis_type_mismatch);
  RUN_TEST(test_mean_axis_invalid);
  RUN_TEST(test_mean_axis_shape_mismatch);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
