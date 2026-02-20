#include "../helpers.h"

/* --- numc_sum (full reduction to scalar) --- */

static int test_sum_1d_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {6};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  size_t sshape[] = {1};
  NumcArray *out = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  numc_array_write(a, da);

  int err = numc_sum(a, out);
  ASSERT_MSG(err == 0, "sum float32 should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 21.0f, "sum([1..6]) == 21");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sum_1d_int32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {5};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  size_t sshape[] = {1};
  NumcArray *out = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_INT32);

  int32_t da[] = {10, 20, 30, 40, 50};
  numc_array_write(a, da);

  int err = numc_sum(a, out);
  ASSERT_MSG(err == 0, "sum int32 should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 150, "sum([10,20,30,40,50]) == 150");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sum_2d_float64(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a   = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT64);
  size_t sshape[] = {1};
  NumcArray *out = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_FLOAT64);

  double da[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  numc_array_write(a, da);

  int err = numc_sum(a, out);
  ASSERT_MSG(err == 0, "sum float64 2d should succeed");

  double *r = (double *)numc_array_data(out);
  ASSERT_MSG(r[0] == 21.0, "sum 2x3 == 21");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sum_transposed(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  size_t sshape[] = {1};
  NumcArray *out = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  numc_array_write(a, da);

  /* Transpose makes it non-contiguous */
  size_t axes[] = {1, 0};
  numc_array_transpose(a, axes);
  ASSERT_MSG(!numc_array_is_contiguous(a), "transposed should be non-contiguous");

  int err = numc_sum(a, out);
  ASSERT_MSG(err == 0, "sum transposed should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 21.0f, "sum of transposed == 21 (same elements)");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sum_null(void) {
  int err = numc_sum(NULL, NULL);
  ASSERT_MSG(err == NUMC_ERR_NULL, "sum(NULL) should return NUMC_ERR_NULL");
  return 0;
}

static int test_sum_type_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  size_t sshape[] = {1};
  NumcArray *out = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_FLOAT32);

  int err = numc_sum(a, out);
  ASSERT_MSG(err == NUMC_ERR_TYPE, "sum type mismatch should return NUMC_ERR_TYPE");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sum_out_not_scalar(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);

  int err = numc_sum(a, out);
  ASSERT_MSG(err == NUMC_ERR_SHAPE, "sum with non-scalar out should return NUMC_ERR_SHAPE");

  numc_ctx_free(ctx);
  return 0;
}

/* --- numc_sum_axis --- */

static int test_sum_axis0_2d_float32(void) {
  /* [[1, 2, 3], [4, 5, 6]] -> sum axis=0 -> [5, 7, 9] */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  size_t oshape[] = {3};
  NumcArray *out = numc_array_zeros(ctx, oshape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  numc_array_write(a, da);

  int err = numc_sum_axis(a, 0, 0, out);
  ASSERT_MSG(err == 0, "sum_axis(0) should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 5.0f, "sum_axis0[0] == 5");
  ASSERT_MSG(r[1] == 7.0f, "sum_axis0[1] == 7");
  ASSERT_MSG(r[2] == 9.0f, "sum_axis0[2] == 9");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sum_axis1_2d_float32(void) {
  /* [[1, 2, 3], [4, 5, 6]] -> sum axis=1 -> [6, 15] */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  size_t oshape[] = {2};
  NumcArray *out = numc_array_zeros(ctx, oshape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  numc_array_write(a, da);

  int err = numc_sum_axis(a, 1, 0, out);
  ASSERT_MSG(err == 0, "sum_axis(1) should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 6.0f,  "sum_axis1[0] == 6");
  ASSERT_MSG(r[1] == 15.0f, "sum_axis1[1] == 15");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sum_axis_1d(void) {
  /* [10, 20, 30] -> sum axis=0 -> scalar 60 */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  size_t sshape[] = {1};
  NumcArray *out = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_INT32);

  int32_t da[] = {10, 20, 30};
  numc_array_write(a, da);

  int err = numc_sum_axis(a, 0, 0, out);
  ASSERT_MSG(err == 0, "sum_axis 1d should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 60, "sum_axis0 of [10,20,30] == 60");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sum_axis0_3d(void) {
  /* shape (2,2,3), sum axis=0 -> (2,3)
   * [[[1,2,3],[4,5,6]],
   *  [[7,8,9],[10,11,12]]]
   * axis=0 -> [[8,10,12],[14,16,18]] */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 3, NUMC_DTYPE_INT32);
  size_t oshape[] = {2, 3};
  NumcArray *out = numc_array_zeros(ctx, oshape, 2, NUMC_DTYPE_INT32);

  int32_t da[] = {1,2,3, 4,5,6, 7,8,9, 10,11,12};
  numc_array_write(a, da);

  int err = numc_sum_axis(a, 0, 0, out);
  ASSERT_MSG(err == 0, "sum_axis(0) 3d should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 8,  "r[0] == 1+7");
  ASSERT_MSG(r[1] == 10, "r[1] == 2+8");
  ASSERT_MSG(r[2] == 12, "r[2] == 3+9");
  ASSERT_MSG(r[3] == 14, "r[3] == 4+10");
  ASSERT_MSG(r[4] == 16, "r[4] == 5+11");
  ASSERT_MSG(r[5] == 18, "r[5] == 6+12");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sum_axis1_3d(void) {
  /* shape (2,2,3), sum axis=1 -> (2,3)
   * [[[1,2,3],[4,5,6]],
   *  [[7,8,9],[10,11,12]]]
   * axis=1 -> [[5,7,9],[17,19,21]] */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 3, NUMC_DTYPE_INT32);
  size_t oshape[] = {2, 3};
  NumcArray *out = numc_array_zeros(ctx, oshape, 2, NUMC_DTYPE_INT32);

  int32_t da[] = {1,2,3, 4,5,6, 7,8,9, 10,11,12};
  numc_array_write(a, da);

  int err = numc_sum_axis(a, 1, 0, out);
  ASSERT_MSG(err == 0, "sum_axis(1) 3d should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 5,  "r[0] == 1+4");
  ASSERT_MSG(r[1] == 7,  "r[1] == 2+5");
  ASSERT_MSG(r[2] == 9,  "r[2] == 3+6");
  ASSERT_MSG(r[3] == 17, "r[3] == 7+10");
  ASSERT_MSG(r[4] == 19, "r[4] == 8+11");
  ASSERT_MSG(r[5] == 21, "r[5] == 9+12");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sum_axis2_3d(void) {
  /* shape (2,2,3), sum axis=2 -> (2,2)
   * [[[1,2,3],[4,5,6]],
   *  [[7,8,9],[10,11,12]]]
   * axis=2 -> [[6,15],[24,33]] */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 3, NUMC_DTYPE_INT32);
  size_t oshape[] = {2, 2};
  NumcArray *out = numc_array_zeros(ctx, oshape, 2, NUMC_DTYPE_INT32);

  int32_t da[] = {1,2,3, 4,5,6, 7,8,9, 10,11,12};
  numc_array_write(a, da);

  int err = numc_sum_axis(a, 2, 0, out);
  ASSERT_MSG(err == 0, "sum_axis(2) 3d should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 6,  "r[0] == 1+2+3");
  ASSERT_MSG(r[1] == 15, "r[1] == 4+5+6");
  ASSERT_MSG(r[2] == 24, "r[2] == 7+8+9");
  ASSERT_MSG(r[3] == 33, "r[3] == 10+11+12");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sum_axis_keepdim(void) {
  /* [[1, 2, 3], [4, 5, 6]] -> sum axis=0, keepdim=1 -> [[5, 7, 9]] (1x3) */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  size_t oshape[] = {1, 3};
  NumcArray *out = numc_array_zeros(ctx, oshape, 2, NUMC_DTYPE_FLOAT32);

  float da[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  numc_array_write(a, da);

  int err = numc_sum_axis(a, 0, 1, out);
  ASSERT_MSG(err == 0, "sum_axis keepdim should succeed");

  ASSERT_MSG(numc_array_ndim(out) == 2, "keepdim output should be 2d");
  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 5.0f, "keepdim sum[0] == 5");
  ASSERT_MSG(r[1] == 7.0f, "keepdim sum[1] == 7");
  ASSERT_MSG(r[2] == 9.0f, "keepdim sum[2] == 9");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sum_axis_keepdim_axis1(void) {
  /* [[1, 2, 3], [4, 5, 6]] -> sum axis=1, keepdim=1 -> [[6], [15]] (2x1) */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  size_t oshape[] = {2, 1};
  NumcArray *out = numc_array_zeros(ctx, oshape, 2, NUMC_DTYPE_FLOAT32);

  float da[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  numc_array_write(a, da);

  int err = numc_sum_axis(a, 1, 1, out);
  ASSERT_MSG(err == 0, "sum_axis(1) keepdim should succeed");

  ASSERT_MSG(numc_array_ndim(out) == 2, "keepdim output should be 2d");
  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 6.0f,  "keepdim sum[0] == 6");
  ASSERT_MSG(r[1] == 15.0f, "keepdim sum[1] == 15");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sum_axis_transposed(void) {
  /* 2x3 transposed to 3x2, sum axis=0 -> [6, 15] */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);

  int32_t da[] = {1, 2, 3, 4, 5, 6};
  numc_array_write(a, da);

  /* Transpose: now shape is (3,2), data is [[1,4],[2,5],[3,6]] */
  size_t axes[] = {1, 0};
  numc_array_transpose(a, axes);

  size_t oshape[] = {2};
  NumcArray *out = numc_array_zeros(ctx, oshape, 1, NUMC_DTYPE_INT32);

  /* sum axis=0 of (3,2): [1+2+3, 4+5+6] = [6, 15] */
  int err = numc_sum_axis(a, 0, 0, out);
  ASSERT_MSG(err == 0, "sum_axis transposed should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 6,  "transposed sum_axis0[0] == 6");
  ASSERT_MSG(r[1] == 15, "transposed sum_axis0[1] == 15");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sum_axis_null(void) {
  int err = numc_sum_axis(NULL, 0, 0, NULL);
  ASSERT_MSG(err == NUMC_ERR_NULL, "sum_axis(NULL) should return NUMC_ERR_NULL");
  return 0;
}

static int test_sum_axis_type_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a   = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  size_t oshape[] = {3};
  NumcArray *out = numc_array_zeros(ctx, oshape, 1, NUMC_DTYPE_FLOAT32);

  int err = numc_sum_axis(a, 0, 0, out);
  ASSERT_MSG(err == NUMC_ERR_TYPE, "sum_axis type mismatch should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sum_axis_invalid(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  size_t oshape[] = {2};
  NumcArray *out = numc_array_zeros(ctx, oshape, 1, NUMC_DTYPE_INT32);

  int err = numc_sum_axis(a, 5, 0, out);
  ASSERT_MSG(err == NUMC_ERR_SHAPE, "sum_axis with invalid axis should fail");

  err = numc_sum_axis(a, -1, 0, out);
  ASSERT_MSG(err == NUMC_ERR_SHAPE, "sum_axis with negative axis should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sum_axis_shape_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  /* Wrong output shape: should be (3,) for axis=0, not (2,) */
  size_t oshape[] = {2};
  NumcArray *out = numc_array_zeros(ctx, oshape, 1, NUMC_DTYPE_INT32);

  int err = numc_sum_axis(a, 0, 0, out);
  ASSERT_MSG(err == NUMC_ERR_SHAPE, "sum_axis with wrong output shape should fail");

  numc_ctx_free(ctx);
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== reduction/test_sum ===\n\n");

  printf("numc_sum:\n");
  RUN_TEST(test_sum_1d_float32);
  RUN_TEST(test_sum_1d_int32);
  RUN_TEST(test_sum_2d_float64);
  RUN_TEST(test_sum_transposed);
  RUN_TEST(test_sum_null);
  RUN_TEST(test_sum_type_mismatch);
  RUN_TEST(test_sum_out_not_scalar);

  printf("\nnumc_sum_axis:\n");
  RUN_TEST(test_sum_axis0_2d_float32);
  RUN_TEST(test_sum_axis1_2d_float32);
  RUN_TEST(test_sum_axis_1d);
  RUN_TEST(test_sum_axis0_3d);
  RUN_TEST(test_sum_axis1_3d);
  RUN_TEST(test_sum_axis2_3d);
  RUN_TEST(test_sum_axis_keepdim);
  RUN_TEST(test_sum_axis_keepdim_axis1);
  RUN_TEST(test_sum_axis_transposed);
  RUN_TEST(test_sum_axis_null);
  RUN_TEST(test_sum_axis_type_mismatch);
  RUN_TEST(test_sum_axis_invalid);
  RUN_TEST(test_sum_axis_shape_mismatch);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
