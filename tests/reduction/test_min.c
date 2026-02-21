#include "../helpers.h"

/* --- numc_min (full reduction to scalar) --- */

static int test_min_1d_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {6};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  size_t sshape[] = {1};
  NumcArray *out = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {3.0f, 1.0f, 5.0f, 2.0f, 6.0f, 4.0f};
  numc_array_write(a, da);

  int err = numc_min(a, out);
  ASSERT_MSG(err == 0, "min float32 should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1.0f, "min([3,1,5,2,6,4]) == 1");

  numc_ctx_free(ctx);
  return 0;
}

static int test_min_1d_int32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {5};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  size_t sshape[] = {1};
  NumcArray *out = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_INT32);

  int32_t da[] = {-10, 50, 20, -30, 40};
  numc_array_write(a, da);

  int err = numc_min(a, out);
  ASSERT_MSG(err == 0, "min int32 should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == -30, "min([-10,50,20,-30,40]) == -30");

  numc_ctx_free(ctx);
  return 0;
}

static int test_min_1d_uint8(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_UINT8);
  size_t sshape[] = {1};
  NumcArray *out = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_UINT8);

  uint8_t da[] = {255, 42, 0, 100};
  numc_array_write(a, da);

  int err = numc_min(a, out);
  ASSERT_MSG(err == 0, "min uint8 should succeed");

  uint8_t *r = (uint8_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 0, "min([255,42,0,100]) == 0");

  numc_ctx_free(ctx);
  return 0;
}

static int test_min_2d_float64(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a   = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT64);
  size_t sshape[] = {1};
  NumcArray *out = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_FLOAT64);

  double da[] = {1.5, -2.5, 3.5, -4.5, 5.5, -6.5};
  numc_array_write(a, da);

  int err = numc_min(a, out);
  ASSERT_MSG(err == 0, "min float64 2d should succeed");

  double *r = (double *)numc_array_data(out);
  ASSERT_MSG(r[0] == -6.5, "min 2x3 == -6.5");

  numc_ctx_free(ctx);
  return 0;
}

static int test_min_positive_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  size_t sshape[] = {1};
  NumcArray *out = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {5.0f, 2.0f, 8.0f, 1.0f};
  numc_array_write(a, da);

  int err = numc_min(a, out);
  ASSERT_MSG(err == 0, "min all-positive should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1.0f, "min([5,2,8,1]) == 1");

  numc_ctx_free(ctx);
  return 0;
}

static int test_min_transposed(void) {
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

  int err = numc_min(a, out);
  ASSERT_MSG(err == 0, "min transposed should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1.0f, "min of transposed == 1 (same elements)");

  numc_ctx_free(ctx);
  return 0;
}

static int test_min_null(void) {
  int err = numc_min(NULL, NULL);
  ASSERT_MSG(err == NUMC_ERR_NULL, "min(NULL) should return NUMC_ERR_NULL");
  return 0;
}

static int test_min_type_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  size_t sshape[] = {1};
  NumcArray *out = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_FLOAT32);

  int err = numc_min(a, out);
  ASSERT_MSG(err == NUMC_ERR_TYPE, "min type mismatch should return NUMC_ERR_TYPE");

  numc_ctx_free(ctx);
  return 0;
}

static int test_min_out_not_scalar(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);

  int err = numc_min(a, out);
  ASSERT_MSG(err == NUMC_ERR_SHAPE, "min with non-scalar out should return NUMC_ERR_SHAPE");

  numc_ctx_free(ctx);
  return 0;
}

/* --- numc_min_axis --- */

static int test_min_axis0_2d_float32(void) {
  /* [[1, 5, 3], [4, 2, 6]] -> min axis=0 -> [1, 2, 3] */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  size_t oshape[] = {3};
  NumcArray *out = numc_array_zeros(ctx, oshape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {1.0f, 5.0f, 3.0f, 4.0f, 2.0f, 6.0f};
  numc_array_write(a, da);

  int err = numc_min_axis(a, 0, 0, out);
  ASSERT_MSG(err == 0, "min_axis(0) should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1.0f, "min_axis0[0] == min(1,4) = 1");
  ASSERT_MSG(r[1] == 2.0f, "min_axis0[1] == min(5,2) = 2");
  ASSERT_MSG(r[2] == 3.0f, "min_axis0[2] == min(3,6) = 3");

  numc_ctx_free(ctx);
  return 0;
}

static int test_min_axis1_2d_float32(void) {
  /* [[1, 5, 3], [4, 2, 6]] -> min axis=1 -> [1, 2] */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  size_t oshape[] = {2};
  NumcArray *out = numc_array_zeros(ctx, oshape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {1.0f, 5.0f, 3.0f, 4.0f, 2.0f, 6.0f};
  numc_array_write(a, da);

  int err = numc_min_axis(a, 1, 0, out);
  ASSERT_MSG(err == 0, "min_axis(1) should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1.0f, "min_axis1[0] == min(1,5,3) = 1");
  ASSERT_MSG(r[1] == 2.0f, "min_axis1[1] == min(4,2,6) = 2");

  numc_ctx_free(ctx);
  return 0;
}

static int test_min_axis_1d(void) {
  /* [30, 10, 20] -> min axis=0 -> scalar 10 */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  size_t sshape[] = {1};
  NumcArray *out = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_INT32);

  int32_t da[] = {30, 10, 20};
  numc_array_write(a, da);

  int err = numc_min_axis(a, 0, 0, out);
  ASSERT_MSG(err == 0, "min_axis 1d should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 10, "min_axis0 of [30,10,20] == 10");

  numc_ctx_free(ctx);
  return 0;
}

static int test_min_axis0_3d(void) {
  /* shape (2,2,3):
   * [[[1,2,3],[4,5,6]],
   *  [[7,8,9],[10,11,12]]]
   * min axis=0 -> [[1,2,3],[4,5,6]] */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 3, NUMC_DTYPE_INT32);
  size_t oshape[] = {2, 3};
  NumcArray *out = numc_array_zeros(ctx, oshape, 2, NUMC_DTYPE_INT32);

  int32_t da[] = {1,2,3, 4,5,6, 7,8,9, 10,11,12};
  numc_array_write(a, da);

  int err = numc_min_axis(a, 0, 0, out);
  ASSERT_MSG(err == 0, "min_axis(0) 3d should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1, "r[0] == min(1,7)");
  ASSERT_MSG(r[1] == 2, "r[1] == min(2,8)");
  ASSERT_MSG(r[2] == 3, "r[2] == min(3,9)");
  ASSERT_MSG(r[3] == 4, "r[3] == min(4,10)");
  ASSERT_MSG(r[4] == 5, "r[4] == min(5,11)");
  ASSERT_MSG(r[5] == 6, "r[5] == min(6,12)");

  numc_ctx_free(ctx);
  return 0;
}

static int test_min_axis2_3d(void) {
  /* shape (2,2,3):
   * [[[1,2,3],[4,5,6]],
   *  [[7,8,9],[10,11,12]]]
   * min axis=2 -> [[1,4],[7,10]] */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 3, NUMC_DTYPE_INT32);
  size_t oshape[] = {2, 2};
  NumcArray *out = numc_array_zeros(ctx, oshape, 2, NUMC_DTYPE_INT32);

  int32_t da[] = {1,2,3, 4,5,6, 7,8,9, 10,11,12};
  numc_array_write(a, da);

  int err = numc_min_axis(a, 2, 0, out);
  ASSERT_MSG(err == 0, "min_axis(2) 3d should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1,  "r[0] == min(1,2,3)");
  ASSERT_MSG(r[1] == 4,  "r[1] == min(4,5,6)");
  ASSERT_MSG(r[2] == 7,  "r[2] == min(7,8,9)");
  ASSERT_MSG(r[3] == 10, "r[3] == min(10,11,12)");

  numc_ctx_free(ctx);
  return 0;
}

static int test_min_axis_keepdim(void) {
  /* [[1, 5, 3], [4, 2, 6]] -> min axis=0, keepdim=1 -> [[1, 2, 3]] (1x3) */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  size_t oshape[] = {1, 3};
  NumcArray *out = numc_array_zeros(ctx, oshape, 2, NUMC_DTYPE_FLOAT32);

  float da[] = {1.0f, 5.0f, 3.0f, 4.0f, 2.0f, 6.0f};
  numc_array_write(a, da);

  int err = numc_min_axis(a, 0, 1, out);
  ASSERT_MSG(err == 0, "min_axis keepdim should succeed");

  ASSERT_MSG(numc_array_ndim(out) == 2, "keepdim output should be 2d");
  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1.0f, "keepdim min[0] == 1");
  ASSERT_MSG(r[1] == 2.0f, "keepdim min[1] == 2");
  ASSERT_MSG(r[2] == 3.0f, "keepdim min[2] == 3");

  numc_ctx_free(ctx);
  return 0;
}

static int test_min_axis_transposed(void) {
  /* 2x3 transposed to 3x2, min axis=0
   * Original: [[1,2,3],[4,5,6]]
   * Transposed (3x2): [[1,4],[2,5],[3,6]]
   * min axis=0: [1, 4] */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);

  int32_t da[] = {1, 2, 3, 4, 5, 6};
  numc_array_write(a, da);

  size_t axes[] = {1, 0};
  numc_array_transpose(a, axes);

  size_t oshape[] = {2};
  NumcArray *out = numc_array_zeros(ctx, oshape, 1, NUMC_DTYPE_INT32);

  int err = numc_min_axis(a, 0, 0, out);
  ASSERT_MSG(err == 0, "min_axis transposed should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1, "transposed min_axis0[0] == 1");
  ASSERT_MSG(r[1] == 4, "transposed min_axis0[1] == 4");

  numc_ctx_free(ctx);
  return 0;
}

static int test_min_axis_null(void) {
  int err = numc_min_axis(NULL, 0, 0, NULL);
  ASSERT_MSG(err == NUMC_ERR_NULL, "min_axis(NULL) should return NUMC_ERR_NULL");
  return 0;
}

static int test_min_axis_type_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a   = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  size_t oshape[] = {3};
  NumcArray *out = numc_array_zeros(ctx, oshape, 1, NUMC_DTYPE_FLOAT32);

  int err = numc_min_axis(a, 0, 0, out);
  ASSERT_MSG(err == NUMC_ERR_TYPE, "min_axis type mismatch should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_min_axis_invalid(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  size_t oshape[] = {2};
  NumcArray *out = numc_array_zeros(ctx, oshape, 1, NUMC_DTYPE_INT32);

  int err = numc_min_axis(a, 5, 0, out);
  ASSERT_MSG(err == NUMC_ERR_SHAPE, "min_axis with invalid axis should fail");

  err = numc_min_axis(a, -1, 0, out);
  ASSERT_MSG(err == NUMC_ERR_SHAPE, "min_axis with negative axis should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_min_axis_shape_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  size_t oshape[] = {2};
  NumcArray *out = numc_array_zeros(ctx, oshape, 1, NUMC_DTYPE_INT32);

  int err = numc_min_axis(a, 0, 0, out);
  ASSERT_MSG(err == NUMC_ERR_SHAPE, "min_axis with wrong output shape should fail");

  numc_ctx_free(ctx);
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== reduction/test_min ===\n\n");

  printf("numc_min:\n");
  RUN_TEST(test_min_1d_float32);
  RUN_TEST(test_min_1d_int32);
  RUN_TEST(test_min_1d_uint8);
  RUN_TEST(test_min_2d_float64);
  RUN_TEST(test_min_positive_float32);
  RUN_TEST(test_min_transposed);
  RUN_TEST(test_min_null);
  RUN_TEST(test_min_type_mismatch);
  RUN_TEST(test_min_out_not_scalar);

  printf("\nnumc_min_axis:\n");
  RUN_TEST(test_min_axis0_2d_float32);
  RUN_TEST(test_min_axis1_2d_float32);
  RUN_TEST(test_min_axis_1d);
  RUN_TEST(test_min_axis0_3d);
  RUN_TEST(test_min_axis2_3d);
  RUN_TEST(test_min_axis_keepdim);
  RUN_TEST(test_min_axis_transposed);
  RUN_TEST(test_min_axis_null);
  RUN_TEST(test_min_axis_type_mismatch);
  RUN_TEST(test_min_axis_invalid);
  RUN_TEST(test_min_axis_shape_mismatch);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
