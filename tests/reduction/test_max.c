#include "../helpers.h"

/* --- numc_max (full reduction to scalar) --- */

static int test_max_1d_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {6};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  size_t sshape[] = {1};
  NumcArray *out = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {3.0f, 1.0f, 5.0f, 2.0f, 6.0f, 4.0f};
  numc_array_write(a, da);

  int err = numc_max(a, out);
  ASSERT_MSG(err == 0, "max float32 should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 6.0f, "max([3,1,5,2,6,4]) == 6");

  numc_ctx_free(ctx);
  return 0;
}

static int test_max_1d_int32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {5};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  size_t sshape[] = {1};
  NumcArray *out = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_INT32);

  int32_t da[] = {-10, 50, 20, -30, 40};
  numc_array_write(a, da);

  int err = numc_max(a, out);
  ASSERT_MSG(err == 0, "max int32 should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 50, "max([-10,50,20,-30,40]) == 50");

  numc_ctx_free(ctx);
  return 0;
}

static int test_max_1d_int8(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT8);
  size_t sshape[] = {1};
  NumcArray *out = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_INT8);

  int8_t da[] = {-128, 0, 127, -1};
  numc_array_write(a, da);

  int err = numc_max(a, out);
  ASSERT_MSG(err == 0, "max int8 should succeed");

  int8_t *r = (int8_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 127, "max([-128,0,127,-1]) == 127");

  numc_ctx_free(ctx);
  return 0;
}

static int test_max_2d_float64(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a   = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT64);
  size_t sshape[] = {1};
  NumcArray *out = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_FLOAT64);

  double da[] = {1.5, -2.5, 3.5, -4.5, 5.5, -6.5};
  numc_array_write(a, da);

  int err = numc_max(a, out);
  ASSERT_MSG(err == 0, "max float64 2d should succeed");

  double *r = (double *)numc_array_data(out);
  ASSERT_MSG(r[0] == 5.5, "max 2x3 == 5.5");

  numc_ctx_free(ctx);
  return 0;
}

static int test_max_negative_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  size_t sshape[] = {1};
  NumcArray *out = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {-5.0f, -2.0f, -8.0f, -1.0f};
  numc_array_write(a, da);

  int err = numc_max(a, out);
  ASSERT_MSG(err == 0, "max all-negative should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == -1.0f, "max([-5,-2,-8,-1]) == -1");

  numc_ctx_free(ctx);
  return 0;
}

static int test_max_transposed(void) {
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

  int err = numc_max(a, out);
  ASSERT_MSG(err == 0, "max transposed should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 6.0f, "max of transposed == 6 (same elements)");

  numc_ctx_free(ctx);
  return 0;
}

static int test_max_null(void) {
  int err = numc_max(NULL, NULL);
  ASSERT_MSG(err == NUMC_ERR_NULL, "max(NULL) should return NUMC_ERR_NULL");
  return 0;
}

static int test_max_type_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  size_t sshape[] = {1};
  NumcArray *out = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_FLOAT32);

  int err = numc_max(a, out);
  ASSERT_MSG(err == NUMC_ERR_TYPE, "max type mismatch should return NUMC_ERR_TYPE");

  numc_ctx_free(ctx);
  return 0;
}

static int test_max_out_not_scalar(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);

  int err = numc_max(a, out);
  ASSERT_MSG(err == NUMC_ERR_SHAPE, "max with non-scalar out should return NUMC_ERR_SHAPE");

  numc_ctx_free(ctx);
  return 0;
}

/* --- numc_max_axis --- */

static int test_max_axis0_2d_float32(void) {
  /* [[1, 5, 3], [4, 2, 6]] -> max axis=0 -> [4, 5, 6] */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  size_t oshape[] = {3};
  NumcArray *out = numc_array_zeros(ctx, oshape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {1.0f, 5.0f, 3.0f, 4.0f, 2.0f, 6.0f};
  numc_array_write(a, da);

  int err = numc_max_axis(a, 0, 0, out);
  ASSERT_MSG(err == 0, "max_axis(0) should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 4.0f, "max_axis0[0] == max(1,4) = 4");
  ASSERT_MSG(r[1] == 5.0f, "max_axis0[1] == max(5,2) = 5");
  ASSERT_MSG(r[2] == 6.0f, "max_axis0[2] == max(3,6) = 6");

  numc_ctx_free(ctx);
  return 0;
}

static int test_max_axis1_2d_float32(void) {
  /* [[1, 5, 3], [4, 2, 6]] -> max axis=1 -> [5, 6] */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  size_t oshape[] = {2};
  NumcArray *out = numc_array_zeros(ctx, oshape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {1.0f, 5.0f, 3.0f, 4.0f, 2.0f, 6.0f};
  numc_array_write(a, da);

  int err = numc_max_axis(a, 1, 0, out);
  ASSERT_MSG(err == 0, "max_axis(1) should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 5.0f, "max_axis1[0] == max(1,5,3) = 5");
  ASSERT_MSG(r[1] == 6.0f, "max_axis1[1] == max(4,2,6) = 6");

  numc_ctx_free(ctx);
  return 0;
}

static int test_max_axis_1d(void) {
  /* [30, 10, 20] -> max axis=0 -> scalar 30 */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  size_t sshape[] = {1};
  NumcArray *out = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_INT32);

  int32_t da[] = {30, 10, 20};
  numc_array_write(a, da);

  int err = numc_max_axis(a, 0, 0, out);
  ASSERT_MSG(err == 0, "max_axis 1d should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 30, "max_axis0 of [30,10,20] == 30");

  numc_ctx_free(ctx);
  return 0;
}

static int test_max_axis0_3d(void) {
  /* shape (2,2,3):
   * [[[1,2,3],[4,5,6]],
   *  [[7,8,9],[10,11,12]]]
   * max axis=0 -> [[7,8,9],[10,11,12]] */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 3, NUMC_DTYPE_INT32);
  size_t oshape[] = {2, 3};
  NumcArray *out = numc_array_zeros(ctx, oshape, 2, NUMC_DTYPE_INT32);

  int32_t da[] = {1,2,3, 4,5,6, 7,8,9, 10,11,12};
  numc_array_write(a, da);

  int err = numc_max_axis(a, 0, 0, out);
  ASSERT_MSG(err == 0, "max_axis(0) 3d should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 7,  "r[0] == max(1,7)");
  ASSERT_MSG(r[1] == 8,  "r[1] == max(2,8)");
  ASSERT_MSG(r[2] == 9,  "r[2] == max(3,9)");
  ASSERT_MSG(r[3] == 10, "r[3] == max(4,10)");
  ASSERT_MSG(r[4] == 11, "r[4] == max(5,11)");
  ASSERT_MSG(r[5] == 12, "r[5] == max(6,12)");

  numc_ctx_free(ctx);
  return 0;
}

static int test_max_axis2_3d(void) {
  /* shape (2,2,3):
   * [[[1,2,3],[4,5,6]],
   *  [[7,8,9],[10,11,12]]]
   * max axis=2 -> [[3,6],[9,12]] */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 3, NUMC_DTYPE_INT32);
  size_t oshape[] = {2, 2};
  NumcArray *out = numc_array_zeros(ctx, oshape, 2, NUMC_DTYPE_INT32);

  int32_t da[] = {1,2,3, 4,5,6, 7,8,9, 10,11,12};
  numc_array_write(a, da);

  int err = numc_max_axis(a, 2, 0, out);
  ASSERT_MSG(err == 0, "max_axis(2) 3d should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 3,  "r[0] == max(1,2,3)");
  ASSERT_MSG(r[1] == 6,  "r[1] == max(4,5,6)");
  ASSERT_MSG(r[2] == 9,  "r[2] == max(7,8,9)");
  ASSERT_MSG(r[3] == 12, "r[3] == max(10,11,12)");

  numc_ctx_free(ctx);
  return 0;
}

static int test_max_axis_keepdim(void) {
  /* [[1, 5, 3], [4, 2, 6]] -> max axis=0, keepdim=1 -> [[4, 5, 6]] (1x3) */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  size_t oshape[] = {1, 3};
  NumcArray *out = numc_array_zeros(ctx, oshape, 2, NUMC_DTYPE_FLOAT32);

  float da[] = {1.0f, 5.0f, 3.0f, 4.0f, 2.0f, 6.0f};
  numc_array_write(a, da);

  int err = numc_max_axis(a, 0, 1, out);
  ASSERT_MSG(err == 0, "max_axis keepdim should succeed");

  ASSERT_MSG(numc_array_ndim(out) == 2, "keepdim output should be 2d");
  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 4.0f, "keepdim max[0] == 4");
  ASSERT_MSG(r[1] == 5.0f, "keepdim max[1] == 5");
  ASSERT_MSG(r[2] == 6.0f, "keepdim max[2] == 6");

  numc_ctx_free(ctx);
  return 0;
}

static int test_max_axis_transposed(void) {
  /* 2x3 transposed to 3x2, max axis=0
   * Original: [[1,2,3],[4,5,6]]
   * Transposed (3x2): [[1,4],[2,5],[3,6]]
   * max axis=0: [3, 6] */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);

  int32_t da[] = {1, 2, 3, 4, 5, 6};
  numc_array_write(a, da);

  size_t axes[] = {1, 0};
  numc_array_transpose(a, axes);

  size_t oshape[] = {2};
  NumcArray *out = numc_array_zeros(ctx, oshape, 1, NUMC_DTYPE_INT32);

  int err = numc_max_axis(a, 0, 0, out);
  ASSERT_MSG(err == 0, "max_axis transposed should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 3, "transposed max_axis0[0] == 3");
  ASSERT_MSG(r[1] == 6, "transposed max_axis0[1] == 6");

  numc_ctx_free(ctx);
  return 0;
}

static int test_max_axis_null(void) {
  int err = numc_max_axis(NULL, 0, 0, NULL);
  ASSERT_MSG(err == NUMC_ERR_NULL, "max_axis(NULL) should return NUMC_ERR_NULL");
  return 0;
}

static int test_max_axis_type_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a   = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  size_t oshape[] = {3};
  NumcArray *out = numc_array_zeros(ctx, oshape, 1, NUMC_DTYPE_FLOAT32);

  int err = numc_max_axis(a, 0, 0, out);
  ASSERT_MSG(err == NUMC_ERR_TYPE, "max_axis type mismatch should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_max_axis_invalid(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  size_t oshape[] = {2};
  NumcArray *out = numc_array_zeros(ctx, oshape, 1, NUMC_DTYPE_INT32);

  int err = numc_max_axis(a, 5, 0, out);
  ASSERT_MSG(err == NUMC_ERR_SHAPE, "max_axis with invalid axis should fail");

  err = numc_max_axis(a, -1, 0, out);
  ASSERT_MSG(err == NUMC_ERR_SHAPE, "max_axis with negative axis should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_max_axis_shape_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  size_t oshape[] = {2};
  NumcArray *out = numc_array_zeros(ctx, oshape, 1, NUMC_DTYPE_INT32);

  int err = numc_max_axis(a, 0, 0, out);
  ASSERT_MSG(err == NUMC_ERR_SHAPE, "max_axis with wrong output shape should fail");

  numc_ctx_free(ctx);
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== reduction/test_max ===\n\n");

  printf("numc_max:\n");
  RUN_TEST(test_max_1d_float32);
  RUN_TEST(test_max_1d_int32);
  RUN_TEST(test_max_1d_int8);
  RUN_TEST(test_max_2d_float64);
  RUN_TEST(test_max_negative_float32);
  RUN_TEST(test_max_transposed);
  RUN_TEST(test_max_null);
  RUN_TEST(test_max_type_mismatch);
  RUN_TEST(test_max_out_not_scalar);

  printf("\nnumc_max_axis:\n");
  RUN_TEST(test_max_axis0_2d_float32);
  RUN_TEST(test_max_axis1_2d_float32);
  RUN_TEST(test_max_axis_1d);
  RUN_TEST(test_max_axis0_3d);
  RUN_TEST(test_max_axis2_3d);
  RUN_TEST(test_max_axis_keepdim);
  RUN_TEST(test_max_axis_transposed);
  RUN_TEST(test_max_axis_null);
  RUN_TEST(test_max_axis_type_mismatch);
  RUN_TEST(test_max_axis_invalid);
  RUN_TEST(test_max_axis_shape_mismatch);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
