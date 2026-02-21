#include "../helpers.h"

/* --- numc_argmin (full reduction to scalar) --- */

static int test_argmin_1d_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {6};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  size_t sshape[] = {1};
  NumcArray *out = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_INT64);

  float da[] = {3.0f, 1.0f, 5.0f, 2.0f, 6.0f, 4.0f};
  numc_array_write(a, da);

  int err = numc_argmin(a, out);
  ASSERT_MSG(err == 0, "argmin float32 should succeed");

  int64_t *r = (int64_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1, "argmin([3,1,5,2,6,4]) == 1");

  numc_ctx_free(ctx);
  return 0;
}

static int test_argmin_1d_int32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {5};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  size_t sshape[] = {1};
  NumcArray *out = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_INT64);

  int32_t da[] = {-10, 50, 20, -30, 40};
  numc_array_write(a, da);

  int err = numc_argmin(a, out);
  ASSERT_MSG(err == 0, "argmin int32 should succeed");

  int64_t *r = (int64_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 3, "argmin([-10,50,20,-30,40]) == 3");

  numc_ctx_free(ctx);
  return 0;
}

static int test_argmin_1d_int8(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT8);
  size_t sshape[] = {1};
  NumcArray *out = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_INT64);

  int8_t da[] = {-128, 0, 127, -1};
  numc_array_write(a, da);

  int err = numc_argmin(a, out);
  ASSERT_MSG(err == 0, "argmin int8 should succeed");

  int64_t *r = (int64_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 0, "argmin([-128,0,127,-1]) == 0");

  numc_ctx_free(ctx);
  return 0;
}

static int test_argmin_2d_float64(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT64);
  size_t sshape[] = {1};
  NumcArray *out = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_INT64);

  double da[] = {1.5, -2.5, 3.5, -4.5, 5.5, -6.5};
  numc_array_write(a, da);

  int err = numc_argmin(a, out);
  ASSERT_MSG(err == 0, "argmin float64 2d should succeed");

  int64_t *r = (int64_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 5, "argmin 2x3 == 5 (element -6.5)");

  numc_ctx_free(ctx);
  return 0;
}

static int test_argmin_negative_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  size_t sshape[] = {1};
  NumcArray *out = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_INT64);

  float da[] = {-5.0f, -2.0f, -8.0f, -1.0f};
  numc_array_write(a, da);

  int err = numc_argmin(a, out);
  ASSERT_MSG(err == 0, "argmin all-negative should succeed");

  int64_t *r = (int64_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 2, "argmin([-5,-2,-8,-1]) == 2");

  numc_ctx_free(ctx);
  return 0;
}

static int test_argmin_transposed(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  size_t sshape[] = {1};
  NumcArray *out = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_INT64);

  /* Original: [[1,2,3],[4,5,6]] -> transposed (3x2): [[1,4],[2,5],[3,6]]
   * Flat traversal of transposed: 1,4,2,5,3,6 -> argmin = 0 (element 1) */
  float da[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  numc_array_write(a, da);

  size_t axes[] = {1, 0};
  numc_array_transpose(a, axes);
  ASSERT_MSG(!numc_array_is_contiguous(a),
             "transposed should be non-contiguous");

  int err = numc_argmin(a, out);
  ASSERT_MSG(err == 0, "argmin transposed should succeed");

  int64_t *r = (int64_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 0, "argmin of transposed == 0 (element 1)");

  numc_ctx_free(ctx);
  return 0;
}

static int test_argmin_null(void) {
  int err = numc_argmin(NULL, NULL);
  ASSERT_MSG(err == NUMC_ERR_NULL,
             "argmin(NULL) should return NUMC_ERR_NULL");
  return 0;
}

static int test_argmin_out_not_int64(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  size_t sshape[] = {1};
  NumcArray *out = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_FLOAT32);

  int err = numc_argmin(a, out);
  ASSERT_MSG(err == NUMC_ERR_TYPE,
             "argmin with non-INT64 out should return NUMC_ERR_TYPE");

  numc_ctx_free(ctx);
  return 0;
}

static int test_argmin_out_not_scalar(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT64);

  int err = numc_argmin(a, out);
  ASSERT_MSG(err == NUMC_ERR_SHAPE,
             "argmin with non-scalar out should return NUMC_ERR_SHAPE");

  numc_ctx_free(ctx);
  return 0;
}

/* --- numc_argmin_axis --- */

static int test_argmin_axis0_2d_float32(void) {
  /* [[1, 5, 3], [4, 2, 6]] -> argmin axis=0 -> [0, 1, 0] */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  size_t oshape[] = {3};
  NumcArray *out = numc_array_zeros(ctx, oshape, 1, NUMC_DTYPE_INT64);

  float da[] = {1.0f, 5.0f, 3.0f, 4.0f, 2.0f, 6.0f};
  numc_array_write(a, da);

  int err = numc_argmin_axis(a, 0, 0, out);
  ASSERT_MSG(err == 0, "argmin_axis(0) should succeed");

  int64_t *r = (int64_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 0, "argmin_axis0[0] == 0 (1 < 4)");
  ASSERT_MSG(r[1] == 1, "argmin_axis0[1] == 1 (2 < 5)");
  ASSERT_MSG(r[2] == 0, "argmin_axis0[2] == 0 (3 < 6)");

  numc_ctx_free(ctx);
  return 0;
}

static int test_argmin_axis1_2d_float32(void) {
  /* [[1, 5, 3], [4, 2, 6]] -> argmin axis=1 -> [0, 1] */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  size_t oshape[] = {2};
  NumcArray *out = numc_array_zeros(ctx, oshape, 1, NUMC_DTYPE_INT64);

  float da[] = {1.0f, 5.0f, 3.0f, 4.0f, 2.0f, 6.0f};
  numc_array_write(a, da);

  int err = numc_argmin_axis(a, 1, 0, out);
  ASSERT_MSG(err == 0, "argmin_axis(1) should succeed");

  int64_t *r = (int64_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 0, "argmin_axis1[0] == 0 (1 is min of [1,5,3])");
  ASSERT_MSG(r[1] == 1, "argmin_axis1[1] == 1 (2 is min of [4,2,6])");

  numc_ctx_free(ctx);
  return 0;
}

static int test_argmin_axis_1d(void) {
  /* [30, 10, 20] -> argmin axis=0 -> scalar 1 */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  size_t sshape[] = {1};
  NumcArray *out = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_INT64);

  int32_t da[] = {30, 10, 20};
  numc_array_write(a, da);

  int err = numc_argmin_axis(a, 0, 0, out);
  ASSERT_MSG(err == 0, "argmin_axis 1d should succeed");

  int64_t *r = (int64_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1, "argmin_axis0 of [30,10,20] == 1");

  numc_ctx_free(ctx);
  return 0;
}

static int test_argmin_axis0_3d(void) {
  /* shape (2,2,3):
   * [[[1,2,3],[4,5,6]],
   *  [[7,8,9],[10,11,12]]]
   * argmin axis=0 -> [[0,0,0],[0,0,0]] (all from first slice) */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 3, NUMC_DTYPE_INT32);
  size_t oshape[] = {2, 3};
  NumcArray *out = numc_array_zeros(ctx, oshape, 2, NUMC_DTYPE_INT64);

  int32_t da[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  numc_array_write(a, da);

  int err = numc_argmin_axis(a, 0, 0, out);
  ASSERT_MSG(err == 0, "argmin_axis(0) 3d should succeed");

  int64_t *r = (int64_t *)numc_array_data(out);
  for (int i = 0; i < 6; i++)
    ASSERT_MSG(r[i] == 0, "all argmin along axis=0 should be 0");

  numc_ctx_free(ctx);
  return 0;
}

static int test_argmin_axis2_3d(void) {
  /* shape (2,2,3):
   * [[[1,2,3],[4,5,6]],
   *  [[7,8,9],[10,11,12]]]
   * argmin axis=2 -> [[0,0],[0,0]] (first element of each row) */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 3, NUMC_DTYPE_INT32);
  size_t oshape[] = {2, 2};
  NumcArray *out = numc_array_zeros(ctx, oshape, 2, NUMC_DTYPE_INT64);

  int32_t da[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  numc_array_write(a, da);

  int err = numc_argmin_axis(a, 2, 0, out);
  ASSERT_MSG(err == 0, "argmin_axis(2) 3d should succeed");

  int64_t *r = (int64_t *)numc_array_data(out);
  for (int i = 0; i < 4; i++)
    ASSERT_MSG(r[i] == 0, "all argmin along axis=2 should be 0");

  numc_ctx_free(ctx);
  return 0;
}

static int test_argmin_axis_keepdim(void) {
  /* [[1, 5, 3], [4, 2, 6]] -> argmin axis=0 keepdim -> [[0, 1, 0]] (1x3) */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  size_t oshape[] = {1, 3};
  NumcArray *out = numc_array_zeros(ctx, oshape, 2, NUMC_DTYPE_INT64);

  float da[] = {1.0f, 5.0f, 3.0f, 4.0f, 2.0f, 6.0f};
  numc_array_write(a, da);

  int err = numc_argmin_axis(a, 0, 1, out);
  ASSERT_MSG(err == 0, "argmin_axis keepdim should succeed");

  ASSERT_MSG(numc_array_ndim(out) == 2, "keepdim output should be 2d");
  int64_t *r = (int64_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 0, "keepdim argmin[0] == 0");
  ASSERT_MSG(r[1] == 1, "keepdim argmin[1] == 1");
  ASSERT_MSG(r[2] == 0, "keepdim argmin[2] == 0");

  numc_ctx_free(ctx);
  return 0;
}

static int test_argmin_axis_transposed(void) {
  /* 2x3 transposed to 3x2, argmin axis=0
   * Original: [[1,2,3],[4,5,6]]
   * Transposed (3x2): [[1,4],[2,5],[3,6]]
   * argmin axis=0: [0, 0] (index 0 = row [1,4]) */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);

  int32_t da[] = {1, 2, 3, 4, 5, 6};
  numc_array_write(a, da);

  size_t axes[] = {1, 0};
  numc_array_transpose(a, axes);

  size_t oshape[] = {2};
  NumcArray *out = numc_array_zeros(ctx, oshape, 1, NUMC_DTYPE_INT64);

  int err = numc_argmin_axis(a, 0, 0, out);
  ASSERT_MSG(err == 0, "argmin_axis transposed should succeed");

  int64_t *r = (int64_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 0, "transposed argmin_axis0[0] == 0");
  ASSERT_MSG(r[1] == 0, "transposed argmin_axis0[1] == 0");

  numc_ctx_free(ctx);
  return 0;
}

static int test_argmin_axis_null(void) {
  int err = numc_argmin_axis(NULL, 0, 0, NULL);
  ASSERT_MSG(err == NUMC_ERR_NULL,
             "argmin_axis(NULL) should return NUMC_ERR_NULL");
  return 0;
}

static int test_argmin_axis_out_not_int64(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  size_t oshape[] = {3};
  NumcArray *out = numc_array_zeros(ctx, oshape, 1, NUMC_DTYPE_INT32);

  int err = numc_argmin_axis(a, 0, 0, out);
  ASSERT_MSG(err == NUMC_ERR_TYPE,
             "argmin_axis with non-INT64 out should return NUMC_ERR_TYPE");

  numc_ctx_free(ctx);
  return 0;
}

static int test_argmin_axis_invalid(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  size_t oshape[] = {2};
  NumcArray *out = numc_array_zeros(ctx, oshape, 1, NUMC_DTYPE_INT64);

  int err = numc_argmin_axis(a, 5, 0, out);
  ASSERT_MSG(err == NUMC_ERR_SHAPE,
             "argmin_axis with invalid axis should fail");

  err = numc_argmin_axis(a, -1, 0, out);
  ASSERT_MSG(err == NUMC_ERR_SHAPE,
             "argmin_axis with negative axis should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_argmin_axis_shape_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  size_t oshape[] = {2};
  NumcArray *out = numc_array_zeros(ctx, oshape, 1, NUMC_DTYPE_INT64);

  int err = numc_argmin_axis(a, 0, 0, out);
  ASSERT_MSG(err == NUMC_ERR_SHAPE,
             "argmin_axis with wrong output shape should fail");

  numc_ctx_free(ctx);
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== reduction/test_argmin ===\n\n");

  printf("numc_argmin:\n");
  RUN_TEST(test_argmin_1d_float32);
  RUN_TEST(test_argmin_1d_int32);
  RUN_TEST(test_argmin_1d_int8);
  RUN_TEST(test_argmin_2d_float64);
  RUN_TEST(test_argmin_negative_float32);
  RUN_TEST(test_argmin_transposed);
  RUN_TEST(test_argmin_null);
  RUN_TEST(test_argmin_out_not_int64);
  RUN_TEST(test_argmin_out_not_scalar);

  printf("\nnumc_argmin_axis:\n");
  RUN_TEST(test_argmin_axis0_2d_float32);
  RUN_TEST(test_argmin_axis1_2d_float32);
  RUN_TEST(test_argmin_axis_1d);
  RUN_TEST(test_argmin_axis0_3d);
  RUN_TEST(test_argmin_axis2_3d);
  RUN_TEST(test_argmin_axis_keepdim);
  RUN_TEST(test_argmin_axis_transposed);
  RUN_TEST(test_argmin_axis_null);
  RUN_TEST(test_argmin_axis_out_not_int64);
  RUN_TEST(test_argmin_axis_invalid);
  RUN_TEST(test_argmin_axis_shape_mismatch);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
