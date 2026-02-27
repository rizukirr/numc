#include "../helpers.h"

/* All happy-path tests use (2,3) x (3,2) -> (2,2):
 *
 * a = [[1, 2, 3],   b = [[ 7,  8],    out = [[ 58,  64],
 *      [4, 5, 6]]        [ 9, 10],           [139, 154]]
 *                        [11, 12]]
 */

static int test_matmul_naive_int8(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape_a[] = {2, 3};
  size_t shape_b[] = {3, 2};
  size_t shape_c[] = {2, 2};
  NumcArray *a = numc_array_create(ctx, shape_a, 2, NUMC_DTYPE_INT8);
  NumcArray *b = numc_array_create(ctx, shape_b, 2, NUMC_DTYPE_INT8);
  NumcArray *c = numc_array_zeros(ctx, shape_c, 2, NUMC_DTYPE_INT8);

  /* Use small values so results fit in int8 range [-128, 127]:
   * out = [[22, 28], [49, 64]] */
  int8_t da[][3] = {{1, 2, 3}, {4, 5, 6}};
  int8_t db[][2] = {{1, 2}, {3, 4}, {5, 6}};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_matmul_naive(a, b, c);
  ASSERT_MSG(err == 0, "matmul_naive int8 should succeed");

  int8_t *r = (int8_t *)numc_array_data(c);
  ASSERT_MSG(r[0] == 22 && r[1] == 28 && r[2] == 49 && r[3] == 64,
             "matmul_naive int8 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_matmul_naive_int16(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape_a[] = {2, 3};
  size_t shape_b[] = {3, 2};
  size_t shape_c[] = {2, 2};
  NumcArray *a = numc_array_create(ctx, shape_a, 2, NUMC_DTYPE_INT16);
  NumcArray *b = numc_array_create(ctx, shape_b, 2, NUMC_DTYPE_INT16);
  NumcArray *c = numc_array_zeros(ctx, shape_c, 2, NUMC_DTYPE_INT16);

  int16_t da[][3] = {{1, 2, 3}, {4, 5, 6}};
  int16_t db[][2] = {{7, 8}, {9, 10}, {11, 12}};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_matmul_naive(a, b, c);
  ASSERT_MSG(err == 0, "matmul_naive int16 should succeed");

  int16_t *r = (int16_t *)numc_array_data(c);
  ASSERT_MSG(r[0] == 58 && r[1] == 64 && r[2] == 139 && r[3] == 154,
             "matmul_naive int16 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_matmul_naive_int32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape_a[] = {2, 3};
  size_t shape_b[] = {3, 2};
  size_t shape_c[] = {2, 2};
  NumcArray *a = numc_array_create(ctx, shape_a, 2, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_create(ctx, shape_b, 2, NUMC_DTYPE_INT32);
  NumcArray *c = numc_array_zeros(ctx, shape_c, 2, NUMC_DTYPE_INT32);

  int32_t da[][3] = {{1, 2, 3}, {4, 5, 6}};
  int32_t db[][2] = {{7, 8}, {9, 10}, {11, 12}};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_matmul_naive(a, b, c);
  ASSERT_MSG(err == 0, "matmul_naive int32 should succeed");

  int32_t *r = (int32_t *)numc_array_data(c);
  ASSERT_MSG(r[0] == 58 && r[1] == 64 && r[2] == 139 && r[3] == 154,
             "matmul_naive int32 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_matmul_naive_int64(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape_a[] = {2, 3};
  size_t shape_b[] = {3, 2};
  size_t shape_c[] = {2, 2};
  NumcArray *a = numc_array_create(ctx, shape_a, 2, NUMC_DTYPE_INT64);
  NumcArray *b = numc_array_create(ctx, shape_b, 2, NUMC_DTYPE_INT64);
  NumcArray *c = numc_array_zeros(ctx, shape_c, 2, NUMC_DTYPE_INT64);

  int64_t da[][3] = {{1, 2, 3}, {4, 5, 6}};
  int64_t db[][2] = {{7, 8}, {9, 10}, {11, 12}};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_matmul_naive(a, b, c);
  ASSERT_MSG(err == 0, "matmul_naive int64 should succeed");

  int64_t *r = (int64_t *)numc_array_data(c);
  ASSERT_MSG(r[0] == 58 && r[1] == 64 && r[2] == 139 && r[3] == 154,
             "matmul_naive int64 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_matmul_naive_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape_a[] = {2, 3};
  size_t shape_b[] = {3, 2};
  size_t shape_c[] = {2, 2};
  NumcArray *a = numc_array_create(ctx, shape_a, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_create(ctx, shape_b, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *c = numc_array_zeros(ctx, shape_c, 2, NUMC_DTYPE_FLOAT32);

  float da[][3] = {{1, 2, 3}, {4, 5, 6}};
  float db[][2] = {{7, 8}, {9, 10}, {11, 12}};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_matmul_naive(a, b, c);
  ASSERT_MSG(err == 0, "matmul_naive float32 should succeed");

  float *r = (float *)numc_array_data(c);
  ASSERT_MSG(r[0] == 58.0f && r[1] == 64.0f && r[2] == 139.0f && r[3] == 154.0f,
             "matmul_naive float32 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_matmul_naive_float64(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape_a[] = {2, 3};
  size_t shape_b[] = {3, 2};
  size_t shape_c[] = {2, 2};
  NumcArray *a = numc_array_create(ctx, shape_a, 2, NUMC_DTYPE_FLOAT64);
  NumcArray *b = numc_array_create(ctx, shape_b, 2, NUMC_DTYPE_FLOAT64);
  NumcArray *c = numc_array_zeros(ctx, shape_c, 2, NUMC_DTYPE_FLOAT64);

  double da[][3] = {{1, 2, 3}, {4, 5, 6}};
  double db[][2] = {{7, 8}, {9, 10}, {11, 12}};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_matmul_naive(a, b, c);
  ASSERT_MSG(err == 0, "matmul_naive float64 should succeed");

  double *r = (double *)numc_array_data(c);
  ASSERT_MSG(r[0] == 58.0 && r[1] == 64.0 && r[2] == 139.0 && r[3] == 154.0,
             "matmul_naive float64 results should match");

  numc_ctx_free(ctx);
  return 0;
}

/* ── Error cases ─────────────────────────────────────────────────────── */

static int test_matmul_inner_dims_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape_a[] = {2, 3};
  size_t shape_b[] = {4, 2}; /* inner dim: a.shape[1]=3 != b.shape[0]=4 */
  size_t shape_c[] = {2, 2};
  NumcArray *a = numc_array_create(ctx, shape_a, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_create(ctx, shape_b, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *c = numc_array_zeros(ctx, shape_c, 2, NUMC_DTYPE_FLOAT32);

  int err = numc_matmul_naive(a, b, c);
  ASSERT_MSG(err == NUMC_ERR_SHAPE,
             "matmul_naive with inner dims mismatch should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_matmul_wrong_out_shape(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape_a[] = {2, 3};
  size_t shape_b[] = {3, 2};
  size_t shape_c[] = {3, 3}; /* should be (2,2) */
  NumcArray *a = numc_array_create(ctx, shape_a, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_create(ctx, shape_b, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *c = numc_array_zeros(ctx, shape_c, 2, NUMC_DTYPE_FLOAT32);

  int err = numc_matmul_naive(a, b, c);
  ASSERT_MSG(err == NUMC_ERR_SHAPE, "wrong out shape should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_matmul_wrong_ndim(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sh1[] = {6};      /* 1D */
  size_t sh2[] = {2, 2};
  NumcArray *a = numc_array_create(ctx, sh1, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_create(ctx, sh2, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *c = numc_array_zeros(ctx, sh2, 2, NUMC_DTYPE_FLOAT32);

  int err = numc_matmul_naive(a, b, c);
  ASSERT_MSG(err == NUMC_ERR_SHAPE, "1D input should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_matmul_dtype_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sh[] = {2, 2};
  NumcArray *a = numc_array_create(ctx, sh, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_create(ctx, sh, 2, NUMC_DTYPE_FLOAT64);
  NumcArray *c = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_FLOAT32);

  int err = numc_matmul_naive(a, b, c);
  ASSERT_MSG(err == NUMC_ERR_TYPE, "dtype mismatch should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_matmul_null(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sh[]  = {2, 2};
  NumcArray *a = numc_array_create(ctx, sh, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_create(ctx, sh, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *c = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_FLOAT32);

  ASSERT_MSG(numc_matmul_naive(NULL, b, c) == NUMC_ERR_NULL, "NULL a should fail");
  ASSERT_MSG(numc_matmul_naive(a, NULL, c) == NUMC_ERR_NULL, "NULL b should fail");
  ASSERT_MSG(numc_matmul_naive(a, b, NULL) == NUMC_ERR_NULL, "NULL out should fail");

  numc_ctx_free(ctx);
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== matmul/test_matmul_naive ===\n\n");

  printf("numc_matmul_naive:\n");
  RUN_TEST(test_matmul_naive_int8);
  RUN_TEST(test_matmul_naive_int16);
  RUN_TEST(test_matmul_naive_int32);
  RUN_TEST(test_matmul_naive_int64);
  RUN_TEST(test_matmul_naive_float32);
  RUN_TEST(test_matmul_naive_float64);

  printf("\nError cases:\n");
  RUN_TEST(test_matmul_inner_dims_mismatch);
  RUN_TEST(test_matmul_wrong_out_shape);
  RUN_TEST(test_matmul_wrong_ndim);
  RUN_TEST(test_matmul_dtype_mismatch);
  RUN_TEST(test_matmul_null);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
