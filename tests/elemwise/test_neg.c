#include "../helpers.h"

static int test_neg_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {1.0f, -2.0f, 3.0f, -4.0f};
  numc_array_write(a, da);

  int err = numc_neg(a, out);
  ASSERT_MSG(err == 0, "neg should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == -1.0f && r[1] == 2.0f && r[2] == -3.0f && r[3] == 4.0f,
             "neg results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_neg_int32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);

  int32_t da[] = {10, -20, 30, -40};
  numc_array_write(a, da);

  int err = numc_neg(a, out);
  ASSERT_MSG(err == 0, "neg int32 should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == -10 && r[1] == 20 && r[2] == -30 && r[3] == 40,
             "neg int32 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_neg_int8(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT8);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT8);

  int8_t da[] = {5, -10, 15, -20};
  numc_array_write(a, da);

  int err = numc_neg(a, out);
  ASSERT_MSG(err == 0, "neg int8 should succeed");

  int8_t *r = (int8_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == -5 && r[1] == 10 && r[2] == -15 && r[3] == 20,
             "neg int8 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_neg_float64(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT64);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT64);

  double da[] = {1.5, -2.5, 3.5};
  numc_array_write(a, da);

  int err = numc_neg(a, out);
  ASSERT_MSG(err == 0, "neg float64 should succeed");

  double *r = (double *)numc_array_data(out);
  ASSERT_MSG(r[0] == -1.5 && r[1] == 2.5 && r[2] == -3.5,
             "neg float64 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_neg_2d(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_INT32);

  int32_t da[] = {1, -2, 3, -4, 5, -6};
  numc_array_write(a, da);

  int err = numc_neg(a, out);
  ASSERT_MSG(err == 0, "2D neg should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == -1 && r[1] == 2 && r[2] == -3, "neg 2D row 0");
  ASSERT_MSG(r[3] == 4 && r[4] == -5 && r[5] == 6, "neg 2D row 1");

  numc_ctx_free(ctx);
  return 0;
}

static int test_neg_strided(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);

  int32_t da[] = {1, -2, 3, -4, 5, -6};
  numc_array_write(a, da);

  // Transpose to make it non-contiguous
  size_t axes[] = {1, 0};
  numc_array_transpose(a, axes);

  // out shape matches transposed: {3, 2}
  size_t out_shape[] = {3, 2};
  NumcArray *out = numc_array_zeros(ctx, out_shape, 2, NUMC_DTYPE_INT32);

  int err = numc_neg(a, out);
  ASSERT_MSG(err == 0, "strided neg should succeed");

  // Transposed a: [[1,-4],[-2,5],[3,-6]]
  // negated: [[-1,4],[2,-5],[-3,6]]
  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == -1 && r[1] == 4, "strided row 0");
  ASSERT_MSG(r[2] == 2 && r[3] == -5, "strided row 1");
  ASSERT_MSG(r[4] == -3 && r[5] == 6, "strided row 2");

  numc_ctx_free(ctx);
  return 0;
}

static int test_neg_zeros(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  int err = numc_neg(a, out);
  ASSERT_MSG(err == 0, "neg of zeros should succeed");

  float *r = (float *)numc_array_data(out);
  for (size_t i = 0; i < 4; i++) {
    ASSERT_MSG(r[i] == 0.0f, "neg of zero should be zero");
  }

  numc_ctx_free(ctx);
  return 0;
}

static int test_neg_null(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  ASSERT_MSG(numc_neg(NULL, out) != 0, "neg with NULL a should fail");
  ASSERT_MSG(numc_neg(a, NULL) != 0, "neg with NULL out should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_neg_type_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);

  ASSERT_MSG(numc_neg(a, out) != 0, "neg with dtype mismatch should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_neg_shape_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape_a[] = {4};
  size_t shape_o[] = {5};
  NumcArray *a = numc_array_zeros(ctx, shape_a, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape_o, 1, NUMC_DTYPE_FLOAT32);

  ASSERT_MSG(numc_neg(a, out) != 0, "neg with shape mismatch should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_neg_inplace_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {1.0f, -2.0f, 3.0f, -4.0f};
  numc_array_write(a, da);

  int err = numc_neg_inplace(a);
  ASSERT_MSG(err == 0, "neg_inplace should succeed");

  float *r = (float *)numc_array_data(a);
  ASSERT_MSG(r[0] == -1.0f && r[1] == 2.0f && r[2] == -3.0f && r[3] == 4.0f,
             "neg_inplace results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_neg_inplace_int32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);

  int32_t da[] = {10, -20, 30, -40};
  numc_array_write(a, da);

  int err = numc_neg_inplace(a);
  ASSERT_MSG(err == 0, "neg_inplace int32 should succeed");

  int32_t *r = (int32_t *)numc_array_data(a);
  ASSERT_MSG(r[0] == -10 && r[1] == 20 && r[2] == -30 && r[3] == 40,
             "neg_inplace int32 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_neg_inplace_int8(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT8);

  int8_t da[] = {5, -10, 15, -20};
  numc_array_write(a, da);

  int err = numc_neg_inplace(a);
  ASSERT_MSG(err == 0, "neg_inplace int8 should succeed");

  int8_t *r = (int8_t *)numc_array_data(a);
  ASSERT_MSG(r[0] == -5 && r[1] == 10 && r[2] == -15 && r[3] == 20,
             "neg_inplace int8 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_neg_inplace_float64(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT64);

  double da[] = {1.5, -2.5, 3.5};
  numc_array_write(a, da);

  int err = numc_neg_inplace(a);
  ASSERT_MSG(err == 0, "neg_inplace float64 should succeed");

  double *r = (double *)numc_array_data(a);
  ASSERT_MSG(r[0] == -1.5 && r[1] == 2.5 && r[2] == -3.5,
             "neg_inplace float64 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_neg_inplace_2d(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);

  int32_t da[] = {1, -2, 3, -4, 5, -6};
  numc_array_write(a, da);

  int err = numc_neg_inplace(a);
  ASSERT_MSG(err == 0, "2D neg_inplace should succeed");

  int32_t *r = (int32_t *)numc_array_data(a);
  ASSERT_MSG(r[0] == -1 && r[1] == 2 && r[2] == -3, "neg_inplace 2D row 0");
  ASSERT_MSG(r[3] == 4 && r[4] == -5 && r[5] == 6, "neg_inplace 2D row 1");

  numc_ctx_free(ctx);
  return 0;
}

static int test_neg_inplace_contiguous_2d(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3, 2};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);

  int32_t da[] = {1, -2, 3, -4, 5, -6};
  numc_array_write(a, da);

  int err = numc_neg_inplace(a);
  ASSERT_MSG(err == 0, "neg_inplace contiguous 2d should succeed");

  int32_t *r = (int32_t *)numc_array_data(a);
  ASSERT_MSG(r[0] == -1 && r[1] == 2, "inplace row 0");
  ASSERT_MSG(r[2] == -3 && r[3] == 4, "inplace row 1");
  ASSERT_MSG(r[4] == -5 && r[5] == 6, "inplace row 2");

  numc_ctx_free(ctx);
  return 0;
}

static int test_neg_inplace_zeros(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  int err = numc_neg_inplace(a);
  ASSERT_MSG(err == 0, "neg_inplace of zeros should succeed");

  float *r = (float *)numc_array_data(a);
  for (size_t i = 0; i < 4; i++) {
    ASSERT_MSG(r[i] == 0.0f, "neg_inplace of zero should be zero");
  }

  numc_ctx_free(ctx);
  return 0;
}

static int test_neg_inplace_null(void) {
  ASSERT_MSG(numc_neg_inplace(NULL) != 0, "neg_inplace with NULL should fail");
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== elemwise/test_neg ===\n\n");

  printf("numc_neg:\n");
  RUN_TEST(test_neg_float32);
  RUN_TEST(test_neg_int32);
  RUN_TEST(test_neg_int8);
  RUN_TEST(test_neg_float64);
  RUN_TEST(test_neg_2d);
  RUN_TEST(test_neg_strided);
  RUN_TEST(test_neg_zeros);
  RUN_TEST(test_neg_null);
  RUN_TEST(test_neg_type_mismatch);
  RUN_TEST(test_neg_shape_mismatch);

  printf("\nnumc_neg_inplace:\n");
  RUN_TEST(test_neg_inplace_float32);
  RUN_TEST(test_neg_inplace_int32);
  RUN_TEST(test_neg_inplace_int8);
  RUN_TEST(test_neg_inplace_float64);
  RUN_TEST(test_neg_inplace_2d);
  RUN_TEST(test_neg_inplace_contiguous_2d);
  RUN_TEST(test_neg_inplace_zeros);
  RUN_TEST(test_neg_inplace_null);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
