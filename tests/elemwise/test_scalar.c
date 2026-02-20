#include "../helpers.h"

static int test_add_scalar_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {1.0f, 2.0f, 3.0f, 4.0f};
  numc_array_write(a, da);

  int err = numc_add_scalar(a, 10.0, out);
  ASSERT_MSG(err == 0, "add_scalar should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 11.0f && r[1] == 12.0f && r[2] == 13.0f && r[3] == 14.0f,
             "add_scalar results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sub_scalar_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {10.0f, 20.0f, 30.0f, 40.0f};
  numc_array_write(a, da);

  int err = numc_sub_scalar(a, 5.0, out);
  ASSERT_MSG(err == 0, "sub_scalar should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 5.0f && r[1] == 15.0f && r[2] == 25.0f && r[3] == 35.0f,
             "sub_scalar results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_mul_scalar_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {1.0f, 2.0f, 3.0f, 4.0f};
  numc_array_write(a, da);

  int err = numc_mul_scalar(a, 3.0, out);
  ASSERT_MSG(err == 0, "mul_scalar should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 3.0f && r[1] == 6.0f && r[2] == 9.0f && r[3] == 12.0f,
             "mul_scalar results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_div_scalar_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {10.0f, 20.0f, 30.0f, 40.0f};
  numc_array_write(a, da);

  int err = numc_div_scalar(a, 10.0, out);
  ASSERT_MSG(err == 0, "div_scalar should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1.0f && r[1] == 2.0f && r[2] == 3.0f && r[3] == 4.0f,
             "div_scalar results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_scalar_op_int32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);

  int32_t da[] = {10, 20, 30};
  numc_array_write(a, da);

  int err = numc_add_scalar(a, 5.0, out);
  ASSERT_MSG(err == 0, "add_scalar int32 should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 15 && r[1] == 25 && r[2] == 35,
             "add_scalar int32 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_scalar_op_null(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  ASSERT_MSG(numc_add_scalar(NULL, 1.0, out) != 0,
             "add_scalar with NULL a should fail");
  ASSERT_MSG(numc_add_scalar(a, 1.0, NULL) != 0,
             "add_scalar with NULL out should fail");
  ASSERT_MSG(numc_sub_scalar(NULL, 1.0, out) != 0,
             "sub_scalar with NULL should fail");
  ASSERT_MSG(numc_mul_scalar(NULL, 1.0, out) != 0,
             "mul_scalar with NULL should fail");
  ASSERT_MSG(numc_div_scalar(NULL, 1.0, out) != 0,
             "div_scalar with NULL should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_scalar_op_type_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);

  ASSERT_MSG(numc_add_scalar(a, 1.0, out) != 0,
             "scalar op with dtype mismatch should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_scalar_op_shape_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape_a[] = {4};
  size_t shape_o[] = {5};
  NumcArray *a = numc_array_zeros(ctx, shape_a, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape_o, 1, NUMC_DTYPE_FLOAT32);

  ASSERT_MSG(numc_add_scalar(a, 1.0, out) != 0,
             "scalar op with shape mismatch should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_add_scalar_inplace_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {1.0f, 2.0f, 3.0f, 4.0f};
  numc_array_write(a, da);

  int err = numc_add_scalar_inplace(a, 10.0);
  ASSERT_MSG(err == 0, "add_scalar_inplace should succeed");

  float *r = (float *)numc_array_data(a);
  ASSERT_MSG(r[0] == 11.0f && r[1] == 12.0f && r[2] == 13.0f && r[3] == 14.0f,
             "add_scalar_inplace results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sub_scalar_inplace_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {10.0f, 20.0f, 30.0f, 40.0f};
  numc_array_write(a, da);

  int err = numc_sub_scalar_inplace(a, 5.0);
  ASSERT_MSG(err == 0, "sub_scalar_inplace should succeed");

  float *r = (float *)numc_array_data(a);
  ASSERT_MSG(r[0] == 5.0f && r[1] == 15.0f && r[2] == 25.0f && r[3] == 35.0f,
             "sub_scalar_inplace results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_mul_scalar_inplace_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {1.0f, 2.0f, 3.0f, 4.0f};
  numc_array_write(a, da);

  int err = numc_mul_scalar_inplace(a, 3.0);
  ASSERT_MSG(err == 0, "mul_scalar_inplace should succeed");

  float *r = (float *)numc_array_data(a);
  ASSERT_MSG(r[0] == 3.0f && r[1] == 6.0f && r[2] == 9.0f && r[3] == 12.0f,
             "mul_scalar_inplace results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_div_scalar_inplace_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {10.0f, 20.0f, 30.0f, 40.0f};
  numc_array_write(a, da);

  int err = numc_div_scalar_inplace(a, 10.0);
  ASSERT_MSG(err == 0, "div_scalar_inplace should succeed");

  float *r = (float *)numc_array_data(a);
  ASSERT_MSG(r[0] == 1.0f && r[1] == 2.0f && r[2] == 3.0f && r[3] == 4.0f,
             "div_scalar_inplace results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_scalar_inplace_int32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);

  int32_t da[] = {10, 20, 30};
  numc_array_write(a, da);

  int err = numc_mul_scalar_inplace(a, 2.0);
  ASSERT_MSG(err == 0, "mul_scalar_inplace int32 should succeed");

  int32_t *r = (int32_t *)numc_array_data(a);
  ASSERT_MSG(r[0] == 20 && r[1] == 40 && r[2] == 60,
             "mul_scalar_inplace int32 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_scalar_inplace_null(void) {
  ASSERT_MSG(numc_add_scalar_inplace(NULL, 1.0) != 0,
             "add_scalar_inplace with NULL should fail");
  ASSERT_MSG(numc_sub_scalar_inplace(NULL, 1.0) != 0,
             "sub_scalar_inplace with NULL should fail");
  ASSERT_MSG(numc_mul_scalar_inplace(NULL, 1.0) != 0,
             "mul_scalar_inplace with NULL should fail");
  ASSERT_MSG(numc_div_scalar_inplace(NULL, 1.0) != 0,
             "div_scalar_inplace with NULL should fail");
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== elemwise/test_scalar ===\n\n");

  printf("Scalar ops:\n");
  RUN_TEST(test_add_scalar_float32);
  RUN_TEST(test_sub_scalar_float32);
  RUN_TEST(test_mul_scalar_float32);
  RUN_TEST(test_div_scalar_float32);
  RUN_TEST(test_scalar_op_int32);

  printf("\nScalar error cases:\n");
  RUN_TEST(test_scalar_op_null);
  RUN_TEST(test_scalar_op_type_mismatch);
  RUN_TEST(test_scalar_op_shape_mismatch);

  printf("\nScalar inplace ops:\n");
  RUN_TEST(test_add_scalar_inplace_float32);
  RUN_TEST(test_sub_scalar_inplace_float32);
  RUN_TEST(test_mul_scalar_inplace_float32);
  RUN_TEST(test_div_scalar_inplace_float32);
  RUN_TEST(test_scalar_inplace_int32);
  RUN_TEST(test_scalar_inplace_null);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
