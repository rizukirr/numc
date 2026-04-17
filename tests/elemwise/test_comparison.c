#include "../helpers.h"

/* -- numc_eq ----------------------------------------------------------- */

static int test_eq_int32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_UINT8);

  int32_t da[] = {1, 2, 3, 4};
  int32_t db[] = {1, 5, 3, 0};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_eq(a, b, out);
  ASSERT_MSG(err == 0, "eq int32 should succeed");

  uint8_t *r = (uint8_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1, "1 == 1 -> 1");
  ASSERT_MSG(r[1] == 0, "2 == 5 -> 0");
  ASSERT_MSG(r[2] == 1, "3 == 3 -> 1");
  ASSERT_MSG(r[3] == 0, "4 == 0 -> 0");

  numc_ctx_free(ctx);
  return 0;
}

static int test_eq_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_UINT8);

  float da[] = {1.0f, 2.5f, 3.0f, 4.0f};
  float db[] = {1.0f, 2.5f, 0.0f, 5.0f};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_eq(a, b, out);
  ASSERT_MSG(err == 0, "eq float32 should succeed");

  uint8_t *r = (uint8_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1, "1.0 == 1.0 -> 1");
  ASSERT_MSG(r[1] == 1, "2.5 == 2.5 -> 1");
  ASSERT_MSG(r[2] == 0, "3.0 == 0.0 -> 0");
  ASSERT_MSG(r[3] == 0, "4.0 == 5.0 -> 0");

  numc_ctx_free(ctx);
  return 0;
}

static int test_eq_scalar_int32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_UINT8);

  int32_t da[] = {1, 2, 3, 2};
  numc_array_write(a, da);

  int err = numc_eq_scalar(a, 2.0, out);
  ASSERT_MSG(err == 0, "eq_scalar int32 should succeed");

  uint8_t *r = (uint8_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 0, "1 == 2 -> 0");
  ASSERT_MSG(r[1] == 1, "2 == 2 -> 1");
  ASSERT_MSG(r[2] == 0, "3 == 2 -> 0");
  ASSERT_MSG(r[3] == 1, "2 == 2 -> 1");

  numc_ctx_free(ctx);
  return 0;
}

/* -- numc_gt ----------------------------------------------------------- */

static int test_gt_int32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_UINT8);

  int32_t da[] = {5, 2, 3, 4};
  int32_t db[] = {1, 2, 7, 0};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_gt(a, b, out);
  ASSERT_MSG(err == 0, "gt int32 should succeed");

  uint8_t *r = (uint8_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1, "5 > 1 -> 1");
  ASSERT_MSG(r[1] == 0, "2 > 2 -> 0");
  ASSERT_MSG(r[2] == 0, "3 > 7 -> 0");
  ASSERT_MSG(r[3] == 1, "4 > 0 -> 1");

  numc_ctx_free(ctx);
  return 0;
}

static int test_gt_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_UINT8);

  float da[] = {3.0f, 1.0f, 2.5f, 2.5f};
  float db[] = {1.0f, 5.0f, 2.5f, 2.0f};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_gt(a, b, out);
  ASSERT_MSG(err == 0, "gt float32 should succeed");

  uint8_t *r = (uint8_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1, "3.0 > 1.0 -> 1");
  ASSERT_MSG(r[1] == 0, "1.0 > 5.0 -> 0");
  ASSERT_MSG(r[2] == 0, "2.5 > 2.5 -> 0");
  ASSERT_MSG(r[3] == 1, "2.5 > 2.0 -> 1");

  numc_ctx_free(ctx);
  return 0;
}

static int test_gt_scalar_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_UINT8);

  float da[] = {1.0f, 3.0f, 2.0f, 5.0f};
  numc_array_write(a, da);

  int err = numc_gt_scalar(a, 2.0, out);
  ASSERT_MSG(err == 0, "gt_scalar float32 should succeed");

  uint8_t *r = (uint8_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 0, "1.0 > 2.0 -> 0");
  ASSERT_MSG(r[1] == 1, "3.0 > 2.0 -> 1");
  ASSERT_MSG(r[2] == 0, "2.0 > 2.0 -> 0");
  ASSERT_MSG(r[3] == 1, "5.0 > 2.0 -> 1");

  numc_ctx_free(ctx);
  return 0;
}

/* -- numc_lt ----------------------------------------------------------- */

static int test_lt_int32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_UINT8);

  int32_t da[] = {1, 2, 7, 0};
  int32_t db[] = {5, 2, 3, 4};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_lt(a, b, out);
  ASSERT_MSG(err == 0, "lt int32 should succeed");

  uint8_t *r = (uint8_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1, "1 < 5 -> 1");
  ASSERT_MSG(r[1] == 0, "2 < 2 -> 0");
  ASSERT_MSG(r[2] == 0, "7 < 3 -> 0");
  ASSERT_MSG(r[3] == 1, "0 < 4 -> 1");

  numc_ctx_free(ctx);
  return 0;
}

static int test_lt_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_UINT8);

  float da[] = {1.0f, 5.0f, 2.5f, 2.0f};
  float db[] = {3.0f, 1.0f, 2.5f, 2.5f};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_lt(a, b, out);
  ASSERT_MSG(err == 0, "lt float32 should succeed");

  uint8_t *r = (uint8_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1, "1.0 < 3.0 -> 1");
  ASSERT_MSG(r[1] == 0, "5.0 < 1.0 -> 0");
  ASSERT_MSG(r[2] == 0, "2.5 < 2.5 -> 0");
  ASSERT_MSG(r[3] == 1, "2.0 < 2.5 -> 1");

  numc_ctx_free(ctx);
  return 0;
}

static int test_lt_scalar_int32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_UINT8);

  int32_t da[] = {1, 3, 2, 5};
  numc_array_write(a, da);

  int err = numc_lt_scalar(a, 3.0, out);
  ASSERT_MSG(err == 0, "lt_scalar int32 should succeed");

  uint8_t *r = (uint8_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1, "1 < 3 -> 1");
  ASSERT_MSG(r[1] == 0, "3 < 3 -> 0");
  ASSERT_MSG(r[2] == 1, "2 < 3 -> 1");
  ASSERT_MSG(r[3] == 0, "5 < 3 -> 0");

  numc_ctx_free(ctx);
  return 0;
}

/* -- numc_ge ----------------------------------------------------------- */

static int test_ge_int32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_UINT8);

  int32_t da[] = {5, 2, 3, 0};
  int32_t db[] = {1, 2, 7, 4};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_ge(a, b, out);
  ASSERT_MSG(err == 0, "ge int32 should succeed");

  uint8_t *r = (uint8_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1, "5 >= 1 -> 1");
  ASSERT_MSG(r[1] == 1, "2 >= 2 -> 1");
  ASSERT_MSG(r[2] == 0, "3 >= 7 -> 0");
  ASSERT_MSG(r[3] == 0, "0 >= 4 -> 0");

  numc_ctx_free(ctx);
  return 0;
}

static int test_ge_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_UINT8);

  float da[] = {3.0f, 1.0f, 2.5f, 2.5f};
  float db[] = {1.0f, 5.0f, 2.5f, 3.0f};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_ge(a, b, out);
  ASSERT_MSG(err == 0, "ge float32 should succeed");

  uint8_t *r = (uint8_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1, "3.0 >= 1.0 -> 1");
  ASSERT_MSG(r[1] == 0, "1.0 >= 5.0 -> 0");
  ASSERT_MSG(r[2] == 1, "2.5 >= 2.5 -> 1");
  ASSERT_MSG(r[3] == 0, "2.5 >= 3.0 -> 0");

  numc_ctx_free(ctx);
  return 0;
}

static int test_ge_scalar_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_UINT8);

  float da[] = {1.0f, 3.0f, 2.0f, 5.0f};
  numc_array_write(a, da);

  int err = numc_ge_scalar(a, 3.0, out);
  ASSERT_MSG(err == 0, "ge_scalar float32 should succeed");

  uint8_t *r = (uint8_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 0, "1.0 >= 3.0 -> 0");
  ASSERT_MSG(r[1] == 1, "3.0 >= 3.0 -> 1");
  ASSERT_MSG(r[2] == 0, "2.0 >= 3.0 -> 0");
  ASSERT_MSG(r[3] == 1, "5.0 >= 3.0 -> 1");

  numc_ctx_free(ctx);
  return 0;
}

/* -- numc_le ----------------------------------------------------------- */

static int test_le_int32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_UINT8);

  int32_t da[] = {1, 2, 7, 4};
  int32_t db[] = {5, 2, 3, 0};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_le(a, b, out);
  ASSERT_MSG(err == 0, "le int32 should succeed");

  uint8_t *r = (uint8_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1, "1 <= 5 -> 1");
  ASSERT_MSG(r[1] == 1, "2 <= 2 -> 1");
  ASSERT_MSG(r[2] == 0, "7 <= 3 -> 0");
  ASSERT_MSG(r[3] == 0, "4 <= 0 -> 0");

  numc_ctx_free(ctx);
  return 0;
}

static int test_le_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_UINT8);

  float da[] = {1.0f, 5.0f, 2.5f, 3.0f};
  float db[] = {3.0f, 1.0f, 2.5f, 2.5f};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_le(a, b, out);
  ASSERT_MSG(err == 0, "le float32 should succeed");

  uint8_t *r = (uint8_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1, "1.0 <= 3.0 -> 1");
  ASSERT_MSG(r[1] == 0, "5.0 <= 1.0 -> 0");
  ASSERT_MSG(r[2] == 1, "2.5 <= 2.5 -> 1");
  ASSERT_MSG(r[3] == 0, "3.0 <= 2.5 -> 0");

  numc_ctx_free(ctx);
  return 0;
}

static int test_le_scalar_int32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_UINT8);

  int32_t da[] = {1, 3, 5, 2};
  numc_array_write(a, da);

  int err = numc_le_scalar(a, 3.0, out);
  ASSERT_MSG(err == 0, "le_scalar int32 should succeed");

  uint8_t *r = (uint8_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1, "1 <= 3 -> 1");
  ASSERT_MSG(r[1] == 1, "3 <= 3 -> 1");
  ASSERT_MSG(r[2] == 0, "5 <= 3 -> 0");
  ASSERT_MSG(r[3] == 1, "2 <= 3 -> 1");

  numc_ctx_free(ctx);
  return 0;
}

/* -- 2D test ----------------------------------------------------------- */

static int test_comparison_2d(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_UINT8);

  int32_t da[] = {1, 2, 3, 4, 5, 6};
  int32_t db[] = {1, 3, 2, 4, 6, 5};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_eq(a, b, out);
  ASSERT_MSG(err == 0, "2D eq should succeed");

  uint8_t *r = (uint8_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1 && r[1] == 0 && r[2] == 0, "2D eq row 0");
  ASSERT_MSG(r[3] == 1 && r[4] == 0 && r[5] == 0, "2D eq row 1");

  numc_ctx_free(ctx);
  return 0;
}

/* -- Strided (transposed) test ----------------------------------------- */

static int test_comparison_strided(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);

  int32_t da[] = {1, 2, 3, 4, 5, 6};
  int32_t db[] = {1, 5, 3, 4, 2, 6};
  numc_array_write(a, da);
  numc_array_write(b, db);

  size_t axes[] = {1, 0};
  numc_array_transpose(a, axes);
  numc_array_transpose(b, axes);

  // Transposed shapes: {3, 2}
  // a^T: [[1,4],[2,5],[3,6]], b^T: [[1,4],[5,2],[3,6]]
  size_t out_shape[] = {3, 2};
  NumcArray *out = numc_array_zeros(ctx, out_shape, 2, NUMC_DTYPE_UINT8);

  int err = numc_gt(a, b, out);
  ASSERT_MSG(err == 0, "strided gt should succeed");

  uint8_t *r = (uint8_t *)numc_array_data(out);
  // [1>1, 4>4, 2>5, 5>2, 3>3, 6>6] = [0, 0, 0, 1, 0, 0]
  ASSERT_MSG(r[0] == 0 && r[1] == 0, "strided gt row 0");
  ASSERT_MSG(r[2] == 0 && r[3] == 1, "strided gt row 1");
  ASSERT_MSG(r[4] == 0 && r[5] == 0, "strided gt row 2");

  numc_ctx_free(ctx);
  return 0;
}

/* -- Error cases ------------------------------------------------------- */

static int test_comparison_null(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_UINT8);

  ASSERT_MSG(numc_eq(NULL, a, out) != 0, "eq with NULL a should fail");
  ASSERT_MSG(numc_eq(a, NULL, out) != 0, "eq with NULL b should fail");
  ASSERT_MSG(numc_eq(a, a, NULL) != 0, "eq with NULL out should fail");
  ASSERT_MSG(numc_gt(NULL, a, out) != 0, "gt with NULL should fail");
  ASSERT_MSG(numc_lt(NULL, a, out) != 0, "lt with NULL should fail");
  ASSERT_MSG(numc_ge(NULL, a, out) != 0, "ge with NULL should fail");
  ASSERT_MSG(numc_le(NULL, a, out) != 0, "le with NULL should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_comparison_type_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_UINT8);

  ASSERT_MSG(numc_eq(a, b, out) != 0, "eq with dtype mismatch should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_comparison_shape_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape_a[] = {4};
  size_t shape_b[] = {5};
  NumcArray *a = numc_array_zeros(ctx, shape_a, 1, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_zeros(ctx, shape_b, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape_a, 1, NUMC_DTYPE_UINT8);

  ASSERT_MSG(numc_eq(a, b, out) != 0, "eq with shape mismatch should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_comparison_scalar_null(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_UINT8);

  ASSERT_MSG(numc_eq_scalar(NULL, 1.0, out) != 0,
             "eq_scalar with NULL a should fail");
  ASSERT_MSG(numc_eq_scalar(a, 1.0, NULL) != 0,
             "eq_scalar with NULL out should fail");
  ASSERT_MSG(numc_gt_scalar(NULL, 1.0, out) != 0,
             "gt_scalar with NULL should fail");
  ASSERT_MSG(numc_lt_scalar(NULL, 1.0, out) != 0,
             "lt_scalar with NULL should fail");
  ASSERT_MSG(numc_ge_scalar(NULL, 1.0, out) != 0,
             "ge_scalar with NULL should fail");
  ASSERT_MSG(numc_le_scalar(NULL, 1.0, out) != 0,
             "le_scalar with NULL should fail");

  numc_ctx_free(ctx);
  return 0;
}

/* -- Output type validation -------------------------------------------- */

static int test_comparison_wrong_output_type(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out_bad = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);

  ASSERT_MSG(numc_eq(a, b, out_bad) != 0,
             "eq with non-uint8 output should fail");

  numc_ctx_free(ctx);
  return 0;
}

/* -- main -------------------------------------------------------------- */

int main(void) {
  int passes = 0, fails = 0;
  printf("=== elemwise/test_comparison ===\n\n");

  printf("numc_eq:\n");
  RUN_TEST(test_eq_int32);
  RUN_TEST(test_eq_float32);
  RUN_TEST(test_eq_scalar_int32);

  printf("\nnumc_gt:\n");
  RUN_TEST(test_gt_int32);
  RUN_TEST(test_gt_float32);
  RUN_TEST(test_gt_scalar_float32);

  printf("\nnumc_lt:\n");
  RUN_TEST(test_lt_int32);
  RUN_TEST(test_lt_float32);
  RUN_TEST(test_lt_scalar_int32);

  printf("\nnumc_ge:\n");
  RUN_TEST(test_ge_int32);
  RUN_TEST(test_ge_float32);
  RUN_TEST(test_ge_scalar_float32);

  printf("\nnumc_le:\n");
  RUN_TEST(test_le_int32);
  RUN_TEST(test_le_float32);
  RUN_TEST(test_le_scalar_int32);

  printf("\nMulti-dimensional:\n");
  RUN_TEST(test_comparison_2d);
  RUN_TEST(test_comparison_strided);

  printf("\nError cases:\n");
  RUN_TEST(test_comparison_null);
  RUN_TEST(test_comparison_type_mismatch);
  RUN_TEST(test_comparison_shape_mismatch);
  RUN_TEST(test_comparison_scalar_null);
  RUN_TEST(test_comparison_wrong_output_type);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
