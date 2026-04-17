#include "../helpers.h"

/* -- Basic int32 ------------------------------------------------------- */

static int test_where_int32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *cond = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);

  int32_t dc[] = {1, 0, 1, 0};
  int32_t da[] = {10, 20, 30, 40};
  int32_t db[] = {90, 80, 70, 60};
  numc_array_write(cond, dc);
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_where(cond, a, b, out);
  ASSERT_MSG(err == 0, "where int32 should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 10, "cond=1 -> a[0]=10");
  ASSERT_MSG(r[1] == 80, "cond=0 -> b[1]=80");
  ASSERT_MSG(r[2] == 30, "cond=1 -> a[2]=30");
  ASSERT_MSG(r[3] == 60, "cond=0 -> b[3]=60");

  numc_ctx_free(ctx);
  return 0;
}

/* -- Basic float32 ----------------------------------------------------- */

static int test_where_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *cond = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float dc[] = {1.0f, 0.0f, 5.0f, 0.0f};
  float da[] = {1.5f, 2.5f, 3.5f, 4.5f};
  float db[] = {9.5f, 8.5f, 7.5f, 6.5f};
  numc_array_write(cond, dc);
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_where(cond, a, b, out);
  ASSERT_MSG(err == 0, "where float32 should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1.5f, "cond=1.0 -> a[0]=1.5");
  ASSERT_MSG(r[1] == 8.5f, "cond=0.0 -> b[1]=8.5");
  ASSERT_MSG(r[2] == 3.5f, "cond=5.0 -> a[2]=3.5");
  ASSERT_MSG(r[3] == 6.5f, "cond=0.0 -> b[3]=6.5");

  numc_ctx_free(ctx);
  return 0;
}

/* -- 2D arrays --------------------------------------------------------- */

static int test_where_2d(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *cond = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_INT32);

  int32_t dc[] = {1, 0, 1, 0, 1, 0};
  int32_t da[] = {10, 20, 30, 40, 50, 60};
  int32_t db[] = {11, 22, 33, 44, 55, 66};
  numc_array_write(cond, dc);
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_where(cond, a, b, out);
  ASSERT_MSG(err == 0, "where 2D should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 10 && r[1] == 22 && r[2] == 30, "2D where row 0");
  ASSERT_MSG(r[3] == 44 && r[4] == 50 && r[5] == 66, "2D where row 1");

  numc_ctx_free(ctx);
  return 0;
}

/* -- Strided (transposed) ---------------------------------------------- */

static int test_where_strided(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *cond = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);

  int32_t dc[] = {1, 0, 1, 0, 1, 0};
  int32_t da[] = {10, 20, 30, 40, 50, 60};
  int32_t db[] = {11, 22, 33, 44, 55, 66};
  numc_array_write(cond, dc);
  numc_array_write(a, da);
  numc_array_write(b, db);

  size_t axes[] = {1, 0};
  numc_array_transpose(cond, axes);
  numc_array_transpose(a, axes);
  numc_array_transpose(b, axes);

  // Transposed shape: {3, 2}
  // cond^T: [[1,0],[0,1],[1,0]]
  // a^T:    [[10,40],[20,50],[30,60]]
  // b^T:    [[11,44],[22,55],[33,66]]
  // expected: [[10,44],[22,50],[30,66]]
  size_t out_shape[] = {3, 2};
  NumcArray *out = numc_array_zeros(ctx, out_shape, 2, NUMC_DTYPE_INT32);

  int err = numc_where(cond, a, b, out);
  ASSERT_MSG(err == 0, "strided where should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 10 && r[1] == 44, "strided where row 0");
  ASSERT_MSG(r[2] == 22 && r[3] == 50, "strided where row 1");
  ASSERT_MSG(r[4] == 30 && r[5] == 66, "strided where row 2");

  numc_ctx_free(ctx);
  return 0;
}

/* -- Condition from comparison op -------------------------------------- */

static int test_where_from_gt(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *x = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *y = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *cond = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);

  int32_t dx[] = {5, 2, 8, 1};
  int32_t dy[] = {3, 4, 6, 9};
  numc_array_write(x, dx);
  numc_array_write(y, dy);

  // Build condition manually: x > y -> [1, 0, 1, 0]
  int32_t dc[] = {1, 0, 1, 0};
  numc_array_write(cond, dc);
  int err = 0;
  ASSERT_MSG(err == 0, "gt should succeed");

  // out = where(cond, x, y) -> [5, 4, 8, 9]
  err = numc_where(cond, x, y, out);
  ASSERT_MSG(err == 0, "where from gt should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 5, "5>3 -> pick x=5");
  ASSERT_MSG(r[1] == 4, "2<=4 -> pick y=4");
  ASSERT_MSG(r[2] == 8, "8>6 -> pick x=8");
  ASSERT_MSG(r[3] == 9, "1<=9 -> pick y=9");

  numc_ctx_free(ctx);
  return 0;
}

/* -- Error: NULL pointers ---------------------------------------------- */

static int test_where_null(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);

  ASSERT_MSG(numc_where(NULL, a, a, out) != 0, "where with NULL cond");
  ASSERT_MSG(numc_where(a, NULL, a, out) != 0, "where with NULL a");
  ASSERT_MSG(numc_where(a, a, NULL, out) != 0, "where with NULL b");
  ASSERT_MSG(numc_where(a, a, a, NULL) != 0, "where with NULL out");

  numc_ctx_free(ctx);
  return 0;
}

/* -- Error: dtype mismatch --------------------------------------------- */

static int test_where_type_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *cond = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *a = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  ASSERT_MSG(numc_where(cond, a, b, out) != 0, "where dtype mismatch");

  numc_ctx_free(ctx);
  return 0;
}

/* -- Error: shape mismatch --------------------------------------------- */

static int test_where_shape_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape4[] = {4};
  size_t shape5[] = {5};
  NumcArray *cond = numc_array_zeros(ctx, shape4, 1, NUMC_DTYPE_INT32);
  NumcArray *a = numc_array_zeros(ctx, shape4, 1, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_zeros(ctx, shape5, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape4, 1, NUMC_DTYPE_INT32);

  ASSERT_MSG(numc_where(cond, a, b, out) != 0, "where shape mismatch");

  numc_ctx_free(ctx);
  return 0;
}

/* -- main -------------------------------------------------------------- */

int main(void) {
  int passes = 0, fails = 0;
  printf("=== elemwise/test_where ===\n\n");

  printf("Basic:\n");
  RUN_TEST(test_where_int32);
  RUN_TEST(test_where_float32);

  printf("\nMulti-dimensional:\n");
  RUN_TEST(test_where_2d);
  RUN_TEST(test_where_strided);

  printf("\nComparison chaining:\n");
  RUN_TEST(test_where_from_gt);

  printf("\nError cases:\n");
  RUN_TEST(test_where_null);
  RUN_TEST(test_where_type_mismatch);
  RUN_TEST(test_where_shape_mismatch);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
