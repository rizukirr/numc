#include "../helpers.h"

/* ── Broadcast dim 0: (1,4) + (3,4) → (3,4) ───────────────────────── */

static int test_broadcast_dim0(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sa[] = {1, 4}, sb[] = {3, 4}, so[] = {3, 4};
  NumcArray *a = numc_array_create(ctx, sa, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_create(ctx, sb, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, so, 2, NUMC_DTYPE_FLOAT32);

  float da[] = {1, 2, 3, 4};
  float db[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_add(a, b, out);
  ASSERT_MSG(err == 0, "broadcast dim0 add should succeed");

  float *r = (float *)numc_array_data(out);
  /* row 0: 1+10, 2+20, 3+30, 4+40 */
  ASSERT_MSG(r[0] == 11.0f && r[1] == 22.0f && r[2] == 33.0f && r[3] == 44.0f,
             "row 0");
  /* row 1: 1+50, 2+60, 3+70, 4+80 */
  ASSERT_MSG(r[4] == 51.0f && r[5] == 62.0f && r[6] == 73.0f && r[7] == 84.0f,
             "row 1");
  /* row 2: 1+90, 2+100, 3+110, 4+120 */
  ASSERT_MSG(r[8] == 91.0f && r[9] == 102.0f && r[10] == 113.0f &&
                 r[11] == 124.0f,
             "row 2");

  numc_ctx_free(ctx);
  return 0;
}

/* ── Broadcast dim 1: (3,1) + (3,4) → (3,4) ───────────────────────── */

static int test_broadcast_dim1(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sa[] = {3, 1}, sb[] = {3, 4}, so[] = {3, 4};
  NumcArray *a = numc_array_create(ctx, sa, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_create(ctx, sb, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, so, 2, NUMC_DTYPE_FLOAT32);

  float da[] = {1, 2, 3};
  float db[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_add(a, b, out);
  ASSERT_MSG(err == 0, "broadcast dim1 add should succeed");

  float *r = (float *)numc_array_data(out);
  /* row 0: 1+10, 1+20, 1+30, 1+40 */
  ASSERT_MSG(r[0] == 11.0f && r[1] == 21.0f && r[2] == 31.0f && r[3] == 41.0f,
             "row 0");
  /* row 1: 2+50, 2+60, 2+70, 2+80 */
  ASSERT_MSG(r[4] == 52.0f && r[5] == 62.0f && r[6] == 72.0f && r[7] == 82.0f,
             "row 1");
  /* row 2: 3+90, 3+100, 3+110, 3+120 */
  ASSERT_MSG(r[8] == 93.0f && r[9] == 103.0f && r[10] == 113.0f &&
                 r[11] == 123.0f,
             "row 2");

  numc_ctx_free(ctx);
  return 0;
}

/* ── Both dims broadcast: (3,1) + (1,4) → (3,4) ───────────────────── */

static int test_broadcast_both(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sa[] = {3, 1}, sb[] = {1, 4}, so[] = {3, 4};
  NumcArray *a = numc_array_create(ctx, sa, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_create(ctx, sb, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, so, 2, NUMC_DTYPE_FLOAT32);

  float da[] = {1, 2, 3};
  float db[] = {10, 20, 30, 40};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_add(a, b, out);
  ASSERT_MSG(err == 0, "broadcast both add should succeed");

  float *r = (float *)numc_array_data(out);
  /* row 0: 1+10, 1+20, 1+30, 1+40 */
  ASSERT_MSG(r[0] == 11.0f && r[1] == 21.0f && r[2] == 31.0f && r[3] == 41.0f,
             "row 0");
  /* row 1: 2+10, 2+20, 2+30, 2+40 */
  ASSERT_MSG(r[4] == 12.0f && r[5] == 22.0f && r[6] == 32.0f && r[7] == 42.0f,
             "row 1");
  /* row 2: 3+10, 3+20, 3+30, 3+40 */
  ASSERT_MSG(r[8] == 13.0f && r[9] == 23.0f && r[10] == 33.0f &&
                 r[11] == 43.0f,
             "row 2");

  numc_ctx_free(ctx);
  return 0;
}

/* ── Scalar-like broadcast: (1,1) + (3,4) → (3,4) ─────────────────── */

static int test_broadcast_scalar_like(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sa[] = {1, 1}, sb[] = {3, 4}, so[] = {3, 4};
  NumcArray *a = numc_array_create(ctx, sa, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_create(ctx, sb, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, so, 2, NUMC_DTYPE_FLOAT32);

  float da[] = {100};
  float db[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_add(a, b, out);
  ASSERT_MSG(err == 0, "scalar-like broadcast should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 101.0f && r[5] == 106.0f && r[11] == 112.0f,
             "scalar-like results");

  numc_ctx_free(ctx);
  return 0;
}

/* ── Rank mismatch: (4,) + (3,4) → (3,4) ──────────────────────────── */

static int test_broadcast_rank(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sa[] = {4}, sb[] = {3, 4}, so[] = {3, 4};
  NumcArray *a = numc_array_create(ctx, sa, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_create(ctx, sb, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, so, 2, NUMC_DTYPE_FLOAT32);

  float da[] = {1, 2, 3, 4};
  float db[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_add(a, b, out);
  ASSERT_MSG(err == 0, "rank broadcast add should succeed");

  float *r = (float *)numc_array_data(out);
  /* a is treated as (1,4) → broadcast dim 0 */
  ASSERT_MSG(r[0] == 11.0f && r[1] == 22.0f && r[2] == 33.0f && r[3] == 44.0f,
             "row 0");
  ASSERT_MSG(r[4] == 51.0f && r[5] == 62.0f && r[6] == 73.0f && r[7] == 84.0f,
             "row 1");
  ASSERT_MSG(r[8] == 91.0f && r[9] == 102.0f && r[10] == 113.0f &&
                 r[11] == 124.0f,
             "row 2");

  numc_ctx_free(ctx);
  return 0;
}

/* ── 3D broadcast: (2,1,4) + (1,3,1) → (2,3,4) ───────────────────── */

static int test_broadcast_3d(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sa[] = {2, 1, 4}, sb[] = {1, 3, 1}, so[] = {2, 3, 4};
  NumcArray *a = numc_array_create(ctx, sa, 3, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_create(ctx, sb, 3, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, so, 3, NUMC_DTYPE_FLOAT32);

  float da[] = {1, 2, 3, 4, 5, 6, 7, 8};
  float db[] = {10, 20, 30};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_add(a, b, out);
  ASSERT_MSG(err == 0, "3D broadcast add should succeed");

  float *r = (float *)numc_array_data(out);
  /* out[0,0,:] = a[0,0,:] + b[0,0,0] = [1,2,3,4] + 10 = [11,12,13,14] */
  ASSERT_MSG(r[0] == 11.0f && r[1] == 12.0f && r[2] == 13.0f && r[3] == 14.0f,
             "out[0,0,:]");
  /* out[0,1,:] = a[0,0,:] + b[0,1,0] = [1,2,3,4] + 20 = [21,22,23,24] */
  ASSERT_MSG(r[4] == 21.0f && r[5] == 22.0f && r[6] == 23.0f && r[7] == 24.0f,
             "out[0,1,:]");
  /* out[0,2,:] = a[0,0,:] + b[0,2,0] = [1,2,3,4] + 30 = [31,32,33,34] */
  ASSERT_MSG(r[8] == 31.0f && r[9] == 32.0f && r[10] == 33.0f &&
                 r[11] == 34.0f,
             "out[0,2,:]");
  /* out[1,0,:] = a[1,0,:] + b[0,0,0] = [5,6,7,8] + 10 = [15,16,17,18] */
  ASSERT_MSG(r[12] == 15.0f && r[13] == 16.0f && r[14] == 17.0f &&
                 r[15] == 18.0f,
             "out[1,0,:]");

  numc_ctx_free(ctx);
  return 0;
}

/* ── Broadcast with int32 ──────────────────────────────────────────── */

static int test_broadcast_int32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sa[] = {3, 1}, sb[] = {1, 4}, so[] = {3, 4};
  NumcArray *a = numc_array_create(ctx, sa, 2, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_create(ctx, sb, 2, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, so, 2, NUMC_DTYPE_INT32);

  int32_t da[] = {1, 2, 3};
  int32_t db[] = {10, 20, 30, 40};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_add(a, b, out);
  ASSERT_MSG(err == 0, "broadcast int32 should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 11 && r[1] == 21 && r[2] == 31 && r[3] == 41, "row 0");
  ASSERT_MSG(r[4] == 12 && r[5] == 22 && r[6] == 32 && r[7] == 42, "row 1");
  ASSERT_MSG(r[8] == 13 && r[9] == 23 && r[10] == 33 && r[11] == 43, "row 2");

  numc_ctx_free(ctx);
  return 0;
}

/* ── Broadcast with int8 ───────────────────────────────────────────── */

static int test_broadcast_int8(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sa[] = {3, 1}, sb[] = {1, 4}, so[] = {3, 4};
  NumcArray *a = numc_array_create(ctx, sa, 2, NUMC_DTYPE_INT8);
  NumcArray *b = numc_array_create(ctx, sb, 2, NUMC_DTYPE_INT8);
  NumcArray *out = numc_array_zeros(ctx, so, 2, NUMC_DTYPE_INT8);

  int8_t da[] = {1, 2, 3};
  int8_t db[] = {10, 20, 30, 40};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_add(a, b, out);
  ASSERT_MSG(err == 0, "broadcast int8 should succeed");

  int8_t *r = (int8_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 11 && r[1] == 21 && r[2] == 31 && r[3] == 41, "row 0");
  ASSERT_MSG(r[4] == 12 && r[5] == 22 && r[6] == 32 && r[7] == 42, "row 1");
  ASSERT_MSG(r[8] == 13 && r[9] == 23 && r[10] == 33 && r[11] == 43, "row 2");

  numc_ctx_free(ctx);
  return 0;
}

/* ── Non-contiguous (transposed) + broadcast ───────────────────────── */

static int test_broadcast_noncontiguous(void) {
  NumcCtx *ctx = numc_ctx_create();

  /* Create a (4,3) array and transpose to (3,4) — non-contiguous */
  size_t shape_orig[] = {4, 3};
  NumcArray *b = numc_array_create(ctx, shape_orig, 2, NUMC_DTYPE_FLOAT32);
  float db[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  numc_array_write(b, db);

  size_t axes[] = {1, 0};
  numc_array_transpose(b, axes);
  /* b is now (3,4) non-contiguous: [[1,4,7,10],[2,5,8,11],[3,6,9,12]] */

  /* a = (1,4) — broadcasts along dim 0 */
  size_t sa[] = {1, 4};
  NumcArray *a = numc_array_create(ctx, sa, 2, NUMC_DTYPE_FLOAT32);
  float da[] = {100, 200, 300, 400};
  numc_array_write(a, da);

  size_t so[] = {3, 4};
  NumcArray *out = numc_array_zeros(ctx, so, 2, NUMC_DTYPE_FLOAT32);

  int err = numc_add(a, b, out);
  ASSERT_MSG(err == 0, "non-contiguous broadcast should succeed");

  float *r = (float *)numc_array_data(out);
  /* row 0: 100+1, 200+4, 300+7, 400+10 */
  ASSERT_MSG(r[0] == 101.0f && r[1] == 204.0f && r[2] == 307.0f &&
                 r[3] == 410.0f,
             "row 0");
  /* row 1: 100+2, 200+5, 300+8, 400+11 */
  ASSERT_MSG(r[4] == 102.0f && r[5] == 205.0f && r[6] == 308.0f &&
                 r[7] == 411.0f,
             "row 1");
  /* row 2: 100+3, 200+6, 300+9, 400+12 */
  ASSERT_MSG(r[8] == 103.0f && r[9] == 206.0f && r[10] == 309.0f &&
                 r[11] == 412.0f,
             "row 2");

  numc_ctx_free(ctx);
  return 0;
}

/* ── In-place broadcast: maximum_inplace with (3,4) and (1,4) ──────── */

static int test_broadcast_inplace(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sa[] = {3, 4}, sb[] = {1, 4};
  NumcArray *a = numc_array_create(ctx, sa, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_create(ctx, sb, 2, NUMC_DTYPE_FLOAT32);

  float da[] = {1, 5, 3, 7, 2, 6, 4, 8, 9, 0, 5, 3};
  float db[] = {3, 3, 3, 3};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_maximum_inplace(a, b);
  ASSERT_MSG(err == 0, "inplace broadcast maximum should succeed");

  float *r = (float *)numc_array_data(a);
  /* max(each, 3): row 0: 3,5,3,7  row 1: 3,6,4,8  row 2: 9,3,5,3 */
  ASSERT_MSG(r[0] == 3.0f && r[1] == 5.0f && r[2] == 3.0f && r[3] == 7.0f,
             "row 0");
  ASSERT_MSG(r[4] == 3.0f && r[5] == 6.0f && r[6] == 4.0f && r[7] == 8.0f,
             "row 1");
  ASSERT_MSG(r[8] == 9.0f && r[9] == 3.0f && r[10] == 5.0f && r[11] == 3.0f,
             "row 2");

  numc_ctx_free(ctx);
  return 0;
}

/* ── Error: incompatible broadcast shapes ──────────────────────────── */

static int test_broadcast_error_incompatible(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sa[] = {4}, sb[] = {5}, so[] = {5};
  NumcArray *a = numc_array_zeros(ctx, sa, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_zeros(ctx, sb, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, so, 1, NUMC_DTYPE_FLOAT32);

  ASSERT_MSG(numc_add(a, b, out) != 0, "(4) + (5) should fail");

  /* 2D incompatible */
  size_t sa2[] = {3, 4}, sb2[] = {2, 4}, so2[] = {3, 4};
  NumcArray *a2 = numc_array_zeros(ctx, sa2, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *b2 = numc_array_zeros(ctx, sb2, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *out2 = numc_array_zeros(ctx, so2, 2, NUMC_DTYPE_FLOAT32);

  ASSERT_MSG(numc_add(a2, b2, out2) != 0, "(3,4) + (2,4) should fail");

  numc_ctx_free(ctx);
  return 0;
}

/* ── Error: wrong output shape ─────────────────────────────────────── */

static int test_broadcast_error_wrong_output(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sa[] = {3, 1}, sb[] = {1, 4};
  NumcArray *a = numc_array_zeros(ctx, sa, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_zeros(ctx, sb, 2, NUMC_DTYPE_FLOAT32);

  /* Wrong output shape: (3,3) instead of (3,4) */
  size_t so_bad[] = {3, 3};
  NumcArray *out = numc_array_zeros(ctx, so_bad, 2, NUMC_DTYPE_FLOAT32);
  ASSERT_MSG(numc_add(a, b, out) != 0, "wrong output shape should fail");

  /* Wrong output ndim */
  size_t so_bad2[] = {12};
  NumcArray *out2 = numc_array_zeros(ctx, so_bad2, 1, NUMC_DTYPE_FLOAT32);
  ASSERT_MSG(numc_add(a, b, out2) != 0, "wrong output ndim should fail");

  numc_ctx_free(ctx);
  return 0;
}

/* ── Other ops with broadcast: mul, maximum, minimum ───────────────── */

static int test_broadcast_ops(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sa[] = {3, 1}, sb[] = {1, 4}, so[] = {3, 4};
  NumcArray *a = numc_array_create(ctx, sa, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_create(ctx, sb, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, so, 2, NUMC_DTYPE_FLOAT32);

  float da[] = {2, 3, 4};
  float db[] = {1, 5, 2, 6};
  numc_array_write(a, da);
  numc_array_write(b, db);

  /* mul: (3,1) * (1,4) → (3,4) */
  int err = numc_mul(a, b, out);
  ASSERT_MSG(err == 0, "broadcast mul should succeed");
  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 2.0f && r[1] == 10.0f && r[2] == 4.0f && r[3] == 12.0f,
             "mul row 0");
  ASSERT_MSG(r[4] == 3.0f && r[5] == 15.0f && r[6] == 6.0f && r[7] == 18.0f,
             "mul row 1");

  /* maximum: (3,1) vs (1,4) → (3,4) */
  err = numc_maximum(a, b, out);
  ASSERT_MSG(err == 0, "broadcast maximum should succeed");
  r = (float *)numc_array_data(out);
  /* max(2, [1,5,2,6]) = [2,5,2,6] */
  ASSERT_MSG(r[0] == 2.0f && r[1] == 5.0f && r[2] == 2.0f && r[3] == 6.0f,
             "maximum row 0");
  /* max(3, [1,5,2,6]) = [3,5,3,6] */
  ASSERT_MSG(r[4] == 3.0f && r[5] == 5.0f && r[6] == 3.0f && r[7] == 6.0f,
             "maximum row 1");

  /* minimum: (3,1) vs (1,4) → (3,4) */
  err = numc_minimum(a, b, out);
  ASSERT_MSG(err == 0, "broadcast minimum should succeed");
  r = (float *)numc_array_data(out);
  /* min(2, [1,5,2,6]) = [1,2,2,2] */
  ASSERT_MSG(r[0] == 1.0f && r[1] == 2.0f && r[2] == 2.0f && r[3] == 2.0f,
             "minimum row 0");
  /* min(3, [1,5,2,6]) = [1,3,2,3] */
  ASSERT_MSG(r[4] == 1.0f && r[5] == 3.0f && r[6] == 2.0f && r[7] == 3.0f,
             "minimum row 1");

  numc_ctx_free(ctx);
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== elemwise/test_broadcast ===\n\n");

  printf("Broadcasting:\n");
  RUN_TEST(test_broadcast_dim0);
  RUN_TEST(test_broadcast_dim1);
  RUN_TEST(test_broadcast_both);
  RUN_TEST(test_broadcast_scalar_like);
  RUN_TEST(test_broadcast_rank);
  RUN_TEST(test_broadcast_3d);

  printf("\nMultiple dtypes:\n");
  RUN_TEST(test_broadcast_int32);
  RUN_TEST(test_broadcast_int8);

  printf("\nNon-contiguous + inplace:\n");
  RUN_TEST(test_broadcast_noncontiguous);
  RUN_TEST(test_broadcast_inplace);

  printf("\nError cases:\n");
  RUN_TEST(test_broadcast_error_incompatible);
  RUN_TEST(test_broadcast_error_wrong_output);

  printf("\nOther ops:\n");
  RUN_TEST(test_broadcast_ops);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
