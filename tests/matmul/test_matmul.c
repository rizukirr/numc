#include "../helpers.h"
#include <math.h>

/*
 * Tests for the unified numc_matmul() API, which dispatches to packed
 * SIMD GEMM for large ops and naive kernels otherwise.
 *
 * We use 64x64 matrices to exercise the packed GEMM path.
 */

#define N 64

/* ── Float32 ──────────────────────────────────────────────────── */

static int test_matmul_f32_identity(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sh[] = {N, N};

  /* A = sequential values, B = identity matrix */
  NumcArray *a = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *c = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_FLOAT32);

  float *da = (float *)numc_array_data(a);
  float *db = (float *)numc_array_data(b);
  for (size_t i = 0; i < (size_t)N * N; i++)
    da[i] = (float)(i + 1);
  for (size_t i = 0; i < N; i++)
    db[i * N + i] = 1.0f;

  int err = numc_matmul(a, b, c);
  ASSERT_MSG_CTX(err == 0, "matmul f32 identity should succeed", ctx);

  /* A * I = A */
  float *rc = (float *)numc_array_data(c);
  for (size_t i = 0; i < (size_t)N * N; i++) {
    ASSERT_MSG_CTX(rc[i] == da[i], "matmul f32 A*I should equal A", ctx);
  }

  numc_ctx_free(ctx);
  return 0;
}

static int test_matmul_f32_known_result(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sha[] = {2, N};
  size_t shb[] = {N, 2};
  size_t shc[] = {2, 2};

  NumcArray *a = numc_array_zeros(ctx, sha, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_zeros(ctx, shb, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *c = numc_array_zeros(ctx, shc, 2, NUMC_DTYPE_FLOAT32);

  /* a[i][j] = 1.0 for all, b[i][j] = 1.0 for all
   * C = A @ B => each element = N (dot product of N ones) */
  float *da = (float *)numc_array_data(a);
  float *db = (float *)numc_array_data(b);
  for (size_t i = 0; i < (size_t)2 * N; i++)
    da[i] = 1.0f;
  for (size_t i = 0; i < (size_t)N * 2; i++)
    db[i] = 1.0f;

  int err = numc_matmul(a, b, c);
  ASSERT_MSG_CTX(err == 0, "matmul f32 ones should succeed", ctx);

  float *rc = (float *)numc_array_data(c);
  for (size_t i = 0; i < 4; i++) {
    ASSERT_MSG_CTX(rc[i] == (float)N, "matmul f32 ones result should equal N",
                   ctx);
  }

  numc_ctx_free(ctx);
  return 0;
}

/* ── Float64 ──────────────────────────────────────────────────── */

static int test_matmul_f64_identity(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sh[] = {N, N};

  NumcArray *a = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_FLOAT64);
  NumcArray *b = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_FLOAT64);
  NumcArray *c = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_FLOAT64);

  double *da = (double *)numc_array_data(a);
  double *db = (double *)numc_array_data(b);
  for (size_t i = 0; i < (size_t)N * N; i++)
    da[i] = (double)(i + 1);
  for (size_t i = 0; i < N; i++)
    db[i * N + i] = 1.0;

  int err = numc_matmul(a, b, c);
  ASSERT_MSG_CTX(err == 0, "matmul f64 identity should succeed", ctx);

  double *rc = (double *)numc_array_data(c);
  for (size_t i = 0; i < (size_t)N * N; i++) {
    ASSERT_MSG_CTX(rc[i] == da[i], "matmul f64 A*I should equal A", ctx);
  }

  numc_ctx_free(ctx);
  return 0;
}

static int test_matmul_f64_known_result(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sha[] = {2, N};
  size_t shb[] = {N, 2};
  size_t shc[] = {2, 2};

  NumcArray *a = numc_array_zeros(ctx, sha, 2, NUMC_DTYPE_FLOAT64);
  NumcArray *b = numc_array_zeros(ctx, shb, 2, NUMC_DTYPE_FLOAT64);
  NumcArray *c = numc_array_zeros(ctx, shc, 2, NUMC_DTYPE_FLOAT64);

  double *da = (double *)numc_array_data(a);
  double *db = (double *)numc_array_data(b);
  for (size_t i = 0; i < (size_t)2 * N; i++)
    da[i] = 1.0;
  for (size_t i = 0; i < (size_t)N * 2; i++)
    db[i] = 1.0;

  int err = numc_matmul(a, b, c);
  ASSERT_MSG_CTX(err == 0, "matmul f64 ones should succeed", ctx);

  double *rc = (double *)numc_array_data(c);
  for (size_t i = 0; i < 4; i++) {
    ASSERT_MSG_CTX(rc[i] == (double)N, "matmul f64 ones result should equal N",
                   ctx);
  }

  numc_ctx_free(ctx);
  return 0;
}

/* ── Rectangular shapes ────────────────────────────────────────── */

static int test_matmul_f32_rect(void) {
  /* (4 x 128) @ (128 x 8) = (4 x 8), ops = 4*128*8 = 4096 * ... > 32k */
  size_t M = 4, K = 128, NN = 64; /* 4*128*64 = 32768, exactly at threshold */
  NumcCtx *ctx = numc_ctx_create();
  size_t sha[] = {M, K}, shb[] = {K, NN}, shc[] = {M, NN};

  NumcArray *a = numc_array_zeros(ctx, sha, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_zeros(ctx, shb, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *c = numc_array_zeros(ctx, shc, 2, NUMC_DTYPE_FLOAT32);

  /* Fill with 1s: each output element = K */
  float *da = (float *)numc_array_data(a);
  float *db = (float *)numc_array_data(b);
  for (size_t i = 0; i < (size_t)M * K; i++)
    da[i] = 1.0f;
  for (size_t i = 0; i < (size_t)K * NN; i++)
    db[i] = 1.0f;

  int err = numc_matmul(a, b, c);
  ASSERT_MSG_CTX(err == 0, "matmul f32 rect should succeed", ctx);

  float *rc = (float *)numc_array_data(c);
  for (size_t i = 0; i < (size_t)M * NN; i++) {
    ASSERT_MSG_CTX(rc[i] == (float)K, "matmul f32 rect result should equal K",
                   ctx);
  }

  numc_ctx_free(ctx);
  return 0;
}

/* ── Verify GEMM and naive agree ───────────────────────────────── */

static int test_matmul_gemm_vs_naive_f32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sh[] = {N, N};

  NumcArray *a = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *c_gemm = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *c_naive = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_FLOAT32);

  /* Fill with small integers to avoid float rounding differences */
  float *da = (float *)numc_array_data(a);
  float *db = (float *)numc_array_data(b);
  for (size_t i = 0; i < (size_t)N * N; i++) {
    da[i] = (float)((i % 7) + 1);
    db[i] = (float)((i % 5) + 1);
  }

  int err1 = numc_matmul(a, b, c_gemm);
  int err2 = numc_matmul_naive(a, b, c_naive);
  ASSERT_MSG_CTX(err1 == 0 && err2 == 0, "both matmul paths should succeed",
                 ctx);

  float *rb = (float *)numc_array_data(c_gemm);
  float *rn = (float *)numc_array_data(c_naive);
  for (size_t i = 0; i < (size_t)N * N; i++) {
    float diff = fabsf(rb[i] - rn[i]);
    ASSERT_MSG_CTX(diff < 1e-3f, "BLIS and naive f32 results should match",
                   ctx);
  }

  numc_ctx_free(ctx);
  return 0;
}

static int test_matmul_gemm_vs_naive_f64(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sh[] = {N, N};

  NumcArray *a = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_FLOAT64);
  NumcArray *b = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_FLOAT64);
  NumcArray *c_gemm = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_FLOAT64);
  NumcArray *c_naive = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_FLOAT64);

  double *da = (double *)numc_array_data(a);
  double *db = (double *)numc_array_data(b);
  for (size_t i = 0; i < (size_t)N * N; i++) {
    da[i] = (double)((i % 7) + 1);
    db[i] = (double)((i % 5) + 1);
  }

  int err1 = numc_matmul(a, b, c_gemm);
  int err2 = numc_matmul_naive(a, b, c_naive);
  ASSERT_MSG_CTX(err1 == 0 && err2 == 0, "both matmul paths should succeed",
                 ctx);

  double *rb = (double *)numc_array_data(c_gemm);
  double *rn = (double *)numc_array_data(c_naive);
  for (size_t i = 0; i < (size_t)N * N; i++) {
    double diff = fabs(rb[i] - rn[i]);
    ASSERT_MSG_CTX(diff < 1e-10, "BLIS and naive f64 results should match",
                   ctx);
  }

  numc_ctx_free(ctx);
  return 0;
}

static int test_matmul_packed_256_f32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {256, 256};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *c_fast = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *c_naive = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);

  float *ad = (float *)numc_array_data(a);
  float *bd = (float *)numc_array_data(b);
  for (size_t i = 0; i < 256 * 256; i++) {
    ad[i] = (float)((i % 7) + 1);
    bd[i] = (float)((i % 5) + 1);
  }

  int err1 = numc_matmul(a, b, c_fast);
  int err2 = numc_matmul_naive(a, b, c_naive);
  ASSERT_MSG_CTX(err1 == 0, "matmul failed", ctx);
  ASSERT_MSG_CTX(err2 == 0, "matmul_naive failed", ctx);

  const float *r1 = (const float *)numc_array_data(c_fast);
  const float *r2 = (const float *)numc_array_data(c_naive);
  for (size_t i = 0; i < 256 * 256; i++) {
    float diff = r1[i] - r2[i];
    if (diff < 0)
      diff = -diff;
    ASSERT_MSG_CTX(diff < 1e-2f, "packed vs naive mismatch at 256x256", ctx);
  }

  numc_ctx_free(ctx);
  return 0;
}

static int test_matmul_packed_256_f64(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {256, 256};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT64);
  NumcArray *b = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT64);
  NumcArray *c_fast = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT64);
  NumcArray *c_naive = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT64);

  double *ad = (double *)numc_array_data(a);
  double *bd = (double *)numc_array_data(b);
  for (size_t i = 0; i < 256 * 256; i++) {
    ad[i] = (double)((i % 7) + 1);
    bd[i] = (double)((i % 5) + 1);
  }

  int err1 = numc_matmul(a, b, c_fast);
  int err2 = numc_matmul_naive(a, b, c_naive);
  ASSERT_MSG_CTX(err1 == 0, "matmul failed", ctx);
  ASSERT_MSG_CTX(err2 == 0, "matmul_naive failed", ctx);

  const double *r1 = (const double *)numc_array_data(c_fast);
  const double *r2 = (const double *)numc_array_data(c_naive);
  for (size_t i = 0; i < 256 * 256; i++) {
    double diff = r1[i] - r2[i];
    if (diff < 0)
      diff = -diff;
    ASSERT_MSG_CTX(diff < 1e-6, "packed vs naive mismatch at 256x256", ctx);
  }

  numc_ctx_free(ctx);
  return 0;
}

/* ── Int32 ────────────────────────────────────────────────────── */

static int test_matmul_i32_identity(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sh[] = {N, N};
  NumcArray *a = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_INT32);
  NumcArray *c = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_INT32);
  int32_t *da = (int32_t *)numc_array_data(a);
  int32_t *db = (int32_t *)numc_array_data(b);
  for (size_t i = 0; i < (size_t)N * N; i++)
    da[i] = (int32_t)((i % 10) + 1);
  for (size_t i = 0; i < N; i++)
    db[i * N + i] = 1;
  int err = numc_matmul(a, b, c);
  ASSERT_MSG_CTX(err == 0, "matmul i32 identity should succeed", ctx);
  int32_t *rc = (int32_t *)numc_array_data(c);
  for (size_t i = 0; i < (size_t)N * N; i++)
    ASSERT_MSG_CTX(rc[i] == da[i], "matmul i32 A*I should equal A", ctx);
  numc_ctx_free(ctx);
  return 0;
}

static int test_matmul_gemm_vs_naive_i32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sh[] = {N, N};
  NumcArray *a = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_INT32);
  NumcArray *c_gemm = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_INT32);
  NumcArray *c_naive = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_INT32);
  int32_t *da = (int32_t *)numc_array_data(a);
  int32_t *db = (int32_t *)numc_array_data(b);
  for (size_t i = 0; i < (size_t)N * N; i++) {
    da[i] = (int32_t)((i % 7) + 1);
    db[i] = (int32_t)((i % 5) + 1);
  }
  int err1 = numc_matmul(a, b, c_gemm);
  int err2 = numc_matmul_naive(a, b, c_naive);
  ASSERT_MSG_CTX(err1 == 0 && err2 == 0, "both matmul paths should succeed", ctx);
  int32_t *rg = (int32_t *)numc_array_data(c_gemm);
  int32_t *rn = (int32_t *)numc_array_data(c_naive);
  for (size_t i = 0; i < (size_t)N * N; i++)
    ASSERT_MSG_CTX(rg[i] == rn[i], "GEMM and naive i32 results should match", ctx);
  numc_ctx_free(ctx);
  return 0;
}

/* ── UInt32 ───────────────────────────────────────────────────── */

static int test_matmul_u32_identity(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sh[] = {N, N};
  NumcArray *a = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_UINT32);
  NumcArray *b = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_UINT32);
  NumcArray *c = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_UINT32);
  uint32_t *da = (uint32_t *)numc_array_data(a);
  uint32_t *db = (uint32_t *)numc_array_data(b);
  for (size_t i = 0; i < (size_t)N * N; i++)
    da[i] = (uint32_t)((i % 10) + 1);
  for (size_t i = 0; i < N; i++)
    db[i * N + i] = 1;
  int err = numc_matmul(a, b, c);
  ASSERT_MSG_CTX(err == 0, "matmul u32 identity should succeed", ctx);
  uint32_t *rc = (uint32_t *)numc_array_data(c);
  for (size_t i = 0; i < (size_t)N * N; i++)
    ASSERT_MSG_CTX(rc[i] == da[i], "matmul u32 A*I should equal A", ctx);
  numc_ctx_free(ctx);
  return 0;
}

/* ── Int16 ────────────────────────────────────────────────────── */

static int test_matmul_i16_identity(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sh[] = {N, N};
  NumcArray *a = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_INT16);
  NumcArray *b = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_INT16);
  NumcArray *c = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_INT16);
  int16_t *da = (int16_t *)numc_array_data(a);
  int16_t *db = (int16_t *)numc_array_data(b);
  for (size_t i = 0; i < (size_t)N * N; i++)
    da[i] = (int16_t)((i % 10) + 1);
  for (size_t i = 0; i < N; i++)
    db[i * N + i] = 1;
  int err = numc_matmul(a, b, c);
  ASSERT_MSG_CTX(err == 0, "matmul i16 identity should succeed", ctx);
  int16_t *rc = (int16_t *)numc_array_data(c);
  for (size_t i = 0; i < (size_t)N * N; i++)
    ASSERT_MSG_CTX(rc[i] == da[i], "matmul i16 A*I should equal A", ctx);
  numc_ctx_free(ctx);
  return 0;
}

static int test_matmul_gemm_vs_naive_i16(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sh[] = {N, N};
  NumcArray *a = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_INT16);
  NumcArray *b = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_INT16);
  NumcArray *c_gemm = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_INT16);
  NumcArray *c_naive = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_INT16);
  int16_t *da = (int16_t *)numc_array_data(a);
  int16_t *db = (int16_t *)numc_array_data(b);
  for (size_t i = 0; i < (size_t)N * N; i++) {
    da[i] = (int16_t)((i % 7) + 1);
    db[i] = (int16_t)((i % 5) + 1);
  }
  int err1 = numc_matmul(a, b, c_gemm);
  int err2 = numc_matmul_naive(a, b, c_naive);
  ASSERT_MSG_CTX(err1 == 0 && err2 == 0, "both should succeed", ctx);
  int16_t *rg = (int16_t *)numc_array_data(c_gemm);
  int16_t *rn = (int16_t *)numc_array_data(c_naive);
  for (size_t i = 0; i < (size_t)N * N; i++)
    ASSERT_MSG_CTX(rg[i] == rn[i], "GEMM and naive i16 should match", ctx);
  numc_ctx_free(ctx);
  return 0;
}

/* ── UInt16 ───────────────────────────────────────────────────── */

static int test_matmul_u16_identity(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sh[] = {N, N};
  NumcArray *a = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_UINT16);
  NumcArray *b = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_UINT16);
  NumcArray *c = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_UINT16);
  uint16_t *da = (uint16_t *)numc_array_data(a);
  uint16_t *db = (uint16_t *)numc_array_data(b);
  for (size_t i = 0; i < (size_t)N * N; i++)
    da[i] = (uint16_t)((i % 10) + 1);
  for (size_t i = 0; i < N; i++)
    db[i * N + i] = 1;
  int err = numc_matmul(a, b, c);
  ASSERT_MSG_CTX(err == 0, "matmul u16 identity should succeed", ctx);
  uint16_t *rc = (uint16_t *)numc_array_data(c);
  for (size_t i = 0; i < (size_t)N * N; i++)
    ASSERT_MSG_CTX(rc[i] == da[i], "matmul u16 A*I should equal A", ctx);
  numc_ctx_free(ctx);
  return 0;
}

/* ── Int64 ────────────────────────────────────────────────────── */

static int test_matmul_i64_identity(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sh[] = {N, N};
  NumcArray *a = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_INT64);
  NumcArray *b = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_INT64);
  NumcArray *c = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_INT64);
  int64_t *da = (int64_t *)numc_array_data(a);
  int64_t *db = (int64_t *)numc_array_data(b);
  for (size_t i = 0; i < (size_t)N * N; i++)
    da[i] = (int64_t)((i % 10) + 1);
  for (size_t i = 0; i < N; i++)
    db[i * N + i] = 1;
  int err = numc_matmul(a, b, c);
  ASSERT_MSG_CTX(err == 0, "matmul i64 identity should succeed", ctx);
  int64_t *rc = (int64_t *)numc_array_data(c);
  for (size_t i = 0; i < (size_t)N * N; i++)
    ASSERT_MSG_CTX(rc[i] == da[i], "matmul i64 A*I should equal A", ctx);
  numc_ctx_free(ctx);
  return 0;
}

static int test_matmul_gemm_vs_naive_i64(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sh[] = {N, N};
  NumcArray *a = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_INT64);
  NumcArray *b = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_INT64);
  NumcArray *c_gemm = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_INT64);
  NumcArray *c_naive = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_INT64);
  int64_t *da = (int64_t *)numc_array_data(a);
  int64_t *db = (int64_t *)numc_array_data(b);
  for (size_t i = 0; i < (size_t)N * N; i++) {
    da[i] = (int64_t)((i % 7) + 1);
    db[i] = (int64_t)((i % 5) + 1);
  }
  int err1 = numc_matmul(a, b, c_gemm);
  int err2 = numc_matmul_naive(a, b, c_naive);
  ASSERT_MSG_CTX(err1 == 0 && err2 == 0, "both should succeed", ctx);
  int64_t *rg = (int64_t *)numc_array_data(c_gemm);
  int64_t *rn = (int64_t *)numc_array_data(c_naive);
  for (size_t i = 0; i < (size_t)N * N; i++)
    ASSERT_MSG_CTX(rg[i] == rn[i], "GEMM and naive i64 should match", ctx);
  numc_ctx_free(ctx);
  return 0;
}

/* ── UInt64 ───────────────────────────────────────────────────── */

static int test_matmul_u64_identity(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sh[] = {N, N};
  NumcArray *a = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_UINT64);
  NumcArray *b = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_UINT64);
  NumcArray *c = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_UINT64);
  uint64_t *da = (uint64_t *)numc_array_data(a);
  uint64_t *db = (uint64_t *)numc_array_data(b);
  for (size_t i = 0; i < (size_t)N * N; i++)
    da[i] = (uint64_t)((i % 10) + 1);
  for (size_t i = 0; i < N; i++)
    db[i * N + i] = 1;
  int err = numc_matmul(a, b, c);
  ASSERT_MSG_CTX(err == 0, "matmul u64 identity should succeed", ctx);
  uint64_t *rc = (uint64_t *)numc_array_data(c);
  for (size_t i = 0; i < (size_t)N * N; i++)
    ASSERT_MSG_CTX(rc[i] == da[i], "matmul u64 A*I should equal A", ctx);
  numc_ctx_free(ctx);
  return 0;
}

/* ── Int8 ─────────────────────────────────────────────────────── */

static int test_matmul_i8_identity(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sh[] = {N, N};
  NumcArray *a = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_INT8);
  NumcArray *b = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_INT8);
  NumcArray *c = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_INT8);
  int8_t *da = (int8_t *)numc_array_data(a);
  int8_t *db = (int8_t *)numc_array_data(b);
  for (size_t i = 0; i < (size_t)N * N; i++)
    da[i] = (int8_t)((i % 5) + 1);
  for (size_t i = 0; i < N; i++)
    db[i * N + i] = 1;
  int err = numc_matmul(a, b, c);
  ASSERT_MSG_CTX(err == 0, "matmul i8 identity should succeed", ctx);
  int8_t *rc = (int8_t *)numc_array_data(c);
  for (size_t i = 0; i < (size_t)N * N; i++)
    ASSERT_MSG_CTX(rc[i] == da[i], "matmul i8 A*I should equal A", ctx);
  numc_ctx_free(ctx);
  return 0;
}

static int test_matmul_gemm_vs_naive_i8(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sh[] = {N, N};
  NumcArray *a = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_INT8);
  NumcArray *b = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_INT8);
  NumcArray *c_gemm = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_INT8);
  NumcArray *c_naive = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_INT8);
  int8_t *da = (int8_t *)numc_array_data(a);
  int8_t *db = (int8_t *)numc_array_data(b);
  for (size_t i = 0; i < (size_t)N * N; i++) {
    da[i] = (int8_t)((i % 3) + 1);
    db[i] = (int8_t)((i % 2) + 1);
  }
  int err1 = numc_matmul(a, b, c_gemm);
  int err2 = numc_matmul_naive(a, b, c_naive);
  ASSERT_MSG_CTX(err1 == 0 && err2 == 0, "both should succeed", ctx);
  int8_t *rg = (int8_t *)numc_array_data(c_gemm);
  int8_t *rn = (int8_t *)numc_array_data(c_naive);
  for (size_t i = 0; i < (size_t)N * N; i++)
    ASSERT_MSG_CTX(rg[i] == rn[i], "GEMM and naive i8 should match", ctx);
  numc_ctx_free(ctx);
  return 0;
}

/* ── UInt8 ────────────────────────────────────────────────────── */

static int test_matmul_u8_identity(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sh[] = {N, N};
  NumcArray *a = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_UINT8);
  NumcArray *b = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_UINT8);
  NumcArray *c = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_UINT8);
  uint8_t *da = (uint8_t *)numc_array_data(a);
  uint8_t *db = (uint8_t *)numc_array_data(b);
  for (size_t i = 0; i < (size_t)N * N; i++)
    da[i] = (uint8_t)((i % 5) + 1);
  for (size_t i = 0; i < N; i++)
    db[i * N + i] = 1;
  int err = numc_matmul(a, b, c);
  ASSERT_MSG_CTX(err == 0, "matmul u8 identity should succeed", ctx);
  uint8_t *rc = (uint8_t *)numc_array_data(c);
  for (size_t i = 0; i < (size_t)N * N; i++)
    ASSERT_MSG_CTX(rc[i] == da[i], "matmul u8 A*I should equal A", ctx);
  numc_ctx_free(ctx);
  return 0;
}

/* ── main ───────────────────────────────────────────────────────── */

int main(void) {
  int passes = 0, fails = 0;
  printf("=== matmul/test_matmul (unified API) ===\n\n");

  printf("Float32:\n");
  RUN_TEST(test_matmul_f32_identity);
  RUN_TEST(test_matmul_f32_known_result);
  RUN_TEST(test_matmul_f32_rect);

  printf("\nFloat64:\n");
  RUN_TEST(test_matmul_f64_identity);
  RUN_TEST(test_matmul_f64_known_result);

  printf("\nPacked vs naive cross-validation:\n");
  RUN_TEST(test_matmul_gemm_vs_naive_f32);
  RUN_TEST(test_matmul_gemm_vs_naive_f64);

  printf("\nPacked GEMM 256x256 cross-validation:\n");
  RUN_TEST(test_matmul_packed_256_f32);
  RUN_TEST(test_matmul_packed_256_f64);

  printf("\nInt32:\n");
  RUN_TEST(test_matmul_i32_identity);
  RUN_TEST(test_matmul_gemm_vs_naive_i32);

  printf("\nUInt32:\n");
  RUN_TEST(test_matmul_u32_identity);

  printf("\nInt16:\n");
  RUN_TEST(test_matmul_i16_identity);
  RUN_TEST(test_matmul_gemm_vs_naive_i16);

  printf("\nUInt16:\n");
  RUN_TEST(test_matmul_u16_identity);

  printf("\nInt64:\n");
  RUN_TEST(test_matmul_i64_identity);
  RUN_TEST(test_matmul_gemm_vs_naive_i64);

  printf("\nUInt64:\n");
  RUN_TEST(test_matmul_u64_identity);

  printf("\nInt8:\n");
  RUN_TEST(test_matmul_i8_identity);
  RUN_TEST(test_matmul_gemm_vs_naive_i8);

  printf("\nUInt8:\n");
  RUN_TEST(test_matmul_u8_identity);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
