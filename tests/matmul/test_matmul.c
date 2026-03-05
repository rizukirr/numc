#include "../helpers.h"
#include <math.h>

/*
 * Tests for the unified numc_matmul() API, which dispatches to BLIS
 * (sgemm/dgemm) for large float ops and naive kernels otherwise.
 * BLIS threshold: >= 32k ops (vendored) or >= 65k ops (system).
 *
 * We use 64x64 matrices (64*64*64 = 262144 ops > 32k) to ensure
 * the BLIS path is exercised when available.
 */

#define N 64 /* 64x64 => 262144 ops, well above BLIS threshold */

/* ── Float32 BLIS path ──────────────────────────────────────────── */

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

/* ── Float64 BLIS path ──────────────────────────────────────────── */

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

/* ── Rectangular shapes through BLIS ────────────────────────────── */

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

/* ── Verify BLIS and naive agree ────────────────────────────────── */

static int test_matmul_blis_vs_naive_f32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sh[] = {N, N};

  NumcArray *a = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *c_blis = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *c_naive = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_FLOAT32);

  /* Fill with small integers to avoid float rounding differences */
  float *da = (float *)numc_array_data(a);
  float *db = (float *)numc_array_data(b);
  for (size_t i = 0; i < (size_t)N * N; i++) {
    da[i] = (float)((i % 7) + 1);
    db[i] = (float)((i % 5) + 1);
  }

  int err1 = numc_matmul(a, b, c_blis);
  int err2 = numc_matmul_naive(a, b, c_naive);
  ASSERT_MSG_CTX(err1 == 0 && err2 == 0, "both matmul paths should succeed",
                 ctx);

  float *rb = (float *)numc_array_data(c_blis);
  float *rn = (float *)numc_array_data(c_naive);
  for (size_t i = 0; i < (size_t)N * N; i++) {
    float diff = fabsf(rb[i] - rn[i]);
    ASSERT_MSG_CTX(diff < 1e-3f, "BLIS and naive f32 results should match",
                   ctx);
  }

  numc_ctx_free(ctx);
  return 0;
}

static int test_matmul_blis_vs_naive_f64(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t sh[] = {N, N};

  NumcArray *a = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_FLOAT64);
  NumcArray *b = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_FLOAT64);
  NumcArray *c_blis = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_FLOAT64);
  NumcArray *c_naive = numc_array_zeros(ctx, sh, 2, NUMC_DTYPE_FLOAT64);

  double *da = (double *)numc_array_data(a);
  double *db = (double *)numc_array_data(b);
  for (size_t i = 0; i < (size_t)N * N; i++) {
    da[i] = (double)((i % 7) + 1);
    db[i] = (double)((i % 5) + 1);
  }

  int err1 = numc_matmul(a, b, c_blis);
  int err2 = numc_matmul_naive(a, b, c_naive);
  ASSERT_MSG_CTX(err1 == 0 && err2 == 0, "both matmul paths should succeed",
                 ctx);

  double *rb = (double *)numc_array_data(c_blis);
  double *rn = (double *)numc_array_data(c_naive);
  for (size_t i = 0; i < (size_t)N * N; i++) {
    double diff = fabs(rb[i] - rn[i]);
    ASSERT_MSG_CTX(diff < 1e-10, "BLIS and naive f64 results should match",
                   ctx);
  }

  numc_ctx_free(ctx);
  return 0;
}

/* ── main ───────────────────────────────────────────────────────── */

int main(void) {
  int passes = 0, fails = 0;
  printf("=== matmul/test_matmul (unified API + BLIS path) ===\n\n");

  printf("Float32 (BLIS sgemm path):\n");
  RUN_TEST(test_matmul_f32_identity);
  RUN_TEST(test_matmul_f32_known_result);
  RUN_TEST(test_matmul_f32_rect);

  printf("\nFloat64 (BLIS dgemm path):\n");
  RUN_TEST(test_matmul_f64_identity);
  RUN_TEST(test_matmul_f64_known_result);

  printf("\nBLIS vs naive cross-validation:\n");
  RUN_TEST(test_matmul_blis_vs_naive_f32);
  RUN_TEST(test_matmul_blis_vs_naive_f64);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
