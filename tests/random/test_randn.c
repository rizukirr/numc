#include "../helpers.h"

#include <math.h>
#include <stdint.h>

/* ── Box-Muller spare caching — consecutive calls differ ────────────*/

static int test_randn_consecutive_differ(void) {
  NumcCtx *ctx = numc_ctx_create();
  numc_manual_seed(42);

  NumcArray *a =
      numc_array_randn(ctx, (size_t[]){64}, 1, NUMC_DTYPE_FLOAT32);
  ASSERT_MSG(a != NULL, "randn float32 must not be NULL");

  float *p = (float *)numc_array_data(a);
  /* With Box-Muller spare, pairs of elements come from the same call.
   * Across 64 elements we expect non-identical adjacent pairs. */
  int pairs_differ = 0;
  for (size_t i = 0; i + 1 < 64; i += 2) {
    if (p[i] != p[i + 1])
      pairs_differ++;
  }
  ASSERT_MSG(pairs_differ > 0,
             "randn pairs should not all be equal (spare caching)");

  numc_ctx_free(ctx);
  return 0;
}

/* ── float32 N(0,1): mean ~ 0, std ~ 1 over large sample ───────────*/

static int test_randn_float32_stats(void) {
  NumcCtx *ctx = numc_ctx_create();
  numc_manual_seed(1234);

  const size_t n = 4096;
  NumcArray *a =
      numc_array_randn(ctx, (size_t[]){n}, 1, NUMC_DTYPE_FLOAT32);
  ASSERT_MSG(a != NULL, "randn float32 large must not be NULL");

  float *p = (float *)numc_array_data(a);
  double sum = 0.0, sum2 = 0.0;
  for (size_t i = 0; i < n; i++) {
    sum += p[i];
    sum2 += (double)p[i] * p[i];
  }
  double mean = sum / (double)n;
  double var  = sum2 / (double)n - mean * mean;
  double std  = sqrt(var);

  /* With n=4096 the CLT gives very tight bounds:
   * |mean| < 3 * (1/sqrt(n)) ~ 0.047   =>  use 0.15 for safety
   * std in (0.9, 1.1)                                              */
  ASSERT_MSG(fabs(mean) < 0.15,
             "randn float32 mean should be close to 0");
  ASSERT_MSG(std > 0.9 && std < 1.1,
             "randn float32 std should be close to 1");

  numc_ctx_free(ctx);
  return 0;
}

/* ── float64 N(0,1): mean ~ 0, std ~ 1 over large sample ───────────*/

static int test_randn_float64_stats(void) {
  NumcCtx *ctx = numc_ctx_create();
  numc_manual_seed(5678);

  const size_t n = 4096;
  NumcArray *a =
      numc_array_randn(ctx, (size_t[]){n}, 1, NUMC_DTYPE_FLOAT64);
  ASSERT_MSG(a != NULL, "randn float64 large must not be NULL");

  double *p = (double *)numc_array_data(a);
  double sum = 0.0, sum2 = 0.0;
  for (size_t i = 0; i < n; i++) {
    sum  += p[i];
    sum2 += p[i] * p[i];
  }
  double mean = sum / (double)n;
  double var  = sum2 / (double)n - mean * mean;
  double std  = sqrt(var);

  ASSERT_MSG(fabs(mean) < 0.15,
             "randn float64 mean should be close to 0");
  ASSERT_MSG(std > 0.9 && std < 1.1,
             "randn float64 std should be close to 1");

  numc_ctx_free(ctx);
  return 0;
}

/* ── reproducibility ────────────────────────────────────────────────*/

static int test_randn_seed_reproducible(void) {
  NumcCtx *ctx = numc_ctx_create();

  numc_manual_seed(99);
  NumcArray *a =
      numc_array_randn(ctx, (size_t[]){16}, 1, NUMC_DTYPE_FLOAT64);

  numc_manual_seed(99);
  NumcArray *b =
      numc_array_randn(ctx, (size_t[]){16}, 1, NUMC_DTYPE_FLOAT64);

  ASSERT_MSG(a != NULL && b != NULL, "randn arrays must not be NULL");

  double *pa = (double *)numc_array_data(a);
  double *pb = (double *)numc_array_data(b);
  for (size_t i = 0; i < 16; i++) {
    ASSERT_MSG(pa[i] == pb[i],
               "same seed must produce same randn sequence");
  }

  numc_ctx_free(ctx);
  return 0;
}

/* ── integer dtypes produce near-zero values (N(0,1) cast) ─────────*/

static int test_randn_int32_near_zero(void) {
  NumcCtx *ctx = numc_ctx_create();
  numc_manual_seed(77);

  const size_t n = 512;
  NumcArray *a =
      numc_array_randn(ctx, (size_t[]){n}, 1, NUMC_DTYPE_INT32);
  ASSERT_MSG(a != NULL, "randn int32 must not be NULL");

  int32_t *p = (int32_t *)numc_array_data(a);
  /* N(0,1) cast to int: ~99.7% of values in [-3, 3] */
  size_t in_range = 0;
  for (size_t i = 0; i < n; i++) {
    if (p[i] >= -3 && p[i] <= 3)
      in_range++;
  }
  /* Expect at least 95% in [-3, 3] */
  ASSERT_MSG(in_range > n * 95 / 100,
             "randn int32 should mostly be near zero");

  numc_ctx_free(ctx);
  return 0;
}

/* ── multi-dimensional ──────────────────────────────────────────────*/

static int test_randn_2d(void) {
  NumcCtx *ctx = numc_ctx_create();
  numc_manual_seed(33);

  NumcArray *a =
      numc_array_randn(ctx, (size_t[]){8, 16}, 2, NUMC_DTYPE_FLOAT32);
  ASSERT_MSG(a != NULL, "randn 2D must not be NULL");
  ASSERT_MSG(numc_array_size(a) == 128, "randn 2D size should be 128");
  ASSERT_MSG(numc_array_ndim(a) == 2, "randn 2D ndim should be 2");

  numc_ctx_free(ctx);
  return 0;
}

/* ── error cases ────────────────────────────────────────────────────*/

static int test_randn_null_ctx(void) {
  NumcArray *a =
      numc_array_randn(NULL, (size_t[]){4}, 1, NUMC_DTYPE_FLOAT32);
  ASSERT_MSG(a == NULL, "randn with NULL ctx must return NULL");
  return 0;
}

static int test_randn_null_shape(void) {
  NumcCtx *ctx = numc_ctx_create();
  NumcArray *a = numc_array_randn(ctx, NULL, 1, NUMC_DTYPE_FLOAT32);
  ASSERT_MSG(a == NULL, "randn with NULL shape must return NULL");
  numc_ctx_free(ctx);
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== random/test_randn ===\n\n");

  printf("numc_array_randn correctness:\n");
  RUN_TEST(test_randn_consecutive_differ);
  RUN_TEST(test_randn_float32_stats);
  RUN_TEST(test_randn_float64_stats);
  RUN_TEST(test_randn_seed_reproducible);

  printf("\nnumc_array_randn integer dtypes:\n");
  RUN_TEST(test_randn_int32_near_zero);

  printf("\nnumc_array_randn multi-dimensional:\n");
  RUN_TEST(test_randn_2d);

  printf("\nnumc_array_randn error cases:\n");
  RUN_TEST(test_randn_null_ctx);
  RUN_TEST(test_randn_null_shape);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
