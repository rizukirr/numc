#include "../helpers.h"

#include <math.h>
#include <stdint.h>

/* ── He (Kaiming) initialization ────────────────────────────────────
 *
 * Expected: N(0, sqrt(2/fan_in))
 * => mean ~ 0, std ~ sqrt(2/fan_in)
 */

static int test_he_float32_shape(void) {
  NumcCtx *ctx = numc_ctx_create();
  numc_manual_seed(42);

  NumcArray *w = numc_array_random_he(ctx, (size_t[]){256, 128}, 2,
                                      NUMC_DTYPE_FLOAT32, 256);
  ASSERT_MSG(w != NULL, "he float32 must not be NULL");
  ASSERT_MSG(numc_array_ndim(w) == 2, "he ndim should be 2");
  ASSERT_MSG(numc_array_size(w) == 256 * 128, "he size should be 256*128");

  numc_ctx_free(ctx);
  return 0;
}

static int test_he_float32_stats(void) {
  NumcCtx *ctx = numc_ctx_create();
  numc_manual_seed(1111);

  const size_t fan_in = 512;
  const size_t n = 4096;
  NumcArray *w =
      numc_array_random_he(ctx, (size_t[]){n}, 1, NUMC_DTYPE_FLOAT32, fan_in);
  ASSERT_MSG(w != NULL, "he float32 stats must not be NULL");

  float *p = (float *)numc_array_data(w);
  double sum = 0.0, sum2 = 0.0;
  for (size_t i = 0; i < n; i++) {
    sum += p[i];
    sum2 += (double)p[i] * p[i];
  }
  double mean = sum / (double)n;
  double var = sum2 / (double)n - mean * mean;
  double std = sqrt(var);

  double expected_std = sqrt(2.0 / (double)fan_in);

  ASSERT_MSG(fabs(mean) < 0.05, "he float32 mean should be close to 0");
  /* Allow ±20% tolerance on std */
  ASSERT_MSG(fabs(std - expected_std) < 0.2 * expected_std,
             "he float32 std should be close to sqrt(2/fan_in)");

  numc_ctx_free(ctx);
  return 0;
}

static int test_he_float64_stats(void) {
  NumcCtx *ctx = numc_ctx_create();
  numc_manual_seed(2222);

  const size_t fan_in = 256;
  const size_t n = 4096;
  NumcArray *w =
      numc_array_random_he(ctx, (size_t[]){n}, 1, NUMC_DTYPE_FLOAT64, fan_in);
  ASSERT_MSG(w != NULL, "he float64 stats must not be NULL");

  double *p = (double *)numc_array_data(w);
  double sum = 0.0, sum2 = 0.0;
  for (size_t i = 0; i < n; i++) {
    sum += p[i];
    sum2 += p[i] * p[i];
  }
  double mean = sum / (double)n;
  double var = sum2 / (double)n - mean * mean;
  double std = sqrt(var);

  double expected_std = sqrt(2.0 / (double)fan_in);

  ASSERT_MSG(fabs(mean) < 0.05, "he float64 mean should be close to 0");
  ASSERT_MSG(fabs(std - expected_std) < 0.2 * expected_std,
             "he float64 std should be close to sqrt(2/fan_in)");

  numc_ctx_free(ctx);
  return 0;
}

static int test_he_error_zero_fan_in(void) {
  NumcCtx *ctx = numc_ctx_create();
  NumcArray *w =
      numc_array_random_he(ctx, (size_t[]){4}, 1, NUMC_DTYPE_FLOAT32, 0);
  ASSERT_MSG(w == NULL, "he with fan_in=0 must return NULL");
  numc_ctx_free(ctx);
  return 0;
}

/* ── Xavier (Glorot) initialization ─────────────────────────────────
 *
 * Expected: uniform [-limit, limit), limit = sqrt(6/(fan_in+fan_out))
 * => mean ~ 0, all values in (-limit, limit)
 */

static int test_xavier_float32_shape(void) {
  NumcCtx *ctx = numc_ctx_create();
  numc_manual_seed(42);

  NumcArray *w = numc_array_random_xavier(ctx, (size_t[]){128, 64}, 2,
                                          NUMC_DTYPE_FLOAT32, 128, 64);
  ASSERT_MSG(w != NULL, "xavier float32 must not be NULL");
  ASSERT_MSG(numc_array_ndim(w) == 2, "xavier ndim should be 2");
  ASSERT_MSG(numc_array_size(w) == 128 * 64, "xavier size should be 128*64");

  numc_ctx_free(ctx);
  return 0;
}

static int test_xavier_float32_range(void) {
  NumcCtx *ctx = numc_ctx_create();
  numc_manual_seed(3333);

  const size_t fan_in = 128, fan_out = 64;
  const size_t n = 2048;
  NumcArray *w = numc_array_random_xavier(ctx, (size_t[]){n}, 1,
                                          NUMC_DTYPE_FLOAT32, fan_in, fan_out);
  ASSERT_MSG(w != NULL, "xavier float32 range must not be NULL");

  double limit = sqrt(6.0 / (double)(fan_in + fan_out));
  float flimit = (float)limit;

  float *p = (float *)numc_array_data(w);
  for (size_t i = 0; i < n; i++) {
    ASSERT_MSG(p[i] >= -flimit && p[i] < flimit,
               "xavier float32 values must be in [-limit, limit)");
  }

  numc_ctx_free(ctx);
  return 0;
}

static int test_xavier_float32_stats(void) {
  NumcCtx *ctx = numc_ctx_create();
  numc_manual_seed(4444);

  const size_t fan_in = 256, fan_out = 128;
  const size_t n = 4096;
  NumcArray *w = numc_array_random_xavier(ctx, (size_t[]){n}, 1,
                                          NUMC_DTYPE_FLOAT32, fan_in, fan_out);
  ASSERT_MSG(w != NULL, "xavier float32 stats must not be NULL");

  float *p = (float *)numc_array_data(w);
  double sum = 0.0;
  for (size_t i = 0; i < n; i++)
    sum += p[i];
  double mean = sum / (double)n;

  ASSERT_MSG(fabs(mean) < 0.05, "xavier float32 mean should be close to 0");

  numc_ctx_free(ctx);
  return 0;
}

static int test_xavier_float64_range(void) {
  NumcCtx *ctx = numc_ctx_create();
  numc_manual_seed(5555);

  const size_t fan_in = 64, fan_out = 32;
  const size_t n = 1024;
  NumcArray *w = numc_array_random_xavier(ctx, (size_t[]){n}, 1,
                                          NUMC_DTYPE_FLOAT64, fan_in, fan_out);
  ASSERT_MSG(w != NULL, "xavier float64 range must not be NULL");

  double limit = sqrt(6.0 / (double)(fan_in + fan_out));
  double *p = (double *)numc_array_data(w);
  for (size_t i = 0; i < n; i++) {
    ASSERT_MSG(p[i] >= -limit && p[i] < limit,
               "xavier float64 values must be in [-limit, limit)");
  }

  numc_ctx_free(ctx);
  return 0;
}

static int test_xavier_error_zero_fan(void) {
  NumcCtx *ctx = numc_ctx_create();

  NumcArray *a = numc_array_random_xavier(ctx, (size_t[]){4}, 1,
                                          NUMC_DTYPE_FLOAT32, 0, 64);
  ASSERT_MSG(a == NULL, "xavier with fan_in=0 must return NULL");

  NumcArray *b = numc_array_random_xavier(ctx, (size_t[]){4}, 1,
                                          NUMC_DTYPE_FLOAT32, 64, 0);
  ASSERT_MSG(b == NULL, "xavier with fan_out=0 must return NULL");

  numc_ctx_free(ctx);
  return 0;
}

/* ── He vs Xavier are different ─────────────────────────────────────*/

static int test_he_xavier_differ(void) {
  NumcCtx *ctx = numc_ctx_create();

  numc_manual_seed(9999);
  NumcArray *he =
      numc_array_random_he(ctx, (size_t[]){64}, 1, NUMC_DTYPE_FLOAT32, 128);

  numc_manual_seed(9999);
  NumcArray *xav = numc_array_random_xavier(ctx, (size_t[]){64}, 1,
                                            NUMC_DTYPE_FLOAT32, 128, 64);

  ASSERT_MSG(he != NULL && xav != NULL,
             "he and xavier arrays must not be NULL");

  float *ph = (float *)numc_array_data(he);
  float *px = (float *)numc_array_data(xav);
  int differs = 0;
  for (size_t i = 0; i < 64; i++) {
    if (ph[i] != px[i])
      differs = 1;
  }
  ASSERT_MSG(differs, "he and xavier should produce different values");

  numc_ctx_free(ctx);
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== random/test_random_init ===\n\n");

  printf("numc_array_random_he:\n");
  RUN_TEST(test_he_float32_shape);
  RUN_TEST(test_he_float32_stats);
  RUN_TEST(test_he_float64_stats);
  RUN_TEST(test_he_error_zero_fan_in);

  printf("\nnumc_array_random_xavier:\n");
  RUN_TEST(test_xavier_float32_shape);
  RUN_TEST(test_xavier_float32_range);
  RUN_TEST(test_xavier_float32_stats);
  RUN_TEST(test_xavier_float64_range);
  RUN_TEST(test_xavier_error_zero_fan);

  printf("\nhe vs xavier:\n");
  RUN_TEST(test_he_xavier_differ);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
