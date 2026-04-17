#include "../helpers.h"

#include <math.h>

/* -- He init: mean ~ 0, std ~ sqrt(2/fan_in) -----------------------*/

static int test_he_float32_stats(void) {
  NumcCtx *ctx = numc_ctx_create();
  numc_manual_seed(42);

  const size_t n = 4096;
  const size_t fan_in = 512;
  NumcArray *a =
      numc_array_random_he(ctx, (size_t[]){n}, 1, NUMC_DTYPE_FLOAT32, fan_in);
  ASSERT_MSG(a != NULL, "random_he float32 must not be NULL");

  float *p = (float *)numc_array_data(a);
  double sum = 0.0, sum2 = 0.0;
  for (size_t i = 0; i < n; i++) {
    sum += p[i];
    sum2 += (double)p[i] * p[i];
  }
  double mean = sum / (double)n;
  double var = sum2 / (double)n - mean * mean;
  double std = sqrt(var);
  double expected_std = sqrt(2.0 / (double)fan_in);

  ASSERT_MSG(fabs(mean) < 0.1, "random_he float32 mean should be close to 0");
  /* allow ±25% tolerance on std */
  ASSERT_MSG(std > expected_std * 0.75 && std < expected_std * 1.25,
             "random_he float32 std should be close to sqrt(2/fan_in)");

  numc_ctx_free(ctx);
  return 0;
}

static int test_he_float64_stats(void) {
  NumcCtx *ctx = numc_ctx_create();
  numc_manual_seed(1234);

  const size_t n = 4096;
  const size_t fan_in = 256;
  NumcArray *a =
      numc_array_random_he(ctx, (size_t[]){n}, 1, NUMC_DTYPE_FLOAT64, fan_in);
  ASSERT_MSG(a != NULL, "random_he float64 must not be NULL");

  double *p = (double *)numc_array_data(a);
  double sum = 0.0, sum2 = 0.0;
  for (size_t i = 0; i < n; i++) {
    sum += p[i];
    sum2 += p[i] * p[i];
  }
  double mean = sum / (double)n;
  double var = sum2 / (double)n - mean * mean;
  double std = sqrt(var);
  double expected_std = sqrt(2.0 / (double)fan_in);

  ASSERT_MSG(fabs(mean) < 0.1, "random_he float64 mean should be close to 0");
  ASSERT_MSG(std > expected_std * 0.75 && std < expected_std * 1.25,
             "random_he float64 std should be close to sqrt(2/fan_in)");

  numc_ctx_free(ctx);
  return 0;
}

/* -- 2D shape --------------------------------------------------------*/

static int test_he_2d(void) {
  NumcCtx *ctx = numc_ctx_create();
  numc_manual_seed(7);

  NumcArray *a =
      numc_array_random_he(ctx, (size_t[]){32, 64}, 2, NUMC_DTYPE_FLOAT32, 64);
  ASSERT_MSG(a != NULL, "random_he 2D must not be NULL");
  ASSERT_MSG(numc_array_size(a) == 2048, "random_he 2D size should be 2048");
  ASSERT_MSG(numc_array_ndim(a) == 2, "random_he 2D ndim should be 2");

  numc_ctx_free(ctx);
  return 0;
}

/* -- error cases ----------------------------------------------------*/

static int test_he_null_ctx(void) {
  NumcArray *a =
      numc_array_random_he(NULL, (size_t[]){4}, 1, NUMC_DTYPE_FLOAT32, 4);
  ASSERT_MSG(a == NULL, "random_he with NULL ctx must return NULL");
  return 0;
}

static int test_he_zero_fan_in(void) {
  NumcCtx *ctx = numc_ctx_create();
  NumcArray *a =
      numc_array_random_he(ctx, (size_t[]){4}, 1, NUMC_DTYPE_FLOAT32, 0);
  ASSERT_MSG(a == NULL, "random_he with fan_in=0 must return NULL");
  numc_ctx_free(ctx);
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== random/test_random_he ===\n\n");

  printf("numc_array_random_he correctness:\n");
  RUN_TEST(test_he_float32_stats);
  RUN_TEST(test_he_float64_stats);

  printf("\nnumc_array_random_he multi-dimensional:\n");
  RUN_TEST(test_he_2d);

  printf("\nnumc_array_random_he error cases:\n");
  RUN_TEST(test_he_null_ctx);
  RUN_TEST(test_he_zero_fan_in);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
