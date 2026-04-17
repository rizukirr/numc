#include "../helpers.h"

#include <math.h>

/* -- Xavier init: uniform [-limit, limit), limit=sqrt(6/(fi+fo)) ---*/

static int test_xavier_float32_range(void) {
  NumcCtx *ctx = numc_ctx_create();
  numc_manual_seed(42);

  const size_t n = 4096;
  const size_t fan_in = 256, fan_out = 128;
  double limit = sqrt(6.0 / (double)(fan_in + fan_out));

  NumcArray *a = numc_array_random_xavier(ctx, (size_t[]){n}, 1,
                                          NUMC_DTYPE_FLOAT32, fan_in, fan_out);
  ASSERT_MSG(a != NULL, "random_xavier float32 must not be NULL");

  float *p = (float *)numc_array_data(a);
  for (size_t i = 0; i < n; i++) {
    ASSERT_MSG((double)p[i] >= -limit && (double)p[i] < limit,
               "random_xavier float32 values must be in [-limit, limit)");
  }

  numc_ctx_free(ctx);
  return 0;
}

static int test_xavier_float64_range(void) {
  NumcCtx *ctx = numc_ctx_create();
  numc_manual_seed(1234);

  const size_t n = 4096;
  const size_t fan_in = 512, fan_out = 256;
  double limit = sqrt(6.0 / (double)(fan_in + fan_out));

  NumcArray *a = numc_array_random_xavier(ctx, (size_t[]){n}, 1,
                                          NUMC_DTYPE_FLOAT64, fan_in, fan_out);
  ASSERT_MSG(a != NULL, "random_xavier float64 must not be NULL");

  double *p = (double *)numc_array_data(a);
  for (size_t i = 0; i < n; i++) {
    ASSERT_MSG(p[i] >= -limit && p[i] < limit,
               "random_xavier float64 values must be in [-limit, limit)");
  }

  numc_ctx_free(ctx);
  return 0;
}

/* -- mean ~ 0 over large sample -------------------------------------*/

static int test_xavier_float32_mean(void) {
  NumcCtx *ctx = numc_ctx_create();
  numc_manual_seed(99);

  const size_t n = 4096;
  const size_t fan_in = 128, fan_out = 128;

  NumcArray *a = numc_array_random_xavier(ctx, (size_t[]){n}, 1,
                                          NUMC_DTYPE_FLOAT32, fan_in, fan_out);
  ASSERT_MSG(a != NULL, "random_xavier float32 mean-check must not be NULL");

  float *p = (float *)numc_array_data(a);
  double sum = 0.0;
  for (size_t i = 0; i < n; i++)
    sum += p[i];
  double mean = sum / (double)n;

  ASSERT_MSG(fabs(mean) < 0.1,
             "random_xavier float32 mean should be close to 0");

  numc_ctx_free(ctx);
  return 0;
}

/* -- 2D shape --------------------------------------------------------*/

static int test_xavier_2d(void) {
  NumcCtx *ctx = numc_ctx_create();
  numc_manual_seed(7);

  NumcArray *a = numc_array_random_xavier(ctx, (size_t[]){16, 32}, 2,
                                          NUMC_DTYPE_FLOAT32, 32, 16);
  ASSERT_MSG(a != NULL, "random_xavier 2D must not be NULL");
  ASSERT_MSG(numc_array_size(a) == 512, "random_xavier 2D size should be 512");
  ASSERT_MSG(numc_array_ndim(a) == 2, "random_xavier 2D ndim should be 2");

  numc_ctx_free(ctx);
  return 0;
}

/* -- error cases ----------------------------------------------------*/

static int test_xavier_null_ctx(void) {
  NumcArray *a = numc_array_random_xavier(NULL, (size_t[]){4}, 1,
                                          NUMC_DTYPE_FLOAT32, 4, 4);
  ASSERT_MSG(a == NULL, "random_xavier with NULL ctx must return NULL");
  return 0;
}

static int test_xavier_zero_fan_in(void) {
  NumcCtx *ctx = numc_ctx_create();
  NumcArray *a =
      numc_array_random_xavier(ctx, (size_t[]){4}, 1, NUMC_DTYPE_FLOAT32, 0, 4);
  ASSERT_MSG(a == NULL, "random_xavier with fan_in=0 must return NULL");
  numc_ctx_free(ctx);
  return 0;
}

static int test_xavier_zero_fan_out(void) {
  NumcCtx *ctx = numc_ctx_create();
  NumcArray *a =
      numc_array_random_xavier(ctx, (size_t[]){4}, 1, NUMC_DTYPE_FLOAT32, 4, 0);
  ASSERT_MSG(a == NULL, "random_xavier with fan_out=0 must return NULL");
  numc_ctx_free(ctx);
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== random/test_random_xavier ===\n\n");

  printf("numc_array_random_xavier correctness:\n");
  RUN_TEST(test_xavier_float32_range);
  RUN_TEST(test_xavier_float64_range);
  RUN_TEST(test_xavier_float32_mean);

  printf("\nnumc_array_random_xavier multi-dimensional:\n");
  RUN_TEST(test_xavier_2d);

  printf("\nnumc_array_random_xavier error cases:\n");
  RUN_TEST(test_xavier_null_ctx);
  RUN_TEST(test_xavier_zero_fan_in);
  RUN_TEST(test_xavier_zero_fan_out);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
