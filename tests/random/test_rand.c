#include "../helpers.h"

#include <math.h>
#include <stdint.h>

/* ── numc_manual_seed reproducibility ──────────────────────────────*/

static int test_seed_reproducible(void) {
  NumcCtx *ctx = numc_ctx_create();

  numc_manual_seed(42);
  NumcArray *a = numc_array_rand(ctx, (size_t[]){8}, 1, NUMC_DTYPE_FLOAT32);

  numc_manual_seed(42);
  NumcArray *b = numc_array_rand(ctx, (size_t[]){8}, 1, NUMC_DTYPE_FLOAT32);

  ASSERT_MSG(a != NULL && b != NULL, "rand arrays must not be NULL");

  float *pa = (float *)numc_array_data(a);
  float *pb = (float *)numc_array_data(b);
  for (size_t i = 0; i < 8; i++) {
    ASSERT_MSG(pa[i] == pb[i], "same seed must produce same sequence");
  }

  numc_ctx_free(ctx);
  return 0;
}

static int test_seed_different(void) {
  NumcCtx *ctx = numc_ctx_create();

  numc_manual_seed(1);
  NumcArray *a = numc_array_rand(ctx, (size_t[]){8}, 1, NUMC_DTYPE_FLOAT64);

  numc_manual_seed(2);
  NumcArray *b = numc_array_rand(ctx, (size_t[]){8}, 1, NUMC_DTYPE_FLOAT64);

  ASSERT_MSG(a != NULL && b != NULL, "rand arrays must not be NULL");

  double *pa = (double *)numc_array_data(a);
  double *pb = (double *)numc_array_data(b);
  int differs = 0;
  for (size_t i = 0; i < 8; i++) {
    if (pa[i] != pb[i])
      differs = 1;
  }
  ASSERT_MSG(differs, "different seeds must produce different sequences");

  numc_ctx_free(ctx);
  return 0;
}

/* ── float32 uniform [0, 1) ─────────────────────────────────────────*/

static int test_rand_float32_range(void) {
  NumcCtx *ctx = numc_ctx_create();
  numc_manual_seed(123);

  NumcArray *a = numc_array_rand(ctx, (size_t[]){1024}, 1, NUMC_DTYPE_FLOAT32);
  ASSERT_MSG(a != NULL, "rand float32 must not be NULL");

  float *p = (float *)numc_array_data(a);
  for (size_t i = 0; i < 1024; i++) {
    ASSERT_MSG(p[i] >= 0.0f && p[i] < 1.0f, "rand float32 must be in [0, 1)");
  }

  numc_ctx_free(ctx);
  return 0;
}

static int test_rand_float32_not_all_zero(void) {
  NumcCtx *ctx = numc_ctx_create();
  numc_manual_seed(7);

  NumcArray *a = numc_array_rand(ctx, (size_t[]){64}, 1, NUMC_DTYPE_FLOAT32);
  ASSERT_MSG(a != NULL, "rand float32 must not be NULL");

  float *p = (float *)numc_array_data(a);
  float sum = 0.0f;
  for (size_t i = 0; i < 64; i++)
    sum += p[i];
  ASSERT_MSG(sum > 0.0f, "rand float32 array should not be all zeros");

  numc_ctx_free(ctx);
  return 0;
}

/* ── float64 uniform [0, 1) ─────────────────────────────────────────*/

static int test_rand_float64_range(void) {
  NumcCtx *ctx = numc_ctx_create();
  numc_manual_seed(456);

  NumcArray *a = numc_array_rand(ctx, (size_t[]){1024}, 1, NUMC_DTYPE_FLOAT64);
  ASSERT_MSG(a != NULL, "rand float64 must not be NULL");

  double *p = (double *)numc_array_data(a);
  for (size_t i = 0; i < 1024; i++) {
    ASSERT_MSG(p[i] >= 0.0 && p[i] < 1.0, "rand float64 must be in [0, 1)");
  }

  numc_ctx_free(ctx);
  return 0;
}

/* ── integer types — basic sanity ──────────────────────────────────*/

static int test_rand_int32_not_uniform_zero(void) {
  NumcCtx *ctx = numc_ctx_create();
  numc_manual_seed(99);

  NumcArray *a = numc_array_rand(ctx, (size_t[]){64}, 1, NUMC_DTYPE_INT32);
  ASSERT_MSG(a != NULL, "rand int32 must not be NULL");

  int32_t *p = (int32_t *)numc_array_data(a);
  int nonzero = 0;
  for (size_t i = 0; i < 64; i++) {
    if (p[i] != 0)
      nonzero++;
  }
  ASSERT_MSG(nonzero > 0, "rand int32 should produce non-zero values");

  numc_ctx_free(ctx);
  return 0;
}

static int test_rand_uint8_range(void) {
  NumcCtx *ctx = numc_ctx_create();
  numc_manual_seed(55);

  NumcArray *a = numc_array_rand(ctx, (size_t[]){256}, 1, NUMC_DTYPE_UINT8);
  ASSERT_MSG(a != NULL, "rand uint8 must not be NULL");

  uint8_t *p = (uint8_t *)numc_array_data(a);
  /* uint8 is always in [0, 255] by type — just check array was created */
  (void)p;

  numc_ctx_free(ctx);
  return 0;
}

/* ── multi-dimensional ──────────────────────────────────────────────*/

static int test_rand_2d(void) {
  NumcCtx *ctx = numc_ctx_create();
  numc_manual_seed(11);

  NumcArray *a = numc_array_rand(ctx, (size_t[]){4, 8}, 2, NUMC_DTYPE_FLOAT32);
  ASSERT_MSG(a != NULL, "rand 2D must not be NULL");
  ASSERT_MSG(numc_array_ndim(a) == 2, "rand 2D ndim should be 2");
  ASSERT_MSG(numc_array_size(a) == 32, "rand 2D size should be 32");

  float *p = (float *)numc_array_data(a);
  for (size_t i = 0; i < 32; i++) {
    ASSERT_MSG(p[i] >= 0.0f && p[i] < 1.0f,
               "rand 2D float32 must be in [0, 1)");
  }

  numc_ctx_free(ctx);
  return 0;
}

static int test_rand_3d(void) {
  NumcCtx *ctx = numc_ctx_create();
  numc_manual_seed(22);

  NumcArray *a =
      numc_array_rand(ctx, (size_t[]){2, 3, 4}, 3, NUMC_DTYPE_FLOAT64);
  ASSERT_MSG(a != NULL, "rand 3D must not be NULL");
  ASSERT_MSG(numc_array_size(a) == 24, "rand 3D size should be 24");

  double *p = (double *)numc_array_data(a);
  for (size_t i = 0; i < 24; i++) {
    ASSERT_MSG(p[i] >= 0.0 && p[i] < 1.0, "rand 3D float64 must be in [0, 1)");
  }

  numc_ctx_free(ctx);
  return 0;
}

/* ── error cases ────────────────────────────────────────────────────*/

static int test_rand_null_ctx(void) {
  NumcArray *a = numc_array_rand(NULL, (size_t[]){4}, 1, NUMC_DTYPE_FLOAT32);
  ASSERT_MSG(a == NULL, "rand with NULL ctx must return NULL");
  return 0;
}

static int test_rand_null_shape(void) {
  NumcCtx *ctx = numc_ctx_create();
  NumcArray *a = numc_array_rand(ctx, NULL, 1, NUMC_DTYPE_FLOAT32);
  ASSERT_MSG(a == NULL, "rand with NULL shape must return NULL");
  numc_ctx_free(ctx);
  return 0;
}

static int test_rand_zero_dim(void) {
  NumcCtx *ctx = numc_ctx_create();
  NumcArray *a = numc_array_rand(ctx, (size_t[]){4}, 0, NUMC_DTYPE_FLOAT32);
  ASSERT_MSG(a == NULL, "rand with dim=0 must return NULL");
  numc_ctx_free(ctx);
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== random/test_rand ===\n\n");

  printf("numc_manual_seed:\n");
  RUN_TEST(test_seed_reproducible);
  RUN_TEST(test_seed_different);

  printf("\nnumc_array_rand float:\n");
  RUN_TEST(test_rand_float32_range);
  RUN_TEST(test_rand_float32_not_all_zero);
  RUN_TEST(test_rand_float64_range);

  printf("\nnumc_array_rand integer:\n");
  RUN_TEST(test_rand_int32_not_uniform_zero);
  RUN_TEST(test_rand_uint8_range);

  printf("\nnumc_array_rand multi-dimensional:\n");
  RUN_TEST(test_rand_2d);
  RUN_TEST(test_rand_3d);

  printf("\nnumc_array_rand error cases:\n");
  RUN_TEST(test_rand_null_ctx);
  RUN_TEST(test_rand_null_shape);
  RUN_TEST(test_rand_zero_dim);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
