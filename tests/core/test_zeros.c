#include "../helpers.h"

static int test_array_zeros(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *arr = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);
  ASSERT_MSG(arr != NULL, "zeros should not return NULL");

  int32_t *data = (int32_t *)numc_array_data(arr);
  for (size_t i = 0; i < 4; i++) {
    ASSERT_MSG(data[i] == 0, "all elements should be zero");
  }

  numc_ctx_free(ctx);
  return 0;
}

static int test_array_zeros_float(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {8};
  NumcArray *arr = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  float *data = (float *)numc_array_data(arr);
  for (size_t i = 0; i < 8; i++) {
    ASSERT_MSG(data[i] == 0.0f, "all float elements should be zero");
  }
  numc_ctx_free(ctx);
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== core/test_zeros ===\n\n");

  printf("Zeros:\n");
  RUN_TEST(test_array_zeros);
  RUN_TEST(test_array_zeros_float);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
