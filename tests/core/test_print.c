#include "../helpers.h"

static int test_array_print_smoke(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *arr = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  int32_t data[] = {1, 2, 3, 4, 5, 6};
  numc_array_write(arr, data);

  // Just verify it doesn't crash — output goes to stdout
  numc_array_print(arr);

  numc_ctx_free(ctx);
  return 0;
}

static int test_array_print_float(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *arr = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  float data[] = {1.5f, 2.5f, 3.5f};
  numc_array_write(arr, data);

  numc_array_print(arr);

  numc_ctx_free(ctx);
  return 0;
}

static int test_array_print_1d(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {5};
  NumcArray *arr = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT8);
  numc_array_print(arr);
  numc_ctx_free(ctx);
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== core/test_print ===\n\n");

  printf("Print (smoke tests):\n");
  RUN_TEST(test_array_print_smoke);
  RUN_TEST(test_array_print_float);
  RUN_TEST(test_array_print_1d);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
