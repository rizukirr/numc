#include "../helpers.h"

static int test_array_fill(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {6};
  int32_t val = 42;
  NumcArray *arr = numc_array_fill(ctx, shape, 1, NUMC_DTYPE_INT32, &val);
  ASSERT_MSG(arr != NULL, "fill should not return NULL");

  int32_t *data = (int32_t *)numc_array_data(arr);
  for (size_t i = 0; i < 6; i++) {
    ASSERT_MSG(data[i] == 42, "all elements should be 42");
  }

  numc_ctx_free(ctx);
  return 0;
}

static int test_array_fill_float(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3, 3};
  float val = 3.14f;
  NumcArray *arr = numc_array_fill(ctx, shape, 2, NUMC_DTYPE_FLOAT32, &val);

  float *data = (float *)numc_array_data(arr);
  for (size_t i = 0; i < 9; i++) {
    ASSERT_MSG(data[i] == 3.14f, "all elements should be 3.14");
  }

  numc_ctx_free(ctx);
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== core/test_fill ===\n\n");

  printf("Fill:\n");
  RUN_TEST(test_array_fill);
  RUN_TEST(test_array_fill_float);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
