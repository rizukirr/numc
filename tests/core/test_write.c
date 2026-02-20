#include "../helpers.h"

static int test_array_write(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *arr = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);

  int32_t src[] = {10, 20, 30, 40};
  numc_array_write(arr, src);

  int32_t *data = (int32_t *)numc_array_data(arr);
  for (size_t i = 0; i < 4; i++) {
    ASSERT_MSG(data[i] == src[i], "written data should match");
  }

  numc_ctx_free(ctx);
  return 0;
}

static int test_array_write_null(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *arr = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  numc_array_write(arr, NULL);  // should not crash
  numc_array_write(NULL, NULL); // should not crash
  numc_ctx_free(ctx);
  return 0;
}

static int test_array_write_2d(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *arr = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);

  float src[2][3] = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
  numc_array_write(arr, src);

  float *data = (float *)numc_array_data(arr);
  ASSERT_MSG(data[0] == 1.0f && data[5] == 6.0f,
             "2D write should be row-major");

  numc_ctx_free(ctx);
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== core/test_write ===\n\n");

  printf("Write:\n");
  RUN_TEST(test_array_write);
  RUN_TEST(test_array_write_null);
  RUN_TEST(test_array_write_2d);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
