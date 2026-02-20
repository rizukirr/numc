#include "../helpers.h"

static int test_array_reshape_basic(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 6};
  NumcArray *arr = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_INT32);

  size_t new_shape[] = {3, 4};
  int ret = numc_array_reshape(arr, new_shape, 2);
  ASSERT_MSG(ret == 0, "reshape should succeed");
  ASSERT_MSG(numc_array_size(arr) == 12, "size should remain 12");
  ASSERT_MSG(numc_array_ndim(arr) == 2, "ndim should still be 2");

  size_t out_shape[2];
  numc_array_shape(arr, out_shape);
  ASSERT_MSG(out_shape[0] == 3 && out_shape[1] == 4, "shape should be {3, 4}");

  numc_ctx_free(ctx);
  return 0;
}

static int test_array_reshape_to_1d(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3, 4};
  NumcArray *arr = numc_array_zeros(ctx, shape, 3, NUMC_DTYPE_FLOAT32);

  size_t new_shape[] = {24};
  int ret = numc_array_reshape(arr, new_shape, 1);
  ASSERT_MSG(ret == 0, "reshape to 1D should succeed");
  ASSERT_MSG(numc_array_ndim(arr) == 1, "ndim should be 1");
  ASSERT_MSG(numc_array_size(arr) == 24, "size should be 24");

  numc_ctx_free(ctx);
  return 0;
}

static int test_array_reshape_bad_size(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3, 4};
  NumcArray *arr = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_INT32);

  size_t new_shape[] = {5, 5}; // 25 != 12
  int ret = numc_array_reshape(arr, new_shape, 2);
  ASSERT_MSG(ret == -1, "reshape with wrong size should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_array_reshape_null(void) {
  ASSERT_MSG(numc_array_reshape(NULL, (size_t[]){4}, 1) == -1,
             "reshape NULL should return -1");
  return 0;
}

static int test_array_reshape_copy_basic(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 6};
  NumcArray *arr = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_INT32);
  int32_t data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  numc_array_write(arr, data);

  size_t new_shape[] = {3, 4};
  NumcArray *reshaped = numc_array_reshape_copy(arr, new_shape, 2);
  ASSERT_MSG(reshaped != NULL, "reshape_copy should not return NULL");
  ASSERT_MSG(numc_array_size(reshaped) == 12, "copy size should be 12");

  // Original should be unchanged
  size_t orig_shape[2];
  numc_array_shape(arr, orig_shape);
  ASSERT_MSG(orig_shape[0] == 2 && orig_shape[1] == 6,
             "original shape should be unchanged");

  // Data should be preserved
  int32_t *rdata = (int32_t *)numc_array_data(reshaped);
  ASSERT_MSG(rdata[0] == 1 && rdata[11] == 12,
             "data should be preserved in copy");

  numc_ctx_free(ctx);
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== core/test_reshape ===\n\n");

  printf("Reshape:\n");
  RUN_TEST(test_array_reshape_basic);
  RUN_TEST(test_array_reshape_to_1d);
  RUN_TEST(test_array_reshape_bad_size);
  RUN_TEST(test_array_reshape_null);
  RUN_TEST(test_array_reshape_copy_basic);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
