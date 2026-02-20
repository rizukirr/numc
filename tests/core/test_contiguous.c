#include "../helpers.h"

static int test_array_is_contiguous(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3, 4};
  NumcArray *arr = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  ASSERT_MSG(numc_array_is_contiguous(arr) == true,
             "newly created array should be contiguous");
  numc_ctx_free(ctx);
  return 0;
}

static int test_array_contiguous_after_transpose(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *arr = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  int32_t data[] = {1, 2, 3, 4, 5, 6};
  numc_array_write(arr, data);

  size_t axes[] = {1, 0};
  numc_array_transpose(arr, axes);
  ASSERT_MSG(numc_array_is_contiguous(arr) == false,
             "transposed array should not be contiguous");

  int ret = numc_array_contiguous(arr);
  ASSERT_MSG(ret == 0, "contiguous should succeed");
  ASSERT_MSG(numc_array_is_contiguous(arr) == true,
             "array should be contiguous after conversion");

  // After transpose and contiguous: shape is {3, 2}, data should be transposed
  int32_t *cdata = (int32_t *)numc_array_data(arr);
  // Original row-major: [[1,2,3],[4,5,6]]
  // Transposed: [[1,4],[2,5],[3,6]]
  ASSERT_MSG(cdata[0] == 1 && cdata[1] == 4, "transposed row 0: {1, 4}");
  ASSERT_MSG(cdata[2] == 2 && cdata[3] == 5, "transposed row 1: {2, 5}");
  ASSERT_MSG(cdata[4] == 3 && cdata[5] == 6, "transposed row 2: {3, 6}");

  numc_ctx_free(ctx);
  return 0;
}

static int test_array_contiguous_null(void) {
  ASSERT_MSG(numc_array_contiguous(NULL) == -1,
             "contiguous on NULL should return -1");
  return 0;
}

static int test_array_contiguous_already(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *arr = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);
  int ret = numc_array_contiguous(arr);
  ASSERT_MSG(ret == 0, "contiguous on already-contiguous should return 0");
  numc_ctx_free(ctx);
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== core/test_contiguous ===\n\n");

  printf("Contiguous:\n");
  RUN_TEST(test_array_is_contiguous);
  RUN_TEST(test_array_contiguous_after_transpose);
  RUN_TEST(test_array_contiguous_null);
  RUN_TEST(test_array_contiguous_already);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
