#include "../helpers.h"

static int test_array_copy(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *arr = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);
  int32_t src[] = {1, 2, 3, 4};
  numc_array_write(arr, src);

  NumcArray *copy = numc_array_copy(arr);
  ASSERT_MSG(copy != NULL, "copy should not be NULL");
  ASSERT_MSG(numc_array_size(copy) == 4, "copy size should match");
  ASSERT_MSG(numc_array_dtype(copy) == NUMC_DTYPE_INT32,
             "copy dtype should match");

  int32_t *copy_data = (int32_t *)numc_array_data(copy);
  int32_t *orig_data = (int32_t *)numc_array_data(arr);
  for (size_t i = 0; i < 4; i++) {
    ASSERT_MSG(copy_data[i] == orig_data[i], "copy data should match original");
  }

  // Verify it's a deep copy — modifying copy shouldn't affect original
  copy_data[0] = 999;
  ASSERT_MSG(orig_data[0] == 1, "modifying copy should not affect original");

  numc_ctx_free(ctx);
  return 0;
}

static int test_array_copy_null(void) {
  NumcArray *copy = numc_array_copy(NULL);
  ASSERT_MSG(copy == NULL, "copy of NULL should return NULL");
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== core/test_copy ===\n\n");

  printf("Copy:\n");
  RUN_TEST(test_array_copy);
  RUN_TEST(test_array_copy_null);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
