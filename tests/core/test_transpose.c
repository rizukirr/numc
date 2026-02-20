#include "../helpers.h"

static int test_array_transpose_2d(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *arr = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  int32_t data[] = {1, 2, 3, 4, 5, 6};
  numc_array_write(arr, data);

  size_t axes[] = {1, 0};
  int ret = numc_array_transpose(arr, axes);
  ASSERT_MSG(ret == 0, "transpose should succeed");

  size_t out_shape[2];
  numc_array_shape(arr, out_shape);
  ASSERT_MSG(out_shape[0] == 3 && out_shape[1] == 2,
             "transposed shape should be {3, 2}");

  numc_ctx_free(ctx);
  return 0;
}

static int test_array_transpose_3d(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3, 4};
  NumcArray *arr = numc_array_zeros(ctx, shape, 3, NUMC_DTYPE_INT32);

  size_t axes[] = {2, 0, 1}; // (2,3,4) -> (4,2,3)
  int ret = numc_array_transpose(arr, axes);
  ASSERT_MSG(ret == 0, "3D transpose should succeed");

  size_t out_shape[3];
  numc_array_shape(arr, out_shape);
  ASSERT_MSG(out_shape[0] == 4 && out_shape[1] == 2 && out_shape[2] == 3,
             "transposed shape should be {4, 2, 3}");

  numc_ctx_free(ctx);
  return 0;
}

static int test_array_transpose_invalid_axis(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *arr = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_INT32);

  size_t axes[] = {0, 5}; // 5 is out of bounds
  int ret = numc_array_transpose(arr, axes);
  ASSERT_MSG(ret == -1, "transpose with invalid axis should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_array_transpose_duplicate_axis(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *arr = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_INT32);

  size_t axes[] = {0, 0}; // duplicate
  int ret = numc_array_transpose(arr, axes);
  ASSERT_MSG(ret == -1, "transpose with duplicate axis should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_array_transpose_copy_basic(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *arr = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  int32_t data[] = {1, 2, 3, 4, 5, 6};
  numc_array_write(arr, data);

  size_t axes[] = {1, 0};
  NumcArray *copy = numc_array_transpose_copy(arr, axes);
  ASSERT_MSG(copy != NULL, "transpose_copy should not return NULL");

  // Original should be unchanged
  size_t orig_shape[2];
  numc_array_shape(arr, orig_shape);
  ASSERT_MSG(orig_shape[0] == 2 && orig_shape[1] == 3,
             "original shape should be unchanged");

  // Copy should be transposed
  size_t copy_shape[2];
  numc_array_shape(copy, copy_shape);
  ASSERT_MSG(copy_shape[0] == 3 && copy_shape[1] == 2,
             "copy shape should be transposed");

  numc_ctx_free(ctx);
  return 0;
}

static int test_array_transpose_copy_null(void) {
  NumcArray *copy = numc_array_transpose_copy(NULL, (size_t[]){1, 0});
  ASSERT_MSG(copy == NULL, "transpose_copy of NULL should return NULL");
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== core/test_transpose ===\n\n");

  printf("Transpose:\n");
  RUN_TEST(test_array_transpose_2d);
  RUN_TEST(test_array_transpose_3d);
  RUN_TEST(test_array_transpose_invalid_axis);
  RUN_TEST(test_array_transpose_duplicate_axis);
  RUN_TEST(test_array_transpose_copy_basic);
  RUN_TEST(test_array_transpose_copy_null);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
