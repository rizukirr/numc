#include "../helpers.h"

static int test_array_shape_strides(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3, 4};
  NumcArray *arr = numc_array_create(ctx, shape, 3, NUMC_DTYPE_INT32);

  size_t out_shape[3];
  numc_array_shape(arr, out_shape);
  ASSERT_MSG(out_shape[0] == 2 && out_shape[1] == 3 && out_shape[2] == 4,
             "shape should be {2, 3, 4}");

  size_t out_strides[3];
  numc_array_strides(arr, out_strides);
  // Row-major: strides in bytes: {3*4*4, 4*4, 4} = {48, 16, 4}
  ASSERT_MSG(out_strides[0] == 48, "stride[0] should be 48");
  ASSERT_MSG(out_strides[1] == 16, "stride[1] should be 16");
  ASSERT_MSG(out_strides[2] == 4, "stride[2] should be 4");

  numc_ctx_free(ctx);
  return 0;
}

static int test_array_data_not_null(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {5};
  NumcArray *arr = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  ASSERT_MSG(numc_array_data(arr) != NULL, "data pointer should not be NULL");
  numc_ctx_free(ctx);
  return 0;
}

static int test_array_capacity(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {10};
  NumcArray *arr = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT64);
  ASSERT_MSG(numc_array_capacity(arr) == 10 * sizeof(double),
             "capacity should be size * elem_size");
  numc_ctx_free(ctx);
  return 0;
}

static int test_multiple_arrays_same_ctx(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {100};

  NumcArray *arrays[10];
  for (int i = 0; i < 10; i++) {
    arrays[i] = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
    ASSERT_MSG(arrays[i] != NULL,
               "should create multiple arrays from same ctx");
  }

  // All freed together
  numc_ctx_free(ctx);
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== core/test_properties ===\n\n");

  printf("Shape & strides:\n");
  RUN_TEST(test_array_shape_strides);
  RUN_TEST(test_array_data_not_null);

  printf("\nCapacity:\n");
  RUN_TEST(test_array_capacity);

  printf("\nMultiple arrays:\n");
  RUN_TEST(test_multiple_arrays_same_ctx);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
