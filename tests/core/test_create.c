#include "../helpers.h"

static int test_array_create_basic(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3, 4};
  NumcArray *arr = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  ASSERT_MSG(arr != NULL, "array should not be NULL");
  ASSERT_MSG(numc_array_size(arr) == 12, "size should be 12");
  ASSERT_MSG(numc_array_ndim(arr) == 2, "ndim should be 2");
  ASSERT_MSG(numc_array_elem_size(arr) == sizeof(float),
             "elem_size should be 4");
  ASSERT_MSG(numc_array_dtype(arr) == NUMC_DTYPE_FLOAT32,
             "dtype should be FLOAT32");
  numc_ctx_free(ctx);
  return 0;
}

static int test_array_create_1d(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {10};
  NumcArray *arr = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  ASSERT_MSG(arr != NULL, "array should not be NULL");
  ASSERT_MSG(numc_array_size(arr) == 10, "size should be 10");
  ASSERT_MSG(numc_array_ndim(arr) == 1, "ndim should be 1");
  numc_ctx_free(ctx);
  return 0;
}

static int test_array_create_high_dim(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 2, 2, 2, 2, 2, 2, 2};
  NumcArray *arr = numc_array_create(ctx, shape, 8, NUMC_DTYPE_FLOAT64);
  ASSERT_MSG(arr != NULL, "8-dim array should not be NULL");
  ASSERT_MSG(numc_array_size(arr) == 256, "size should be 256");
  ASSERT_MSG(numc_array_ndim(arr) == 8, "ndim should be 8");
  numc_ctx_free(ctx);
  return 0;
}

static int test_array_create_null_ctx(void) {
  size_t shape[] = {3};
  NumcArray *arr = numc_array_create(NULL, shape, 1, NUMC_DTYPE_INT32);
  ASSERT_MSG(arr == NULL, "should return NULL with NULL ctx");
  return 0;
}

static int test_array_create_null_shape(void) {
  NumcCtx *ctx = numc_ctx_create();
  NumcArray *arr = numc_array_create(ctx, NULL, 1, NUMC_DTYPE_INT32);
  ASSERT_MSG(arr == NULL, "should return NULL with NULL shape");
  numc_ctx_free(ctx);
  return 0;
}

static int test_array_create_zero_dim(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *arr = numc_array_create(ctx, shape, 0, NUMC_DTYPE_INT32);
  ASSERT_MSG(arr == NULL, "should return NULL with zero dim");
  numc_ctx_free(ctx);
  return 0;
}

static int test_array_create_all_dtypes(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcDType dtypes[] = {
      NUMC_DTYPE_INT8,    NUMC_DTYPE_INT16,  NUMC_DTYPE_INT32,
      NUMC_DTYPE_INT64,   NUMC_DTYPE_UINT8,  NUMC_DTYPE_UINT16,
      NUMC_DTYPE_UINT32,  NUMC_DTYPE_UINT64, NUMC_DTYPE_FLOAT32,
      NUMC_DTYPE_FLOAT64,
  };
  for (int i = 0; i < 10; i++) {
    NumcArray *arr = numc_array_create(ctx, shape, 1, dtypes[i]);
    ASSERT_MSG(arr != NULL, "should create array for all dtypes");
    ASSERT_MSG(numc_array_dtype(arr) == dtypes[i], "dtype should match");
  }
  numc_ctx_free(ctx);
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== core/test_create ===\n\n");

  printf("Array creation:\n");
  RUN_TEST(test_array_create_basic);
  RUN_TEST(test_array_create_1d);
  RUN_TEST(test_array_create_high_dim);
  RUN_TEST(test_array_create_null_ctx);
  RUN_TEST(test_array_create_null_shape);
  RUN_TEST(test_array_create_zero_dim);
  RUN_TEST(test_array_create_all_dtypes);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
