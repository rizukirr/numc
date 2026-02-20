#include "../helpers.h"

static int test_array_slice_basic(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {10};
  NumcArray *arr = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  int32_t data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  numc_array_write(arr, data);

  NumcArray *view = numc_slice(arr, .axis = 0, .start = 2, .stop = 7);
  ASSERT_MSG(view != NULL, "slice should not return NULL");
  ASSERT_MSG(numc_array_size(view) == 5, "slice size should be 5");

  // Verify it's a view — data pointer should be offset
  int32_t *vdata = (int32_t *)numc_array_data(view);
  ASSERT_MSG(vdata[0] == 2, "first element should be 2");
  ASSERT_MSG(vdata[4] == 6, "last element should be 6");

  numc_ctx_free(ctx);
  return 0;
}

static int test_array_slice_step(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {10};
  NumcArray *arr = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  int32_t data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  numc_array_write(arr, data);

  NumcArray *view =
      numc_slice(arr, .axis = 0, .start = 0, .stop = 10, .step = 2);
  ASSERT_MSG(view != NULL, "slice with step should not return NULL");
  ASSERT_MSG(numc_array_size(view) == 5, "slice size with step 2 should be 5");

  numc_ctx_free(ctx);
  return 0;
}

static int test_array_slice_full(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {5};
  NumcArray *arr = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);

  // stop=0 means full extent
  NumcArray *view = numc_slice(arr, .axis = 0, .start = 0, .stop = 0);
  ASSERT_MSG(view != NULL, "full slice should not return NULL");
  ASSERT_MSG(numc_array_size(view) == 5, "full slice size should be 5");

  numc_ctx_free(ctx);
  return 0;
}

static int test_array_slice_2d(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4, 6};
  NumcArray *arr = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_INT32);

  // Slice axis 0: rows 1..3
  NumcArray *view = numc_slice(arr, .axis = 0, .start = 1, .stop = 3);
  ASSERT_MSG(view != NULL, "2D slice should not return NULL");
  ASSERT_MSG(numc_array_size(view) == 12,
             "sliced shape should be {2, 6} = 12 elements");

  size_t out_shape[2];
  numc_array_shape(view, out_shape);
  ASSERT_MSG(out_shape[0] == 2, "sliced dim 0 should be 2");
  ASSERT_MSG(out_shape[1] == 6, "sliced dim 1 should be 6");

  numc_ctx_free(ctx);
  return 0;
}

static int test_array_slice_out_of_bounds(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {5};
  NumcArray *arr = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);

  // axis >= ndim
  NumcArray *view = numc_slice(arr, .axis = 1, .start = 0, .stop = 3);
  ASSERT_MSG(view == NULL, "slice with invalid axis should return NULL");

  numc_ctx_free(ctx);
  return 0;
}

static int test_array_slice_null(void) {
  NumcArray *view = numc_slice(NULL, .axis = 0, .start = 0, .stop = 3);
  ASSERT_MSG(view == NULL, "slice of NULL should return NULL");
  return 0;
}

static int test_array_slice_is_view(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {6};
  NumcArray *arr = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  int32_t data[] = {10, 20, 30, 40, 50, 60};
  numc_array_write(arr, data);

  NumcArray *view = numc_slice(arr, .axis = 0, .start = 1, .stop = 4);

  // Modify original, view should reflect change
  int32_t *orig = (int32_t *)numc_array_data(arr);
  orig[2] = 999;

  int32_t *vdata = (int32_t *)numc_array_data(view);
  ASSERT_MSG(vdata[1] == 999,
             "view should reflect changes to original (shared data)");

  numc_ctx_free(ctx);
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== core/test_slice ===\n\n");

  printf("Slice:\n");
  RUN_TEST(test_array_slice_basic);
  RUN_TEST(test_array_slice_step);
  RUN_TEST(test_array_slice_full);
  RUN_TEST(test_array_slice_2d);
  RUN_TEST(test_array_slice_out_of_bounds);
  RUN_TEST(test_array_slice_null);
  RUN_TEST(test_array_slice_is_view);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
