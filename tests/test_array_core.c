#include <numc/numc.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>


/* --- Helpers --- */

#define ASSERT_MSG(cond, msg)                                                  \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg);           \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define RUN_TEST(fn)                                                           \
  do {                                                                         \
    printf("  %-60s", #fn);                                                    \
    int _r = fn();                                                             \
    if (_r) {                                                                  \
      printf("FAIL\n");                                                        \
      fails++;                                                                 \
    } else {                                                                   \
      printf("OK\n");                                                          \
      passes++;                                                                \
    }                                                                          \
  } while (0)

/* --- Context tests --- */

static int test_ctx_create(void) {
  NumcCtx *ctx = numc_ctx_create();
  ASSERT_MSG(ctx != NULL, "ctx should not be NULL");
  numc_ctx_free(ctx);
  return 0;
}

static int test_ctx_free_null(void) {
  numc_ctx_free(NULL); // should not crash
  return 0;
}

/* --- Array creation tests --- */

static int test_array_create_basic(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3, 4};
  NumcArray *arr = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  ASSERT_MSG(arr != NULL, "array should not be NULL");
  ASSERT_MSG(numc_array_size(arr) == 12, "size should be 12");
  ASSERT_MSG(numc_array_ndim(arr) == 2, "ndim should be 2");
  ASSERT_MSG(numc_array_elem_size(arr) == sizeof(float), "elem_size should be 4");
  ASSERT_MSG(numc_array_dtype(arr) == NUMC_DTYPE_FLOAT32, "dtype should be FLOAT32");
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
      NUMC_DTYPE_INT8,    NUMC_DTYPE_INT16,   NUMC_DTYPE_INT32,
      NUMC_DTYPE_INT64,   NUMC_DTYPE_UINT8,   NUMC_DTYPE_UINT16,
      NUMC_DTYPE_UINT32,  NUMC_DTYPE_UINT64,  NUMC_DTYPE_FLOAT32,
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

/* --- Shape & strides tests --- */

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

/* --- Zeros tests --- */

static int test_array_zeros(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *arr = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);
  ASSERT_MSG(arr != NULL, "zeros should not return NULL");

  int32_t *data = (int32_t *)numc_array_data(arr);
  for (size_t i = 0; i < 4; i++) {
    ASSERT_MSG(data[i] == 0, "all elements should be zero");
  }

  numc_ctx_free(ctx);
  return 0;
}

static int test_array_zeros_float(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {8};
  NumcArray *arr = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  float *data = (float *)numc_array_data(arr);
  for (size_t i = 0; i < 8; i++) {
    ASSERT_MSG(data[i] == 0.0f, "all float elements should be zero");
  }
  numc_ctx_free(ctx);
  return 0;
}

/* --- Fill tests --- */

static int test_array_fill(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {6};
  int32_t val = 42;
  NumcArray *arr =
      numc_array_fill(ctx, shape, 1, NUMC_DTYPE_INT32, &val);
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
  NumcArray *arr =
      numc_array_fill(ctx, shape, 2, NUMC_DTYPE_FLOAT32, &val);

  float *data = (float *)numc_array_data(arr);
  for (size_t i = 0; i < 9; i++) {
    ASSERT_MSG(data[i] == 3.14f, "all elements should be 3.14");
  }

  numc_ctx_free(ctx);
  return 0;
}

/* --- Write tests --- */

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
  numc_array_write(arr, NULL); // should not crash
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
  ASSERT_MSG(data[0] == 1.0f && data[5] == 6.0f, "2D write should be row-major");

  numc_ctx_free(ctx);
  return 0;
}

/* --- Copy tests --- */

static int test_array_copy(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *arr = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);
  int32_t src[] = {1, 2, 3, 4};
  numc_array_write(arr, src);

  NumcArray *copy = numc_array_copy(arr);
  ASSERT_MSG(copy != NULL, "copy should not be NULL");
  ASSERT_MSG(numc_array_size(copy) == 4, "copy size should match");
  ASSERT_MSG(numc_array_dtype(copy) == NUMC_DTYPE_INT32, "copy dtype should match");

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

/* --- Capacity tests --- */

static int test_array_capacity(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {10};
  NumcArray *arr = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT64);
  ASSERT_MSG(numc_array_capacity(arr) == 10 * sizeof(double),
             "capacity should be size * elem_size");
  numc_ctx_free(ctx);
  return 0;
}

/* --- Contiguous tests --- */

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

/* --- Reshape tests --- */

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
  ASSERT_MSG(out_shape[0] == 3 && out_shape[1] == 4,
             "shape should be {3, 4}");

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

/* --- Transpose tests --- */

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

/* --- Slice tests --- */

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

  NumcArray *view = numc_slice(arr, .axis = 0, .start = 0, .stop = 10, .step = 2);
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
  ASSERT_MSG(numc_array_size(view) == 12, "sliced shape should be {2, 6} = 12 elements");

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
  ASSERT_MSG(vdata[1] == 999, "view should reflect changes to original (shared data)");

  numc_ctx_free(ctx);
  return 0;
}

/* --- Multiple arrays from same context --- */

static int test_multiple_arrays_same_ctx(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {100};

  NumcArray *arrays[10];
  for (int i = 0; i < 10; i++) {
    arrays[i] = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
    ASSERT_MSG(arrays[i] != NULL, "should create multiple arrays from same ctx");
  }

  // All freed together
  numc_ctx_free(ctx);
  return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Math: Element-wise binary ops
 * ═══════════════════════════════════════════════════════════════════════ */

static int test_add_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *b   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float db[] = {10.0f, 20.0f, 30.0f, 40.0f};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_add(a, b, out);
  ASSERT_MSG(err == 0, "add should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 11.0f && r[1] == 22.0f && r[2] == 33.0f && r[3] == 44.0f,
             "add results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sub_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *b   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {10.0f, 20.0f, 30.0f, 40.0f};
  float db[] = {1.0f, 2.0f, 3.0f, 4.0f};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_sub(a, b, out);
  ASSERT_MSG(err == 0, "sub should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 9.0f && r[1] == 18.0f && r[2] == 27.0f && r[3] == 36.0f,
             "sub results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_mul_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *b   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {2.0f, 3.0f, 4.0f, 5.0f};
  float db[] = {10.0f, 10.0f, 10.0f, 10.0f};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_mul(a, b, out);
  ASSERT_MSG(err == 0, "mul should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 20.0f && r[1] == 30.0f && r[2] == 40.0f && r[3] == 50.0f,
             "mul results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_div_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *b   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {20.0f, 30.0f, 40.0f, 50.0f};
  float db[] = {10.0f, 10.0f, 10.0f, 10.0f};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_div(a, b, out);
  ASSERT_MSG(err == 0, "div should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 2.0f && r[1] == 3.0f && r[2] == 4.0f && r[3] == 5.0f,
             "div results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_add_int32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *b   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);

  int32_t da[] = {1, 2, 3, 4};
  int32_t db[] = {100, 200, 300, 400};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_add(a, b, out);
  ASSERT_MSG(err == 0, "add int32 should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 101 && r[1] == 202 && r[2] == 303 && r[3] == 404,
             "add int32 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_add_int8(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT8);
  NumcArray *b   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT8);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT8);

  int8_t da[] = {1, 2, 3, 4};
  int8_t db[] = {10, 20, 30, 40};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_add(a, b, out);
  ASSERT_MSG(err == 0, "add int8 should succeed");

  int8_t *r = (int8_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 11 && r[1] == 22 && r[2] == 33 && r[3] == 44,
             "add int8 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_add_float64(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT64);
  NumcArray *b   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT64);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT64);

  double da[] = {1.5, 2.5, 3.5};
  double db[] = {0.5, 0.5, 0.5};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_add(a, b, out);
  ASSERT_MSG(err == 0, "add float64 should succeed");

  double *r = (double *)numc_array_data(out);
  ASSERT_MSG(r[0] == 2.0 && r[1] == 3.0 && r[2] == 4.0,
             "add float64 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_binary_op_2d(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a   = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  NumcArray *b   = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_INT32);

  int32_t da[] = {1, 2, 3, 4, 5, 6};
  int32_t db[] = {10, 10, 10, 10, 10, 10};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_add(a, b, out);
  ASSERT_MSG(err == 0, "2D add should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 11 && r[5] == 16, "2D add results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_binary_op_strided(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);

  int32_t da[] = {1, 2, 3, 4, 5, 6};
  int32_t db[] = {10, 20, 30, 40, 50, 60};
  numc_array_write(a, da);
  numc_array_write(b, db);

  // Transpose both to make them non-contiguous
  size_t axes[] = {1, 0};
  numc_array_transpose(a, axes);
  numc_array_transpose(b, axes);

  // out shape matches transposed: {3, 2}
  size_t out_shape[] = {3, 2};
  NumcArray *out = numc_array_zeros(ctx, out_shape, 2, NUMC_DTYPE_INT32);

  int err = numc_add(a, b, out);
  ASSERT_MSG(err == 0, "strided add should succeed");

  // Transposed a: [[1,4],[2,5],[3,6]], b: [[10,40],[20,50],[30,60]]
  // sum: [[11,44],[22,55],[33,66]]
  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 11 && r[1] == 44, "strided row 0");
  ASSERT_MSG(r[2] == 22 && r[3] == 55, "strided row 1");
  ASSERT_MSG(r[4] == 33 && r[5] == 66, "strided row 2");

  numc_ctx_free(ctx);
  return 0;
}

/* --- Math error cases --- */

static int test_binary_op_null(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  ASSERT_MSG(numc_add(NULL, a, out) != 0, "add with NULL a should fail");
  ASSERT_MSG(numc_add(a, NULL, out) != 0, "add with NULL b should fail");
  ASSERT_MSG(numc_add(a, a, NULL) != 0, "add with NULL out should fail");
  ASSERT_MSG(numc_sub(NULL, a, out) != 0, "sub with NULL should fail");
  ASSERT_MSG(numc_mul(NULL, a, out) != 0, "mul with NULL should fail");
  ASSERT_MSG(numc_div(NULL, a, out) != 0, "div with NULL should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_binary_op_type_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *b   = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  ASSERT_MSG(numc_add(a, b, out) != 0, "add with dtype mismatch should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_binary_op_shape_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape_a[] = {4};
  size_t shape_b[] = {5};
  NumcArray *a   = numc_array_zeros(ctx, shape_a, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *b   = numc_array_zeros(ctx, shape_b, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape_a, 1, NUMC_DTYPE_FLOAT32);

  ASSERT_MSG(numc_add(a, b, out) != 0, "add with shape mismatch should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_binary_op_dim_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape_1d[] = {6};
  size_t shape_2d[] = {2, 3};
  NumcArray *a   = numc_array_zeros(ctx, shape_1d, 1, NUMC_DTYPE_INT32);
  NumcArray *b   = numc_array_zeros(ctx, shape_2d, 2, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape_1d, 1, NUMC_DTYPE_INT32);

  ASSERT_MSG(numc_add(a, b, out) != 0, "add with dim mismatch should fail");

  numc_ctx_free(ctx);
  return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Math: Scalar ops
 * ═══════════════════════════════════════════════════════════════════════ */

static int test_add_scalar_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {1.0f, 2.0f, 3.0f, 4.0f};
  numc_array_write(a, da);

  int err = numc_add_scalar(a, 10.0, out);
  ASSERT_MSG(err == 0, "add_scalar should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 11.0f && r[1] == 12.0f && r[2] == 13.0f && r[3] == 14.0f,
             "add_scalar results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sub_scalar_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {10.0f, 20.0f, 30.0f, 40.0f};
  numc_array_write(a, da);

  int err = numc_sub_scalar(a, 5.0, out);
  ASSERT_MSG(err == 0, "sub_scalar should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 5.0f && r[1] == 15.0f && r[2] == 25.0f && r[3] == 35.0f,
             "sub_scalar results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_mul_scalar_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {1.0f, 2.0f, 3.0f, 4.0f};
  numc_array_write(a, da);

  int err = numc_mul_scalar(a, 3.0, out);
  ASSERT_MSG(err == 0, "mul_scalar should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 3.0f && r[1] == 6.0f && r[2] == 9.0f && r[3] == 12.0f,
             "mul_scalar results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_div_scalar_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {10.0f, 20.0f, 30.0f, 40.0f};
  numc_array_write(a, da);

  int err = numc_div_scalar(a, 10.0, out);
  ASSERT_MSG(err == 0, "div_scalar should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 1.0f && r[1] == 2.0f && r[2] == 3.0f && r[3] == 4.0f,
             "div_scalar results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_scalar_op_int32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);

  int32_t da[] = {10, 20, 30};
  numc_array_write(a, da);

  int err = numc_add_scalar(a, 5.0, out);
  ASSERT_MSG(err == 0, "add_scalar int32 should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == 15 && r[1] == 25 && r[2] == 35,
             "add_scalar int32 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_scalar_op_null(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  ASSERT_MSG(numc_add_scalar(NULL, 1.0, out) != 0, "add_scalar with NULL a should fail");
  ASSERT_MSG(numc_add_scalar(a, 1.0, NULL) != 0, "add_scalar with NULL out should fail");
  ASSERT_MSG(numc_sub_scalar(NULL, 1.0, out) != 0, "sub_scalar with NULL should fail");
  ASSERT_MSG(numc_mul_scalar(NULL, 1.0, out) != 0, "mul_scalar with NULL should fail");
  ASSERT_MSG(numc_div_scalar(NULL, 1.0, out) != 0, "div_scalar with NULL should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_scalar_op_type_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);

  ASSERT_MSG(numc_add_scalar(a, 1.0, out) != 0,
             "scalar op with dtype mismatch should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_scalar_op_shape_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape_a[] = {4};
  size_t shape_o[] = {5};
  NumcArray *a   = numc_array_zeros(ctx, shape_a, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape_o, 1, NUMC_DTYPE_FLOAT32);

  ASSERT_MSG(numc_add_scalar(a, 1.0, out) != 0,
             "scalar op with shape mismatch should fail");

  numc_ctx_free(ctx);
  return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Math: Scalar inplace ops
 * ═══════════════════════════════════════════════════════════════════════ */

static int test_add_scalar_inplace_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {1.0f, 2.0f, 3.0f, 4.0f};
  numc_array_write(a, da);

  int err = numc_add_scalar_inplace(a, 10.0);
  ASSERT_MSG(err == 0, "add_scalar_inplace should succeed");

  float *r = (float *)numc_array_data(a);
  ASSERT_MSG(r[0] == 11.0f && r[1] == 12.0f && r[2] == 13.0f && r[3] == 14.0f,
             "add_scalar_inplace results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_sub_scalar_inplace_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {10.0f, 20.0f, 30.0f, 40.0f};
  numc_array_write(a, da);

  int err = numc_sub_scalar_inplace(a, 5.0);
  ASSERT_MSG(err == 0, "sub_scalar_inplace should succeed");

  float *r = (float *)numc_array_data(a);
  ASSERT_MSG(r[0] == 5.0f && r[1] == 15.0f && r[2] == 25.0f && r[3] == 35.0f,
             "sub_scalar_inplace results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_mul_scalar_inplace_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {1.0f, 2.0f, 3.0f, 4.0f};
  numc_array_write(a, da);

  int err = numc_mul_scalar_inplace(a, 3.0);
  ASSERT_MSG(err == 0, "mul_scalar_inplace should succeed");

  float *r = (float *)numc_array_data(a);
  ASSERT_MSG(r[0] == 3.0f && r[1] == 6.0f && r[2] == 9.0f && r[3] == 12.0f,
             "mul_scalar_inplace results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_div_scalar_inplace_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {10.0f, 20.0f, 30.0f, 40.0f};
  numc_array_write(a, da);

  int err = numc_div_scalar_inplace(a, 10.0);
  ASSERT_MSG(err == 0, "div_scalar_inplace should succeed");

  float *r = (float *)numc_array_data(a);
  ASSERT_MSG(r[0] == 1.0f && r[1] == 2.0f && r[2] == 3.0f && r[3] == 4.0f,
             "div_scalar_inplace results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_scalar_inplace_int32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);

  int32_t da[] = {10, 20, 30};
  numc_array_write(a, da);

  int err = numc_mul_scalar_inplace(a, 2.0);
  ASSERT_MSG(err == 0, "mul_scalar_inplace int32 should succeed");

  int32_t *r = (int32_t *)numc_array_data(a);
  ASSERT_MSG(r[0] == 20 && r[1] == 40 && r[2] == 60,
             "mul_scalar_inplace int32 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_scalar_inplace_null(void) {
  ASSERT_MSG(numc_add_scalar_inplace(NULL, 1.0) != 0,
             "add_scalar_inplace with NULL should fail");
  ASSERT_MSG(numc_sub_scalar_inplace(NULL, 1.0) != 0,
             "sub_scalar_inplace with NULL should fail");
  ASSERT_MSG(numc_mul_scalar_inplace(NULL, 1.0) != 0,
             "mul_scalar_inplace with NULL should fail");
  ASSERT_MSG(numc_div_scalar_inplace(NULL, 1.0) != 0,
             "div_scalar_inplace with NULL should fail");
  return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Math: Unary ops (neg)
 * ═══════════════════════════════════════════════════════════════════════ */

static int test_neg_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {1.0f, -2.0f, 3.0f, -4.0f};
  numc_array_write(a, da);

  int err = numc_neg(a, out);
  ASSERT_MSG(err == 0, "neg should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == -1.0f && r[1] == 2.0f && r[2] == -3.0f && r[3] == 4.0f,
             "neg results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_neg_int32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);

  int32_t da[] = {10, -20, 30, -40};
  numc_array_write(a, da);

  int err = numc_neg(a, out);
  ASSERT_MSG(err == 0, "neg int32 should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == -10 && r[1] == 20 && r[2] == -30 && r[3] == 40,
             "neg int32 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_neg_int8(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT8);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT8);

  int8_t da[] = {5, -10, 15, -20};
  numc_array_write(a, da);

  int err = numc_neg(a, out);
  ASSERT_MSG(err == 0, "neg int8 should succeed");

  int8_t *r = (int8_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == -5 && r[1] == 10 && r[2] == -15 && r[3] == 20,
             "neg int8 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_neg_float64(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a   = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT64);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT64);

  double da[] = {1.5, -2.5, 3.5};
  numc_array_write(a, da);

  int err = numc_neg(a, out);
  ASSERT_MSG(err == 0, "neg float64 should succeed");

  double *r = (double *)numc_array_data(out);
  ASSERT_MSG(r[0] == -1.5 && r[1] == 2.5 && r[2] == -3.5,
             "neg float64 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_neg_2d(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a   = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_INT32);

  int32_t da[] = {1, -2, 3, -4, 5, -6};
  numc_array_write(a, da);

  int err = numc_neg(a, out);
  ASSERT_MSG(err == 0, "2D neg should succeed");

  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == -1 && r[1] == 2 && r[2] == -3, "neg 2D row 0");
  ASSERT_MSG(r[3] == 4 && r[4] == -5 && r[5] == 6, "neg 2D row 1");

  numc_ctx_free(ctx);
  return 0;
}

static int test_neg_strided(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);

  int32_t da[] = {1, -2, 3, -4, 5, -6};
  numc_array_write(a, da);

  // Transpose to make it non-contiguous
  size_t axes[] = {1, 0};
  numc_array_transpose(a, axes);

  // out shape matches transposed: {3, 2}
  size_t out_shape[] = {3, 2};
  NumcArray *out = numc_array_zeros(ctx, out_shape, 2, NUMC_DTYPE_INT32);

  int err = numc_neg(a, out);
  ASSERT_MSG(err == 0, "strided neg should succeed");

  // Transposed a: [[1,-4],[-2,5],[3,-6]]
  // negated: [[-1,4],[2,-5],[-3,6]]
  int32_t *r = (int32_t *)numc_array_data(out);
  ASSERT_MSG(r[0] == -1 && r[1] == 4, "strided row 0");
  ASSERT_MSG(r[2] == 2 && r[3] == -5, "strided row 1");
  ASSERT_MSG(r[4] == -3 && r[5] == 6, "strided row 2");

  numc_ctx_free(ctx);
  return 0;
}

static int test_neg_zeros(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  int err = numc_neg(a, out);
  ASSERT_MSG(err == 0, "neg of zeros should succeed");

  float *r = (float *)numc_array_data(out);
  for (size_t i = 0; i < 4; i++) {
    ASSERT_MSG(r[i] == 0.0f, "neg of zero should be zero");
  }

  numc_ctx_free(ctx);
  return 0;
}

static int test_neg_null(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  ASSERT_MSG(numc_neg(NULL, out) != 0, "neg with NULL a should fail");
  ASSERT_MSG(numc_neg(a, NULL) != 0, "neg with NULL out should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_neg_type_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a   = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_INT32);

  ASSERT_MSG(numc_neg(a, out) != 0, "neg with dtype mismatch should fail");

  numc_ctx_free(ctx);
  return 0;
}

static int test_neg_shape_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape_a[] = {4};
  size_t shape_o[] = {5};
  NumcArray *a   = numc_array_zeros(ctx, shape_a, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape_o, 1, NUMC_DTYPE_FLOAT32);

  ASSERT_MSG(numc_neg(a, out) != 0, "neg with shape mismatch should fail");

  numc_ctx_free(ctx);
  return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Math: Unary inplace ops (neg_inplace)
 * ═══════════════════════════════════════════════════════════════════════ */

static int test_neg_inplace_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {1.0f, -2.0f, 3.0f, -4.0f};
  numc_array_write(a, da);

  int err = numc_neg_inplace(a);
  ASSERT_MSG(err == 0, "neg_inplace should succeed");

  float *r = (float *)numc_array_data(a);
  ASSERT_MSG(r[0] == -1.0f && r[1] == 2.0f && r[2] == -3.0f && r[3] == 4.0f,
             "neg_inplace results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_neg_inplace_int32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);

  int32_t da[] = {10, -20, 30, -40};
  numc_array_write(a, da);

  int err = numc_neg_inplace(a);
  ASSERT_MSG(err == 0, "neg_inplace int32 should succeed");

  int32_t *r = (int32_t *)numc_array_data(a);
  ASSERT_MSG(r[0] == -10 && r[1] == 20 && r[2] == -30 && r[3] == 40,
             "neg_inplace int32 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_neg_inplace_int8(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT8);

  int8_t da[] = {5, -10, 15, -20};
  numc_array_write(a, da);

  int err = numc_neg_inplace(a);
  ASSERT_MSG(err == 0, "neg_inplace int8 should succeed");

  int8_t *r = (int8_t *)numc_array_data(a);
  ASSERT_MSG(r[0] == -5 && r[1] == 10 && r[2] == -15 && r[3] == 20,
             "neg_inplace int8 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_neg_inplace_float64(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT64);

  double da[] = {1.5, -2.5, 3.5};
  numc_array_write(a, da);

  int err = numc_neg_inplace(a);
  ASSERT_MSG(err == 0, "neg_inplace float64 should succeed");

  double *r = (double *)numc_array_data(a);
  ASSERT_MSG(r[0] == -1.5 && r[1] == 2.5 && r[2] == -3.5,
             "neg_inplace float64 results should match");

  numc_ctx_free(ctx);
  return 0;
}

static int test_neg_inplace_2d(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);

  int32_t da[] = {1, -2, 3, -4, 5, -6};
  numc_array_write(a, da);

  int err = numc_neg_inplace(a);
  ASSERT_MSG(err == 0, "2D neg_inplace should succeed");

  int32_t *r = (int32_t *)numc_array_data(a);
  ASSERT_MSG(r[0] == -1 && r[1] == 2 && r[2] == -3, "neg_inplace 2D row 0");
  ASSERT_MSG(r[3] == 4 && r[4] == -5 && r[5] == 6, "neg_inplace 2D row 1");

  numc_ctx_free(ctx);
  return 0;
}

static int test_neg_inplace_contiguous_2d(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3, 2};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);

  int32_t da[] = {1, -2, 3, -4, 5, -6};
  numc_array_write(a, da);

  int err = numc_neg_inplace(a);
  ASSERT_MSG(err == 0, "neg_inplace contiguous 2d should succeed");

  int32_t *r = (int32_t *)numc_array_data(a);
  ASSERT_MSG(r[0] == -1 && r[1] == 2, "inplace row 0");
  ASSERT_MSG(r[2] == -3 && r[3] == 4, "inplace row 1");
  ASSERT_MSG(r[4] == -5 && r[5] == 6, "inplace row 2");

  numc_ctx_free(ctx);
  return 0;
}

static int test_neg_inplace_zeros(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  int err = numc_neg_inplace(a);
  ASSERT_MSG(err == 0, "neg_inplace of zeros should succeed");

  float *r = (float *)numc_array_data(a);
  for (size_t i = 0; i < 4; i++) {
    ASSERT_MSG(r[i] == 0.0f, "neg_inplace of zero should be zero");
  }

  numc_ctx_free(ctx);
  return 0;
}

static int test_neg_inplace_null(void) {
  ASSERT_MSG(numc_neg_inplace(NULL) != 0, "neg_inplace with NULL should fail");
  return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Error API
 * ═══════════════════════════════════════════════════════════════════════ */

static int test_error_set_get(void) {
  numc_set_error(NUMC_ERR_SHAPE, "test shape error");
  NumcError err = numc_get_error();
  ASSERT_MSG(err.code == NUMC_ERR_SHAPE, "error code should be SHAPE");
  ASSERT_MSG(err.msg != NULL, "error message should not be NULL");
  return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Print (smoke test — just verify it doesn't crash)
 * ═══════════════════════════════════════════════════════════════════════ */

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

/* --- main --- */

int main(void) {
  int passes = 0, fails = 0;

  printf("=== numc Test Suite ===\n\n");

  printf("Context:\n");
  RUN_TEST(test_ctx_create);
  RUN_TEST(test_ctx_free_null);

  printf("\nArray creation:\n");
  RUN_TEST(test_array_create_basic);
  RUN_TEST(test_array_create_1d);
  RUN_TEST(test_array_create_high_dim);
  RUN_TEST(test_array_create_null_ctx);
  RUN_TEST(test_array_create_null_shape);
  RUN_TEST(test_array_create_zero_dim);
  RUN_TEST(test_array_create_all_dtypes);

  printf("\nShape & strides:\n");
  RUN_TEST(test_array_shape_strides);
  RUN_TEST(test_array_data_not_null);

  printf("\nZeros:\n");
  RUN_TEST(test_array_zeros);
  RUN_TEST(test_array_zeros_float);

  printf("\nFill:\n");
  RUN_TEST(test_array_fill);
  RUN_TEST(test_array_fill_float);

  printf("\nWrite:\n");
  RUN_TEST(test_array_write);
  RUN_TEST(test_array_write_null);
  RUN_TEST(test_array_write_2d);

  printf("\nCopy:\n");
  RUN_TEST(test_array_copy);
  RUN_TEST(test_array_copy_null);

  printf("\nCapacity:\n");
  RUN_TEST(test_array_capacity);

  printf("\nContiguous:\n");
  RUN_TEST(test_array_is_contiguous);
  RUN_TEST(test_array_contiguous_after_transpose);
  RUN_TEST(test_array_contiguous_null);
  RUN_TEST(test_array_contiguous_already);

  printf("\nReshape:\n");
  RUN_TEST(test_array_reshape_basic);
  RUN_TEST(test_array_reshape_to_1d);
  RUN_TEST(test_array_reshape_bad_size);
  RUN_TEST(test_array_reshape_null);
  RUN_TEST(test_array_reshape_copy_basic);

  printf("\nTranspose:\n");
  RUN_TEST(test_array_transpose_2d);
  RUN_TEST(test_array_transpose_3d);
  RUN_TEST(test_array_transpose_invalid_axis);
  RUN_TEST(test_array_transpose_duplicate_axis);
  RUN_TEST(test_array_transpose_copy_basic);
  RUN_TEST(test_array_transpose_copy_null);

  printf("\nSlice:\n");
  RUN_TEST(test_array_slice_basic);
  RUN_TEST(test_array_slice_step);
  RUN_TEST(test_array_slice_full);
  RUN_TEST(test_array_slice_2d);
  RUN_TEST(test_array_slice_out_of_bounds);
  RUN_TEST(test_array_slice_null);
  RUN_TEST(test_array_slice_is_view);

  printf("\nMultiple arrays:\n");
  RUN_TEST(test_multiple_arrays_same_ctx);

  printf("\nBinary math ops:\n");
  RUN_TEST(test_add_float32);
  RUN_TEST(test_sub_float32);
  RUN_TEST(test_mul_float32);
  RUN_TEST(test_div_float32);
  RUN_TEST(test_add_int32);
  RUN_TEST(test_add_int8);
  RUN_TEST(test_add_float64);
  RUN_TEST(test_binary_op_2d);
  RUN_TEST(test_binary_op_strided);

  printf("\nBinary math error cases:\n");
  RUN_TEST(test_binary_op_null);
  RUN_TEST(test_binary_op_type_mismatch);
  RUN_TEST(test_binary_op_shape_mismatch);
  RUN_TEST(test_binary_op_dim_mismatch);

  printf("\nScalar ops:\n");
  RUN_TEST(test_add_scalar_float32);
  RUN_TEST(test_sub_scalar_float32);
  RUN_TEST(test_mul_scalar_float32);
  RUN_TEST(test_div_scalar_float32);
  RUN_TEST(test_scalar_op_int32);

  printf("\nScalar error cases:\n");
  RUN_TEST(test_scalar_op_null);
  RUN_TEST(test_scalar_op_type_mismatch);
  RUN_TEST(test_scalar_op_shape_mismatch);

  printf("\nScalar inplace ops:\n");
  RUN_TEST(test_add_scalar_inplace_float32);
  RUN_TEST(test_sub_scalar_inplace_float32);
  RUN_TEST(test_mul_scalar_inplace_float32);
  RUN_TEST(test_div_scalar_inplace_float32);
  RUN_TEST(test_scalar_inplace_int32);
  RUN_TEST(test_scalar_inplace_null);

  printf("\nUnary ops (neg):\n");
  RUN_TEST(test_neg_float32);
  RUN_TEST(test_neg_int32);
  RUN_TEST(test_neg_int8);
  RUN_TEST(test_neg_float64);
  RUN_TEST(test_neg_2d);
  RUN_TEST(test_neg_strided);
  RUN_TEST(test_neg_zeros);
  RUN_TEST(test_neg_null);
  RUN_TEST(test_neg_type_mismatch);
  RUN_TEST(test_neg_shape_mismatch);

  printf("\nUnary inplace ops (neg_inplace):\n");
  RUN_TEST(test_neg_inplace_float32);
  RUN_TEST(test_neg_inplace_int32);
  RUN_TEST(test_neg_inplace_int8);
  RUN_TEST(test_neg_inplace_float64);
  RUN_TEST(test_neg_inplace_2d);
  RUN_TEST(test_neg_inplace_contiguous_2d);
  RUN_TEST(test_neg_inplace_zeros);
  RUN_TEST(test_neg_inplace_null);

  printf("\nError API:\n");
  RUN_TEST(test_error_set_get);

  printf("\nPrint (smoke tests):\n");
  RUN_TEST(test_array_print_smoke);
  RUN_TEST(test_array_print_float);
  RUN_TEST(test_array_print_1d);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
