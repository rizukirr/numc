/**
 * @file test_array_transpose.c
 * @brief Test array transpose functionality
 */

#include "array.h"
#include "dtype.h"
#include <assert.h>
#include <stdio.h>

void test_transpose_2d(void) {
  printf("Testing 2D array transpose...\n");
  
  // Create 3×4 array
  Array *arr = array_create(2, (size_t[]){3, 4}, DTYPE_INT, NULL);
  int *data = (int *)arr->data;
  
  // Fill: [[0,1,2,3], [4,5,6,7], [8,9,10,11]]
  for (int i = 0; i < 12; i++) {
    data[i] = i;
  }
  
  // Store original data pointer and strides
  void *orig_data = arr->data;
  size_t orig_stride0 = arr->strides[0];
  size_t orig_stride1 = arr->strides[1];
  
  // Transpose: swap axes 0 and 1
  size_t axes[] = {1, 0};
  int result = array_transpose(arr, axes);
  assert(result == 0);
  
  // Check shape changed to 4×3
  assert(arr->shape[0] == 4);
  assert(arr->shape[1] == 3);
  
  // Check strides were swapped
  assert(arr->strides[0] == orig_stride1);
  assert(arr->strides[1] == orig_stride0);
  
  // Check data pointer unchanged (view, not copy)
  assert(arr->data == orig_data);
  
  // Verify element access through strides
  // Original [0,0] = 0 -> Transposed [0,0] = 0
  assert(*array_geti(arr, (size_t[]){0, 0}) == 0);
  // Original [0,1] = 1 -> Transposed [1,0] = 1
  assert(*array_geti(arr, (size_t[]){1, 0}) == 1);
  // Original [1,0] = 4 -> Transposed [0,1] = 4
  assert(*array_geti(arr, (size_t[]){0, 1}) == 4);
  // Original [2,3] = 11 -> Transposed [3,2] = 11
  assert(*array_geti(arr, (size_t[]){3, 2}) == 11);
  
  array_free(arr);
  printf("  ✓ 2D transpose works\n");
}

void test_transpose_3d(void) {
  printf("Testing 3D array transpose...\n");
  
  // Create 2×3×4 array
  Array *arr = array_create(3, (size_t[]){2, 3, 4}, DTYPE_INT, NULL);
  int *data = (int *)arr->data;
  
  // Fill with sequential values
  for (int i = 0; i < 24; i++) {
    data[i] = i;
  }
  
  // Store original strides
  size_t orig_stride0 = arr->strides[0];
  size_t orig_stride1 = arr->strides[1];
  size_t orig_stride2 = arr->strides[2];
  
  // Transpose: reverse all axes [2, 1, 0]
  size_t axes[] = {2, 1, 0};
  int result = array_transpose(arr, axes);
  assert(result == 0);
  
  // Check shape changed to 4×3×2
  assert(arr->shape[0] == 4);
  assert(arr->shape[1] == 3);
  assert(arr->shape[2] == 2);
  
  // Check strides were reversed
  assert(arr->strides[0] == orig_stride2);
  assert(arr->strides[1] == orig_stride1);
  assert(arr->strides[2] == orig_stride0);
  
  // Verify specific element access
  // Original [0,0,0] = 0 -> Transposed [0,0,0] = 0
  assert(*array_geti(arr, (size_t[]){0, 0, 0}) == 0);
  // Original [0,0,1] = 1 -> Transposed [1,0,0] = 1
  assert(*array_geti(arr, (size_t[]){1, 0, 0}) == 1);
  // Original [1,2,3] = 23 -> Transposed [3,2,1] = 23
  assert(*array_geti(arr, (size_t[]){3, 2, 1}) == 23);
  
  array_free(arr);
  printf("  ✓ 3D transpose works\n");
}

void test_transpose_3d_partial(void) {
  printf("Testing 3D partial transpose...\n");
  
  // Create 2×3×4 array
  Array *arr = array_create(3, (size_t[]){2, 3, 4}, DTYPE_FLOAT, NULL);
  float *data = (float *)arr->data;
  
  // Fill with sequential values
  for (int i = 0; i < 24; i++) {
    data[i] = (float)i;
  }
  
  size_t orig_stride0 = arr->strides[0];
  size_t orig_stride1 = arr->strides[1];
  size_t orig_stride2 = arr->strides[2];
  
  // Transpose: move last axis to front [2, 0, 1]
  size_t axes[] = {2, 0, 1};
  int result = array_transpose(arr, axes);
  assert(result == 0);
  
  // Check shape changed to 4×2×3
  assert(arr->shape[0] == 4);
  assert(arr->shape[1] == 2);
  assert(arr->shape[2] == 3);
  
  // Check strides were permuted correctly
  assert(arr->strides[0] == orig_stride2);
  assert(arr->strides[1] == orig_stride0);
  assert(arr->strides[2] == orig_stride1);
  
  // Verify element access
  // Original [0,0,0] = 0.0 -> Transposed [0,0,0] = 0.0
  assert(*array_getf(arr, (size_t[]){0, 0, 0}) == 0.0f);
  // Original [0,1,2] = 6.0 -> Transposed [2,0,1] = 6.0
  assert(*array_getf(arr, (size_t[]){2, 0, 1}) == 6.0f);
  
  array_free(arr);
  printf("  ✓ 3D partial transpose works\n");
}

void test_transpose_identity(void) {
  printf("Testing identity transpose (no change)...\n");
  
  // Create 2×3 array
  Array *arr = array_create(2, (size_t[]){2, 3}, DTYPE_INT, NULL);
  int *data = (int *)arr->data;
  for (int i = 0; i < 6; i++) {
    data[i] = i;
  }
  
  size_t orig_stride0 = arr->strides[0];
  size_t orig_stride1 = arr->strides[1];
  
  // Identity transpose: [0, 1] (no swap)
  size_t axes[] = {0, 1};
  int result = array_transpose(arr, axes);
  assert(result == 0);
  
  // Shape should remain unchanged
  assert(arr->shape[0] == 2);
  assert(arr->shape[1] == 3);
  
  // Strides should remain unchanged
  assert(arr->strides[0] == orig_stride0);
  assert(arr->strides[1] == orig_stride1);
  
  array_free(arr);
  printf("  ✓ Identity transpose works\n");
}

void test_transpose_1d(void) {
  printf("Testing 1D array transpose...\n");
  
  // Create 1D array [0, 1, 2, 3, 4]
  Array *arr = array_create(1, (size_t[]){5}, DTYPE_INT, NULL);
  int *data = (int *)arr->data;
  for (int i = 0; i < 5; i++) {
    data[i] = i;
  }
  
  size_t orig_stride0 = arr->strides[0];
  
  // Transpose 1D array (identity, only one axis)
  size_t axes[] = {0};
  int result = array_transpose(arr, axes);
  assert(result == 0);
  
  // Nothing should change
  assert(arr->shape[0] == 5);
  assert(arr->strides[0] == orig_stride0);
  
  array_free(arr);
  printf("  ✓ 1D transpose works\n");
}

void test_transpose_null_array(void) {
  printf("Testing transpose with NULL array...\n");
  
  size_t axes[] = {1, 0};
  int result = array_transpose(NULL, axes);
  assert(result == -1);
  
  printf("  ✓ NULL array handled correctly\n");
}

void test_transpose_data_sharing(void) {
  printf("Testing transpose shares data (view behavior)...\n");
  
  // Create 2×3 array
  Array *arr = array_create(2, (size_t[]){2, 3}, DTYPE_INT, NULL);
  int *data = (int *)arr->data;
  data[0] = 100;
  data[5] = 200;
  
  void *orig_data = arr->data;
  
  // Transpose
  size_t axes[] = {1, 0};
  array_transpose(arr, axes);
  
  // Data pointer should be unchanged
  assert(arr->data == orig_data);
  
  // Modify through transposed view
  *array_geti(arr, (size_t[]){0, 0}) = 999;
  
  // Verify change visible in original data buffer
  assert(((int *)arr->data)[0] == 999);
  
  array_free(arr);
  printf("  ✓ Transpose shares data correctly\n");
}

void test_transpose_4d(void) {
  printf("Testing 4D array transpose...\n");
  
  // Create 2×2×2×2 array
  Array *arr = array_create(4, (size_t[]){2, 2, 2, 2}, DTYPE_INT, NULL);
  int *data = (int *)arr->data;
  
  // Fill with sequential values
  for (int i = 0; i < 16; i++) {
    data[i] = i;
  }
  
  // Transpose: swap first and last pairs [3, 2, 1, 0]
  size_t axes[] = {3, 2, 1, 0};
  int result = array_transpose(arr, axes);
  assert(result == 0);
  
  // Shape should be reversed (still 2×2×2×2 in this case)
  assert(arr->shape[0] == 2);
  assert(arr->shape[1] == 2);
  assert(arr->shape[2] == 2);
  assert(arr->shape[3] == 2);
  
  // Verify element access
  assert(*array_geti(arr, (size_t[]){0, 0, 0, 0}) == 0);
  assert(*array_geti(arr, (size_t[]){1, 1, 1, 1}) == 15);
  
  array_free(arr);
  printf("  ✓ 4D transpose works\n");
}

int main(void) {
  printf("Running array transpose tests...\n\n");
  
  test_transpose_2d();
  test_transpose_3d();
  test_transpose_3d_partial();
  test_transpose_identity();
  test_transpose_1d();
  test_transpose_null_array();
  test_transpose_data_sharing();
  test_transpose_4d();
  
  printf("\n✓ All transpose tests passed!\n");
  return 0;
}
