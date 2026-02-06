/**
 * @file test_array_reshape.c
 * @brief Test array reshaping functionality
 */

#include "array.h"
#include "types.h"
#include <assert.h>
#include <stdio.h>

void test_reshape_2d_to_1d(void) {
  printf("Testing reshape [2, 6] -> [12]...\n");
  
  Array *arr = array_create(&(ArrayCreate){.ndim = 2, .shape = (size_t[]){2, 6}, .numc_type = NUMC_TYPE_INT, .data = NULL, .owns_data = true});
  int *data = (int *)arr->data;
  for (int i = 0; i < 12; i++) {
    data[i] = i;
  }
  
  int result = array_reshape(arr, 1, (size_t[]){12});
  assert(result == 0);
  assert(arr->ndim == 1);
  assert(arr->shape[0] == 12);
  assert(arr->size == 12);
  
  // Data should remain the same
  for (int i = 0; i < 12; i++) {
    assert(data[i] == i);
  }
  
  array_free(arr);
  printf("  ✓ 2D to 1D reshape works\n");
}

void test_reshape_1d_to_2d(void) {
  printf("Testing reshape [12] -> [3, 4]...\n");
  
  Array *arr = array_create(&(ArrayCreate){.ndim = 1, .shape = (size_t[]){12}, .numc_type = NUMC_TYPE_INT, .data = NULL, .owns_data = true});
  int *data = (int *)arr->data;
  for (int i = 0; i < 12; i++) {
    data[i] = i;
  }
  
  int result = array_reshape(arr, 2, (size_t[]){3, 4});
  assert(result == 0);
  assert(arr->ndim == 2);
  assert(arr->shape[0] == 3);
  assert(arr->shape[1] == 4);
  assert(arr->size == 12);
  
  array_free(arr);
  printf("  ✓ 1D to 2D reshape works\n");
}

void test_reshape_invalid_size(void) {
  printf("Testing reshape with mismatched size...\n");
  
  Array *arr = array_create(&(ArrayCreate){.ndim = 1, .shape = (size_t[]){12}, .numc_type = NUMC_TYPE_INT, .data = NULL, .owns_data = true});
  
  // Try to reshape to different size (should fail)
  int result = array_reshape(arr, 2, (size_t[]){2, 5});
  assert(result != 0); // Should fail
  
  // Original shape should be unchanged
  assert(arr->ndim == 1);
  assert(arr->shape[0] == 12);
  
  array_free(arr);
  printf("  ✓ Invalid reshape properly rejected\n");
}

int main(void) {
  printf("=== Running Array Reshape Tests ===\n\n");
  
  test_reshape_2d_to_1d();
  test_reshape_1d_to_2d();
  test_reshape_invalid_size();
  
  printf("\n=== All Array Reshape Tests Passed ===\n");
  return 0;
}
