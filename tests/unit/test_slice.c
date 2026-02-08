/**
 * @file test_array_slice.c
 * @brief Test array slicing functionality
 */

#include <numc/numc.h>
#include <assert.h>
#include <stdio.h>

void test_basic_slice(void) {
  printf("Testing basic array slice...\n");
  
  // Create 1D array [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  Array *arr = array_create(&(ArrayCreate){.ndim = 1, .shape = (size_t[]){10}, .numc_type = NUMC_TYPE_INT, .data = NULL, .owns_data = true});
  int *data = (int *)arr->data;
  for (int i = 0; i < 10; i++) {
    data[i] = i;
  }
  
  // Slice [2:7:1] -> [2, 3, 4, 5, 6]
  Array *sliced = array_slice(arr, (size_t[]){2}, (size_t[]){7}, (size_t[]){1});
  assert(sliced != NULL);
  assert(sliced->size == 5);
  assert(sliced->owns_data == 0); // View doesn't own data
  
  int *sliced_data = (int *)sliced->data;
  for (int i = 0; i < 5; i++) {
    assert(sliced_data[i] == i + 2);
  }
  
  array_free(sliced);
  array_free(arr);
  printf("  ✓ Basic slice works\n");
}

void test_slice_with_step(void) {
  printf("Testing slice with step...\n");
  
  // Create 1D array [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  Array *arr = array_create(&(ArrayCreate){.ndim = 1, .shape = (size_t[]){10}, .numc_type = NUMC_TYPE_INT, .data = NULL, .owns_data = true});
  int *data = (int *)arr->data;
  for (int i = 0; i < 10; i++) {
    data[i] = i;
  }
  
  // Slice [0:10:2] -> [0, 2, 4, 6, 8]
  Array *sliced = array_slice(arr, (size_t[]){0}, (size_t[]){10}, (size_t[]){2});
  assert(sliced != NULL);
  assert(sliced->size == 5);
  
  // Use array_get for proper stride-aware access
  for (int i = 0; i < 5; i++) {
    int *ptr = (int *)array_get(sliced, (size_t[]){i});
    assert(*ptr == i * 2);
  }
  
  array_free(sliced);
  array_free(arr);
  printf("  ✓ Slice with step works\n");
}

void test_2d_slice(void) {
  printf("Testing 2D array slice...\n");
  
  // Create 2D array [4, 5]
  Array *arr = array_create(&(ArrayCreate){.ndim = 2, .shape = (size_t[]){4, 5}, .numc_type = NUMC_TYPE_INT, .data = NULL, .owns_data = true});
  int *data = (int *)arr->data;
  for (int i = 0; i < 20; i++) {
    data[i] = i;
  }
  
  // Slice rows [1:3], cols [1:4] -> shape [2, 3]
  Array *sliced = array_slice(arr, (size_t[]){1, 1}, (size_t[]){3, 4}, (size_t[]){1, 1});
  assert(sliced != NULL);
  assert(sliced->shape[0] == 2);
  assert(sliced->shape[1] == 3);
  assert(sliced->size == 6);
  
  array_free(sliced);
  array_free(arr);
  printf("  ✓ 2D slice works\n");
}

int main(void) {
  printf("=== Running Array Slice Tests ===\n\n");
  
  test_basic_slice();
  test_slice_with_step();
  test_2d_slice();
  
  printf("\n=== All Array Slice Tests Passed ===\n");
  return 0;
}
