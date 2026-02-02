/**
 * @file test_array_copy.c
 * @brief Test array copying functionality
 */

#include "array.h"
#include "types.h"
#include <assert.h>
#include <stdio.h>

void test_copy_contiguous(void) {
  printf("Testing copy of contiguous array...\n");

  Array *arr = array_create(2, (size_t[]){3, 4}, DTYPE_INT, NULL);
  int *data = (int *)arr->data;
  for (int i = 0; i < 12; i++) {
    data[i] = i;
  }

  Array *copy = array_copy(arr);
  assert(copy != NULL);
  assert(copy->ndim == arr->ndim);
  assert(copy->shape[0] == arr->shape[0]);
  assert(copy->shape[1] == arr->shape[1]);
  assert(copy->size == arr->size);
  assert(copy->data != arr->data); // Different memory

  // Check data is copied
  int *copy_data = (int *)copy->data;
  for (int i = 0; i < 12; i++) {
    assert(copy_data[i] == data[i]);
  }

  // Modify original, copy should be unchanged
  data[0] = 999;
  assert(copy_data[0] == 0); // Still original value

  array_free(arr);
  array_free(copy);
  printf("  ✓ Contiguous array copy works\n");
}

void test_copy_slice(void) {
  printf("Testing copy of sliced array...\n");

  // Create array and slice it
  Array *arr = array_create(1, (size_t[]){10}, DTYPE_INT, NULL);
  int *data = (int *)arr->data;
  for (int i = 0; i < 10; i++) {
    data[i] = i;
  }

  Array *slice = array_slice(arr, (size_t[]){2}, (size_t[]){8}, (size_t[]){2});
  // Slice should be [2, 4, 6]

  Array *copy = array_to_contiguous(slice);
  assert(copy != NULL);
  assert(copy->size == slice->size);
  assert(array_is_contiguous(copy)); // Copy should be contiguous

  int *copy_data = (int *)copy->data;
  int expected[] = {2, 4, 6};
  for (int i = 0; i < 3; i++) {
    assert(copy_data[i] == expected[i]);
  }

  array_free(copy);
  array_free(slice);
  array_free(arr);
  printf("  ✓ Sliced array copy works\n");
}

void test_is_contiguous(void) {
  printf("Testing array_is_contiguous...\n");

  // Contiguous array
  Array *arr = array_create(2, (size_t[]){3, 4}, DTYPE_INT, NULL);
  assert(array_is_contiguous(arr) != 0);

  // Non-contiguous slice
  Array *slice =
      array_slice(arr, (size_t[]){0, 0}, (size_t[]){3, 4}, (size_t[]){1, 2});
  assert(array_is_contiguous(slice) == 0);

  array_free(slice);
  array_free(arr);
  printf("  ✓ Contiguity detection works\n");
}

int main(void) {
  printf("=== Running Array Copy Tests ===\n\n");

  test_copy_contiguous();
  test_copy_slice();
  test_is_contiguous();

  printf("\n=== All Array Copy Tests Passed ===\n");
  return 0;
}
