/**
 * @file test_array_create.c
 * @brief Test array creation functions
 */

#include "array.h"
#include "dtype.h"
#include <assert.h>
#include <stdio.h>

void test_array_create(void) {
  printf("Testing array_create...\n");
  
  Array *arr = array_create(2, (size_t[]){3, 4}, DTYPE_INT);
  assert(arr != NULL);
  assert(arr->ndim == 2);
  assert(arr->shape[0] == 3);
  assert(arr->shape[1] == 4);
  assert(arr->size == 12);
  assert(arr->dtype == DTYPE_INT);
  assert(arr->elem_size == 4);
  assert(arr->owns_data == 1);
  
  array_free(arr);
  printf("  ✓ array_create works\n");
}

void test_array_zeros(void) {
  printf("Testing array_zeros...\n");
  
  Array *arr = array_zeros(1, (size_t[]){5}, DTYPE_INT);
  assert(arr != NULL);
  
  int *data = (int *)arr->data;
  for (int i = 0; i < 5; i++) {
    assert(data[i] == 0);
  }
  
  array_free(arr);
  printf("  ✓ array_zeros works\n");
}

void test_array_ones(void) {
  printf("Testing array_ones...\n");
  
  // Test integer ones
  Array *arr_int = array_ones(1, (size_t[]){5}, DTYPE_INT);
  assert(arr_int != NULL);
  int *data_int = (int *)arr_int->data;
  for (int i = 0; i < 5; i++) {
    assert(data_int[i] == 1);
  }
  array_free(arr_int);
  
  // Test float ones
  Array *arr_float = array_ones(1, (size_t[]){5}, DTYPE_FLOAT);
  assert(arr_float != NULL);
  float *data_float = (float *)arr_float->data;
  for (int i = 0; i < 5; i++) {
    assert(data_float[i] == 1.0f);
  }
  array_free(arr_float);
  
  printf("  ✓ array_ones works\n");
}

void test_array_fill(void) {
  printf("Testing array_fill...\n");
  
  int value = 42;
  Array *arr = array_fill(1, (size_t[]){5}, DTYPE_INT, &value);
  assert(arr != NULL);
  
  int *data = (int *)arr->data;
  for (int i = 0; i < 5; i++) {
    assert(data[i] == 42);
  }
  
  array_free(arr);
  printf("  ✓ array_fill works\n");
}

void test_array_batch(void) {
  printf("Testing array_batch...\n");
  
  int batch_data[] = {10, 20, 30, 40, 50};
  Array *arr = array_batch(1, (size_t[]){5}, DTYPE_INT, batch_data);
  assert(arr != NULL);
  
  int *data = (int *)arr->data;
  for (int i = 0; i < 5; i++) {
    assert(data[i] == batch_data[i]);
  }
  
  array_free(arr);
  printf("  ✓ array_batch works\n");
}

int main(void) {
  printf("=== Running Array Creation Tests ===\n\n");
  
  test_array_create();
  test_array_zeros();
  test_array_ones();
  test_array_fill();
  test_array_batch();
  
  printf("\n=== All Array Creation Tests Passed ===\n");
  return 0;
}
