/**
 * @file test_array_concat.c
 * @brief Test array concatenation with fast path optimization
 */

#include "array.h"
#include "shape.h"
#include "types.h"
#include <assert.h>
#include <stdio.h>

void test_concat_axis0_1d(void) {
  printf("Testing 1D concatenation along axis 0...\n");

  // Create [1, 2, 3]
  Array *a = array_create(&(ArrayCreate){.ndim = 1, .shape = (size_t[]){3}, .numc_type = NUMC_TYPE_INT, .data = NULL, .owns_data = true});
  int *data_a = (int *)a->data;
  data_a[0] = 1;
  data_a[1] = 2;
  data_a[2] = 3;

  // Create [4, 5]
  Array *b = array_create(&(ArrayCreate){.ndim = 1, .shape = (size_t[]){2}, .numc_type = NUMC_TYPE_INT, .data = NULL, .owns_data = true});
  int *data_b = (int *)b->data;
  data_b[0] = 4;
  data_b[1] = 5;

  // Concatenate: should get [1, 2, 3, 4, 5]
  Array *result = array_concatenate(a, b, 0);
  assert(result != NULL);
  assert(result->shape[0] == 5);
  assert(result->size == 5);

  int *data_result = (int *)result->data;
  assert(data_result[0] == 1);
  assert(data_result[1] == 2);
  assert(data_result[2] == 3);
  assert(data_result[3] == 4);
  assert(data_result[4] == 5);

  array_free(a);
  array_free(b);
  array_free(result);
  printf("  ✓ 1D concatenation works (fast path)\n");
}

void test_concat_axis0_2d(void) {
  printf("Testing 2D concatenation along axis 0 (rows)...\n");

  // Create [[1, 2, 3], [4, 5, 6]] - shape [2, 3]
  Array *a = array_create(&(ArrayCreate){.ndim = 2, .shape = (size_t[]){2, 3}, .numc_type = NUMC_TYPE_INT, .data = NULL, .owns_data = true});
  int *data_a = (int *)a->data;
  for (int i = 0; i < 6; i++) {
    data_a[i] = i + 1;
  }

  // Create [[7, 8, 9]] - shape [1, 3]
  Array *b = array_create(&(ArrayCreate){.ndim = 2, .shape = (size_t[]){1, 3}, .numc_type = NUMC_TYPE_INT, .data = NULL, .owns_data = true});
  int *data_b = (int *)b->data;
  for (int i = 0; i < 3; i++) {
    data_b[i] = i + 7;
  }

  // Concatenate along axis 0: should get shape [3, 3]
  Array *result = array_concatenate(a, b, 0);
  assert(result != NULL);
  assert(result->shape[0] == 3);
  assert(result->shape[1] == 3);
  assert(result->size == 9);

  int *data_result = (int *)result->data;
  for (int i = 0; i < 9; i++) {
    assert(data_result[i] == i + 1);
  }

  array_free(a);
  array_free(b);
  array_free(result);
  printf("  ✓ 2D row concatenation works (fast path)\n");
}

void test_concat_axis1_2d(void) {
  printf("Testing 2D concatenation along axis 1 (columns)...\n");

  // Create [[1, 2], [3, 4]] - shape [2, 2]
  Array *a = array_create(&(ArrayCreate){.ndim = 2, .shape = (size_t[]){2, 2}, .numc_type = NUMC_TYPE_INT, .data = NULL, .owns_data = true});
  int *data_a = (int *)a->data;
  data_a[0] = 1;
  data_a[1] = 2;
  data_a[2] = 3;
  data_a[3] = 4;

  // Create [[5], [6]] - shape [2, 1]
  Array *b = array_create(&(ArrayCreate){.ndim = 2, .shape = (size_t[]){2, 1}, .numc_type = NUMC_TYPE_INT, .data = NULL, .owns_data = true});
  int *data_b = (int *)b->data;
  data_b[0] = 5;
  data_b[1] = 6;

  // Concatenate along axis 1: should get [[1, 2, 5], [3, 4, 6]] - shape [2, 3]
  Array *result = array_concatenate(a, b, 1);
  assert(result != NULL);
  assert(result->shape[0] == 2);
  assert(result->shape[1] == 3);
  assert(result->size == 6);

  int expected[] = {1, 2, 5, 3, 4, 6};
  int *data_result = (int *)result->data;
  for (int i = 0; i < 6; i++) {
    assert(data_result[i] == expected[i]);
  }

  array_free(a);
  array_free(b);
  array_free(result);
  printf("  ✓ 2D column concatenation works (slow path)\n");
}

void test_concat_with_slices(void) {
  printf("Testing concatenation with sliced (non-contiguous) arrays...\n");

  // Create [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  Array *arr = array_create(&(ArrayCreate){.ndim = 1, .shape = (size_t[]){10}, .numc_type = NUMC_TYPE_INT, .data = NULL, .owns_data = true});
  int *data = (int *)arr->data;
  for (int i = 0; i < 10; i++) {
    data[i] = i;
  }

  // Slice [0:5] -> [0, 1, 2, 3, 4]
  Array *a = array_slice(arr, (size_t[]){0}, (size_t[]){5}, (size_t[]){1});

  // Slice [5:10] -> [5, 6, 7, 8, 9]
  Array *b = array_slice(arr, (size_t[]){5}, (size_t[]){10}, (size_t[]){1});

  // Both slices are contiguous, so should use fast path
  assert(array_is_contiguous(a));
  assert(array_is_contiguous(b));

  // Concatenate
  Array *result = array_concatenate(a, b, 0);
  assert(result != NULL);
  assert(result->size == 10);

  int *result_data = (int *)result->data;
  for (int i = 0; i < 10; i++) {
    assert(result_data[i] == i);
  }

  array_free(arr);
  array_free(a);
  array_free(b);
  array_free(result);
  printf("  ✓ Concatenation with slices works\n");
}

void test_concat_float_arrays(void) {
  printf("Testing concatenation with float arrays...\n");

  // Create [1.0, 2.0, 3.0]
  Array *a = array_create(&(ArrayCreate){.ndim = 1, .shape = (size_t[]){3}, .numc_type = NUMC_TYPE_FLOAT, .data = NULL, .owns_data = true});
  float *data_a = (float *)a->data;
  data_a[0] = 1.0f;
  data_a[1] = 2.0f;
  data_a[2] = 3.0f;

  // Create [4.0, 5.0]
  Array *b = array_create(&(ArrayCreate){.ndim = 1, .shape = (size_t[]){2}, .numc_type = NUMC_TYPE_FLOAT, .data = NULL, .owns_data = true});
  float *data_b = (float *)b->data;
  data_b[0] = 4.0f;
  data_b[1] = 5.0f;

  // Concatenate
  Array *result = array_concatenate(a, b, 0);
  assert(result != NULL);
  assert(result->size == 5);

  float *data_result = (float *)result->data;
  assert(data_result[0] == 1.0f);
  assert(data_result[1] == 2.0f);
  assert(data_result[2] == 3.0f);
  assert(data_result[3] == 4.0f);
  assert(data_result[4] == 5.0f);

  array_free(a);
  array_free(b);
  array_free(result);
  printf("  ✓ Float concatenation works (fast path)\n");
}

int main(void) {
  printf("=== Running Array Concatenation Tests ===\n\n");

  test_concat_axis0_1d();
  test_concat_axis0_2d();
  test_concat_axis1_2d();
  test_concat_with_slices();
  test_concat_float_arrays();

  printf("\n=== All Array Concatenation Tests Passed ===\n");
  return 0;
}
