/**
 * @file test_arange.c
 * @brief Test array_arange function
 */

#include <numc/numc.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>

void test_arange_basic(void) {
  printf("Testing array_arange basic...\n");

  Array *arr = array_arange(0, 10, 1, NUMC_TYPE_INT);
  assert(arr != NULL);
  assert(arr->size == 10);
  assert(arr->ndim == 1);
  assert(arr->numc_type == NUMC_TYPE_INT);

  int *data = (int *)arr->data;
  for (int i = 0; i < 10; i++) {
    assert(data[i] == i);
  }

  array_free(arr);
  printf("  ✓ basic arange(0, 10, 1) works\n");
}

void test_arange_step(void) {
  printf("Testing array_arange with step...\n");

  Array *arr = array_arange(5, 20, 2, NUMC_TYPE_INT);
  assert(arr != NULL);
  assert(arr->size == 8);

  int expected[] = {5, 7, 9, 11, 13, 15, 17, 19};
  int *data = (int *)arr->data;
  for (size_t i = 0; i < arr->size; i++) {
    assert(data[i] == expected[i]);
  }

  array_free(arr);
  printf("  ✓ arange(5, 20, 2) works\n");
}

void test_arange_large_step(void) {
  printf("Testing array_arange with large step...\n");

  Array *arr = array_arange(0, 100, 10, NUMC_TYPE_INT);
  assert(arr != NULL);
  assert(arr->size == 10);

  int expected[] = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90};
  int *data = (int *)arr->data;
  for (size_t i = 0; i < arr->size; i++) {
    assert(data[i] == expected[i]);
  }

  array_free(arr);
  printf("  ✓ arange(0, 100, 10) works\n");
}

void test_arange_negative_step(void) {
  printf("Testing array_arange with negative step...\n");

  Array *arr = array_arange(10, 0, -1, NUMC_TYPE_INT);
  assert(arr != NULL);
  assert(arr->size == 10);

  int expected[] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  int *data = (int *)arr->data;
  for (size_t i = 0; i < arr->size; i++) {
    assert(data[i] == expected[i]);
  }

  array_free(arr);

  // Negative step with step size > 1
  Array *arr2 = array_arange(20, 5, -2, NUMC_TYPE_INT);
  assert(arr2 != NULL);
  assert(arr2->size == 8);

  int expected2[] = {20, 18, 16, 14, 12, 10, 8, 6};
  int *data2 = (int *)arr2->data;
  for (size_t i = 0; i < arr2->size; i++) {
    assert(data2[i] == expected2[i]);
  }

  array_free(arr2);
  printf("  ✓ arange with negative step works\n");
}

void test_arange_float(void) {
  printf("Testing array_arange with float type...\n");

  Array *arr = array_arange(0, 5, 1, NUMC_TYPE_FLOAT);
  assert(arr != NULL);
  assert(arr->size == 5);
  assert(arr->numc_type == NUMC_TYPE_FLOAT);

  float *data = (float *)arr->data;
  for (int i = 0; i < 5; i++) {
    assert(data[i] == (float)i);
  }

  array_free(arr);
  printf("  ✓ arange with FLOAT type works\n");
}

void test_arange_single_element(void) {
  printf("Testing array_arange single element...\n");

  Array *arr = array_arange(0, 1, 1, NUMC_TYPE_INT);
  assert(arr != NULL);
  assert(arr->size == 1);
  assert(((int *)arr->data)[0] == 0);

  array_free(arr);
  printf("  ✓ arange single element works\n");
}

void test_arange_invalid(void) {
  printf("Testing array_arange invalid inputs...\n");

  // step == 0
  Array *arr1 = array_arange(0, 10, 0, NUMC_TYPE_INT);
  assert(arr1 == NULL);

  // positive step with start >= stop
  Array *arr2 = array_arange(10, 5, 1, NUMC_TYPE_INT);
  assert(arr2 == NULL);

  // negative step with start <= stop
  Array *arr3 = array_arange(0, 10, -1, NUMC_TYPE_INT);
  assert(arr3 == NULL);

  printf("  ✓ arange invalid inputs rejected\n");
}

int main(void) {
  printf("=== Running Array Arange Tests ===\n\n");

  test_arange_basic();
  test_arange_step();
  test_arange_large_step();
  test_arange_negative_step();
  test_arange_float();
  test_arange_single_element();
  test_arange_invalid();

  printf("\n=== All Array Arange Tests Passed ===\n");
  return 0;
}
