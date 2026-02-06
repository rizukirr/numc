/**
 * @file test_array_full.c
 * @brief Test array_full with different types
 */

#include "array.h"
#include "types.h"
#include <assert.h>
#include <stdio.h>

void test_fill_int(void) {
  printf("Testing array_full with NUMC_TYPE_INT...\n");

  ArrayCreate src = {
      .ndim = 1,
      .shape = (size_t[]){10},
      .numc_type = NUMC_TYPE_INT,
      .data = NULL,
      .owns_data = true,
  };
  int value = 42;
  Array *arr = array_full(&src, &value);
  assert(arr != NULL);

  int *data = (int *)arr->data;
  for (size_t i = 0; i < arr->size; i++) {
    assert(data[i] == 42);
  }

  array_free(arr);
  printf("  ✓ INT fill works\n");
}

void test_fill_float(void) {
  printf("Testing array_full with NUMC_TYPE_FLOAT...\n");

  ArrayCreate src = {
      .ndim = 1,
      .shape = (size_t[]){10},
      .numc_type = NUMC_TYPE_FLOAT,
      .data = NULL,
      .owns_data = true,
  };
  float value = 3.14f;
  Array *arr = array_full(&src, &value);
  assert(arr != NULL);

  float *data = (float *)arr->data;
  for (size_t i = 0; i < arr->size; i++) {
    assert(data[i] == 3.14f);
  }

  array_free(arr);
  printf("  ✓ FLOAT fill works\n");
}

void test_fill_double(void) {
  printf("Testing array_full with NUMC_TYPE_DOUBLE...\n");

  double value = 2.718;
  ArrayCreate src = {
      .ndim = 1,
      .shape = (size_t[]){10},
      .numc_type = NUMC_TYPE_DOUBLE,
      .data = NULL,
      .owns_data = true,
  };
  Array *arr = array_full(&src, &value);
  assert(arr != NULL);

  double *data = (double *)arr->data;
  for (size_t i = 0; i < arr->size; i++) {
    assert(data[i] == 2.718);
  }

  array_free(arr);
  printf("  ✓ DOUBLE fill works\n");
}

void test_fill_2d(void) {
  printf("Testing array_full with 2D array...\n");

  int value = 99;
  ArrayCreate src = {
      .ndim = 2,
      .shape = (size_t[]){3, 4},
      .numc_type = NUMC_TYPE_INT,
      .data = NULL,
      .owns_data = true,
  };
  Array *arr = array_full(&src, &value);
  assert(arr != NULL);
  assert(arr->size == 12);

  int *data = (int *)arr->data;
  for (size_t i = 0; i < arr->size; i++) {
    assert(data[i] == 99);
  }

  array_free(arr);
  printf("  ✓ 2D fill works\n");
}

int main(void) {
  printf("=== Running Array Fill Tests ===\n\n");

  test_fill_int();
  test_fill_float();
  test_fill_double();
  test_fill_2d();

  printf("\n=== All Array Fill Tests Passed ===\n");
  return 0;
}
