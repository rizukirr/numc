/**
 * @file test_linspace.c
 * @brief Test array_linspace function
 */

#include <numc/numc.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>

void test_linspace_basic_float(void) {
  printf("Testing array_linspace basic (FLOAT)...\n");

  Array *arr = array_linspace(0, 10, 5, NUMC_TYPE_FLOAT);
  assert(arr != NULL);
  assert(arr->size == 5);
  assert(arr->ndim == 1);
  assert(arr->numc_type == NUMC_TYPE_FLOAT);

  float expected[] = {0.0f, 2.5f, 5.0f, 7.5f, 10.0f};
  float *data = (float *)arr->data;
  for (size_t i = 0; i < arr->size; i++) {
    assert(fabsf(data[i] - expected[i]) < 1e-6f);
  }

  array_free(arr);
  printf("  ✓ linspace(0, 10, 5, FLOAT) works\n");
}

void test_linspace_basic_int(void) {
  printf("Testing array_linspace basic (INT)...\n");

  Array *arr = array_linspace(0, 100, 11, NUMC_TYPE_INT);
  assert(arr != NULL);
  assert(arr->size == 11);

  int expected[] = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
  int *data = (int *)arr->data;
  for (size_t i = 0; i < arr->size; i++) {
    assert(data[i] == expected[i]);
  }

  array_free(arr);
  printf("  ✓ linspace(0, 100, 11, INT) works\n");
}

void test_linspace_single_element(void) {
  printf("Testing array_linspace single element...\n");

  Array *arr = array_linspace(5, 5, 1, NUMC_TYPE_INT);
  assert(arr != NULL);
  assert(arr->size == 1);
  assert(((int *)arr->data)[0] == 5);

  array_free(arr);

  // num=1 with different start/stop should return start
  Array *arr2 = array_linspace(3, 10, 1, NUMC_TYPE_INT);
  assert(arr2 != NULL);
  assert(arr2->size == 1);
  assert(((int *)arr2->data)[0] == 3);

  array_free(arr2);
  printf("  ✓ linspace single element works\n");
}

void test_linspace_two_elements(void) {
  printf("Testing array_linspace two elements...\n");

  Array *arr = array_linspace(0, 10, 2, NUMC_TYPE_FLOAT);
  assert(arr != NULL);
  assert(arr->size == 2);

  float *data = (float *)arr->data;
  assert(fabsf(data[0] - 0.0f) < 1e-6f);
  assert(fabsf(data[1] - 10.0f) < 1e-6f);

  array_free(arr);
  printf("  ✓ linspace(0, 10, 2) works\n");
}

void test_linspace_descending(void) {
  printf("Testing array_linspace descending...\n");

  Array *arr = array_linspace(10, 0, 5, NUMC_TYPE_FLOAT);
  assert(arr != NULL);
  assert(arr->size == 5);

  float expected[] = {10.0f, 7.5f, 5.0f, 2.5f, 0.0f};
  float *data = (float *)arr->data;
  for (size_t i = 0; i < arr->size; i++) {
    assert(fabsf(data[i] - expected[i]) < 1e-6f);
  }

  array_free(arr);
  printf("  ✓ linspace(10, 0, 5) descending works\n");
}

void test_linspace_negative_range(void) {
  printf("Testing array_linspace negative range...\n");

  Array *arr = array_linspace(-10, 10, 5, NUMC_TYPE_FLOAT);
  assert(arr != NULL);
  assert(arr->size == 5);

  float expected[] = {-10.0f, -5.0f, 0.0f, 5.0f, 10.0f};
  float *data = (float *)arr->data;
  for (size_t i = 0; i < arr->size; i++) {
    assert(fabsf(data[i] - expected[i]) < 1e-6f);
  }

  array_free(arr);
  printf("  ✓ linspace(-10, 10, 5) works\n");
}

void test_linspace_endpoint(void) {
  printf("Testing array_linspace endpoint precision...\n");

  // Verify that the last element is exactly stop
  Array *arr = array_linspace(0, 7, 8, NUMC_TYPE_FLOAT);
  assert(arr != NULL);
  assert(arr->size == 8);

  float *data = (float *)arr->data;
  assert(data[0] == 0.0f);
  assert(data[7] == 7.0f);

  array_free(arr);
  printf("  ✓ linspace endpoint is exact\n");
}

void test_linspace_double(void) {
  printf("Testing array_linspace with DOUBLE...\n");

  Array *arr = array_linspace(0, 1, 5, NUMC_TYPE_DOUBLE);
  assert(arr != NULL);
  assert(arr->size == 5);
  assert(arr->numc_type == NUMC_TYPE_DOUBLE);

  double expected[] = {0.0, 0.25, 0.5, 0.75, 1.0};
  double *data = (double *)arr->data;
  for (size_t i = 0; i < arr->size; i++) {
    assert(fabs(data[i] - expected[i]) < 1e-12);
  }

  array_free(arr);
  printf("  ✓ linspace(0, 1, 5, DOUBLE) works\n");
}

void test_linspace_invalid(void) {
  printf("Testing array_linspace invalid inputs...\n");

  // num == 0
  Array *arr = array_linspace(0, 10, 0, NUMC_TYPE_INT);
  assert(arr == NULL);

  printf("  ✓ linspace invalid inputs rejected\n");
}

int main(void) {
  printf("=== Running Array Linspace Tests ===\n\n");

  test_linspace_basic_float();
  test_linspace_basic_int();
  test_linspace_single_element();
  test_linspace_two_elements();
  test_linspace_descending();
  test_linspace_negative_range();
  test_linspace_endpoint();
  test_linspace_double();
  test_linspace_invalid();

  printf("\n=== All Array Linspace Tests Passed ===\n");
  return 0;
}
