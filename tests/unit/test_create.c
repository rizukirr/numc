/**
 * @file test_array_create.c
 * @brief Test array creation functions
 */

#include <numc/numc.h>
#include <assert.h>
#include <stdio.h>

void test_array_create(void) {
  printf("Testing array_create...\n");

  // Test creating empty array (NULL data)

  ArrayCreate empty_create = {
      .ndim = 2,
      .shape = (size_t[]){3, 4},
      .numc_type = NUMC_TYPE_INT,
      .data = NULL,
      .owns_data = true,
  };
  Array *empty = array_create(&empty_create);
  assert(empty != NULL);
  assert(empty->ndim == 2);
  assert(empty->shape[0] == 3);
  assert(empty->shape[1] == 4);
  assert(empty->size == 12);
  assert(empty->numc_type == NUMC_TYPE_INT);
  assert(empty->elem_size == 4);
  assert(empty->owns_data == 1);
  array_free(empty);

  // Test creating array with initial data
  int init_data[] = {1, 2, 3, 4, 5, 6};
  ArrayCreate arr_with_data_create = {
      .ndim = 2,
      .shape = (size_t[]){2, 3},
      .numc_type = NUMC_TYPE_INT,
      .data = init_data,
      .owns_data = true,
  };
  Array *arr_with_data = array_create(&arr_with_data_create);
  assert(arr_with_data != NULL);
  assert(arr_with_data->ndim == 2);
  assert(arr_with_data->size == 6);

  // Verify data was copied
  int *data = (int *)arr_with_data->data;
  for (int i = 0; i < 6; i++) {
    assert(data[i] == init_data[i]);
  }
  array_free(arr_with_data);

  printf("  ✓ array_create works\n");
}

void test_array_zeros(void) {
  printf("Testing array_zeros...\n");

  Array *arr = array_zeros(1, (size_t[]){5}, NUMC_TYPE_INT);
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
  Array *arr_int = array_ones(1, (size_t[]){5}, NUMC_TYPE_INT);
  assert(arr_int != NULL);
  int *data_int = (int *)arr_int->data;
  for (int i = 0; i < 5; i++) {
    assert(data_int[i] == 1);
  }
  array_free(arr_int);

  // Test float ones
  Array *arr_float = array_ones(1, (size_t[]){5}, NUMC_TYPE_FLOAT);
  assert(arr_float != NULL);
  float *data_float = (float *)arr_float->data;
  for (int i = 0; i < 5; i++) {
    assert(data_float[i] == 1.0f);
  }
  array_free(arr_float);

  printf("  ✓ array_ones works\n");
}

void test_array_full(void) {
  printf("Testing array_fill...\n");

  int value = 42;
  ArrayCreate src = {
      .ndim = 1,
      .shape = (size_t[]){5},
      .numc_type = NUMC_TYPE_INT,
      .data = NULL,
      .owns_data = true,
  };
  Array *arr = array_full(&src, &value);
  assert(arr != NULL);

  int *data = (int *)arr->data;
  for (int i = 0; i < 5; i++) {
    assert(data[i] == 42);
  }

  array_free(arr);
  printf("  ✓ array_fill works\n");
}

int main(void) {
  printf("=== Running Array Creation Tests ===\n\n");

  test_array_create();
  test_array_zeros();
  test_array_ones();
  test_array_full();

  printf("\n=== All Array Creation Tests Passed ===\n");
  return 0;
}
