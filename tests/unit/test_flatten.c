/**
 * @file test_flatten.c
 * @brief Test array_flatten functionality
 */

#include <numc/numc.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

void test_flatten_1d_array(void) {
  // Flattening 1D array should be a no-op
  size_t shape[] = {10};
  Array *arr = array_zeros(1, shape, NUMC_TYPE_INT);

  // Fill with values
  for (size_t i = 0; i < arr->size; i++) {
    ((NUMC_INT *)arr->data)[i] = i;
  }

  int result = array_flatten(arr);
  assert(result == 0);
  assert(arr->ndim == 1);
  assert(arr->shape[0] == 10);
  assert(arr->size == 10);

  // Verify data unchanged
  for (size_t i = 0; i < arr->size; i++) {
    assert(((NUMC_INT *)arr->data)[i] == (NUMC_INT)i);
  }

  array_free(arr);
  printf("✓ test_flatten_1d_array\n");
}

void test_flatten_2d_array(void) {
  // Create 2D array [3, 4]
  size_t shape[] = {3, 4};
  Array *arr = array_zeros(2, shape, NUMC_TYPE_INT);

  // Fill with sequential values: 0, 1, 2, ..., 11
  for (size_t i = 0; i < arr->size; i++) {
    ((NUMC_INT *)arr->data)[i] = i;
  }

  // Flatten
  int result = array_flatten(arr);
  assert(result == 0);
  assert(arr->ndim == 1);
  assert(arr->shape[0] == 12);
  assert(arr->size == 12);

  // Verify data order preserved (row-major)
  for (size_t i = 0; i < arr->size; i++) {
    assert(((NUMC_INT *)arr->data)[i] == (NUMC_INT)i);
  }

  array_free(arr);
  printf("✓ test_flatten_2d_array\n");
}

void test_flatten_3d_array(void) {
  // Create 3D array [2, 3, 4]
  size_t shape[] = {2, 3, 4};
  Array *arr = array_zeros(3, shape, NUMC_TYPE_FLOAT);

  // Fill with sequential values
  for (size_t i = 0; i < arr->size; i++) {
    ((NUMC_FLOAT *)arr->data)[i] = (float)i;
  }

  // Flatten
  int result = array_flatten(arr);
  assert(result == 0);
  assert(arr->ndim == 1);
  assert(arr->shape[0] == 24);  // 2 * 3 * 4
  assert(arr->size == 24);

  // Verify data order preserved
  for (size_t i = 0; i < arr->size; i++) {
    assert(((NUMC_FLOAT *)arr->data)[i] == (float)i);
  }

  array_free(arr);
  printf("✓ test_flatten_3d_array\n");
}

void test_flatten_non_contiguous(void) {
  // Create 2D array and slice it to make non-contiguous
  size_t shape[] = {4, 6};
  Array *arr = array_zeros(2, shape, NUMC_TYPE_INT);

  // Fill with data
  for (size_t i = 0; i < arr->size; i++) {
    ((NUMC_INT *)arr->data)[i] = i;
  }

  // Create slice (every other column)
  size_t start[] = {0, 0};
  size_t stop[] = {4, 6};
  size_t step[] = {1, 2};
  Array *slice = array_slice(arr, start, stop, step);
  assert(!array_is_contiguous(slice));

  // Copy to make it owned
  Array *slice_copy = array_copy(slice);
  assert(slice_copy->owns_data);
  assert(array_is_contiguous(slice_copy));

  // Flatten the copied slice
  int result = array_flatten(slice_copy);
  assert(result == 0);
  assert(slice_copy->ndim == 1);
  assert(slice_copy->shape[0] == 12);  // 4 * 3
  assert(slice_copy->size == 12);

  // Verify data is from columns 0, 2, 4
  NUMC_INT expected[] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22};
  for (size_t i = 0; i < slice_copy->size; i++) {
    assert(((NUMC_INT *)slice_copy->data)[i] == expected[i]);
  }

  array_free(arr);
  array_free(slice);
  array_free(slice_copy);
  printf("✓ test_flatten_non_contiguous\n");
}

void test_flatten_all_types(void) {
  NUMC_TYPE types[] = {NUMC_TYPE_BYTE,  NUMC_TYPE_UBYTE, NUMC_TYPE_SHORT,
                       NUMC_TYPE_USHORT, NUMC_TYPE_INT,   NUMC_TYPE_UINT,
                       NUMC_TYPE_LONG,   NUMC_TYPE_ULONG, NUMC_TYPE_FLOAT,
                       NUMC_TYPE_DOUBLE};

  for (size_t t = 0; t < sizeof(types) / sizeof(NUMC_TYPE); t++) {
    size_t shape[] = {2, 3};
    Array *arr = array_zeros(2, shape, types[t]);

    int result = array_flatten(arr);
    assert(result == 0);
    assert(arr->ndim == 1);
    assert(arr->shape[0] == 6);
    assert(arr->size == 6);
    assert(arr->numc_type == types[t]);

    array_free(arr);
  }

  printf("✓ test_flatten_all_types\n");
}

void test_flatten_large_array(void) {
  // Test with a larger multidimensional array
  size_t shape[] = {10, 20, 5};
  Array *arr = array_zeros(3, shape, NUMC_TYPE_DOUBLE);

  // Fill with data
  for (size_t i = 0; i < arr->size; i++) {
    ((NUMC_DOUBLE *)arr->data)[i] = (double)i * 0.5;
  }

  int result = array_flatten(arr);
  assert(result == 0);
  assert(arr->ndim == 1);
  assert(arr->shape[0] == 1000);  // 10 * 20 * 5
  assert(arr->size == 1000);

  // Spot check some values
  assert(((NUMC_DOUBLE *)arr->data)[0] == 0.0);
  assert(((NUMC_DOUBLE *)arr->data)[100] == 50.0);
  assert(((NUMC_DOUBLE *)arr->data)[999] == 499.5);

  array_free(arr);
  printf("✓ test_flatten_large_array\n");
}

void test_flatten_preserves_data(void) {
  // Create array with specific pattern
  size_t shape[] = {3, 3};
  int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  ArrayCreate create = {
      .ndim = 2,
      .shape = shape,
      .numc_type = NUMC_TYPE_INT,
      .data = data,
      .owns_data = true,
  };
  Array *arr = array_create(&create);

  // Flatten
  int result = array_flatten(arr);
  assert(result == 0);
  assert(arr->ndim == 1);
  assert(arr->shape[0] == 9);

  // Verify data preserved in row-major order
  for (size_t i = 0; i < arr->size; i++) {
    assert(((NUMC_INT *)arr->data)[i] == (NUMC_INT)(i + 1));
  }

  array_free(arr);
  printf("✓ test_flatten_preserves_data\n");
}

void test_flatten_idempotent(void) {
  // Flattening an already flat array should work
  size_t shape[] = {2, 3, 4};
  Array *arr = array_zeros(3, shape, NUMC_TYPE_INT);

  // First flatten
  int result1 = array_flatten(arr);
  assert(result1 == 0);
  assert(arr->ndim == 1);
  assert(arr->shape[0] == 24);

  // Second flatten (should be no-op)
  int result2 = array_flatten(arr);
  assert(result2 == 0);
  assert(arr->ndim == 1);
  assert(arr->shape[0] == 24);

  array_free(arr);
  printf("✓ test_flatten_idempotent\n");
}

int main(void) {
  test_flatten_1d_array();
  test_flatten_2d_array();
  test_flatten_3d_array();
  test_flatten_non_contiguous();
  test_flatten_all_types();
  test_flatten_large_array();
  test_flatten_preserves_data();
  test_flatten_idempotent();

  printf("\n✓ All flatten tests passed\n");
  return 0;
}
