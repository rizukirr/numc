/**
 * @file test_array_add.c
 * @brief Test array element-wise addition
 */

#include "array.h"
#include "dtype.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

void test_add_contiguous(void) {
  size_t shape[] = {3, 4};
  Array *a = array_zeros(2, shape, DTYPE_INT);
  Array *b = array_zeros(2, shape, DTYPE_INT);

  // Fill arrays
  for (size_t i = 0; i < a->size; i++) {
    ((NUMC_INT *)a->data)[i] = i;
    ((NUMC_INT *)b->data)[i] = i * 2;
  }

  // Add arrays
  Array *c = array_add(a, b);
  assert(c != NULL);
  assert(c->ndim == 2);
  assert(c->shape[0] == 3);
  assert(c->shape[1] == 4);

  // Check results
  for (size_t i = 0; i < c->size; i++) {
    NUMC_INT expected = i + (i * 2);
    NUMC_INT actual = ((NUMC_INT *)c->data)[i];
    assert(actual == expected);
  }

  array_free(a);
  array_free(b);
  array_free(c);
  printf("✓ test_add_contiguous\n");
}

void test_add_non_contiguous(void) {
  size_t shape[] = {4, 6};
  Array *a = array_zeros(2, shape, DTYPE_FLOAT);
  Array *b = array_zeros(2, shape, DTYPE_FLOAT);

  // Fill arrays
  for (size_t i = 0; i < a->size; i++) {
    ((NUMC_FLOAT *)a->data)[i] = (float)i;
    ((NUMC_FLOAT *)b->data)[i] = (float)i * 2.0f;
  }

  // Create slices (non-contiguous)
  size_t start[] = {0, 0};
  size_t stop[] = {4, 6};
  size_t step[] = {1, 2}; // Every other column

  Array *a_slice = array_slice(a, start, stop, step);
  Array *b_slice = array_slice(b, start, stop, step);

  assert(!array_is_contiguous(a_slice));
  assert(!array_is_contiguous(b_slice));

  // Add slices
  Array *c = array_add(a_slice, b_slice);
  assert(c != NULL);
  assert(c->ndim == 2);
  assert(c->shape[0] == 4);
  assert(c->shape[1] == 3);

  // Check results
  size_t indices[] = {0, 0};
  for (size_t i = 0; i < c->size; i++) {
    size_t a_idx = indices[0] * 6 + indices[1] * 2;
    float expected = (float)a_idx + ((float)a_idx * 2.0f);
    float actual = ((NUMC_FLOAT *)c->data)[i];
    assert(actual == expected);

    // Increment indices
    indices[1]++;
    if (indices[1] >= 3) {
      indices[1] = 0;
      indices[0]++;
    }
  }

  array_free(a);
  array_free(b);
  array_free(a_slice);
  array_free(b_slice);
  array_free(c);
  printf("✓ test_add_non_contiguous\n");
}

void test_add_mismatched_shapes(void) {
  size_t shape1[] = {3, 4};
  size_t shape2[] = {3, 5};
  Array *a = array_zeros(2, shape1, DTYPE_INT);
  Array *b = array_zeros(2, shape2, DTYPE_INT);

  Array *c = array_add(a, b);
  assert(c == NULL); // Should fail

  array_free(a);
  array_free(b);
  printf("✓ test_add_mismatched_shapes\n");
}

void test_add_mismatched_types(void) {
  size_t shape[] = {3, 4};
  Array *a = array_zeros(2, shape, DTYPE_INT);
  Array *b = array_zeros(2, shape, DTYPE_FLOAT);

  Array *c = array_add(a, b);
  assert(c == NULL); // Should fail

  array_free(a);
  array_free(b);
  printf("✓ test_add_mismatched_types\n");
}

void test_add_all_dtypes(void) {
  size_t shape[] = {10};

  DType types[] = {DTYPE_BYTE,  DTYPE_UBYTE, DTYPE_SHORT, DTYPE_USHORT,
                   DTYPE_INT,   DTYPE_UINT,  DTYPE_LONG,  DTYPE_ULONG,
                   DTYPE_FLOAT, DTYPE_DOUBLE};

  for (size_t t = 0; t < sizeof(types) / sizeof(DType); t++) {
    Array *a = array_zeros(1, shape, types[t]);
    Array *b = array_zeros(1, shape, types[t]);

    Array *c = array_add(a, b);
    assert(c != NULL);
    assert(c->dtype == types[t]);

    array_free(a);
    array_free(b);
    array_free(c);
  }

  printf("✓ test_add_all_dtypes\n");
}

int main(void) {
  test_add_contiguous();
  test_add_non_contiguous();
  test_add_mismatched_shapes();
  test_add_mismatched_types();
  test_add_all_dtypes();

  printf("\n✓ All array_add tests passed\n");
  return 0;
}
