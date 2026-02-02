/**
 * @file test_add.c
 * @brief Test array element-wise mathematical operations (add, sub, mul, div)
 */

#include "array.h"
#include "types.h"
#include <assert.h>
#include <math.h>
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

  // Convert to contiguous before adding
  Array *a_cont = array_to_contiguous(a_slice);
  Array *b_cont = array_to_contiguous(b_slice);

  // Add slices
  Array *c = array_add(a_cont, b_cont);
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
  array_free(a_cont);
  array_free(b_cont);
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

// =============================================================================
//                          Subtraction Tests
// =============================================================================

void test_sub_contiguous(void) {
  size_t shape[] = {3, 4};
  Array *a = array_zeros(2, shape, DTYPE_INT);
  Array *b = array_zeros(2, shape, DTYPE_INT);

  // Fill arrays
  for (size_t i = 0; i < a->size; i++) {
    ((NUMC_INT *)a->data)[i] = i * 3;
    ((NUMC_INT *)b->data)[i] = i;
  }

  // Subtract arrays
  Array *c = array_sub(a, b);
  assert(c != NULL);
  assert(c->ndim == 2);
  assert(c->shape[0] == 3);
  assert(c->shape[1] == 4);

  // Check results
  for (size_t i = 0; i < c->size; i++) {
    NUMC_INT expected = (i * 3) - i;
    NUMC_INT actual = ((NUMC_INT *)c->data)[i];
    assert(actual == expected);
  }

  array_free(a);
  array_free(b);
  array_free(c);
  printf("✓ test_sub_contiguous\n");
}

void test_sub_float(void) {
  size_t shape[] = {5};
  Array *a = array_zeros(1, shape, DTYPE_FLOAT);
  Array *b = array_zeros(1, shape, DTYPE_FLOAT);

  // Fill arrays
  for (size_t i = 0; i < a->size; i++) {
    ((NUMC_FLOAT *)a->data)[i] = (float)i * 2.5f;
    ((NUMC_FLOAT *)b->data)[i] = (float)i * 0.5f;
  }

  // Subtract arrays
  Array *c = array_sub(a, b);
  assert(c != NULL);

  // Check results
  for (size_t i = 0; i < c->size; i++) {
    float expected = ((float)i * 2.5f) - ((float)i * 0.5f);
    float actual = ((NUMC_FLOAT *)c->data)[i];
    assert(fabsf(actual - expected) < 1e-6f);
  }

  array_free(a);
  array_free(b);
  array_free(c);
  printf("✓ test_sub_float\n");
}

// =============================================================================
//                          Multiplication Tests
// =============================================================================

void test_mul_contiguous(void) {
  size_t shape[] = {3, 3};
  Array *a = array_zeros(2, shape, DTYPE_INT);
  Array *b = array_zeros(2, shape, DTYPE_INT);

  // Fill arrays
  for (size_t i = 0; i < a->size; i++) {
    ((NUMC_INT *)a->data)[i] = i + 1;
    ((NUMC_INT *)b->data)[i] = (i % 3) + 1;
  }

  // Multiply arrays
  Array *c = array_mul(a, b);
  assert(c != NULL);
  assert(c->ndim == 2);
  assert(c->shape[0] == 3);
  assert(c->shape[1] == 3);

  // Check results
  for (size_t i = 0; i < c->size; i++) {
    NUMC_INT expected = (i + 1) * ((i % 3) + 1);
    NUMC_INT actual = ((NUMC_INT *)c->data)[i];
    assert(actual == expected);
  }

  array_free(a);
  array_free(b);
  array_free(c);
  printf("✓ test_mul_contiguous\n");
}

void test_mul_double(void) {
  size_t shape[] = {4};
  Array *a = array_zeros(1, shape, DTYPE_DOUBLE);
  Array *b = array_zeros(1, shape, DTYPE_DOUBLE);

  // Fill arrays
  for (size_t i = 0; i < a->size; i++) {
    ((NUMC_DOUBLE *)a->data)[i] = (double)i * 1.5;
    ((NUMC_DOUBLE *)b->data)[i] = (double)i * 2.0;
  }

  // Multiply arrays
  Array *c = array_mul(a, b);
  assert(c != NULL);

  // Check results
  for (size_t i = 0; i < c->size; i++) {
    double expected = ((double)i * 1.5) * ((double)i * 2.0);
    double actual = ((NUMC_DOUBLE *)c->data)[i];
    assert(fabs(actual - expected) < 1e-10);
  }

  array_free(a);
  array_free(b);
  array_free(c);
  printf("✓ test_mul_double\n");
}

// =============================================================================
//                          Division Tests
// =============================================================================

void test_div_contiguous(void) {
  size_t shape[] = {4, 2};
  Array *a = array_zeros(2, shape, DTYPE_INT);
  Array *b = array_zeros(2, shape, DTYPE_INT);

  // Fill arrays (avoid division by zero)
  for (size_t i = 0; i < a->size; i++) {
    ((NUMC_INT *)a->data)[i] = (i + 1) * 10;
    ((NUMC_INT *)b->data)[i] = (i + 1) * 2;
  }

  // Divide arrays
  Array *c = array_div(a, b);
  assert(c != NULL);
  assert(c->ndim == 2);
  assert(c->shape[0] == 4);
  assert(c->shape[1] == 2);

  // Check results
  for (size_t i = 0; i < c->size; i++) {
    NUMC_INT expected = ((i + 1) * 10) / ((i + 1) * 2);
    NUMC_INT actual = ((NUMC_INT *)c->data)[i];
    assert(actual == expected);
  }

  array_free(a);
  array_free(b);
  array_free(c);
  printf("✓ test_div_contiguous\n");
}

void test_div_float(void) {
  size_t shape[] = {6};
  Array *a = array_zeros(1, shape, DTYPE_FLOAT);
  Array *b = array_zeros(1, shape, DTYPE_FLOAT);

  // Fill arrays
  for (size_t i = 0; i < a->size; i++) {
    ((NUMC_FLOAT *)a->data)[i] = (float)(i + 1) * 10.0f;
    ((NUMC_FLOAT *)b->data)[i] = (float)(i + 1) * 2.0f;
  }

  // Divide arrays
  Array *c = array_div(a, b);
  assert(c != NULL);

  // Check results
  for (size_t i = 0; i < c->size; i++) {
    float expected = ((float)(i + 1) * 10.0f) / ((float)(i + 1) * 2.0f);
    float actual = ((NUMC_FLOAT *)c->data)[i];
    assert(fabsf(actual - expected) < 1e-6f);
  }

  array_free(a);
  array_free(b);
  array_free(c);
  printf("✓ test_div_float\n");
}

// =============================================================================
//                          Combined Operations Tests
// =============================================================================

void test_all_ops_with_dtypes(void) {
  size_t shape[] = {8};

  DType types[] = {DTYPE_BYTE,  DTYPE_UBYTE, DTYPE_SHORT, DTYPE_USHORT,
                   DTYPE_INT,   DTYPE_UINT,  DTYPE_LONG,  DTYPE_ULONG,
                   DTYPE_FLOAT, DTYPE_DOUBLE};

  for (size_t t = 0; t < sizeof(types) / sizeof(DType); t++) {
    Array *a = array_ones(1, shape, types[t]);
    Array *b = array_ones(1, shape, types[t]);

    // Test add
    Array *result_add = array_add(a, b);
    assert(result_add != NULL);
    assert(result_add->dtype == types[t]);
    array_free(result_add);

    // Test sub
    Array *result_sub = array_sub(a, b);
    assert(result_sub != NULL);
    assert(result_sub->dtype == types[t]);
    array_free(result_sub);

    // Test mul
    Array *result_mul = array_mul(a, b);
    assert(result_mul != NULL);
    assert(result_mul->dtype == types[t]);
    array_free(result_mul);

    // Test div (use ones to avoid division by zero)
    Array *result_div = array_div(a, b);
    assert(result_div != NULL);
    assert(result_div->dtype == types[t]);
    array_free(result_div);

    array_free(a);
    array_free(b);
  }

  printf("✓ test_all_ops_with_dtypes\n");
}

int main(void) {
  // Addition tests
  test_add_contiguous();
  test_add_non_contiguous();
  test_add_mismatched_shapes();
  test_add_mismatched_types();
  test_add_all_dtypes();

  // Subtraction tests
  test_sub_contiguous();
  test_sub_float();

  // Multiplication tests
  test_mul_contiguous();
  test_mul_double();

  // Division tests
  test_div_contiguous();
  test_div_float();

  // Combined tests
  test_all_ops_with_dtypes();

  printf("\n✓ All mathematical operation tests passed\n");
  return 0;
}
