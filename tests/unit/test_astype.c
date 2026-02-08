/**
 * @file test_astype.c
 * @brief Test array_astype (type conversion) functionality
 */

#include <numc/numc.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void test_astype_int_to_float(void) {
  size_t shape[] = {5};
  int data[] = {10, 20, 30, 40, 50};

  ArrayCreate create = {
      .ndim = 1,
      .shape = shape,
      .numc_type = NUMC_TYPE_INT,
      .data = data,
      .owns_data = true,
  };
  Array *arr = array_create(&create);

  // Convert to float
  int result = array_astype(arr, NUMC_TYPE_FLOAT);
  assert(result == 0);
  assert(arr->numc_type == NUMC_TYPE_FLOAT);
  assert(arr->elem_size == sizeof(NUMC_FLOAT));
  assert(arr->size == 5);

  // Verify values
  NUMC_FLOAT *fdata = (NUMC_FLOAT *)arr->data;
  assert(fabsf(fdata[0] - 10.0f) < 1e-6f);
  assert(fabsf(fdata[1] - 20.0f) < 1e-6f);
  assert(fabsf(fdata[2] - 30.0f) < 1e-6f);
  assert(fabsf(fdata[3] - 40.0f) < 1e-6f);
  assert(fabsf(fdata[4] - 50.0f) < 1e-6f);

  array_free(arr);
  printf("✓ test_astype_int_to_float\n");
}

void test_astype_float_to_int(void) {
  size_t shape[] = {4};
  float data[] = {1.7f, 2.3f, 3.9f, 4.1f};

  ArrayCreate create = {
      .ndim = 1,
      .shape = shape,
      .numc_type = NUMC_TYPE_FLOAT,
      .data = data,
      .owns_data = true,
  };
  Array *arr = array_create(&create);

  // Convert to int (truncation)
  int result = array_astype(arr, NUMC_TYPE_INT);
  assert(result == 0);
  assert(arr->numc_type == NUMC_TYPE_INT);
  assert(arr->elem_size == sizeof(NUMC_INT));
  assert(arr->size == 4);

  // Verify values (truncated)
  NUMC_INT *idata = (NUMC_INT *)arr->data;
  assert(idata[0] == 1);
  assert(idata[1] == 2);
  assert(idata[2] == 3);
  assert(idata[3] == 4);

  array_free(arr);
  printf("✓ test_astype_float_to_int\n");
}

void test_astype_double_to_short(void) {
  size_t shape[] = {6};
  double data[] = {100.5, -50.2, 32767.0, -32768.0, 0.0, 1234.9};

  ArrayCreate create = {
      .ndim = 1,
      .shape = shape,
      .numc_type = NUMC_TYPE_DOUBLE,
      .data = data,
      .owns_data = true,
  };
  Array *arr = array_create(&create);

  // Convert to short
  int result = array_astype(arr, NUMC_TYPE_SHORT);
  assert(result == 0);
  assert(arr->numc_type == NUMC_TYPE_SHORT);
  assert(arr->elem_size == sizeof(NUMC_SHORT));
  assert(arr->size == 6);

  // Verify values
  NUMC_SHORT *sdata = (NUMC_SHORT *)arr->data;
  assert(sdata[0] == 100);
  assert(sdata[1] == -50);
  assert(sdata[2] == 32767);
  assert(sdata[3] == -32768);
  assert(sdata[4] == 0);
  assert(sdata[5] == 1234);

  array_free(arr);
  printf("✓ test_astype_double_to_short\n");
}

void test_astype_byte_to_long(void) {
  size_t shape[] = {5};
  int8_t data[] = {1, -2, 127, -128, 0};

  ArrayCreate create = {
      .ndim = 1,
      .shape = shape,
      .numc_type = NUMC_TYPE_BYTE,
      .data = data,
      .owns_data = true,
  };
  Array *arr = array_create(&create);

  // Convert to long (widening)
  int result = array_astype(arr, NUMC_TYPE_LONG);
  assert(result == 0);
  assert(arr->numc_type == NUMC_TYPE_LONG);
  assert(arr->elem_size == sizeof(NUMC_LONG));
  assert(arr->size == 5);

  // Verify values preserved
  NUMC_LONG *ldata = (NUMC_LONG *)arr->data;
  assert(ldata[0] == 1);
  assert(ldata[1] == -2);
  assert(ldata[2] == 127);
  assert(ldata[3] == -128);
  assert(ldata[4] == 0);

  array_free(arr);
  printf("✓ test_astype_byte_to_long\n");
}

void test_astype_signed_to_unsigned(void) {
  size_t shape[] = {5};
  int32_t data[] = {-5, -1, 0, 1, 5};

  ArrayCreate create = {
      .ndim = 1,
      .shape = shape,
      .numc_type = NUMC_TYPE_INT,
      .data = data,
      .owns_data = true,
  };
  Array *arr = array_create(&create);

  // Convert to unsigned (two's complement wraparound)
  int result = array_astype(arr, NUMC_TYPE_UINT);
  assert(result == 0);
  assert(arr->numc_type == NUMC_TYPE_UINT);
  assert(arr->elem_size == sizeof(NUMC_UINT));
  assert(arr->size == 5);

  // Verify values (negative values wrap)
  NUMC_UINT *udata = (NUMC_UINT *)arr->data;
  assert(udata[0] == (NUMC_UINT)-5);  // Wraps to large positive
  assert(udata[1] == (NUMC_UINT)-1);  // Wraps to max uint
  assert(udata[2] == 0);
  assert(udata[3] == 1);
  assert(udata[4] == 5);

  array_free(arr);
  printf("✓ test_astype_signed_to_unsigned\n");
}

void test_astype_same_type(void) {
  size_t shape[] = {4};
  Array *arr = array_zeros(1, shape, NUMC_TYPE_INT);

  // Fill with data
  for (size_t i = 0; i < arr->size; i++) {
    ((NUMC_INT *)arr->data)[i] = i;
  }

  // Convert to same type (should be no-op but still succeed)
  int result = array_astype(arr, NUMC_TYPE_INT);
  assert(result == 0);
  assert(arr->numc_type == NUMC_TYPE_INT);
  assert(arr->size == 4);

  // Verify data unchanged
  for (size_t i = 0; i < arr->size; i++) {
    assert(((NUMC_INT *)arr->data)[i] == (NUMC_INT)i);
  }

  array_free(arr);
  printf("✓ test_astype_same_type\n");
}

void test_astype_2d_array(void) {
  size_t shape[] = {2, 3};
  float data[] = {1.5f, 2.7f, 3.2f, 4.9f, 5.1f, 6.8f};

  ArrayCreate create = {
      .ndim = 2,
      .shape = shape,
      .numc_type = NUMC_TYPE_FLOAT,
      .data = data,
      .owns_data = true,
  };
  Array *arr = array_create(&create);

  // Convert to int
  int result = array_astype(arr, NUMC_TYPE_INT);
  assert(result == 0);
  assert(arr->numc_type == NUMC_TYPE_INT);
  assert(arr->ndim == 2);
  assert(arr->shape[0] == 2);
  assert(arr->shape[1] == 3);
  assert(arr->size == 6);

  // Verify values (truncated)
  NUMC_INT expected[] = {1, 2, 3, 4, 5, 6};
  NUMC_INT *idata = (NUMC_INT *)arr->data;
  for (size_t i = 0; i < arr->size; i++) {
    assert(idata[i] == expected[i]);
  }

  array_free(arr);
  printf("✓ test_astype_2d_array\n");
}

void test_astype_all_type_pairs(void) {
  // Test key conversion paths
  NUMC_TYPE types[] = {NUMC_TYPE_BYTE,  NUMC_TYPE_UBYTE, NUMC_TYPE_SHORT,
                       NUMC_TYPE_USHORT, NUMC_TYPE_INT,   NUMC_TYPE_UINT,
                       NUMC_TYPE_LONG,   NUMC_TYPE_ULONG, NUMC_TYPE_FLOAT,
                       NUMC_TYPE_DOUBLE};

  size_t shape[] = {3};

  for (size_t src = 0; src < sizeof(types) / sizeof(NUMC_TYPE); src++) {
    for (size_t dst = 0; dst < sizeof(types) / sizeof(NUMC_TYPE); dst++) {
      Array *arr = array_ones(1, shape, types[src]);

      int result = array_astype(arr, types[dst]);
      assert(result == 0);
      assert(arr->numc_type == types[dst]);
      assert(arr->size == 3);
      assert(arr->elem_size == numc_type_size(types[dst]));

      array_free(arr);
    }
  }

  printf("✓ test_astype_all_type_pairs (100 combinations)\n");
}

void test_astype_rejects_view(void) {
  size_t shape[] = {6};
  Array *arr = array_zeros(1, shape, NUMC_TYPE_INT);

  // Create a view (slice)
  size_t start[] = {0};
  size_t stop[] = {6};
  size_t step[] = {2};
  Array *view = array_slice(arr, start, stop, step);
  assert(!view->owns_data);

  // Should reject conversion of view
  int result = array_astype(view, NUMC_TYPE_FLOAT);
  assert(result < 0);  // Should fail

  array_free(arr);
  array_free(view);
  printf("✓ test_astype_rejects_view\n");
}

void test_astype_handles_non_contiguous(void) {
  // Create non-contiguous array by slicing
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
  Array *owned = array_copy(slice);
  assert(owned->owns_data);
  assert(array_is_contiguous(owned));

  // Now convert type
  int result = array_astype(owned, NUMC_TYPE_FLOAT);
  assert(result == 0);
  assert(owned->numc_type == NUMC_TYPE_FLOAT);
  assert(owned->size == 12);  // 4 * 3

  array_free(arr);
  array_free(slice);
  array_free(owned);
  printf("✓ test_astype_handles_non_contiguous\n");
}

void test_astype_preserves_shape(void) {
  // Test that shape and strides are correctly updated
  size_t shape[] = {2, 3, 4};
  Array *arr = array_zeros(3, shape, NUMC_TYPE_BYTE);

  size_t old_strides[3];
  for (size_t i = 0; i < 3; i++) {
    old_strides[i] = arr->strides[i];
  }

  // Convert to double (8 bytes vs 1 byte)
  int result = array_astype(arr, NUMC_TYPE_DOUBLE);
  assert(result == 0);
  assert(arr->numc_type == NUMC_TYPE_DOUBLE);
  assert(arr->elem_size == sizeof(NUMC_DOUBLE));

  // Shape unchanged
  assert(arr->ndim == 3);
  assert(arr->shape[0] == 2);
  assert(arr->shape[1] == 3);
  assert(arr->shape[2] == 4);
  assert(arr->size == 24);

  // Strides scaled by elem_size ratio (8 / 1 = 8x)
  assert(arr->strides[2] == sizeof(NUMC_DOUBLE));
  assert(arr->strides[1] == sizeof(NUMC_DOUBLE) * 4);
  assert(arr->strides[0] == sizeof(NUMC_DOUBLE) * 4 * 3);

  array_free(arr);
  printf("✓ test_astype_preserves_shape\n");
}

void test_astype_large_array(void) {
  // Test with larger array to exercise parallelization
  size_t shape[] = {1000};
  Array *arr = array_zeros(1, shape, NUMC_TYPE_INT);

  // Fill with sequential values
  for (size_t i = 0; i < arr->size; i++) {
    ((NUMC_INT *)arr->data)[i] = i;
  }

  // Convert to double
  int result = array_astype(arr, NUMC_TYPE_DOUBLE);
  assert(result == 0);
  assert(arr->numc_type == NUMC_TYPE_DOUBLE);
  assert(arr->size == 1000);

  // Spot check values
  NUMC_DOUBLE *ddata = (NUMC_DOUBLE *)arr->data;
  assert(fabs(ddata[0] - 0.0) < 1e-10);
  assert(fabs(ddata[500] - 500.0) < 1e-10);
  assert(fabs(ddata[999] - 999.0) < 1e-10);

  array_free(arr);
  printf("✓ test_astype_large_array\n");
}

void test_astype_chained_conversions(void) {
  // Test multiple conversions in sequence
  size_t shape[] = {4};
  int data[] = {10, 20, 30, 40};

  ArrayCreate create = {
      .ndim = 1,
      .shape = shape,
      .numc_type = NUMC_TYPE_INT,
      .data = data,
      .owns_data = true,
  };
  Array *arr = array_create(&create);

  // INT -> FLOAT
  int result1 = array_astype(arr, NUMC_TYPE_FLOAT);
  assert(result1 == 0);
  assert(arr->numc_type == NUMC_TYPE_FLOAT);

  // FLOAT -> DOUBLE
  int result2 = array_astype(arr, NUMC_TYPE_DOUBLE);
  assert(result2 == 0);
  assert(arr->numc_type == NUMC_TYPE_DOUBLE);

  // DOUBLE -> SHORT
  int result3 = array_astype(arr, NUMC_TYPE_SHORT);
  assert(result3 == 0);
  assert(arr->numc_type == NUMC_TYPE_SHORT);

  // Verify final values
  NUMC_SHORT *sdata = (NUMC_SHORT *)arr->data;
  assert(sdata[0] == 10);
  assert(sdata[1] == 20);
  assert(sdata[2] == 30);
  assert(sdata[3] == 40);

  array_free(arr);
  printf("✓ test_astype_chained_conversions\n");
}

void test_astype_capacity_updated(void) {
  // Verify capacity is correctly updated
  size_t shape[] = {10};
  Array *arr = array_zeros(1, shape, NUMC_TYPE_BYTE);  // 1 byte each

  size_t old_capacity = arr->capacity;
  assert(old_capacity == 10);

  // Convert to LONG (8 bytes each)
  int result = array_astype(arr, NUMC_TYPE_LONG);
  assert(result == 0);

  // Capacity should equal size (in elements)
  assert(arr->capacity == arr->size);
  assert(arr->capacity == 10);

  array_free(arr);
  printf("✓ test_astype_capacity_updated\n");
}

int main(void) {
  test_astype_int_to_float();
  test_astype_float_to_int();
  test_astype_double_to_short();
  test_astype_byte_to_long();
  test_astype_signed_to_unsigned();
  test_astype_same_type();
  test_astype_2d_array();
  test_astype_all_type_pairs();
  test_astype_rejects_view();
  test_astype_handles_non_contiguous();
  test_astype_preserves_shape();
  test_astype_large_array();
  test_astype_chained_conversions();
  test_astype_capacity_updated();

  printf("\n✓ All astype tests passed\n");
  return 0;
}
