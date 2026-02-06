/**
 * @file test_array_core.c
 * @brief Test core array functions (offset, bounds_check, get, free)
 */

#include "alloc.h"
#include "array.h"
#include "types.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

void test_array_offset_1d(void) {
  printf("Testing array_offset with 1D array...\n");

  // Create 1D array [10]
  Array *arr = array_create(&(ArrayCreate){.ndim = 1,
                                           .shape = (size_t[]){10},
                                           .numc_type = NUMC_TYPE_INT,
                                           .data = NULL,
                                           .owns_data = true});

  // Check offsets for different indices
  assert(array_offset(arr, (size_t[]){0}) == 0);
  assert(array_offset(arr, (size_t[]){1}) == sizeof(int));
  assert(array_offset(arr, (size_t[]){5}) == 5 * sizeof(int));
  assert(array_offset(arr, (size_t[]){9}) == 9 * sizeof(int));

  array_free(arr);
  printf("  ✓ 1D offset calculation works\n");
}

void test_array_offset_2d(void) {
  printf("Testing array_offset with 2D array...\n");

  // Create 2D array [3, 4] (row-major)
  Array *arr = array_create(&(ArrayCreate){.ndim = 2,
                                           .shape = (size_t[]){3, 4},
                                           .numc_type = NUMC_TYPE_INT,
                                           .data = NULL,
                                           .owns_data = true});

  // stride[0] = 16 (4 elements * 4 bytes)
  // stride[1] = 4 (1 element * 4 bytes)

  // [0,0] -> offset = 0*16 + 0*4 = 0
  assert(array_offset(arr, (size_t[]){0, 0}) == 0);

  // [0,1] -> offset = 0*16 + 1*4 = 4
  assert(array_offset(arr, (size_t[]){0, 1}) == 4);

  // [1,0] -> offset = 1*16 + 0*4 = 16
  assert(array_offset(arr, (size_t[]){1, 0}) == 4 * sizeof(NUMC_INT));

  // [2,3] -> offset = 2*16 + 3*4 = 32 + 12 = 44
  assert(array_offset(arr, (size_t[]){2, 3}) == 44);

  array_free(arr);
  printf("  ✓ 2D offset calculation works\n");
}

void test_array_offset_3d(void) {
  printf("Testing array_offset with 3D array...\n");

  // Create 3D array [2, 3, 4]
  Array *arr = array_create(&(ArrayCreate){.ndim = 3,
                                           .shape = (size_t[]){2, 3, 4},
                                           .numc_type = NUMC_TYPE_INT,
                                           .data = NULL,
                                           .owns_data = true});

  // stride[0] = 48 (3*4 elements * 4 bytes)
  // stride[1] = 16 (4 elements * 4 bytes)
  // stride[2] = 4 (1 element * 4 bytes)

  // [0,0,0] -> 0
  assert(array_offset(arr, (size_t[]){0, 0, 0}) == 0);

  // [1,0,0] -> 1*48 = 48
  assert(array_offset(arr, (size_t[]){1, 0, 0}) == 48);

  // [0,1,0] -> 0*48 + 1*16 = 16
  assert(array_offset(arr, (size_t[]){0, 1, 0}) == 4 * sizeof(NUMC_INT));

  // [0,0,1] -> 0*48 + 0*16 + 1*4 = 4
  assert(array_offset(arr, (size_t[]){0, 0, 1}) == 4);

  // [1,2,3] -> 1*48 + 2*16 + 3*4 = 48 + 32 + 12 = 92
  assert(array_offset(arr, (size_t[]){1, 2, 3}) == 92);

  array_free(arr);
  printf("  ✓ 3D offset calculation works\n");
}

void test_array_offset_null(void) {
  printf("Testing array_offset with NULL inputs...\n");

  Array *arr = array_create(&(ArrayCreate){.ndim = 1,
                                           .shape = (size_t[]){10},
                                           .numc_type = NUMC_TYPE_INT,
                                           .data = NULL,
                                           .owns_data = true});

  // NULL array should return 0
  assert(array_offset(NULL, (size_t[]){0}) == 0);

  // NULL indices should return 0
  assert(array_offset(arr, NULL) == 0);

  array_free(arr);
  printf("  ✓ NULL input handling works\n");
}

void test_array_bounds_check_valid(void) {
  printf("Testing array_bounds_check with valid indices...\n");

  // Create 2D array [3, 4]
  Array *arr = array_create(&(ArrayCreate){.ndim = 2,
                                           .shape = (size_t[]){3, 4},
                                           .numc_type = NUMC_TYPE_INT,
                                           .data = NULL,
                                           .owns_data = true});

  // Valid indices should return 0
  assert(array_bounds_check(arr, (size_t[]){0, 0}) == 0);
  assert(array_bounds_check(arr, (size_t[]){0, 3}) == 0);
  assert(array_bounds_check(arr, (size_t[]){2, 0}) == 0);
  assert(array_bounds_check(arr, (size_t[]){2, 3}) == 0);
  assert(array_bounds_check(arr, (size_t[]){1, 2}) == 0);

  array_free(arr);
  printf("  ✓ Valid indices accepted\n");
}

void test_array_bounds_check_invalid(void) {
  printf("Testing array_bounds_check with invalid indices...\n");

  // Create 2D array [3, 4]
  Array *arr = array_create(&(ArrayCreate){.ndim = 2,
                                           .shape = (size_t[]){3, 4},
                                           .numc_type = NUMC_TYPE_INT,
                                           .data = NULL,
                                           .owns_data = true});

  // Out of bounds indices should return -1
  assert(array_bounds_check(arr, (size_t[]){3, 0}) == -1); // row out of bounds
  assert(array_bounds_check(arr, (size_t[]){0, 4}) == -1); // col out of bounds
  assert(array_bounds_check(arr, (size_t[]){3, 4}) == -1); // both out of bounds
  assert(array_bounds_check(arr, (size_t[]){10, 10}) ==
         -1); // far out of bounds

  array_free(arr);
  printf("  ✓ Invalid indices rejected\n");
}

void test_array_bounds_check_null(void) {
  printf("Testing array_bounds_check with NULL inputs...\n");

  Array *arr = array_create(&(ArrayCreate){.ndim = 1,
                                           .shape = (size_t[]){10},
                                           .numc_type = NUMC_TYPE_INT,
                                           .data = NULL,
                                           .owns_data = true});

  // NULL array should return -1
  assert(array_bounds_check(NULL, (size_t[]){0}) == -1);

  // NULL indices should return -1
  assert(array_bounds_check(arr, NULL) == -1);

  array_free(arr);
  printf("  ✓ NULL input handling works\n");
}

void test_array_get_basic(void) {
  printf("Testing array_get with different data types...\n");

  // Test with int array
  Array *arr_int = array_create(&(ArrayCreate){.ndim = 1,
                                               .shape = (size_t[]){5},
                                               .numc_type = NUMC_TYPE_INT,
                                               .data = NULL,
                                               .owns_data = true});
  int *data_int = (int *)arr_int->data;
  for (int i = 0; i < 5; i++) {
    data_int[i] = i * 10;
  }

  // Get elements
  assert(*(int *)array_get(arr_int, (size_t[]){0}) == 0);
  assert(*(int *)array_get(arr_int, (size_t[]){2}) == 20);
  assert(*(int *)array_get(arr_int, (size_t[]){4}) == 40);

  array_free(arr_int);

  // Test with float array
  Array *arr_float = array_create(&(ArrayCreate){.ndim = 1,
                                                 .shape = (size_t[]){3},
                                                 .numc_type = NUMC_TYPE_FLOAT,
                                                 .data = NULL,
                                                 .owns_data = true});
  float *data_float = (float *)arr_float->data;
  data_float[0] = 1.5f;
  data_float[1] = 2.5f;
  data_float[2] = 3.5f;

  assert(*(float *)array_get(arr_float, (size_t[]){0}) == 1.5f);
  assert(*(float *)array_get(arr_float, (size_t[]){1}) == 2.5f);
  assert(*(float *)array_get(arr_float, (size_t[]){2}) == 3.5f);

  array_free(arr_float);
  printf("  ✓ array_get works with different types\n");
}

void test_array_get_2d(void) {
  printf("Testing array_get with 2D array...\n");

  // Create 2D array [3, 4]
  Array *arr = array_create(&(ArrayCreate){.ndim = 2,
                                           .shape = (size_t[]){3, 4},
                                           .numc_type = NUMC_TYPE_INT,
                                           .data = NULL,
                                           .owns_data = true});
  int *data = (int *)arr->data;

  // Fill: [[0,1,2,3], [4,5,6,7], [8,9,10,11]]
  for (int i = 0; i < 12; i++) {
    data[i] = i;
  }

  // Test element access
  assert(*(int *)array_get(arr, (size_t[]){0, 0}) == 0);
  assert(*(int *)array_get(arr, (size_t[]){0, 3}) == 3);
  assert(*(int *)array_get(arr, (size_t[]){1, 0}) == 4);
  assert(*(int *)array_get(arr, (size_t[]){1, 2}) == 6);
  assert(*(int *)array_get(arr, (size_t[]){2, 3}) == 11);

  array_free(arr);
  printf("  ✓ 2D array_get works\n");
}

void test_array_get_null(void) {
  printf("Testing array_get with NULL inputs...\n");

  Array *arr = array_create(&(ArrayCreate){.ndim = 1,
                                           .shape = (size_t[]){10},
                                           .numc_type = NUMC_TYPE_INT,
                                           .data = NULL,
                                           .owns_data = true});

  // NULL array should return NULL
  assert(array_get(NULL, (size_t[]){0}) == NULL);

  // NULL indices should return NULL
  assert(array_get(arr, NULL) == NULL);

  array_free(arr);
  printf("  ✓ NULL input handling works\n");
}

void test_array_free_basic(void) {
  printf("Testing array_free basic functionality...\n");

  // Create and free array
  Array *arr = array_create(&(ArrayCreate){.ndim = 2,
                                           .shape = (size_t[]){10, 10},
                                           .numc_type = NUMC_TYPE_INT,
                                           .data = NULL,
                                           .owns_data = true});
  assert(arr != NULL);

  // Free should not crash
  array_free(arr);

  // Freeing NULL should not crash
  array_free(NULL);

  printf("  ✓ array_free works\n");
}

void test_array_free_ownership(void) {
  printf("Testing array_free with ownership flags...\n");

  // Create array that owns data
  Array *arr1 = array_create(&(ArrayCreate){.ndim = 1,
                                            .shape = (size_t[]){5},
                                            .numc_type = NUMC_TYPE_INT,
                                            .data = NULL,
                                            .owns_data = true});
  assert(arr1->owns_data == 1);
  array_free(arr1);

  // Create slice (doesn't own data)
  Array *base = array_create(&(ArrayCreate){.ndim = 1,
                                            .shape = (size_t[]){10},
                                            .numc_type = NUMC_TYPE_INT,
                                            .data = NULL,
                                            .owns_data = true});
  Array *slice = array_slice(base, (size_t[]){0}, (size_t[]){5}, (size_t[]){1});
  assert(slice->owns_data == 0);

  void *base_data = base->data;

  // Free slice (should not free data)
  array_free(slice);

  // Base data should still be valid
  assert(base->data == base_data);

  array_free(base);
  printf("  ✓ Ownership handling works\n");
}

void test_type_safe_accessors(void) {
  printf("Testing type-safe accessor functions...\n");

  // Test array_geti
  Array *arr_int = array_create(&(ArrayCreate){.ndim = 1,
                                               .shape = (size_t[]){3},
                                               .numc_type = NUMC_TYPE_INT,
                                               .data = NULL,
                                               .owns_data = true});
  int *data_int = (int *)arr_int->data;
  data_int[0] = 10;
  data_int[1] = 20;
  data_int[2] = 30;
  assert(*array_get_int32(arr_int, (size_t[]){0}) == 10);
  assert(*array_get_int32(arr_int, (size_t[]){2}) == 30);
  array_free(arr_int);

  // Test array_getf
  Array *arr_float = array_create(&(ArrayCreate){.ndim = 1,
                                                 .shape = (size_t[]){2},
                                                 .numc_type = NUMC_TYPE_FLOAT,
                                                 .data = NULL,
                                                 .owns_data = true});
  float *data_float = (float *)arr_float->data;
  data_float[0] = 1.5f;
  data_float[1] = 2.5f;
  assert(*array_get_float32(arr_float, (size_t[]){0}) == 1.5f);
  assert(*array_get_float32(arr_float, (size_t[]){1}) == 2.5f);
  array_free(arr_float);

  // Test array_getui
  Array *arr_uint = array_create(&(ArrayCreate){.ndim = 1,
                                                .shape = (size_t[]){2},
                                                .numc_type = NUMC_TYPE_UINT,
                                                .data = NULL,
                                                .owns_data = true});
  unsigned int *data_uint = (unsigned int *)arr_uint->data;
  data_uint[0] = 100;
  data_uint[1] = 200;
  assert(*array_get_uint32(arr_uint, (size_t[]){0}) == 100);
  assert(*array_get_uint32(arr_uint, (size_t[]){1}) == 200);
  array_free(arr_uint);

  // Test array_getl
  Array *arr_long = array_create(&(ArrayCreate){.ndim = 1,
                                                .shape = (size_t[]){2},
                                                .numc_type = NUMC_TYPE_LONG,
                                                .data = NULL,
                                                .owns_data = true});
  long *data_long = (long *)arr_long->data;
  data_long[0] = 1000L;
  data_long[1] = 2000L;
  assert(*array_get_int64(arr_long, (size_t[]){0}) == 1000L);
  assert(*array_get_int64(arr_long, (size_t[]){1}) == 2000L);
  array_free(arr_long);

  // Test array_getul
  Array *arr_ulong = array_create(&(ArrayCreate){.ndim = 1,
                                                 .shape = (size_t[]){2},
                                                 .numc_type = NUMC_TYPE_ULONG,
                                                 .data = NULL,
                                                 .owns_data = true});
  unsigned long *data_ulong = (unsigned long *)arr_ulong->data;
  data_ulong[0] = 10000UL;
  data_ulong[1] = 20000UL;
  assert(*array_get_uint64(arr_ulong, (size_t[]){0}) == 10000UL);
  assert(*array_get_uint64(arr_ulong, (size_t[]){1}) == 20000UL);
  array_free(arr_ulong);

  // Test array_gets
  Array *arr_short = array_create(&(ArrayCreate){.ndim = 1,
                                                 .shape = (size_t[]){2},
                                                 .numc_type = NUMC_TYPE_SHORT,
                                                 .data = NULL,
                                                 .owns_data = true});
  short *data_short = (short *)arr_short->data;
  data_short[0] = 100;
  data_short[1] = 200;
  assert(*array_get_int16(arr_short, (size_t[]){0}) == 100);
  assert(*array_get_int16(arr_short, (size_t[]){1}) == 200);
  array_free(arr_short);

  // Test array_getus
  Array *arr_ushort = array_create(&(ArrayCreate){.ndim = 1,
                                                  .shape = (size_t[]){2},
                                                  .numc_type = NUMC_TYPE_USHORT,
                                                  .data = NULL,
                                                  .owns_data = true});
  unsigned short *data_ushort = (unsigned short *)arr_ushort->data;
  data_ushort[0] = 300;
  data_ushort[1] = 400;
  assert(*array_get_uint16(arr_ushort, (size_t[]){0}) == 300);
  assert(*array_get_uint16(arr_ushort, (size_t[]){1}) == 400);
  array_free(arr_ushort);

  // Test array_getb
  Array *arr_byte = array_create(&(ArrayCreate){.ndim = 1,
                                                .shape = (size_t[]){2},
                                                .numc_type = NUMC_TYPE_BYTE,
                                                .data = NULL,
                                                .owns_data = true});
  signed char *data_byte = (signed char *)arr_byte->data;
  data_byte[0] = -10;
  data_byte[1] = 20;
  assert(*array_get_int8(arr_byte, (size_t[]){0}) == -10);
  assert(*array_get_int8(arr_byte, (size_t[]){1}) == 20);
  array_free(arr_byte);

  // Test array_getub
  Array *arr_ubyte = array_create(&(ArrayCreate){.ndim = 1,
                                                 .shape = (size_t[]){2},
                                                 .numc_type = NUMC_TYPE_UBYTE,
                                                 .data = NULL,
                                                 .owns_data = true});
  unsigned char *data_ubyte = (unsigned char *)arr_ubyte->data;
  data_ubyte[0] = 50;
  data_ubyte[1] = 100;
  assert(*array_get_uint8(arr_ubyte, (size_t[]){0}) == 50);
  assert(*array_get_uint8(arr_ubyte, (size_t[]){1}) == 100);
  array_free(arr_ubyte);

  printf("  ✓ All type-safe accessors work\n");
}

int main(void) {
  printf("=== Running Core Array Function Tests ===\n\n");

  // array_offset tests
  test_array_offset_1d();
  test_array_offset_2d();
  test_array_offset_3d();
  test_array_offset_null();

  // array_bounds_check tests
  test_array_bounds_check_valid();
  test_array_bounds_check_invalid();
  test_array_bounds_check_null();

  // array_get tests
  test_array_get_basic();
  test_array_get_2d();
  test_array_get_null();

  // array_free tests
  test_array_free_basic();
  test_array_free_ownership();

  // Type-safe accessor tests
  test_type_safe_accessors();

  printf("\n=== All Core Array Function Tests Passed ===\n");
  return 0;
}
