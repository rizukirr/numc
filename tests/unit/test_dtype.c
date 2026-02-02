/**
 * @file test_dtype.c
 * @brief Test DType system functionality
 */

#include "types.h"
#include <assert.h>
#include <stdio.h>

void test_dtype_size(void) {
  printf("Testing dtype_size...\n");
  assert(dtype_size(DTYPE_BYTE) == 1);
  assert(dtype_size(DTYPE_UBYTE) == 1);
  assert(dtype_size(DTYPE_SHORT) == 2);
  assert(dtype_size(DTYPE_USHORT) == 2);
  assert(dtype_size(DTYPE_INT) == 4);
  assert(dtype_size(DTYPE_UINT) == 4);
  assert(dtype_size(DTYPE_LONG) == 8);
  assert(dtype_size(DTYPE_ULONG) == 8);
  assert(dtype_size(DTYPE_FLOAT) == 4);
  assert(dtype_size(DTYPE_DOUBLE) == 8);
  printf("  ✓ All dtype sizes correct\n");
}

void test_dtype_is_float(void) {
  printf("Testing dtype_is_float...\n");
  assert(dtype_is_float(DTYPE_FLOAT) == 1);
  assert(dtype_is_float(DTYPE_DOUBLE) == 1);
  assert(dtype_is_float(DTYPE_INT) == 0);
  assert(dtype_is_float(DTYPE_BYTE) == 0);
  assert(dtype_is_float(DTYPE_LONG) == 0);
  printf("  ✓ Float type detection works\n");
}

void test_dtype_is_signed(void) {
  printf("Testing dtype_is_signed...\n");
  assert(dtype_is_signed(DTYPE_BYTE) == 1);
  assert(dtype_is_signed(DTYPE_SHORT) == 1);
  assert(dtype_is_signed(DTYPE_INT) == 1);
  assert(dtype_is_signed(DTYPE_LONG) == 1);
  assert(dtype_is_signed(DTYPE_UBYTE) == 0);
  assert(dtype_is_signed(DTYPE_USHORT) == 0);
  assert(dtype_is_signed(DTYPE_UINT) == 0);
  assert(dtype_is_signed(DTYPE_ULONG) == 0);
  printf("  ✓ Signed type detection works\n");
}

int main(void) {
  printf("=== Running DType Tests ===\n\n");
  
  test_dtype_size();
  test_dtype_is_float();
  test_dtype_is_signed();
  
  printf("\n=== All DType Tests Passed ===\n");
  return 0;
}
