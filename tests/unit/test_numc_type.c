/**
 * @file test_numc_type.c
 * @brief Test NUMC_TYPE system functionality
 */

#include "types.h"
#include <assert.h>
#include <stdio.h>

void test_numc_type_size(void) {
  printf("Testing numc_type_size...\n");
  assert(numc_type_size(NUMC_TYPE_BYTE) == 1);
  assert(numc_type_size(NUMC_TYPE_UBYTE) == 1);
  assert(numc_type_size(NUMC_TYPE_SHORT) == 2);
  assert(numc_type_size(NUMC_TYPE_USHORT) == 2);
  assert(numc_type_size(NUMC_TYPE_INT) == 4);
  assert(numc_type_size(NUMC_TYPE_UINT) == 4);
  assert(numc_type_size(NUMC_TYPE_LONG) == 8);
  assert(numc_type_size(NUMC_TYPE_ULONG) == 8);
  assert(numc_type_size(NUMC_TYPE_FLOAT) == 4);
  assert(numc_type_size(NUMC_TYPE_DOUBLE) == 8);
  printf("  ✓ All numc_type sizes correct\n");
}

int main(void) {
  printf("=== Running NUMC_TYPE Tests ===\n\n");

  test_numc_type_size();

  printf("\n=== All NUMC_TYPE Tests Passed ===\n");
  return 0;
}
