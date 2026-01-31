/**
 * @file test_memory_alignment.c
 * @brief Test SIMD memory alignment
 */

#include "array.h"
#include "dtype.h"
#include "memory.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

void test_array_alignment(void) {
  printf("Testing array data alignment...\n");
  
  // Test different types
  Array *arr_int = array_create(1, (size_t[]){100}, DTYPE_INT);
  Array *arr_float = array_create(1, (size_t[]){100}, DTYPE_FLOAT);
  Array *arr_double = array_create(2, (size_t[]){10, 10}, DTYPE_DOUBLE);
  Array *arr_long = array_create(3, (size_t[]){5, 5, 4}, DTYPE_LONG);
  
  // All should be 16-byte aligned for SIMD
  assert(((uintptr_t)arr_int->data % NUMC_ALIGN) == 0);
  assert(((uintptr_t)arr_float->data % NUMC_ALIGN) == 0);
  assert(((uintptr_t)arr_double->data % NUMC_ALIGN) == 0);
  assert(((uintptr_t)arr_long->data % NUMC_ALIGN) == 0);
  
  array_free(arr_int);
  array_free(arr_float);
  array_free(arr_double);
  array_free(arr_long);
  
  printf("  ✓ All arrays are 16-byte aligned\n");
}

void test_aligned_calloc(void) {
  printf("Testing aligned_calloc...\n");
  
  void *ptr = aligned_calloc(NUMC_ALIGN, 1024);
  assert(ptr != NULL);
  assert(((uintptr_t)ptr % NUMC_ALIGN) == 0);
  
  // Check it's zero-initialized
  unsigned char *bytes = (unsigned char *)ptr;
  for (int i = 0; i < 1024; i++) {
    assert(bytes[i] == 0);
  }
  
  aligned_free(ptr);
  printf("  ✓ aligned_calloc works\n");
}

void test_aligned_realloc(void) {
  printf("Testing aligned_realloc...\n");
  
  // Allocate 100 bytes
  void *ptr = aligned_calloc(NUMC_ALIGN, 100);
  assert(ptr != NULL);
  
  // Fill with data
  int *data = (int *)ptr;
  for (int i = 0; i < 25; i++) {
    data[i] = i;
  }
  
  // Realloc to 200 bytes
  void *new_ptr = aligned_realloc(ptr, NUMC_ALIGN, 100, 200);
  assert(new_ptr != NULL);
  assert(((uintptr_t)new_ptr % NUMC_ALIGN) == 0);
  
  // Check old data preserved
  int *new_data = (int *)new_ptr;
  for (int i = 0; i < 25; i++) {
    assert(new_data[i] == i);
  }
  
  aligned_free(new_ptr);
  printf("  ✓ aligned_realloc works\n");
}

int main(void) {
  printf("=== Running Memory Alignment Tests ===\n\n");
  
  test_array_alignment();
  test_aligned_calloc();
  test_aligned_realloc();
  
  printf("\n=== All Memory Alignment Tests Passed ===\n");
  return 0;
}
