/**
 * @file test_memory_alignment.c
 * @brief Test SIMD memory alignment
 */

#include <numc/numc.h>
#include "internal.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

void test_array_alignment(void) {
  printf("Testing array data alignment...\n");
  
  // Test different types
  Array *arr_int = array_create(&(ArrayCreate){.ndim = 1, .shape = (size_t[]){100}, .numc_type = NUMC_TYPE_INT, .data = NULL, .owns_data = true});
  Array *arr_float = array_create(&(ArrayCreate){.ndim = 1, .shape = (size_t[]){100}, .numc_type = NUMC_TYPE_FLOAT, .data = NULL, .owns_data = true});
  Array *arr_double = array_create(&(ArrayCreate){.ndim = 2, .shape = (size_t[]){10, 10}, .numc_type = NUMC_TYPE_DOUBLE, .data = NULL, .owns_data = true});
  Array *arr_long = array_create(&(ArrayCreate){.ndim = 3, .shape = (size_t[]){5, 5, 4}, .numc_type = NUMC_TYPE_LONG, .data = NULL, .owns_data = true});
  
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

void test_numc_calloc(void) {
  printf("Testing numc_calloc...\n");
  
  void *ptr = numc_calloc(NUMC_ALIGN, 1024);
  assert(ptr != NULL);
  assert(((uintptr_t)ptr % NUMC_ALIGN) == 0);
  
  // Check it's zero-initialized
  unsigned char *bytes = (unsigned char *)ptr;
  for (int i = 0; i < 1024; i++) {
    assert(bytes[i] == 0);
  }
  
  numc_free(ptr);
  printf("  ✓ numc_calloc works\n");
}

void test_numc_realloc(void) {
  printf("Testing numc_realloc...\n");
  
  // Allocate 100 bytes
  void *ptr = numc_calloc(NUMC_ALIGN, 100);
  assert(ptr != NULL);
  
  // Fill with data
  int *data = (int *)ptr;
  for (int i = 0; i < 25; i++) {
    data[i] = i;
  }
  
  // Realloc to 200 bytes
  void *new_ptr = numc_realloc(ptr, NUMC_ALIGN, 100, 200);
  assert(new_ptr != NULL);
  assert(((uintptr_t)new_ptr % NUMC_ALIGN) == 0);
  
  // Check old data preserved
  int *new_data = (int *)new_ptr;
  for (int i = 0; i < 25; i++) {
    assert(new_data[i] == i);
  }
  
  numc_free(new_ptr);
  printf("  ✓ numc_realloc works\n");
}

int main(void) {
  printf("=== Running Memory Alignment Tests ===\n\n");
  
  test_array_alignment();
  test_numc_calloc();
  test_numc_realloc();
  
  printf("\n=== All Memory Alignment Tests Passed ===\n");
  return 0;
}
