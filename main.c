#include "array.h"
#include "arrio.h"
#include "dtype.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

int main(void) {
  printf("=== Testing DType API with array_ones ===\n");
  size_t a_shape[] = {2, 2, 2};
  Array *arr = array_ones(3, a_shape, DTYPE_FLOAT);

  float *data = (float *)arr->data;
  size_t offset = 0;
  array_print(data, arr->shape, arr->ndim, 0, &offset);
  printf("\n");

  array_free(arr);

  printf("=== Testing DType API with array_fill ===\n");
  size_t b_shape[] = {2, 2, 2};
  NUMC_FLOAT elem = 1.5f;
  Array *arr2 = array_fill(3, b_shape, DTYPE_FLOAT, &elem);

  float *data2 = (float *)arr2->data;
  size_t offset2 = 0;
  array_print(data2, arr2->shape, arr2->ndim, 0, &offset2);
  printf("\n");

  array_free(arr2);
  return 0;
}
