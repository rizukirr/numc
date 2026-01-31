#include "array.h"
#include "arrio.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

int main(void) {
  printf("=== Testing static array to dynamic Array conversion ===\n");
  size_t a_shape[] = {2, 2, 2};
  Array *arr = array_zeros(3, a_shape, sizeof(float));

  float *data = (float *)arr->data;
  size_t offset = 0;
  array_print(data, arr->shape, arr->ndim, 0, &offset);
  printf("\n");

  array_free(arr);
  return 0;
}
