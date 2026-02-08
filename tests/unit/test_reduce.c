/**
 * @file test_reduce.c
 * @brief Test reduction operations (sum, min, max, dot)
 */

#include <numc/numc.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void test_sum_int(void) {
  size_t shape[] = {10};
  Array *a = array_zeros(1, shape, NUMC_TYPE_INT);

  // Fill with 1..10
  for (size_t i = 0; i < a->size; i++)
    ((NUMC_INT *)a->data)[i] = (NUMC_INT)(i + 1);

  NUMC_INT result = 0;
  int ret = array_sum(a, &result);
  assert(ret == 0);
  assert(result == 55);

  array_free(a);
  printf("  test_sum_int\n");
}

void test_sum_float(void) {
  size_t shape[] = {5};
  Array *a = array_zeros(1, shape, NUMC_TYPE_FLOAT);

  ((NUMC_FLOAT *)a->data)[0] = 1.5f;
  ((NUMC_FLOAT *)a->data)[1] = 2.5f;
  ((NUMC_FLOAT *)a->data)[2] = 3.0f;
  ((NUMC_FLOAT *)a->data)[3] = 4.0f;
  ((NUMC_FLOAT *)a->data)[4] = 0.5f;

  NUMC_FLOAT result = 0;
  int ret = array_sum(a, &result);
  assert(ret == 0);
  assert(fabsf(result - 11.5f) < 1e-6f);

  array_free(a);
  printf("  test_sum_float\n");
}

void test_min_max(void) {
  size_t shape[] = {6};
  Array *a = array_zeros(1, shape, NUMC_TYPE_INT);

  NUMC_INT values[] = {5, -3, 8, 1, -7, 4};
  for (size_t i = 0; i < 6; i++)
    ((NUMC_INT *)a->data)[i] = values[i];

  NUMC_INT min_val = 0, max_val = 0;
  assert(array_min(a, &min_val) == 0);
  assert(array_max(a, &max_val) == 0);
  assert(min_val == -7);
  assert(max_val == 8);

  array_free(a);
  printf("  test_min_max\n");
}

void test_dot_float(void) {
  size_t shape[] = {4};
  Array *a = array_zeros(1, shape, NUMC_TYPE_FLOAT);
  Array *b = array_zeros(1, shape, NUMC_TYPE_FLOAT);

  // a = [1, 2, 3, 4], b = [2, 3, 4, 5]
  // dot = 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
  for (size_t i = 0; i < 4; i++) {
    ((NUMC_FLOAT *)a->data)[i] = (float)(i + 1);
    ((NUMC_FLOAT *)b->data)[i] = (float)(i + 2);
  }

  NUMC_FLOAT result = 0;
  int ret = array_dot(a, b, &result);
  assert(ret == 0);
  assert(fabsf(result - 40.0f) < 1e-6f);

  array_free(a);
  array_free(b);
  printf("  test_dot_float\n");
}

void test_dot_mismatched(void) {
  size_t shape1[] = {4};
  size_t shape2[] = {5};
  Array *a = array_zeros(1, shape1, NUMC_TYPE_INT);
  Array *b = array_zeros(1, shape2, NUMC_TYPE_INT);

  NUMC_INT result = 0;
  int ret = array_dot(a, b, &result);
  assert(ret < 0);

  array_free(a);
  array_free(b);
  printf("  test_dot_mismatched\n");
}

void test_reduce_null(void) {
  NUMC_INT result = 0;
  assert(array_sum(NULL, &result) < 0);
  assert(array_min(NULL, &result) < 0);
  assert(array_max(NULL, &result) < 0);
  assert(array_dot(NULL, NULL, &result) < 0);
  printf("  test_reduce_null\n");
}

void test_reduce_all_numc_types(void) {
  size_t shape[] = {5};

  NUMC_TYPE types[] = {NUMC_TYPE_BYTE,   NUMC_TYPE_UBYTE, NUMC_TYPE_SHORT,
                       NUMC_TYPE_USHORT, NUMC_TYPE_INT,   NUMC_TYPE_UINT,
                       NUMC_TYPE_LONG,   NUMC_TYPE_ULONG, NUMC_TYPE_FLOAT,
                       NUMC_TYPE_DOUBLE};

  for (size_t t = 0; t < sizeof(types) / sizeof(NUMC_TYPE); t++) {
    Array *a = array_ones(1, shape, types[t]);

    // sum of 5 ones = 5, stored in native type
    uint8_t out_buf[8] = {0};
    int ret = array_sum(a, out_buf);
    assert(ret == 0);

    array_free(a);
  }

  printf("  test_reduce_all_numc_types\n");
}

int main(void) {
  test_sum_int();
  test_sum_float();
  test_min_max();
  test_dot_float();
  test_dot_mismatched();
  test_reduce_null();
  test_reduce_all_numc_types();

  printf("\n  All reduction tests passed\n");
  return 0;
}
