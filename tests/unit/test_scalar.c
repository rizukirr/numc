/**
 * @file test_scalar.c
 * @brief Test scalar-array operations (adds, subs, muls, divs)
 */

#include "array.h"
#include "types.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void test_adds_int(void) {
  size_t shape[] = {3};
  Array *a = array_zeros(1, shape, NUMC_TYPE_INT);
  Array *out = array_zeros(1, shape, NUMC_TYPE_INT);

  // a = [1, 2, 3], scalar = 5 => [6, 7, 8]
  ((NUMC_INT *)a->data)[0] = 1;
  ((NUMC_INT *)a->data)[1] = 2;
  ((NUMC_INT *)a->data)[2] = 3;

  NUMC_INT scalar = 5;
  int ret = array_add_scalar(a, &scalar, out);
  assert(ret == 0);
  assert(((NUMC_INT *)out->data)[0] == 6);
  assert(((NUMC_INT *)out->data)[1] == 7);
  assert(((NUMC_INT *)out->data)[2] == 8);

  array_free(a);
  array_free(out);
  printf("  test_adds_int\n");
}

void test_subs_int(void) {
  size_t shape[] = {3};
  Array *a = array_zeros(1, shape, NUMC_TYPE_INT);
  Array *out = array_zeros(1, shape, NUMC_TYPE_INT);

  ((NUMC_INT *)a->data)[0] = 10;
  ((NUMC_INT *)a->data)[1] = 20;
  ((NUMC_INT *)a->data)[2] = 30;

  NUMC_INT scalar = 5;
  int ret = array_subtract_scalar(a, &scalar, out);
  assert(ret == 0);
  assert(((NUMC_INT *)out->data)[0] == 5);
  assert(((NUMC_INT *)out->data)[1] == 15);
  assert(((NUMC_INT *)out->data)[2] == 25);

  array_free(a);
  array_free(out);
  printf("  test_subs_int\n");
}

void test_muls_float(void) {
  size_t shape[] = {4};
  Array *a = array_zeros(1, shape, NUMC_TYPE_FLOAT);
  Array *out = array_zeros(1, shape, NUMC_TYPE_FLOAT);

  for (size_t i = 0; i < 4; i++)
    ((NUMC_FLOAT *)a->data)[i] = (float)(i + 1);

  NUMC_FLOAT scalar = 2.0f;
  int ret = array_multiply_scalar(a, &scalar, out);
  assert(ret == 0);
  assert(fabsf(((NUMC_FLOAT *)out->data)[0] - 2.0f) < 1e-6f);
  assert(fabsf(((NUMC_FLOAT *)out->data)[1] - 4.0f) < 1e-6f);
  assert(fabsf(((NUMC_FLOAT *)out->data)[2] - 6.0f) < 1e-6f);
  assert(fabsf(((NUMC_FLOAT *)out->data)[3] - 8.0f) < 1e-6f);

  array_free(a);
  array_free(out);
  printf("  test_muls_float\n");
}

void test_divs_double(void) {
  size_t shape[] = {3};
  Array *a = array_zeros(1, shape, NUMC_TYPE_DOUBLE);
  Array *out = array_zeros(1, shape, NUMC_TYPE_DOUBLE);

  ((NUMC_DOUBLE *)a->data)[0] = 10.0;
  ((NUMC_DOUBLE *)a->data)[1] = 20.0;
  ((NUMC_DOUBLE *)a->data)[2] = 30.0;

  NUMC_DOUBLE scalar = 5.0;
  int ret = array_divide_scalar(a, &scalar, out);
  assert(ret == 0);
  assert(fabs(((NUMC_DOUBLE *)out->data)[0] - 2.0) < 1e-10);
  assert(fabs(((NUMC_DOUBLE *)out->data)[1] - 4.0) < 1e-10);
  assert(fabs(((NUMC_DOUBLE *)out->data)[2] - 6.0) < 1e-10);

  array_free(a);
  array_free(out);
  printf("  test_divs_double\n");
}

void test_scalar_mismatched_numc_type(void) {
  size_t shape[] = {3};
  Array *a = array_zeros(1, shape, NUMC_TYPE_INT);
  Array *out = array_zeros(1, shape, NUMC_TYPE_FLOAT);

  NUMC_INT scalar = 1;
  int ret = array_add_scalar(a, &scalar, out);
  assert(ret == -1);

  array_free(a);
  array_free(out);
  printf("  test_scalar_mismatched_numc_type\n");
}

void test_scalar_null(void) {
  size_t shape[] = {3};
  Array *a = array_zeros(1, shape, NUMC_TYPE_INT);
  Array *out = array_zeros(1, shape, NUMC_TYPE_INT);

  assert(array_add_scalar(NULL, NULL, out) == -1);
  assert(array_add_scalar(a, NULL, out) == -1);

  array_free(a);
  array_free(out);
  printf("  test_scalar_null\n");
}

void test_scalar_all_numc_types(void) {
  size_t shape[] = {8};

  NUMC_TYPE types[] = {NUMC_TYPE_BYTE,  NUMC_TYPE_UBYTE, NUMC_TYPE_SHORT, NUMC_TYPE_USHORT,
                   NUMC_TYPE_INT,   NUMC_TYPE_UINT,  NUMC_TYPE_LONG,  NUMC_TYPE_ULONG,
                   NUMC_TYPE_FLOAT, NUMC_TYPE_DOUBLE};

  for (size_t t = 0; t < sizeof(types) / sizeof(NUMC_TYPE); t++) {
    Array *a = array_ones(1, shape, types[t]);
    Array *out = array_zeros(1, shape, types[t]);

    // Use a scalar of 1 (matching type) — stored as bytes, cast inside kernel
    uint8_t scalar_buf[8] = {0};
    // Set scalar to 1 for the given type size
    switch (numc_type_size(types[t])) {
    case 1:
      scalar_buf[0] = 1;
      break;
    case 2:
      *(uint16_t *)scalar_buf = 1;
      break;
    case 4:
      if (types[t] == NUMC_TYPE_FLOAT)
        *(float *)scalar_buf = 1.0f;
      else
        *(uint32_t *)scalar_buf = 1;
      break;
    case 8:
      if (types[t] == NUMC_TYPE_DOUBLE)
        *(double *)scalar_buf = 1.0;
      else
        *(uint64_t *)scalar_buf = 1;
      break;
    }

    int ret = array_add_scalar(a, scalar_buf, out);
    assert(ret == 0);
    assert(out->numc_type == types[t]);

    array_free(a);
    array_free(out);
  }

  printf("  test_scalar_all_numc_types\n");
}

void test_scalar_2d(void) {
  size_t shape[] = {3, 4};
  Array *a = array_zeros(2, shape, NUMC_TYPE_INT);
  Array *out = array_zeros(2, shape, NUMC_TYPE_INT);

  for (size_t i = 0; i < a->size; i++)
    ((NUMC_INT *)a->data)[i] = (NUMC_INT)i;

  NUMC_INT scalar = 10;
  int ret = array_add_scalar(a, &scalar, out);
  assert(ret == 0);

  for (size_t i = 0; i < out->size; i++)
    assert(((NUMC_INT *)out->data)[i] == (NUMC_INT)(i + 10));

  array_free(a);
  array_free(out);
  printf("  test_scalar_2d\n");
}

int main(void) {
  test_adds_int();
  test_subs_int();
  test_muls_float();
  test_divs_double();
  test_scalar_mismatched_numc_type();
  test_scalar_null();
  test_scalar_all_numc_types();
  test_scalar_2d();

  printf("\n  All scalar operation tests passed\n");
  return 0;
}
