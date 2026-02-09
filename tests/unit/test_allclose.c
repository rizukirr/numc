/**
 * @file test_allclose.c
 * @brief Test array_allclose functionality
 */

#include <numc/numc.h>
#include <assert.h>
#include <stdio.h>

void test_allclose_identical(void) {
  printf("Testing allclose with identical arrays...\n");

  int data[] = {1, 2, 3, 4, 5};
  Array *a = array_create(&(ArrayCreate){
      .ndim = 1, .shape = (size_t[]){5},
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true});
  Array *b = array_create(&(ArrayCreate){
      .ndim = 1, .shape = (size_t[]){5},
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true});

  assert(array_allclose(a, b, 1e-5, 1e-8) == 1);

  array_free(b);
  array_free(a);
  printf("  ✓ Identical arrays are allclose\n");
}

void test_allclose_within_atol(void) {
  printf("Testing allclose within absolute tolerance...\n");

  float data_a[] = {1.0f, 2.0f, 3.0f};
  float data_b[] = {1.0000001f, 2.0000001f, 3.0000001f};
  Array *a = array_create(&(ArrayCreate){
      .ndim = 1, .shape = (size_t[]){3},
      .numc_type = NUMC_TYPE_FLOAT, .data = data_a, .owns_data = true});
  Array *b = array_create(&(ArrayCreate){
      .ndim = 1, .shape = (size_t[]){3},
      .numc_type = NUMC_TYPE_FLOAT, .data = data_b, .owns_data = true});

  assert(array_allclose(a, b, 0.0, 1e-5) == 1);

  array_free(b);
  array_free(a);
  printf("  ✓ Arrays within absolute tolerance pass\n");
}

void test_allclose_within_rtol(void) {
  printf("Testing allclose within relative tolerance...\n");

  float data_a[] = {100.0f, 200.0f, 300.0f};
  float data_b[] = {100.001f, 200.002f, 300.003f};
  Array *a = array_create(&(ArrayCreate){
      .ndim = 1, .shape = (size_t[]){3},
      .numc_type = NUMC_TYPE_FLOAT, .data = data_a, .owns_data = true});
  Array *b = array_create(&(ArrayCreate){
      .ndim = 1, .shape = (size_t[]){3},
      .numc_type = NUMC_TYPE_FLOAT, .data = data_b, .owns_data = true});

  assert(array_allclose(a, b, 1e-4, 0.0) == 1);

  array_free(b);
  array_free(a);
  printf("  ✓ Arrays within relative tolerance pass\n");
}

void test_allclose_outside_tolerance(void) {
  printf("Testing allclose outside tolerance...\n");

  float data_a[] = {1.0f, 2.0f, 3.0f};
  float data_b[] = {1.0f, 2.5f, 3.0f};
  Array *a = array_create(&(ArrayCreate){
      .ndim = 1, .shape = (size_t[]){3},
      .numc_type = NUMC_TYPE_FLOAT, .data = data_a, .owns_data = true});
  Array *b = array_create(&(ArrayCreate){
      .ndim = 1, .shape = (size_t[]){3},
      .numc_type = NUMC_TYPE_FLOAT, .data = data_b, .owns_data = true});

  assert(array_allclose(a, b, 1e-5, 1e-8) == 0);

  array_free(b);
  array_free(a);
  printf("  ✓ Arrays outside tolerance fail\n");
}

void test_allclose_double_rounding(void) {
  printf("Testing allclose with double rounding (0.1+0.2 vs 0.3)...\n");

  double val_a = 0.1 + 0.2;
  double val_b = 0.3;
  Array *a = array_create(&(ArrayCreate){
      .ndim = 1, .shape = (size_t[]){1},
      .numc_type = NUMC_TYPE_DOUBLE, .data = &val_a, .owns_data = true});
  Array *b = array_create(&(ArrayCreate){
      .ndim = 1, .shape = (size_t[]){1},
      .numc_type = NUMC_TYPE_DOUBLE, .data = &val_b, .owns_data = true});

  // Exact equality fails
  Array *eq = array_equal(a, b);
  double *eq_data = (double *)eq->data;
  assert(eq_data[0] == 0.0);

  // allclose passes
  assert(array_allclose(a, b, 1e-5, 1e-8) == 1);

  array_free(eq);
  array_free(b);
  array_free(a);
  printf("  ✓ Double rounding handled by allclose\n");
}

void test_allclose_exact_zero_tolerance(void) {
  printf("Testing allclose with zero tolerance (exact match)...\n");

  float data_a[] = {1.0f, 2.0f, 3.0f};
  float data_b[] = {1.0f, 2.0f, 3.0f};
  float data_c[] = {1.0f, 2.001f, 3.0f};
  Array *a = array_create(&(ArrayCreate){
      .ndim = 1, .shape = (size_t[]){3},
      .numc_type = NUMC_TYPE_FLOAT, .data = data_a, .owns_data = true});
  Array *b = array_create(&(ArrayCreate){
      .ndim = 1, .shape = (size_t[]){3},
      .numc_type = NUMC_TYPE_FLOAT, .data = data_b, .owns_data = true});
  Array *c = array_create(&(ArrayCreate){
      .ndim = 1, .shape = (size_t[]){3},
      .numc_type = NUMC_TYPE_FLOAT, .data = data_c, .owns_data = true});

  assert(array_allclose(a, b, 0.0, 0.0) == 1);
  assert(array_allclose(a, c, 0.0, 0.0) == 0);

  array_free(c);
  array_free(b);
  array_free(a);
  printf("  ✓ Zero tolerance requires exact match\n");
}

void test_allclose_double(void) {
  printf("Testing allclose with double arrays...\n");

  double data_a[] = {1.0, 2.0, 3.0, 4.0};
  double data_b[] = {1.0 + 1e-10, 2.0 + 1e-10, 3.0 + 1e-10, 4.0 + 1e-10};
  Array *a = array_create(&(ArrayCreate){
      .ndim = 1, .shape = (size_t[]){4},
      .numc_type = NUMC_TYPE_DOUBLE, .data = data_a, .owns_data = true});
  Array *b = array_create(&(ArrayCreate){
      .ndim = 1, .shape = (size_t[]){4},
      .numc_type = NUMC_TYPE_DOUBLE, .data = data_b, .owns_data = true});

  assert(array_allclose(a, b, 1e-5, 1e-8) == 1);

  array_free(b);
  array_free(a);
  printf("  ✓ Double allclose works correctly\n");
}

void test_allclose_2d(void) {
  printf("Testing allclose with 2D arrays...\n");

  float data_a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  float data_b[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Array *a = array_create(&(ArrayCreate){
      .ndim = 2, .shape = (size_t[]){2, 3},
      .numc_type = NUMC_TYPE_FLOAT, .data = data_a, .owns_data = true});
  Array *b = array_create(&(ArrayCreate){
      .ndim = 2, .shape = (size_t[]){2, 3},
      .numc_type = NUMC_TYPE_FLOAT, .data = data_b, .owns_data = true});

  assert(array_allclose(a, b, 1e-5, 1e-8) == 1);

  array_free(b);
  array_free(a);
  printf("  ✓ 2D array allclose works correctly\n");
}

void test_allclose_null(void) {
  printf("Testing allclose with NULL arguments...\n");

  Array *a = array_create(&(ArrayCreate){
      .ndim = 1, .shape = (size_t[]){3},
      .numc_type = NUMC_TYPE_INT, .owns_data = true});

  assert(array_allclose(NULL, a, 1e-5, 1e-8) < 0);
  assert(array_allclose(a, NULL, 1e-5, 1e-8) < 0);
  assert(array_allclose(NULL, NULL, 1e-5, 1e-8) < 0);

  array_free(a);
  printf("  ✓ NULL arguments return negative error code\n");
}

void test_allclose_type_mismatch(void) {
  printf("Testing allclose with mismatched types...\n");

  int idata[] = {1, 2, 3};
  float fdata[] = {1.0f, 2.0f, 3.0f};
  Array *a = array_create(&(ArrayCreate){
      .ndim = 1, .shape = (size_t[]){3},
      .numc_type = NUMC_TYPE_INT, .data = idata, .owns_data = true});
  Array *b = array_create(&(ArrayCreate){
      .ndim = 1, .shape = (size_t[]){3},
      .numc_type = NUMC_TYPE_FLOAT, .data = fdata, .owns_data = true});

  assert(array_allclose(a, b, 1e-5, 1e-8) < 0);

  array_free(b);
  array_free(a);
  printf("  ✓ Type mismatch returns negative error code\n");
}

void test_allclose_ndim_mismatch(void) {
  printf("Testing allclose with mismatched ndim...\n");

  int data[] = {1, 2, 3, 4, 5, 6};
  Array *a = array_create(&(ArrayCreate){
      .ndim = 1, .shape = (size_t[]){6},
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true});
  Array *b = array_create(&(ArrayCreate){
      .ndim = 2, .shape = (size_t[]){2, 3},
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true});

  assert(array_allclose(a, b, 1e-5, 1e-8) < 0);

  array_free(b);
  array_free(a);
  printf("  ✓ ndim mismatch returns negative error code\n");
}

void test_allclose_shape_mismatch(void) {
  printf("Testing allclose with mismatched shapes...\n");

  int data[] = {1, 2, 3, 4, 5, 6};
  Array *a = array_create(&(ArrayCreate){
      .ndim = 2, .shape = (size_t[]){2, 3},
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true});
  Array *b = array_create(&(ArrayCreate){
      .ndim = 2, .shape = (size_t[]){3, 2},
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true});

  assert(array_allclose(a, b, 1e-5, 1e-8) < 0);

  array_free(b);
  array_free(a);
  printf("  ✓ Shape mismatch returns negative error code\n");
}

int main(void) {
  printf("=== Running Array Allclose Tests ===\n\n");

  test_allclose_identical();
  test_allclose_within_atol();
  test_allclose_within_rtol();
  test_allclose_outside_tolerance();
  test_allclose_double_rounding();
  test_allclose_exact_zero_tolerance();
  test_allclose_double();
  test_allclose_2d();
  test_allclose_null();
  test_allclose_type_mismatch();
  test_allclose_ndim_mismatch();
  test_allclose_shape_mismatch();

  printf("\n=== All Array Allclose Tests Passed ===\n");
  return 0;
}
