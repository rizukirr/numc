/**
 * @file test_equal.c
 * @brief Test array_equal functionality
 */

#include <numc/numc.h>
#include <assert.h>
#include <stdio.h>

void test_equal_identical(void) {
  printf("Testing equal with identical arrays...\n");

  int data[] = {1, 2, 3, 4, 5};
  Array *a = array_create(&(ArrayCreate){
      .ndim = 1, .shape = (size_t[]){5},
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true});
  Array *b = array_create(&(ArrayCreate){
      .ndim = 1, .shape = (size_t[]){5},
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true});

  Array *eq = array_equal(a, b);
  assert(eq != NULL);
  assert(eq->size == 5);
  assert(eq->numc_type == NUMC_TYPE_INT);

  int *out = (int *)eq->data;
  for (size_t i = 0; i < 5; i++) {
    assert(out[i] == 1);
  }

  array_free(eq);
  array_free(b);
  array_free(a);
  printf("  ✓ Identical arrays compare as all 1s\n");
}

void test_equal_different(void) {
  printf("Testing equal with different arrays...\n");

  int data_a[] = {1, 2, 3, 4, 5};
  int data_b[] = {1, 0, 3, 0, 5};
  Array *a = array_create(&(ArrayCreate){
      .ndim = 1, .shape = (size_t[]){5},
      .numc_type = NUMC_TYPE_INT, .data = data_a, .owns_data = true});
  Array *b = array_create(&(ArrayCreate){
      .ndim = 1, .shape = (size_t[]){5},
      .numc_type = NUMC_TYPE_INT, .data = data_b, .owns_data = true});

  Array *eq = array_equal(a, b);
  assert(eq != NULL);

  int *out = (int *)eq->data;
  int expected[] = {1, 0, 1, 0, 1};
  for (size_t i = 0; i < 5; i++) {
    assert(out[i] == expected[i]);
  }

  array_free(eq);
  array_free(b);
  array_free(a);
  printf("  ✓ Different arrays produce correct element-wise result\n");
}

void test_equal_all_different(void) {
  printf("Testing equal with completely different arrays...\n");

  int data_a[] = {1, 2, 3};
  int data_b[] = {4, 5, 6};
  Array *a = array_create(&(ArrayCreate){
      .ndim = 1, .shape = (size_t[]){3},
      .numc_type = NUMC_TYPE_INT, .data = data_a, .owns_data = true});
  Array *b = array_create(&(ArrayCreate){
      .ndim = 1, .shape = (size_t[]){3},
      .numc_type = NUMC_TYPE_INT, .data = data_b, .owns_data = true});

  Array *eq = array_equal(a, b);
  assert(eq != NULL);

  int *out = (int *)eq->data;
  for (size_t i = 0; i < 3; i++) {
    assert(out[i] == 0);
  }

  array_free(eq);
  array_free(b);
  array_free(a);
  printf("  ✓ All-different arrays compare as all 0s\n");
}

void test_equal_float(void) {
  printf("Testing equal with float arrays...\n");

  float data_a[] = {1.0f, 2.5f, 3.0f, 4.5f};
  float data_b[] = {1.0f, 2.5f, 0.0f, 4.5f};
  Array *a = array_create(&(ArrayCreate){
      .ndim = 1, .shape = (size_t[]){4},
      .numc_type = NUMC_TYPE_FLOAT, .data = data_a, .owns_data = true});
  Array *b = array_create(&(ArrayCreate){
      .ndim = 1, .shape = (size_t[]){4},
      .numc_type = NUMC_TYPE_FLOAT, .data = data_b, .owns_data = true});

  Array *eq = array_equal(a, b);
  assert(eq != NULL);
  assert(eq->numc_type == NUMC_TYPE_FLOAT);

  float *out = (float *)eq->data;
  float expected[] = {1.0f, 1.0f, 0.0f, 1.0f};
  for (size_t i = 0; i < 4; i++) {
    assert(out[i] == expected[i]);
  }

  array_free(eq);
  array_free(b);
  array_free(a);
  printf("  ✓ Float equality works correctly\n");
}

void test_equal_2d(void) {
  printf("Testing equal with 2D arrays...\n");

  int data_a[] = {1, 2, 3, 4, 5, 6};
  int data_b[] = {1, 0, 3, 0, 5, 0};
  Array *a = array_create(&(ArrayCreate){
      .ndim = 2, .shape = (size_t[]){2, 3},
      .numc_type = NUMC_TYPE_INT, .data = data_a, .owns_data = true});
  Array *b = array_create(&(ArrayCreate){
      .ndim = 2, .shape = (size_t[]){2, 3},
      .numc_type = NUMC_TYPE_INT, .data = data_b, .owns_data = true});

  Array *eq = array_equal(a, b);
  assert(eq != NULL);
  assert(eq->ndim == 2);
  assert(eq->shape[0] == 2);
  assert(eq->shape[1] == 3);

  int *out = (int *)eq->data;
  int expected[] = {1, 0, 1, 0, 1, 0};
  for (size_t i = 0; i < 6; i++) {
    assert(out[i] == expected[i]);
  }

  array_free(eq);
  array_free(b);
  array_free(a);
  printf("  ✓ 2D array equality works correctly\n");
}

void test_equal_null(void) {
  printf("Testing equal with NULL arguments...\n");

  Array *a = array_create(&(ArrayCreate){
      .ndim = 1, .shape = (size_t[]){3},
      .numc_type = NUMC_TYPE_INT, .owns_data = true});

  assert(array_equal(NULL, a) == NULL);
  assert(array_equal(a, NULL) == NULL);
  assert(array_equal(NULL, NULL) == NULL);

  array_free(a);
  printf("  ✓ NULL arguments return NULL\n");
}

void test_equal_type_mismatch(void) {
  printf("Testing equal with mismatched types...\n");

  int idata[] = {1, 2, 3};
  float fdata[] = {1.0f, 2.0f, 3.0f};
  Array *a = array_create(&(ArrayCreate){
      .ndim = 1, .shape = (size_t[]){3},
      .numc_type = NUMC_TYPE_INT, .data = idata, .owns_data = true});
  Array *b = array_create(&(ArrayCreate){
      .ndim = 1, .shape = (size_t[]){3},
      .numc_type = NUMC_TYPE_FLOAT, .data = fdata, .owns_data = true});

  assert(array_equal(a, b) == NULL);

  array_free(b);
  array_free(a);
  printf("  ✓ Type mismatch returns NULL\n");
}

void test_equal_ndim_mismatch(void) {
  printf("Testing equal with mismatched ndim...\n");

  int data[] = {1, 2, 3, 4, 5, 6};
  Array *a = array_create(&(ArrayCreate){
      .ndim = 1, .shape = (size_t[]){6},
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true});
  Array *b = array_create(&(ArrayCreate){
      .ndim = 2, .shape = (size_t[]){2, 3},
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true});

  assert(array_equal(a, b) == NULL);

  array_free(b);
  array_free(a);
  printf("  ✓ ndim mismatch returns NULL\n");
}

void test_equal_output_owns_data(void) {
  printf("Testing that result owns its data...\n");

  int data[] = {1, 2, 3};
  Array *a = array_create(&(ArrayCreate){
      .ndim = 1, .shape = (size_t[]){3},
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true});
  Array *b = array_create(&(ArrayCreate){
      .ndim = 1, .shape = (size_t[]){3},
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true});

  Array *eq = array_equal(a, b);
  assert(eq != NULL);
  assert(eq->owns_data == true);
  assert(eq->data != a->data);
  assert(eq->data != b->data);

  array_free(eq);
  array_free(b);
  array_free(a);
  printf("  ✓ Result owns its own data\n");
}

int main(void) {
  printf("=== Running Array Equal Tests ===\n\n");

  test_equal_identical();
  test_equal_different();
  test_equal_all_different();
  test_equal_float();
  test_equal_2d();
  test_equal_null();
  test_equal_type_mismatch();
  test_equal_ndim_mismatch();
  test_equal_output_owns_data();

  printf("\n=== All Array Equal Tests Passed ===\n");
  return 0;
}
