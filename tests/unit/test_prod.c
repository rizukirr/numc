#include <assert.h>
#include <math.h>
#include <numc/numc.h>
#include <stdio.h>

static void test_prod_int(void) {
  printf("  test_prod_int...");
  int data[] = {1, 2, 3, 4, 5};
  size_t shape[] = {5};
  ArrayCreate create = {
      .ndim = 1, .shape = shape,
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true,
  };
  Array *a = array_create(&create);
  assert(a);

  int result = 0;
  assert(array_prod(a, &result) == NUMC_OK);
  assert(result == 120); /* 1*2*3*4*5 */

  array_free(a);
  printf(" PASSED\n");
}

static void test_prod_float(void) {
  printf("  test_prod_float...");
  float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  size_t shape[] = {4};
  ArrayCreate create = {
      .ndim = 1, .shape = shape,
      .numc_type = NUMC_TYPE_FLOAT, .data = data, .owns_data = true,
  };
  Array *a = array_create(&create);
  assert(a);

  float result = 0.0f;
  assert(array_prod(a, &result) == NUMC_OK);
  assert(fabsf(result - 24.0f) < 1e-6f); /* 1*2*3*4 */

  array_free(a);
  printf(" PASSED\n");
}

static void test_prod_double(void) {
  printf("  test_prod_double...");
  double data[] = {1.5, 2.0, 3.0};
  size_t shape[] = {3};
  ArrayCreate create = {
      .ndim = 1, .shape = shape,
      .numc_type = NUMC_TYPE_DOUBLE, .data = data, .owns_data = true,
  };
  Array *a = array_create(&create);
  assert(a);

  double result = 0.0;
  assert(array_prod(a, &result) == NUMC_OK);
  assert(fabs(result - 9.0) < 1e-10); /* 1.5*2.0*3.0 */

  array_free(a);
  printf(" PASSED\n");
}

static void test_prod_single_element(void) {
  printf("  test_prod_single_element...");
  int data[] = {42};
  size_t shape[] = {1};
  ArrayCreate create = {
      .ndim = 1, .shape = shape,
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true,
  };
  Array *a = array_create(&create);
  assert(a);

  int result = 0;
  assert(array_prod(a, &result) == NUMC_OK);
  assert(result == 42);

  array_free(a);
  printf(" PASSED\n");
}

static void test_prod_with_zero(void) {
  printf("  test_prod_with_zero...");
  int data[] = {5, 3, 0, 7, 2};
  size_t shape[] = {5};
  ArrayCreate create = {
      .ndim = 1, .shape = shape,
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true,
  };
  Array *a = array_create(&create);
  assert(a);

  int result = 1;
  assert(array_prod(a, &result) == NUMC_OK);
  assert(result == 0);

  array_free(a);
  printf(" PASSED\n");
}

static void test_prod_2d(void) {
  printf("  test_prod_2d...");
  int data[] = {1, 2, 3, 4, 5, 6};
  size_t shape[] = {2, 3};
  ArrayCreate create = {
      .ndim = 2, .shape = shape,
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true,
  };
  Array *a = array_create(&create);
  assert(a);

  int result = 0;
  assert(array_prod(a, &result) == NUMC_OK);
  assert(result == 720); /* 1*2*3*4*5*6 */

  array_free(a);
  printf(" PASSED\n");
}

static void test_prod_null(void) {
  printf("  test_prod_null...");
  int result = 0;
  assert(array_prod(NULL, &result) == NUMC_ERR_NULL);

  int data[] = {1, 2, 3};
  size_t shape[] = {3};
  ArrayCreate create = {
      .ndim = 1, .shape = shape,
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true,
  };
  Array *a = array_create(&create);
  assert(array_prod(a, NULL) == NUMC_ERR_NULL);
  array_free(a);
  printf(" PASSED\n");
}

int main(void) {
  printf("Running array_prod tests:\n");
  test_prod_int();
  test_prod_float();
  test_prod_double();
  test_prod_single_element();
  test_prod_with_zero();
  test_prod_2d();
  test_prod_null();
  printf("All array_prod tests passed!\n");
  return 0;
}
