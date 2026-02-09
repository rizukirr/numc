#include <assert.h>
#include <math.h>
#include <numc/numc.h>
#include <stdio.h>

static void test_std_1d(void) {
  printf("  test_std_1d...");
  /* [2, 4, 4, 4, 5, 5, 7, 9]
     mean = 5.0
     deviations: -3, -1, -1, -1, 0, 0, 2, 4
     squared: 9, 1, 1, 1, 0, 0, 4, 16 → sum = 32
     variance = 32/8 = 4.0, std = 2.0 */
  int data[] = {2, 4, 4, 4, 5, 5, 7, 9};
  size_t shape[] = {8};
  ArrayCreate create = {
      .ndim = 1, .shape = shape,
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true,
  };
  Array *a = array_create(&create);
  assert(a);

  Array *s = array_std_axis(a, 0);
  assert(s);
  assert(s->numc_type == NUMC_TYPE_DOUBLE);
  assert(s->size == 1);
  double val = ((double *)s->data)[0];
  assert(fabs(val - 2.0) < 1e-10);

  array_free(s);
  array_free(a);
  printf(" PASSED\n");
}

static void test_std_2d_axis0(void) {
  printf("  test_std_2d_axis0...");
  /* [[1, 5],
      [3, 7]]
     axis=0: std([1,3]) = std([5,7])
     mean([1,3]) = 2, deviations: -1, 1, var = 1, std = 1.0
     mean([5,7]) = 6, deviations: -1, 1, var = 1, std = 1.0 */
  int data[] = {1, 5, 3, 7};
  size_t shape[] = {2, 2};
  ArrayCreate create = {
      .ndim = 2, .shape = shape,
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true,
  };
  Array *a = array_create(&create);
  assert(a);

  Array *s = array_std_axis(a, 0);
  assert(s);
  assert(s->ndim == 1);
  assert(s->size == 2);
  double *out = (double *)s->data;
  assert(fabs(out[0] - 1.0) < 1e-10);
  assert(fabs(out[1] - 1.0) < 1e-10);

  array_free(s);
  array_free(a);
  printf(" PASSED\n");
}

static void test_std_2d_axis1(void) {
  printf("  test_std_2d_axis1...");
  /* [[1, 2, 3],
      [4, 5, 6]]
     axis=1:
     row0: mean=2, deviations: -1,0,1, var=2/3, std=sqrt(2/3)
     row1: mean=5, deviations: -1,0,1, var=2/3, std=sqrt(2/3) */
  int data[] = {1, 2, 3, 4, 5, 6};
  size_t shape[] = {2, 3};
  ArrayCreate create = {
      .ndim = 2, .shape = shape,
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true,
  };
  Array *a = array_create(&create);
  assert(a);

  Array *s = array_std_axis(a, 1);
  assert(s);
  assert(s->ndim == 1);
  assert(s->size == 2);
  double expected = sqrt(2.0 / 3.0);
  double *out = (double *)s->data;
  assert(fabs(out[0] - expected) < 1e-10);
  assert(fabs(out[1] - expected) < 1e-10);

  array_free(s);
  array_free(a);
  printf(" PASSED\n");
}

static void test_std_constant(void) {
  printf("  test_std_constant...");
  /* All same values → std = 0 */
  int data[] = {5, 5, 5, 5};
  size_t shape[] = {4};
  ArrayCreate create = {
      .ndim = 1, .shape = shape,
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true,
  };
  Array *a = array_create(&create);
  assert(a);

  Array *s = array_std_axis(a, 0);
  assert(s);
  assert(fabs(((double *)s->data)[0]) < 1e-10);

  array_free(s);
  array_free(a);
  printf(" PASSED\n");
}

static void test_std_float(void) {
  printf("  test_std_float...");
  /* [1.0, 3.0] → mean=2.0, deviations: -1, 1, var=1, std=1.0 */
  float data[] = {1.0f, 3.0f};
  size_t shape[] = {2};
  ArrayCreate create = {
      .ndim = 1, .shape = shape,
      .numc_type = NUMC_TYPE_FLOAT, .data = data, .owns_data = true,
  };
  Array *a = array_create(&create);
  assert(a);

  Array *s = array_std_axis(a, 0);
  assert(s);
  assert(s->numc_type == NUMC_TYPE_DOUBLE);
  assert(fabs(((double *)s->data)[0] - 1.0) < 1e-10);

  array_free(s);
  array_free(a);
  printf(" PASSED\n");
}

static void test_std_3d(void) {
  printf("  test_std_3d...");
  /* shape (2,2,2): std along axis=1
     [[[1,2],[3,4]],
      [[5,6],[7,8]]]
     axis=1 → shape (2,2)
     slice (0,:,0): [1,3] → mean=2, var=1, std=1.0
     slice (0,:,1): [2,4] → mean=3, var=1, std=1.0
     slice (1,:,0): [5,7] → mean=6, var=1, std=1.0
     slice (1,:,1): [6,8] → mean=7, var=1, std=1.0 */
  int data[] = {1, 2, 3, 4, 5, 6, 7, 8};
  size_t shape[] = {2, 2, 2};
  ArrayCreate create = {
      .ndim = 3, .shape = shape,
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true,
  };
  Array *a = array_create(&create);
  assert(a);

  Array *s = array_std_axis(a, 1);
  assert(s);
  assert(s->ndim == 2);
  assert(s->shape[0] == 2);
  assert(s->shape[1] == 2);
  double *out = (double *)s->data;
  for (size_t i = 0; i < 4; i++) {
    assert(fabs(out[i] - 1.0) < 1e-10);
  }

  array_free(s);
  array_free(a);
  printf(" PASSED\n");
}

static void test_std_null(void) {
  printf("  test_std_null...");
  assert(array_std_axis(NULL, 0) == NULL);
  printf(" PASSED\n");
}

static void test_std_invalid_axis(void) {
  printf("  test_std_invalid_axis...");
  int data[] = {1, 2, 3};
  size_t shape[] = {3};
  ArrayCreate create = {
      .ndim = 1, .shape = shape,
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true,
  };
  Array *a = array_create(&create);
  assert(a);
  assert(array_std_axis(a, 5) == NULL);
  array_free(a);
  printf(" PASSED\n");
}

/* ======================== full reduction ======================== */

static void test_std_full_basic(void) {
  printf("  test_std_full_basic...");
  /* [2, 4, 4, 4, 5, 5, 7, 9] → std = 2.0 */
  int data[] = {2, 4, 4, 4, 5, 5, 7, 9};
  size_t shape[] = {8};
  ArrayCreate create = {
      .ndim = 1, .shape = shape,
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true,
  };
  Array *a = array_create(&create);
  assert(a);
  double result = 0.0;
  assert(array_std(a, &result) == 0);
  assert(fabs(result - 2.0) < 1e-10);
  array_free(a);
  printf(" PASSED\n");
}

static void test_std_full_2d(void) {
  printf("  test_std_full_2d...");
  /* [[1, 5], [3, 7]] as flat [1,5,3,7]
     mean = 4.0, deviations: -3, 1, -1, 3
     var = (9+1+1+9)/4 = 5.0, std = sqrt(5) */
  int data[] = {1, 5, 3, 7};
  size_t shape[] = {2, 2};
  ArrayCreate create = {
      .ndim = 2, .shape = shape,
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true,
  };
  Array *a = array_create(&create);
  assert(a);
  double result = 0.0;
  assert(array_std(a, &result) == 0);
  assert(fabs(result - sqrt(5.0)) < 1e-10);
  array_free(a);
  printf(" PASSED\n");
}

static void test_std_full_constant(void) {
  printf("  test_std_full_constant...");
  int data[] = {5, 5, 5, 5};
  size_t shape[] = {4};
  ArrayCreate create = {
      .ndim = 1, .shape = shape,
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true,
  };
  Array *a = array_create(&create);
  assert(a);
  double result = 0.0;
  assert(array_std(a, &result) == 0);
  assert(fabs(result) < 1e-10);
  array_free(a);
  printf(" PASSED\n");
}

static void test_std_full_null(void) {
  printf("  test_std_full_null...");
  double result;
  assert(array_std(NULL, &result) != 0);
  printf(" PASSED\n");
}

int main(void) {
  printf("Running array_std tests:\n");

  printf(" -- axis reduction --\n");
  test_std_1d();
  test_std_2d_axis0();
  test_std_2d_axis1();
  test_std_constant();
  test_std_float();
  test_std_3d();
  test_std_null();
  test_std_invalid_axis();

  printf(" -- full reduction --\n");
  test_std_full_basic();
  test_std_full_2d();
  test_std_full_constant();
  test_std_full_null();

  printf("All array_std tests passed!\n");
  return 0;
}
