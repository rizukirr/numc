#include <assert.h>
#include <math.h>
#include <numc/numc.h>
#include <stdio.h>

static void test_mean_1d(void) {
  printf("  test_mean_1d...");
  int data[] = {2, 4, 6, 8, 10};
  size_t shape[] = {5};
  ArrayCreate create = {
      .ndim = 1, .shape = shape,
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true,
  };
  Array *a = array_create(&create);
  assert(a);

  Array *m = array_mean_axis(a, 0);
  assert(m);
  assert(m->numc_type == NUMC_TYPE_DOUBLE);
  assert(m->ndim == 1);
  assert(m->size == 1);
  double val = ((double *)m->data)[0];
  assert(fabs(val - 6.0) < 1e-10);

  array_free(m);
  array_free(a);
  printf(" PASSED\n");
}

static void test_mean_2d_axis0(void) {
  printf("  test_mean_2d_axis0...");
  /* [[1, 2, 3],
      [4, 5, 6]]  → mean(axis=0) = [2.5, 3.5, 4.5] */
  int data[] = {1, 2, 3, 4, 5, 6};
  size_t shape[] = {2, 3};
  ArrayCreate create = {
      .ndim = 2, .shape = shape,
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true,
  };
  Array *a = array_create(&create);
  assert(a);

  Array *m = array_mean_axis(a, 0);
  assert(m);
  assert(m->numc_type == NUMC_TYPE_DOUBLE);
  assert(m->ndim == 1);
  assert(m->size == 3);

  double *out = (double *)m->data;
  assert(fabs(out[0] - 2.5) < 1e-10);
  assert(fabs(out[1] - 3.5) < 1e-10);
  assert(fabs(out[2] - 4.5) < 1e-10);

  array_free(m);
  array_free(a);
  printf(" PASSED\n");
}

static void test_mean_2d_axis1(void) {
  printf("  test_mean_2d_axis1...");
  /* [[1, 2, 3],
      [4, 5, 6]]  → mean(axis=1) = [2.0, 5.0] */
  int data[] = {1, 2, 3, 4, 5, 6};
  size_t shape[] = {2, 3};
  ArrayCreate create = {
      .ndim = 2, .shape = shape,
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true,
  };
  Array *a = array_create(&create);
  assert(a);

  Array *m = array_mean_axis(a, 1);
  assert(m);
  assert(m->numc_type == NUMC_TYPE_DOUBLE);
  assert(m->ndim == 1);
  assert(m->size == 2);

  double *out = (double *)m->data;
  assert(fabs(out[0] - 2.0) < 1e-10);
  assert(fabs(out[1] - 5.0) < 1e-10);

  array_free(m);
  array_free(a);
  printf(" PASSED\n");
}

static void test_mean_3d(void) {
  printf("  test_mean_3d...");
  /* shape (2,3,2): mean along axis=1 → shape (2,2)
     [[[1,2],[3,4],[5,6]],
      [[7,8],[9,10],[11,12]]]
     mean(axis=1):
       row0: [(1+3+5)/3, (2+4+6)/3] = [3.0, 4.0]
       row1: [(7+9+11)/3, (8+10+12)/3] = [9.0, 10.0] */
  int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  size_t shape[] = {2, 3, 2};
  ArrayCreate create = {
      .ndim = 3, .shape = shape,
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true,
  };
  Array *a = array_create(&create);
  assert(a);

  Array *m = array_mean_axis(a, 1);
  assert(m);
  assert(m->numc_type == NUMC_TYPE_DOUBLE);
  assert(m->ndim == 2);
  assert(m->shape[0] == 2);
  assert(m->shape[1] == 2);
  assert(m->size == 4);

  double *out = (double *)m->data;
  assert(fabs(out[0] - 3.0) < 1e-10);
  assert(fabs(out[1] - 4.0) < 1e-10);
  assert(fabs(out[2] - 9.0) < 1e-10);
  assert(fabs(out[3] - 10.0) < 1e-10);

  array_free(m);
  array_free(a);
  printf(" PASSED\n");
}

static void test_mean_float_input(void) {
  printf("  test_mean_float_input...");
  float data[] = {1.5f, 2.5f, 3.5f, 4.5f};
  size_t shape[] = {2, 2};
  ArrayCreate create = {
      .ndim = 2, .shape = shape,
      .numc_type = NUMC_TYPE_FLOAT, .data = data, .owns_data = true,
  };
  Array *a = array_create(&create);
  assert(a);

  /* mean(axis=0) = [(1.5+3.5)/2, (2.5+4.5)/2] = [2.5, 3.5] */
  Array *m = array_mean_axis(a, 0);
  assert(m);
  assert(m->numc_type == NUMC_TYPE_DOUBLE);

  double *out = (double *)m->data;
  assert(fabs(out[0] - 2.5) < 1e-10);
  assert(fabs(out[1] - 3.5) < 1e-10);

  array_free(m);
  array_free(a);
  printf(" PASSED\n");
}

static void test_mean_null(void) {
  printf("  test_mean_null...");
  Array *m = array_mean_axis(NULL, 0);
  assert(m == NULL);
  printf(" PASSED\n");
}

static void test_mean_invalid_axis(void) {
  printf("  test_mean_invalid_axis...");
  int data[] = {1, 2, 3};
  size_t shape[] = {3};
  ArrayCreate create = {
      .ndim = 1, .shape = shape,
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true,
  };
  Array *a = array_create(&create);
  assert(a);

  Array *m = array_mean_axis(a, 5); /* axis out of bounds */
  assert(m == NULL);

  array_free(a);
  printf(" PASSED\n");
}

static void test_mean_3d_axis0(void) {
  printf("  test_mean_3d_axis0...");
  /* shape (2,3,2): mean along axis=0 → shape (3,2)
     [[[1,2],[3,4],[5,6]],
      [[7,8],[9,10],[11,12]]]
     mean(axis=0):
       [(1+7)/2, (2+8)/2] = [4.0, 5.0]
       [(3+9)/2, (4+10)/2] = [6.0, 7.0]
       [(5+11)/2, (6+12)/2] = [8.0, 9.0] */
  int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  size_t shape[] = {2, 3, 2};
  ArrayCreate create = {
      .ndim = 3, .shape = shape,
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true,
  };
  Array *a = array_create(&create);
  assert(a);

  Array *m = array_mean_axis(a, 0);
  assert(m);
  assert(m->ndim == 2);
  assert(m->shape[0] == 3);
  assert(m->shape[1] == 2);

  double *out = (double *)m->data;
  assert(fabs(out[0] - 4.0) < 1e-10);
  assert(fabs(out[1] - 5.0) < 1e-10);
  assert(fabs(out[2] - 6.0) < 1e-10);
  assert(fabs(out[3] - 7.0) < 1e-10);
  assert(fabs(out[4] - 8.0) < 1e-10);
  assert(fabs(out[5] - 9.0) < 1e-10);

  array_free(m);
  array_free(a);
  printf(" PASSED\n");
}

static void test_mean_3d_axis2(void) {
  printf("  test_mean_3d_axis2...");
  /* shape (2,3,2): mean along axis=2 → shape (2,3)
     [[[1,2],[3,4],[5,6]],
      [[7,8],[9,10],[11,12]]]
     mean(axis=2):
       [(1+2)/2, (3+4)/2, (5+6)/2] = [1.5, 3.5, 5.5]
       [(7+8)/2, (9+10)/2, (11+12)/2] = [7.5, 9.5, 11.5] */
  int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  size_t shape[] = {2, 3, 2};
  ArrayCreate create = {
      .ndim = 3, .shape = shape,
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true,
  };
  Array *a = array_create(&create);
  assert(a);

  Array *m = array_mean_axis(a, 2);
  assert(m);
  assert(m->ndim == 2);
  assert(m->shape[0] == 2);
  assert(m->shape[1] == 3);

  double *out = (double *)m->data;
  assert(fabs(out[0] - 1.5) < 1e-10);
  assert(fabs(out[1] - 3.5) < 1e-10);
  assert(fabs(out[2] - 5.5) < 1e-10);
  assert(fabs(out[3] - 7.5) < 1e-10);
  assert(fabs(out[4] - 9.5) < 1e-10);
  assert(fabs(out[5] - 11.5) < 1e-10);

  array_free(m);
  array_free(a);
  printf(" PASSED\n");
}

/* ======================== full reduction ======================== */

static void test_mean_full_int(void) {
  printf("  test_mean_full_int...");
  int data[] = {2, 4, 6, 8, 10};
  size_t shape[] = {5};
  ArrayCreate create = {
      .ndim = 1, .shape = shape,
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true,
  };
  Array *a = array_create(&create);
  assert(a);
  double result = 0.0;
  assert(array_mean(a, &result) == 0);
  assert(fabs(result - 6.0) < 1e-10);
  array_free(a);
  printf(" PASSED\n");
}

static void test_mean_full_2d(void) {
  printf("  test_mean_full_2d...");
  /* mean of all elements: (1+2+3+4+5+6)/6 = 3.5 */
  int data[] = {1, 2, 3, 4, 5, 6};
  size_t shape[] = {2, 3};
  ArrayCreate create = {
      .ndim = 2, .shape = shape,
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true,
  };
  Array *a = array_create(&create);
  assert(a);
  double result = 0.0;
  assert(array_mean(a, &result) == 0);
  assert(fabs(result - 3.5) < 1e-10);
  array_free(a);
  printf(" PASSED\n");
}

static void test_mean_full_float(void) {
  printf("  test_mean_full_float...");
  float data[] = {1.5f, 2.5f, 3.5f, 4.5f};
  size_t shape[] = {4};
  ArrayCreate create = {
      .ndim = 1, .shape = shape,
      .numc_type = NUMC_TYPE_FLOAT, .data = data, .owns_data = true,
  };
  Array *a = array_create(&create);
  assert(a);
  double result = 0.0;
  assert(array_mean(a, &result) == 0);
  assert(fabs(result - 3.0) < 1e-10);
  array_free(a);
  printf(" PASSED\n");
}

static void test_mean_full_null(void) {
  printf("  test_mean_full_null...");
  double result;
  assert(array_mean(NULL, &result) != 0);
  printf(" PASSED\n");
}

int main(void) {
  printf("Running array_mean tests:\n");

  printf(" -- axis reduction --\n");
  test_mean_1d();
  test_mean_2d_axis0();
  test_mean_2d_axis1();
  test_mean_3d();
  test_mean_3d_axis0();
  test_mean_3d_axis2();
  test_mean_float_input();
  test_mean_null();
  test_mean_invalid_axis();

  printf(" -- full reduction --\n");
  test_mean_full_int();
  test_mean_full_2d();
  test_mean_full_float();
  test_mean_full_null();

  printf("All array_mean tests passed!\n");
  return 0;
}
