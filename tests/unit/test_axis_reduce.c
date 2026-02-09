#include <assert.h>
#include <math.h>
#include <numc/numc.h>
#include <stdio.h>

/* Helper: create a 2x3 INT array [[1,2,3],[4,5,6]] */
static Array *make_2x3(void) {
  int data[] = {1, 2, 3, 4, 5, 6};
  size_t shape[] = {2, 3};
  ArrayCreate c = {
      .ndim = 2, .shape = shape,
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true,
  };
  return array_create(&c);
}

/* ======================== sum_axis ======================== */

static void test_sum_axis_2d_axis0(void) {
  printf("  test_sum_axis_2d_axis0...");
  Array *a = make_2x3();
  /* sum(axis=0) = [1+4, 2+5, 3+6] = [5, 7, 9] */
  Array *r = array_sum_axis(a, 0);
  assert(r);
  assert(r->numc_type == NUMC_TYPE_INT);
  assert(r->ndim == 1);
  assert(r->size == 3);
  int *d = (int *)r->data;
  assert(d[0] == 5 && d[1] == 7 && d[2] == 9);
  array_free(r);
  array_free(a);
  printf(" PASSED\n");
}

static void test_sum_axis_2d_axis1(void) {
  printf("  test_sum_axis_2d_axis1...");
  Array *a = make_2x3();
  /* sum(axis=1) = [1+2+3, 4+5+6] = [6, 15] */
  Array *r = array_sum_axis(a, 1);
  assert(r);
  assert(r->ndim == 1);
  assert(r->size == 2);
  int *d = (int *)r->data;
  assert(d[0] == 6 && d[1] == 15);
  array_free(r);
  array_free(a);
  printf(" PASSED\n");
}

static void test_sum_axis_1d(void) {
  printf("  test_sum_axis_1d...");
  int data[] = {10, 20, 30};
  size_t shape[] = {3};
  ArrayCreate c = {
      .ndim = 1, .shape = shape,
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true,
  };
  Array *a = array_create(&c);
  Array *r = array_sum_axis(a, 0);
  assert(r);
  assert(r->size == 1);
  assert(((int *)r->data)[0] == 60);
  array_free(r);
  array_free(a);
  printf(" PASSED\n");
}

static void test_sum_axis_float(void) {
  printf("  test_sum_axis_float...");
  float data[] = {1.5f, 2.5f, 3.5f, 4.5f};
  size_t shape[] = {2, 2};
  ArrayCreate c = {
      .ndim = 2, .shape = shape,
      .numc_type = NUMC_TYPE_FLOAT, .data = data, .owns_data = true,
  };
  Array *a = array_create(&c);
  Array *r = array_sum_axis(a, 0);
  assert(r);
  assert(r->numc_type == NUMC_TYPE_FLOAT);
  float *d = (float *)r->data;
  assert(fabsf(d[0] - 5.0f) < 1e-6f);
  assert(fabsf(d[1] - 7.0f) < 1e-6f);
  array_free(r);
  array_free(a);
  printf(" PASSED\n");
}

/* ======================== prod_axis ======================== */

static void test_prod_axis_2d_axis0(void) {
  printf("  test_prod_axis_2d_axis0...");
  Array *a = make_2x3();
  /* prod(axis=0) = [1*4, 2*5, 3*6] = [4, 10, 18] */
  Array *r = array_prod_axis(a, 0);
  assert(r);
  assert(r->numc_type == NUMC_TYPE_INT);
  int *d = (int *)r->data;
  assert(d[0] == 4 && d[1] == 10 && d[2] == 18);
  array_free(r);
  array_free(a);
  printf(" PASSED\n");
}

static void test_prod_axis_2d_axis1(void) {
  printf("  test_prod_axis_2d_axis1...");
  Array *a = make_2x3();
  /* prod(axis=1) = [1*2*3, 4*5*6] = [6, 120] */
  Array *r = array_prod_axis(a, 1);
  assert(r);
  int *d = (int *)r->data;
  assert(d[0] == 6 && d[1] == 120);
  array_free(r);
  array_free(a);
  printf(" PASSED\n");
}

/* ======================== min_axis ======================== */

static void test_min_axis_2d_axis0(void) {
  printf("  test_min_axis_2d_axis0...");
  Array *a = make_2x3();
  /* min(axis=0) = [min(1,4), min(2,5), min(3,6)] = [1, 2, 3] */
  Array *r = array_min_axis(a, 0);
  assert(r);
  assert(r->numc_type == NUMC_TYPE_INT);
  int *d = (int *)r->data;
  assert(d[0] == 1 && d[1] == 2 && d[2] == 3);
  array_free(r);
  array_free(a);
  printf(" PASSED\n");
}

static void test_min_axis_2d_axis1(void) {
  printf("  test_min_axis_2d_axis1...");
  Array *a = make_2x3();
  /* min(axis=1) = [min(1,2,3), min(4,5,6)] = [1, 4] */
  Array *r = array_min_axis(a, 1);
  assert(r);
  int *d = (int *)r->data;
  assert(d[0] == 1 && d[1] == 4);
  array_free(r);
  array_free(a);
  printf(" PASSED\n");
}

static void test_min_axis_with_negatives(void) {
  printf("  test_min_axis_with_negatives...");
  int data[] = {3, -1, 2, -5, 4, 0};
  size_t shape[] = {2, 3};
  ArrayCreate c = {
      .ndim = 2, .shape = shape,
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true,
  };
  Array *a = array_create(&c);
  /* min(axis=0) = [min(3,-5), min(-1,4), min(2,0)] = [-5, -1, 0] */
  Array *r = array_min_axis(a, 0);
  assert(r);
  int *d = (int *)r->data;
  assert(d[0] == -5 && d[1] == -1 && d[2] == 0);
  array_free(r);
  array_free(a);
  printf(" PASSED\n");
}

/* ======================== max_axis ======================== */

static void test_max_axis_2d_axis0(void) {
  printf("  test_max_axis_2d_axis0...");
  Array *a = make_2x3();
  /* max(axis=0) = [max(1,4), max(2,5), max(3,6)] = [4, 5, 6] */
  Array *r = array_max_axis(a, 0);
  assert(r);
  assert(r->numc_type == NUMC_TYPE_INT);
  int *d = (int *)r->data;
  assert(d[0] == 4 && d[1] == 5 && d[2] == 6);
  array_free(r);
  array_free(a);
  printf(" PASSED\n");
}

static void test_max_axis_2d_axis1(void) {
  printf("  test_max_axis_2d_axis1...");
  Array *a = make_2x3();
  /* max(axis=1) = [max(1,2,3), max(4,5,6)] = [3, 6] */
  Array *r = array_max_axis(a, 1);
  assert(r);
  int *d = (int *)r->data;
  assert(d[0] == 3 && d[1] == 6);
  array_free(r);
  array_free(a);
  printf(" PASSED\n");
}

/* ======================== 3D tests ======================== */

static void test_sum_axis_3d(void) {
  printf("  test_sum_axis_3d...");
  /* shape (2,2,3): sum along axis=1
     [[[1,2,3],[4,5,6]],
      [[7,8,9],[10,11,12]]]
     sum(axis=1) → shape (2,3)
     [[1+4, 2+5, 3+6], [7+10, 8+11, 9+12]] = [[5,7,9],[17,19,21]] */
  int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  size_t shape[] = {2, 2, 3};
  ArrayCreate c = {
      .ndim = 3, .shape = shape,
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true,
  };
  Array *a = array_create(&c);
  Array *r = array_sum_axis(a, 1);
  assert(r);
  assert(r->ndim == 2);
  assert(r->shape[0] == 2 && r->shape[1] == 3);
  int *d = (int *)r->data;
  assert(d[0] == 5 && d[1] == 7 && d[2] == 9);
  assert(d[3] == 17 && d[4] == 19 && d[5] == 21);
  array_free(r);
  array_free(a);
  printf(" PASSED\n");
}

/* ======================== error tests ======================== */

static void test_axis_reduce_null(void) {
  printf("  test_axis_reduce_null...");
  assert(array_sum_axis(NULL, 0) == NULL);
  assert(array_prod_axis(NULL, 0) == NULL);
  assert(array_min_axis(NULL, 0) == NULL);
  assert(array_max_axis(NULL, 0) == NULL);
  printf(" PASSED\n");
}

static void test_axis_reduce_invalid_axis(void) {
  printf("  test_axis_reduce_invalid_axis...");
  int data[] = {1, 2, 3};
  size_t shape[] = {3};
  ArrayCreate c = {
      .ndim = 1, .shape = shape,
      .numc_type = NUMC_TYPE_INT, .data = data, .owns_data = true,
  };
  Array *a = array_create(&c);
  assert(array_sum_axis(a, 5) == NULL);
  assert(array_min_axis(a, 2) == NULL);
  array_free(a);
  printf(" PASSED\n");
}

int main(void) {
  printf("Running axis reduction tests:\n");
  test_sum_axis_2d_axis0();
  test_sum_axis_2d_axis1();
  test_sum_axis_1d();
  test_sum_axis_float();
  test_prod_axis_2d_axis0();
  test_prod_axis_2d_axis1();
  test_min_axis_2d_axis0();
  test_min_axis_2d_axis1();
  test_min_axis_with_negatives();
  test_max_axis_2d_axis0();
  test_max_axis_2d_axis1();
  test_sum_axis_3d();
  test_axis_reduce_null();
  test_axis_reduce_invalid_axis();
  printf("All axis reduction tests passed!\n");
  return 0;
}
