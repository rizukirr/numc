#include "../helpers.h"
#include "numc/array.h"
#include "numc/dtype.h"

/* ------------------------------------------------------------------ */
/* Negative cases                                                     */
/* ------------------------------------------------------------------ */

static int test_array_stack_err_null(void) {
  int err = numc_array_stack(NULL, 0, 0, NULL);
  ASSERT_MSG(err == -1, "stack(NULL) should fail");
  return 0;
}

static int test_array_stack_err_zero_n(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 2};
  NumcArray *a = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_INT32);

  size_t out_shape[] = {1, 2, 2};
  NumcArray *out = numc_array_zeros(ctx, out_shape, 3, NUMC_DTYPE_INT32);

  NumcArray *arr[1] = {a};
  int ret = numc_array_stack(arr, 0, 0, out);
  ASSERT_MSG_CTX(ret == -1, "n == 0 should fail", ctx);
  numc_ctx_free(ctx);
  return 0;
}

static int test_array_stack_err_dim_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 2};
  NumcArray *a = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_INT32);

  size_t shape2[] = {2, 2, 2};
  NumcArray *b = numc_array_zeros(ctx, shape2, 3, NUMC_DTYPE_INT32);

  size_t out_shape[] = {2, 2, 2};
  NumcArray *out = numc_array_zeros(ctx, out_shape, 3, NUMC_DTYPE_INT32);

  NumcArray *arr[2] = {a, b};
  int ret = numc_array_stack(arr, 2, 0, out);
  ASSERT_MSG_CTX(ret == -1, "dim mismatch should fail", ctx);
  numc_ctx_free(ctx);
  return 0;
}

static int test_array_stack_err_shape_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 2};
  NumcArray *a = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_INT32);

  size_t shape2[] = {2, 3};
  NumcArray *b = numc_array_zeros(ctx, shape2, 2, NUMC_DTYPE_INT32);

  size_t out_shape[] = {2, 2, 2};
  NumcArray *out = numc_array_zeros(ctx, out_shape, 3, NUMC_DTYPE_INT32);

  NumcArray *arr[2] = {a, b};
  int ret = numc_array_stack(arr, 2, 0, out);
  ASSERT_MSG_CTX(ret == -1, "shape mismatch should fail", ctx);
  numc_ctx_free(ctx);
  return 0;
}

static int test_array_stack_err_type_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 2};
  NumcArray *a = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_INT32);

  size_t out_shape[] = {2, 2, 2};
  NumcArray *out = numc_array_zeros(ctx, out_shape, 3, NUMC_DTYPE_INT16);

  NumcArray *arr[2] = {a, b};
  int ret = numc_array_stack(arr, 2, 0, out);
  ASSERT_MSG_CTX(ret == -1, "dtype mismatch should fail", ctx);
  numc_ctx_free(ctx);
  return 0;
}

static int test_array_stack_err_bad_axis(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 2};
  NumcArray *a = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_INT32);

  size_t out_shape[] = {2, 2, 2};
  NumcArray *out = numc_array_zeros(ctx, out_shape, 3, NUMC_DTYPE_INT32);

  NumcArray *arr[2] = {a, b};
  /* rank is 2, so the new axis is valid in [0, 2]; axis 3 is out of range */
  int ret = numc_array_stack(arr, 2, 3, out);
  ASSERT_MSG_CTX(ret == -1, "axis > ndim should fail", ctx);
  numc_ctx_free(ctx);
  return 0;
}

static int test_array_stack_err_out_shape(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 2};
  NumcArray *a = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_INT32);

  /* correct stacked shape would be {2,2,2}; this one is wrong on purpose */
  size_t out_shape[] = {3, 2, 2};
  NumcArray *out = numc_array_zeros(ctx, out_shape, 3, NUMC_DTYPE_INT32);

  NumcArray *arr[2] = {a, b};
  int ret = numc_array_stack(arr, 2, 0, out);
  ASSERT_MSG_CTX(ret == -1, "wrong out shape should fail", ctx);
  numc_ctx_free(ctx);
  return 0;
}

/* ------------------------------------------------------------------ */
/* Positive cases                                                     */
/* ------------------------------------------------------------------ */

static int test_array_stack_axis_0(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 2};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  int da[2][2] = {{1, 2}, {3, 4}};
  numc_array_write(a, da);

  NumcArray *b = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  int db[2][2] = {{5, 6}, {7, 8}};
  numc_array_write(b, db);

  size_t out_shape[] = {2, 2, 2}; /* {n, 2, 2} */
  NumcArray *out = numc_array_zeros(ctx, out_shape, 3, NUMC_DTYPE_INT32);

  NumcArray *arr[2] = {a, b};
  int ret = numc_array_stack(arr, 2, 0, out);
  ASSERT_MSG_CTX(ret == 0, "axis-0 stack should succeed", ctx);

  /* out[0] == a, out[1] == b */
  int *od = numc_array_data(out);
  int expected[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  for (int i = 0; i < 8; i++)
    ASSERT_MSG_CTX(od[i] == expected[i], "axis-0 stack data mismatch", ctx);

  numc_ctx_free(ctx);
  return 0;
}

static int test_array_stack_axis_1(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 2};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  int da[2][2] = {{1, 2}, {3, 4}};
  numc_array_write(a, da);

  NumcArray *b = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  int db[2][2] = {{5, 6}, {7, 8}};
  numc_array_write(b, db);

  size_t out_shape[] = {2, 2, 2}; /* {2, n, 2} */
  NumcArray *out = numc_array_zeros(ctx, out_shape, 3, NUMC_DTYPE_INT32);

  NumcArray *arr[2] = {a, b};
  int ret = numc_array_stack(arr, 2, 1, out);
  ASSERT_MSG_CTX(ret == 0, "axis-1 stack should succeed", ctx);

  /* out[0] = {a row0, b row0}, out[1] = {a row1, b row1} */
  int *od = numc_array_data(out);
  int expected[8] = {1, 2, 5, 6, 3, 4, 7, 8};
  for (int i = 0; i < 8; i++)
    ASSERT_MSG_CTX(od[i] == expected[i], "axis-1 stack data mismatch", ctx);

  numc_ctx_free(ctx);
  return 0;
}

static int test_array_stack_axis_last(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 2};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  int da[2][2] = {{1, 2}, {3, 4}};
  numc_array_write(a, da);

  NumcArray *b = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  int db[2][2] = {{5, 6}, {7, 8}};
  numc_array_write(b, db);

  size_t out_shape[] = {2, 2, 2}; /* {2, 2, n}, new axis appended */
  NumcArray *out = numc_array_zeros(ctx, out_shape, 3, NUMC_DTYPE_INT32);

  NumcArray *arr[2] = {a, b};
  int ret = numc_array_stack(arr, 2, 2, out);
  ASSERT_MSG_CTX(ret == 0, "axis-last stack should succeed", ctx);

  /* fully interleaved: out[i][j] = {a[i][j], b[i][j]} */
  int *od = numc_array_data(out);
  int expected[8] = {1, 5, 2, 6, 3, 7, 4, 8};
  for (int i = 0; i < 8; i++)
    ASSERT_MSG_CTX(od[i] == expected[i], "axis-last stack data mismatch", ctx);

  numc_ctx_free(ctx);
  return 0;
}

static int test_array_stack_1d(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {3};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  int da[3] = {1, 2, 3};
  numc_array_write(a, da);

  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  int db[3] = {4, 5, 6};
  numc_array_write(b, db);

  NumcArray *arr[2] = {a, b};

  /* axis=0 -> (2,3): rows are the whole arrays */
  size_t out0_shape[] = {2, 3};
  NumcArray *out0 = numc_array_zeros(ctx, out0_shape, 2, NUMC_DTYPE_INT32);
  ASSERT_MSG_CTX(numc_array_stack(arr, 2, 0, out0) == 0, "1d axis-0 stack", ctx);
  int *o0 = numc_array_data(out0);
  int exp0[6] = {1, 2, 3, 4, 5, 6};
  for (int i = 0; i < 6; i++)
    ASSERT_MSG_CTX(o0[i] == exp0[i], "1d axis-0 data mismatch", ctx);

  /* axis=1 -> (3,2): elements paired -> {1,4},{2,5},{3,6} */
  size_t out1_shape[] = {3, 2};
  NumcArray *out1 = numc_array_zeros(ctx, out1_shape, 2, NUMC_DTYPE_INT32);
  ASSERT_MSG_CTX(numc_array_stack(arr, 2, 1, out1) == 0, "1d axis-1 stack", ctx);
  int *o1 = numc_array_data(out1);
  int exp1[6] = {1, 4, 2, 5, 3, 6};
  for (int i = 0; i < 6; i++)
    ASSERT_MSG_CTX(o1[i] == exp1[i], "1d axis-1 data mismatch", ctx);

  numc_ctx_free(ctx);
  return 0;
}

/* ------------------------------------------------------------------ */
/* Edge cases                                                         */
/* ------------------------------------------------------------------ */

static int test_array_stack_single(void) {
  /* n == 1: result just gains a leading size-1 axis, data unchanged */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 2};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  int da[2][2] = {{1, 2}, {3, 4}};
  numc_array_write(a, da);

  size_t out_shape[] = {1, 2, 2};
  NumcArray *out = numc_array_zeros(ctx, out_shape, 3, NUMC_DTYPE_INT32);

  NumcArray *arr[1] = {a};
  int ret = numc_array_stack(arr, 1, 0, out);
  ASSERT_MSG_CTX(ret == 0, "single-array stack should succeed", ctx);

  int *od = numc_array_data(out);
  int expected[4] = {1, 2, 3, 4};
  for (int i = 0; i < 4; i++)
    ASSERT_MSG_CTX(od[i] == expected[i], "single stack data mismatch", ctx);

  numc_ctx_free(ctx);
  return 0;
}

static int test_array_stack_three(void) {
  /* n == 3 along last axis -> innermost groups of 3 */
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  NumcArray *c = numc_array_create(ctx, shape, 1, NUMC_DTYPE_INT32);
  numc_array_write(a, (int[]){1, 2});
  numc_array_write(b, (int[]){3, 4});
  numc_array_write(c, (int[]){5, 6});

  size_t out_shape[] = {2, 3};
  NumcArray *out = numc_array_zeros(ctx, out_shape, 2, NUMC_DTYPE_INT32);

  NumcArray *arr[3] = {a, b, c};
  int ret = numc_array_stack(arr, 3, 1, out);
  ASSERT_MSG_CTX(ret == 0, "three-array stack should succeed", ctx);

  int *od = numc_array_data(out);
  int expected[6] = {1, 3, 5, 2, 4, 6};
  for (int i = 0; i < 6; i++)
    ASSERT_MSG_CTX(od[i] == expected[i], "three stack data mismatch", ctx);

  numc_ctx_free(ctx);
  return 0;
}

static int test_array_stack_noncontig(void) {
  NumcCtx *ctx = numc_ctx_create();

  /* {2,4} parents; column-slice each to a NON-contiguous {2,2} view. */
  size_t shape[] = {2, 4};
  NumcArray *af = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  int da[2][4] = {{10, 11, 12, 13}, {14, 15, 16, 17}};
  numc_array_write(af, da);

  NumcArray *bf = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  int db[2][4] = {{20, 21, 22, 23}, {24, 25, 26, 27}};
  numc_array_write(bf, db);

  NumcArray *a =
      NUMC_SLICE(af, .axis = 1, .start = 1, .stop = 3); /* 11,12 / 15,16 */
  NumcArray *b =
      NUMC_SLICE(bf, .axis = 1, .start = 1, .stop = 3); /* 21,22 / 25,26 */
  ASSERT_MSG_CTX(!numc_array_is_contiguous(a),
                 "slice view must be non-contiguous", ctx);

  size_t out_shape[] = {2, 2, 2};
  NumcArray *out = numc_array_zeros(ctx, out_shape, 3, NUMC_DTYPE_INT32);

  NumcArray *arr[2] = {a, b};
  int ret = numc_array_stack(arr, 2, 0, out);
  ASSERT_MSG_CTX(ret == 0, "stack of sliced views should succeed", ctx);

  int *od = numc_array_data(out);
  int expected[8] = {11, 12, 15, 16, 21, 22, 25, 26};
  for (int i = 0; i < 8; i++)
    ASSERT_MSG_CTX(od[i] == expected[i], "non-contig stack data mismatch", ctx);

  numc_ctx_free(ctx);
  return 0;
}

static int stack_dtype_case(NumcCtx *ctx, NumcDType dt) {
  size_t shape[] = {2, 2};
  NumcArray *a = numc_array_create(ctx, shape, 2, dt);
  NumcArray *b = numc_array_create(ctx, shape, 2, dt);
  size_t nbytes = numc_array_capacity(a); /* size * elem_size */

  unsigned char abuf[64], bbuf[64];
  for (size_t i = 0; i < nbytes; i++) {
    abuf[i] = (unsigned char)(0x10 + i);
    bbuf[i] = (unsigned char)(0xA0 + i);
  }
  numc_array_write(a, abuf);
  numc_array_write(b, bbuf);

  size_t out_shape[] = {2, 2, 2}; /* axis-0: a block then b block */
  NumcArray *out = numc_array_zeros(ctx, out_shape, 3, dt);
  NumcArray *arr[2] = {a, b};
  ASSERT_MSG(numc_array_stack(arr, 2, 0, out) == 0, "stack should succeed");

  unsigned char *od = numc_array_data(out);
  ASSERT_MSG(memcmp(od, abuf, nbytes) == 0, "first slice (a) byte mismatch");
  ASSERT_MSG(memcmp(od + nbytes, bbuf, nbytes) == 0,
             "second slice (b) byte mismatch");
  return 0;
}

static int test_array_stack_dtype_sweep(void) {
  NumcCtx *ctx = numc_ctx_create();
  NumcDType dts[] = {NUMC_DTYPE_INT8,   NUMC_DTYPE_INT16,  NUMC_DTYPE_INT32,
                     NUMC_DTYPE_INT64,  NUMC_DTYPE_UINT8,  NUMC_DTYPE_UINT16,
                     NUMC_DTYPE_UINT32, NUMC_DTYPE_UINT64, NUMC_DTYPE_FLOAT32,
                     NUMC_DTYPE_FLOAT64};
  for (size_t i = 0; i < sizeof(dts) / sizeof(dts[0]); i++)
    if (stack_dtype_case(ctx, dts[i]) != 0) {
      numc_ctx_free(ctx);
      return 1;
    }
  numc_ctx_free(ctx);
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== core/test_stack ===\n\n");

  printf("Stack (negative):\n");
  RUN_TEST(test_array_stack_err_null);
  RUN_TEST(test_array_stack_err_zero_n);
  RUN_TEST(test_array_stack_err_dim_mismatch);
  RUN_TEST(test_array_stack_err_shape_mismatch);
  RUN_TEST(test_array_stack_err_type_mismatch);
  RUN_TEST(test_array_stack_err_bad_axis);
  RUN_TEST(test_array_stack_err_out_shape);

  printf("\nStack (positive):\n");
  RUN_TEST(test_array_stack_axis_0);
  RUN_TEST(test_array_stack_axis_1);
  RUN_TEST(test_array_stack_axis_last);
  RUN_TEST(test_array_stack_1d);

  printf("\nStack (edge):\n");
  RUN_TEST(test_array_stack_single);
  RUN_TEST(test_array_stack_three);
  RUN_TEST(test_array_stack_noncontig);
  RUN_TEST(test_array_stack_dtype_sweep);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
