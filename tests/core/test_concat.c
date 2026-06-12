#include "../helpers.h"
#include "numc/array.h"
#include "numc/dtype.h"

static int test_array_concat_err_null(void) {
  int err = numc_array_concat(NULL, 0, 0, NULL);
  ASSERT_MSG(err == -1, "concat should fail");
  return 0;
}

static int test_array_concat_err_dim_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  size_t dim = 2;
  NumcArray *a = numc_array_zeros(ctx, shape, dim, NUMC_DTYPE_INT32);

  size_t shape2[] = {2, 2, 3};
  size_t dim2 = 3;
  NumcArray *b = numc_array_zeros(ctx, shape2, dim2, NUMC_DTYPE_INT32);

  NumcArray *arr[2] = {a, b};

  int ret = numc_array_concat(arr, 2, 0, a);
  ASSERT_MSG(ret == -1, "dim mismatch should fail");
  numc_ctx_free(ctx);
  return 0;
}

static int test_array_concat_err_shape_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  size_t dim = 2;
  NumcArray *a = numc_array_zeros(ctx, shape, dim, NUMC_DTYPE_INT32);

  size_t shape2[] = {2, 2};
  size_t dim2 = 2;
  NumcArray *b = numc_array_zeros(ctx, shape2, dim2, NUMC_DTYPE_INT32);

  size_t out_shape[] = {2, 5};
  NumcArray *out = numc_array_zeros(ctx, out_shape, 2, NUMC_DTYPE_INT32);

  NumcArray *arr[2] = {a, b};
  int ret = numc_array_concat(arr, 2, 0, out);
  ASSERT_MSG(ret == -1, "shape mismatch should fail");
  numc_ctx_free(ctx);
  return 0;
}

static int test_array_concat_err_type_mismatch(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 2};
  size_t dim = 2;
  NumcArray *a = numc_array_zeros(ctx, shape, dim, NUMC_DTYPE_INT32);

  size_t shape2[] = {2, 2};
  size_t dim2 = 2;
  NumcArray *b = numc_array_zeros(ctx, shape2, dim2, NUMC_DTYPE_INT32);

  size_t out_shape[] = {4, 2};
  NumcArray *out = numc_array_zeros(ctx, out_shape, 2, NUMC_DTYPE_INT16);

  NumcArray *arr[2] = {a, b};
  int ret = numc_array_concat(arr, 2, 0, out);
  ASSERT_MSG(ret == -1, "type mismatch should fail");
  numc_ctx_free(ctx);
  return 0;
}


static int test_array_concat_axis(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_INT32);

  size_t shape2[] = {2, 2};
  NumcArray *b = numc_array_zeros(ctx, shape2, 2, NUMC_DTYPE_INT32);

  size_t out_shape[] = {2, 5};
  NumcArray *out = numc_array_zeros(ctx, out_shape, 2, NUMC_DTYPE_INT32);

  NumcArray *arr[2] = {a, b};
  int ret = numc_array_concat(arr, 2, 1, out);
  ASSERT_MSG(ret == 0, "concat should succeed");
  numc_ctx_free(ctx);
  return 0;
}

static int test_array_concat_bad_axis(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_INT32);

  NumcArray *arr[2] = {a, b};
  int ret = numc_array_concat(arr, 2, 0, a);
  ASSERT_MSG(ret == -1, "concat should fail");
  numc_ctx_free(ctx);
  return 0;
}

static int test_array_concat_axis_0(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 2};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  int da[2][2] = {{1, 2}, {3, 4}};
  numc_array_write(a, da);

  NumcArray *b = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  int db[2][2] = {{5, 6}, {7, 8}};
  numc_array_write(b, db);

  size_t out_shape[] = {4, 2};
  NumcArray *out = numc_array_zeros(ctx, out_shape, 2, NUMC_DTYPE_INT32);

  NumcArray *arr[2] = {a, b};
  int ret = numc_array_concat(arr, 2, 0, out);
  ASSERT_MSG(ret == 0, "concat should succeed");

  int *od = numc_array_data(out);
  int expected[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  for (int i = 0; i < 8; i++)
    ASSERT_MSG(od[i] == expected[i], "concat data mismatch");

  numc_ctx_free(ctx);
  return 0;
}

static int test_array_concat_axis_1(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {2, 2};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  int da[2][2] = {{1, 2}, {3, 4}};
  numc_array_write(a, da);

  NumcArray *b = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  int db[2][2] = {{5, 6}, {7, 8}};
  numc_array_write(b, db);

  size_t out_shape[] = {2, 4};
  NumcArray *out = numc_array_zeros(ctx, out_shape, 2, NUMC_DTYPE_INT32);

  NumcArray *arr[2] = {a, b};
  int ret = numc_array_concat(arr, 2, 1, out);
  ASSERT_MSG(ret == 0, "concat should succeed");

  int *od = numc_array_data(out);
  int expected[8] = {1, 2, 5, 6, 3, 4, 7, 8}; /* row0: a|b, row1: a|b */
  for (int i = 0; i < 8; i++)
    ASSERT_MSG(od[i] == expected[i], "axis-1 concat data mismatch");

  numc_ctx_free(ctx);
  return 0;
}

static int test_array_concat_noncontig(void) {
  NumcCtx *ctx = numc_ctx_create();

  /* Two {2,4} parents; column-slice each to a NON-contiguous {2,2} view. */
  size_t shape[] = {2, 4};
  NumcArray *af = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  int da[2][4] = {{10, 11, 12, 13}, {14, 15, 16, 17}};
  numc_array_write(af, da);

  NumcArray *bf = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  int db[2][4] = {{20, 21, 22, 23}, {24, 25, 26, 27}};
  numc_array_write(bf, db);

  /* columns 1..3 -> {2,2}; row stride stays 4, so the view is strided. */
  NumcArray *a =
      NUMC_SLICE(af, .axis = 1, .start = 1, .stop = 3); /* 11,12 / 15,16 */
  NumcArray *b =
      NUMC_SLICE(bf, .axis = 1, .start = 1, .stop = 3); /* 21,22 / 25,26 */
  ASSERT_MSG(!numc_array_is_contiguous(a), "slice view must be non-contiguous");

  size_t out_shape[] = {4, 2};
  NumcArray *out = numc_array_zeros(ctx, out_shape, 2, NUMC_DTYPE_INT32);

  NumcArray *arr[2] = {a, b};
  int ret = numc_array_concat(arr, 2, 0, out);
  ASSERT_MSG(ret == 0, "concat of sliced views should succeed");

  int *od = numc_array_data(out);
  int expected[8] = {11, 12, 15, 16, 21, 22, 25, 26};
  for (int i = 0; i < 8; i++)
    ASSERT_MSG(od[i] == expected[i], "non-contig concat data mismatch");

  numc_ctx_free(ctx);
  return 0;
}

static int concat_dtype_case(NumcCtx *ctx, NumcDType dt) {
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

  size_t out_shape[] = {4, 2};
  NumcArray *out = numc_array_zeros(ctx, out_shape, 2, dt);
  NumcArray *arr[2] = {a, b};
  ASSERT_MSG(numc_array_concat(arr, 2, 0, out) == 0, "concat should succeed");

  unsigned char *od = numc_array_data(out);
  ASSERT_MSG(memcmp(od, abuf, nbytes) == 0, "first half (a) byte mismatch");
  ASSERT_MSG(memcmp(od + nbytes, bbuf, nbytes) == 0,
             "second half (b) byte mismatch");
  return 0;
}

static int test_array_concat_dtype_sweep(void) {
  NumcCtx *ctx = numc_ctx_create();
  NumcDType dts[] = {NUMC_DTYPE_INT8,   NUMC_DTYPE_INT16,  NUMC_DTYPE_INT32,
                     NUMC_DTYPE_INT64,  NUMC_DTYPE_UINT8,  NUMC_DTYPE_UINT16,
                     NUMC_DTYPE_UINT32, NUMC_DTYPE_UINT64, NUMC_DTYPE_FLOAT32,
                     NUMC_DTYPE_FLOAT64};
  for (size_t i = 0; i < sizeof(dts) / sizeof(dts[0]); i++)
    if (concat_dtype_case(ctx, dts[i]) != 0) {
      numc_ctx_free(ctx);
      return 1;
    }
  numc_ctx_free(ctx);
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== core/test_concat ===\n\n");

  printf("Concat:\n");
  RUN_TEST(test_array_concat_err_null);
  RUN_TEST(test_array_concat_err_dim_mismatch);
  RUN_TEST(test_array_concat_bad_axis);
  RUN_TEST(test_array_concat_axis);
  RUN_TEST(test_array_concat_err_shape_mismatch);
  RUN_TEST(test_array_concat_err_type_mismatch);
  RUN_TEST(test_array_concat_axis_0);
  RUN_TEST(test_array_concat_axis_1);
  RUN_TEST(test_array_concat_noncontig);
  RUN_TEST(test_array_concat_dtype_sweep);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
