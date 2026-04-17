#include "../helpers.h"

#include <stdint.h>

/* -- Basic: int32 labels → float32 one-hot -------------------------- */

static int test_one_hot_basic(void) {
  NumcCtx *ctx = numc_ctx_create();

  int32_t data[] = {0, 2, 1, 3};
  NumcArray *labels =
      numc_array_create(ctx, (size_t[]){4}, 1, NUMC_DTYPE_INT32);
  numc_array_write(labels, data);

  NumcArray *oh = numc_one_hot(ctx, labels, 4, NUMC_DTYPE_FLOAT32);
  ASSERT_MSG_CTX(oh != NULL, "one_hot should succeed", ctx);

  size_t shape[2];
  numc_array_shape(oh, shape);
  ASSERT_MSG_CTX(numc_array_ndim(oh) == 2, "ndim should be 2", ctx);
  ASSERT_MSG_CTX(shape[0] == 4, "rows should be 4", ctx);
  ASSERT_MSG_CTX(shape[1] == 4, "cols should be 4", ctx);

  float *p = (float *)numc_array_data(oh);
  /* Row 0: label=0 → [1, 0, 0, 0] */
  ASSERT_MSG_CTX(p[0] == 1.0f && p[1] == 0.0f && p[2] == 0.0f && p[3] == 0.0f,
                 "row 0", ctx);
  /* Row 1: label=2 → [0, 0, 1, 0] */
  ASSERT_MSG_CTX(p[4] == 0.0f && p[5] == 0.0f && p[6] == 1.0f && p[7] == 0.0f,
                 "row 1", ctx);
  /* Row 2: label=1 → [0, 1, 0, 0] */
  ASSERT_MSG_CTX(p[8] == 0.0f && p[9] == 1.0f && p[10] == 0.0f && p[11] == 0.0f,
                 "row 2", ctx);
  /* Row 3: label=3 → [0, 0, 0, 1] */
  ASSERT_MSG_CTX(p[12] == 0.0f && p[13] == 0.0f && p[14] == 0.0f &&
                     p[15] == 1.0f,
                 "row 3", ctx);

  numc_ctx_free(ctx);
  return 0;
}

/* -- float64 output ------------------------------------------------- */

static int test_one_hot_float64(void) {
  NumcCtx *ctx = numc_ctx_create();

  int64_t data[] = {1, 0};
  NumcArray *labels =
      numc_array_create(ctx, (size_t[]){2}, 1, NUMC_DTYPE_INT64);
  numc_array_write(labels, data);

  NumcArray *oh = numc_one_hot(ctx, labels, 3, NUMC_DTYPE_FLOAT64);
  ASSERT_MSG_CTX(oh != NULL, "f64 one_hot should succeed", ctx);

  double *p = (double *)numc_array_data(oh);
  /* Row 0: label=1 → [0, 1, 0] */
  ASSERT_MSG_CTX(p[0] == 0.0 && p[1] == 1.0 && p[2] == 0.0, "f64 row 0", ctx);
  /* Row 1: label=0 → [1, 0, 0] */
  ASSERT_MSG_CTX(p[3] == 1.0 && p[4] == 0.0 && p[5] == 0.0, "f64 row 1", ctx);

  numc_ctx_free(ctx);
  return 0;
}

/* -- uint8 labels --------------------------------------------------- */

static int test_one_hot_uint8_labels(void) {
  NumcCtx *ctx = numc_ctx_create();

  uint8_t data[] = {0, 9, 5};
  NumcArray *labels =
      numc_array_create(ctx, (size_t[]){3}, 1, NUMC_DTYPE_UINT8);
  numc_array_write(labels, data);

  NumcArray *oh = numc_one_hot(ctx, labels, 10, NUMC_DTYPE_FLOAT32);
  ASSERT_MSG_CTX(oh != NULL, "uint8 one_hot should succeed", ctx);

  float *p = (float *)numc_array_data(oh);
  ASSERT_MSG_CTX(p[0] == 1.0f, "row 0 col 0", ctx);
  ASSERT_MSG_CTX(p[1 * 10 + 9] == 1.0f, "row 1 col 9", ctx);
  ASSERT_MSG_CTX(p[2 * 10 + 5] == 1.0f, "row 2 col 5", ctx);

  /* Verify rest is zero by summing each row */
  for (int r = 0; r < 3; r++) {
    float sum = 0.0f;
    for (int c = 0; c < 10; c++)
      sum += p[r * 10 + c];
    ASSERT_MSG_CTX(sum == 1.0f, "each row should sum to 1", ctx);
  }

  numc_ctx_free(ctx);
  return 0;
}

/* -- Out-of-bounds labels are silently skipped ---------------------- */

static int test_one_hot_oob(void) {
  NumcCtx *ctx = numc_ctx_create();

  int32_t data[] = {0, 5, -1};
  NumcArray *labels =
      numc_array_create(ctx, (size_t[]){3}, 1, NUMC_DTYPE_INT32);
  numc_array_write(labels, data);

  NumcArray *oh = numc_one_hot(ctx, labels, 3, NUMC_DTYPE_FLOAT32);
  ASSERT_MSG_CTX(oh != NULL, "oob one_hot should succeed", ctx);

  float *p = (float *)numc_array_data(oh);
  /* Row 0: label=0 → [1, 0, 0] */
  ASSERT_MSG_CTX(p[0] == 1.0f && p[1] == 0.0f && p[2] == 0.0f, "row 0", ctx);
  /* Row 1: label=5 → out of bounds → [0, 0, 0] */
  ASSERT_MSG_CTX(p[3] == 0.0f && p[4] == 0.0f && p[5] == 0.0f,
                 "oob row stays zero", ctx);
  /* Row 2: label=-1 → negative → [0, 0, 0] */
  ASSERT_MSG_CTX(p[6] == 0.0f && p[7] == 0.0f && p[8] == 0.0f,
                 "negative label stays zero", ctx);

  numc_ctx_free(ctx);
  return 0;
}

/* -- Error: float labels rejected ----------------------------------- */

static int test_one_hot_reject_float_labels(void) {
  NumcCtx *ctx = numc_ctx_create();

  NumcArray *labels =
      numc_array_zeros(ctx, (size_t[]){3}, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *oh = numc_one_hot(ctx, labels, 3, NUMC_DTYPE_FLOAT32);
  ASSERT_MSG_CTX(oh == NULL, "float labels should be rejected", ctx);

  numc_ctx_free(ctx);
  return 0;
}

/* -- Error: 2-D labels rejected ------------------------------------- */

static int test_one_hot_reject_2d(void) {
  NumcCtx *ctx = numc_ctx_create();

  NumcArray *labels =
      numc_array_zeros(ctx, (size_t[]){3, 2}, 2, NUMC_DTYPE_INT32);
  NumcArray *oh = numc_one_hot(ctx, labels, 3, NUMC_DTYPE_FLOAT32);
  ASSERT_MSG_CTX(oh == NULL, "2-D labels should be rejected", ctx);

  numc_ctx_free(ctx);
  return 0;
}

/* -- Error: integer output dtype rejected --------------------------- */

static int test_one_hot_reject_int_output(void) {
  NumcCtx *ctx = numc_ctx_create();

  NumcArray *labels = numc_array_zeros(ctx, (size_t[]){3}, 1, NUMC_DTYPE_INT32);
  NumcArray *oh = numc_one_hot(ctx, labels, 3, NUMC_DTYPE_INT32);
  ASSERT_MSG_CTX(oh == NULL, "int output dtype should be rejected", ctx);

  numc_ctx_free(ctx);
  return 0;
}

/* -- main ----------------------------------------------------------- */

int main(void) {
  int passes = 0, fails = 0;

  RUN_TEST(test_one_hot_basic);
  RUN_TEST(test_one_hot_float64);
  RUN_TEST(test_one_hot_uint8_labels);
  RUN_TEST(test_one_hot_oob);
  RUN_TEST(test_one_hot_reject_float_labels);
  RUN_TEST(test_one_hot_reject_2d);
  RUN_TEST(test_one_hot_reject_int_output);

  printf("\n  %d passed, %d failed\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
