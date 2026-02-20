#include "../helpers.h"

static int test_sub_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {10.0f, 20.0f, 30.0f, 40.0f};
  float db[] = {1.0f, 2.0f, 3.0f, 4.0f};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_sub(a, b, out);
  ASSERT_MSG(err == 0, "sub should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 9.0f && r[1] == 18.0f && r[2] == 27.0f && r[3] == 36.0f,
             "sub results should match");

  numc_ctx_free(ctx);
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== elemwise/test_sub ===\n\n");

  printf("numc_sub:\n");
  RUN_TEST(test_sub_float32);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
