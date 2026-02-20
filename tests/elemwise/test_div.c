#include "../helpers.h"

static int test_div_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {20.0f, 30.0f, 40.0f, 50.0f};
  float db[] = {10.0f, 10.0f, 10.0f, 10.0f};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_div(a, b, out);
  ASSERT_MSG(err == 0, "div should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 2.0f && r[1] == 3.0f && r[2] == 4.0f && r[3] == 5.0f,
             "div results should match");

  numc_ctx_free(ctx);
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== elemwise/test_div ===\n\n");

  printf("numc_div:\n");
  RUN_TEST(test_div_float32);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
