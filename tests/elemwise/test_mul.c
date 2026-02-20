#include "../helpers.h"

static int test_mul_float32(void) {
  NumcCtx *ctx = numc_ctx_create();
  size_t shape[] = {4};
  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  float da[] = {2.0f, 3.0f, 4.0f, 5.0f};
  float db[] = {10.0f, 10.0f, 10.0f, 10.0f};
  numc_array_write(a, da);
  numc_array_write(b, db);

  int err = numc_mul(a, b, out);
  ASSERT_MSG(err == 0, "mul should succeed");

  float *r = (float *)numc_array_data(out);
  ASSERT_MSG(r[0] == 20.0f && r[1] == 30.0f && r[2] == 40.0f && r[3] == 50.0f,
             "mul results should match");

  numc_ctx_free(ctx);
  return 0;
}

int main(void) {
  int passes = 0, fails = 0;
  printf("=== elemwise/test_mul ===\n\n");

  printf("numc_mul:\n");
  RUN_TEST(test_mul_float32);

  printf("\n=== Results: %d passed, %d failed ===\n", passes, fails);
  return fails > 0 ? 1 : 0;
}
