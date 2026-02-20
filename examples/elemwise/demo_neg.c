#include "../helpers.h"

static void demo_neg(NumcCtx *ctx) {
  section("Neg");

  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_FLOAT32);

  float da[] = {10, -20, 30, 40, 50, 60};
  numc_array_write(a, da);

  printf("a:\n");
  numc_array_print(a);

  label("numc_neg (a)");
  numc_neg(a, out);
  numc_array_print(out);

  label("numc_neg_inplace (a)");
  numc_neg_inplace(a);
  numc_array_print(a);
}

int main(void) {
  NumcCtx *ctx = numc_ctx_create();
  demo_neg(ctx);
  numc_ctx_free(ctx);
  return 0;
}
