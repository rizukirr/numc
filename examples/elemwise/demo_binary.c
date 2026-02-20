#include "../helpers.h"

static void demo_math_binary(NumcCtx *ctx) {
  section("Element-wise Binary Ops");

  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_FLOAT32);

  float da[] = {10, 20, 30, 40, 50, 60};
  float db[] = {1, 2, 3, 4, 5, 6};
  numc_array_write(a, da);
  numc_array_write(b, db);

  printf("a:\n");
  numc_array_print(a);
  printf("b:\n");
  numc_array_print(b);

  label("numc_add (a + b)");
  numc_add(a, b, out);
  numc_array_print(out);

  label("numc_sub (a - b)");
  numc_sub(a, b, out);
  numc_array_print(out);

  label("numc_mul (a * b)");
  numc_mul(a, b, out);
  numc_array_print(out);

  label("numc_div (a / b)");
  numc_div(a, b, out);
  numc_array_print(out);
}

int main(void) {
  NumcCtx *ctx = numc_ctx_create();
  demo_math_binary(ctx);
  numc_ctx_free(ctx);
  return 0;
}
