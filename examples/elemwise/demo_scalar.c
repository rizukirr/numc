#include "../helpers.h"

static void demo_math_scalar(NumcCtx *ctx) {
  section("Element-wise Scalar Ops");

  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_FLOAT32);

  float da[] = {10, 20, 30, 40, 50, 60};
  numc_array_write(a, da);

  printf("a:\n");
  numc_array_print(a);

  label("numc_add_scalar (a + 100)");
  numc_add_scalar(a, 100.0, out);
  numc_array_print(out);

  label("numc_sub_scalar (a - 5)");
  numc_sub_scalar(a, 5.0, out);
  numc_array_print(out);

  label("numc_mul_scalar (a * 0.5)");
  numc_mul_scalar(a, 0.5, out);
  numc_array_print(out);

  label("numc_div_scalar (a / 3)");
  numc_div_scalar(a, 3.0, out);
  numc_array_print(out);
}

static void demo_math_scalar_inplace(NumcCtx *ctx) {
  section("Scalar Inplace Ops");

  size_t shape[] = {2, 3};
  float da[] = {10, 20, 30, 40, 50, 60};

  label("numc_add_scalar_inplace (a += 1000)");
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  numc_array_write(a, da);
  numc_add_scalar_inplace(a, 1000.0);
  numc_array_print(a);

  label("numc_sub_scalar_inplace (a -= 5)");
  a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  numc_array_write(a, da);
  numc_sub_scalar_inplace(a, 5.0);
  numc_array_print(a);

  label("numc_mul_scalar_inplace (a *= 2)");
  a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  numc_array_write(a, da);
  numc_mul_scalar_inplace(a, 2.0);
  numc_array_print(a);

  label("numc_div_scalar_inplace (a /= 10)");
  a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  numc_array_write(a, da);
  numc_div_scalar_inplace(a, 10.0);
  numc_array_print(a);
}

int main(void) {
  NumcCtx *ctx = numc_ctx_create();
  demo_math_scalar(ctx);
  demo_math_scalar_inplace(ctx);
  numc_ctx_free(ctx);
  return 0;
}
