#include "../helpers.h"

static void demo_maximum_minimum(NumcCtx *ctx) {
  section("Maximum / Minimum");

  size_t shape[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_create(ctx, shape, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_FLOAT32);

  float da[] = {1, 5, 3, 8, 2, 7};
  float db[] = {4, 2, 6, 1, 9, 3};
  numc_array_write(a, da);
  numc_array_write(b, db);

  printf("a:\n");
  numc_array_print(a);
  printf("b:\n");
  numc_array_print(b);

  label("numc_maximum (a, b)");
  numc_maximum(a, b, out);
  numc_array_print(out);

  label("numc_minimum (a, b)");
  numc_minimum(a, b, out);
  numc_array_print(out);

  /* int32 */
  label("int32: maximum and minimum");
  NumcArray *i1 = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  NumcArray *i2 = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  NumcArray *iout = numc_array_zeros(ctx, shape, 2, NUMC_DTYPE_INT32);
  int32_t di1[] = {-10, 20, -30, 40, -50, 60};
  int32_t di2[] = {10, -20, 30, -40, 50, -60};
  numc_array_write(i1, di1);
  numc_array_write(i2, di2);
  printf("a:\n");
  numc_array_print(i1);
  printf("b:\n");
  numc_array_print(i2);
  numc_maximum(i1, i2, iout);
  printf("max: ");
  numc_array_print(iout);
  numc_minimum(i1, i2, iout);
  printf("min: ");
  numc_array_print(iout);

  /* Inplace variants */
  label("numc_maximum_inplace (a = max(a, b), int32)");
  NumcArray *ma = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  NumcArray *mb = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  numc_array_write(ma, di1);
  numc_array_write(mb, di2);
  printf("a: ");
  numc_array_print(ma);
  printf("b: ");
  numc_array_print(mb);
  numc_maximum_inplace(ma, mb);
  printf("a: ");
  numc_array_print(ma);

  label("numc_minimum_inplace (a = min(a, b), int32)");
  NumcArray *na = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  NumcArray *nb = numc_array_create(ctx, shape, 2, NUMC_DTYPE_INT32);
  numc_array_write(na, di1);
  numc_array_write(nb, di2);
  printf("a: ");
  numc_array_print(na);
  printf("b: ");
  numc_array_print(nb);
  numc_minimum_inplace(na, nb);
  printf("a: ");
  numc_array_print(na);
}

int main(void) {
  NumcCtx *ctx = numc_ctx_create();
  demo_maximum_minimum(ctx);
  numc_ctx_free(ctx);
  return 0;
}
