#include "../helpers.h"
#include "numc/math.h"
#include <numc/numc.h>

static void demo_matmul(NumcCtx *ctx) {
  section("Matrix Multiplication");

  /* Create two matrices */
  size_t shape1[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape1, 2, NUMC_DTYPE_FLOAT32);
  float data1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  numc_array_write(a, data1);

  size_t shape2[] = {3, 2};
  NumcArray *b = numc_array_create(ctx, shape2, 2, NUMC_DTYPE_FLOAT32);
  float data2[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  numc_array_write(b, data2);

  printf("a:\n");
  numc_array_print(a);
  printf("b:\n");
  numc_array_print(b);

  size_t shape3[] = {2, 2};
  NumcArray *out = numc_array_zeros(ctx, shape3, 2, NUMC_DTYPE_FLOAT32);

  /* Perform matrix multiplication */
  label("Perform matrix multiplication");
  numc_matmul_naive(a, b, out);
  numc_array_print(out);
}

int main(void) {
  NumcCtx *ctx = numc_ctx_create();
  demo_matmul(ctx);
  numc_ctx_free(ctx);
  return 0;
}
