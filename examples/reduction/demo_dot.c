#include "../helpers.h"
#include <numc/numc.h>
#include <stdio.h>

static void demo_dot(NumcCtx *ctx) {
  section("N-Dimensional Dot Product");

  /* 1. 1D-1D Inner Product */
  label("1D-1D Vector Inner Product");
  float d1[] = {1.0f, 2.0f, 3.0f};
  float d2[] = {4.0f, 5.0f, 6.0f};
  NumcArray *a1 = numc_array_create(ctx, (size_t[]){3}, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *b1 = numc_array_create(ctx, (size_t[]){3}, 1, NUMC_DTYPE_FLOAT32);
  numc_array_write(a1, d1);
  numc_array_write(b1, d2);
  NumcArray *out1 = numc_array_zeros(ctx, (size_t[]){1}, 1, NUMC_DTYPE_FLOAT32);
  numc_dot(a1, b1, out1);
  printf("a: ");
  numc_array_print(a1);
  printf("b: ");
  numc_array_print(b1);
  printf("a . b = ");
  numc_array_print(out1);

  /* 2. 2D-2D Matrix Multiply */
  label("2D-2D Matrix Multiplication (equivalent to matmul)");
  float m1[] = {1, 2, 3, 4, 5, 6};    // 2x3
  float m2[] = {7, 8, 9, 10, 11, 12}; // 3x2
  NumcArray *a2 =
      numc_array_create(ctx, (size_t[]){2, 3}, 2, NUMC_DTYPE_FLOAT32);
  NumcArray *b2 =
      numc_array_create(ctx, (size_t[]){3, 2}, 2, NUMC_DTYPE_FLOAT32);
  numc_array_write(a2, m1);
  numc_array_write(b2, m2);
  NumcArray *out2 =
      numc_array_create(ctx, (size_t[]){2, 2}, 2, NUMC_DTYPE_FLOAT32);
  numc_dot(a2, b2, out2);
  printf("A:\n");
  numc_array_print(a2);
  printf("B:\n");
  numc_array_print(b2);
  printf("A . B =\n");
  numc_array_print(out2);

  /* 3. ND-1D Tensor-Vector */
  label("ND-1D Tensor-Vector Product (2x3 . 3 -> 2)");
  float v1[] = {1, 1, 1};
  NumcArray *b3 = numc_array_create(ctx, (size_t[]){3}, 1, NUMC_DTYPE_FLOAT32);
  numc_array_write(b3, v1);
  NumcArray *out3 =
      numc_array_create(ctx, (size_t[]){2}, 1, NUMC_DTYPE_FLOAT32);
  numc_dot(a2, b3, out3);
  printf("Tensor A (2x3):\n");
  numc_array_print(a2);
  printf("Vector v (3): ");
  numc_array_print(b3);
  printf("A . v = ");
  numc_array_print(out3);

  /* 4. Scalar Case */
  label("0D Scalar . ND Product");
  float val = 10.0f;
  NumcArray *a4 = numc_array_fill(ctx, NULL, 0, NUMC_DTYPE_FLOAT32, &val);
  NumcArray *out4 =
      numc_array_create(ctx, (size_t[]){2, 3}, 2, NUMC_DTYPE_FLOAT32);
  numc_dot(a4, a2, out4);
  printf("Scalar s: ");
  numc_array_print(a4);
  printf("Tensor A (2x3):\n");
  numc_array_print(a2);
  printf("s . A =\n");
  numc_array_print(out4);
}

int main() {
  NumcCtx *ctx = numc_ctx_create();
  demo_dot(ctx);
  numc_ctx_free(ctx);
  return 0;
}
