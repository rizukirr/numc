#include <numc/numc.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

int main() {
  NumcCtx *ctx = numc_ctx_create();
  assert(ctx != NULL);

  size_t shape[] = {4};
  float data_a[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float data_b[] = {2.0f, 3.0f, 4.0f, 5.0f};
  float data_c[] = {10.0f, 20.0f, 30.0f, 40.0f};

  NumcArray *a = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *c = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_create(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

  numc_array_write(a, data_a);
  numc_array_write(b, data_b);
  numc_array_write(c, data_c);

  int err = numc_fma(a, b, c, out);
  assert(err == 0);

  float *out_ptr = (float *)numc_array_data(out);
  for (int i = 0; i < 4; i++) {
    float expected = data_a[i] * data_b[i] + data_c[i];
    assert(fabsf(out_ptr[i] - expected) < 1e-6f);
  }

  printf("FMA test passed!\n");
  numc_ctx_free(ctx);
  return 0;
}
