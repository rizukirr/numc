#include "../helpers.h"

static void demo_properties(NumcCtx *ctx) {
  section("Properties");

  size_t shape[] = {2, 3, 4};
  float val = 1.0f;
  NumcArray *a = numc_array_fill(ctx, shape, 3, NUMC_DTYPE_FLOAT32, &val);

  printf("ndim:      %zu\n", numc_array_ndim(a));
  printf("size:      %zu\n", numc_array_size(a));
  printf("capacity:  %zu\n", numc_array_capacity(a));
  printf("elem_size: %zu\n", numc_array_elem_size(a));
  printf("dtype:     %d (NUMC_DTYPE_FLOAT32 = %d)\n", numc_array_dtype(a),
         NUMC_DTYPE_FLOAT32);

  size_t ndim = numc_array_ndim(a);
  size_t s[ndim], st[ndim];
  numc_array_shape(a, s);
  numc_array_strides(a, st);

  printf("shape:     [");
  for (size_t i = 0; i < ndim; i++)
    printf("%zu%s", s[i], i + 1 < ndim ? ", " : "");
  printf("]\n");

  printf("strides:   [");
  for (size_t i = 0; i < ndim; i++)
    printf("%zu%s", st[i], i + 1 < ndim ? ", " : "");
  printf("] (bytes)\n");

  printf("data ptr:  %p\n", numc_array_data(a));
  printf("contiguous: %s\n", numc_array_is_contiguous(a) ? "true" : "false");
}

int main(void) {
  NumcCtx *ctx = numc_ctx_create();
  demo_properties(ctx);
  numc_ctx_free(ctx);
  return 0;
}
