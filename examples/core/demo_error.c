#include "../helpers.h"

static void demo_error(NumcCtx *ctx) {
  section("Error Handling");

  label("shape mismatch (add 2x3 + 3x2)");
  size_t s1[] = {2, 3}, s2[] = {3, 2};
  NumcArray *a = numc_array_zeros(ctx, s1, 2, NUMC_DTYPE_INT32);
  NumcArray *b = numc_array_zeros(ctx, s2, 2, NUMC_DTYPE_INT32);
  NumcArray *out = numc_array_zeros(ctx, s1, 2, NUMC_DTYPE_INT32);
  int err = numc_add(a, b, out);
  printf("numc_add returned: %d (NUMC_ERR_SHAPE = %d)\n", err, NUMC_ERR_SHAPE);

  label("dtype mismatch (add int32 + float32)");
  NumcArray *c = numc_array_zeros(ctx, s1, 2, NUMC_DTYPE_FLOAT32);
  err = numc_add(a, c, out);
  printf("numc_add returned: %d (NUMC_ERR_TYPE = %d)\n", err, NUMC_ERR_TYPE);

  label("null pointer");
  err = numc_add(NULL, b, out);
  printf("numc_add returned: %d (NUMC_ERR_NULL = %d)\n", err, NUMC_ERR_NULL);

  label("numc_set_error / numc_get_error");
  numc_set_error(-99, "custom error message");
  NumcError e = numc_get_error();
  printf("code: %d, msg: \"%s\"\n", e.code, e.msg);
}

int main(void) {
  NumcCtx *ctx = numc_ctx_create();
  demo_error(ctx);
  numc_ctx_free(ctx);
  return 0;
}
