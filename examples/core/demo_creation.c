#include "../helpers.h"

static void demo_array_creation(NumcCtx *ctx) {
  section("Array Creation");

  /* numc_array_create — uninitialized */
  label("numc_array_create (2x3 float32, then write data)");
  size_t shape1[] = {2, 3};
  NumcArray *a = numc_array_create(ctx, shape1, 2, NUMC_DTYPE_FLOAT32);
  float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  numc_array_write(a, data);
  numc_array_print(a);

  /* numc_array_zeros */
  label("numc_array_zeros (3x3 int32)");
  size_t shape2[] = {3, 3};
  NumcArray *z = numc_array_zeros(ctx, shape2, 2, NUMC_DTYPE_INT32);
  numc_array_print(z);

  /* numc_array_fill */
  label("numc_array_fill (2x4 float64, filled with 3.14)");
  size_t shape3[] = {2, 4};
  double fill_val = 3.14;
  NumcArray *f = numc_array_fill(ctx, shape3, 2, NUMC_DTYPE_FLOAT64, &fill_val);
  numc_array_print(f);

  /* numc_array_copy */
  label("numc_array_copy (deep copy of the float32 array)");
  NumcArray *c = numc_array_copy(a);
  numc_array_print(c);

  /* numc_array_write — multi-dimensional */
  label("numc_array_write (2x2x4 int32)");
  size_t shape4[] = {2, 2, 4};
  NumcArray *w = numc_array_create(ctx, shape4, 3, NUMC_DTYPE_INT32);
  int32_t data3d[][2][4] = {
      {{1, 2, 3, 4}, {5, 6, 7, 8}},
      {{9, 10, 11, 12}, {13, 14, 15, 16}},
  };
  numc_array_write(w, data3d);
  numc_array_print(w);
}

int main(void) {
  NumcCtx *ctx = numc_ctx_create();
  demo_array_creation(ctx);
  numc_ctx_free(ctx);
  return 0;
}
