/*
 * bench/numc/bench_csv_scalar.c — Scalar and scalar-inplace benchmarks
 */

#include "bench_common.h"

typedef int (*ScalarOp)(const NumcArray *, double, NumcArray *);
typedef int (*ScalarInplace)(NumcArray *, double);

static void bench_scalar_op(const char *name, ScalarOp op, size_t size) {
  for (int d = 0; d < N_DTYPES; d++) {
    NumcDType dt = ALL_DTYPES[d];
    NumcCtx *ctx = numc_ctx_create();
    size_t shape[] = {size};
    char val[8];
    fill_value(dt, val);

    NumcArray *a = numc_array_fill(ctx, shape, 1, dt, val);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, dt);
    if (!a || !out) {
      numc_ctx_free(ctx);
      continue;
    }

    double us;
    BENCH_MIN_LOOP(op(a, 2.0, out), BENCH_WARMUP, BENCH_ITERS);

    char shape_str[32];
    snprintf(shape_str, sizeof(shape_str), "(%zu)", size);
    bench_csv("scalar", name, dtype_name(dt), size, shape_str, us);
    numc_ctx_free(ctx);
  }
}

static void bench_scalar_inplace(const char *name, ScalarInplace op,
                                 size_t size) {
  for (int d = 0; d < N_DTYPES; d++) {
    NumcDType dt = ALL_DTYPES[d];
    NumcCtx *ctx = numc_ctx_create();
    size_t shape[] = {size};
    char val[8];
    fill_value(dt, val);

    NumcArray *a = numc_array_fill(ctx, shape, 1, dt, val);
    if (!a) {
      numc_ctx_free(ctx);
      continue;
    }

    double us;
    BENCH_MIN_LOOP(op(a, 1.01), BENCH_WARMUP, BENCH_ITERS);

    char shape_str[32];
    snprintf(shape_str, sizeof(shape_str), "(%zu)", size);
    bench_csv("scalar_inplace", name, dtype_name(dt), size, shape_str, us);
    numc_ctx_free(ctx);
  }
}

int main(int argc, char **argv) {
  bench_cpu_warmup();
  if (bench_should_print_header(argc, argv))
    bench_csv_header();

  bench_scalar_op("add_scalar", numc_add_scalar, BENCH_SIZE);
  bench_scalar_op("sub_scalar", numc_sub_scalar, BENCH_SIZE);
  bench_scalar_op("mul_scalar", numc_mul_scalar, BENCH_SIZE);
  bench_scalar_op("div_scalar", numc_div_scalar, BENCH_SIZE);

  bench_scalar_inplace("add_scalar_inplace", numc_add_scalar_inplace,
                       BENCH_SIZE);
  bench_scalar_inplace("sub_scalar_inplace", numc_sub_scalar_inplace,
                       BENCH_SIZE);
  bench_scalar_inplace("mul_scalar_inplace", numc_mul_scalar_inplace,
                       BENCH_SIZE);
  bench_scalar_inplace("div_scalar_inplace", numc_div_scalar_inplace,
                       BENCH_SIZE);

  return 0;
}
