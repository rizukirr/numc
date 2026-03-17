/*
 * bench/numc/bench_csv_binary.c — Binary element-wise + pow benchmarks
 */

#include "bench_common.h"

typedef int (*BinaryOp)(const NumcArray *, const NumcArray *, NumcArray *);

static void bench_binary(const char *name, BinaryOp op, size_t size) {
  for (int d = 0; d < N_DTYPES; d++) {
    NumcDType dt = ALL_DTYPES[d];
    NumcCtx *ctx = numc_ctx_create();
    size_t shape[] = {size};
    char val[8];
    fill_value(dt, val);

    NumcArray *a = numc_array_fill(ctx, shape, 1, dt, val);
    NumcArray *b = numc_array_fill(ctx, shape, 1, dt, val);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, dt);
    if (!a || !b || !out) {
      numc_ctx_free(ctx);
      continue;
    }

    double us;
    BENCH_MIN_LOOP(op(a, b, out), BENCH_WARMUP, BENCH_ITERS);

    char shape_str[32];
    snprintf(shape_str, sizeof(shape_str), "(%zu)", size);
    bench_csv("binary", name, dtype_name(dt), size, shape_str, us);
    numc_ctx_free(ctx);
  }
}

static void bench_pow(size_t size) {
  for (int d = 0; d < N_DTYPES; d++) {
    NumcDType dt = ALL_DTYPES[d];
    NumcCtx *ctx = numc_ctx_create();
    size_t shape[] = {size};
    char vbase[8], vexp[8];
    fill_value(dt, vbase);
    fill_pow_exp(dt, vexp);

    NumcArray *a = numc_array_fill(ctx, shape, 1, dt, vbase);
    NumcArray *b = numc_array_fill(ctx, shape, 1, dt, vexp);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, dt);
    if (!a || !b || !out) {
      numc_ctx_free(ctx);
      continue;
    }

    double us;
    BENCH_MIN_LOOP(numc_pow(a, b, out), BENCH_WARMUP, BENCH_ITERS);

    char shape_str[32];
    snprintf(shape_str, sizeof(shape_str), "(%zu)", size);
    bench_csv("binary", "pow", dtype_name(dt), size, shape_str, us);
    numc_ctx_free(ctx);
  }
}

int main(int argc, char **argv) {
  if (bench_should_print_header(argc, argv))
    bench_csv_header();

  bench_binary("add", numc_add, BENCH_SIZE);
  bench_binary("sub", numc_sub, BENCH_SIZE);
  bench_binary("mul", numc_mul, BENCH_SIZE);
  bench_binary("div", numc_div, BENCH_SIZE);
  bench_binary("maximum", numc_maximum, BENCH_SIZE);
  bench_binary("minimum", numc_minimum, BENCH_SIZE);
  bench_pow(BENCH_SIZE);

  return 0;
}
