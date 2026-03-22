/*
 * bench/numc/bench_csv_ternary.c — FMA and where benchmarks
 */

#include "bench_common.h"

static void bench_fma(size_t size) {
  for (int d = 0; d < N_DTYPES; d++) {
    NumcDType dt = ALL_DTYPES[d];
    NumcCtx *ctx = numc_ctx_create();
    size_t shape[] = {size};
    char val[8];
    fill_value(dt, val);

    NumcArray *a = numc_array_fill(ctx, shape, 1, dt, val);
    NumcArray *b = numc_array_fill(ctx, shape, 1, dt, val);
    NumcArray *c = numc_array_fill(ctx, shape, 1, dt, val);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, dt);
    if (!a || !b || !c || !out) {
      numc_ctx_free(ctx);
      continue;
    }

    double us;
    BENCH_MIN_LOOP(numc_fma(a, b, c, out), BENCH_WARMUP, BENCH_ITERS);

    char shape_str[32];
    snprintf(shape_str, sizeof(shape_str), "(%zu)", size);
    bench_csv("ternary", "fma", dtype_name(dt), size, shape_str, us);
    numc_ctx_free(ctx);
  }
}

static void bench_where(size_t size) {
  for (int d = 0; d < N_DTYPES; d++) {
    NumcDType dt = ALL_DTYPES[d];
    NumcCtx *ctx = numc_ctx_create();
    size_t shape[] = {size};
    char val[8];
    fill_value(dt, val);
    char one[8] = {0};
    *(int8_t *)one = 1;

    NumcArray *cond = numc_array_fill(ctx, shape, 1, NUMC_DTYPE_INT8, one);
    NumcArray *a = numc_array_fill(ctx, shape, 1, dt, val);
    NumcArray *b = numc_array_fill(ctx, shape, 1, dt, val);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, dt);
    if (!cond || !a || !b || !out) {
      numc_ctx_free(ctx);
      continue;
    }

    double us;
    BENCH_MIN_LOOP(numc_where(cond, a, b, out), BENCH_WARMUP, BENCH_ITERS);

    char shape_str[32];
    snprintf(shape_str, sizeof(shape_str), "(%zu)", size);
    bench_csv("ternary", "where", dtype_name(dt), size, shape_str, us);
    numc_ctx_free(ctx);
  }
}

int main(int argc, char **argv) {
  bench_cpu_warmup();
  if (bench_should_print_header(argc, argv))
    bench_csv_header();

  bench_fma(BENCH_SIZE);
  bench_where(BENCH_SIZE);

  return 0;
}
