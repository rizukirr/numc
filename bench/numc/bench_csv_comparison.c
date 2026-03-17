/*
 * bench/numc/bench_csv_comparison.c — Comparison and comparison-scalar
 * benchmarks
 */

#include "bench_common.h"

typedef int (*BinaryOp)(const NumcArray *, const NumcArray *, NumcArray *);
typedef int (*ScalarOp)(const NumcArray *, double, NumcArray *);

static void bench_comparison(const char *name, BinaryOp op, size_t size) {
  for (int d = 0; d < N_DTYPES; d++) {
    NumcDType dt = ALL_DTYPES[d];
    NumcCtx *ctx = numc_ctx_create();
    size_t shape[] = {size};
    char val[8];
    fill_value(dt, val);

    NumcArray *a = numc_array_fill(ctx, shape, 1, dt, val);
    NumcArray *b = numc_array_fill(ctx, shape, 1, dt, val);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_UINT8);
    if (!a || !b || !out) {
      numc_ctx_free(ctx);
      continue;
    }

    double us;
    BENCH_MIN_LOOP(op(a, b, out), BENCH_WARMUP, BENCH_ITERS);

    char shape_str[32];
    snprintf(shape_str, sizeof(shape_str), "(%zu)", size);
    bench_csv("comparison", name, dtype_name(dt), size, shape_str, us);
    numc_ctx_free(ctx);
  }
}

static void bench_comparison_scalar(const char *name, ScalarOp op,
                                    size_t size) {
  for (int d = 0; d < N_DTYPES; d++) {
    NumcDType dt = ALL_DTYPES[d];
    NumcCtx *ctx = numc_ctx_create();
    size_t shape[] = {size};
    char val[8];
    fill_value(dt, val);

    NumcArray *a = numc_array_fill(ctx, shape, 1, dt, val);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_UINT8);
    if (!a || !out) {
      numc_ctx_free(ctx);
      continue;
    }

    double us;
    BENCH_MIN_LOOP(op(a, 2.0, out), BENCH_WARMUP, BENCH_ITERS);

    char shape_str[32];
    snprintf(shape_str, sizeof(shape_str), "(%zu)", size);
    bench_csv("comparison_scalar", name, dtype_name(dt), size, shape_str, us);
    numc_ctx_free(ctx);
  }
}

int main(int argc, char **argv) {
  if (bench_should_print_header(argc, argv))
    bench_csv_header();

  bench_comparison("eq", numc_eq, BENCH_SIZE);
  bench_comparison("gt", numc_gt, BENCH_SIZE);
  bench_comparison("lt", numc_lt, BENCH_SIZE);
  bench_comparison("ge", numc_ge, BENCH_SIZE);
  bench_comparison("le", numc_le, BENCH_SIZE);

  bench_comparison_scalar("eq_scalar", numc_eq_scalar, BENCH_SIZE);
  bench_comparison_scalar("gt_scalar", numc_gt_scalar, BENCH_SIZE);
  bench_comparison_scalar("lt_scalar", numc_lt_scalar, BENCH_SIZE);
  bench_comparison_scalar("ge_scalar", numc_ge_scalar, BENCH_SIZE);
  bench_comparison_scalar("le_scalar", numc_le_scalar, BENCH_SIZE);

  return 0;
}
