/*
 * bench/numc/bench_csv_unary.c — Unary, unary-inplace, and clip benchmarks
 */

#include "bench_common.h"

typedef int (*UnaryOp)(NumcArray *, NumcArray *);
typedef int (*UnaryInplace)(NumcArray *);

static void bench_unary(const char *name, UnaryOp op, size_t size,
                        int skip_unsigned, int use_exp_fill) {
  for (int d = 0; d < N_DTYPES; d++) {
    NumcDType dt = ALL_DTYPES[d];
    if (skip_unsigned && dtype_is_unsigned(dt))
      continue;

    NumcCtx *ctx = numc_ctx_create();
    size_t shape[] = {size};
    char val[8];
    if (use_exp_fill)
      fill_value_exp(dt, val);
    else
      fill_value(dt, val);

    NumcArray *a = numc_array_fill(ctx, shape, 1, dt, val);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, dt);
    if (!a || !out) {
      numc_ctx_free(ctx);
      continue;
    }

    double us;
    BENCH_MIN_LOOP(op(a, out), BENCH_WARMUP, BENCH_ITERS);

    char shape_str[32];
    snprintf(shape_str, sizeof(shape_str), "(%zu)", size);
    bench_csv("unary", name, dtype_name(dt), size, shape_str, us);
    numc_ctx_free(ctx);
  }
}

static void bench_unary_inplace(const char *name, UnaryInplace op, size_t size,
                                int skip_unsigned, int use_exp_fill) {
  for (int d = 0; d < N_DTYPES; d++) {
    NumcDType dt = ALL_DTYPES[d];
    if (skip_unsigned && dtype_is_unsigned(dt))
      continue;

    NumcCtx *ctx = numc_ctx_create();
    size_t shape[] = {size};
    char val[8];
    if (use_exp_fill)
      fill_value_exp(dt, val);
    else
      fill_value(dt, val);

    NumcArray *a = numc_array_fill(ctx, shape, 1, dt, val);
    if (!a) {
      numc_ctx_free(ctx);
      continue;
    }

    double us;
    BENCH_MIN_LOOP(op(a), BENCH_WARMUP, BENCH_ITERS);

    char shape_str[32];
    snprintf(shape_str, sizeof(shape_str), "(%zu)", size);
    bench_csv("unary_inplace", name, dtype_name(dt), size, shape_str, us);
    numc_ctx_free(ctx);
  }
}

static void bench_clip(size_t size) {
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
    BENCH_MIN_LOOP(numc_clip(a, out, 1.0, 5.0), BENCH_WARMUP, BENCH_ITERS);

    char shape_str[32];
    snprintf(shape_str, sizeof(shape_str), "(%zu)", size);
    bench_csv("unary", "clip", dtype_name(dt), size, shape_str, us);
    numc_ctx_free(ctx);
  }
}

int main(int argc, char **argv) {
  if (bench_should_print_header(argc, argv))
    bench_csv_header();

  /* Unary ops */
  bench_unary("neg", numc_neg, BENCH_SIZE, 0, 0);
  bench_unary("abs", numc_abs, BENCH_SIZE, 1, 0);
  bench_unary("log", numc_log, BENCH_SIZE, 0, 0);
  bench_unary("exp", numc_exp, BENCH_SIZE, 0, 1);
  bench_unary("sqrt", numc_sqrt, BENCH_SIZE, 0, 0);
  bench_clip(BENCH_SIZE);

  /* Unary inplace */
  bench_unary_inplace("neg_inplace", numc_neg_inplace, BENCH_SIZE, 0, 0);
  bench_unary_inplace("abs_inplace", numc_abs_inplace, BENCH_SIZE, 1, 0);
  bench_unary_inplace("log_inplace", numc_log_inplace, BENCH_SIZE, 0, 0);
  bench_unary_inplace("exp_inplace", numc_exp_inplace, BENCH_SIZE, 0, 1);
  bench_unary_inplace("sqrt_inplace", numc_sqrt_inplace, BENCH_SIZE, 0, 0);

  return 0;
}
