/*
 * bench/numc/bench_csv_reduction.c — Full and axis reduction benchmarks
 */

#include "bench_common.h"

typedef int (*ReduceFullFn)(const NumcArray *, NumcArray *);
typedef int (*ReduceAxisFn)(const NumcArray *, int, int, NumcArray *);

static void bench_reduce_full(const char *name, ReduceFullFn fn, size_t size,
                              int out_int64) {
  for (int d = 0; d < N_DTYPES; d++) {
    NumcDType dt = ALL_DTYPES[d];
    NumcCtx *ctx = numc_ctx_create();
    size_t shape[] = {size};
    char val[8];
    fill_value(dt, val);

    NumcArray *a = numc_array_fill(ctx, shape, 1, dt, val);
    size_t sshape[] = {1};
    NumcDType odt = out_int64 ? NUMC_DTYPE_INT64 : dt;
    NumcArray *out = numc_array_zeros(ctx, sshape, 1, odt);
    if (!a || !out) {
      numc_ctx_free(ctx);
      continue;
    }

    double us;
    BENCH_MIN_LOOP(fn(a, out), BENCH_WARMUP, BENCH_ITERS);

    char shape_str[32];
    snprintf(shape_str, sizeof(shape_str), "(%zu)", size);
    bench_csv("reduction", name, dtype_name(dt), size, shape_str, us);
    numc_ctx_free(ctx);
  }
}

static void bench_reduce_axis(const char *name, ReduceAxisFn fn, int axis,
                              size_t rows, size_t cols, int out_int64) {
  size_t total = rows * cols;
  for (int d = 0; d < N_DTYPES; d++) {
    NumcDType dt = ALL_DTYPES[d];
    NumcCtx *ctx = numc_ctx_create();
    size_t shape[] = {rows, cols};
    char val[8];
    fill_value(dt, val);

    NumcArray *a = numc_array_fill(ctx, shape, 2, dt, val);
    size_t oshape[] = {axis == 0 ? cols : rows};
    NumcDType odt = out_int64 ? NUMC_DTYPE_INT64 : dt;
    NumcArray *out = numc_array_zeros(ctx, oshape, 1, odt);
    if (!a || !out) {
      numc_ctx_free(ctx);
      continue;
    }

    double us;
    BENCH_MIN_LOOP(fn(a, axis, 0, out), BENCH_WARMUP, BENCH_ITERS);

    char op_name[64], shape_str[64];
    snprintf(op_name, sizeof(op_name), "%s_axis%d", name, axis);
    snprintf(shape_str, sizeof(shape_str), "(%zux%zu)", rows, cols);
    bench_csv("reduction", op_name, dtype_name(dt), total, shape_str, us);
    numc_ctx_free(ctx);
  }
}

int main(int argc, char **argv) {
  if (bench_should_print_header(argc, argv))
    bench_csv_header();

  /* Full reductions */
  bench_reduce_full("sum", numc_sum, BENCH_SIZE, 0);
  bench_reduce_full("mean", numc_mean, BENCH_SIZE, 0);
  bench_reduce_full("max", numc_max, BENCH_SIZE, 0);
  bench_reduce_full("min", numc_min, BENCH_SIZE, 0);
  bench_reduce_full("argmax", numc_argmax, BENCH_SIZE, 1);
  bench_reduce_full("argmin", numc_argmin, BENCH_SIZE, 1);

  /* Axis reductions (1000x1000) */
  bench_reduce_axis("sum", numc_sum_axis, 0, 1000, 1000, 0);
  bench_reduce_axis("sum", numc_sum_axis, 1, 1000, 1000, 0);
  bench_reduce_axis("mean", numc_mean_axis, 0, 1000, 1000, 0);
  bench_reduce_axis("mean", numc_mean_axis, 1, 1000, 1000, 0);
  bench_reduce_axis("max", numc_max_axis, 0, 1000, 1000, 0);
  bench_reduce_axis("max", numc_max_axis, 1, 1000, 1000, 0);
  bench_reduce_axis("min", numc_min_axis, 0, 1000, 1000, 0);
  bench_reduce_axis("min", numc_min_axis, 1, 1000, 1000, 0);
  bench_reduce_axis("argmax", numc_argmax_axis, 0, 1000, 1000, 1);
  bench_reduce_axis("argmax", numc_argmax_axis, 1, 1000, 1000, 1);
  bench_reduce_axis("argmin", numc_argmin_axis, 0, 1000, 1000, 1);
  bench_reduce_axis("argmin", numc_argmin_axis, 1, 1000, 1000, 1);

  return 0;
}
