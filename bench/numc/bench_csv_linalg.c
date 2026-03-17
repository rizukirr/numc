/*
 * bench/numc/bench_csv_linalg.c — Dot product benchmark
 */

#include "bench_common.h"

static void bench_dot(size_t size) {
  for (int d = 0; d < N_FLOAT; d++) {
    NumcDType dt = FLOAT_DTYPES[d];
    NumcCtx *ctx = numc_ctx_create();
    size_t shape[] = {size};

    NumcArray *a = numc_array_randn(ctx, shape, 1, dt);
    NumcArray *b = numc_array_randn(ctx, shape, 1, dt);
    NumcArray *out = numc_array_zeros(ctx, (size_t[]){1}, 1, dt);
    if (!a || !b || !out) {
      numc_ctx_free(ctx);
      continue;
    }

    double us;
    BENCH_MIN_LOOP(numc_dot(a, b, out), BENCH_WARMUP, 1000);

    char shape_str[32];
    snprintf(shape_str, sizeof(shape_str), "(%zu)", size);
    bench_csv("linalg", "dot", dtype_name(dt), size, shape_str, us);
    numc_ctx_free(ctx);
  }
}

int main(int argc, char **argv) {
  if (bench_should_print_header(argc, argv))
    bench_csv_header();

  bench_dot(BENCH_SIZE);

  return 0;
}
