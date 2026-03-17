/*
 * bench/numc/bench_csv_random.c — Random number generation benchmarks
 */

#include "bench_common.h"

static void bench_random(const char *name, size_t size,
                         NumcArray *(*gen)(NumcCtx *, const size_t *, size_t,
                                           NumcDType)) {
  for (int d = 0; d < N_FLOAT; d++) {
    NumcDType dt = FLOAT_DTYPES[d];
    size_t shape[] = {size};

    /* Warmup */
    for (int i = 0; i < 5; i++) {
      NumcCtx *ctx = numc_ctx_create();
      gen(ctx, shape, 1, dt);
      numc_ctx_free(ctx);
    }

    double us;
    {
      double _min_us = DBL_MAX;
      for (int _i = 0; _i < 50; _i++) {
        NumcCtx *ctx = numc_ctx_create();
        double _t0 = time_us();
        gen(ctx, shape, 1, dt);
        double _elapsed = time_us() - _t0;
        numc_ctx_free(ctx);
        if (_elapsed < _min_us)
          _min_us = _elapsed;
      }
      us = _min_us;
    }

    char shape_str[32];
    snprintf(shape_str, sizeof(shape_str), "(%zu)", size);
    bench_csv("random", name, dtype_name(dt), size, shape_str, us);
  }
}

int main(int argc, char **argv) {
  if (bench_should_print_header(argc, argv))
    bench_csv_header();

  bench_random("rand", BENCH_SIZE, numc_array_rand);
  bench_random("randn", BENCH_SIZE, numc_array_randn);

  return 0;
}
