/*
 * bench/numc/bench_csv_matmul.c — Matrix multiplication benchmarks
 */

#include "bench_common.h"

static void bench_matmul(size_t M, size_t K, size_t N, int warmup, int iters) {
  /* Re-warm OMP thread pool */
#ifdef _OPENMP
#pragma omp parallel
  { (void)0; }
#endif

  numc_manual_seed(42);

  NumcDType dtypes[] = {
      NUMC_DTYPE_FLOAT32, NUMC_DTYPE_FLOAT64, NUMC_DTYPE_INT8,
      NUMC_DTYPE_INT16,   NUMC_DTYPE_INT32,   NUMC_DTYPE_INT64,
      NUMC_DTYPE_UINT8,   NUMC_DTYPE_UINT16,  NUMC_DTYPE_UINT32,
      NUMC_DTYPE_UINT64,
  };

  for (int d = 0; d < 10; d++) {
    NumcDType dt = dtypes[d];
    NumcCtx *ctx = numc_ctx_create();
    size_t sha[] = {M, K}, shb[] = {K, N}, sho[] = {M, N};

    NumcArray *a, *b;
    if (dt == NUMC_DTYPE_FLOAT32 || dt == NUMC_DTYPE_FLOAT64) {
      a = numc_array_rand(ctx, sha, 2, dt);
      b = numc_array_rand(ctx, shb, 2, dt);
    } else {
      char val[8] = {0};
      *(int32_t *)val = 2;
      a = numc_array_fill(ctx, sha, 2, dt, val);
      b = numc_array_fill(ctx, shb, 2, dt, val);
    }
    NumcArray *out = numc_array_zeros(ctx, sho, 2, dt);
    if (!a || !b || !out) {
      numc_ctx_free(ctx);
      continue;
    }

    for (int i = 0; i < warmup; i++)
      numc_matmul(a, b, out);

    double min_us = DBL_MAX;
    for (int i = 0; i < iters; i++) {
      double t0 = time_us();
      numc_matmul(a, b, out);
      double elapsed = time_us() - t0;
      if (elapsed < min_us)
        min_us = elapsed;
    }

    size_t total = M * N;
    char shape_str[64];
    snprintf(shape_str, sizeof(shape_str), "(%zux%zu)@(%zux%zu)", M, K, K, N);
    bench_csv("matmul", "matmul", dtype_name(dt), total, shape_str, min_us);
    numc_ctx_free(ctx);
  }
}

int main(int argc, char **argv) {
  bench_cpu_warmup();
  if (bench_should_print_header(argc, argv))
    bench_csv_header();

  bench_matmul(64, 64, 64, 200, 2000);
  bench_matmul(128, 128, 128, 100, 500);
  bench_matmul(256, 256, 256, 50, 100);
  bench_matmul(512, 512, 512, 20, 50);
  bench_matmul(1024, 1024, 1024, 10, 20);

  return 0;
}
