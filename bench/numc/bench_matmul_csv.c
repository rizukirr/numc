/*
 * bench/numc/bench_matmul_csv.c — Matmul-only CSV benchmark
 *
 * Outputs CSV to stdout (same format as bench.c, but matmul only).
 * Uses per-iteration timing with MINIMUM reported (most stable metric).
 * Used by: ./run.sh bench matmul
 */

#include <numc/numc.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <float.h>

static double time_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static const char *dtype_name(NumcDType dt) {
  static const char *names[] = {
      [NUMC_DTYPE_INT8] = "int8",       [NUMC_DTYPE_INT16] = "int16",
      [NUMC_DTYPE_INT32] = "int32",     [NUMC_DTYPE_INT64] = "int64",
      [NUMC_DTYPE_UINT8] = "uint8",     [NUMC_DTYPE_UINT16] = "uint16",
      [NUMC_DTYPE_UINT32] = "uint32",   [NUMC_DTYPE_UINT64] = "uint64",
      [NUMC_DTYPE_FLOAT32] = "float32", [NUMC_DTYPE_FLOAT64] = "float64",
  };
  return names[dt];
}

static int dtype_is_float(NumcDType dt) {
  return dt == NUMC_DTYPE_FLOAT32 || dt == NUMC_DTYPE_FLOAT64;
}

static void csv(const char *cat, const char *op, const char *dt, size_t size,
                const char *shape, double us, double gflops) {
  printf("numc,%s,%s,%s,%zu,%s,%.4f,%.4f\n", cat, op, dt, size, shape, us,
         gflops);
}

static void bench_matmul(size_t M, size_t K, size_t N, int warmup, int iters) {
#ifdef _OPENMP
#pragma omp parallel
  { (void)0; }
#endif

  /* Deterministic seed for reproducible random data */
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

    /* Use random data instead of constant fill to avoid special-case opts */
    NumcArray *a, *b;
    if (dtype_is_float(dt)) {
      a = numc_array_rand(ctx, sha, 2, dt);
      b = numc_array_rand(ctx, shb, 2, dt);
    } else {
      /* For integer types, use fill with value 2 (rand gives full-range bits
       * which can overflow in matmul for large sizes) */
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

    /* Warmup */
    for (int i = 0; i < warmup; i++)
      numc_matmul(a, b, out);

    /* Per-iteration timing: report minimum */
    double min_us = DBL_MAX;
    for (int i = 0; i < iters; i++) {
      double t0 = time_us();
      numc_matmul(a, b, out);
      double elapsed = time_us() - t0;
      if (elapsed < min_us)
        min_us = elapsed;
    }

    /* GFLOPS = 2*M*K*N / time_us / 1e6  (for float; FLOPs for int is same) */
    double flops = 2.0 * (double)M * (double)K * (double)N;
    double gflops = flops / min_us / 1e3; /* us->s = 1e6, /1e9 for G => /1e3 */

    size_t total = M * N;
    char shape_str[64];
    snprintf(shape_str, sizeof(shape_str), "(%zux%zu)@(%zux%zu)", M, K, K, N);
    csv("matmul", "matmul", dtype_name(dt), total, shape_str, min_us, gflops);
    numc_ctx_free(ctx);
  }
}

int main(void) {
  printf("library,category,operation,dtype,size,shape,time_us,gflops\n");

  bench_matmul(64, 64, 64, 200, 2000);
  bench_matmul(128, 128, 128, 100, 500);
  bench_matmul(256, 256, 256, 50, 100);
  bench_matmul(512, 512, 512, 20, 50);
  bench_matmul(1024, 1024, 1024, 10, 20);

  return 0;
}
