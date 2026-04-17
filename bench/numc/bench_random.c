/*
 * bench_random.c — Random number generation benchmark
 *
 * Tests: randn (allocating)
 * Varies: dtype (float32, float64), array size (1M)
 * Reports: avg time (us), throughput (Mops/s)
 */

#include <numc/numc.h>
#include <stdio.h>
#include <string.h>
#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <time.h>
#endif

/* -- Config ---------------------------------------------------------- */

#define WARMUP 5
#define ITERS  50

/* -- Timer ----------------------------------------------------------- */

static double time_us(void) {
#if defined(_WIN32) || defined(_WIN64)
  LARGE_INTEGER freq, cnt;
  QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&cnt);
  return (double)cnt.QuadPart / (double)freq.QuadPart * 1e6;
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
#endif
}

/* -- Helpers --------------------------------------------------------- */

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

/* -- Benchmark: randn ------------------------------------------------ */

static void bench_randn(size_t size) {
  printf("=============================================="
         "===================================\n");
  printf("  RANDN  (%zu elements, %d iters)\n", size, ITERS);
  printf("\n  %-8s %10s %10s\n", "dtype", "time (us)", "Mop/s");
  printf("  ------------------------------\n");

  NumcDType dtypes[] = {NUMC_DTYPE_FLOAT32, NUMC_DTYPE_FLOAT64};
  int ndtypes = sizeof(dtypes) / sizeof(dtypes[0]);

  for (int d = 0; d < ndtypes; d++) {
    NumcDType dt = dtypes[d];
    size_t shape[] = {size};

    /* Warmup */
    for (int i = 0; i < WARMUP; i++) {
      NumcCtx *ctx = numc_ctx_create();
      NumcArray *a = numc_array_randn(ctx, shape, 1, dt);
      numc_ctx_free(ctx);
    }

    double t0 = time_us();
    for (int i = 0; i < ITERS; i++) {
      NumcCtx *ctx = numc_ctx_create();
      NumcArray *a = numc_array_randn(ctx, shape, 1, dt);
      numc_ctx_free(ctx);
    }
    double t1 = time_us();

    double avg_us = (t1 - t0) / ITERS;
    printf("  %-8s %10.2f %10.1f\n", dtype_name(dt), avg_us, size / avg_us);
  }
}

/* -- main ------------------------------------------------------------ */

int main(void) {
  printf("\n  numc random benchmark\n");
  printf("  build: "
#ifdef __clang__
         "clang " __clang_version__
#elif defined(__GNUC__)
         "gcc " __VERSION__
#else
         "unknown"
#endif
#ifdef _OPENMP
         " | OpenMP"
#endif
         "\n");

  bench_randn(1000000);

  printf("\n");
  return 0;
}
