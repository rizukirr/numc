/*
 * Minimal matmul benchmark for perf profiling.
 * Usage: ./perf_matmul <dtype> <size>
 *   dtype: f32 or f64
 *   size:  matrix dimension (square)
 */
#include <numc/numc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <time.h>
#endif

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

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage: %s <f32|f64> <size>\n", argv[0]);
    return 1;
  }

  NumcDType dt;
  if (strcmp(argv[1], "f32") == 0)
    dt = NUMC_DTYPE_FLOAT32;
  else if (strcmp(argv[1], "f64") == 0)
    dt = NUMC_DTYPE_FLOAT64;
  else {
    fprintf(stderr, "Unknown dtype: %s\n", argv[1]);
    return 1;
  }

  size_t N = (size_t)atoi(argv[2]);
  /* More iterations for smaller sizes */
  int iters;
  if (N <= 64)
    iters = 5000;
  else if (N <= 128)
    iters = 1000;
  else if (N <= 256)
    iters = 200;
  else if (N <= 512)
    iters = 50;
  else
    iters = 20;

  int warmup = iters / 5;
  if (warmup < 2)
    warmup = 2;

#ifdef _OPENMP
#pragma omp parallel
  { (void)0; }
#endif

  NumcCtx *ctx = numc_ctx_create();
  size_t sha[] = {N, N}, shb[] = {N, N}, sho[] = {N, N};
  char val[8] = {0};
  if (dt == NUMC_DTYPE_FLOAT32)
    *(float *)val = 1.0f;
  else
    *(double *)val = 1.0;

  NumcArray *a = numc_array_fill(ctx, sha, 2, dt, val);
  NumcArray *b = numc_array_fill(ctx, shb, 2, dt, val);
  NumcArray *out = numc_array_zeros(ctx, sho, 2, dt);

  /* Warmup */
  for (int i = 0; i < warmup; i++)
    numc_matmul(a, b, out);

  /* Timed region */
  double t0 = time_us();
  for (int i = 0; i < iters; i++)
    numc_matmul(a, b, out);
  double elapsed = (time_us() - t0) / iters;

  double flops = 2.0 * N * N * N;
  double gflops = flops / (elapsed * 1e3);

  printf("%s %zux%zu: %.2f us/iter (%.2f GFLOPS), %d iters\n", argv[1], N, N,
         elapsed, gflops, iters);

  numc_ctx_free(ctx);
  return 0;
}
