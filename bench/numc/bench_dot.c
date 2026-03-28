#include <numc/numc.h>
#include <stdio.h>
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

int main() {
  size_t N = 1000000;
  size_t shape[] = {N};
  NumcCtx *ctx = numc_ctx_create();

  NumcArray *a = numc_array_randn(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *b = numc_array_randn(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
  NumcArray *out = numc_array_zeros(ctx, (size_t[]){1}, 1, NUMC_DTYPE_FLOAT32);

  int iters = 1000;

  double t0 = time_us();
  for (int i = 0; i < iters; i++) {
    numc_dot(a, b, out);
  }
  double t_dot = (time_us() - t0) / iters;

  printf("\nnumc Dot Product Benchmark (N=%zu, %d iters)\n", N, iters);
  printf("numc_dot: %.2f us\n", t_dot);

  numc_ctx_free(ctx);
  return 0;
}
