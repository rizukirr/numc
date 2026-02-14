#include "array/_array_core.h"
#include "array/array_core.h"
#include "array/array_dtype.h"
#include "math/math.h"
#include <stdio.h>
#include <time.h>

#define BENCHMARK_ITERATIONS 100

static double get_time_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

static double benchmark_add(const NumcArray *a, const NumcArray *b,
                            NumcArray *out, int iterations) {
  double start = get_time_ms();
  for (int i = 0; i < iterations; i++) {
    array_add(a, b, out);
  }
  double end = get_time_ms();
  return (end - start) / 1000.0;
}

static void benchmark_type(NumcCtx *ctx, NumcDType dtype,
                           const char *type_name, size_t size) {
  size_t shape[] = {size};
  NumcArray *a = array_zeros(ctx, shape, 1, dtype);
  NumcArray *b = array_zeros(ctx, shape, 1, dtype);
  NumcArray *out = array_zeros(ctx, shape, 1, dtype);
  if (!a || !b || !out) {
    fprintf(stderr, "Failed to allocate arrays for %s\n", type_name);
    return;
  }

  // warmup
  for (int i = 0; i < 10; i++)
    array_add(a, b, out);

  double add_time = benchmark_add(a, b, out, BENCHMARK_ITERATIONS);
  double avg_add = (add_time / BENCHMARK_ITERATIONS) * 1000000;

  printf("  %-10s Add: %8.2f us  (%10.2f Mops/sec)\n", type_name, avg_add,
         size / avg_add);
}

int main(void) {
  printf("===================================================\n");
  printf("       ADD Operation Benchmark (1M elements)        \n");
  printf("===================================================\n\n");

  const size_t size = 1000000;

  NumcCtx *ctx = array_create_ctx();
  if (!ctx) {
    fprintf(stderr, "Failed to create context\n");
    return 1;
  }

  benchmark_type(ctx, NUMC_DTYPE_INT8, "INT8", size);
  benchmark_type(ctx, NUMC_DTYPE_UINT8, "UINT8", size);
  benchmark_type(ctx, NUMC_DTYPE_INT16, "INT16", size);
  benchmark_type(ctx, NUMC_DTYPE_UINT16, "UINT16", size);
  benchmark_type(ctx, NUMC_DTYPE_INT32, "INT32", size);
  benchmark_type(ctx, NUMC_DTYPE_UINT32, "UINT32", size);
  benchmark_type(ctx, NUMC_DTYPE_INT64, "INT64", size);
  benchmark_type(ctx, NUMC_DTYPE_UINT64, "UINT64", size);
  benchmark_type(ctx, NUMC_DTYPE_FLOAT32, "FLOAT32", size);
  benchmark_type(ctx, NUMC_DTYPE_FLOAT64, "FLOAT64", size);

  array_free(ctx);
  return 0;
}
