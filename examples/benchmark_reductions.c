/**
 * @file benchmark_reductions.c
 * @brief Focused benchmark for reduction operations (prod, dot, mean, std)
 */

#include <numc/numc.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>

#define BENCHMARK_ITERATIONS 100

static double get_time_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

void benchmark_type(NUMC_TYPE numc_type, const char *type_name, size_t size) {
  size_t shape[] = {size};
  Array *a = array_ones(1, shape, numc_type);
  Array *b = array_ones(1, shape, numc_type);

  // Result storage
  int64_t result_int = 0;
  double result_float = 0.0;
  void *result = (numc_type == NUMC_TYPE_FLOAT || numc_type == NUMC_TYPE_DOUBLE)
                     ? (void *)&result_float
                     : (void *)&result_int;

  // Use volatile pointers to prevent compiler optimization
  int (*volatile vprod)(const Array *, void *) = array_prod;
  int (*volatile vdot)(const Array *, const Array *, void *) = array_dot;
  int (*volatile vmean)(const Array *, double *) = array_mean;
  int (*volatile vstd)(const Array *, double *) = array_std;

  // Warmup
  for (int i = 0; i < 10; i++) {
    vprod(a, result);
    vdot(a, b, result);
    vmean(a, &result_float);
    vstd(a, &result_float);
  }

  // Benchmark prod (get_time_ms returns milliseconds, convert to microseconds)
  double start = get_time_ms();
  for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
    vprod(a, result);
  }
  double prod_time = ((get_time_ms() - start) / BENCHMARK_ITERATIONS) * 1000;

  // Benchmark dot
  start = get_time_ms();
  for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
    vdot(a, b, result);
  }
  double dot_time = ((get_time_ms() - start) / BENCHMARK_ITERATIONS) * 1000;

  // Benchmark mean
  start = get_time_ms();
  for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
    vmean(a, &result_float);
  }
  double mean_time = ((get_time_ms() - start) / BENCHMARK_ITERATIONS) * 1000;

  // Benchmark std
  start = get_time_ms();
  for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
    vstd(a, &result_float);
  }
  double std_time = ((get_time_ms() - start) / BENCHMARK_ITERATIONS) * 1000;

  printf("  %-10s Prod: %7.2f μs  Dot: %7.2f μs  Mean: %7.2f μs  Std: %7.2f μs\n",
         type_name, prod_time, dot_time, mean_time, std_time);

  array_free(a);
  array_free(b);
}

int main(void) {
  printf("╔═══════════════════════════════════════════════════════════════════╗\n");
  printf("║    Reduction Operations Benchmark (prod/dot/mean/std)            ║\n");
  printf("╚═══════════════════════════════════════════════════════════════════╝\n");
  printf("\n");

  const size_t size = 1000000;

  benchmark_type(NUMC_TYPE_BYTE, "BYTE", size);
  benchmark_type(NUMC_TYPE_UBYTE, "UBYTE", size);
  benchmark_type(NUMC_TYPE_SHORT, "SHORT", size);
  benchmark_type(NUMC_TYPE_USHORT, "USHORT", size);
  benchmark_type(NUMC_TYPE_INT, "INT32", size);
  benchmark_type(NUMC_TYPE_UINT, "UINT32", size);
  benchmark_type(NUMC_TYPE_LONG, "INT64", size);
  benchmark_type(NUMC_TYPE_ULONG, "UINT64", size);
  benchmark_type(NUMC_TYPE_FLOAT, "FLOAT", size);
  benchmark_type(NUMC_TYPE_DOUBLE, "DOUBLE", size);

  return 0;
}
