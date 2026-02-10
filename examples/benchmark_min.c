/**
 * @file benchmark_min.c
 * @brief Focused benchmark for MIN reduction operation
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

double benchmark_min(const Array *a, void *result, int iterations) {
  double start = get_time_ms();
  for (int i = 0; i < iterations; i++) {
    array_min(a, result);
  }
  double end = get_time_ms();
  return (end - start) / 1000.0;
}

void benchmark_type(NUMC_TYPE numc_type, const char *type_name, size_t size) {
  size_t shape[] = {size};
  Array *a = array_ones(1, shape, numc_type);

  // Result storage (use largest type)
  int64_t result_int = 0;
  double result_float = 0.0;
  void *result = (numc_type == NUMC_TYPE_FLOAT || numc_type == NUMC_TYPE_DOUBLE)
                     ? (void *)&result_float
                     : (void *)&result_int;

  // Warmup
  for (int i = 0; i < 10; i++) {
    array_min(a, result);
  }

  // Benchmark
  double min_time = benchmark_min(a, result, BENCHMARK_ITERATIONS);
  double avg_min = (min_time / BENCHMARK_ITERATIONS) * 1000000;

  printf("  %-10s Min: %8.2f μs  (%10.2f Mops/sec)\n",
         type_name, avg_min, size / avg_min);

  array_free(a);
}

int main(void) {
  printf("╔═══════════════════════════════════════════════════════════════════╗\n");
  printf("║             MIN Reduction Benchmark (1M elements)                ║\n");
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
