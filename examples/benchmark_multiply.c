/**
 * @file benchmark_multiply.c
 * @brief Focused benchmark for MULTIPLY operation only
 */

#include <numc/numc.h>
#include <stdio.h>
#include <time.h>

#define BENCHMARK_ITERATIONS 100

static double get_time_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

double benchmark_multiply(const Array *a, const Array *b, Array *out, int iterations) {
  double start = get_time_ms();
  for (int i = 0; i < iterations; i++) {
    array_multiply(a, b, out);
  }
  double end = get_time_ms();
  return (end - start) / 1000.0;
}

void benchmark_type(NUMC_TYPE numc_type, const char *type_name, size_t size) {
  size_t shape[] = {size};
  Array *a = array_ones(1, shape, numc_type);
  Array *b = array_ones(1, shape, numc_type);
  Array *out = array_zeros(1, shape, numc_type);

  // Warmup
  for (int i = 0; i < 10; i++) {
    array_multiply(a, b, out);
  }

  // Benchmark
  double mul_time = benchmark_multiply(a, b, out, BENCHMARK_ITERATIONS);
  double avg_mul = (mul_time / BENCHMARK_ITERATIONS) * 1000000;

  printf("  %-10s Multiply: %8.2f μs  (%10.2f Mops/sec)\n",
         type_name, avg_mul, size / avg_mul);

  array_free(a);
  array_free(b);
  array_free(out);
}

int main(void) {
  printf("╔═══════════════════════════════════════════════════════════════════╗\n");
  printf("║           MULTIPLY Operation Benchmark (1M elements)             ║\n");
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
