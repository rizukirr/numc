/**
 * @file benchmark_scalar.c
 * @brief Focused benchmark for SCALAR operations (add/sub/mul/div with scalar)
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

double benchmark_scalar_op(int (*op_func)(const Array *, const void *, Array *),
                           const Array *a, const void *scalar, Array *out, int iterations) {
  double start = get_time_ms();
  for (int i = 0; i < iterations; i++) {
    op_func(a, scalar, out);
  }
  double end = get_time_ms();
  return (end - start) / 1000.0;
}

void benchmark_type(NUMC_TYPE numc_type, const char *type_name, size_t size) {
  size_t shape[] = {size};
  Array *a = array_ones(1, shape, numc_type);
  Array *out = array_zeros(1, shape, numc_type);

  // Scalar value
  int64_t scalar_int = 2;
  double scalar_float = 2.0;
  void *scalar = (numc_type == NUMC_TYPE_FLOAT || numc_type == NUMC_TYPE_DOUBLE)
                     ? (void *)&scalar_float
                     : (void *)&scalar_int;

  // Warmup
  for (int i = 0; i < 10; i++) {
    array_add_scalar(a, scalar, out);
  }

  // Benchmark each operation
  double add_time = benchmark_scalar_op(array_add_scalar, a, scalar, out, BENCHMARK_ITERATIONS);
  double sub_time = benchmark_scalar_op(array_subtract_scalar, a, scalar, out, BENCHMARK_ITERATIONS);
  double mul_time = benchmark_scalar_op(array_multiply_scalar, a, scalar, out, BENCHMARK_ITERATIONS);
  double div_time = benchmark_scalar_op(array_divide_scalar, a, scalar, out, BENCHMARK_ITERATIONS);

  double avg_add = (add_time / BENCHMARK_ITERATIONS) * 1000000;
  double avg_sub = (sub_time / BENCHMARK_ITERATIONS) * 1000000;
  double avg_mul = (mul_time / BENCHMARK_ITERATIONS) * 1000000;
  double avg_div = (div_time / BENCHMARK_ITERATIONS) * 1000000;

  printf("  %-10s Add: %7.2f μs  Sub: %7.2f μs  Mul: %7.2f μs  Div: %7.2f μs\n",
         type_name, avg_add, avg_sub, avg_mul, avg_div);

  array_free(a);
  array_free(out);
}

int main(void) {
  printf("╔═══════════════════════════════════════════════════════════════════╗\n");
  printf("║         SCALAR Operations Benchmark (1M elements)                ║\n");
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
