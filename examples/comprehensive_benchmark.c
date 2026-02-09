/**
 * @file comprehensive_benchmark.c
 * @brief Comprehensive benchmark for all NUMC operations
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

void print_header(const char *title) {
  printf("\n");
  printf("╔═══════════════════════════════════════════════════════════════════╗"
         "\n");
  printf("║ %-65s ║\n", title);
  printf("╚═══════════════════════════════════════════════════════════════════╝"
         "\n");
}

// Benchmark binary operations (array + array)
double benchmark_binary_op(int (*op_func)(const Array *, const Array *,
                                          Array *),
                           const Array *a, const Array *b, Array *out,
                           int iterations) {
  int (*volatile vop)(const Array *, const Array *, Array *) = op_func;

  double start = get_time_ms();
  for (int i = 0; i < iterations; i++) {
    vop(a, b, out);
  }
  double end = get_time_ms();
  return (end - start) / 1000.0;
}

// Benchmark reduction operations (array -> scalar)
double benchmark_reduce_op(int (*op_func)(const Array *, void *),
                           const Array *a, void *out, int iterations) {
  int (*volatile vop)(const Array *, void *) = op_func;

  double start = get_time_ms();
  for (int i = 0; i < iterations; i++) {
    vop(a, out);
  }
  double end = get_time_ms();
  return (end - start) / 1000.0;
}

// Benchmark double reduction operations (array -> double scalar)
double benchmark_double_reduce_op(int (*op_func)(const Array *, double *),
                                  const Array *a, double *out, int iterations) {
  int (*volatile vop)(const Array *, double *) = op_func;

  double start = get_time_ms();
  for (int i = 0; i < iterations; i++) {
    vop(a, out);
  }
  double end = get_time_ms();
  return (end - start) / 1000.0;
}

// Benchmark dot product (array, array -> scalar)
double benchmark_dot_op(int (*op_func)(const Array *, const Array *, void *),
                        const Array *a, const Array *b, void *out,
                        int iterations) {
  int (*volatile vop)(const Array *, const Array *, void *) = op_func;

  double start = get_time_ms();
  for (int i = 0; i < iterations; i++) {
    vop(a, b, out);
  }
  double end = get_time_ms();
  return (end - start) / 1000.0;
}

// Benchmark scalar operations (array + scalar)
double benchmark_scalar_op(int (*op_func)(const Array *, const void *, Array *),
                           const Array *a, const void *scalar, Array *out,
                           int iterations) {
  int (*volatile vop)(const Array *, const void *, Array *) = op_func;

  double start = get_time_ms();
  for (int i = 0; i < iterations; i++) {
    vop(a, scalar, out);
  }
  double end = get_time_ms();
  return (end - start) / 1000.0;
}

// Benchmark axis reduction operations (array, axis -> new array)
double benchmark_axis_reduce_op(Array *(*op_func)(const Array *, size_t),
                                const Array *a, size_t axis, int iterations) {
  Array *(*volatile vop)(const Array *, size_t) = op_func;

  double start = get_time_ms();
  for (int i = 0; i < iterations; i++) {
    Array *result = vop(a, axis);
    array_free(result);
  }
  double end = get_time_ms();
  return (end - start) / 1000.0;
}

void benchmark_all_operations(NUMC_TYPE numc_type, const char *type_name,
                              size_t size) {
  printf("\n%s Array (%zu elements):\n", type_name, size);

  size_t shape[] = {size};
  Array *a = array_ones(1, shape, numc_type);
  Array *b = array_ones(1, shape, numc_type);

  ArrayCreate create = {
      .ndim = 1,
      .shape = shape,
      .numc_type = numc_type,
      .data = NULL,
      .owns_data = true,
  };
  Array *out = array_create(&create);

  // Allocate result storage for reductions
  // Use the largest type (int64_t/double) to ensure we have enough storage
  NUMC_LONG result_int = 0;
  NUMC_DOUBLE result_float = 0.0;
  void *result = (numc_type == NUMC_TYPE_FLOAT || numc_type == NUMC_TYPE_DOUBLE)
                     ? (void *)&result_float
                     : (void *)&result_int;

  // Scalar value for scalar operations
  // Use the largest type (int64_t/double) to ensure we have enough storage
  NUMC_LONG scalar_int = 2;
  NUMC_DOUBLE scalar_float = 2.0;
  void *scalar = (numc_type == NUMC_TYPE_FLOAT || numc_type == NUMC_TYPE_DOUBLE)
                     ? (void *)&scalar_float
                     : (void *)&scalar_int;

  printf("\n  Binary Operations:\n");

  // Addition
  double add_time =
      benchmark_binary_op(array_add, a, b, out, BENCHMARK_ITERATIONS);
  double avg_add = (add_time / BENCHMARK_ITERATIONS) * 1000000;
  printf("    Add:         %10.2f μs  (%10.2f Mops/sec)\n", avg_add,
         size / avg_add);

  // Subtraction
  double sub_time =
      benchmark_binary_op(array_subtract, a, b, out, BENCHMARK_ITERATIONS);
  double avg_sub = (sub_time / BENCHMARK_ITERATIONS) * 1000000;
  printf("    Subtract:    %10.2f μs  (%10.2f Mops/sec)\n", avg_sub,
         size / avg_sub);

  // Multiplication
  double mul_time =
      benchmark_binary_op(array_multiply, a, b, out, BENCHMARK_ITERATIONS);
  double avg_mul = (mul_time / BENCHMARK_ITERATIONS) * 1000000;
  printf("    Multiply:    %10.2f μs  (%10.2f Mops/sec)\n", avg_mul,
         size / avg_mul);

  // Division
  double div_time =
      benchmark_binary_op(array_divide, a, b, out, BENCHMARK_ITERATIONS);
  double avg_div = (div_time / BENCHMARK_ITERATIONS) * 1000000;
  printf("    Divide:      %10.2f μs  (%10.2f Mops/sec)\n", avg_div,
         size / avg_div);

  printf("\n  Scalar Operations:\n");

  // Add scalar
  double adds_time = benchmark_scalar_op(array_add_scalar, a, scalar, out,
                                         BENCHMARK_ITERATIONS);
  double avg_adds = (adds_time / BENCHMARK_ITERATIONS) * 1000000;
  printf("    Add scalar:  %10.2f μs  (%10.2f Mops/sec)\n", avg_adds,
         size / avg_adds);

  // Subtract scalar
  double subs_time = benchmark_scalar_op(array_subtract_scalar, a, scalar, out,
                                         BENCHMARK_ITERATIONS);
  double avg_subs = (subs_time / BENCHMARK_ITERATIONS) * 1000000;
  printf("    Sub scalar:  %10.2f μs  (%10.2f Mops/sec)\n", avg_subs,
         size / avg_subs);

  // Multiply scalar
  double muls_time = benchmark_scalar_op(array_multiply_scalar, a, scalar, out,
                                         BENCHMARK_ITERATIONS);
  double avg_muls = (muls_time / BENCHMARK_ITERATIONS) * 1000000;
  printf("    Mul scalar:  %10.2f μs  (%10.2f Mops/sec)\n", avg_muls,
         size / avg_muls);

  // Divide scalar
  double divs_time = benchmark_scalar_op(array_divide_scalar, a, scalar, out,
                                         BENCHMARK_ITERATIONS);
  double avg_divs = (divs_time / BENCHMARK_ITERATIONS) * 1000000;
  printf("    Div scalar:  %10.2f μs  (%10.2f Mops/sec)\n", avg_divs,
         size / avg_divs);

  printf("\n  Reduction Operations:\n");

  // Sum
  double sum_time =
      benchmark_reduce_op(array_sum, a, result, BENCHMARK_ITERATIONS);
  double avg_sum = (sum_time / BENCHMARK_ITERATIONS) * 1000000;
  printf("    Sum:         %10.2f μs  (%10.2f Mops/sec)\n", avg_sum,
         size / avg_sum);

  // Min
  double min_time =
      benchmark_reduce_op(array_min, a, result, BENCHMARK_ITERATIONS);
  double avg_min = (min_time / BENCHMARK_ITERATIONS) * 1000000;
  printf("    Min:         %10.2f μs  (%10.2f Mops/sec)\n", avg_min,
         size / avg_min);

  // Max
  double max_time =
      benchmark_reduce_op(array_max, a, result, BENCHMARK_ITERATIONS);
  double avg_max = (max_time / BENCHMARK_ITERATIONS) * 1000000;
  printf("    Max:         %10.2f μs  (%10.2f Mops/sec)\n", avg_max,
         size / avg_max);

  // Prod
  double prod_time =
      benchmark_reduce_op(array_prod, a, result, BENCHMARK_ITERATIONS);
  double avg_prod = (prod_time / BENCHMARK_ITERATIONS) * 1000000;
  printf("    Prod:        %10.2f μs  (%10.2f Mops/sec)\n", avg_prod,
         size / avg_prod);

  // Dot product
  double dot_time =
      benchmark_dot_op(array_dot, a, b, result, BENCHMARK_ITERATIONS);
  double avg_dot = (dot_time / BENCHMARK_ITERATIONS) * 1000000;
  printf("    Dot product: %10.2f μs  (%10.2f Mops/sec)\n", avg_dot,
         size / avg_dot);

  // Mean (full reduction)
  NUMC_DOUBLE mean_result = 0.0;
  double mean_full_time = benchmark_double_reduce_op(array_mean, a, &mean_result,
                                                     BENCHMARK_ITERATIONS);
  double avg_mean_full = (mean_full_time / BENCHMARK_ITERATIONS) * 1000000;
  printf("    Mean:        %10.2f μs  (%10.2f Mops/sec)\n", avg_mean_full,
         size / avg_mean_full);

  // Std (full reduction)
  NUMC_DOUBLE std_result = 0.0;
  double std_full_time = benchmark_double_reduce_op(array_std, a, &std_result,
                                                    BENCHMARK_ITERATIONS);
  double avg_std_full = (std_full_time / BENCHMARK_ITERATIONS) * 1000000;
  printf("    Std:         %10.2f μs  (%10.2f Mops/sec)\n", avg_std_full,
         size / avg_std_full);

  printf("\n  Axis Reduction Operations (axis=0 on 2D %zux%zu):\n",
         (size_t)1000, size / 1000);

  // Reshape to 2D for axis benchmarks: 1000 x (size/1000)
  size_t rows = 1000;
  size_t cols = size / rows;
  size_t shape_2d[] = {rows, cols};
  Array *a2d = array_ones(2, shape_2d, numc_type);

  // Mean axis=0
  double mean_time =
      benchmark_axis_reduce_op(array_mean_axis, a2d, 0, BENCHMARK_ITERATIONS);
  double avg_mean = (mean_time / BENCHMARK_ITERATIONS) * 1000000;
  printf("    Mean:        %10.2f μs  (%10.2f Mops/sec)\n", avg_mean,
         size / avg_mean);

  // Mean axis=1
  double mean1_time =
      benchmark_axis_reduce_op(array_mean_axis, a2d, 1, BENCHMARK_ITERATIONS);
  double avg_mean1 = (mean1_time / BENCHMARK_ITERATIONS) * 1000000;
  printf("    Mean(ax=1):  %10.2f μs  (%10.2f Mops/sec)\n", avg_mean1,
         size / avg_mean1);

  // Std axis=0
  double std_time =
      benchmark_axis_reduce_op(array_std_axis, a2d, 0, BENCHMARK_ITERATIONS);
  double avg_std = (std_time / BENCHMARK_ITERATIONS) * 1000000;
  printf("    Std:         %10.2f μs  (%10.2f Mops/sec)\n", avg_std,
         size / avg_std);

  // Std axis=1
  double std1_time =
      benchmark_axis_reduce_op(array_std_axis, a2d, 1, BENCHMARK_ITERATIONS);
  double avg_std1 = (std1_time / BENCHMARK_ITERATIONS) * 1000000;
  printf("    Std(ax=1):   %10.2f μs  (%10.2f Mops/sec)\n", avg_std1,
         size / avg_std1);

  array_free(a2d);
  array_free(a);
  array_free(b);
  array_free(out);
}

void benchmark_by_size(void) {
  print_header("Performance by Array Size (FLOAT)");

  printf("\nBenchmark Configuration:\n");
  printf("  Iterations: %d per operation\n", BENCHMARK_ITERATIONS);
  printf("  Data type: FLOAT (32-bit)\n");
  printf("  Memory layout: Contiguous\n");

  printf("\n--- Small Arrays (L1 Cache) ---\n");
  benchmark_all_operations(NUMC_TYPE_FLOAT, "FLOAT", 1024);

  printf("\n--- Medium Arrays (L2 Cache) ---\n");
  benchmark_all_operations(NUMC_TYPE_FLOAT, "FLOAT", 65536);

  printf("\n--- Large Arrays (L3 Cache) ---\n");
  benchmark_all_operations(NUMC_TYPE_FLOAT, "FLOAT", 524288);

  printf("\n--- Very Large Arrays (Main Memory) ---\n");
  benchmark_all_operations(NUMC_TYPE_FLOAT, "FLOAT", 4194304);
}

void benchmark_by_type(void) {
  print_header("Performance by Data Type (1M elements)");

  printf("\nArray size: 1,000,000 elements\n");

  benchmark_all_operations(NUMC_TYPE_BYTE, "BYTE (INT8)", 1000000);
  benchmark_all_operations(NUMC_TYPE_UBYTE, "UBYTE (UINT8)", 1000000);
  benchmark_all_operations(NUMC_TYPE_SHORT, "SHORT (INT16)", 1000000);
  benchmark_all_operations(NUMC_TYPE_USHORT, "USHORT (UINT16)", 1000000);
  benchmark_all_operations(NUMC_TYPE_INT, "INT32", 1000000);
  benchmark_all_operations(NUMC_TYPE_UINT, "UINT32", 1000000);
  benchmark_all_operations(NUMC_TYPE_LONG, "INT64", 1000000);
  benchmark_all_operations(NUMC_TYPE_ULONG, "UINT64", 1000000);
  benchmark_all_operations(NUMC_TYPE_FLOAT, "FLOAT", 1000000);
  benchmark_all_operations(NUMC_TYPE_DOUBLE, "DOUBLE", 1000000);
}

int main(void) {
  printf("\n");
  printf("╔═══════════════════════════════════════════════════════════════════╗"
         "\n");
  printf("║              NUMC Comprehensive Performance Benchmark             "
         "║\n");
  printf("║         Testing All Operations Across Multiple Data Types         "
         "║\n");
  printf("╚═══════════════════════════════════════════════════════════════════╝"
         "\n");

  printf("\nOperations tested:\n");
  printf("  Binary:     add, subtract, multiply, divide\n");
  printf("  Scalar:     add_scalar, subtract_scalar, multiply_scalar, "
         "divide_scalar\n");
  printf("  Reduction:  sum, min, max, prod, dot\n");
  printf("  Axis:       mean(ax=0,1), std(ax=0,1)\n");

  benchmark_by_type();
  benchmark_by_size();

  printf("\n");
  print_header("Summary");
  printf("\nAll operations completed successfully.\n");
  printf("Results show performance across:\n");
  printf("  - 10 data types (BYTE, UBYTE, SHORT, USHORT, INT32, UINT32, INT64, "
         "UINT64, FLOAT, DOUBLE)\n");
  printf("  - 17 operation types\n");
  printf("  - Multiple array sizes (1K to 4M elements)\n\n");

  return 0;
}
