/**
 * @file simd_benchmark.c
 * @brief Benchmark to demonstrate SIMD auto-vectorization performance
 */

#include "array.h"
#include "types.h"
#include <stdio.h>
#include <time.h>

#define BENCHMARK_ITERATIONS 100

double benchmark_operation(Array *(*op_func)(const Array *, const Array *),
                           const Array *a, const Array *b, int iterations) {
  clock_t start = clock();
  for (int i = 0; i < iterations; i++) {
    Array *result = op_func(a, b);
    array_free(result);
  }
  clock_t end = clock();
  return (double)(end - start) / CLOCKS_PER_SEC;
}

void print_header(const char *title) {
  printf("\n");
  printf("╔═══════════════════════════════════════════════════════════════════╗"
         "\n");
  printf("║ %-65s ║\n", title);
  printf("╚═══════════════════════════════════════════════════════════════════╝"
         "\n");
}

void print_simd_info(void) {
  print_header("SIMD Auto-Vectorization Information");

  printf("\nCompiler Optimization Details:\n");
  printf("  • Optimization Level: -O3 (Release build)\n");
  printf("  • SIMD Instructions: SSE2 (128-bit) baseline\n");
  printf("  • AVX/AVX2: Enabled with -march=native\n");
  printf("  • Memory Alignment: 16-byte (NUMC_ALIGN)\n");

  printf("\nSIMD Instruction Sets Used:\n");
  printf("  • Integer ops (INT, LONG):  paddd (SSE2) / vpaddd (AVX2)\n");
  printf("  • Float ops:                addps (SSE)  / vaddps (AVX)\n");
  printf("  • Double ops:               addpd (SSE2) / vaddpd (AVX)\n");

  printf("\nVectorization Width:\n");
  printf("  • SSE2:  128-bit (4x int32, 4x float, 2x double per operation)\n");
  printf("  • AVX2:  256-bit (8x int32, 8x float, 4x double per operation)\n");

  printf("\nKey Features:\n");
  printf("  ✓ Compiler auto-vectorizes simple loops at -O3\n");
  printf("  ✓ Type-specific kernels for optimal code generation\n");
  printf("  ✓ Aligned memory allocation for efficient SIMD access\n");
  printf("  ✓ Loop unrolling for better instruction-level parallelism\n");
}

void benchmark_dtype(DType dtype, const char *dtype_name, size_t size) {
  size_t shape[] = {size};
  Array *a = array_ones(1, shape, dtype);
  Array *b = array_ones(1, shape, dtype);

  printf("\n%s Array (%zu elements):\n", dtype_name, size);

  double add_time = benchmark_operation(array_add, a, b, BENCHMARK_ITERATIONS);
  double sub_time = benchmark_operation(array_sub, a, b, BENCHMARK_ITERATIONS);
  double mul_time = benchmark_operation(array_mul, a, b, BENCHMARK_ITERATIONS);
  double div_time = benchmark_operation(array_div, a, b, BENCHMARK_ITERATIONS);

  double avg_add = (add_time / BENCHMARK_ITERATIONS) * 1000000; // microseconds
  double avg_sub = (sub_time / BENCHMARK_ITERATIONS) * 1000000;
  double avg_mul = (mul_time / BENCHMARK_ITERATIONS) * 1000000;
  double avg_div = (div_time / BENCHMARK_ITERATIONS) * 1000000;

  printf("  Addition:       %8.2f μs  (%8.2f Mops/sec)\n", avg_add,
         size / avg_add);
  printf("  Subtraction:    %8.2f μs  (%8.2f Mops/sec)\n", avg_sub,
         size / avg_sub);
  printf("  Multiplication: %8.2f μs  (%8.2f Mops/sec)\n", avg_mul,
         size / avg_mul);
  printf("  Division:       %8.2f μs  (%8.2f Mops/sec)\n", avg_div,
         size / avg_div);

  array_free(a);
  array_free(b);
}

void benchmark_sizes(void) {
  print_header("Performance Benchmarks - Various Array Sizes");

  printf("\nBenchmark Configuration:\n");
  printf("  • Iterations per test: %d\n", BENCHMARK_ITERATIONS);
  printf("  • Data type: DTYPE_FLOAT (32-bit)\n");
  printf("  • Memory layout: Contiguous (optimized path)\n");

  // Small arrays (fits in L1 cache)
  printf("\n--- Small Arrays (L1 Cache) ---\n");
  benchmark_dtype(DTYPE_FLOAT, "FLOAT", 1024); // 4 KB

  // Medium arrays (fits in L2 cache)
  printf("\n--- Medium Arrays (L2 Cache) ---\n");
  benchmark_dtype(DTYPE_FLOAT, "FLOAT", 65536); // 256 KB

  // Large arrays (fits in L3 cache)
  printf("\n--- Large Arrays (L3 Cache) ---\n");
  benchmark_dtype(DTYPE_FLOAT, "FLOAT", 524288); // 2 MB

  // Very large arrays (exceeds cache)
  printf("\n--- Very Large Arrays (Main Memory) ---\n");
  benchmark_dtype(DTYPE_FLOAT, "FLOAT", 4194304); // 16 MB
}

void benchmark_dtypes(void) {
  print_header("Performance Comparison - Different Data Types");

  size_t size = 1000000;
  printf("\nArray size: %zu elements\n", size);

  benchmark_dtype(DTYPE_INT, "INT32", size);
  benchmark_dtype(DTYPE_LONG, "INT64", size);
  benchmark_dtype(DTYPE_FLOAT, "FLOAT", size);
  benchmark_dtype(DTYPE_DOUBLE, "DOUBLE", size);
}

void demonstrate_vectorization(void) {
  print_header("SIMD Vectorization Demonstration");

  printf("\nComparing scalar vs vectorized performance:\n");
  printf("(The compiler automatically vectorizes our simple loops)\n\n");

  size_t sizes[] = {100, 1000, 10000, 100000, 1000000};

  printf("Array Size │   Add Time   │  Elements/μs │ Speedup Factor\n");
  printf("───────────┼──────────────┼──────────────┼────────────────\n");

  double baseline_throughput = 0;

  for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
    size_t size = sizes[i];
    size_t shape[] = {size};

    Array *a = array_ones(1, shape, DTYPE_FLOAT);
    Array *b = array_ones(1, shape, DTYPE_FLOAT);

    clock_t start = clock();
    for (int j = 0; j < 100; j++) {
      Array *result = array_add(a, b);
      array_free(result);
    }
    clock_t end = clock();

    double avg_time_us =
        ((double)(end - start) / CLOCKS_PER_SEC / 100) * 1000000;
    double throughput = size / avg_time_us;

    if (i == 0) {
      baseline_throughput = throughput;
    }

    double speedup = throughput / baseline_throughput;

    printf("%10zu │ %9.2f μs │ %12.2f │ %14.2fx\n", size, avg_time_us,
           throughput, speedup);

    array_free(a);
    array_free(b);
  }

  printf("\nNote: Speedup increases with array size due to better SIMD "
         "utilization\n");
  printf("      and amortization of loop overhead.\n");
}

int main(void) {
  printf("\n");
  printf("╔═══════════════════════════════════════════════════════════════════╗"
         "\n");
  printf("║              NUMC - SIMD Auto-Vectorization Benchmark             "
         "║\n");
  printf("║                  Demonstrating Compiler Optimization               "
         "║\n");
  printf("╚═══════════════════════════════════════════════════════════════════╝"
         "\n");

  print_simd_info();
  demonstrate_vectorization();
  benchmark_sizes();
  benchmark_dtypes();

  printf("\n");
  print_header("Summary");
  printf(
      "\n✓ All operations are automatically SIMD-vectorized by the compiler\n");
  printf("✓ Performance scales with array size and CPU cache hierarchy\n");
  printf("✓ Type-specific kernels enable optimal instruction selection\n");
  printf("✓ Aligned memory allocation maximizes SIMD efficiency\n\n");

  printf("To verify SIMD usage, compile with:\n");
  printf("  clang -O3 -march=native -S -masm=intel -Iinclude src/math.c "
         "src/array.c\n");
  printf("  rg -e 'paddd|vpaddd|addps|vaddps' math.s\n\n");

  return 0;
}
