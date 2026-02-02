#include "array.h"
#include <stdio.h>
#include <time.h>

#define BENCHMARK_ITERATIONS 100
#define ARRAY_SIZE (10 * 1024 * 1024) // 10M elements

static double get_time_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

static void benchmark_fill(DType dtype, const char *name, size_t elem_size) {
  size_t shape[] = {ARRAY_SIZE};
  
  double total_time = 0.0;
  for (int iter = 0; iter < BENCHMARK_ITERATIONS; iter++) {
    double start = get_time_ms();
    
    // Create and fill array with a value
    Array *arr = NULL;
    switch (dtype) {
      case DTYPE_BYTE: {
        int8_t val = 42;
        arr = array_fill(1, shape, dtype, &val);
        break;
      }
      case DTYPE_UBYTE: {
        uint8_t val = 42;
        arr = array_fill(1, shape, dtype, &val);
        break;
      }
      case DTYPE_SHORT: {
        int16_t val = 42;
        arr = array_fill(1, shape, dtype, &val);
        break;
      }
      case DTYPE_USHORT: {
        uint16_t val = 42;
        arr = array_fill(1, shape, dtype, &val);
        break;
      }
      case DTYPE_INT: {
        int32_t val = 42;
        arr = array_fill(1, shape, dtype, &val);
        break;
      }
      case DTYPE_UINT: {
        uint32_t val = 42;
        arr = array_fill(1, shape, dtype, &val);
        break;
      }
      case DTYPE_LONG: {
        int64_t val = 42;
        arr = array_fill(1, shape, dtype, &val);
        break;
      }
      case DTYPE_ULONG: {
        uint64_t val = 42;
        arr = array_fill(1, shape, dtype, &val);
        break;
      }
      case DTYPE_FLOAT: {
        float val = 42.0f;
        arr = array_fill(1, shape, dtype, &val);
        break;
      }
      case DTYPE_DOUBLE: {
        double val = 42.0;
        arr = array_fill(1, shape, dtype, &val);
        break;
      }
    }
    
    double end = get_time_ms();
    
    if (!arr) {
      fprintf(stderr, "Failed to create array for %s\n", name);
      return;
    }
    
    total_time += (end - start);
    array_free(arr);
  }
  
  double avg_time = total_time / BENCHMARK_ITERATIONS;
  double bytes_per_sec = (ARRAY_SIZE * elem_size) / (avg_time / 1000.0);
  double gb_per_sec = bytes_per_sec / (1024.0 * 1024.0 * 1024.0);
  
  printf("%-10s: %2zu bytes, %6.3f ms/iter, %6.2f GB/s\n", 
         name, elem_size, avg_time, gb_per_sec);
}

int main(void) {
  printf("Fill Performance Benchmark (%d iterations, %d elements)\n", 
         BENCHMARK_ITERATIONS, ARRAY_SIZE);
  printf("================================================================\n\n");
  printf("Data Type   Size     Time/Iter     Throughput\n");
  printf("----------------------------------------------------------------\n");
  
  // 8-bit types (1 byte)
  benchmark_fill(DTYPE_BYTE, "BYTE", 1);
  benchmark_fill(DTYPE_UBYTE, "UBYTE", 1);
  
  // 16-bit types (2 bytes)
  benchmark_fill(DTYPE_SHORT, "SHORT", 2);
  benchmark_fill(DTYPE_USHORT, "USHORT", 2);
  
  // 32-bit types (4 bytes)
  benchmark_fill(DTYPE_INT, "INT", 4);
  benchmark_fill(DTYPE_UINT, "UINT", 4);
  benchmark_fill(DTYPE_FLOAT, "FLOAT", 4);
  
  // 64-bit types (8 bytes)
  benchmark_fill(DTYPE_LONG, "LONG", 8);
  benchmark_fill(DTYPE_ULONG, "ULONG", 8);
  benchmark_fill(DTYPE_DOUBLE, "DOUBLE", 8);
  
  printf("\n");
  printf("Note: Higher GB/s is better (includes allocation overhead)\n");
  printf("      All types should show good vectorization performance\n");
  
  return 0;
}
