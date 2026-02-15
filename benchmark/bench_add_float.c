#include "array/_array_core.h"
#include "array/array_core.h"
#include "array/array_dtype.h"
#include "math/math.h"
#include <stdio.h>
#include <time.h>

#define WARMUP 20
#define ITERATIONS 500

static double get_time_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e9 + ts.tv_nsec;
}

static double bench(void (*fn)(const NumcArray *, const NumcArray *, NumcArray *),
                    const NumcArray *a, const NumcArray *b, NumcArray *out,
                    int iterations) {
  double start = get_time_ns();
  for (int i = 0; i < iterations; i++) {
    fn(a, b, out);
  }
  double end = get_time_ns();
  return (end - start) / iterations; // ns per call
}

static void call_array_add(const NumcArray *a, const NumcArray *b,
                           NumcArray *out) {
  array_add(a, b, out);
}

static void call_array_add_float(const NumcArray *a, const NumcArray *b,
                                 NumcArray *out) {
  array_add_float(a, b, out);
}

int main(void) {
  const size_t sizes[] = {1000, 10000, 100000, 1000000, 10000000};
  const int nsizes = sizeof(sizes) / sizeof(sizes[0]);

  printf("=============================================================\n");
  printf("  array_add (auto-vec) vs array_add_float (explicit AVX2)\n");
  printf("  Warmup: %d | Iterations: %d\n", WARMUP, ITERATIONS);
  printf("=============================================================\n");
  printf("  %-12s %14s %14s %10s\n", "Size", "array_add", "add_float", "Speedup");
  printf("-------------------------------------------------------------\n");

  for (int s = 0; s < nsizes; s++) {
    size_t n = sizes[s];

    NumcCtx *ctx = array_create_ctx();
    size_t shape[] = {n};
    NumcArray *a = array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
    NumcArray *b = array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
    NumcArray *out = array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);

    if (!a || !b || !out) {
      fprintf(stderr, "alloc failed for size %zu\n", n);
      array_free(ctx);
      continue;
    }

    // fill with something non-trivial
    float *pa = (float *)a->data;
    float *pb = (float *)b->data;
    for (size_t i = 0; i < n; i++) {
      pa[i] = (float)i * 0.5f;
      pb[i] = (float)i * 0.3f;
    }

    // warmup
    for (int i = 0; i < WARMUP; i++) {
      array_add(a, b, out);
      array_add_float(a, b, out);
    }

    double t_generic = bench(call_array_add, a, b, out, ITERATIONS);
    double t_avx = bench(call_array_add_float, a, b, out, ITERATIONS);

    double speedup = t_generic / t_avx;

    const char *unit_g = "ns";
    double v_g = t_generic;
    if (v_g >= 1e6) { v_g /= 1e6; unit_g = "ms"; }
    else if (v_g >= 1e3) { v_g /= 1e3; unit_g = "us"; }

    const char *unit_a = "ns";
    double v_a = t_avx;
    if (v_a >= 1e6) { v_a /= 1e6; unit_a = "ms"; }
    else if (v_a >= 1e3) { v_a /= 1e3; unit_a = "us"; }

    printf("  %-12zu %10.2f %-2s %10.2f %-2s %9.2fx\n",
           n, v_g, unit_g, v_a, unit_a, speedup);

    array_free(ctx);
  }

  printf("=============================================================\n");
  return 0;
}
