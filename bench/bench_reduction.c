/*
 * bench_reduction.c — Reduction operation benchmark
 *
 * Tests: sum (full), sum_axis (axis=0, axis=-1)
 * Varies: dtype (all 10), array size (100–1M)
 * Reports: avg time (us), throughput (Mops/s)
 */

#include <numc/numc.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

/* ── Config ────────────────────────────────────────────────────────── */

#define WARMUP 20
#define ITERS 200

/* ── Timer ─────────────────────────────────────────────────────────── */

static double time_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

/* ── Helpers ───────────────────────────────────────────────────────── */

static const char *dtype_name(NumcDType dt) {
  static const char *names[] = {
      [NUMC_DTYPE_INT8] = "int8",       [NUMC_DTYPE_INT16] = "int16",
      [NUMC_DTYPE_INT32] = "int32",     [NUMC_DTYPE_INT64] = "int64",
      [NUMC_DTYPE_UINT8] = "uint8",     [NUMC_DTYPE_UINT16] = "uint16",
      [NUMC_DTYPE_UINT32] = "uint32",   [NUMC_DTYPE_UINT64] = "uint64",
      [NUMC_DTYPE_FLOAT32] = "float32", [NUMC_DTYPE_FLOAT64] = "float64",
  };
  return names[dt];
}

static void fill_value(NumcDType dt, char buf[static 8]) {
  memset(buf, 0, 8);
  switch (dt) {
  case NUMC_DTYPE_INT8:
    *(int8_t *)buf = 1;
    break;
  case NUMC_DTYPE_INT16:
    *(int16_t *)buf = 1;
    break;
  case NUMC_DTYPE_INT32:
    *(int32_t *)buf = 1;
    break;
  case NUMC_DTYPE_INT64:
    *(int64_t *)buf = 1;
    break;
  case NUMC_DTYPE_UINT8:
    *(uint8_t *)buf = 1;
    break;
  case NUMC_DTYPE_UINT16:
    *(uint16_t *)buf = 1;
    break;
  case NUMC_DTYPE_UINT32:
    *(uint32_t *)buf = 1;
    break;
  case NUMC_DTYPE_UINT64:
    *(uint64_t *)buf = 1;
    break;
  case NUMC_DTYPE_FLOAT32:
    *(float *)buf = 1.0f;
    break;
  case NUMC_DTYPE_FLOAT64:
    *(double *)buf = 1.0;
    break;
  }
}

static const NumcDType ALL_DTYPES[] = {
    NUMC_DTYPE_INT8,    NUMC_DTYPE_UINT8,   NUMC_DTYPE_INT16,
    NUMC_DTYPE_UINT16,  NUMC_DTYPE_INT32,   NUMC_DTYPE_UINT32,
    NUMC_DTYPE_INT64,   NUMC_DTYPE_UINT64,  NUMC_DTYPE_FLOAT32,
    NUMC_DTYPE_FLOAT64,
};
static const int N_DTYPES = sizeof(ALL_DTYPES) / sizeof(ALL_DTYPES[0]);

/* ── Benchmark: full sum — all dtypes ──────────────────────────────── */

static void bench_sum_full(size_t size) {
  printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
         "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
  printf("  SUM (full reduction)  (%zu elements, %d iters)\n", size, ITERS);
  printf("\n  %-8s %10s %10s\n", "dtype", "time (us)", "Mop/s");
  printf("  ────────────────────────────────\n");

  for (int d = 0; d < N_DTYPES; d++) {
    NumcDType dt = ALL_DTYPES[d];
    NumcCtx *ctx = numc_ctx_create();
    size_t shape[] = {size};
    char val[8];
    fill_value(dt, val);

    NumcArray *a = numc_array_fill(ctx, shape, 1, dt, val);
    size_t sshape[] = {1};
    NumcArray *out = numc_array_zeros(ctx, sshape, 1, dt);
    if (!a || !out) {
      fprintf(stderr, "  alloc failed for %s\n", dtype_name(dt));
      numc_ctx_free(ctx);
      continue;
    }

    /* Warmup */
    for (int i = 0; i < WARMUP; i++)
      numc_sum(a, out);

    double t0 = time_us();
    for (int i = 0; i < ITERS; i++)
      numc_sum(a, out);
    double t1 = time_us();

    double us = (t1 - t0) / ITERS;
    double mops = size / us;

    printf("  %-8s %10.2f %10.1f\n", dtype_name(dt), us, mops);
    numc_ctx_free(ctx);
  }
}

/* ── Benchmark: sum axis=0 on 2D array ─────────────────────────────── */

static void bench_sum_axis0(size_t rows, size_t cols) {
  size_t total = rows * cols;
  printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
         "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
  printf("  SUM AXIS=0  (%zux%zu = %zu elements, %d iters)\n", rows, cols, total,
         ITERS);
  printf("\n  %-8s %10s %10s\n", "dtype", "time (us)", "Mop/s");
  printf("  ────────────────────────────────\n");

  for (int d = 0; d < N_DTYPES; d++) {
    NumcDType dt = ALL_DTYPES[d];
    NumcCtx *ctx = numc_ctx_create();
    size_t shape[] = {rows, cols};
    char val[8];
    fill_value(dt, val);

    NumcArray *a = numc_array_fill(ctx, shape, 2, dt, val);
    size_t oshape[] = {cols};
    NumcArray *out = numc_array_zeros(ctx, oshape, 1, dt);
    if (!a || !out) {
      fprintf(stderr, "  alloc failed for %s\n", dtype_name(dt));
      numc_ctx_free(ctx);
      continue;
    }

    for (int i = 0; i < WARMUP; i++)
      numc_sum_axis(a, 0, 0, out);

    double t0 = time_us();
    for (int i = 0; i < ITERS; i++)
      numc_sum_axis(a, 0, 0, out);
    double t1 = time_us();

    double us = (t1 - t0) / ITERS;
    double mops = total / us;

    printf("  %-8s %10.2f %10.1f\n", dtype_name(dt), us, mops);
    numc_ctx_free(ctx);
  }
}

/* ── Benchmark: sum axis=-1 (last) on 2D array ────────────────────── */

static void bench_sum_axis_last(size_t rows, size_t cols) {
  size_t total = rows * cols;
  printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
         "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
  printf("  SUM AXIS=1  (%zux%zu = %zu elements, %d iters)\n", rows, cols, total,
         ITERS);
  printf("\n  %-8s %10s %10s\n", "dtype", "time (us)", "Mop/s");
  printf("  ────────────────────────────────\n");

  for (int d = 0; d < N_DTYPES; d++) {
    NumcDType dt = ALL_DTYPES[d];
    NumcCtx *ctx = numc_ctx_create();
    size_t shape[] = {rows, cols};
    char val[8];
    fill_value(dt, val);

    NumcArray *a = numc_array_fill(ctx, shape, 2, dt, val);
    size_t oshape[] = {rows};
    NumcArray *out = numc_array_zeros(ctx, oshape, 1, dt);
    if (!a || !out) {
      fprintf(stderr, "  alloc failed for %s\n", dtype_name(dt));
      numc_ctx_free(ctx);
      continue;
    }

    for (int i = 0; i < WARMUP; i++)
      numc_sum_axis(a, 1, 0, out);

    double t0 = time_us();
    for (int i = 0; i < ITERS; i++)
      numc_sum_axis(a, 1, 0, out);
    double t1 = time_us();

    double us = (t1 - t0) / ITERS;
    double mops = total / us;

    printf("  %-8s %10.2f %10.1f\n", dtype_name(dt), us, mops);
    numc_ctx_free(ctx);
  }
}

/* ── Benchmark: size scaling (float32 full sum) ────────────────────── */

static void bench_scaling(void) {
  printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
         "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
  printf("  SIZE SCALING  (float32 sum, %d iters)\n", ITERS);
  printf("\n  %10s %10s %10s %10s\n", "elements", "time (us)", "Mop/s", "GB/s");
  printf("  ──────────────────────────────────────────\n");

  size_t sizes[] = {100, 1000, 10000, 100000, 1000000};
  int nsizes = sizeof(sizes) / sizeof(sizes[0]);

  for (int s = 0; s < nsizes; s++) {
    size_t n = sizes[s];
    NumcCtx *ctx = numc_ctx_create();
    size_t shape[] = {n};
    float va = 1.0f;
    NumcArray *a = numc_array_fill(ctx, shape, 1, NUMC_DTYPE_FLOAT32, &va);
    size_t sshape[] = {1};
    NumcArray *out = numc_array_zeros(ctx, sshape, 1, NUMC_DTYPE_FLOAT32);
    if (!a || !out) {
      numc_ctx_free(ctx);
      continue;
    }

    for (int i = 0; i < WARMUP; i++)
      numc_sum(a, out);

    double t0 = time_us();
    for (int i = 0; i < ITERS; i++)
      numc_sum(a, out);
    double t1 = time_us();

    double us = (t1 - t0) / ITERS;
    double mops = n / us;
    /* bandwidth: reads 1 array = n * 4 bytes */
    double gbs = (n * sizeof(float)) / (us * 1e3);

    printf("  %10zu %10.2f %10.1f %10.2f\n", n, us, mops, gbs);
    numc_ctx_free(ctx);
  }
}

/* ── main ──────────────────────────────────────────────────────────── */

int main(void) {
  printf("\n  numc reduction benchmark\n");
  printf("  build: "
#ifdef __clang__
         "clang " __clang_version__
#elif defined(__GNUC__)
         "gcc " __VERSION__
#else
         "unknown"
#endif
#ifdef _OPENMP
         " | OpenMP"
#endif
         "\n\n");

  /* 1. Full sum — all dtypes, 1M elements */
  bench_sum_full(1000000);

  /* 2. Axis=0 reduction — 1000x1000 (reduce rows) */
  bench_sum_axis0(1000, 1000);

  /* 3. Axis=1 reduction — 1000x1000 (reduce cols, contiguous inner dim) */
  bench_sum_axis_last(1000, 1000);

  /* 4. Size scaling */
  bench_scaling();

  printf("\n");
  return 0;
}
