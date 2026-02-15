/*
 * bench_scalar.c — Element-wise scalar operation benchmark
 *
 * Tests: add_scalar, sub_scalar, mul_scalar, div_scalar (allocating + inplace)
 * Varies: dtype (all 10), array size (1K–1M)
 * Reports: avg time (us), throughput (Mops/s)
 */

#include <numc/numc.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

/* ── Config ────────────────────────────────────────────────────────── */

#define WARMUP 20
#define ITERS  200

typedef int (*ScalarOp)(const NumcArray *, double, NumcArray *);
typedef int (*ScalarInplace)(NumcArray *, double);

/* ── Timer ─────────────────────────────────────────────────────────── */

static double time_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static double bench_scalar(ScalarOp op, const NumcArray *a, double scalar,
                           NumcArray *out, int iters) {
  for (int i = 0; i < WARMUP; i++)
    op(a, scalar, out);

  double t0 = time_us();
  for (int i = 0; i < iters; i++)
    op(a, scalar, out);
  double t1 = time_us();

  return (t1 - t0) / iters;
}

static double bench_inplace(ScalarInplace op, NumcArray *a, double scalar,
                            int iters) {
  for (int i = 0; i < WARMUP; i++)
    op(a, scalar);

  double t0 = time_us();
  for (int i = 0; i < iters; i++)
    op(a, scalar);
  double t1 = time_us();

  return (t1 - t0) / iters;
}

/* ── Helpers ───────────────────────────────────────────────────────── */

static const char *dtype_name(NumcDType dt) {
  static const char *names[] = {
    [NUMC_DTYPE_INT8]    = "int8",    [NUMC_DTYPE_INT16]   = "int16",
    [NUMC_DTYPE_INT32]   = "int32",   [NUMC_DTYPE_INT64]   = "int64",
    [NUMC_DTYPE_UINT8]   = "uint8",   [NUMC_DTYPE_UINT16]  = "uint16",
    [NUMC_DTYPE_UINT32]  = "uint32",  [NUMC_DTYPE_UINT64]  = "uint64",
    [NUMC_DTYPE_FLOAT32] = "float32", [NUMC_DTYPE_FLOAT64] = "float64",
  };
  return names[dt];
}

static void print_header(const char *title) {
  printf("\n  %-8s %8s %8s %8s %8s   %8s %8s %8s %8s\n",
         title,
         "add", "sub", "mul", "div",
         "add", "sub", "mul", "div");
  printf("  %-8s %8s %8s %8s %8s   %8s %8s %8s %8s\n",
         "",
         "(us)", "(us)", "(us)", "(us)",
         "(Mop/s)", "(Mop/s)", "(Mop/s)", "(Mop/s)");
  printf("  ────────────────────────────────────────────"
         "──────────────────────────────────────\n");
}

static void fill_value(NumcDType dt, char buf[static 8]) {
  memset(buf, 0, 8);
  switch (dt) {
    case NUMC_DTYPE_INT8:    *(int8_t *)buf   = 3;    break;
    case NUMC_DTYPE_INT16:   *(int16_t *)buf  = 7;    break;
    case NUMC_DTYPE_INT32:   *(int32_t *)buf  = 42;   break;
    case NUMC_DTYPE_INT64:   *(int64_t *)buf  = 42;   break;
    case NUMC_DTYPE_UINT8:   *(uint8_t *)buf  = 3;    break;
    case NUMC_DTYPE_UINT16:  *(uint16_t *)buf = 7;    break;
    case NUMC_DTYPE_UINT32:  *(uint32_t *)buf = 42;   break;
    case NUMC_DTYPE_UINT64:  *(uint64_t *)buf = 42;   break;
    case NUMC_DTYPE_FLOAT32: *(float *)buf    = 1.5f; break;
    case NUMC_DTYPE_FLOAT64: *(double *)buf   = 1.5;  break;
  }
}

static const NumcDType ALL_DTYPES[] = {
  NUMC_DTYPE_INT8,    NUMC_DTYPE_UINT8,
  NUMC_DTYPE_INT16,   NUMC_DTYPE_UINT16,
  NUMC_DTYPE_INT32,   NUMC_DTYPE_UINT32,
  NUMC_DTYPE_INT64,   NUMC_DTYPE_UINT64,
  NUMC_DTYPE_FLOAT32, NUMC_DTYPE_FLOAT64,
};
static const int N_DTYPES = sizeof(ALL_DTYPES) / sizeof(ALL_DTYPES[0]);

/* ── Benchmark: scalar allocating ops ──────────────────────────────── */

static void bench_scalar_ops(NumcCtx *ctx, size_t size) {
  printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
         "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
  printf("  SCALAR OPS  (%zu elements, %d iters)\n", size, ITERS);
  print_header("dtype");

  ScalarOp ops[] = {numc_add_scalar, numc_sub_scalar,
                    numc_mul_scalar, numc_div_scalar};

  for (int d = 0; d < N_DTYPES; d++) {
    NumcDType dt = ALL_DTYPES[d];
    size_t shape[] = {size};

    char val[8];
    fill_value(dt, val);
    NumcArray *a   = numc_array_fill(ctx, shape, 1, dt, val);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, dt);
    if (!a || !out) {
      fprintf(stderr, "  alloc failed for %s\n", dtype_name(dt));
      continue;
    }

    double us[4], mops[4];
    for (int op = 0; op < 4; op++) {
      us[op]   = bench_scalar(ops[op], a, 2.0, out, ITERS);
      mops[op] = size / us[op];
    }

    printf("  %-8s %8.2f %8.2f %8.2f %8.2f   %8.1f %8.1f %8.1f %8.1f\n",
           dtype_name(dt),
           us[0], us[1], us[2], us[3],
           mops[0], mops[1], mops[2], mops[3]);
  }
}

/* ── Benchmark: scalar inplace ops ─────────────────────────────────── */

static void bench_scalar_inplace_ops(NumcCtx *ctx, size_t size) {
  printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
         "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
  printf("  SCALAR INPLACE  (%zu elements, %d iters)\n", size, ITERS);
  print_header("dtype");

  ScalarInplace ops[] = {numc_add_scalar_inplace, numc_sub_scalar_inplace,
                         numc_mul_scalar_inplace, numc_div_scalar_inplace};

  for (int d = 0; d < N_DTYPES; d++) {
    NumcDType dt = ALL_DTYPES[d];
    size_t shape[] = {size};

    char val[8];
    fill_value(dt, val);
    NumcArray *a = numc_array_fill(ctx, shape, 1, dt, val);
    if (!a) {
      fprintf(stderr, "  alloc failed for %s\n", dtype_name(dt));
      continue;
    }

    double us[4], mops[4];
    for (int op = 0; op < 4; op++) {
      us[op]   = bench_inplace(ops[op], a, 1.01, ITERS);
      mops[op] = size / us[op];
    }

    printf("  %-8s %8.2f %8.2f %8.2f %8.2f   %8.1f %8.1f %8.1f %8.1f\n",
           dtype_name(dt),
           us[0], us[1], us[2], us[3],
           mops[0], mops[1], mops[2], mops[3]);
  }
}

/* ── Benchmark: size scaling (scalar add) ──────────────────────────── */

static void bench_scaling(NumcCtx *ctx) {
  printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
         "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
  printf("  SIZE SCALING  (float32 add_scalar, %d iters)\n", ITERS);
  printf("\n  %10s %10s %10s %10s\n",
         "elements", "time (us)", "Mops/s", "GB/s");
  printf("  ──────────────────────────────────────────\n");

  size_t sizes[] = {100, 1000, 10000, 100000, 1000000};
  int nsizes = sizeof(sizes) / sizeof(sizes[0]);

  for (int s = 0; s < nsizes; s++) {
    size_t n = sizes[s];
    size_t shape[] = {n};
    float va = 1.5f;
    NumcArray *a   = numc_array_fill(ctx, shape, 1, NUMC_DTYPE_FLOAT32, &va);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
    if (!a || !out) continue;

    double us = bench_scalar(numc_add_scalar, a, 2.0, out, ITERS);
    double mops = n / us;
    /* bandwidth: reads 1 array + writes 1 = 2 * n * 4 bytes */
    double gbs = (2.0 * n * sizeof(float)) / (us * 1e3);

    printf("  %10zu %10.2f %10.1f %10.2f\n", n, us, mops, gbs);
  }
}

/* ── main ──────────────────────────────────────────────────────────── */

int main(void) {
  printf("\n  numc scalar operation benchmark\n");
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
         "\n");

  /* 1. Scalar allocating — all dtypes */
  NumcCtx *ctx = numc_ctx_create();
  if (!ctx) {
    fprintf(stderr, "Failed to create context\n");
    return 1;
  }
  bench_scalar_ops(ctx, 1000000);
  numc_ctx_free(ctx);

  /* 2. Scalar inplace — all dtypes */
  ctx = numc_ctx_create();
  bench_scalar_inplace_ops(ctx, 1000000);
  numc_ctx_free(ctx);

  /* 3. Size scaling */
  ctx = numc_ctx_create();
  bench_scaling(ctx);
  numc_ctx_free(ctx);

  printf("\n");
  return 0;
}
