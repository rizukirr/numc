/*
 * bench_pow.c — Element-wise pow operation benchmark
 *
 * Tests: pow (allocating + inplace) on two arrays
 * Varies: dtype (all 10), array size
 * Reports: avg time (us), throughput (Mops/s)
 */

#include <numc/numc.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

/* ── Config ────────────────────────────────────────────────────────── */

#define WARMUP 20
#define ITERS 200

typedef int (*PowOp)(NumcArray *, NumcArray *, NumcArray *);
typedef int (*PowInplace)(NumcArray *, NumcArray *);

/* ── Timer ─────────────────────────────────────────────────────────── */

static double time_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static double bench_pow_op(PowOp op, NumcArray *a, NumcArray *b, NumcArray *out,
                           int iters) {
  for (int i = 0; i < WARMUP; i++)
    op(a, b, out);

  double t0 = time_us();
  for (int i = 0; i < iters; i++)
    op(a, b, out);
  double t1 = time_us();

  return (t1 - t0) / iters;
}

static double bench_pow_inplace_op(PowInplace op, NumcArray *a, NumcArray *b,
                                   int iters) {
  for (int i = 0; i < WARMUP; i++)
    op(a, b);

  double t0 = time_us();
  for (int i = 0; i < iters; i++)
    op(a, b);
  double t1 = time_us();

  return (t1 - t0) / iters;
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

/* Base values: small positive, safe for all types */
static void fill_base(NumcDType dt, char buf[static 8]) {
  memset(buf, 0, 8);
  switch (dt) {
  case NUMC_DTYPE_INT8:
    *(int8_t *)buf = 3;
    break;
  case NUMC_DTYPE_INT16:
    *(int16_t *)buf = 3;
    break;
  case NUMC_DTYPE_INT32:
    *(int32_t *)buf = 3;
    break;
  case NUMC_DTYPE_INT64:
    *(int64_t *)buf = 3;
    break;
  case NUMC_DTYPE_UINT8:
    *(uint8_t *)buf = 3;
    break;
  case NUMC_DTYPE_UINT16:
    *(uint16_t *)buf = 3;
    break;
  case NUMC_DTYPE_UINT32:
    *(uint32_t *)buf = 3;
    break;
  case NUMC_DTYPE_UINT64:
    *(uint64_t *)buf = 3;
    break;
  case NUMC_DTYPE_FLOAT32:
    *(float *)buf = 2.0f;
    break;
  case NUMC_DTYPE_FLOAT64:
    *(double *)buf = 2.0;
    break;
  }
}

/* Exponent values: small to avoid integer overflow */
static void fill_exp(NumcDType dt, char buf[static 8]) {
  memset(buf, 0, 8);
  switch (dt) {
  case NUMC_DTYPE_INT8:
    *(int8_t *)buf = 3;
    break;
  case NUMC_DTYPE_INT16:
    *(int16_t *)buf = 3;
    break;
  case NUMC_DTYPE_INT32:
    *(int32_t *)buf = 3;
    break;
  case NUMC_DTYPE_INT64:
    *(int64_t *)buf = 3;
    break;
  case NUMC_DTYPE_UINT8:
    *(uint8_t *)buf = 3;
    break;
  case NUMC_DTYPE_UINT16:
    *(uint16_t *)buf = 3;
    break;
  case NUMC_DTYPE_UINT32:
    *(uint32_t *)buf = 3;
    break;
  case NUMC_DTYPE_UINT64:
    *(uint64_t *)buf = 3;
    break;
  case NUMC_DTYPE_FLOAT32:
    *(float *)buf = 3.0f;
    break;
  case NUMC_DTYPE_FLOAT64:
    *(double *)buf = 3.0;
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

/* ── Benchmark: contiguous pow ─────────────────────────────────────── */

static void bench_contiguous(NumcCtx *ctx, size_t size) {
  printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
         "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
  printf("  POW CONTIGUOUS  (%zu elements, %d iters)\n", size, ITERS);
  printf("\n  %-8s %10s %10s   %10s %10s\n", "dtype", "pow", "inplace", "pow",
         "inplace");
  printf("  %-8s %10s %10s   %10s %10s\n", "", "(us)", "(us)", "(Mop/s)",
         "(Mop/s)");
  printf("  ──────────────────────────────────────────────────────────\n");

  for (int d = 0; d < N_DTYPES; d++) {
    NumcDType dt = ALL_DTYPES[d];
    size_t shape[] = {size};

    char vbase[8], vexp[8];
    fill_base(dt, vbase);
    fill_exp(dt, vexp);

    NumcArray *a = numc_array_fill(ctx, shape, 1, dt, vbase);
    NumcArray *b = numc_array_fill(ctx, shape, 1, dt, vexp);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, dt);
    NumcArray *a_ip = numc_array_fill(ctx, shape, 1, dt, vbase);
    if (!a || !b || !out || !a_ip) {
      fprintf(stderr, "  alloc failed for %s\n", dtype_name(dt));
      continue;
    }

    double us_pow = bench_pow_op(numc_pow, a, b, out, ITERS);
    double us_ip = bench_pow_inplace_op(numc_pow_inplace, a_ip, b, ITERS);

    printf("  %-8s %10.2f %10.2f   %10.1f %10.1f\n", dtype_name(dt), us_pow,
           us_ip, size / us_pow, size / us_ip);
  }
}

/* ── Benchmark: size scaling ───────────────────────────────────────── */

static void bench_scaling(NumcCtx *ctx) {
  printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
         "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
  printf("  SIZE SCALING  (float32 pow, %d iters)\n", ITERS);
  printf("\n  %10s %10s %10s %10s\n", "elements", "time (us)", "Mops/s",
         "GB/s");
  printf("  ──────────────────────────────────────────\n");

  size_t sizes[] = {100, 1000, 10000, 100000, 1000000};
  int nsizes = sizeof(sizes) / sizeof(sizes[0]);

  for (int s = 0; s < nsizes; s++) {
    size_t n = sizes[s];
    size_t shape[] = {n};
    float vb = 2.0f, ve = 3.0f;
    NumcArray *a = numc_array_fill(ctx, shape, 1, NUMC_DTYPE_FLOAT32, &vb);
    NumcArray *b = numc_array_fill(ctx, shape, 1, NUMC_DTYPE_FLOAT32, &ve);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
    if (!a || !b || !out)
      continue;

    double us = bench_pow_op(numc_pow, a, b, out, ITERS);
    double mops = n / us;
    /* bandwidth: reads 2 arrays + writes 1 = 3 * n * 4 bytes */
    double gbs = (3.0 * n * sizeof(float)) / (us * 1e3);

    printf("  %10zu %10.2f %10.1f %10.2f\n", n, us, mops, gbs);
  }
}

/* ── Benchmark: int vs float comparison ────────────────────────────── */

static void bench_int_vs_float(NumcCtx *ctx, size_t size) {
  printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
         "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
  printf("  INT vs FLOAT path comparison  (%zu elements, %d iters)\n", size,
         ITERS);
  printf("  int path: exponentiation by squaring (exact)\n");
  printf("  float path: fused exp(b * log(a))\n");
  printf("\n  %-8s %10s %10s\n", "dtype", "time (us)", "Mop/s");
  printf("  ──────────────────────────────\n");

  NumcDType dtypes[] = {NUMC_DTYPE_INT32, NUMC_DTYPE_INT64,
                        NUMC_DTYPE_FLOAT32, NUMC_DTYPE_FLOAT64};
  int ndtypes = sizeof(dtypes) / sizeof(dtypes[0]);

  for (int d = 0; d < ndtypes; d++) {
    NumcDType dt = dtypes[d];
    size_t shape[] = {size};

    char vbase[8], vexp[8];
    fill_base(dt, vbase);
    fill_exp(dt, vexp);

    NumcArray *a = numc_array_fill(ctx, shape, 1, dt, vbase);
    NumcArray *b = numc_array_fill(ctx, shape, 1, dt, vexp);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, dt);
    if (!a || !b || !out)
      continue;

    double us = bench_pow_op(numc_pow, a, b, out, ITERS);
    printf("  %-8s %10.2f %10.1f\n", dtype_name(dt), us, size / us);
  }
}

/* ── main ──────────────────────────────────────────────────────────── */

int main(void) {
  printf("\n  numc pow benchmark\n");
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

  /* 1. Contiguous — all dtypes */
  NumcCtx *ctx = numc_ctx_create();
  if (!ctx) {
    fprintf(stderr, "Failed to create context\n");
    return 1;
  }
  bench_contiguous(ctx, 1000000);
  numc_ctx_free(ctx);

  /* 2. Int vs float path comparison */
  ctx = numc_ctx_create();
  bench_int_vs_float(ctx, 1000000);
  numc_ctx_free(ctx);

  /* 3. Size scaling */
  ctx = numc_ctx_create();
  bench_scaling(ctx);
  numc_ctx_free(ctx);

  printf("\n");
  return 0;
}
