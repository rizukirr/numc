/*
 * bench_unary.c — Element-wise unary operation benchmark
 *
 * Tests: log, exp, abs (allocating + inplace)
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

typedef int (*UnaryOp)(NumcArray *, NumcArray *);
typedef int (*UnaryInplace)(NumcArray *);

/* ── Timer ─────────────────────────────────────────────────────────── */

static double time_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static double bench_unary(UnaryOp op, NumcArray *a, NumcArray *out, int iters) {
  for (int i = 0; i < WARMUP; i++)
    op(a, out);

  double t0 = time_us();
  for (int i = 0; i < iters; i++)
    op(a, out);
  double t1 = time_us();

  return (t1 - t0) / iters;
}

static double bench_inplace(UnaryInplace op, NumcArray *a, int iters) {
  for (int i = 0; i < WARMUP; i++)
    op(a);

  double t0 = time_us();
  for (int i = 0; i < iters; i++)
    op(a);
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

static void print_header(const char *title) {
  printf("\n  %-8s %8s %8s %8s %8s   %8s %8s %8s %8s\n", title, "log", "exp",
         "abs", "sqrt", "log", "exp", "abs", "sqrt");
  printf("  %-8s %8s %8s %8s %8s   %8s %8s %8s %8s\n", "", "(us)", "(us)",
         "(us)", "(us)", "(Mop/s)", "(Mop/s)", "(Mop/s)", "(Mop/s)");
  printf("  ──────────────────────────────────────────────────────────────"
         "──────────────\n");
}

/* Positive values ≥ 1 — safe for log (no -inf), safe for abs. */
static void fill_value(NumcDType dt, char buf[static 8]) {
  memset(buf, 0, 8);
  switch (dt) {
  case NUMC_DTYPE_INT8:
    *(int8_t *)buf = 3;
    break;
  case NUMC_DTYPE_INT16:
    *(int16_t *)buf = 7;
    break;
  case NUMC_DTYPE_INT32:
    *(int32_t *)buf = 42;
    break;
  case NUMC_DTYPE_INT64:
    *(int64_t *)buf = 42;
    break;
  case NUMC_DTYPE_UINT8:
    *(uint8_t *)buf = 3;
    break;
  case NUMC_DTYPE_UINT16:
    *(uint16_t *)buf = 7;
    break;
  case NUMC_DTYPE_UINT32:
    *(uint32_t *)buf = 42;
    break;
  case NUMC_DTYPE_UINT64:
    *(uint64_t *)buf = 42;
    break;
  case NUMC_DTYPE_FLOAT32:
    *(float *)buf = 1.5f;
    break;
  case NUMC_DTYPE_FLOAT64:
    *(double *)buf = 1.5;
    break;
  }
}

/*
 * Small values for exp — keeps the result in range when cast back to integer.
 * exp(2) ≈ 7.39 → fits int8/uint8; no integer overflow.
 */
static void fill_value_exp(NumcDType dt, char buf[static 8]) {
  memset(buf, 0, 8);
  switch (dt) {
  case NUMC_DTYPE_INT8:
    *(int8_t *)buf = 2;
    break;
  case NUMC_DTYPE_INT16:
    *(int16_t *)buf = 2;
    break;
  case NUMC_DTYPE_INT32:
    *(int32_t *)buf = 2;
    break;
  case NUMC_DTYPE_INT64:
    *(int64_t *)buf = 2;
    break;
  case NUMC_DTYPE_UINT8:
    *(uint8_t *)buf = 2;
    break;
  case NUMC_DTYPE_UINT16:
    *(uint16_t *)buf = 2;
    break;
  case NUMC_DTYPE_UINT32:
    *(uint32_t *)buf = 2;
    break;
  case NUMC_DTYPE_UINT64:
    *(uint64_t *)buf = 2;
    break;
  case NUMC_DTYPE_FLOAT32:
    *(float *)buf = 1.5f;
    break;
  case NUMC_DTYPE_FLOAT64:
    *(double *)buf = 1.5;
    break;
  }
}

static const NumcDType ALL_DTYPES[] = {
    NUMC_DTYPE_INT8,    NUMC_DTYPE_UINT8,   NUMC_DTYPE_INT16, NUMC_DTYPE_UINT16,
    NUMC_DTYPE_INT32,   NUMC_DTYPE_UINT32,  NUMC_DTYPE_INT64, NUMC_DTYPE_UINT64,
    NUMC_DTYPE_FLOAT32, NUMC_DTYPE_FLOAT64,
};
static const int N_DTYPES = sizeof(ALL_DTYPES) / sizeof(ALL_DTYPES[0]);

static int dtype_is_unsigned(NumcDType dt) {
  return dt == NUMC_DTYPE_UINT8 || dt == NUMC_DTYPE_UINT16 ||
         dt == NUMC_DTYPE_UINT32 || dt == NUMC_DTYPE_UINT64;
}

/* ── Benchmark: unary allocating ops ───────────────────────────────── */

static void bench_unary_ops(NumcCtx *ctx, size_t size) {
  printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
         "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
  printf("  UNARY OPS  (%zu elements, %d iters)\n", size, ITERS);
  print_header("dtype");

  for (int d = 0; d < N_DTYPES; d++) {
    NumcDType dt = ALL_DTYPES[d];
    size_t shape[] = {size};

    char val[8], val_exp[8];
    fill_value(dt, val);
    fill_value_exp(dt, val_exp);

    NumcArray *a_log = numc_array_fill(ctx, shape, 1, dt, val);
    NumcArray *a_exp = numc_array_fill(ctx, shape, 1, dt, val_exp);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, dt);
    if (!a_log || !a_exp || !out) {
      fprintf(stderr, "  alloc failed for %s\n", dtype_name(dt));
      continue;
    }

    double us[2], mops[2];
    us[0] = bench_unary(numc_log, a_log, out, ITERS);
    us[1] = bench_unary(numc_exp, a_exp, out, ITERS);
    for (int i = 0; i < 2; i++)
      mops[i] = size / us[i];

    NumcArray *a_sqrt = numc_array_fill(ctx, shape, 1, dt, val);
    if (!a_sqrt) {
      fprintf(stderr, "  alloc failed for %s sqrt\n", dtype_name(dt));
      continue;
    }
    double us_sqrt = bench_unary(numc_sqrt, a_sqrt, out, ITERS);
    double mops_sqrt = size / us_sqrt;

    if (dtype_is_unsigned(dt)) {
      printf("  %-8s %8.2f %8.2f %8s %8.2f   %8.1f %8.1f %8s %8.1f\n",
             dtype_name(dt), us[0], us[1], "n/a", us_sqrt, mops[0], mops[1],
             "n/a", mops_sqrt);
    } else {
      NumcArray *a_abs = numc_array_fill(ctx, shape, 1, dt, val);
      if (!a_abs) {
        fprintf(stderr, "  alloc failed for %s abs\n", dtype_name(dt));
        printf("  %-8s %8.2f %8.2f %8s %8.2f   %8.1f %8.1f %8s %8.1f\n",
               dtype_name(dt), us[0], us[1], "err", us_sqrt, mops[0], mops[1],
               "err", mops_sqrt);
        continue;
      }
      double us_abs = bench_unary(numc_abs, a_abs, out, ITERS);
      double mops_abs = size / us_abs;
      printf("  %-8s %8.2f %8.2f %8.2f %8.2f   %8.1f %8.1f %8.1f %8.1f\n",
             dtype_name(dt), us[0], us[1], us_abs, us_sqrt, mops[0], mops[1],
             mops_abs, mops_sqrt);
    }
  }
}

/* ── Benchmark: unary inplace ops ──────────────────────────────────── */

static void bench_unary_inplace_ops(NumcCtx *ctx, size_t size) {
  printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
         "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
  printf("  UNARY INPLACE  (%zu elements, %d iters)\n", size, ITERS);
  print_header("dtype");

  for (int d = 0; d < N_DTYPES; d++) {
    NumcDType dt = ALL_DTYPES[d];
    size_t shape[] = {size};

    char val[8], val_exp[8];
    fill_value(dt, val);
    fill_value_exp(dt, val_exp);

    NumcArray *a_log = numc_array_fill(ctx, shape, 1, dt, val);
    NumcArray *a_exp = numc_array_fill(ctx, shape, 1, dt, val_exp);
    if (!a_log || !a_exp) {
      fprintf(stderr, "  alloc failed for %s\n", dtype_name(dt));
      continue;
    }

    double us[2], mops[2];
    us[0] = bench_inplace(numc_log_inplace, a_log, ITERS);
    us[1] = bench_inplace(numc_exp_inplace, a_exp, ITERS);
    for (int i = 0; i < 2; i++)
      mops[i] = size / us[i];

    NumcArray *a_sqrt = numc_array_fill(ctx, shape, 1, dt, val);
    if (!a_sqrt) {
      fprintf(stderr, "  alloc failed for %s sqrt\n", dtype_name(dt));
      continue;
    }
    double us_sqrt = bench_inplace(numc_sqrt_inplace, a_sqrt, ITERS);
    double mops_sqrt = size / us_sqrt;

    if (dtype_is_unsigned(dt)) {
      printf("  %-8s %8.2f %8.2f %8s %8.2f   %8.1f %8.1f %8s %8.1f\n",
             dtype_name(dt), us[0], us[1], "n/a", us_sqrt, mops[0], mops[1],
             "n/a", mops_sqrt);
    } else {
      NumcArray *a_abs = numc_array_fill(ctx, shape, 1, dt, val);
      if (!a_abs) {
        fprintf(stderr, "  alloc failed for %s abs\n", dtype_name(dt));
        printf("  %-8s %8.2f %8.2f %8s %8.2f   %8.1f %8.1f %8s %8.1f\n",
               dtype_name(dt), us[0], us[1], "err", us_sqrt, mops[0], mops[1],
               "err", mops_sqrt);
        continue;
      }
      double us_abs = bench_inplace(numc_abs_inplace, a_abs, ITERS);
      double mops_abs = size / us_abs;
      printf("  %-8s %8.2f %8.2f %8.2f %8.2f   %8.1f %8.1f %8.1f %8.1f\n",
             dtype_name(dt), us[0], us[1], us_abs, us_sqrt, mops[0], mops[1],
             mops_abs, mops_sqrt);
    }
  }
}

/* ── Benchmark: size scaling (float32 sqrt) ────────────────────────── */

static void bench_scaling(NumcCtx *ctx) {
  printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
         "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
  printf("  SIZE SCALING  (float32 sqrt, %d iters)\n", ITERS);
  printf("\n  %10s %10s %10s %10s\n", "elements", "time (us)", "Mops/s",
         "GB/s");
  printf("  ──────────────────────────────────────────\n");

  size_t sizes[] = {100, 1000, 10000, 100000, 1000000};
  int nsizes = sizeof(sizes) / sizeof(sizes[0]);

  for (int s = 0; s < nsizes; s++) {
    size_t n = sizes[s];
    size_t shape[] = {n};
    float va = 1.5f;
    NumcArray *a = numc_array_fill(ctx, shape, 1, NUMC_DTYPE_FLOAT32, &va);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
    if (!a || !out)
      continue;

    double us = bench_unary(numc_sqrt, a, out, ITERS);
    double mops = n / us;
    /* bandwidth: reads 1 array + writes 1 = 2 * n * 4 bytes */
    double gbs = (2.0 * n * sizeof(float)) / (us * 1e3);

    printf("  %10zu %10.2f %10.1f %10.2f\n", n, us, mops, gbs);
  }
}

/* ── main ──────────────────────────────────────────────────────────── */

int main(void) {
  printf("\n  numc unary operation benchmark\n");
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

  /* 1. Unary allocating — all dtypes */
  NumcCtx *ctx = numc_ctx_create();
  if (!ctx) {
    fprintf(stderr, "Failed to create context\n");
    return 1;
  }
  bench_unary_ops(ctx, 1000000);
  numc_ctx_free(ctx);

  /* 2. Unary inplace — all dtypes */
  ctx = numc_ctx_create();
  bench_unary_inplace_ops(ctx, 1000000);
  numc_ctx_free(ctx);

  /* 3. Size scaling */
  ctx = numc_ctx_create();
  bench_scaling(ctx);
  numc_ctx_free(ctx);

  printf("\n");
  return 0;
}
