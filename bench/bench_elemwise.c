/*
 * bench_elemwise.c — Element-wise binary (ND) operation benchmark
 *
 * Tests: add, sub, mul, div on two arrays
 * Varies: dtype (all 10), array size, memory layout (contiguous vs strided)
 * Reports: avg time (us), throughput (Mops/s)
 */

#include <numc/numc.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

/* ── Config ────────────────────────────────────────────────────────── */

#define WARMUP 20
#define ITERS  200

typedef int (*ElemwiseOp)(const NumcArray *, const NumcArray *, NumcArray *);

/* ── Timer ─────────────────────────────────────────────────────────── */

static double time_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static double bench(ElemwiseOp op, const NumcArray *a, const NumcArray *b,
                    NumcArray *out, int iters) {
  for (int i = 0; i < WARMUP; i++)
    op(a, b, out);

  double t0 = time_us();
  for (int i = 0; i < iters; i++)
    op(a, b, out);
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

/* ── Benchmark: contiguous binary ops ──────────────────────────────── */

static void bench_contiguous(NumcCtx *ctx, size_t size) {
  printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
         "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
  printf("  CONTIGUOUS BINARY  (%zu elements, %d iters)\n", size, ITERS);
  print_header("dtype");

  ElemwiseOp ops[] = {numc_add, numc_sub, numc_mul, numc_div};

  for (int d = 0; d < N_DTYPES; d++) {
    NumcDType dt = ALL_DTYPES[d];
    size_t shape[] = {size};

    char val[8];
    fill_value(dt, val);
    NumcArray *a   = numc_array_fill(ctx, shape, 1, dt, val);
    NumcArray *b   = numc_array_fill(ctx, shape, 1, dt, val);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, dt);
    if (!a || !b || !out) {
      fprintf(stderr, "  alloc failed for %s\n", dtype_name(dt));
      continue;
    }

    double us[4], mops[4];
    for (int op = 0; op < 4; op++) {
      us[op]   = bench(ops[op], a, b, out, ITERS);
      mops[op] = size / us[op];
    }

    printf("  %-8s %8.2f %8.2f %8.2f %8.2f   %8.1f %8.1f %8.1f %8.1f\n",
           dtype_name(dt),
           us[0], us[1], us[2], us[3],
           mops[0], mops[1], mops[2], mops[3]);
  }
}

/* ── Benchmark: strided (transposed view) ──────────────────────────── */

static void bench_strided(NumcCtx *ctx, size_t rows, size_t cols) {
  size_t total = rows * cols;

  printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
         "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
  printf("  STRIDED  (%zux%zu transposed, %zu elements, %d iters)\n",
         rows, cols, total, ITERS);
  print_header("dtype");

  ElemwiseOp ops[] = {numc_add, numc_sub, numc_mul, numc_div};

  NumcDType dtypes[] = {NUMC_DTYPE_INT32, NUMC_DTYPE_FLOAT32, NUMC_DTYPE_FLOAT64};
  const int ndtypes = sizeof(dtypes) / sizeof(dtypes[0]);

  for (int d = 0; d < ndtypes; d++) {
    NumcDType dt = dtypes[d];
    size_t shape[] = {rows, cols};

    char val[8];
    fill_value(dt, val);
    NumcArray *a   = numc_array_fill(ctx, shape, 2, dt, val);
    NumcArray *b   = numc_array_fill(ctx, shape, 2, dt, val);
    NumcArray *out = numc_array_zeros(ctx, shape, 2, dt);
    if (!a || !b || !out) continue;

    size_t axes[] = {1, 0};
    numc_array_transpose(a, axes);
    numc_array_transpose(b, axes);
    size_t out_shape[] = {cols, rows};
    numc_array_reshape(out, out_shape, 2);

    double us[4], mops[4];
    for (int op = 0; op < 4; op++) {
      us[op]   = bench(ops[op], a, b, out, ITERS);
      mops[op] = total / us[op];
    }

    printf("  %-8s %8.2f %8.2f %8.2f %8.2f   %8.1f %8.1f %8.1f %8.1f\n",
           dtype_name(dt),
           us[0], us[1], us[2], us[3],
           mops[0], mops[1], mops[2], mops[3]);
  }
}

/* ── Benchmark: broadcast patterns ─────────────────────────────────── */

static void bench_bcast_pattern(NumcCtx *ctx, const size_t *sa, size_t na,
                                const size_t *sb, size_t nb, const size_t *so,
                                size_t no, size_t total) {
  ElemwiseOp ops[] = {numc_add, numc_sub, numc_mul, numc_div};
  NumcDType dtypes[] = {NUMC_DTYPE_INT32, NUMC_DTYPE_FLOAT32, NUMC_DTYPE_FLOAT64};

  for (int d = 0; d < 3; d++) {
    NumcDType dt = dtypes[d];
    char val[8];
    fill_value(dt, val);
    NumcArray *a   = numc_array_fill(ctx, sa, na, dt, val);
    NumcArray *b   = numc_array_fill(ctx, sb, nb, dt, val);
    NumcArray *out = numc_array_zeros(ctx, so, no, dt);
    if (!a || !b || !out) continue;

    double us[4], mops[4];
    for (int op = 0; op < 4; op++) {
      us[op]   = bench(ops[op], a, b, out, ITERS);
      mops[op] = total / us[op];
    }

    printf("  %-8s %8.2f %8.2f %8.2f %8.2f   %8.1f %8.1f %8.1f %8.1f\n",
           dtype_name(dt),
           us[0], us[1], us[2], us[3],
           mops[0], mops[1], mops[2], mops[3]);
  }
}

static void bench_broadcast(NumcCtx *ctx, size_t M, size_t N) {
  size_t total = M * N;

  /* Row broadcast: (1,N) + (M,N) → (M,N) */
  printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
         "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
  printf("  BROADCAST ROW  (1,%zu) + (%zu,%zu) -> (%zu,%zu), %d iters\n",
         N, M, N, M, N, ITERS);
  print_header("dtype");
  {
    size_t sa[] = {1, N}, sb[] = {M, N}, so[] = {M, N};
    bench_bcast_pattern(ctx, sa, 2, sb, 2, so, 2, total);
  }

  /* Outer broadcast: (M,1) + (1,N) → (M,N) */
  printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
         "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
  printf("  BROADCAST OUTER  (%zu,1) + (1,%zu) -> (%zu,%zu), %d iters\n",
         M, N, M, N, ITERS);
  print_header("dtype");
  {
    size_t sa[] = {M, 1}, sb[] = {1, N}, so[] = {M, N};
    bench_bcast_pattern(ctx, sa, 2, sb, 2, so, 2, total);
  }

  /* Rank broadcast: (N,) + (M,N) → (M,N) */
  printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
         "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
  printf("  BROADCAST RANK  (%zu,) + (%zu,%zu) -> (%zu,%zu), %d iters\n",
         N, M, N, M, N, ITERS);
  print_header("dtype");
  {
    size_t sa[] = {N}, sb[] = {M, N}, so[] = {M, N};
    bench_bcast_pattern(ctx, sa, 1, sb, 2, so, 2, total);
  }
}

/* ── Benchmark: scaling across sizes ───────────────────────────────── */

static void bench_scaling(NumcCtx *ctx) {
  printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
         "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
  printf("  SIZE SCALING  (float32 add, %d iters)\n", ITERS);
  printf("\n  %10s %10s %10s %10s\n", "elements", "time (us)", "Mops/s", "GB/s");
  printf("  ──────────────────────────────────────────\n");

  size_t sizes[] = {100, 1000, 10000, 100000, 1000000};
  int nsizes = sizeof(sizes) / sizeof(sizes[0]);

  for (int s = 0; s < nsizes; s++) {
    size_t n = sizes[s];
    size_t shape[] = {n};
    float va = 1.5f, vb = 2.5f;
    NumcArray *a   = numc_array_fill(ctx, shape, 1, NUMC_DTYPE_FLOAT32, &va);
    NumcArray *b   = numc_array_fill(ctx, shape, 1, NUMC_DTYPE_FLOAT32, &vb);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_FLOAT32);
    if (!a || !b || !out) continue;

    double us = bench(numc_add, a, b, out, ITERS);
    double mops = n / us;
    double gbs = (3.0 * n * sizeof(float)) / (us * 1e3);

    printf("  %10zu %10.2f %10.1f %10.2f\n", n, us, mops, gbs);
  }
}

/* ── main ──────────────────────────────────────────────────────────── */

int main(void) {
  printf("\n  numc element-wise binary benchmark\n");
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

  /* 1. Contiguous — all dtypes, 1M elements */
  NumcCtx *ctx = numc_ctx_create();
  if (!ctx) {
    fprintf(stderr, "Failed to create context\n");
    return 1;
  }
  bench_contiguous(ctx, 1000000);
  numc_ctx_free(ctx);

  /* 2. Strided (transposed) */
  ctx = numc_ctx_create();
  bench_strided(ctx, 1000, 1000);
  numc_ctx_free(ctx);

  /* 3. Broadcast */
  ctx = numc_ctx_create();
  bench_broadcast(ctx, 1000, 1000);
  numc_ctx_free(ctx);

  /* 4. Size scaling */
  ctx = numc_ctx_create();
  bench_scaling(ctx);
  numc_ctx_free(ctx);

  printf("\n");
  return 0;
}
