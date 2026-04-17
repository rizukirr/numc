/*
 * bench/numc/bench.c — Comprehensive numc benchmark (CSV output)
 *
 * Benchmarks ALL numc math operations and outputs CSV to stdout.
 * CSV columns:
 * library,category,operation,dtype,size,shape,time_us,throughput_mops
 *
 * Build: linked via CMakeLists.txt against numc::numc
 * Run:   ./bench_numc_csv > results.csv
 */

#include <numc/numc.h>
#include <stdio.h>
#include <string.h>
#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <time.h>
#endif
#include <float.h>

/* -- Config ---------------------------------------------------------- */

#define WARMUP 20
#define ITERS  200
#define SIZE   1000000

/* -- Timer ----------------------------------------------------------- */

static double time_us(void) {
#if defined(_WIN32) || defined(_WIN64)
  LARGE_INTEGER freq, cnt;
  QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&cnt);
  return (double)cnt.QuadPart / (double)freq.QuadPart * 1e6;
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
#endif
}

/* Burn ~200ms of CPU to force turbo-boost ramp-up on hybrid CPUs. */
static void bench_cpu_warmup(void) {
  volatile double sink = 0.0;
  double t0 = time_us();
  while (time_us() - t0 < 200000.0)
    for (int i = 0; i < 100000; i++)
      sink += (double)i * 1.0001;
  (void)sink;
}

/* -- Helpers --------------------------------------------------------- */

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

/* Small values safe for exp (avoids overflow) */
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

static void fill_pow_exp(NumcDType dt, char buf[static 8]) {
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
    NUMC_DTYPE_INT8,    NUMC_DTYPE_UINT8,   NUMC_DTYPE_INT16, NUMC_DTYPE_UINT16,
    NUMC_DTYPE_INT32,   NUMC_DTYPE_UINT32,  NUMC_DTYPE_INT64, NUMC_DTYPE_UINT64,
    NUMC_DTYPE_FLOAT32, NUMC_DTYPE_FLOAT64,
};
static const int N_DTYPES = sizeof(ALL_DTYPES) / sizeof(ALL_DTYPES[0]);

static const NumcDType FLOAT_DTYPES[] = {
    NUMC_DTYPE_FLOAT32,
    NUMC_DTYPE_FLOAT64,
};
static const int N_FLOAT = 2;

static int dtype_is_unsigned(NumcDType dt) {
  return dt == NUMC_DTYPE_UINT8 || dt == NUMC_DTYPE_UINT16 ||
         dt == NUMC_DTYPE_UINT32 || dt == NUMC_DTYPE_UINT64;
}

/* Print a CSV row */
static void csv(const char *cat, const char *op, const char *dt, size_t size,
                const char *shape, double us) {
  printf("numc,%s,%s,%s,%zu,%s,%.4f,%.4f\n", cat, op, dt, size, shape, us,
         size / us);
}

/* -- Binary element-wise -------------------------------------------- */

typedef int (*BinaryOp)(const NumcArray *, const NumcArray *, NumcArray *);

static void bench_binary(const char *name, BinaryOp op, size_t size) {
  for (int d = 0; d < N_DTYPES; d++) {
    NumcDType dt = ALL_DTYPES[d];
    NumcCtx *ctx = numc_ctx_create();
    size_t shape[] = {size};
    char val[8];
    fill_value(dt, val);

    NumcArray *a = numc_array_fill(ctx, shape, 1, dt, val);
    NumcArray *b = numc_array_fill(ctx, shape, 1, dt, val);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, dt);
    if (!a || !b || !out) {
      numc_ctx_free(ctx);
      continue;
    }

    for (int i = 0; i < WARMUP; i++)
      op(a, b, out);
    double t0 = time_us();
    for (int i = 0; i < ITERS; i++)
      op(a, b, out);
    double us = (time_us() - t0) / ITERS;

    char shape_str[32];
    snprintf(shape_str, sizeof(shape_str), "(%zu)", size);
    csv("binary", name, dtype_name(dt), size, shape_str, us);
    numc_ctx_free(ctx);
  }
}

/* -- Scalar ops ----------------------------------------------------- */

typedef int (*ScalarOp)(const NumcArray *, double, NumcArray *);

static void bench_scalar_op(const char *name, ScalarOp op, size_t size) {
  for (int d = 0; d < N_DTYPES; d++) {
    NumcDType dt = ALL_DTYPES[d];
    NumcCtx *ctx = numc_ctx_create();
    size_t shape[] = {size};
    char val[8];
    fill_value(dt, val);

    NumcArray *a = numc_array_fill(ctx, shape, 1, dt, val);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, dt);
    if (!a || !out) {
      numc_ctx_free(ctx);
      continue;
    }

    for (int i = 0; i < WARMUP; i++)
      op(a, 2.0, out);
    double t0 = time_us();
    for (int i = 0; i < ITERS; i++)
      op(a, 2.0, out);
    double us = (time_us() - t0) / ITERS;

    char shape_str[32];
    snprintf(shape_str, sizeof(shape_str), "(%zu)", size);
    csv("scalar", name, dtype_name(dt), size, shape_str, us);
    numc_ctx_free(ctx);
  }
}

/* -- Scalar inplace ops --------------------------------------------- */

typedef int (*ScalarInplace)(NumcArray *, double);

static void bench_scalar_inplace(const char *name, ScalarInplace op,
                                 size_t size) {
  for (int d = 0; d < N_DTYPES; d++) {
    NumcDType dt = ALL_DTYPES[d];
    NumcCtx *ctx = numc_ctx_create();
    size_t shape[] = {size};
    char val[8];
    fill_value(dt, val);

    NumcArray *a = numc_array_fill(ctx, shape, 1, dt, val);
    if (!a) {
      numc_ctx_free(ctx);
      continue;
    }

    for (int i = 0; i < WARMUP; i++)
      op(a, 1.01);
    double t0 = time_us();
    for (int i = 0; i < ITERS; i++)
      op(a, 1.01);
    double us = (time_us() - t0) / ITERS;

    char shape_str[32];
    snprintf(shape_str, sizeof(shape_str), "(%zu)", size);
    csv("scalar_inplace", name, dtype_name(dt), size, shape_str, us);
    numc_ctx_free(ctx);
  }
}

/* -- Unary ops ------------------------------------------------------ */

typedef int (*UnaryOp)(NumcArray *, NumcArray *);

static void bench_unary(const char *name, UnaryOp op, size_t size,
                        int skip_unsigned, int use_exp_fill) {
  for (int d = 0; d < N_DTYPES; d++) {
    NumcDType dt = ALL_DTYPES[d];
    if (skip_unsigned && dtype_is_unsigned(dt))
      continue;

    NumcCtx *ctx = numc_ctx_create();
    size_t shape[] = {size};
    char val[8];
    if (use_exp_fill)
      fill_value_exp(dt, val);
    else
      fill_value(dt, val);

    NumcArray *a = numc_array_fill(ctx, shape, 1, dt, val);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, dt);
    if (!a || !out) {
      numc_ctx_free(ctx);
      continue;
    }

    for (int i = 0; i < WARMUP; i++)
      op(a, out);
    double t0 = time_us();
    for (int i = 0; i < ITERS; i++)
      op(a, out);
    double us = (time_us() - t0) / ITERS;

    char shape_str[32];
    snprintf(shape_str, sizeof(shape_str), "(%zu)", size);
    csv("unary", name, dtype_name(dt), size, shape_str, us);
    numc_ctx_free(ctx);
  }
}

/* -- Unary inplace ops ---------------------------------------------- */

typedef int (*UnaryInplace)(NumcArray *);

static void bench_unary_inplace(const char *name, UnaryInplace op, size_t size,
                                int skip_unsigned, int use_exp_fill) {
  for (int d = 0; d < N_DTYPES; d++) {
    NumcDType dt = ALL_DTYPES[d];
    if (skip_unsigned && dtype_is_unsigned(dt))
      continue;

    NumcCtx *ctx = numc_ctx_create();
    size_t shape[] = {size};
    char val[8];
    if (use_exp_fill)
      fill_value_exp(dt, val);
    else
      fill_value(dt, val);

    NumcArray *a = numc_array_fill(ctx, shape, 1, dt, val);
    if (!a) {
      numc_ctx_free(ctx);
      continue;
    }

    for (int i = 0; i < WARMUP; i++)
      op(a);
    double t0 = time_us();
    for (int i = 0; i < ITERS; i++)
      op(a);
    double us = (time_us() - t0) / ITERS;

    char shape_str[32];
    snprintf(shape_str, sizeof(shape_str), "(%zu)", size);
    csv("unary_inplace", name, dtype_name(dt), size, shape_str, us);
    numc_ctx_free(ctx);
  }
}

/* -- Clip ----------------------------------------------------------- */

static void bench_clip(size_t size) {
  for (int d = 0; d < N_DTYPES; d++) {
    NumcDType dt = ALL_DTYPES[d];
    NumcCtx *ctx = numc_ctx_create();
    size_t shape[] = {size};
    char val[8];
    fill_value(dt, val);

    NumcArray *a = numc_array_fill(ctx, shape, 1, dt, val);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, dt);
    if (!a || !out) {
      numc_ctx_free(ctx);
      continue;
    }

    for (int i = 0; i < WARMUP; i++)
      numc_clip(a, out, 1.0, 5.0);
    double t0 = time_us();
    for (int i = 0; i < ITERS; i++)
      numc_clip(a, out, 1.0, 5.0);
    double us = (time_us() - t0) / ITERS;

    char shape_str[32];
    snprintf(shape_str, sizeof(shape_str), "(%zu)", size);
    csv("unary", "clip", dtype_name(dt), size, shape_str, us);
    numc_ctx_free(ctx);
  }
}

/* -- Comparison ops ------------------------------------------------- */

static void bench_comparison(const char *name, BinaryOp op, size_t size) {
  for (int d = 0; d < N_DTYPES; d++) {
    NumcDType dt = ALL_DTYPES[d];
    NumcCtx *ctx = numc_ctx_create();
    size_t shape[] = {size};
    char val[8];
    fill_value(dt, val);

    NumcArray *a = numc_array_fill(ctx, shape, 1, dt, val);
    NumcArray *b = numc_array_fill(ctx, shape, 1, dt, val);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_UINT8);
    if (!a || !b || !out) {
      numc_ctx_free(ctx);
      continue;
    }

    for (int i = 0; i < WARMUP; i++)
      op(a, b, out);
    double t0 = time_us();
    for (int i = 0; i < ITERS; i++)
      op(a, b, out);
    double us = (time_us() - t0) / ITERS;

    char shape_str[32];
    snprintf(shape_str, sizeof(shape_str), "(%zu)", size);
    csv("comparison", name, dtype_name(dt), size, shape_str, us);
    numc_ctx_free(ctx);
  }
}

/* -- Comparison scalar ops ------------------------------------------ */

static void bench_comparison_scalar(const char *name, ScalarOp op,
                                    size_t size) {
  for (int d = 0; d < N_DTYPES; d++) {
    NumcDType dt = ALL_DTYPES[d];
    NumcCtx *ctx = numc_ctx_create();
    size_t shape[] = {size};
    char val[8];
    fill_value(dt, val);

    NumcArray *a = numc_array_fill(ctx, shape, 1, dt, val);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_UINT8);
    if (!a || !out) {
      numc_ctx_free(ctx);
      continue;
    }

    for (int i = 0; i < WARMUP; i++)
      op(a, 2.0, out);
    double t0 = time_us();
    for (int i = 0; i < ITERS; i++)
      op(a, 2.0, out);
    double us = (time_us() - t0) / ITERS;

    char shape_str[32];
    snprintf(shape_str, sizeof(shape_str), "(%zu)", size);
    csv("comparison_scalar", name, dtype_name(dt), size, shape_str, us);
    numc_ctx_free(ctx);
  }
}

/* -- Pow ------------------------------------------------------------ */

static void bench_pow(size_t size) {
  for (int d = 0; d < N_DTYPES; d++) {
    NumcDType dt = ALL_DTYPES[d];
    NumcCtx *ctx = numc_ctx_create();
    size_t shape[] = {size};
    char vbase[8], vexp[8];
    fill_value(dt, vbase);
    fill_pow_exp(dt, vexp);

    NumcArray *a = numc_array_fill(ctx, shape, 1, dt, vbase);
    NumcArray *b = numc_array_fill(ctx, shape, 1, dt, vexp);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, dt);
    if (!a || !b || !out) {
      numc_ctx_free(ctx);
      continue;
    }

    for (int i = 0; i < WARMUP; i++)
      numc_pow(a, b, out);
    double t0 = time_us();
    for (int i = 0; i < ITERS; i++)
      numc_pow(a, b, out);
    double us = (time_us() - t0) / ITERS;

    char shape_str[32];
    snprintf(shape_str, sizeof(shape_str), "(%zu)", size);
    csv("binary", "pow", dtype_name(dt), size, shape_str, us);
    numc_ctx_free(ctx);
  }
}

/* -- FMA ------------------------------------------------------------ */

static void bench_fma(size_t size) {
  for (int d = 0; d < N_DTYPES; d++) {
    NumcDType dt = ALL_DTYPES[d];
    NumcCtx *ctx = numc_ctx_create();
    size_t shape[] = {size};
    char val[8];
    fill_value(dt, val);

    NumcArray *a = numc_array_fill(ctx, shape, 1, dt, val);
    NumcArray *b = numc_array_fill(ctx, shape, 1, dt, val);
    NumcArray *c = numc_array_fill(ctx, shape, 1, dt, val);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, dt);
    if (!a || !b || !c || !out) {
      numc_ctx_free(ctx);
      continue;
    }

    for (int i = 0; i < WARMUP; i++)
      numc_fma(a, b, c, out);
    double t0 = time_us();
    for (int i = 0; i < ITERS; i++)
      numc_fma(a, b, c, out);
    double us = (time_us() - t0) / ITERS;

    char shape_str[32];
    snprintf(shape_str, sizeof(shape_str), "(%zu)", size);
    csv("ternary", "fma", dtype_name(dt), size, shape_str, us);
    numc_ctx_free(ctx);
  }
}

/* -- Where ---------------------------------------------------------- */

static void bench_where(size_t size) {
  for (int d = 0; d < N_DTYPES; d++) {
    NumcDType dt = ALL_DTYPES[d];
    NumcCtx *ctx = numc_ctx_create();
    size_t shape[] = {size};
    char val[8];
    fill_value(dt, val);
    char one[8] = {0};
    *(int8_t *)one = 1;

    NumcArray *cond = numc_array_fill(ctx, shape, 1, NUMC_DTYPE_INT8, one);
    NumcArray *a = numc_array_fill(ctx, shape, 1, dt, val);
    NumcArray *b = numc_array_fill(ctx, shape, 1, dt, val);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, dt);
    if (!cond || !a || !b || !out) {
      numc_ctx_free(ctx);
      continue;
    }

    for (int i = 0; i < WARMUP; i++)
      numc_where(cond, a, b, out);
    double t0 = time_us();
    for (int i = 0; i < ITERS; i++)
      numc_where(cond, a, b, out);
    double us = (time_us() - t0) / ITERS;

    char shape_str[32];
    snprintf(shape_str, sizeof(shape_str), "(%zu)", size);
    csv("ternary", "where", dtype_name(dt), size, shape_str, us);
    numc_ctx_free(ctx);
  }
}

/* -- Full reductions ------------------------------------------------ */

typedef int (*ReduceFullFn)(const NumcArray *, NumcArray *);

static void bench_reduce_full(const char *name, ReduceFullFn fn, size_t size,
                              int out_int64) {
  for (int d = 0; d < N_DTYPES; d++) {
    NumcDType dt = ALL_DTYPES[d];
    NumcCtx *ctx = numc_ctx_create();
    size_t shape[] = {size};
    char val[8];
    fill_value(dt, val);

    NumcArray *a = numc_array_fill(ctx, shape, 1, dt, val);
    size_t sshape[] = {1};
    NumcDType odt = out_int64 ? NUMC_DTYPE_INT64 : dt;
    NumcArray *out = numc_array_zeros(ctx, sshape, 1, odt);
    if (!a || !out) {
      numc_ctx_free(ctx);
      continue;
    }

    for (int i = 0; i < WARMUP; i++)
      fn(a, out);
    double t0 = time_us();
    for (int i = 0; i < ITERS; i++)
      fn(a, out);
    double us = (time_us() - t0) / ITERS;

    char shape_str[32];
    snprintf(shape_str, sizeof(shape_str), "(%zu)", size);
    csv("reduction", name, dtype_name(dt), size, shape_str, us);
    numc_ctx_free(ctx);
  }
}

/* -- Axis reductions ------------------------------------------------ */

typedef int (*ReduceAxisFn)(const NumcArray *, int, int, NumcArray *);

static void bench_reduce_axis(const char *name, ReduceAxisFn fn, int axis,
                              size_t rows, size_t cols, int out_int64) {
  size_t total = rows * cols;
  for (int d = 0; d < N_DTYPES; d++) {
    NumcDType dt = ALL_DTYPES[d];
    NumcCtx *ctx = numc_ctx_create();
    size_t shape[] = {rows, cols};
    char val[8];
    fill_value(dt, val);

    NumcArray *a = numc_array_fill(ctx, shape, 2, dt, val);
    size_t oshape[] = {axis == 0 ? cols : rows};
    NumcDType odt = out_int64 ? NUMC_DTYPE_INT64 : dt;
    NumcArray *out = numc_array_zeros(ctx, oshape, 1, odt);
    if (!a || !out) {
      numc_ctx_free(ctx);
      continue;
    }

    for (int i = 0; i < WARMUP; i++)
      fn(a, axis, 0, out);
    double t0 = time_us();
    for (int i = 0; i < ITERS; i++)
      fn(a, axis, 0, out);
    double us = (time_us() - t0) / ITERS;

    char op_name[64], shape_str[64];
    snprintf(op_name, sizeof(op_name), "%s_axis%d", name, axis);
    snprintf(shape_str, sizeof(shape_str), "(%zux%zu)", rows, cols);
    csv("reduction", op_name, dtype_name(dt), total, shape_str, us);
    numc_ctx_free(ctx);
  }
}

/* -- Matmul --------------------------------------------------------- */

static void bench_matmul(size_t M, size_t K, size_t N, int warmup, int iters) {
  /* Re-warm OMP thread pool: after heavy naive integer matmuls from previous
   * sizes, libomp threads may be sleeping (KMP_BLOCKTIME expired). A dummy
   * parallel region wakes them before packed GEMM timing. */
#ifdef _OPENMP
#pragma omp parallel
  { (void)0; }
#endif

  /* Deterministic seed for reproducible random data */
  numc_manual_seed(42);

  NumcDType dtypes[] = {
      NUMC_DTYPE_FLOAT32, NUMC_DTYPE_FLOAT64, NUMC_DTYPE_INT8,
      NUMC_DTYPE_INT16,   NUMC_DTYPE_INT32,   NUMC_DTYPE_INT64,
      NUMC_DTYPE_UINT8,   NUMC_DTYPE_UINT16,  NUMC_DTYPE_UINT32,
      NUMC_DTYPE_UINT64,
  };

  for (int d = 0; d < 10; d++) {
    NumcDType dt = dtypes[d];
    NumcCtx *ctx = numc_ctx_create();
    size_t sha[] = {M, K}, shb[] = {K, N}, sho[] = {M, N};

    /* Use random data for float types to avoid special-case optimizations */
    NumcArray *a, *b;
    if (dt == NUMC_DTYPE_FLOAT32 || dt == NUMC_DTYPE_FLOAT64) {
      a = numc_array_rand(ctx, sha, 2, dt);
      b = numc_array_rand(ctx, shb, 2, dt);
    } else {
      char val[8] = {0};
      *(int32_t *)val = 2;
      a = numc_array_fill(ctx, sha, 2, dt, val);
      b = numc_array_fill(ctx, shb, 2, dt, val);
    }
    NumcArray *out = numc_array_zeros(ctx, sho, 2, dt);
    if (!a || !b || !out) {
      numc_ctx_free(ctx);
      continue;
    }

    for (int i = 0; i < warmup; i++)
      numc_matmul(a, b, out);

    /* Per-iteration timing: report minimum (most stable) */
    double min_us = DBL_MAX;
    for (int i = 0; i < iters; i++) {
      double t0 = time_us();
      numc_matmul(a, b, out);
      double elapsed = time_us() - t0;
      if (elapsed < min_us)
        min_us = elapsed;
    }

    size_t total = M * N;
    char shape_str[64];
    snprintf(shape_str, sizeof(shape_str), "(%zux%zu)@(%zux%zu)", M, K, K, N);
    csv("matmul", "matmul", dtype_name(dt), total, shape_str, min_us);
    numc_ctx_free(ctx);
  }
}

/* -- Dot product ---------------------------------------------------- */

static void bench_dot(size_t size) {
  for (int d = 0; d < N_DTYPES; d++) {
    NumcDType dt = ALL_DTYPES[d];
    NumcCtx *ctx = numc_ctx_create();
    size_t shape[] = {size};

    NumcArray *a = numc_array_randn(ctx, shape, 1, dt);
    NumcArray *b = numc_array_randn(ctx, shape, 1, dt);
    NumcArray *out = numc_array_zeros(ctx, (size_t[]){1}, 1, dt);
    if (!a || !b || !out) {
      numc_ctx_free(ctx);
      continue;
    }

    int iters = 1000;
    for (int i = 0; i < WARMUP; i++)
      numc_dot(a, b, out);
    double t0 = time_us();
    for (int i = 0; i < iters; i++)
      numc_dot(a, b, out);
    double us = (time_us() - t0) / iters;

    char shape_str[32];
    snprintf(shape_str, sizeof(shape_str), "(%zu)", size);
    csv("linalg", "dot", dtype_name(dt), size, shape_str, us);
    numc_ctx_free(ctx);
  }
}

/* -- Random --------------------------------------------------------- */

static void bench_random(const char *name, size_t size,
                         NumcArray *(*gen)(NumcCtx *, const size_t *, size_t,
                                           NumcDType)) {
  int iters = 50;
  for (int d = 0; d < N_DTYPES; d++) {
    NumcDType dt = ALL_DTYPES[d];
    size_t shape[] = {size};

    /* Warmup */
    for (int i = 0; i < 5; i++) {
      NumcCtx *ctx = numc_ctx_create();
      gen(ctx, shape, 1, dt);
      numc_ctx_free(ctx);
    }

    double t0 = time_us();
    for (int i = 0; i < iters; i++) {
      NumcCtx *ctx = numc_ctx_create();
      gen(ctx, shape, 1, dt);
      numc_ctx_free(ctx);
    }
    double us = (time_us() - t0) / iters;

    char shape_str[32];
    snprintf(shape_str, sizeof(shape_str), "(%zu)", size);
    csv("random", name, dtype_name(dt), size, shape_str, us);
  }
}

/* -- main ------------------------------------------------------------ */

int main(void) {
  /* Pre-warm CPU to force turbo-boost ramp-up before measurements */
  bench_cpu_warmup();

  /* CSV header */
  printf(
      "library,category,operation,dtype,size,shape,time_us,throughput_mops\n");

  /* Binary element-wise */
  bench_binary("add", numc_add, SIZE);
  bench_binary("sub", numc_sub, SIZE);
  bench_binary("mul", numc_mul, SIZE);
  bench_binary("div", numc_div, SIZE);
  bench_binary("maximum", numc_maximum, SIZE);
  bench_binary("minimum", numc_minimum, SIZE);
  bench_pow(SIZE);

  /* FMA and Where */
  bench_fma(SIZE);
  bench_where(SIZE);

  /* Scalar ops */
  bench_scalar_op("add_scalar", numc_add_scalar, SIZE);
  bench_scalar_op("sub_scalar", numc_sub_scalar, SIZE);
  bench_scalar_op("mul_scalar", numc_mul_scalar, SIZE);
  bench_scalar_op("div_scalar", numc_div_scalar, SIZE);

  /* Scalar inplace */
  bench_scalar_inplace("add_scalar_inplace", numc_add_scalar_inplace, SIZE);
  bench_scalar_inplace("sub_scalar_inplace", numc_sub_scalar_inplace, SIZE);
  bench_scalar_inplace("mul_scalar_inplace", numc_mul_scalar_inplace, SIZE);
  bench_scalar_inplace("div_scalar_inplace", numc_div_scalar_inplace, SIZE);

  /* Unary ops */
  bench_unary("neg", numc_neg, SIZE, 0, 0);
  bench_unary("abs", numc_abs, SIZE, 1, 0); /* skip unsigned */
  bench_unary("log", numc_log, SIZE, 0, 0);
  bench_unary("exp", numc_exp, SIZE, 0, 1); /* use small values */
  bench_unary("sqrt", numc_sqrt, SIZE, 0, 0);
  bench_unary("tanh", numc_tanh, SIZE, 0, 0);
  bench_clip(SIZE);

  /* Unary inplace */
  bench_unary_inplace("neg_inplace", numc_neg_inplace, SIZE, 0, 0);
  bench_unary_inplace("abs_inplace", numc_abs_inplace, SIZE, 1, 0);
  bench_unary_inplace("log_inplace", numc_log_inplace, SIZE, 0, 0);
  bench_unary_inplace("exp_inplace", numc_exp_inplace, SIZE, 0, 1);
  bench_unary_inplace("sqrt_inplace", numc_sqrt_inplace, SIZE, 0, 0);
  bench_unary_inplace("tanh_inplace", numc_tanh_inplace, SIZE, 0, 0);

  /* Comparison ops */
  bench_comparison("eq", numc_eq, SIZE);
  bench_comparison("gt", numc_gt, SIZE);
  bench_comparison("lt", numc_lt, SIZE);
  bench_comparison("ge", numc_ge, SIZE);
  bench_comparison("le", numc_le, SIZE);

  /* Comparison scalar ops */
  bench_comparison_scalar("eq_scalar", numc_eq_scalar, SIZE);
  bench_comparison_scalar("gt_scalar", numc_gt_scalar, SIZE);
  bench_comparison_scalar("lt_scalar", numc_lt_scalar, SIZE);
  bench_comparison_scalar("ge_scalar", numc_ge_scalar, SIZE);
  bench_comparison_scalar("le_scalar", numc_le_scalar, SIZE);

  /* Full reductions */
  bench_reduce_full("sum", numc_sum, SIZE, 0);
  bench_reduce_full("mean", numc_mean, SIZE, 0);
  bench_reduce_full("max", numc_max, SIZE, 0);
  bench_reduce_full("min", numc_min, SIZE, 0);
  bench_reduce_full("argmax", numc_argmax, SIZE, 1);
  bench_reduce_full("argmin", numc_argmin, SIZE, 1);

  /* Axis reductions (1000x1000) */
  bench_reduce_axis("sum", numc_sum_axis, 0, 1000, 1000, 0);
  bench_reduce_axis("sum", numc_sum_axis, 1, 1000, 1000, 0);
  bench_reduce_axis("mean", numc_mean_axis, 0, 1000, 1000, 0);
  bench_reduce_axis("mean", numc_mean_axis, 1, 1000, 1000, 0);
  bench_reduce_axis("max", numc_max_axis, 0, 1000, 1000, 0);
  bench_reduce_axis("max", numc_max_axis, 1, 1000, 1000, 0);
  bench_reduce_axis("min", numc_min_axis, 0, 1000, 1000, 0);
  bench_reduce_axis("min", numc_min_axis, 1, 1000, 1000, 0);
  bench_reduce_axis("argmax", numc_argmax_axis, 0, 1000, 1000, 1);
  bench_reduce_axis("argmax", numc_argmax_axis, 1, 1000, 1000, 1);
  bench_reduce_axis("argmin", numc_argmin_axis, 0, 1000, 1000, 1);
  bench_reduce_axis("argmin", numc_argmin_axis, 1, 1000, 1000, 1);

  /* Matmul — various sizes (aligned with bench_matmul_csv.c) */
  bench_matmul(64, 64, 64, 200, 2000);
  bench_matmul(128, 128, 128, 100, 500);
  bench_matmul(256, 256, 256, 50, 100);
  bench_matmul(512, 512, 512, 20, 50);
  bench_matmul(1024, 1024, 1024, 10, 20);

  /* Dot product */
  bench_dot(SIZE);

  /* Random */
  bench_random("rand", SIZE, numc_array_rand);
  bench_random("randn", SIZE, numc_array_randn);

  return 0;
}
