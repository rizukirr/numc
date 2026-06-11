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

/* -- Timing macros ---------------------------------------------------- *
 * Report the MINIMUM per-iteration time, consistently across every
 * category (matmul already did this). Min is the most stable statistic
 * under OS scheduling noise; using it everywhere keeps categories
 * comparable instead of mixing min (matmul) with mean (everything else).
 *
 * BENCH_MIN       — warm up, then time CALL `niters` times.
 * BENCH_MIN_RESET — same, but run RESET (untimed) before every call, so
 *                   in-place ops measure a fresh, valid input each time
 *                   instead of accumulating toward inf/NaN. */
#define BENCH_MIN(CALL, niters, OUT)        \
  do {                                      \
    for (int _w = 0; _w < WARMUP; _w++)     \
      CALL;                                 \
    double _min = DBL_MAX;                  \
    for (int _i = 0; _i < (niters); _i++) { \
      double _t0 = time_us();               \
      CALL;                                 \
      double _e = time_us() - _t0;          \
      if (_e < _min)                        \
        _min = _e;                          \
    }                                       \
    (OUT) = _min;                           \
  } while (0)

#define BENCH_MIN_RESET(RESET, CALL, niters, OUT) \
  do {                                            \
    for (int _w = 0; _w < WARMUP; _w++) {         \
      RESET;                                      \
      CALL;                                       \
    }                                             \
    double _min = DBL_MAX;                        \
    for (int _i = 0; _i < (niters); _i++) {       \
      RESET;                                      \
      double _t0 = time_us();                     \
      CALL;                                       \
      double _e = time_us() - _t0;                \
      if (_e < _min)                              \
        _min = _e;                                \
    }                                             \
    (OUT) = _min;                                 \
  } while (0)

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

/* -- Spread data fills ---------------------------------------------- *
 * Constant input makes every data-dependent branch (comparisons,
 * maximum/minimum, clip, argmax/argmin, where) perfectly predictable,
 * which flatters those kernels. These helpers write reproducible,
 * well-spread per-element data instead. */

/* Cheap reproducible per-element hash (no RNG state, no seed). */
static unsigned bench_hash(size_t i, unsigned phase) {
  unsigned h = (unsigned)i + phase * 2654435761u;
  h ^= h >> 13;
  h *= 0x9E3779B1u;
  h ^= h >> 15;
  return h;
}

/* Spread values: ints in [0,15], floats in [0.0,8.0). The float range
 * straddles the 2.0 scalar-comparison threshold and the clip(1.0,5.0)
 * bounds used below, so those branches see a realistic mix. `phase`
 * shifts the sequence so two operands differ element-wise. */
static void fill_varied(NumcArray *arr, unsigned phase) {
  size_t n = numc_array_size(arr);
  void *data = numc_array_data(arr);
  switch (numc_array_dtype(arr)) {
#define NUMC_BENCH_FILL(DT, CT, EXPR)    \
  case DT:                               \
    for (size_t i = 0; i < n; i++) {     \
      unsigned h = bench_hash(i, phase); \
      ((CT *)data)[i] = (CT)(EXPR);      \
    }                                    \
    break;
    NUMC_BENCH_FILL(NUMC_DTYPE_INT8, int8_t, h % 16)
    NUMC_BENCH_FILL(NUMC_DTYPE_UINT8, uint8_t, h % 16)
    NUMC_BENCH_FILL(NUMC_DTYPE_INT16, int16_t, h % 16)
    NUMC_BENCH_FILL(NUMC_DTYPE_UINT16, uint16_t, h % 16)
    NUMC_BENCH_FILL(NUMC_DTYPE_INT32, int32_t, h % 16)
    NUMC_BENCH_FILL(NUMC_DTYPE_UINT32, uint32_t, h % 16)
    NUMC_BENCH_FILL(NUMC_DTYPE_INT64, int64_t, h % 16)
    NUMC_BENCH_FILL(NUMC_DTYPE_UINT64, uint64_t, h % 16)
    NUMC_BENCH_FILL(NUMC_DTYPE_FLOAT32, float, (float)(h % 800) * 0.01f)
    NUMC_BENCH_FILL(NUMC_DTYPE_FLOAT64, double, (double)(h % 800) * 0.01)
#undef NUMC_BENCH_FILL
  }
}

/* Like fill_varied but strictly nonzero: ints in [1,15], floats in
 * [1.0,8.0). Safe to use as a divisor (no integer divide-by-zero). */
static void fill_nonzero(NumcArray *arr, unsigned phase) {
  size_t n = numc_array_size(arr);
  void *data = numc_array_data(arr);
  switch (numc_array_dtype(arr)) {
#define NUMC_BENCH_FILL(DT, CT, EXPR)    \
  case DT:                               \
    for (size_t i = 0; i < n; i++) {     \
      unsigned h = bench_hash(i, phase); \
      ((CT *)data)[i] = (CT)(EXPR);      \
    }                                    \
    break;
    NUMC_BENCH_FILL(NUMC_DTYPE_INT8, int8_t, 1 + h % 15)
    NUMC_BENCH_FILL(NUMC_DTYPE_UINT8, uint8_t, 1 + h % 15)
    NUMC_BENCH_FILL(NUMC_DTYPE_INT16, int16_t, 1 + h % 15)
    NUMC_BENCH_FILL(NUMC_DTYPE_UINT16, uint16_t, 1 + h % 15)
    NUMC_BENCH_FILL(NUMC_DTYPE_INT32, int32_t, 1 + h % 15)
    NUMC_BENCH_FILL(NUMC_DTYPE_UINT32, uint32_t, 1 + h % 15)
    NUMC_BENCH_FILL(NUMC_DTYPE_INT64, int64_t, 1 + h % 15)
    NUMC_BENCH_FILL(NUMC_DTYPE_UINT64, uint64_t, 1 + h % 15)
    NUMC_BENCH_FILL(NUMC_DTYPE_FLOAT32, float, 1.0f + (float)(h % 700) * 0.01f)
    NUMC_BENCH_FILL(NUMC_DTYPE_FLOAT64, double, 1.0 + (double)(h % 700) * 0.01)
#undef NUMC_BENCH_FILL
  }
}

/* Spread values that straddle the clip(1.0,5.0) bounds: ints in [0,9],
 * floats in [0.0,6.0), so both the lower and upper clamp branches fire. */
static void fill_clip(NumcArray *arr) {
  size_t n = numc_array_size(arr);
  void *data = numc_array_data(arr);
  switch (numc_array_dtype(arr)) {
#define NUMC_BENCH_FILL(DT, CT, EXPR) \
  case DT:                            \
    for (size_t i = 0; i < n; i++) {  \
      unsigned h = bench_hash(i, 0);  \
      ((CT *)data)[i] = (CT)(EXPR);   \
    }                                 \
    break;
    NUMC_BENCH_FILL(NUMC_DTYPE_INT8, int8_t, h % 10)
    NUMC_BENCH_FILL(NUMC_DTYPE_UINT8, uint8_t, h % 10)
    NUMC_BENCH_FILL(NUMC_DTYPE_INT16, int16_t, h % 10)
    NUMC_BENCH_FILL(NUMC_DTYPE_UINT16, uint16_t, h % 10)
    NUMC_BENCH_FILL(NUMC_DTYPE_INT32, int32_t, h % 10)
    NUMC_BENCH_FILL(NUMC_DTYPE_UINT32, uint32_t, h % 10)
    NUMC_BENCH_FILL(NUMC_DTYPE_INT64, int64_t, h % 10)
    NUMC_BENCH_FILL(NUMC_DTYPE_UINT64, uint64_t, h % 10)
    NUMC_BENCH_FILL(NUMC_DTYPE_FLOAT32, float, (float)(h % 600) * 0.01f)
    NUMC_BENCH_FILL(NUMC_DTYPE_FLOAT64, double, (double)(h % 600) * 0.01)
#undef NUMC_BENCH_FILL
  }
}

/* Broadcast a constant value across every element, in place. Used to
 * reset an in-place op's input to a fresh, valid value before each timed
 * iteration. */
static void reset_const(NumcArray *arr, const char val[static 8]) {
  size_t n = numc_array_size(arr);
  size_t es = numc_array_elem_size(arr);
  char *d = numc_array_data(arr);
  for (size_t i = 0; i < n; i++)
    memcpy(d + i * es, val, es);
}

/* Fill a 0/1 condition mask with an exact 50/50 split (alternating), so
 * `where` actually selects from both branches instead of always one. The
 * mask is typed to match the operands: numc_where currently requires the
 * condition's dtype to equal a/b/out (see _check_ternary). */
static void fill_cond_half(NumcArray *arr) {
  size_t n = numc_array_size(arr);
  void *data = numc_array_data(arr);
  switch (numc_array_dtype(arr)) {
#define NUMC_BENCH_FILL(DT, CT)      \
  case DT:                           \
    for (size_t i = 0; i < n; i++)   \
      ((CT *)data)[i] = (CT)(i & 1); \
    break;
    NUMC_BENCH_FILL(NUMC_DTYPE_INT8, int8_t)
    NUMC_BENCH_FILL(NUMC_DTYPE_UINT8, uint8_t)
    NUMC_BENCH_FILL(NUMC_DTYPE_INT16, int16_t)
    NUMC_BENCH_FILL(NUMC_DTYPE_UINT16, uint16_t)
    NUMC_BENCH_FILL(NUMC_DTYPE_INT32, int32_t)
    NUMC_BENCH_FILL(NUMC_DTYPE_UINT32, uint32_t)
    NUMC_BENCH_FILL(NUMC_DTYPE_INT64, int64_t)
    NUMC_BENCH_FILL(NUMC_DTYPE_UINT64, uint64_t)
    NUMC_BENCH_FILL(NUMC_DTYPE_FLOAT32, float)
    NUMC_BENCH_FILL(NUMC_DTYPE_FLOAT64, double)
#undef NUMC_BENCH_FILL
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
    NumcArray *a = numc_array_zeros(ctx, shape, 1, dt);
    NumcArray *b = numc_array_zeros(ctx, shape, 1, dt);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, dt);
    if (!a || !b || !out) {
      numc_ctx_free(ctx);
      continue;
    }
    fill_varied(a, 0);
    fill_nonzero(b, 1); /* nonzero -> safe divisor for div */

    double us;
    BENCH_MIN(op(a, b, out), ITERS, us);

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
    NumcArray *a = numc_array_zeros(ctx, shape, 1, dt);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, dt);
    if (!a || !out) {
      numc_ctx_free(ctx);
      continue;
    }
    fill_varied(a, 0);

    double us;
    BENCH_MIN(op(a, 2.0, out), ITERS, us);

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
    NumcArray *a = numc_array_zeros(ctx, shape, 1, dt);
    if (!a) {
      numc_ctx_free(ctx);
      continue;
    }

    /* Reset before each timed call: in-place op would otherwise compound
     * across iterations (e.g. mul_scalar drifting toward overflow). */
    double us;
    BENCH_MIN_RESET(fill_varied(a, 0), op(a, 1.01), ITERS, us);

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

    /* Transcendentals keep controlled, domain-safe inputs (log/sqrt need
     * non-negative, exp needs small values); only the statistic changes. */
    double us;
    BENCH_MIN(op(a, out), ITERS, us);

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

    /* Reset to the fresh fill before each call: in-place exp/log would
     * otherwise compound to inf/NaN within a few iterations and end up
     * timing a different code path. `val` holds the domain-safe value. */
    double us;
    BENCH_MIN_RESET(reset_const(a, val), op(a), ITERS, us);

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
    NumcArray *a = numc_array_zeros(ctx, shape, 1, dt);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, dt);
    if (!a || !out) {
      numc_ctx_free(ctx);
      continue;
    }
    fill_clip(a); /* spread straddles both clip bounds */

    double us;
    BENCH_MIN(numc_clip(a, out, 1.0, 5.0), ITERS, us);

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
    NumcArray *a = numc_array_zeros(ctx, shape, 1, dt);
    NumcArray *b = numc_array_zeros(ctx, shape, 1, dt);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_UINT8);
    if (!a || !b || !out) {
      numc_ctx_free(ctx);
      continue;
    }
    fill_varied(a, 0);
    fill_varied(b, 1); /* differs from a -> realistic mix of true/false */

    double us;
    BENCH_MIN(op(a, b, out), ITERS, us);

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
    NumcArray *a = numc_array_zeros(ctx, shape, 1, dt);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, NUMC_DTYPE_UINT8);
    if (!a || !out) {
      numc_ctx_free(ctx);
      continue;
    }
    fill_varied(a, 0); /* spread straddles the 2.0 threshold */

    double us;
    BENCH_MIN(op(a, 2.0, out), ITERS, us);

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

    /* Controlled base/exponent kept (random pow easily overflows/NaNs). */
    double us;
    BENCH_MIN(numc_pow(a, b, out), ITERS, us);

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
    NumcArray *a = numc_array_zeros(ctx, shape, 1, dt);
    NumcArray *b = numc_array_zeros(ctx, shape, 1, dt);
    NumcArray *c = numc_array_zeros(ctx, shape, 1, dt);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, dt);
    if (!a || !b || !c || !out) {
      numc_ctx_free(ctx);
      continue;
    }
    fill_varied(a, 0);
    fill_varied(b, 1);
    fill_varied(c, 2);

    double us;
    BENCH_MIN(numc_fma(a, b, c, out), ITERS, us);

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
    NumcArray *cond = numc_array_zeros(ctx, shape, 1, dt);
    NumcArray *a = numc_array_zeros(ctx, shape, 1, dt);
    NumcArray *b = numc_array_zeros(ctx, shape, 1, dt);
    NumcArray *out = numc_array_zeros(ctx, shape, 1, dt);
    if (!cond || !a || !b || !out) {
      numc_ctx_free(ctx);
      continue;
    }
    fill_cond_half(cond); /* 50/50 -> exercises both select branches */
    fill_varied(a, 0);
    fill_varied(b, 1);

    double us;
    BENCH_MIN(numc_where(cond, a, b, out), ITERS, us);

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
    NumcArray *a = numc_array_zeros(ctx, shape, 1, dt);
    size_t sshape[] = {1};
    NumcDType odt = out_int64 ? NUMC_DTYPE_INT64 : dt;
    NumcArray *out = numc_array_zeros(ctx, sshape, 1, odt);
    if (!a || !out) {
      numc_ctx_free(ctx);
      continue;
    }
    fill_varied(a, 0);

    double us;
    BENCH_MIN(fn(a, out), ITERS, us);

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
    NumcArray *a = numc_array_zeros(ctx, shape, 2, dt);
    size_t oshape[] = {axis == 0 ? cols : rows};
    NumcDType odt = out_int64 ? NUMC_DTYPE_INT64 : dt;
    NumcArray *out = numc_array_zeros(ctx, oshape, 1, odt);
    if (!a || !out) {
      numc_ctx_free(ctx);
      continue;
    }
    fill_varied(a, 0);

    double us;
    BENCH_MIN(fn(a, axis, 0, out), ITERS, us);

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
    double us;
    BENCH_MIN(numc_dot(a, b, out), iters, us);

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

    /* Time only the generation call (which includes its own array
     * allocation); the surrounding ctx create/free is excluded so this
     * measures the RNG fill rather than arena lifecycle overhead. */
    double us = DBL_MAX;
    for (int i = 0; i < iters; i++) {
      NumcCtx *ctx = numc_ctx_create();
      double t0 = time_us();
      gen(ctx, shape, 1, dt);
      double e = time_us() - t0;
      numc_ctx_free(ctx);
      if (e < us)
        us = e;
    }

    char shape_str[32];
    snprintf(shape_str, sizeof(shape_str), "(%zu)", size);
    csv("random", name, dtype_name(dt), size, shape_str, us);
  }
}

/* -- Cache / bandwidth sweep ---------------------------------------- *
 * The same op (add) across sizes that grow from L1-resident up into
 * DRAM, so the CSV shows how throughput falls off once the working set
 * leaves cache. The default SIZE=1M elemwise numbers above are largely
 * L3-resident; this sweep makes that cache dependence explicit. Working
 * set per point is 3 buffers (a, b, out) * elem_size * size. */
static void bench_cache_sweep(void) {
  static const size_t sizes[] = {
      1024, 8192, 65536, 524288, 4194304, 16777216,
  };
  static const NumcDType dts[] = {NUMC_DTYPE_FLOAT32, NUMC_DTYPE_FLOAT64};
  for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
    size_t size = sizes[s];
    /* Fewer iterations for the largest, DRAM-bound sizes (each one streams
     * hundreds of MB per pass). */
    int iters = size >= 4194304 ? 30 : 200;
    for (int d = 0; d < 2; d++) {
      NumcDType dt = dts[d];
      NumcCtx *ctx = numc_ctx_create();
      size_t shape[] = {size};
      NumcArray *a = numc_array_zeros(ctx, shape, 1, dt);
      NumcArray *b = numc_array_zeros(ctx, shape, 1, dt);
      NumcArray *out = numc_array_zeros(ctx, shape, 1, dt);
      if (!a || !b || !out) {
        numc_ctx_free(ctx);
        continue;
      }
      fill_varied(a, 0);
      fill_nonzero(b, 1);

      double us;
      BENCH_MIN(numc_add(a, b, out), iters, us);

      char shape_str[32];
      snprintf(shape_str, sizeof(shape_str), "(%zu)", size);
      csv("cache", "add", dtype_name(dt), size, shape_str, us);
      numc_ctx_free(ctx);
    }
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

  /* Cache / bandwidth sweep (L1 -> DRAM) */
  bench_cache_sweep();

  return 0;
}
