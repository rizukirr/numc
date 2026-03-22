/*
 * bench/numc/bench_common.h — Shared benchmark infrastructure for numc
 *
 * Include this header in per-function benchmark files to get:
 * - Timer, min-time benchmark macro
 * - dtype arrays, dtype helpers
 * - fill_value / fill_value_exp / fill_pow_exp
 * - CSV output helpers
 *
 * All functions are static/static inline to avoid linker issues when
 * included from multiple translation units.
 */

#ifndef NUMC_BENCH_COMMON_H
#define NUMC_BENCH_COMMON_H

#include <numc/numc.h>
#include <numc/math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <float.h>
#include <math.h>

/* ── Config ────────────────────────────────────────────────────────── */

#define BENCH_WARMUP 20
#define BENCH_ITERS  200
#define BENCH_SIZE   1000000

/* ── Timer ─────────────────────────────────────────────────────────── */

static inline double time_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

/**
 * @brief Minimum-time benchmark helper for zero-arg callables.
 *
 * Matches NumPy bench.py methodology: report the minimum per-iteration
 * time (most stable, least affected by OS scheduling noise).
 *
 * Requires a pre-declared `double us;` variable in the calling scope.
 */
#define BENCH_MIN_LOOP(CALL, warmup, iters) \
  do {                                      \
    for (int _w = 0; _w < (warmup); _w++)   \
      CALL;                                 \
    double _min_us = DBL_MAX;               \
    for (int _i = 0; _i < (iters); _i++) {  \
      double _t0 = time_us();               \
      CALL;                                 \
      double _elapsed = time_us() - _t0;    \
      if (_elapsed < _min_us)               \
        _min_us = _elapsed;                 \
    }                                       \
    us = _min_us;                           \
  } while (0)

/* ── CPU frequency warmup ─────────────────────────────────────────── */

/**
 * @brief Burn ~200ms of CPU time to force turbo-boost ramp-up.
 *
 * On hybrid CPUs (Intel Alder Lake / Raptor Lake), the CPU starts at a
 * low P-state and ramps up to turbo within ~50-100ms of sustained load.
 * Without pre-warming, the first benchmark in a run pays a ~3-5x penalty.
 * Call this once at the start of main() before any measurements.
 */
static inline void bench_cpu_warmup(void) {
  volatile double sink = 0.0;
  double t0 = time_us();
  while (time_us() - t0 < 200000.0) { /* 200ms */
    for (int i = 0; i < 100000; i++)
      sink += (double)i * 1.0001;
  }
  (void)sink;
}

/* ── Dtype arrays ─────────────────────────────────────────────────── */

static const NumcDType ALL_DTYPES[] = {
    NUMC_DTYPE_INT8,    NUMC_DTYPE_UINT8,   NUMC_DTYPE_INT16, NUMC_DTYPE_UINT16,
    NUMC_DTYPE_INT32,   NUMC_DTYPE_UINT32,  NUMC_DTYPE_INT64, NUMC_DTYPE_UINT64,
    NUMC_DTYPE_FLOAT32, NUMC_DTYPE_FLOAT64,
};
static const int N_DTYPES = 10;

static const NumcDType FLOAT_DTYPES[] = {
    NUMC_DTYPE_FLOAT32,
    NUMC_DTYPE_FLOAT64,
};
static const int N_FLOAT = 2;

/* ── Dtype helpers ────────────────────────────────────────────────── */

static inline const char *dtype_name(NumcDType dt) {
  static const char *names[] = {
      [NUMC_DTYPE_INT8] = "int8",       [NUMC_DTYPE_INT16] = "int16",
      [NUMC_DTYPE_INT32] = "int32",     [NUMC_DTYPE_INT64] = "int64",
      [NUMC_DTYPE_UINT8] = "uint8",     [NUMC_DTYPE_UINT16] = "uint16",
      [NUMC_DTYPE_UINT32] = "uint32",   [NUMC_DTYPE_UINT64] = "uint64",
      [NUMC_DTYPE_FLOAT32] = "float32", [NUMC_DTYPE_FLOAT64] = "float64",
  };
  return names[dt];
}

static inline int dtype_is_unsigned(NumcDType dt) {
  return dt == NUMC_DTYPE_UINT8 || dt == NUMC_DTYPE_UINT16 ||
         dt == NUMC_DTYPE_UINT32 || dt == NUMC_DTYPE_UINT64;
}

/* ── Fill-value helpers ───────────────────────────────────────────── */

static inline void fill_value(NumcDType dt, char buf[8]) {
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
static inline void fill_value_exp(NumcDType dt, char buf[8]) {
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

static inline void fill_pow_exp(NumcDType dt, char buf[8]) {
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

/* ── CSV output ───────────────────────────────────────────────────── */

static inline void bench_csv(const char *cat, const char *op, const char *dt,
                             size_t size, const char *shape, double us) {
  printf("numc,%s,%s,%s,%zu,%s,%.4f,%.4f\n", cat, op, dt, size, shape, us,
         size / us);
}

static inline void bench_csv_header(void) {
  printf(
      "library,category,operation,dtype,size,shape,time_us,throughput_mops\n");
}

static inline int bench_should_print_header(int argc, char **argv) {
  if (argc >= 2 && strcmp(argv[1], "--no-header") == 0)
    return 0;
  return 1;
}

#endif /* NUMC_BENCH_COMMON_H */
