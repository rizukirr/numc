/**
 * @file kernel.h
 * @brief xoshiro256** PRNG with 4-stream SoA layout.
 *
 * Implements a vectorization-friendly PRNG using four interleaved
 * xoshiro256** generators for uniform and normal distribution sampling.
 */
#ifndef NUMC_RANDOM_KERNEL_H
#define NUMC_RANDOM_KERNEL_H

#include "internal.h"
#include "helpers.h"
#include <numc/dtype.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

/* ── xoshiro256** PRNG — 4-stream parallel state ────────────────────
 *
 * To enable auto-vectorization the state is stored in Structure-of-Arrays
 * (SoA) layout: four independent xoshiro256** streams run in lockstep.
 * Each stream is seeded 2^128 steps apart via the xoshiro jump polynomial,
 * guaranteeing non-overlapping sequences.
 *
 * SoA layout (prng_s[word][lane]):
 *   prng_s[0][0..3]  = s0 for streams 0-3
 *   prng_s[1][0..3]  = s1 for streams 0-3
 *   prng_s[2][0..3]  = s2 for streams 0-3
 *   prng_s[3][0..3]  = s3 for streams 0-3
 *
 * The fill loop advances all 4 streams each iteration and writes 4
 * elements at once. Because all operations are pure arithmetic on local
 * arrays, the compiler can fully vectorize to AVX2 (4x uint64_t in YMM).
 *
 * Period per stream:  2^256 - 1
 * Gap between streams: 2^128 steps  (~3.4 * 10^38)
 */

#define NUMC_PRNG_LANES 4

static _Thread_local uint64_t prng_s[4][NUMC_PRNG_LANES];
static _Thread_local bool prng_seeded = false;

/* ── splitmix64: seed expansion ─────────────────────────────────────*/

static inline uint64_t _splitmix64(uint64_t *x) {
  uint64_t z = (*x += UINT64_C(0x9e3779b97f4a7c15));
  z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
  return z ^ (z >> 31);
}

static inline uint64_t numc_rotl64(const uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}

/* ── xoshiro256** jump — advance one stream by 2^128 steps ──────────
 *
 * Jump polynomial from the official xoshiro reference implementation.
 * Applying this to stream k's state gives stream k+1's starting point.
 */
static inline void _xoshiro_jump(uint64_t s[4]) {
  static const uint64_t jump[4] = {
      UINT64_C(0x180ec6d33cfd0aba),
      UINT64_C(0xd5a61266f0c9392c),
      UINT64_C(0xa9582618e03fc9aa),
      UINT64_C(0x39abdc4529b1661c),
  };

  uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  for (int i = 0; i < 4; i++) {
    for (int b = 0; b < 64; b++) {
      if (jump[i] & (UINT64_C(1) << b)) {
        s0 ^= s[0];
        s1 ^= s[1];
        s2 ^= s[2];
        s3 ^= s[3];
      }
      /* advance one step */
      const uint64_t t = s[1] << 17;
      s[2] ^= s[0];
      s[3] ^= s[1];
      s[1] ^= s[2];
      s[0] ^= s[3];
      s[2] ^= t;
      s[3] = numc_rotl64(s[3], 45);
    }
  }
  s[0] = s0;
  s[1] = s1;
  s[2] = s2;
  s[3] = s3;
}

/* ── Seed all 4 streams ──────────────────────────────────────────────*/

static inline void prng_seed(uint64_t seed) {
  /* Seed stream 0 via splitmix64 */
  uint64_t base[4];
  base[0] = _splitmix64(&seed);
  base[1] = _splitmix64(&seed);
  base[2] = _splitmix64(&seed);
  base[3] = _splitmix64(&seed);

  /* Store stream 0 in SoA column 0 */
  for (int w = 0; w < 4; w++)
    prng_s[w][0] = base[w];

  /* Derive streams 1-3 by jumping 2^128 steps each */
  for (int lane = 1; lane < NUMC_PRNG_LANES; lane++) {
    _xoshiro_jump(base);
    for (int w = 0; w < 4; w++)
      prng_s[w][lane] = base[w];
  }

  prng_seeded = true;
}

static inline void _prng_ensure_seeded(void) {
  if (!prng_seeded)
    prng_seed(UINT64_C(0xdeadbeefcafe1234));
}

/* ── Vectorizable 4-wide xoshiro256** step ───────────────────────────
 *
 * Operates on local uint64_t[4] arrays — no global reads/writes inside
 * the loop body. The compiler sees a pure arithmetic recurrence on
 * local variables and emits AVX2 VPMULUDQ / VPXOR / VPSLLQ / VPSRLQ.
 *
 * Caller loads state from prng_s into locals before the loop and
 * stores back after, keeping the hot state entirely in YMM registers.
 */

#define XOSHIRO_STEP_4(s0, s1, s2, s3, out)       \
  do {                                            \
    uint64_t _r0 = numc_rotl64(s1[0] * 5, 7) * 9; \
    uint64_t _r1 = numc_rotl64(s1[1] * 5, 7) * 9; \
    uint64_t _r2 = numc_rotl64(s1[2] * 5, 7) * 9; \
    uint64_t _r3 = numc_rotl64(s1[3] * 5, 7) * 9; \
    uint64_t _t0 = s1[0] << 17;                   \
    uint64_t _t1 = s1[1] << 17;                   \
    uint64_t _t2 = s1[2] << 17;                   \
    uint64_t _t3 = s1[3] << 17;                   \
    s2[0] ^= s0[0];                               \
    s2[1] ^= s0[1];                               \
    s2[2] ^= s0[2];                               \
    s2[3] ^= s0[3];                               \
    s3[0] ^= s1[0];                               \
    s3[1] ^= s1[1];                               \
    s3[2] ^= s1[2];                               \
    s3[3] ^= s1[3];                               \
    s1[0] ^= s2[0];                               \
    s1[1] ^= s2[1];                               \
    s1[2] ^= s2[2];                               \
    s1[3] ^= s2[3];                               \
    s0[0] ^= s3[0];                               \
    s0[1] ^= s3[1];                               \
    s0[2] ^= s3[2];                               \
    s0[3] ^= s3[3];                               \
    s2[0] ^= _t0;                                 \
    s2[1] ^= _t1;                                 \
    s2[2] ^= _t2;                                 \
    s2[3] ^= _t3;                                 \
    s3[0] = numc_rotl64(s3[0], 45);               \
    s3[1] = numc_rotl64(s3[1], 45);               \
    s3[2] = numc_rotl64(s3[2], 45);               \
    s3[3] = numc_rotl64(s3[3], 45);               \
    (out)[0] = _r0;                               \
    (out)[1] = _r1;                               \
    (out)[2] = _r2;                               \
    (out)[3] = _r3;                               \
  } while (0)

/* ── Scalar fallback: single-stream step ────────────────────────────
 *
 * Used for the tail (n % 4 != 0) and for randn (Box-Muller pairs
 * don't benefit from SIMD due to the transcendental functions).
 */

static inline uint64_t _prng_next_scalar(void) {
  _prng_ensure_seeded();
  /* use lane 0 only */
  const uint64_t result = numc_rotl64(prng_s[1][0] * 5, 7) * 9;
  const uint64_t t = prng_s[1][0] << 17;
  prng_s[2][0] ^= prng_s[0][0];
  prng_s[3][0] ^= prng_s[1][0];
  prng_s[1][0] ^= prng_s[2][0];
  prng_s[0][0] ^= prng_s[3][0];
  prng_s[2][0] ^= t;
  prng_s[3][0] = numc_rotl64(prng_s[3][0], 45);
  return result;
}

/* ── Uniform [0, 1) helpers — operate on raw uint64_t ───────────────
 *
 * IEEE 754 bit trick: OR in the exponent for [1.0, 2.0), subtract 1.0.
 * Inlined into the fill loop so the compiler fuses bit manipulation
 * with the store, avoiding intermediate float temporaries.
 */

static inline double _u64_to_f64(uint64_t raw) {
  const uint64_t bits = (raw >> 11) | UINT64_C(0x3FF0000000000000);
  double v;
  memcpy(&v, &bits, sizeof v);
  return v - 1.0;
}

static inline float _u64_to_f32(uint64_t raw) {
  const uint32_t bits = (uint32_t)(raw >> 41) | UINT32_C(0x3F800000);
  float v;
  memcpy(&v, &bits, sizeof v);
  return v - 1.0f;
}

/* Scalar wrappers used by Box-Muller and tail loops */
static inline double _prng_f64(void) {
  return _u64_to_f64(_prng_next_scalar());
}
static inline float _prng_f32(void) {
  return _u64_to_f32(_prng_next_scalar());
}

/* ── Box-Muller — scalar only (transcendentals block vectorization) ──
 *
 * sin/cos/log cannot be SIMD'd by the auto-vectorizer without libmvec.
 * Box-Muller remains sequential; the spare caches one extra sample.
 */

static _Thread_local double bm_spare;
static _Thread_local bool bm_spare_ready = false;

static inline double _prng_normal_f64(void) {
  if (bm_spare_ready) {
    bm_spare_ready = false;
    return bm_spare;
  }
  double u1;
  do {
    u1 = _prng_f64();
  } while (u1 == 0.0);
  double u2 = _prng_f64();
  const double mag = sqrt(-2.0 * _log_f64(u1));
  const double two_pi_u2 = 6.283185307179586476925 * u2;
  bm_spare = mag * _sin_f64(two_pi_u2);
  bm_spare_ready = true;
  return mag * _cos_f64(two_pi_u2);
}

static inline float _prng_normal_f32(void) {
  return (float)_prng_normal_f64();
}

/* ── Per-dtype fill kernel typedef ──────────────────────────────────*/

typedef void (*NumcRandKernel)(char *out, size_t n);

/* ── DEFINE_RAND_KERNEL ──────────────────────────────────────────────
 *
 * Two execution paths selected by total byte count (same threshold as
 * all other numc kernels: NUMC_OMP_BYTE_THRESHOLD = 1 MB):
 *
 * SMALL path (< 1 MB): single-threaded, 4-wide SIMD.
 *   - Loads SoA state into 4 local arrays (hot in registers/L1)
 *   - XOSHIRO_STEP_4 advances 4 independent streams per iteration
 *   - Scalar tail handles n % 4 remainder via lane 0
 *
 * LARGE path (>= 1 MB): multi-threaded via OpenMP + per-thread SIMD.
 *   - Thread count = total_bytes / NUMC_OMP_BYTES_PER_THREAD
 *   - Each thread derives a private sub-state by jumping the global
 *     state by (thread_id * chunk_size) steps, ensuring non-overlapping
 *     sequences with no locking.
 *   - Each thread runs the same 4-wide SIMD inner loop on its slice.
 *   - After all threads finish, global state is advanced past all
 *     elements via a single sequential writeback from thread 0's
 *     final state (threads are assigned contiguous chunks).
 *
 * CONVERT_EXPR(raw) maps a raw uint64_t to C_TYPE — must be a pure
 * expression on a local variable so the SIMD inner loop stays clean.
 */

/* Helper: derive a private PRNG state for one OMP thread by advancing
 * the base state by `skip` individual xoshiro steps.  Each step is
 * one call to _prng_next_scalar(), so this is O(skip) — only called
 * once per thread at the start of the parallel region.             */
static inline void prng_skip(uint64_t s[4], size_t skip) {
  for (size_t k = 0; k < skip; k++) {
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = numc_rotl64(s[3], 45);
  }
}

/* ── _prng_get_tid: safe omp_get_thread_num wrapper ─────────────────
 *
 * Returns the OMP thread ID when OpenMP is available, 0 otherwise.
 * Declared here so DEFINE_RAND_KERNEL can call it without #ifdef inside
 * the macro body.  The compiler will constant-fold the HAVE_OMP==0 path.
 */
static inline int _prng_get_tid(void) {
#ifdef HAVE_OMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

/* ── DEFINE_RAND_KERNEL ──────────────────────────────────────────────
 *
 * Stamps _kern_rand_TYPE_ENUM with two execution paths:
 *
 * SMALL path (< 1 MB): single-threaded 4-wide SIMD.
 *   Loads SoA state into locals, runs XOSHIRO_STEP_4 per 4 elements,
 *   stores state back, handles tail with lane-0 scalar step.
 *
 * LARGE path (>= 1 MB): OMP parallel when HAVE_OMP, else single-thread.
 *   Each thread derives a private sub-state by advancing lane-0 of the
 *   global SoA by (tid * chunk) xoshiro steps via prng_skip(), then
 *   runs the same 4-wide SIMD loop on its slice.
 *   _prng_gettid() wraps omp_get_thread_num() safely without #ifdef
 *   inside the macro body.
 *   After the parallel region, lane-0 is advanced past all n elements.
 */
#define DEFINE_RAND_KERNEL(TYPE_ENUM, C_TYPE, CONVERT_EXPR)              \
  static void _kern_rand_##TYPE_ENUM(char *out, size_t n) {              \
    _prng_ensure_seeded();                                               \
    C_TYPE *restrict po = (C_TYPE *)out;                                 \
    size_t total_bytes = n * sizeof(C_TYPE);                             \
    if (total_bytes <= NUMC_OMP_BYTE_THRESHOLD) {                        \
      /* ── Small path: single-threaded 4-wide SIMD ───────────────── */ \
      uint64_t s0[4], s1[4], s2[4], s3[4];                               \
      memcpy(s0, prng_s[0], sizeof s0);                                  \
      memcpy(s1, prng_s[1], sizeof s1);                                  \
      memcpy(s2, prng_s[2], sizeof s2);                                  \
      memcpy(s3, prng_s[3], sizeof s3);                                  \
      size_t i = 0;                                                      \
      uint64_t raw[4];                                                   \
      for (; i + NUMC_PRNG_LANES <= n; i += NUMC_PRNG_LANES) {           \
        XOSHIRO_STEP_4(s0, s1, s2, s3, raw);                             \
        po[i + 0] = (CONVERT_EXPR(raw[0]));                              \
        po[i + 1] = (CONVERT_EXPR(raw[1]));                              \
        po[i + 2] = (CONVERT_EXPR(raw[2]));                              \
        po[i + 3] = (CONVERT_EXPR(raw[3]));                              \
      }                                                                  \
      memcpy(prng_s[0], s0, sizeof s0);                                  \
      memcpy(prng_s[1], s1, sizeof s1);                                  \
      memcpy(prng_s[2], s2, sizeof s2);                                  \
      memcpy(prng_s[3], s3, sizeof s3);                                  \
      for (; i < n; i++)                                                 \
        po[i] = (CONVERT_EXPR(_prng_next_scalar()));                     \
    } else {                                                             \
      /* ── Large path: per-thread sub-states, OMP when available ─── */ \
      int _nthreads = (int)(total_bytes / NUMC_OMP_BYTES_PER_THREAD);    \
      if (_nthreads < 1)                                                 \
        _nthreads = 1;                                                   \
      uint64_t base[4] = {                                               \
          prng_s[0][0],                                                  \
          prng_s[1][0],                                                  \
          prng_s[2][0],                                                  \
          prng_s[3][0],                                                  \
      };                                                                 \
      size_t chunk = (n + (size_t)_nthreads - 1) / (size_t)_nthreads;    \
      NUMC_PRAGMA(omp parallel num_threads(_nthreads)) {                 \
        int tid = _prng_get_tid();                                       \
        size_t start = (size_t)tid * chunk;                              \
        size_t end = start + chunk < n ? start + chunk : n;              \
        uint64_t ts[4] = {base[0], base[1], base[2], base[3]};           \
        prng_skip(ts, start);                                            \
        uint64_t s0[4], s1[4], s2[4], s3[4];                             \
        for (int l = 0; l < 4; l++) {                                    \
          uint64_t lt[4] = {ts[0], ts[1], ts[2], ts[3]};                 \
          prng_skip(lt, (size_t)l);                                      \
          s0[l] = lt[0];                                                 \
          s1[l] = lt[1];                                                 \
          s2[l] = lt[2];                                                 \
          s3[l] = lt[3];                                                 \
        }                                                                \
        size_t _i = start;                                               \
        uint64_t _raw[4];                                                \
        for (; _i + NUMC_PRNG_LANES <= end; _i += NUMC_PRNG_LANES) {     \
          XOSHIRO_STEP_4(s0, s1, s2, s3, _raw);                          \
          po[_i + 0] = (CONVERT_EXPR(_raw[0]));                          \
          po[_i + 1] = (CONVERT_EXPR(_raw[1]));                          \
          po[_i + 2] = (CONVERT_EXPR(_raw[2]));                          \
          po[_i + 3] = (CONVERT_EXPR(_raw[3]));                          \
        }                                                                \
        uint64_t _ts[4] = {s0[0], s1[0], s2[0], s3[0]};                  \
        for (; _i < end; _i++) {                                         \
          const uint64_t _res = numc_rotl64(_ts[1] * 5, 7) * 9;          \
          const uint64_t _t = _ts[1] << 17;                              \
          _ts[2] ^= _ts[0];                                              \
          _ts[3] ^= _ts[1];                                              \
          _ts[1] ^= _ts[2];                                              \
          _ts[0] ^= _ts[3];                                              \
          _ts[2] ^= _t;                                                  \
          _ts[3] = numc_rotl64(_ts[3], 45);                              \
          po[_i] = (CONVERT_EXPR(_res));                                 \
        }                                                                \
      } /* end omp parallel */                                           \
      prng_skip(base, n);                                                \
      prng_s[0][0] = base[0];                                            \
      prng_s[1][0] = base[1];                                            \
      prng_s[2][0] = base[2];                                            \
      prng_s[3][0] = base[3];                                            \
      /* advance lanes 1-3 by n steps to keep all lanes in sync */       \
      for (int _lane = 1; _lane < NUMC_PRNG_LANES; _lane++) {            \
        uint64_t _ls[4] = {                                              \
            prng_s[0][_lane],                                            \
            prng_s[1][_lane],                                            \
            prng_s[2][_lane],                                            \
            prng_s[3][_lane],                                            \
        };                                                               \
        prng_skip(_ls, n);                                               \
        prng_s[0][_lane] = _ls[0];                                       \
        prng_s[1][_lane] = _ls[1];                                       \
        prng_s[2][_lane] = _ls[2];                                       \
        prng_s[3][_lane] = _ls[3];                                       \
      }                                                                  \
    }                                                                    \
  }

/* ── DEFINE_RANDN_KERNEL ─────────────────────────────────────────────
 *
 * randn uses Box-Muller (log/sin/cos) which cannot be auto-vectorized
 * without libmvec. Parallelized with NUMC_OMP_FOR over a simple scalar
 * loop — each OMP thread has its own _Thread_local PRNG state so there
 * is no contention; the spare cache (_bm_spare) is also thread-local.
 *
 * Note: _Thread_local state is initialized lazily per thread via
 * _prng_ensure_seeded(), so worker threads auto-seed from the default
 * constant unless numc_manual_seed() was called on that thread.
 * For reproducible parallel randn the caller should seed each thread
 * explicitly (or accept that parallel threads get independent streams).
 */

#define DEFINE_RANDN_KERNEL(TYPE_ENUM, C_TYPE, EXPR)         \
  static void _kern_randn_##TYPE_ENUM(char *out, size_t n) { \
    C_TYPE *restrict po = (C_TYPE *)out;                     \
    NUMC_OMP_FOR_NOSIMD(                                     \
        n, sizeof(C_TYPE),                                   \
        for (size_t i = 0; i < n; i++) { po[i] = (EXPR); }); \
  }

/* ── Dispatch table entry helper ────────────────────────────────────*/

#define ER(OP, TE) [TE] = _kern_##OP##_##TE

#endif /* NUMC_RANDOM_KERNEL_H */
