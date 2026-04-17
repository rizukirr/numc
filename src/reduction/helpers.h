/**
 * @file helpers.h
 * @brief Pairwise summation and multi-accumulator helpers for reductions.
 *
 * Provides numerically stable summation routines that maintain O(log n)
 * error growth, along with 4-way unrolled accumulation for better ILP.
 */
#ifndef NUMC_REDUCTION_HELPERS_H
#define NUMC_REDUCTION_HELPERS_H

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "arch_dispatch.h"
#if NUMC_HAVE_AVX512
#include "intrinsics/reduce_avx512.h"
#elif NUMC_HAVE_AVX2
#include "intrinsics/reduce_avx2.h"
#endif
#if NUMC_HAVE_SVE
#include "intrinsics/reduce_sve.h"
#elif NUMC_HAVE_NEON
#include "intrinsics/reduce_neon.h"
#endif
#if NUMC_HAVE_RVV
#include "intrinsics/reduce_rvv.h"
#endif

/* -- Pairwise summation for float types ------------------------------
 *
 * IEEE-754 non-associativity prevents the compiler from vectorizing
 * a serial `acc += val` loop — it emits scalar vaddss/vaddsd with a
 * single accumulator (latency-bound, ~4 cycles per add).
 *
 * Pairwise summation uses 8 independent accumulators in the leaf,
 * which the SLP vectorizer packs into vaddps/vaddpd (throughput-bound,
 * 8 or 4 floats per vector add). Recursive splitting keeps the
 * accumulators independent across blocks.
 *
 * Block size 128: matches NumPy's pairwise_sum implementation.
 * 8 accumulators: fills one ymm (float32) or two ymm (float64).
 */

#define PAIRWISE_BLOCKSIZE 128

/**
 * @brief Pairwise summation for float32 (accurate and vectorizable).
 *
 * @param a Pointer to the float32 data.
 * @param n Number of elements.
 * @return Sum of all elements.
 */
static inline float _pairwise_sum_f32(const float *restrict a, size_t n) {
  if (n <= PAIRWISE_BLOCKSIZE) {
    float r0 = 0, r1 = 0, r2 = 0, r3 = 0;
    float r4 = 0, r5 = 0, r6 = 0, r7 = 0;
    size_t i = 0, n8 = n & ~(size_t)7;
    for (; i < n8; i += 8) {
      r0 += a[i];
      r1 += a[i + 1];
      r2 += a[i + 2];
      r3 += a[i + 3];
      r4 += a[i + 4];
      r5 += a[i + 5];
      r6 += a[i + 6];
      r7 += a[i + 7];
    }
    float sum = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
    for (; i < n; i++)
      sum += a[i];
    return sum;
  }
  size_t half = n / 2;
  return _pairwise_sum_f32(a, half) + _pairwise_sum_f32(a + half, n - half);
}

/**
 * @brief Pairwise summation for float64 (accurate and vectorizable).
 *
 * @param a Pointer to the float64 data.
 * @param n Number of elements.
 * @return Sum of all elements.
 */
static inline double _pairwise_sum_f64(const double *restrict a, size_t n) {
  if (n <= PAIRWISE_BLOCKSIZE) {
    double r0 = 0, r1 = 0, r2 = 0, r3 = 0;
    double r4 = 0, r5 = 0, r6 = 0, r7 = 0;
    size_t i = 0, n8 = n & ~(size_t)7;
    for (; i < n8; i += 8) {
      r0 += a[i];
      r1 += a[i + 1];
      r2 += a[i + 2];
      r3 += a[i + 3];
      r4 += a[i + 4];
      r5 += a[i + 5];
      r6 += a[i + 6];
      r7 += a[i + 7];
    }
    double sum = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
    for (; i < n; i++)
      sum += a[i];
    return sum;
  }
  size_t half = n / 2;
  return _pairwise_sum_f64(a, half) + _pairwise_sum_f64(a + half, n - half);
}

/* -- Multi-accumulator max/min for float types -------------------------
 *
 * Same technique as pairwise summation: 8 independent accumulators
 * that the SLP vectorizer packs into vmaxps/vminps (float32) or
 * vmaxpd/vminpd (float64).
 *
 * Unlike pairwise sum, no recursion needed — max/min is exact
 * (no rounding error), so a flat multi-accumulator loop suffices.
 */

#define NUMC_VMAX(a, b) ((a) > (b) ? (a) : (b))
#define NUMC_VMIN(a, b) ((a) < (b) ? (a) : (b))

#define DEFINE_VEC_MINMAX(NAME, TYPE, INIT, OP)                          \
  static inline TYPE _vec_##NAME(const TYPE *restrict a, size_t n) {     \
    TYPE r0 = INIT, r1 = INIT, r2 = INIT, r3 = INIT;                     \
    TYPE r4 = INIT, r5 = INIT, r6 = INIT, r7 = INIT;                     \
    size_t i = 0, n8 = n & ~(size_t)7;                                   \
    for (; i < n8; i += 8) {                                             \
      r0 = OP(a[i], r0);                                                 \
      r1 = OP(a[i + 1], r1);                                             \
      r2 = OP(a[i + 2], r2);                                             \
      r3 = OP(a[i + 3], r3);                                             \
      r4 = OP(a[i + 4], r4);                                             \
      r5 = OP(a[i + 5], r5);                                             \
      r6 = OP(a[i + 6], r6);                                             \
      r7 = OP(a[i + 7], r7);                                             \
    }                                                                    \
    TYPE m = OP(OP(OP(r0, r1), OP(r2, r3)), OP(OP(r4, r5), OP(r6, r7))); \
    for (; i < n; i++)                                                   \
      m = OP(a[i], m);                                                   \
    return m;                                                            \
  }

DEFINE_VEC_MINMAX(max_f32, float, -INFINITY, NUMC_VMAX)
DEFINE_VEC_MINMAX(max_f64, double, -INFINITY, NUMC_VMAX)
DEFINE_VEC_MINMAX(min_f32, float, INFINITY, NUMC_VMIN)
DEFINE_VEC_MINMAX(min_f64, double, INFINITY, NUMC_VMIN)

/* Integer min/max: SIMD dispatch by ISA priority.
 * Each ISA provides reduce_{min,max}_{i8..u64}_SUFFIX functions. */

#define _VEC_WRAP(NAME, CT, SIMD_FN)                    \
  static inline CT _vec_##NAME(const CT *a, size_t n) { \
    return SIMD_FN(a, n);                               \
  }

#if NUMC_HAVE_AVX512


_VEC_WRAP(max_i8, int8_t, reduce_max_i8_avx512)
_VEC_WRAP(max_i16, int16_t, reduce_max_i16_avx512)
_VEC_WRAP(max_i32, int32_t, reduce_max_i32_avx512)
_VEC_WRAP(max_i64, int64_t, reduce_max_i64_avx512)
_VEC_WRAP(max_u8, uint8_t, reduce_max_u8_avx512)
_VEC_WRAP(max_u16, uint16_t, reduce_max_u16_avx512)
_VEC_WRAP(max_u32, uint32_t, reduce_max_u32_avx512)
_VEC_WRAP(max_u64, uint64_t, reduce_max_u64_avx512)
_VEC_WRAP(min_i8, int8_t, reduce_min_i8_avx512)
_VEC_WRAP(min_i16, int16_t, reduce_min_i16_avx512)
_VEC_WRAP(min_i32, int32_t, reduce_min_i32_avx512)
_VEC_WRAP(min_i64, int64_t, reduce_min_i64_avx512)
_VEC_WRAP(min_u8, uint8_t, reduce_min_u8_avx512)
_VEC_WRAP(min_u16, uint16_t, reduce_min_u16_avx512)
_VEC_WRAP(min_u32, uint32_t, reduce_min_u32_avx512)
_VEC_WRAP(min_u64, uint64_t, reduce_min_u64_avx512)

#elif NUMC_HAVE_AVX2


_VEC_WRAP(max_i8, int8_t, reduce_max_i8_avx2)
_VEC_WRAP(max_i16, int16_t, reduce_max_i16_avx2)
_VEC_WRAP(max_i32, int32_t, reduce_max_i32_avx2)
_VEC_WRAP(max_i64, int64_t, reduce_max_i64_avx2)
_VEC_WRAP(max_u8, uint8_t, reduce_max_u8_avx2)
_VEC_WRAP(max_u16, uint16_t, reduce_max_u16_avx2)
_VEC_WRAP(max_u32, uint32_t, reduce_max_u32_avx2)
_VEC_WRAP(max_u64, uint64_t, reduce_max_u64_avx2)
_VEC_WRAP(min_i8, int8_t, reduce_min_i8_avx2)
_VEC_WRAP(min_i16, int16_t, reduce_min_i16_avx2)
_VEC_WRAP(min_i32, int32_t, reduce_min_i32_avx2)
_VEC_WRAP(min_i64, int64_t, reduce_min_i64_avx2)
_VEC_WRAP(min_u8, uint8_t, reduce_min_u8_avx2)
_VEC_WRAP(min_u16, uint16_t, reduce_min_u16_avx2)
_VEC_WRAP(min_u32, uint32_t, reduce_min_u32_avx2)
_VEC_WRAP(min_u64, uint64_t, reduce_min_u64_avx2)

#elif NUMC_HAVE_SVE


_VEC_WRAP(max_i8, int8_t, reduce_max_i8_sve)
_VEC_WRAP(max_i16, int16_t, reduce_max_i16_sve)
_VEC_WRAP(max_i32, int32_t, reduce_max_i32_sve)
_VEC_WRAP(max_i64, int64_t, reduce_max_i64_sve)
_VEC_WRAP(max_u8, uint8_t, reduce_max_u8_sve)
_VEC_WRAP(max_u16, uint16_t, reduce_max_u16_sve)
_VEC_WRAP(max_u32, uint32_t, reduce_max_u32_sve)
_VEC_WRAP(max_u64, uint64_t, reduce_max_u64_sve)
_VEC_WRAP(min_i8, int8_t, reduce_min_i8_sve)
_VEC_WRAP(min_i16, int16_t, reduce_min_i16_sve)
_VEC_WRAP(min_i32, int32_t, reduce_min_i32_sve)
_VEC_WRAP(min_i64, int64_t, reduce_min_i64_sve)
_VEC_WRAP(min_u8, uint8_t, reduce_min_u8_sve)
_VEC_WRAP(min_u16, uint16_t, reduce_min_u16_sve)
_VEC_WRAP(min_u32, uint32_t, reduce_min_u32_sve)
_VEC_WRAP(min_u64, uint64_t, reduce_min_u64_sve)

#elif NUMC_HAVE_NEON


_VEC_WRAP(max_i8, int8_t, reduce_max_i8_neon)
_VEC_WRAP(max_i16, int16_t, reduce_max_i16_neon)
_VEC_WRAP(max_i32, int32_t, reduce_max_i32_neon)
_VEC_WRAP(max_i64, int64_t, reduce_max_i64_neon)
_VEC_WRAP(max_u8, uint8_t, reduce_max_u8_neon)
_VEC_WRAP(max_u16, uint16_t, reduce_max_u16_neon)
_VEC_WRAP(max_u32, uint32_t, reduce_max_u32_neon)
_VEC_WRAP(max_u64, uint64_t, reduce_max_u64_neon)
_VEC_WRAP(min_i8, int8_t, reduce_min_i8_neon)
_VEC_WRAP(min_i16, int16_t, reduce_min_i16_neon)
_VEC_WRAP(min_i32, int32_t, reduce_min_i32_neon)
_VEC_WRAP(min_i64, int64_t, reduce_min_i64_neon)
_VEC_WRAP(min_u8, uint8_t, reduce_min_u8_neon)
_VEC_WRAP(min_u16, uint16_t, reduce_min_u16_neon)
_VEC_WRAP(min_u32, uint32_t, reduce_min_u32_neon)
_VEC_WRAP(min_u64, uint64_t, reduce_min_u64_neon)

#elif NUMC_HAVE_RVV


_VEC_WRAP(max_i8, int8_t, reduce_max_i8_rvv)
_VEC_WRAP(max_i16, int16_t, reduce_max_i16_rvv)
_VEC_WRAP(max_i32, int32_t, reduce_max_i32_rvv)
_VEC_WRAP(max_i64, int64_t, reduce_max_i64_rvv)
_VEC_WRAP(max_u8, uint8_t, reduce_max_u8_rvv)
_VEC_WRAP(max_u16, uint16_t, reduce_max_u16_rvv)
_VEC_WRAP(max_u32, uint32_t, reduce_max_u32_rvv)
_VEC_WRAP(max_u64, uint64_t, reduce_max_u64_rvv)
_VEC_WRAP(min_i8, int8_t, reduce_min_i8_rvv)
_VEC_WRAP(min_i16, int16_t, reduce_min_i16_rvv)
_VEC_WRAP(min_i32, int32_t, reduce_min_i32_rvv)
_VEC_WRAP(min_i64, int64_t, reduce_min_i64_rvv)
_VEC_WRAP(min_u8, uint8_t, reduce_min_u8_rvv)
_VEC_WRAP(min_u16, uint16_t, reduce_min_u16_rvv)
_VEC_WRAP(min_u32, uint32_t, reduce_min_u32_rvv)
_VEC_WRAP(min_u64, uint64_t, reduce_min_u64_rvv)

#else /* scalar fallback */


DEFINE_VEC_MINMAX(max_i8, int8_t, INT8_MIN, NUMC_VMAX)
DEFINE_VEC_MINMAX(max_i16, int16_t, INT16_MIN, NUMC_VMAX)
DEFINE_VEC_MINMAX(max_i32, int32_t, INT32_MIN, NUMC_VMAX)
DEFINE_VEC_MINMAX(max_i64, int64_t, INT64_MIN, NUMC_VMAX)
DEFINE_VEC_MINMAX(max_u8, uint8_t, 0, NUMC_VMAX)
DEFINE_VEC_MINMAX(max_u16, uint16_t, 0, NUMC_VMAX)
DEFINE_VEC_MINMAX(max_u32, uint32_t, 0, NUMC_VMAX)
DEFINE_VEC_MINMAX(max_u64, uint64_t, 0, NUMC_VMAX)
DEFINE_VEC_MINMAX(min_i8, int8_t, INT8_MAX, NUMC_VMIN)
DEFINE_VEC_MINMAX(min_i16, int16_t, INT16_MAX, NUMC_VMIN)
DEFINE_VEC_MINMAX(min_i32, int32_t, INT32_MAX, NUMC_VMIN)
DEFINE_VEC_MINMAX(min_i64, int64_t, INT64_MAX, NUMC_VMIN)
DEFINE_VEC_MINMAX(min_u8, uint8_t, UINT8_MAX, NUMC_VMIN)
DEFINE_VEC_MINMAX(min_u16, uint16_t, UINT16_MAX, NUMC_VMIN)
DEFINE_VEC_MINMAX(min_u32, uint32_t, UINT32_MAX, NUMC_VMIN)
DEFINE_VEC_MINMAX(min_u64, uint64_t, UINT64_MAX, NUMC_VMIN)

#endif

#undef _VEC_WRAP

#undef DEFINE_VEC_MINMAX
#undef NUMC_VMAX
#undef NUMC_VMIN

#endif
