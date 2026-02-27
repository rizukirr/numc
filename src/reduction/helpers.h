#ifndef NUMC_REDUCTION_HELPERS_H
#define NUMC_REDUCTION_HELPERS_H

#include <math.h>
#include <stddef.h>

/* ── Pairwise summation for float types ──────────────────────────────
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

/* ── Multi-accumulator max/min for float types ─────────────────────────
 *
 * Same technique as pairwise summation: 8 independent accumulators
 * that the SLP vectorizer packs into vmaxps/vminps (float32) or
 * vmaxpd/vminpd (float64).
 *
 * Unlike pairwise sum, no recursion needed — max/min is exact
 * (no rounding error), so a flat multi-accumulator loop suffices.
 */

#define NUMC_VMAX_(a, b) ((a) > (b) ? (a) : (b))
#define NUMC_VMIN_(a, b) ((a) < (b) ? (a) : (b))

#define DEFINE_VEC_MINMAX_(NAME, TYPE, INIT, OP)                               \
  static inline TYPE _vec_##NAME(const TYPE *restrict a, size_t n) {           \
    TYPE r0 = INIT, r1 = INIT, r2 = INIT, r3 = INIT;                          \
    TYPE r4 = INIT, r5 = INIT, r6 = INIT, r7 = INIT;                          \
    size_t i = 0, n8 = n & ~(size_t)7;                                        \
    for (; i < n8; i += 8) {                                                   \
      r0 = OP(a[i], r0);                                                      \
      r1 = OP(a[i + 1], r1);                                                  \
      r2 = OP(a[i + 2], r2);                                                  \
      r3 = OP(a[i + 3], r3);                                                  \
      r4 = OP(a[i + 4], r4);                                                  \
      r5 = OP(a[i + 5], r5);                                                  \
      r6 = OP(a[i + 6], r6);                                                  \
      r7 = OP(a[i + 7], r7);                                                  \
    }                                                                          \
    TYPE m = OP(OP(OP(r0, r1), OP(r2, r3)), OP(OP(r4, r5), OP(r6, r7)));      \
    for (; i < n; i++)                                                         \
      m = OP(a[i], m);                                                         \
    return m;                                                                  \
  }

DEFINE_VEC_MINMAX_(max_f32, float, -INFINITY, NUMC_VMAX_)
DEFINE_VEC_MINMAX_(max_f64, double, -INFINITY, NUMC_VMAX_)
DEFINE_VEC_MINMAX_(min_f32, float, INFINITY, NUMC_VMIN_)
DEFINE_VEC_MINMAX_(min_f64, double, INFINITY, NUMC_VMIN_)

#undef DEFINE_VEC_MINMAX_
#undef NUMC_VMAX_
#undef NUMC_VMIN_

#endif
