#ifndef NUMC_MATH_HELPERS_H
#define NUMC_MATH_HELPERS_H

#include <math.h>
#include <stdint.h>
#include <string.h>

/* ── Accurate scalar log helpers (fdlibm, argument reduction + Horner) ──
 *
 * Algorithm (same as fdlibm / musl libm):
 *   1. Argument reduction: x = 2^k * m, m in [sqrt(2)/2, sqrt(2))
 *   2. f = m - 1  (f in [-0.293, 0.414])
 *   3. s = f/(2+f)  (s in [-0.172, 0.172] — keeps polynomial convergent)
 *   4. log(m) = f - hfsq + s*(hfsq + R)  where R is Horner polynomial in s^2
 *   5. log(x) = k*ln2 + log(m)
 *
 * float32: 4 Horner coefficients, max error < 0.5 ULP
 * float64: 7 Horner coefficients, max error < 1 ULP
 */

static inline float _log_f32(float x) {
  static const float ln2 = 6.9314718056e-01f,
                     Lg1 = 6.6666668653e-01f, /* 0x3F2AAAAB */
      Lg2 = 4.0000004172e-01f,                /* 0x3ECCCCCD */
      Lg3 = 2.8571429849e-01f,                /* 0x3E924925 */
      Lg4 = 2.2222198009e-01f;                /* 0x3E638E29 */

  if (x <= 0.0f)
    return 0.0f;

  /* Argument reduction: x = 2^k * m, m in [sqrt(2)/2, sqrt(2)) */
  uint32_t ix;
  memcpy(&ix, &x, sizeof ix);
  int k = (int)((ix >> 23) & 0xffu) - 127;
  ix = (ix & 0x007fffffu) | 0x3f800000u; /* set biased exponent to 0 */
  float m;
  memcpy(&m, &ix, sizeof m);
  if (m > 1.41421356f) {
    m *= 0.5f;
    k++;
  }
  float f = m - 1.0f;

  float s = f / (2.0f + f);
  float z = s * s;
  float w = z * z;
  float t1 = w * (Lg2 + w * Lg4);
  float t2 = z * (Lg1 + w * Lg3);
  float R = t1 + t2;
  float hfsq = 0.5f * f * f;
  return (float)k * ln2 + f - hfsq + s * (hfsq + R);
}

static inline double _log_f64(double x) {
  static const double ln2 = 6.9314718055994530942e-01,
                      Lg1 = 6.6666666666666735130e-01, /* 0x3FE5555555555593 */
      Lg2 = 3.9999999999940941908e-01,                 /* 0x3FD999999997FA04 */
      Lg3 = 2.8571428743662391490e-01,                 /* 0x3FD2492494229359 */
      Lg4 = 2.2221984321497839600e-01,                 /* 0x3FCC71C51D8E78AF */
      Lg5 = 1.8183572161618050120e-01,                 /* 0x3FC7466496CB03DE */
      Lg6 = 1.5313837699209373320e-01,                 /* 0x3FC39A09D078C69F */
      Lg7 = 1.4798198605116585910e-01;                 /* 0x3FC2F112DF3E5244 */

  if (x <= 0.0)
    return 0.0;

  /* Argument reduction: x = 2^k * m, m in [sqrt(2)/2, sqrt(2)) */
  uint64_t ix;
  memcpy(&ix, &x, sizeof ix);
  int k = (int)((ix >> 52) & 0x7ffULL) - 1023;
  ix = (ix & 0x000fffffffffffffULL) | 0x3ff0000000000000ULL;
  double m;
  memcpy(&m, &ix, sizeof m);
  if (m > 1.4142135623730951) {
    m *= 0.5;
    k++;
  }
  double f = m - 1.0;

  double s = f / (2.0 + f);
  double z = s * s;
  double w = z * z;
  double t1 = w * (Lg2 + w * (Lg4 + w * Lg6));
  double t2 = z * (Lg1 + w * (Lg3 + w * (Lg5 + w * Lg7)));
  double R = t1 + t2;
  double hfsq = 0.5 * f * f;
  return (double)k * ln2 + f - hfsq + s * (hfsq + R);
}

/* ── Accurate scalar exp helpers (Cephes-style, argument reduction + Horner) ──
 *
 * Algorithm (Cephes / Julien Pommier):
 *   1. Clamp: x > 88.38 -> inf, x < -103.97 -> 0
 *   2. Argument reduction: n = round(x / ln2), r = x - n*ln2 (compensated)
 *   3. Horner polynomial: exp(r) for |r| <= ln2/2
 *   4. Scale: multiply by 2^n via IEEE 754 exponent field bit-add
 *
 * float32: 6 Remez coefficients, max error < 1 ULP
 * float64: 11 Taylor coefficients (1/n!), truncation error < 0.23 * 2^-53
 */

static inline float _exp_f32(float x) {
  static const float LOG2E = 1.44269504088896341f, /* log2(e) = 1/ln2 */
      LN2HI = 6.93359375000000000e-1f, /* ln2 upper half (355/512, exact)    */
      LN2LO = -2.12194440e-4f,         /* ln2 lower half                     */
      /* Remez-optimized Horner coefficients (Cephes / Julien Pommier)       */
      P0 = 1.9875691500e-4f, P1 = 1.3981999507e-3f, P2 = 8.3334519073e-3f,
                     P3 = 4.1665795894e-2f, P4 = 1.6666665459e-1f,
                     P5 = 5.0000001201e-1f;

  /* Step 0: clamp */
  if (x > 88.3762626647949f)
    return 1.0f / 0.0f; /* +inf */
  if (x < -103.972076f)
    return 0.0f;

  /* Step 1: argument reduction */
  float n = roundf(x * LOG2E);

  /* Step 2: compensated subtraction (two-part ln2 avoids cancellation) */
  float r = x - n * LN2HI;
  r = r - n * LN2LO;

  /* Step 3: Horner polynomial for exp(r), |r| <= ln2/2 */
  float p = P0;
  p = p * r + P1;
  p = p * r + P2;
  p = p * r + P3;
  p = p * r + P4;
  p = p * r + P5;
  p = p * r * r + r + 1.0f;

  /* Step 4: scale by 2^n via IEEE 754 exponent field (bits 23-30) */
  int32_t ni = (int32_t)n;
  uint32_t bits;
  memcpy(&bits, &p, sizeof bits);
  bits += (uint32_t)(ni << 23);
  memcpy(&p, &bits, sizeof p);

  return p;
}

static inline double _exp_f64(double x) {
  static const double LOG2E = 1.44269504088896338700e+00,
                      LN2HI =
                          6.93147180369123816490e-01, /* lower 28 bits zero */
      LN2LO = 1.90821492927058770002e-10,             /* remainder          */
      /* Taylor coefficients 1/n! for n = 2..12 */
      C2 = 5.00000000000000000000e-01, C3 = 1.66666666666666666667e-01,
                      C4 = 4.16666666666666666667e-02,
                      C5 = 8.33333333333333333333e-03,
                      C6 = 1.38888888888888888889e-03,
                      C7 = 1.98412698412698412698e-04,
                      C8 = 2.48015873015873015873e-05,
                      C9 = 2.75573192239858906526e-06,
                      C10 = 2.75573192239858906526e-07,
                      C11 = 2.50521083854417187751e-08,
                      C12 = 2.08767569878680989792e-09;

  /* Step 0: clamp */
  if (x > 709.782712893383996843)
    return 1.0 / 0.0; /* +inf */
  if (x < -745.133219101941217)
    return 0.0;

  /* Step 1: argument reduction */
  double n = round(x * LOG2E);

  /* Step 2: compensated subtraction */
  double r = x - n * LN2HI;
  r = r - n * LN2LO;

  /* Step 3: Horner polynomial for exp(r), |r| <= ln2/2 */
  double p = C12;
  p = p * r + C11;
  p = p * r + C10;
  p = p * r + C9;
  p = p * r + C8;
  p = p * r + C7;
  p = p * r + C6;
  p = p * r + C5;
  p = p * r + C4;
  p = p * r + C3;
  p = p * r + C2;
  p = p * r * r + r + 1.0;

  /* Step 4: scale by 2^n via IEEE 754 exponent field (bits 52-62) */
  int64_t ni = (int64_t)n;
  uint64_t bits;
  memcpy(&bits, &p, sizeof bits);
  bits += (uint64_t)(ni << 52);
  memcpy(&p, &bits, sizeof p);

  return p;
}

/* ── Integer pow helpers ──────────────────────────────────────────────
 *
 * Two strategies, chosen by element width:
 *
 * 8/16-bit — branchless fixed-iteration (auto-vectorizes to SIMD):
 *   Fixed iteration count (7-16) lets the compiler fully unroll the inner
 *   loop. Branchless mask selects base or 1 per exponent bit. The outer
 *   element loop then auto-vectorizes because every element executes the
 *   same number of operations. Unsigned arithmetic for well-defined overflow.
 *
 * 32/64-bit — variable-iteration with early exit (scalar, optimal throughput):
 *   For 32/64-bit types, the fixed iteration count (31/63) overwhelms
 *   the SIMD benefit. Variable-iteration exits early for typical small
 *   exponents (2-4 iterations for exp <= 10). AVX2 also lacks vpmullq
 *   for 64-bit, so int64 can't vectorize regardless.
 *
 * Negative bases work naturally: squaring produces positive values,
 * and the odd-bit selection from the original base preserves sign.
 */

/* ── 8/16-bit: branchless fixed-iteration (vectorizable) ───────────── */

#define DEFINE_POWI_SIGNED(NAME, CT, UCT, BITS)                                \
  __attribute__((always_inline)) static inline CT NAME(CT base, CT exp) {      \
    int neg = exp < 0;                                                         \
    UCT uexp = neg ? 0 : (UCT)exp;                                             \
    UCT ubase = (UCT)base;                                                     \
    UCT result = 1;                                                            \
    for (int bit = 0; bit < BITS; bit++) {                                     \
      UCT mask = -((uexp >> bit) & 1);                                         \
      result *= ((ubase - 1) & mask) + 1;                                      \
      ubase *= ubase;                                                          \
    }                                                                          \
    return neg ? 0 : (CT)result;                                               \
  }

#define DEFINE_POWI_UNSIGNED(NAME, UCT, BITS)                                  \
  __attribute__((always_inline)) static inline UCT NAME(UCT base, UCT exp) {   \
    UCT result = 1;                                                            \
    for (int bit = 0; bit < BITS; bit++) {                                     \
      UCT mask = -((exp >> bit) & 1);                                          \
      result *= ((base - 1) & mask) + 1;                                       \
      base *= base;                                                            \
    }                                                                          \
    return result;                                                             \
  }

DEFINE_POWI_SIGNED(_powi_i8, int8_t, uint8_t, 7)
DEFINE_POWI_SIGNED(_powi_i16, int16_t, uint16_t, 15)

DEFINE_POWI_UNSIGNED(_powi_u8, uint8_t, 8)
DEFINE_POWI_UNSIGNED(_powi_u16, uint16_t, 16)

/* ── 32/64-bit: variable-iteration with early exit (scalar, fast) ──── */

static inline int64_t _powi_signed(int64_t base, int64_t exp) {
  if (exp < 0)
    return 0;
  int64_t result = 1;
  while (exp > 0) {
    if (exp & 1)
      result *= base;
    base *= base;
    exp >>= 1;
  }
  return result;
}

static inline uint64_t _powi_unsigned(uint64_t base, uint64_t exp) {
  uint64_t result = 1;
  while (exp > 0) {
    if (exp & 1)
      result *= base;
    base *= base;
    exp >>= 1;
  }
  return result;
}

#endif
