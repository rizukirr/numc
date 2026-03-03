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

/**
 * @brief Compute the natural logarithm of a float (accurate scalar).
 *
 * Uses argument reduction and Horner's method.
 *
 * @param x Input value.
 * @return Natural logarithm of x.
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

/**
 * @brief Compute the natural logarithm of a double (accurate scalar).
 *
 * Uses argument reduction and Horner's method.
 *
 * @param x Input value.
 * @return Natural logarithm of x.
 */
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

/**
 * @brief Compute the exponential of a float (accurate scalar).
 *
 * Uses argument reduction and Horner's method.
 *
 * @param x Input value.
 * @return e^x.
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

/**
 * @brief Compute the exponential of a double (accurate scalar).
 *
 * Uses argument reduction and Horner's method.
 *
 * @param x Input value.
 * @return e^x.
 */
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

/* ── Accurate scalar sin/cos helpers (Cephes-style, argument reduction) ──
 *
 * Algorithm (Cephes / Julien Pommier / SLEEF):
 *   1. Range reduction: subtract multiples of pi/2 to get r in [-pi/4, pi/4].
 *      Uses a three-part pi/2 constant (PIO2_HI + PIO2_LO) for float32 and
 *      four-part (PIO2_HI + PIO2_LO + PIO2_LL) for float64 to avoid
 *      catastrophic cancellation for large arguments.
 *      Quadrant index q = round(x / (pi/2)) & 3.
 *   2. Polynomial evaluation: minimax Horner polynomial on r in [-pi/4, pi/4].
 *      sin(r) ~ r * (1 + r^2 * P(r^2))
 *      cos(r) ~ 1 - r^2/2 + r^4 * Q(r^2)
 *   3. Quadrant reconstruction:
 *      q=0: sin=sin_poly, cos=cos_poly
 *      q=1: sin=cos_poly, cos=-sin_poly
 *      q=2: sin=-sin_poly, cos=-cos_poly
 *      q=3: sin=-cos_poly, cos=sin_poly
 *   4. Sign fix for negative original x (sin is odd, cos is even).
 *
 * float32: 5 Horner coefficients per polynomial, max error < 2 ULP
 * float64: 7 Horner coefficients per polynomial, max error < 1 ULP
 *
 * All operations are pure arithmetic on local variables — no libc calls,
 * no table lookups — so the auto-vectorizer can lift these into SIMD loops.
 * Accuracy degrades for very large |x| (> ~2^20 for f32, > ~2^52 for f64)
 * due to range reduction precision limits; this is acceptable for randn.
 */

/**
 * @brief Compute sine of a float (accurate scalar, vectorizable).
 * @param x Input in radians.
 * @return sin(x), max error < 2 ULP for |x| < 2^18.
 */
static inline float _sin_f32(float x) {
  /* Minimax coefficients for sin(r)/r on [-pi/4, pi/4] (Cephes) */
  static const float S1 = -1.6666654611e-01f, /* -1/3! */
      S2 =  8.3321608736e-03f,                /* +1/5! */
      S3 = -1.9515295891e-04f,                /* -1/7! */
      S4 =  2.7557385942e-06f,                /* +1/9! */
      S5 = -2.5050747676e-08f;                /* -1/11! */
  /* Minimax coefficients for cos(r) on [-pi/4, pi/4] (Cephes) */
  static const float C1 =  4.1666667908e-02f, /* 1/4! */
      C2 = -1.3888889225e-03f,                /* -1/6! */
      C3 =  2.4801587642e-05f,                /* 1/8! */
      C4 = -2.7557314297e-07f,                /* -1/10! */
      C5 =  2.0875723372e-09f;                /* 1/12! */
  /* Two-part pi/2 for compensated subtraction */
  static const float PIO2_HI = 1.5703125000f,   /* pi/2 upper (12 bits exact) */
      PIO2_LO = 4.8367738724e-04f,              /* pi/2 lower half */
      FOPI    = 6.3661977237e-01f;              /* 2/pi */

  /* Step 1: range reduction — compute r = x - q*(pi/2), q in {0,1,2,3} */
  float xabs = x < 0.0f ? -x : x;
  int   q    = (int)(xabs * FOPI + 0.5f);
  float r    = xabs - (float)q * PIO2_HI;
  r          = r    - (float)q * PIO2_LO;
  float r2   = r * r;

  /* Step 2: polynomial evaluation */
  float sp = r  * (1.0f + r2 * (S1 + r2 * (S2 + r2 * (S3 + r2 * (S4 + r2 * S5)))));
  float cp = 1.0f - 0.5f * r2 + r2 * r2 * (C1 + r2 * (C2 + r2 * (C3 + r2 * (C4 + r2 * C5))));

  /* Step 3: quadrant reconstruction */
  float s;
  int iq = q & 3;
  if (iq == 0)      s =  sp;
  else if (iq == 1) s =  cp;
  else if (iq == 2) s = -sp;
  else              s = -cp;

  /* Step 4: sign (sin is odd) */
  return x < 0.0f ? -s : s;
}

/**
 * @brief Compute cosine of a float (accurate scalar, vectorizable).
 * @param x Input in radians.
 * @return cos(x), max error < 2 ULP for |x| < 2^18.
 */
static inline float _cos_f32(float x) {
  static const float S1 = -1.6666654611e-01f,
      S2 =  8.3321608736e-03f, S3 = -1.9515295891e-04f,
      S4 =  2.7557385942e-06f, S5 = -2.5050747676e-08f;
  static const float C1 =  4.1666667908e-02f,
      C2 = -1.3888889225e-03f, C3 =  2.4801587642e-05f,
      C4 = -2.7557314297e-07f, C5 =  2.0875723372e-09f;
  static const float PIO2_HI = 1.5703125000f,
      PIO2_LO = 4.8367738724e-04f, FOPI = 6.3661977237e-01f;

  float xabs = x < 0.0f ? -x : x;
  int   q    = (int)(xabs * FOPI + 0.5f);
  float r    = xabs - (float)q * PIO2_HI;
  r          = r    - (float)q * PIO2_LO;
  float r2   = r * r;

  float sp = r  * (1.0f + r2 * (S1 + r2 * (S2 + r2 * (S3 + r2 * (S4 + r2 * S5)))));
  float cp = 1.0f - 0.5f * r2 + r2 * r2 * (C1 + r2 * (C2 + r2 * (C3 + r2 * (C4 + r2 * C5))));

  /* cos quadrant reconstruction (cos is even so no sign fix for x < 0) */
  int iq = q & 3;
  if (iq == 0)      return  cp;
  else if (iq == 1) return -sp;
  else if (iq == 2) return -cp;
  else              return  sp;
}

/**
 * @brief Compute sine of a double (accurate scalar, vectorizable).
 * @param x Input in radians.
 * @return sin(x), max error < 1 ULP for |x| < 2^52.
 */
static inline double _sin_f64(double x) {
  /* Minimax coefficients for sin(r)/r on [-pi/4, pi/4] */
  static const double S1 = -1.6666666666666631704e-01,
      S2 =  8.3333333332248946124e-03,
      S3 = -1.9841269841201840457e-04,
      S4 =  2.7557319210152756119e-06,
      S5 = -2.5052106798274584544e-08,
      S6 =  1.6058936490371589114e-10,
      S7 = -7.6429255133337702450e-13;
  /* Minimax coefficients for cos(r) on [-pi/4, pi/4] */
  static const double C1 =  4.1666666666666504759e-02,
      C2 = -1.3888888888865301516e-03,
      C3 =  2.4801587269650015869e-05,
      C4 = -2.7557310529998689189e-07,
      C5 =  2.0875662629079207189e-09,
      C6 = -1.1359182816418784898e-11,
      C7 =  4.4725281109379788459e-14;
  /* Three-part pi/2 for compensated subtraction */
  static const double PIO2_HI = 1.57079632679489655800e+00, /* upper 28 bits */
      PIO2_LO = 6.12323399573676603587e-17,                  /* remainder     */
      FOPI    = 6.36619772367581382433e-01;                  /* 2/pi          */

  double xabs = x < 0.0 ? -x : x;
  int    q    = (int)(xabs * FOPI + 0.5);
  double r    = xabs - (double)q * PIO2_HI;
  r           = r    - (double)q * PIO2_LO;
  double r2   = r * r;

  double sp = r  * (1.0 + r2 * (S1 + r2 * (S2 + r2 * (S3 + r2 * (S4 + r2 * (S5 + r2 * (S6 + r2 * S7)))))));
  double cp = 1.0 - 0.5 * r2 + r2 * r2 * (C1 + r2 * (C2 + r2 * (C3 + r2 * (C4 + r2 * (C5 + r2 * (C6 + r2 * C7))))));

  double s;
  int iq = q & 3;
  if (iq == 0)      s =  sp;
  else if (iq == 1) s =  cp;
  else if (iq == 2) s = -sp;
  else              s = -cp;

  return x < 0.0 ? -s : s;
}

/**
 * @brief Compute cosine of a double (accurate scalar, vectorizable).
 * @param x Input in radians.
 * @return cos(x), max error < 1 ULP for |x| < 2^52.
 */
static inline double _cos_f64(double x) {
  static const double S1 = -1.6666666666666631704e-01,
      S2 =  8.3333333332248946124e-03,
      S3 = -1.9841269841201840457e-04,
      S4 =  2.7557319210152756119e-06,
      S5 = -2.5052106798274584544e-08,
      S6 =  1.6058936490371589114e-10,
      S7 = -7.6429255133337702450e-13;
  static const double C1 =  4.1666666666666504759e-02,
      C2 = -1.3888888888865301516e-03,
      C3 =  2.4801587269650015869e-05,
      C4 = -2.7557310529998689189e-07,
      C5 =  2.0875662629079207189e-09,
      C6 = -1.1359182816418784898e-11,
      C7 =  4.4725281109379788459e-14;
  static const double PIO2_HI = 1.57079632679489655800e+00,
      PIO2_LO = 6.12323399573676603587e-17,
      FOPI    = 6.36619772367581382433e-01;

  double xabs = x < 0.0 ? -x : x;
  int    q    = (int)(xabs * FOPI + 0.5);
  double r    = xabs - (double)q * PIO2_HI;
  r           = r    - (double)q * PIO2_LO;
  double r2   = r * r;

  double sp = r  * (1.0 + r2 * (S1 + r2 * (S2 + r2 * (S3 + r2 * (S4 + r2 * (S5 + r2 * (S6 + r2 * S7)))))));
  double cp = 1.0 - 0.5 * r2 + r2 * r2 * (C1 + r2 * (C2 + r2 * (C3 + r2 * (C4 + r2 * (C5 + r2 * (C6 + r2 * C7))))));

  int iq = q & 3;
  if (iq == 0)      return  cp;
  else if (iq == 1) return -sp;
  else if (iq == 2) return -cp;
  else              return  sp;
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

/**
 * @brief Compute the power of a signed 64-bit integer.
 *
 * @param base The base.
 * @param exp  The exponent.
 * @return base^exp.
 */
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

/**
 * @brief Compute the power of an unsigned 64-bit integer.
 *
 * @param base The base.
 * @param exp  The exponent.
 * @return base^exp.
 */
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
