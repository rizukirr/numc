/**
 * @file math_neon.h
 * @brief Vectorized log/exp/sin/cos intrinsics for AArch64 NEON.
 *
 * Same fdlibm/Cephes algorithms as math_avx2.h, ported to NEON
 * 128-bit intrinsics: 4-wide float32, 2-wide float64.
 */
#ifndef NUMC_MATH_NEON_H
#define NUMC_MATH_NEON_H

#include <arm_neon.h>

/* ===================================================================
   Vectorized log — float32 (4-wide, fdlibm algorithm)
   =================================================================== */

static inline float32x4_t _neon_log_f32(float32x4_t x) {
  const float32x4_t ln2 = vdupq_n_f32(6.9314718056e-01f);
  const float32x4_t lg1 = vdupq_n_f32(6.6666668653e-01f);
  const float32x4_t lg2 = vdupq_n_f32(4.0000004172e-01f);
  const float32x4_t lg3 = vdupq_n_f32(2.8571429849e-01f);
  const float32x4_t lg4 = vdupq_n_f32(2.2222198009e-01f);
  const float32x4_t half = vdupq_n_f32(0.5f);
  const float32x4_t one = vdupq_n_f32(1.0f);
  const float32x4_t two = vdupq_n_f32(2.0f);
  const float32x4_t sqrt2 = vdupq_n_f32(1.41421356f);

  /* Step 1: Argument reduction */
  uint32x4_t ix = vreinterpretq_u32_f32(x);

  /* k = exponent - 127 */
  uint32x4_t exp_bits = vshrq_n_u32(ix, 23);
  exp_bits = vandq_u32(exp_bits, vdupq_n_u32(0xFF));
  int32x4_t k_i = vsubq_s32(vreinterpretq_s32_u32(exp_bits), vdupq_n_s32(127));

  /* m = mantissa with exponent set to 0 (biased 127) */
  uint32x4_t mantissa = vorrq_u32(vandq_u32(ix, vdupq_n_u32(0x007FFFFF)),
                                  vdupq_n_u32(0x3F800000));
  float32x4_t m = vreinterpretq_f32_u32(mantissa);

  /* If m > sqrt(2), halve m and increment k */
  uint32x4_t mask = vcgtq_f32(m, sqrt2);
  m = vbslq_f32(mask, vmulq_f32(m, half), m);
  k_i = vaddq_s32(k_i, vandq_s32(vreinterpretq_s32_u32(mask), vdupq_n_s32(1)));

  float32x4_t k = vcvtq_f32_s32(k_i);

  /* Step 2: f = m - 1 */
  float32x4_t f = vsubq_f32(m, one);

  /* Step 3: s = f / (2 + f) */
  float32x4_t s = vdivq_f32(f, vaddq_f32(two, f));
  float32x4_t z = vmulq_f32(s, s);
  float32x4_t w = vmulq_f32(z, z);

  /* Step 4: Horner polynomial */
  float32x4_t t1 = vfmaq_f32(lg2, w, lg4);
  t1 = vmulq_f32(w, t1);
  float32x4_t t2 = vfmaq_f32(lg1, w, lg3);
  t2 = vmulq_f32(z, t2);
  float32x4_t r = vaddq_f32(t1, t2);

  /* Step 5: result = k*ln2 + f - hfsq + s*(hfsq + r) */
  float32x4_t hfsq = vmulq_f32(half, vmulq_f32(f, f));
  float32x4_t result = vfmaq_f32(
      vaddq_f32(vsubq_f32(f, hfsq), vmulq_f32(s, vaddq_f32(hfsq, r))), k, ln2);

  /* Handle x <= 0: return 0 */
  uint32x4_t pos_mask = vcgtq_f32(x, vdupq_n_f32(0.0f));
  result =
      vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(result), pos_mask));

  return result;
}

/* ===================================================================
   Vectorized exp — float32 (4-wide, Cephes algorithm)
   =================================================================== */

static inline float32x4_t _neon_exp_f32(float32x4_t x) {
  const float32x4_t log2e = vdupq_n_f32(1.44269504088896341f);
  const float32x4_t ln2hi = vdupq_n_f32(6.93359375000000000e-1f);
  const float32x4_t ln2lo = vdupq_n_f32(-2.12194440e-4f);
  const float32x4_t p0 = vdupq_n_f32(1.9875691500e-4f);
  const float32x4_t p1 = vdupq_n_f32(1.3981999507e-3f);
  const float32x4_t p2 = vdupq_n_f32(8.3334519073e-3f);
  const float32x4_t p3 = vdupq_n_f32(4.1665795894e-2f);
  const float32x4_t p4 = vdupq_n_f32(1.6666665459e-1f);
  const float32x4_t p5 = vdupq_n_f32(5.0000001201e-1f);
  const float32x4_t one = vdupq_n_f32(1.0f);

  /* Save original for underflow/overflow masking */
  const float32x4_t x_orig = x;

  /* Clamp */
  const float32x4_t exp_hi = vdupq_n_f32(88.3762626647949f);
  const float32x4_t exp_lo = vdupq_n_f32(-103.972076f);
  x = vmaxq_f32(x, exp_lo);
  x = vminq_f32(x, exp_hi);

  /* n = round(x * log2e) */
  float32x4_t n = vrndnq_f32(vmulq_f32(x, log2e));

  /* r = x - n * ln2 (compensated) */
  float32x4_t r = vfmsq_f32(x, n, ln2hi);
  r = vfmsq_f32(r, n, ln2lo);

  /* Horner polynomial */
  float32x4_t p = p0;
  p = vfmaq_f32(p1, p, r);
  p = vfmaq_f32(p2, p, r);
  p = vfmaq_f32(p3, p, r);
  p = vfmaq_f32(p4, p, r);
  p = vfmaq_f32(p5, p, r);
  p = vfmaq_f32(vaddq_f32(r, one), p, vmulq_f32(r, r));

  /* Scale by 2^n */
  int32x4_t ni = vcvtq_s32_f32(n);
  ni = vshlq_n_s32(ni, 23);
  int32x4_t result_i = vaddq_s32(vreinterpretq_s32_f32(p), ni);

  float32x4_t result = vreinterpretq_f32_s32(result_i);

  /* Underflow: x < -103.972076 → 0, Overflow: x > 88.376 → +inf */
  uint32x4_t underflow = vcltq_f32(x_orig, exp_lo);
  uint32x4_t overflow = vcgtq_f32(x_orig, exp_hi);
  result = vbslq_f32(underflow, vdupq_n_f32(0.0f), result);
  result = vbslq_f32(overflow, vdupq_n_f32(1.0f / 0.0f), result);

  return result;
}

/* ===================================================================
   Vectorized log — float64 (2-wide, fdlibm algorithm)
   =================================================================== */

static inline float64x2_t _neon_log_f64(float64x2_t x) {
  const float64x2_t ln2 = vdupq_n_f64(6.9314718055994530942e-01);
  const float64x2_t lg1 = vdupq_n_f64(6.6666666666666735130e-01);
  const float64x2_t lg2 = vdupq_n_f64(3.9999999999940941908e-01);
  const float64x2_t lg3 = vdupq_n_f64(2.8571428743662391490e-01);
  const float64x2_t lg4 = vdupq_n_f64(2.2221984321497839600e-01);
  const float64x2_t lg5 = vdupq_n_f64(1.8183572161618050120e-01);
  const float64x2_t lg6 = vdupq_n_f64(1.5313837699209373320e-01);
  const float64x2_t lg7 = vdupq_n_f64(1.4798198605116585910e-01);
  const float64x2_t half = vdupq_n_f64(0.5);
  const float64x2_t one = vdupq_n_f64(1.0);
  const float64x2_t two = vdupq_n_f64(2.0);
  const float64x2_t sqrt2 = vdupq_n_f64(1.4142135623730951);

  /* Extract exponent */
  uint64x2_t ix = vreinterpretq_u64_f64(x);
  uint64x2_t exp_bits = vshrq_n_u64(ix, 52);
  exp_bits = vandq_u64(exp_bits, vdupq_n_u64(0x7FF));

  /* k = exponent - 1023 (scalar path since NEON lacks i64 cvt) */
  int64_t k_arr[2];
  k_arr[0] = (int64_t)vgetq_lane_u64(exp_bits, 0) - 1023;
  k_arr[1] = (int64_t)vgetq_lane_u64(exp_bits, 1) - 1023;
  float64x2_t k =
      vsetq_lane_f64((double)k_arr[1], vdupq_n_f64((double)k_arr[0]), 1);

  /* Normalize mantissa */
  uint64x2_t mant_mask = vdupq_n_u64(0x000FFFFFFFFFFFFFULL);
  uint64x2_t mant_bias = vdupq_n_u64(0x3FF0000000000000ULL);
  uint64x2_t mantissa = vorrq_u64(vandq_u64(ix, mant_mask), mant_bias);
  float64x2_t m = vreinterpretq_f64_u64(mantissa);

  /* If m > sqrt(2), halve m and increment k */
  uint64x2_t mask = vcgtq_f64(m, sqrt2);
  m = vbslq_f64(mask, vmulq_f64(m, half), m);
  k = vaddq_f64(
      k, vreinterpretq_f64_u64(vandq_u64(mask, vreinterpretq_u64_f64(one))));

  /* f = m - 1 */
  float64x2_t f = vsubq_f64(m, one);

  /* s = f / (2 + f) */
  float64x2_t s = vdivq_f64(f, vaddq_f64(two, f));
  float64x2_t z = vmulq_f64(s, s);
  float64x2_t w = vmulq_f64(z, z);

  /* Horner polynomial (7 coefficients, interleaved) */
  float64x2_t t1 = vfmaq_f64(lg4, w, lg6);
  t1 = vfmaq_f64(lg2, t1, w);
  t1 = vmulq_f64(w, t1);

  float64x2_t t2 = vfmaq_f64(lg5, w, lg7);
  t2 = vfmaq_f64(lg3, t2, w);
  t2 = vfmaq_f64(lg1, t2, w);
  t2 = vmulq_f64(z, t2);

  float64x2_t r = vaddq_f64(t1, t2);

  float64x2_t hfsq = vmulq_f64(half, vmulq_f64(f, f));
  float64x2_t result = vfmaq_f64(
      vaddq_f64(vsubq_f64(f, hfsq), vmulq_f64(s, vaddq_f64(hfsq, r))), k, ln2);

  /* Handle x <= 0 */
  uint64x2_t pos_mask = vcgtq_f64(x, vdupq_n_f64(0.0));
  result =
      vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(result), pos_mask));

  return result;
}

/* ===================================================================
   Vectorized exp — float64 (2-wide, Cephes algorithm)
   =================================================================== */

static inline float64x2_t _neon_exp_f64(float64x2_t x) {
  const float64x2_t log2e = vdupq_n_f64(1.44269504088896338700e+00);
  const float64x2_t ln2hi = vdupq_n_f64(6.93147180369123816490e-01);
  const float64x2_t ln2lo = vdupq_n_f64(1.90821492927058770002e-10);
  const float64x2_t c2 = vdupq_n_f64(5.00000000000000000000e-01);
  const float64x2_t c3 = vdupq_n_f64(1.66666666666666666667e-01);
  const float64x2_t c4 = vdupq_n_f64(4.16666666666666666667e-02);
  const float64x2_t c5 = vdupq_n_f64(8.33333333333333333333e-03);
  const float64x2_t c6 = vdupq_n_f64(1.38888888888888888889e-03);
  const float64x2_t c7 = vdupq_n_f64(1.98412698412698412698e-04);
  const float64x2_t c8 = vdupq_n_f64(2.48015873015873015873e-05);
  const float64x2_t c9 = vdupq_n_f64(2.75573192239858906526e-06);
  const float64x2_t c10 = vdupq_n_f64(2.75573192239858906526e-07);
  const float64x2_t c11 = vdupq_n_f64(2.50521083854417187751e-08);
  const float64x2_t c12 = vdupq_n_f64(2.08767569878680989792e-09);
  const float64x2_t one = vdupq_n_f64(1.0);

  /* Save original for underflow/overflow masking */
  const float64x2_t exp_lo = vdupq_n_f64(-745.133219101941217);
  const float64x2_t exp_hi = vdupq_n_f64(709.782712893383996843);
  const float64x2_t x_orig = x;

  /* Clamp */
  x = vmaxq_f64(x, exp_lo);
  x = vminq_f64(x, exp_hi);

  /* n = round(x * log2e) */
  float64x2_t n = vrndnq_f64(vmulq_f64(x, log2e));

  /* r = x - n * ln2 (compensated) */
  float64x2_t r = vfmsq_f64(x, n, ln2hi);
  r = vfmsq_f64(r, n, ln2lo);

  /* Horner polynomial */
  float64x2_t p = c12;
  p = vfmaq_f64(c11, p, r);
  p = vfmaq_f64(c10, p, r);
  p = vfmaq_f64(c9, p, r);
  p = vfmaq_f64(c8, p, r);
  p = vfmaq_f64(c7, p, r);
  p = vfmaq_f64(c6, p, r);
  p = vfmaq_f64(c5, p, r);
  p = vfmaq_f64(c4, p, r);
  p = vfmaq_f64(c3, p, r);
  p = vfmaq_f64(c2, p, r);
  p = vfmaq_f64(vaddq_f64(r, one), p, vmulq_f64(r, r));

  /* Scale by 2^n via IEEE 754 exponent field */
  double n_dbl[2];
  vst1q_f64(n_dbl, n);
  int64_t n_arr[2] = {(int64_t)n_dbl[0], (int64_t)n_dbl[1]};
  int64x2_t ni = vld1q_s64(n_arr);
  ni = vshlq_n_s64(ni, 52);
  int64x2_t result_i = vaddq_s64(vreinterpretq_s64_f64(p), ni);

  float64x2_t result = vreinterpretq_f64_s64(result_i);

  /* Underflow: x < -745.13 → 0, Overflow: x > 709.78 → +inf */
  uint64x2_t underflow = vcltq_f64(x_orig, exp_lo);
  uint64x2_t overflow = vcgtq_f64(x_orig, exp_hi);
  result = vbslq_f64(underflow, vdupq_n_f64(0.0), result);
  result = vbslq_f64(overflow, vdupq_n_f64(1.0 / 0.0), result);

  return result;
}

/* ===================================================================
   Vectorized sin — float32 (4-wide, Cephes algorithm)
   =================================================================== */

static inline float32x4_t _neon_sin_f32(float32x4_t x) {
  const float32x4_t minus_cephes_DP1 = vdupq_n_f32(-0.78515625f);
  const float32x4_t minus_cephes_DP2 = vdupq_n_f32(-2.4187564849853515625e-4f);
  const float32x4_t minus_cephes_DP3 = vdupq_n_f32(-3.77489497744594108e-8f);
  const float32x4_t sincof_p0 = vdupq_n_f32(-1.9515295891e-4f);
  const float32x4_t sincof_p1 = vdupq_n_f32(8.3321608736e-3f);
  const float32x4_t sincof_p2 = vdupq_n_f32(-1.6666654611e-1f);
  const float32x4_t coscof_p0 = vdupq_n_f32(2.443315711809948e-5f);
  const float32x4_t coscof_p1 = vdupq_n_f32(-1.388731625493765e-3f);
  const float32x4_t coscof_p2 = vdupq_n_f32(4.166664568298827e-2f);
  const float32x4_t fopi = vdupq_n_f32(1.27323954473516f);
  const float32x4_t half = vdupq_n_f32(0.5f);
  const float32x4_t one = vdupq_n_f32(1.0f);

  /* Extract sign and work with |x| */
  uint32x4_t sign_bit =
      vandq_u32(vreinterpretq_u32_f32(x), vdupq_n_u32(0x80000000));
  float32x4_t xa = vabsq_f32(x);

  /* j = (int)(|x| * 4/pi) */
  float32x4_t y = vmulq_f32(xa, fopi);
  int32x4_t j = vcvtq_s32_f32(y);
  /* Round up to even */
  j = vaddq_s32(j, vandq_s32(j, vdupq_n_s32(1)));
  y = vcvtq_f32_s32(j);

  /* j&4 -> sign swap */
  uint32x4_t swap_sign =
      vshlq_n_u32(vreinterpretq_u32_s32(vandq_s32(j, vdupq_n_s32(4))), 29);
  sign_bit = veorq_u32(sign_bit, swap_sign);

  /* j&2 -> use cosine polynomial */
  uint32x4_t poly_mask =
      vceqq_s32(vandq_s32(j, vdupq_n_s32(2)), vdupq_n_s32(2));

  /* Range reduction */
  xa = vfmaq_f32(xa, y, minus_cephes_DP1);
  xa = vfmaq_f32(xa, y, minus_cephes_DP2);
  xa = vfmaq_f32(xa, y, minus_cephes_DP3);

  float32x4_t z = vmulq_f32(xa, xa);

  /* cos polynomial */
  float32x4_t yc = coscof_p0;
  yc = vfmaq_f32(coscof_p1, yc, z);
  yc = vfmaq_f32(coscof_p2, yc, z);
  yc = vmulq_f32(yc, vmulq_f32(z, z));
  yc = vfmsq_f32(yc, half, z);
  yc = vaddq_f32(yc, one);

  /* sin polynomial */
  float32x4_t ys = sincof_p0;
  ys = vfmaq_f32(sincof_p1, ys, z);
  ys = vfmaq_f32(sincof_p2, ys, z);
  ys = vmulq_f32(ys, vmulq_f32(z, xa));
  ys = vaddq_f32(ys, xa);

  /* Select and apply sign */
  y = vbslq_f32(poly_mask, yc, ys);
  y = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(y), sign_bit));
  return y;
}

/* ===================================================================
   Vectorized cos — float32 (4-wide, Cephes algorithm)
   =================================================================== */

static inline float32x4_t _neon_cos_f32(float32x4_t x) {
  const float32x4_t minus_cephes_DP1 = vdupq_n_f32(-0.78515625f);
  const float32x4_t minus_cephes_DP2 = vdupq_n_f32(-2.4187564849853515625e-4f);
  const float32x4_t minus_cephes_DP3 = vdupq_n_f32(-3.77489497744594108e-8f);
  const float32x4_t sincof_p0 = vdupq_n_f32(-1.9515295891e-4f);
  const float32x4_t sincof_p1 = vdupq_n_f32(8.3321608736e-3f);
  const float32x4_t sincof_p2 = vdupq_n_f32(-1.6666654611e-1f);
  const float32x4_t coscof_p0 = vdupq_n_f32(2.443315711809948e-5f);
  const float32x4_t coscof_p1 = vdupq_n_f32(-1.388731625493765e-3f);
  const float32x4_t coscof_p2 = vdupq_n_f32(4.166664568298827e-2f);
  const float32x4_t fopi = vdupq_n_f32(1.27323954473516f);
  const float32x4_t half = vdupq_n_f32(0.5f);
  const float32x4_t one = vdupq_n_f32(1.0f);

  float32x4_t xa = vabsq_f32(x);

  float32x4_t y = vmulq_f32(xa, fopi);
  int32x4_t j = vcvtq_s32_f32(y);
  j = vaddq_s32(j, vandq_s32(j, vdupq_n_s32(1)));
  y = vcvtq_f32_s32(j);

  /* cos: shift j by 2 */
  j = vsubq_s32(j, vdupq_n_s32(2));

  /* sign: negate when ((j^4)&4) is set */
  uint32x4_t sign_bit =
      vshlq_n_u32(vreinterpretq_u32_s32(
                      vandq_s32(veorq_s32(j, vdupq_n_s32(4)), vdupq_n_s32(4))),
                  29);

  uint32x4_t poly_mask =
      vceqq_s32(vandq_s32(j, vdupq_n_s32(2)), vdupq_n_s32(2));

  xa = vfmaq_f32(xa, y, minus_cephes_DP1);
  xa = vfmaq_f32(xa, y, minus_cephes_DP2);
  xa = vfmaq_f32(xa, y, minus_cephes_DP3);

  float32x4_t z = vmulq_f32(xa, xa);

  float32x4_t yc = coscof_p0;
  yc = vfmaq_f32(coscof_p1, yc, z);
  yc = vfmaq_f32(coscof_p2, yc, z);
  yc = vmulq_f32(yc, vmulq_f32(z, z));
  yc = vfmsq_f32(yc, half, z);
  yc = vaddq_f32(yc, one);

  float32x4_t ys = sincof_p0;
  ys = vfmaq_f32(sincof_p1, ys, z);
  ys = vfmaq_f32(sincof_p2, ys, z);
  ys = vmulq_f32(ys, vmulq_f32(z, xa));
  ys = vaddq_f32(ys, xa);

  y = vbslq_f32(poly_mask, yc, ys);
  y = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(y), sign_bit));
  return y;
}

/* ===================================================================
   Combined sincos — float32 (4-wide)
   =================================================================== */

static inline void _neon_sincos_f32(float32x4_t x, float32x4_t *s,
                                    float32x4_t *c) {
  const float32x4_t minus_cephes_DP1 = vdupq_n_f32(-0.78515625f);
  const float32x4_t minus_cephes_DP2 = vdupq_n_f32(-2.4187564849853515625e-4f);
  const float32x4_t minus_cephes_DP3 = vdupq_n_f32(-3.77489497744594108e-8f);
  const float32x4_t sincof_p0 = vdupq_n_f32(-1.9515295891e-4f);
  const float32x4_t sincof_p1 = vdupq_n_f32(8.3321608736e-3f);
  const float32x4_t sincof_p2 = vdupq_n_f32(-1.6666654611e-1f);
  const float32x4_t coscof_p0 = vdupq_n_f32(2.443315711809948e-5f);
  const float32x4_t coscof_p1 = vdupq_n_f32(-1.388731625493765e-3f);
  const float32x4_t coscof_p2 = vdupq_n_f32(4.166664568298827e-2f);
  const float32x4_t fopi = vdupq_n_f32(1.27323954473516f);
  const float32x4_t half = vdupq_n_f32(0.5f);
  const float32x4_t one = vdupq_n_f32(1.0f);

  uint32x4_t sign_sin =
      vandq_u32(vreinterpretq_u32_f32(x), vdupq_n_u32(0x80000000));
  float32x4_t xa = vabsq_f32(x);

  float32x4_t y = vmulq_f32(xa, fopi);
  int32x4_t j = vcvtq_s32_f32(y);
  j = vaddq_s32(j, vandq_s32(j, vdupq_n_s32(1)));
  y = vcvtq_f32_s32(j);

  int32x4_t j4 = vandq_s32(j, vdupq_n_s32(4));
  sign_sin = veorq_u32(sign_sin, vshlq_n_u32(vreinterpretq_u32_s32(j4), 29));

  /* cos sign */
  uint32x4_t sign_cos =
      vshlq_n_u32(vreinterpretq_u32_s32(vandq_s32(
                      veorq_s32(vsubq_s32(j, vdupq_n_s32(2)), vdupq_n_s32(4)),
                      vdupq_n_s32(4))),
                  29);

  uint32x4_t poly_mask =
      vceqq_s32(vandq_s32(j, vdupq_n_s32(2)), vdupq_n_s32(2));

  xa = vfmaq_f32(xa, y, minus_cephes_DP1);
  xa = vfmaq_f32(xa, y, minus_cephes_DP2);
  xa = vfmaq_f32(xa, y, minus_cephes_DP3);

  float32x4_t z = vmulq_f32(xa, xa);

  float32x4_t yc = coscof_p0;
  yc = vfmaq_f32(coscof_p1, yc, z);
  yc = vfmaq_f32(coscof_p2, yc, z);
  yc = vmulq_f32(yc, vmulq_f32(z, z));
  yc = vfmsq_f32(yc, half, z);
  yc = vaddq_f32(yc, one);

  float32x4_t yz = sincof_p0;
  yz = vfmaq_f32(sincof_p1, yz, z);
  yz = vfmaq_f32(sincof_p2, yz, z);
  yz = vmulq_f32(yz, vmulq_f32(z, xa));
  yz = vaddq_f32(yz, xa);

  /* sin */
  *s = vreinterpretq_f32_u32(
      veorq_u32(vreinterpretq_u32_f32(vbslq_f32(poly_mask, yc, yz)), sign_sin));

  /* cos: opposite poly selection */
  uint32x4_t cos_poly_mask = vmvnq_u32(poly_mask);
  *c = vreinterpretq_f32_u32(veorq_u32(
      vreinterpretq_u32_f32(vbslq_f32(cos_poly_mask, yc, yz)), sign_cos));
}

#endif /* NUMC_MATH_NEON_H */
