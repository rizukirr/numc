/**
 * @file math_sve.h
 * @brief Vectorized log/exp/sin/cos intrinsics for ARM SVE.
 *
 * Same fdlibm/Cephes algorithms as math_avx2.h, ported to SVE
 * scalable-width intrinsics. All operations are predicated.
 */
#ifndef NUMC_MATH_SVE_H
#define NUMC_MATH_SVE_H

#include <arm_sve.h>

/* ===================================================================
   Vectorized log — float32 (VL-wide, fdlibm algorithm)
   =================================================================== */

static inline svfloat32_t _sve_log_f32(svfloat32_t x) {
  svbool_t ptrue = svptrue_b32();

  const svfloat32_t ln2 = svdup_f32(6.9314718056e-01f);
  const svfloat32_t lg1 = svdup_f32(6.6666668653e-01f);
  const svfloat32_t lg2 = svdup_f32(4.0000004172e-01f);
  const svfloat32_t lg3 = svdup_f32(2.8571429849e-01f);
  const svfloat32_t lg4 = svdup_f32(2.2222198009e-01f);
  const svfloat32_t half = svdup_f32(0.5f);
  const svfloat32_t one = svdup_f32(1.0f);
  const svfloat32_t two = svdup_f32(2.0f);
  const svfloat32_t sqrt2 = svdup_f32(1.41421356f);

  /* Step 1: Argument reduction */
  svuint32_t ix = svreinterpret_u32_f32(x);

  /* k = exponent - 127 */
  svuint32_t exp_bits = svlsr_n_u32_x(ptrue, ix, 23);
  exp_bits = svand_n_u32_x(ptrue, exp_bits, 0xFF);
  svint32_t k_i = svsub_n_s32_x(ptrue, svreinterpret_s32_u32(exp_bits), 127);

  /* m = mantissa with exponent set to 0 */
  svuint32_t mantissa =
      svorr_n_u32_x(ptrue, svand_n_u32_x(ptrue, ix, 0x007FFFFF), 0x3F800000);
  svfloat32_t m = svreinterpret_f32_u32(mantissa);

  /* If m > sqrt(2), halve m and increment k */
  svbool_t mask = svcmpgt_f32(ptrue, m, sqrt2);
  m = svsel_f32(mask, svmul_f32_x(ptrue, m, half), m);
  k_i = svadd_s32_m(mask, k_i, svdup_s32(1));

  svfloat32_t k = svcvt_f32_s32_x(ptrue, k_i);

  /* f = m - 1 */
  svfloat32_t f = svsub_f32_x(ptrue, m, one);

  /* s = f / (2 + f) */
  svfloat32_t s = svdiv_f32_x(ptrue, f, svadd_f32_x(ptrue, two, f));
  svfloat32_t z = svmul_f32_x(ptrue, s, s);
  svfloat32_t w = svmul_f32_x(ptrue, z, z);

  /* Horner polynomial */
  svfloat32_t t1 = svmla_f32_x(ptrue, lg2, w, lg4);
  t1 = svmul_f32_x(ptrue, w, t1);
  svfloat32_t t2 = svmla_f32_x(ptrue, lg1, w, lg3);
  t2 = svmul_f32_x(ptrue, z, t2);
  svfloat32_t r = svadd_f32_x(ptrue, t1, t2);

  /* result = k*ln2 + f - hfsq + s*(hfsq + r) */
  svfloat32_t hfsq = svmul_f32_x(ptrue, half, svmul_f32_x(ptrue, f, f));
  svfloat32_t result = svmla_f32_x(
      ptrue,
      svadd_f32_x(ptrue, svsub_f32_x(ptrue, f, hfsq),
                  svmul_f32_x(ptrue, s, svadd_f32_x(ptrue, hfsq, r))),
      k, ln2);

  /* Handle x <= 0 */
  svbool_t pos_mask = svcmpgt_f32(ptrue, x, svdup_f32(0.0f));
  result = svsel_f32(pos_mask, result, svdup_f32(0.0f));

  return result;
}

/* ===================================================================
   Vectorized exp — float32 (VL-wide, Cephes algorithm)
   =================================================================== */

static inline svfloat32_t _sve_exp_f32(svfloat32_t x) {
  svbool_t ptrue = svptrue_b32();

  const svfloat32_t log2e = svdup_f32(1.44269504088896341f);
  const svfloat32_t ln2hi = svdup_f32(6.93359375000000000e-1f);
  const svfloat32_t ln2lo = svdup_f32(-2.12194440e-4f);
  const svfloat32_t p0 = svdup_f32(1.9875691500e-4f);
  const svfloat32_t p1 = svdup_f32(1.3981999507e-3f);
  const svfloat32_t p2 = svdup_f32(8.3334519073e-3f);
  const svfloat32_t p3 = svdup_f32(4.1665795894e-2f);
  const svfloat32_t p4 = svdup_f32(1.6666665459e-1f);
  const svfloat32_t p5 = svdup_f32(5.0000001201e-1f);
  const svfloat32_t one = svdup_f32(1.0f);

  /* Save original for underflow/overflow masking */
  const svfloat32_t exp_lo = svdup_f32(-103.972076f);
  const svfloat32_t exp_hi = svdup_f32(88.3762626647949f);
  const svfloat32_t x_orig = x;

  /* Clamp */
  x = svmax_f32_x(ptrue, x, exp_lo);
  x = svmin_f32_x(ptrue, x, exp_hi);

  /* n = round(x * log2e) */
  svfloat32_t n = svrintn_f32_x(ptrue, svmul_f32_x(ptrue, x, log2e));

  /* r = x - n * ln2 */
  svfloat32_t r = svmls_f32_x(ptrue, x, n, ln2hi);
  r = svmls_f32_x(ptrue, r, n, ln2lo);

  /* Horner polynomial */
  svfloat32_t p = p0;
  p = svmla_f32_x(ptrue, p1, p, r);
  p = svmla_f32_x(ptrue, p2, p, r);
  p = svmla_f32_x(ptrue, p3, p, r);
  p = svmla_f32_x(ptrue, p4, p, r);
  p = svmla_f32_x(ptrue, p5, p, r);
  p = svmla_f32_x(ptrue, svadd_f32_x(ptrue, r, one), p,
                  svmul_f32_x(ptrue, r, r));

  /* Scale by 2^n */
  svint32_t ni = svcvt_s32_f32_x(ptrue, n);
  ni = svlsl_n_s32_x(ptrue, ni, 23);
  svint32_t result_i = svadd_s32_x(ptrue, svreinterpret_s32_f32(p), ni);

  svfloat32_t result = svreinterpret_f32_s32(result_i);

  /* Underflow: x < -103.972076 → 0, Overflow: x > 88.376 → +inf */
  svbool_t underflow = svcmplt_f32(ptrue, x_orig, exp_lo);
  svbool_t overflow = svcmpgt_f32(ptrue, x_orig, exp_hi);
  result = svsel_f32(underflow, svdup_f32(0.0f), result);
  result = svsel_f32(overflow, svdup_f32(1.0f / 0.0f), result);

  return result;
}

/* ===================================================================
   Vectorized log — float64 (VL-wide, fdlibm algorithm)
   =================================================================== */

static inline svfloat64_t _sve_log_f64(svfloat64_t x) {
  svbool_t ptrue = svptrue_b64();

  const svfloat64_t ln2 = svdup_f64(6.9314718055994530942e-01);
  const svfloat64_t lg1 = svdup_f64(6.6666666666666735130e-01);
  const svfloat64_t lg2 = svdup_f64(3.9999999999940941908e-01);
  const svfloat64_t lg3 = svdup_f64(2.8571428743662391490e-01);
  const svfloat64_t lg4 = svdup_f64(2.2221984321497839600e-01);
  const svfloat64_t lg5 = svdup_f64(1.8183572161618050120e-01);
  const svfloat64_t lg6 = svdup_f64(1.5313837699209373320e-01);
  const svfloat64_t lg7 = svdup_f64(1.4798198605116585910e-01);
  const svfloat64_t half = svdup_f64(0.5);
  const svfloat64_t one = svdup_f64(1.0);
  const svfloat64_t two = svdup_f64(2.0);
  const svfloat64_t sqrt2 = svdup_f64(1.4142135623730951);

  svuint64_t ix = svreinterpret_u64_f64(x);
  svuint64_t exp_bits = svlsr_n_u64_x(ptrue, ix, 52);
  exp_bits = svand_n_u64_x(ptrue, exp_bits, 0x7FF);

  /* k = exponent - 1023 */
  svint64_t k_i = svsub_n_s64_x(ptrue, svreinterpret_s64_u64(exp_bits), 1023);

  /* Normalize mantissa */
  svuint64_t mantissa =
      svorr_n_u64_x(ptrue, svand_n_u64_x(ptrue, ix, 0x000FFFFFFFFFFFFFULL),
                    0x3FF0000000000000ULL);
  svfloat64_t m = svreinterpret_f64_u64(mantissa);

  /* If m > sqrt(2), halve m and increment k */
  svbool_t mask = svcmpgt_f64(ptrue, m, sqrt2);
  m = svsel_f64(mask, svmul_f64_x(ptrue, m, half), m);
  k_i = svadd_s64_m(mask, k_i, svdup_s64(1));

  svfloat64_t k = svcvt_f64_s64_x(ptrue, k_i);

  svfloat64_t f = svsub_f64_x(ptrue, m, one);
  svfloat64_t s = svdiv_f64_x(ptrue, f, svadd_f64_x(ptrue, two, f));
  svfloat64_t z = svmul_f64_x(ptrue, s, s);
  svfloat64_t w = svmul_f64_x(ptrue, z, z);

  /* t1 = w*(lg2 + w*(lg4 + w*lg6)) */
  svfloat64_t t1 = svmla_f64_x(ptrue, lg4, w, lg6);
  t1 = svmla_f64_x(ptrue, lg2, t1, w);
  t1 = svmul_f64_x(ptrue, w, t1);

  /* t2 = z*(lg1 + w*(lg3 + w*(lg5 + w*lg7))) */
  svfloat64_t t2 = svmla_f64_x(ptrue, lg5, w, lg7);
  t2 = svmla_f64_x(ptrue, lg3, t2, w);
  t2 = svmla_f64_x(ptrue, lg1, t2, w);
  t2 = svmul_f64_x(ptrue, z, t2);

  svfloat64_t r = svadd_f64_x(ptrue, t1, t2);

  svfloat64_t hfsq = svmul_f64_x(ptrue, half, svmul_f64_x(ptrue, f, f));
  svfloat64_t result = svmla_f64_x(
      ptrue,
      svadd_f64_x(ptrue, svsub_f64_x(ptrue, f, hfsq),
                  svmul_f64_x(ptrue, s, svadd_f64_x(ptrue, hfsq, r))),
      k, ln2);

  svbool_t pos_mask = svcmpgt_f64(ptrue, x, svdup_f64(0.0));
  result = svsel_f64(pos_mask, result, svdup_f64(0.0));

  return result;
}

/* ===================================================================
   Vectorized exp — float64 (VL-wide, Cephes algorithm)
   =================================================================== */

static inline svfloat64_t _sve_exp_f64(svfloat64_t x) {
  svbool_t ptrue = svptrue_b64();

  const svfloat64_t log2e = svdup_f64(1.44269504088896338700e+00);
  const svfloat64_t ln2hi = svdup_f64(6.93147180369123816490e-01);
  const svfloat64_t ln2lo = svdup_f64(1.90821492927058770002e-10);
  const svfloat64_t c2 = svdup_f64(5.00000000000000000000e-01);
  const svfloat64_t c3 = svdup_f64(1.66666666666666666667e-01);
  const svfloat64_t c4 = svdup_f64(4.16666666666666666667e-02);
  const svfloat64_t c5 = svdup_f64(8.33333333333333333333e-03);
  const svfloat64_t c6 = svdup_f64(1.38888888888888888889e-03);
  const svfloat64_t c7 = svdup_f64(1.98412698412698412698e-04);
  const svfloat64_t c8 = svdup_f64(2.48015873015873015873e-05);
  const svfloat64_t c9 = svdup_f64(2.75573192239858906526e-06);
  const svfloat64_t c10 = svdup_f64(2.75573192239858906526e-07);
  const svfloat64_t c11 = svdup_f64(2.50521083854417187751e-08);
  const svfloat64_t c12 = svdup_f64(2.08767569878680989792e-09);
  const svfloat64_t one = svdup_f64(1.0);

  /* Save original for underflow/overflow masking */
  const svfloat64_t exp_lo = svdup_f64(-745.133219101941217);
  const svfloat64_t exp_hi = svdup_f64(709.782712893383996843);
  const svfloat64_t x_orig = x;

  /* Clamp */
  x = svmax_f64_x(ptrue, x, exp_lo);
  x = svmin_f64_x(ptrue, x, exp_hi);

  /* n = round(x * log2e) */
  svfloat64_t n = svrintn_f64_x(ptrue, svmul_f64_x(ptrue, x, log2e));

  /* r = x - n * ln2 */
  svfloat64_t r = svmls_f64_x(ptrue, x, n, ln2hi);
  r = svmls_f64_x(ptrue, r, n, ln2lo);

  /* Horner polynomial */
  svfloat64_t p = c12;
  p = svmla_f64_x(ptrue, c11, p, r);
  p = svmla_f64_x(ptrue, c10, p, r);
  p = svmla_f64_x(ptrue, c9, p, r);
  p = svmla_f64_x(ptrue, c8, p, r);
  p = svmla_f64_x(ptrue, c7, p, r);
  p = svmla_f64_x(ptrue, c6, p, r);
  p = svmla_f64_x(ptrue, c5, p, r);
  p = svmla_f64_x(ptrue, c4, p, r);
  p = svmla_f64_x(ptrue, c3, p, r);
  p = svmla_f64_x(ptrue, c2, p, r);
  p = svmla_f64_x(ptrue, svadd_f64_x(ptrue, r, one), p,
                  svmul_f64_x(ptrue, r, r));

  /* Scale by 2^n */
  svint64_t ni = svcvt_s64_f64_x(ptrue, n);
  ni = svlsl_n_s64_x(ptrue, ni, 52);
  svint64_t result_i = svadd_s64_x(ptrue, svreinterpret_s64_f64(p), ni);

  svfloat64_t result = svreinterpret_f64_s64(result_i);

  /* Underflow: x < -745.13 → 0, Overflow: x > 709.78 → +inf */
  svbool_t underflow = svcmplt_f64(ptrue, x_orig, exp_lo);
  svbool_t overflow = svcmpgt_f64(ptrue, x_orig, exp_hi);
  result = svsel_f64(underflow, svdup_f64(0.0), result);
  result = svsel_f64(overflow, svdup_f64(1.0 / 0.0), result);

  return result;
}

/* ===================================================================
   Vectorized sin — float32 (VL-wide, Cephes algorithm)
   =================================================================== */

static inline svfloat32_t _sve_sin_f32(svfloat32_t x) {
  svbool_t ptrue = svptrue_b32();

  const svfloat32_t minus_cephes_DP1 = svdup_f32(-0.78515625f);
  const svfloat32_t minus_cephes_DP2 = svdup_f32(-2.4187564849853515625e-4f);
  const svfloat32_t minus_cephes_DP3 = svdup_f32(-3.77489497744594108e-8f);
  const svfloat32_t sincof_p0 = svdup_f32(-1.9515295891e-4f);
  const svfloat32_t sincof_p1 = svdup_f32(8.3321608736e-3f);
  const svfloat32_t sincof_p2 = svdup_f32(-1.6666654611e-1f);
  const svfloat32_t coscof_p0 = svdup_f32(2.443315711809948e-5f);
  const svfloat32_t coscof_p1 = svdup_f32(-1.388731625493765e-3f);
  const svfloat32_t coscof_p2 = svdup_f32(4.166664568298827e-2f);
  const svfloat32_t fopi = svdup_f32(1.27323954473516f);
  const svfloat32_t half = svdup_f32(0.5f);
  const svfloat32_t one = svdup_f32(1.0f);

  /* Extract sign and work with |x| */
  svuint32_t sign_bit =
      svand_n_u32_x(ptrue, svreinterpret_u32_f32(x), 0x80000000u);
  svfloat32_t xa = svabs_f32_x(ptrue, x);

  /* j = (int)(|x| * 4/pi), round up to even */
  svfloat32_t y = svmul_f32_x(ptrue, xa, fopi);
  svint32_t j = svcvt_s32_f32_x(ptrue, y);
  j = svadd_s32_x(ptrue, j, svand_n_s32_x(ptrue, j, 1));
  y = svcvt_f32_s32_x(ptrue, j);

  /* j&4 -> sign swap */
  svuint32_t swap_sign = svlsl_n_u32_x(
      ptrue, svreinterpret_u32_s32(svand_n_s32_x(ptrue, j, 4)), 29);
  sign_bit = sveor_u32_x(ptrue, sign_bit, swap_sign);

  /* j&2 -> use cosine polynomial */
  svbool_t poly_mask =
      svcmpeq_s32(ptrue, svand_n_s32_x(ptrue, j, 2), svdup_s32(2));

  /* Range reduction */
  xa = svmla_f32_x(ptrue, xa, y, minus_cephes_DP1);
  xa = svmla_f32_x(ptrue, xa, y, minus_cephes_DP2);
  xa = svmla_f32_x(ptrue, xa, y, minus_cephes_DP3);

  svfloat32_t z = svmul_f32_x(ptrue, xa, xa);

  /* cos polynomial */
  svfloat32_t yc = coscof_p0;
  yc = svmla_f32_x(ptrue, coscof_p1, yc, z);
  yc = svmla_f32_x(ptrue, coscof_p2, yc, z);
  yc = svmul_f32_x(ptrue, yc, svmul_f32_x(ptrue, z, z));
  yc = svmls_f32_x(ptrue, yc, half, z);
  yc = svadd_f32_x(ptrue, yc, one);

  /* sin polynomial */
  svfloat32_t ys = sincof_p0;
  ys = svmla_f32_x(ptrue, sincof_p1, ys, z);
  ys = svmla_f32_x(ptrue, sincof_p2, ys, z);
  ys = svmul_f32_x(ptrue, ys, svmul_f32_x(ptrue, z, xa));
  ys = svadd_f32_x(ptrue, ys, xa);

  /* Select and apply sign */
  y = svsel_f32(poly_mask, yc, ys);
  y = svreinterpret_f32_u32(
      sveor_u32_x(ptrue, svreinterpret_u32_f32(y), sign_bit));
  return y;
}

/* ===================================================================
   Vectorized cos — float32 (VL-wide, Cephes algorithm)
   =================================================================== */

static inline svfloat32_t _sve_cos_f32(svfloat32_t x) {
  svbool_t ptrue = svptrue_b32();

  const svfloat32_t minus_cephes_DP1 = svdup_f32(-0.78515625f);
  const svfloat32_t minus_cephes_DP2 = svdup_f32(-2.4187564849853515625e-4f);
  const svfloat32_t minus_cephes_DP3 = svdup_f32(-3.77489497744594108e-8f);
  const svfloat32_t sincof_p0 = svdup_f32(-1.9515295891e-4f);
  const svfloat32_t sincof_p1 = svdup_f32(8.3321608736e-3f);
  const svfloat32_t sincof_p2 = svdup_f32(-1.6666654611e-1f);
  const svfloat32_t coscof_p0 = svdup_f32(2.443315711809948e-5f);
  const svfloat32_t coscof_p1 = svdup_f32(-1.388731625493765e-3f);
  const svfloat32_t coscof_p2 = svdup_f32(4.166664568298827e-2f);
  const svfloat32_t fopi = svdup_f32(1.27323954473516f);
  const svfloat32_t half = svdup_f32(0.5f);
  const svfloat32_t one = svdup_f32(1.0f);

  svfloat32_t xa = svabs_f32_x(ptrue, x);

  svfloat32_t y = svmul_f32_x(ptrue, xa, fopi);
  svint32_t j = svcvt_s32_f32_x(ptrue, y);
  j = svadd_s32_x(ptrue, j, svand_n_s32_x(ptrue, j, 1));
  y = svcvt_f32_s32_x(ptrue, j);

  /* cos: shift j by 2 */
  j = svsub_n_s32_x(ptrue, j, 2);

  /* sign: (j^4)&4 */
  svuint32_t sign_bit =
      svlsl_n_u32_x(ptrue,
                    svreinterpret_u32_s32(
                        svand_n_s32_x(ptrue, sveor_n_s32_x(ptrue, j, 4), 4)),
                    29);

  svbool_t poly_mask =
      svcmpeq_s32(ptrue, svand_n_s32_x(ptrue, j, 2), svdup_s32(2));

  xa = svmla_f32_x(ptrue, xa, y, minus_cephes_DP1);
  xa = svmla_f32_x(ptrue, xa, y, minus_cephes_DP2);
  xa = svmla_f32_x(ptrue, xa, y, minus_cephes_DP3);

  svfloat32_t z = svmul_f32_x(ptrue, xa, xa);

  svfloat32_t yc = coscof_p0;
  yc = svmla_f32_x(ptrue, coscof_p1, yc, z);
  yc = svmla_f32_x(ptrue, coscof_p2, yc, z);
  yc = svmul_f32_x(ptrue, yc, svmul_f32_x(ptrue, z, z));
  yc = svmls_f32_x(ptrue, yc, half, z);
  yc = svadd_f32_x(ptrue, yc, one);

  svfloat32_t ys = sincof_p0;
  ys = svmla_f32_x(ptrue, sincof_p1, ys, z);
  ys = svmla_f32_x(ptrue, sincof_p2, ys, z);
  ys = svmul_f32_x(ptrue, ys, svmul_f32_x(ptrue, z, xa));
  ys = svadd_f32_x(ptrue, ys, xa);

  y = svsel_f32(poly_mask, yc, ys);
  y = svreinterpret_f32_u32(
      sveor_u32_x(ptrue, svreinterpret_u32_f32(y), sign_bit));
  return y;
}

/* ===================================================================
   Combined sincos — float32 (VL-wide)
   =================================================================== */

static inline void _sve_sincos_f32(svfloat32_t x, svfloat32_t *s,
                                   svfloat32_t *c) {
  svbool_t ptrue = svptrue_b32();

  const svfloat32_t minus_cephes_DP1 = svdup_f32(-0.78515625f);
  const svfloat32_t minus_cephes_DP2 = svdup_f32(-2.4187564849853515625e-4f);
  const svfloat32_t minus_cephes_DP3 = svdup_f32(-3.77489497744594108e-8f);
  const svfloat32_t sincof_p0 = svdup_f32(-1.9515295891e-4f);
  const svfloat32_t sincof_p1 = svdup_f32(8.3321608736e-3f);
  const svfloat32_t sincof_p2 = svdup_f32(-1.6666654611e-1f);
  const svfloat32_t coscof_p0 = svdup_f32(2.443315711809948e-5f);
  const svfloat32_t coscof_p1 = svdup_f32(-1.388731625493765e-3f);
  const svfloat32_t coscof_p2 = svdup_f32(4.166664568298827e-2f);
  const svfloat32_t fopi = svdup_f32(1.27323954473516f);
  const svfloat32_t half = svdup_f32(0.5f);
  const svfloat32_t one = svdup_f32(1.0f);

  svuint32_t sign_sin =
      svand_n_u32_x(ptrue, svreinterpret_u32_f32(x), 0x80000000u);
  svfloat32_t xa = svabs_f32_x(ptrue, x);

  svfloat32_t y = svmul_f32_x(ptrue, xa, fopi);
  svint32_t j = svcvt_s32_f32_x(ptrue, y);
  j = svadd_s32_x(ptrue, j, svand_n_s32_x(ptrue, j, 1));
  y = svcvt_f32_s32_x(ptrue, j);

  svuint32_t j4 = svreinterpret_u32_s32(svand_n_s32_x(ptrue, j, 4));
  sign_sin = sveor_u32_x(ptrue, sign_sin, svlsl_n_u32_x(ptrue, j4, 29));

  /* cos sign */
  svuint32_t sign_cos = svlsl_n_u32_x(
      ptrue,
      svreinterpret_u32_s32(svand_n_s32_x(
          ptrue, sveor_n_s32_x(ptrue, svsub_n_s32_x(ptrue, j, 2), 4), 4)),
      29);

  svbool_t poly_mask =
      svcmpeq_s32(ptrue, svand_n_s32_x(ptrue, j, 2), svdup_s32(2));

  xa = svmla_f32_x(ptrue, xa, y, minus_cephes_DP1);
  xa = svmla_f32_x(ptrue, xa, y, minus_cephes_DP2);
  xa = svmla_f32_x(ptrue, xa, y, minus_cephes_DP3);

  svfloat32_t z = svmul_f32_x(ptrue, xa, xa);

  svfloat32_t yc = coscof_p0;
  yc = svmla_f32_x(ptrue, coscof_p1, yc, z);
  yc = svmla_f32_x(ptrue, coscof_p2, yc, z);
  yc = svmul_f32_x(ptrue, yc, svmul_f32_x(ptrue, z, z));
  yc = svmls_f32_x(ptrue, yc, half, z);
  yc = svadd_f32_x(ptrue, yc, one);

  svfloat32_t yz = sincof_p0;
  yz = svmla_f32_x(ptrue, sincof_p1, yz, z);
  yz = svmla_f32_x(ptrue, sincof_p2, yz, z);
  yz = svmul_f32_x(ptrue, yz, svmul_f32_x(ptrue, z, xa));
  yz = svadd_f32_x(ptrue, yz, xa);

  /* sin */
  *s = svreinterpret_f32_u32(sveor_u32_x(
      ptrue, svreinterpret_u32_f32(svsel_f32(poly_mask, yc, yz)), sign_sin));

  /* cos: opposite selection */
  svbool_t cos_poly_mask = svnot_b_z(ptrue, poly_mask);
  *c = svreinterpret_f32_u32(sveor_u32_x(
      ptrue, svreinterpret_u32_f32(svsel_f32(cos_poly_mask, yc, yz)),
      sign_cos));
}

/* ========================================================================
   Vectorize tanh - float32/float64 (VLA)
   Uses identify: tanh(x) = 2 / (1 + exp(-2x)) -1
   ======================================================================*/

static inline svfloat32_t _sve_tanh_f32(svfloat32_t x) {
  const svbool_t pg = svptrue_b32();
  const svfloat32_t vtwo = svdup_f32(2.0f);
  const svfloat32_t vone = svdup_f32(1.0f);
  const svfloat32_t vneg2 = svdup_f32(-2.0f);

  svfloat32_t e = _sve_exp_f32(svmul_f32_x(pg, vneg2, x));
  return svsub_f32_x(pg, svdiv_f32_x(pg, vtwo, svadd_f32_x(pg, vone, e)), vone);
}


static inline svfloat64_t _sve_tanh_f64(svfloat64_t x) {
  const svbool_t pg = svptrue_b64();
  const svfloat64_t vtwo = svdup_f64(2.0f);
  const svfloat64_t vone = svdup_f64(1.0f);
  const svfloat64_t vneg2 = svdup_f64(-2.0f);

  svfloat64_t e = _sve_exp_f64(svmul_f64_x(pg, vneg2, x));
  return svsub_f64_x(pg, svdiv_f64_x(pg, vtwo, svadd_f64_x(pg, vone, e)), vone);
}

static inline void _fast_tanh_f32_sve(const void *restrict ap,
                                      void *restrict op, size_t n) {
  const float *a = (const float *)ap;
  float *out = (float *)op;
  size_t vl = svcntw();
  for (size_t i = 0; i < n; i += vl) {
    svbool_t pg = svwhilelt_b32((uint32_t)i, (uint32_t)n);
    svfloat32_t va = svld1_f32(pg, a + i);
    svst1_f32(pg, out + i, _sve_tanh_f32(va));
  }
}

static inline void _fast_tanh_f64_sve(const void *restrict ap,
                                      void *restrict op, size_t n) {
  const double *a = (const double *)ap;
  double *out = (double *)op;
  size_t vl = svcntd();
  for (size_t i = 0; i < n; i += vl) {
    svbool_t pg = svwhilelt_b64((uint64_t)i, (uint64_t)n);
    svfloat64_t va = svld1_f64(pg, a + i);
    svst1_f64(pg, out + i, _sve_tanh_f64(va));
  }
}

static inline svfloat32_t _sve_sigmoid_f32(svfloat32_t x, svbool_t pg){
  const svfloat32_t zero = svdup_f32(0.0f);
  const svfloat32_t one = svdup_f32(1.0f);

  svbool pos = svcmpge_f32(pg, x, zero);
  svfloat32_t z_pos = _sve_exp_f32(svneg_f32_m(pg, x), pg);
  svfloat32_t y_pos = _svdiv_f32_m(pg, one, svadd_f32_m(pg, one, z_pos));
  svfloat32_t z_neg = _sve_exp_f32(x, pg);
  svfloat32_t y_neg = svdiv_f32_m(pg, z_neg, svadd_f32_m(pg, one, z_neg));

  return svsel_f32(pos, y_pos, y_neg);
}

static inline svfloat64_t _sve_sigmoid_f64(svfloat64_t x, svbool_t pg){
  const svfloat64_t zero = svdup_f64(0.0);
  const svfloat64_t one = svdup_f64(1.0);

  svbool_t pos = svcmpge_f64(pg, x, zero);
  svfloat64_t z_pos = _sve_exp_f64(svneg_f64(pg, x), pg);
  svfloat64_t y_pos = svdiv_f64_m(pg, one, svadd_f64_m(pg, one, z_pos));
  svfloat64_t z_neg = _sve_exp_f64(x, pg);
  svfloat64_t y_neg = svdiv_f64_m(pg, z_neg, svadd_f64_m(pg, one, z_neg));

  return svsel_f64(pos, y_pos, y_neg);
}

static inline void _sigmoid_f32_sve(const float x, float *out){
  float z = 0.0f;
  if(x >= 0.0f){
    z = _exp_f32(-x);
    *out = 1.0f / (1.0f + z);
  } else {
    z = _exp_f32(x);
    *out = z / (1.0f + z);
  }
}

static inline void _sigmoid_f64_sve(const double x, double *out) {
  double z = 0.0f;
  if(x >= 0.0){
    z = _exp_f64(-x);
    *out = 1.0 / (1.0 + z);
  }else {
    z = _exp_f64(x);
    *out = z / (1.0 + z);
  }
}

static inline void _fast_sigmoid_f32_sve(const void *restrict ap,
                                        void *restrict op, size_t n) {
  const float *a = (const float *)ap;
  float *out = (float *)op;

  size_t i = 0;
  const size_t vl = svcntw();
  for(; i + vl <= n; i += vl){
    svbool_t pg = svptrue_b32();
    svst1_f32(pg, out + i, _sve_sigmoid_f32(svld1_f32(pg, a + i), pg));
  }

  for(; i < n; ++i)
    _sigmoid_f32_sve(a[i], out + i);
}

static inline void _fast_sigmoid_f64_sve(const void *restrict ap,
                                        void *restrict op, size_t n){
  const double *a = (const double *)ap;
  double *out = (double *)op;

  size_t i = 0;
  const size_t vl = svcntd();
  for(; i + vl <= n; i += vl){
    svbool_t pg = svtrue_b64();
    svst1_f64(pg, out + i, _sve_sigmoid_f64(svld1_f64(pg, a + i), pg)):
  }

  for(; i < n; ++i)
    _sigmoid_f64_sve(a[i], out + i);
}

#endif /* NUMC_MATH_SVE_H */
