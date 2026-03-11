/**
 * @file math_avx2.h
 * @brief Vectorized log/exp intrinsics for AVX2+FMA.
 *
 * Implements the same fdlibm/Cephes algorithms as the scalar helpers
 * in helpers.h, using AVX2 SIMD intrinsics for 8-wide float32 and
 * 4-wide float64 processing.
 */
#ifndef NUMC_MATH_AVX2_H
#define NUMC_MATH_AVX2_H

#include <immintrin.h>

/* ===================================================================
   Vectorized log — float32 (fdlibm algorithm, same as _log_f32)
   =================================================================== */

static inline __m256 _mm256_log_ps(__m256 x) {
  /* Constants (same as _log_f32 in helpers.h) */
  const __m256 ln2 = _mm256_set1_ps(6.9314718056e-01f);
  const __m256 lg1 = _mm256_set1_ps(6.6666668653e-01f);
  const __m256 lg2 = _mm256_set1_ps(4.0000004172e-01f);
  const __m256 lg3 = _mm256_set1_ps(2.8571429849e-01f);
  const __m256 lg4 = _mm256_set1_ps(2.2222198009e-01f);
  const __m256 half = _mm256_set1_ps(0.5f);
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 two = _mm256_set1_ps(2.0f);
  const __m256 sqrt2 = _mm256_set1_ps(1.41421356f);

  /* Step 1: Argument reduction — extract exponent and normalize */
  __m256i ix = _mm256_castps_si256(x);

  /* k = exponent - 127 */
  __m256i exp_bits = _mm256_srli_epi32(ix, 23);
  exp_bits = _mm256_and_si256(exp_bits, _mm256_set1_epi32(0xFF));
  __m256i k_i = _mm256_sub_epi32(exp_bits, _mm256_set1_epi32(127));

  /* m = mantissa with exponent set to 0 (biased 127) */
  __m256i mantissa =
      _mm256_or_si256(_mm256_and_si256(ix, _mm256_set1_epi32(0x007FFFFF)),
                      _mm256_set1_epi32(0x3F800000));
  __m256 m = _mm256_castsi256_ps(mantissa);

  /* If m > sqrt(2), halve m and increment k */
  __m256 mask = _mm256_cmp_ps(m, sqrt2, _CMP_GT_OS);
  m = _mm256_blendv_ps(m, _mm256_mul_ps(m, half), mask);
  k_i = _mm256_add_epi32(
      k_i, _mm256_and_si256(_mm256_castps_si256(mask), _mm256_set1_epi32(1)));

  __m256 k = _mm256_cvtepi32_ps(k_i);

  /* Step 2: f = m - 1 */
  __m256 f = _mm256_sub_ps(m, one);

  /* Step 3: s = f / (2 + f) */
  __m256 s = _mm256_div_ps(f, _mm256_add_ps(two, f));
  __m256 z = _mm256_mul_ps(s, s);
  __m256 w = _mm256_mul_ps(z, z);

  /* Step 4: Horner polynomial — t1 = w*(lg2 + w*lg4) */
  __m256 t1 = _mm256_fmadd_ps(w, lg4, lg2);
  t1 = _mm256_mul_ps(w, t1);
  /* t2 = z*(lg1 + w*lg3) */
  __m256 t2 = _mm256_fmadd_ps(w, lg3, lg1);
  t2 = _mm256_mul_ps(z, t2);
  __m256 r = _mm256_add_ps(t1, t2);

  /* Step 5: result = k*ln2 + f - hfsq + s*(hfsq + r) */
  __m256 hfsq = _mm256_mul_ps(half, _mm256_mul_ps(f, f));
  __m256 result =
      _mm256_fmadd_ps(k, ln2,
                      _mm256_add_ps(_mm256_sub_ps(f, hfsq),
                                    _mm256_mul_ps(s, _mm256_add_ps(hfsq, r))));

  /* Handle x <= 0: return 0 (matching scalar behavior) */
  __m256 zero = _mm256_setzero_ps();
  __m256 pos_mask = _mm256_cmp_ps(x, zero, _CMP_GT_OS);
  result = _mm256_and_ps(result, pos_mask);

  return result;
}

/* ===================================================================
   Vectorized exp — float32 (Cephes algorithm, same as _exp_f32)
   =================================================================== */

static inline __m256 _mm256_exp_ps(__m256 x) {
  /* Constants (same as _exp_f32 in helpers.h) */
  const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
  const __m256 ln2hi = _mm256_set1_ps(6.93359375000000000e-1f);
  const __m256 ln2lo = _mm256_set1_ps(-2.12194440e-4f);
  const __m256 p0 = _mm256_set1_ps(1.9875691500e-4f);
  const __m256 p1 = _mm256_set1_ps(1.3981999507e-3f);
  const __m256 p2 = _mm256_set1_ps(8.3334519073e-3f);
  const __m256 p3 = _mm256_set1_ps(4.1665795894e-2f);
  const __m256 p4 = _mm256_set1_ps(1.6666665459e-1f);
  const __m256 p5 = _mm256_set1_ps(5.0000001201e-1f);
  const __m256 one = _mm256_set1_ps(1.0f);

  /* Clamp to avoid overflow/underflow */
  const __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
  const __m256 exp_lo = _mm256_set1_ps(-103.972076f);
  x = _mm256_max_ps(x, exp_lo);
  x = _mm256_min_ps(x, exp_hi);

  /* Step 1: n = round(x * log2e) */
  __m256 n = _mm256_round_ps(_mm256_mul_ps(x, log2e),
                             _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

  /* Step 2: r = x - n * ln2 (compensated) */
  __m256 r = _mm256_fnmadd_ps(n, ln2hi, x);
  r = _mm256_fnmadd_ps(n, ln2lo, r);

  /* Step 3: Horner polynomial */
  __m256 p = p0;
  p = _mm256_fmadd_ps(p, r, p1);
  p = _mm256_fmadd_ps(p, r, p2);
  p = _mm256_fmadd_ps(p, r, p3);
  p = _mm256_fmadd_ps(p, r, p4);
  p = _mm256_fmadd_ps(p, r, p5);
  p = _mm256_fmadd_ps(p, _mm256_mul_ps(r, r), _mm256_add_ps(r, one));

  /* Step 4: scale by 2^n via IEEE 754 exponent field */
  __m256i ni = _mm256_cvtps_epi32(n);
  ni = _mm256_slli_epi32(ni, 23);
  __m256i result_i = _mm256_add_epi32(_mm256_castps_si256(p), ni);

  return _mm256_castsi256_ps(result_i);
}

/* ===================================================================
   Vectorized log — float64 (fdlibm algorithm, same as _log_f64)
   =================================================================== */

static inline __m256d _mm256_log_pd(__m256d x) {
  const __m256d ln2 = _mm256_set1_pd(6.9314718055994530942e-01);
  const __m256d lg1 = _mm256_set1_pd(6.6666666666666735130e-01);
  const __m256d lg2 = _mm256_set1_pd(3.9999999999940941908e-01);
  const __m256d lg3 = _mm256_set1_pd(2.8571428743662391490e-01);
  const __m256d lg4 = _mm256_set1_pd(2.2221984321497839600e-01);
  const __m256d lg5 = _mm256_set1_pd(1.8183572161618050120e-01);
  const __m256d lg6 = _mm256_set1_pd(1.5313837699209373320e-01);
  const __m256d lg7 = _mm256_set1_pd(1.4798198605116585910e-01);
  const __m256d half = _mm256_set1_pd(0.5);
  const __m256d one = _mm256_set1_pd(1.0);
  const __m256d two = _mm256_set1_pd(2.0);
  const __m256d sqrt2 = _mm256_set1_pd(1.4142135623730951);

  /* Extract exponent */
  __m256i ix = _mm256_castpd_si256(x);
  /* For f64, exponent is bits 52-62 */
  __m256i exp_bits = _mm256_srli_epi64(ix, 52);
  __m256i exp_mask = _mm256_set1_epi64x(0x7FF);
  exp_bits = _mm256_and_si256(exp_bits, exp_mask);

  /* k = exponent - 1023. AVX2 lacks epi64 cvt, use scalar. */
  __m256i bias = _mm256_set1_epi64x(1023);
  __m256i k_64 = _mm256_sub_epi64(exp_bits, bias);

  /* Convert k to double via store/reload (no _mm256_cvtepi64_pd) */
  int64_t k_arr[4];
  _mm256_storeu_si256((__m256i *)k_arr, k_64);
  __m256d k = _mm256_set_pd((double)k_arr[3], (double)k_arr[2],
                            (double)k_arr[1], (double)k_arr[0]);

  /* Normalize mantissa: clear exponent, set biased exponent to 0 */
  __m256i mant_mask = _mm256_set1_epi64x(0x000FFFFFFFFFFFFFLL);
  __m256i mant_bias = _mm256_set1_epi64x(0x3FF0000000000000LL);
  __m256i mantissa =
      _mm256_or_si256(_mm256_and_si256(ix, mant_mask), mant_bias);
  __m256d m = _mm256_castsi256_pd(mantissa);

  /* If m > sqrt(2), halve m and increment k */
  __m256d mask = _mm256_cmp_pd(m, sqrt2, _CMP_GT_OS);
  m = _mm256_blendv_pd(m, _mm256_mul_pd(m, half), mask);
  k = _mm256_add_pd(k, _mm256_and_pd(mask, one));

  /* f = m - 1 */
  __m256d f = _mm256_sub_pd(m, one);

  /* s = f / (2 + f) */
  __m256d s = _mm256_div_pd(f, _mm256_add_pd(two, f));
  __m256d z = _mm256_mul_pd(s, s);
  __m256d w = _mm256_mul_pd(z, z);

  /* Horner polynomial (7 coefficients, interleaved) */
  /* t1 = w*(lg2 + w*(lg4 + w*lg6)) */
  __m256d t1 = _mm256_fmadd_pd(w, lg6, lg4);
  t1 = _mm256_fmadd_pd(t1, w, lg2);
  t1 = _mm256_mul_pd(w, t1);

  /* t2 = z*(lg1 + w*(lg3 + w*(lg5 + w*lg7))) */
  __m256d t2 = _mm256_fmadd_pd(w, lg7, lg5);
  t2 = _mm256_fmadd_pd(t2, w, lg3);
  t2 = _mm256_fmadd_pd(t2, w, lg1);
  t2 = _mm256_mul_pd(z, t2);

  __m256d r = _mm256_add_pd(t1, t2);

  __m256d hfsq = _mm256_mul_pd(half, _mm256_mul_pd(f, f));
  __m256d result =
      _mm256_fmadd_pd(k, ln2,
                      _mm256_add_pd(_mm256_sub_pd(f, hfsq),
                                    _mm256_mul_pd(s, _mm256_add_pd(hfsq, r))));

  /* Handle x <= 0: return 0 */
  __m256d zero = _mm256_setzero_pd();
  __m256d pos_mask = _mm256_cmp_pd(x, zero, _CMP_GT_OS);
  result = _mm256_and_pd(result, pos_mask);

  return result;
}

/* ===================================================================
   Vectorized exp — float64 (Cephes algorithm, same as _exp_f64)
   =================================================================== */

static inline __m256d _mm256_exp_pd(__m256d x) {
  const __m256d log2e = _mm256_set1_pd(1.44269504088896338700e+00);
  const __m256d ln2hi = _mm256_set1_pd(6.93147180369123816490e-01);
  const __m256d ln2lo = _mm256_set1_pd(1.90821492927058770002e-10);
  const __m256d c2 = _mm256_set1_pd(5.00000000000000000000e-01);
  const __m256d c3 = _mm256_set1_pd(1.66666666666666666667e-01);
  const __m256d c4 = _mm256_set1_pd(4.16666666666666666667e-02);
  const __m256d c5 = _mm256_set1_pd(8.33333333333333333333e-03);
  const __m256d c6 = _mm256_set1_pd(1.38888888888888888889e-03);
  const __m256d c7 = _mm256_set1_pd(1.98412698412698412698e-04);
  const __m256d c8 = _mm256_set1_pd(2.48015873015873015873e-05);
  const __m256d c9 = _mm256_set1_pd(2.75573192239858906526e-06);
  const __m256d c10 = _mm256_set1_pd(2.75573192239858906526e-07);
  const __m256d c11 = _mm256_set1_pd(2.50521083854417187751e-08);
  const __m256d c12 = _mm256_set1_pd(2.08767569878680989792e-09);
  const __m256d one = _mm256_set1_pd(1.0);

  /* Clamp */
  x = _mm256_max_pd(x, _mm256_set1_pd(-745.133219101941217));
  x = _mm256_min_pd(x, _mm256_set1_pd(709.782712893383996843));

  /* n = round(x * log2e) */
  __m256d n = _mm256_round_pd(_mm256_mul_pd(x, log2e),
                              _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

  /* r = x - n * ln2 (compensated) */
  __m256d r = _mm256_fnmadd_pd(n, ln2hi, x);
  r = _mm256_fnmadd_pd(n, ln2lo, r);

  /* Horner polynomial (c12..c2) */
  __m256d p = c12;
  p = _mm256_fmadd_pd(p, r, c11);
  p = _mm256_fmadd_pd(p, r, c10);
  p = _mm256_fmadd_pd(p, r, c9);
  p = _mm256_fmadd_pd(p, r, c8);
  p = _mm256_fmadd_pd(p, r, c7);
  p = _mm256_fmadd_pd(p, r, c6);
  p = _mm256_fmadd_pd(p, r, c5);
  p = _mm256_fmadd_pd(p, r, c4);
  p = _mm256_fmadd_pd(p, r, c3);
  p = _mm256_fmadd_pd(p, r, c2);
  p = _mm256_fmadd_pd(p, _mm256_mul_pd(r, r), _mm256_add_pd(r, one));

  /* Scale by 2^n via IEEE 754 exponent field (bits 52-62).
   * AVX2 lacks cvtpd_epi64, use scalar extraction. */
  double n_dbl[4];
  _mm256_storeu_pd(n_dbl, n);
  int64_t n_arr[4];
  n_arr[0] = (int64_t)n_dbl[0];
  n_arr[1] = (int64_t)n_dbl[1];
  n_arr[2] = (int64_t)n_dbl[2];
  n_arr[3] = (int64_t)n_dbl[3];
  __m256i ni = _mm256_set_epi64x(n_arr[3], n_arr[2], n_arr[1], n_arr[0]);
  ni = _mm256_slli_epi64(ni, 52);
  __m256i result_i = _mm256_add_epi64(_mm256_castpd_si256(p), ni);

  return _mm256_castsi256_pd(result_i);
}

/* ===================================================================
   Vectorized sin/cos — float32 (Cephes algorithm)

   Range reduction to [-π/4, π/4], then polynomial approximation.
   Used by SIMD Box-Muller randn kernel.
   =================================================================== */

static inline __m256 _mm256_sin_ps(__m256 x) {
  /* Cephes constants */
  const __m256 minus_cephes_DP1 = _mm256_set1_ps(-0.78515625f);
  const __m256 minus_cephes_DP2 = _mm256_set1_ps(-2.4187564849853515625e-4f);
  const __m256 minus_cephes_DP3 = _mm256_set1_ps(-3.77489497744594108e-8f);
  const __m256 sincof_p0 = _mm256_set1_ps(-1.9515295891e-4f);
  const __m256 sincof_p1 = _mm256_set1_ps(8.3321608736e-3f);
  const __m256 sincof_p2 = _mm256_set1_ps(-1.6666654611e-1f);
  const __m256 coscof_p0 = _mm256_set1_ps(2.443315711809948e-5f);
  const __m256 coscof_p1 = _mm256_set1_ps(-1.388731625493765e-3f);
  const __m256 coscof_p2 = _mm256_set1_ps(4.166664568298827e-2f);
  const __m256 fopi = _mm256_set1_ps(1.27323954473516f); /* 4/π */
  const __m256 half = _mm256_set1_ps(0.5f);
  const __m256 one = _mm256_set1_ps(1.0f);

  /* Extract sign and work with |x| */
  __m256i sign_bit = _mm256_and_si256(_mm256_castps_si256(x),
                                      _mm256_set1_epi32((int)0x80000000));
  __m256 xa = _mm256_andnot_ps(
      _mm256_castsi256_ps(_mm256_set1_epi32((int)0x80000000)), x);

  /* j = (int)(|x| * 4/π) — which octant */
  __m256 y = _mm256_mul_ps(xa, fopi);
  __m256i j = _mm256_cvttps_epi32(y);
  /* if j is odd, j++ (round up to even) */
  j = _mm256_add_epi32(j, _mm256_and_si256(j, _mm256_set1_epi32(1)));
  y = _mm256_cvtepi32_ps(j);

  /* j&4 → sign swap for sin */
  __m256i swap_sign =
      _mm256_slli_epi32(_mm256_and_si256(j, _mm256_set1_epi32(4)), 29);
  sign_bit = _mm256_xor_si256(sign_bit, swap_sign);

  /* j&2 → use cosine polynomial instead */
  __m256 poly_mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(
      _mm256_and_si256(j, _mm256_set1_epi32(2)), _mm256_set1_epi32(2)));

  /* Extended-precision range reduction */
  xa = _mm256_fmadd_ps(y, minus_cephes_DP1, xa);
  xa = _mm256_fmadd_ps(y, minus_cephes_DP2, xa);
  xa = _mm256_fmadd_ps(y, minus_cephes_DP3, xa);

  __m256 z = _mm256_mul_ps(xa, xa);

  /* cos polynomial: 1 - z*0.5 + z*z*(p0 + z*(p1 + z*p2)) */
  __m256 yc = coscof_p0;
  yc = _mm256_fmadd_ps(yc, z, coscof_p1);
  yc = _mm256_fmadd_ps(yc, z, coscof_p2);
  yc = _mm256_mul_ps(yc, _mm256_mul_ps(z, z));
  yc = _mm256_fnmadd_ps(half, z, yc);
  yc = _mm256_add_ps(yc, one);

  /* sin polynomial: x + x*z*(p0*z*z + p1*z + p2) */
  __m256 ys = sincof_p0;
  ys = _mm256_fmadd_ps(ys, z, sincof_p1);
  ys = _mm256_fmadd_ps(ys, z, sincof_p2);
  ys = _mm256_mul_ps(ys, _mm256_mul_ps(z, xa));
  ys = _mm256_add_ps(ys, xa);

  /* Select sin or cos polynomial based on octant */
  y = _mm256_blendv_ps(ys, yc, poly_mask);

  /* Apply sign */
  y = _mm256_xor_ps(y, _mm256_castsi256_ps(sign_bit));
  return y;
}

static inline __m256 _mm256_cos_ps(__m256 x) {
  const __m256 minus_cephes_DP1 = _mm256_set1_ps(-0.78515625f);
  const __m256 minus_cephes_DP2 = _mm256_set1_ps(-2.4187564849853515625e-4f);
  const __m256 minus_cephes_DP3 = _mm256_set1_ps(-3.77489497744594108e-8f);
  const __m256 sincof_p0 = _mm256_set1_ps(-1.9515295891e-4f);
  const __m256 sincof_p1 = _mm256_set1_ps(8.3321608736e-3f);
  const __m256 sincof_p2 = _mm256_set1_ps(-1.6666654611e-1f);
  const __m256 coscof_p0 = _mm256_set1_ps(2.443315711809948e-5f);
  const __m256 coscof_p1 = _mm256_set1_ps(-1.388731625493765e-3f);
  const __m256 coscof_p2 = _mm256_set1_ps(4.166664568298827e-2f);
  const __m256 fopi = _mm256_set1_ps(1.27323954473516f);
  const __m256 half = _mm256_set1_ps(0.5f);
  const __m256 one = _mm256_set1_ps(1.0f);

  __m256 xa = _mm256_andnot_ps(
      _mm256_castsi256_ps(_mm256_set1_epi32((int)0x80000000)), x);

  __m256 y = _mm256_mul_ps(xa, fopi);
  __m256i j = _mm256_cvttps_epi32(y);
  j = _mm256_add_epi32(j, _mm256_and_si256(j, _mm256_set1_epi32(1)));
  y = _mm256_cvtepi32_ps(j);

  /* cos: shift j by 2 (cos(x) = sin(x + π/2)) */
  j = _mm256_sub_epi32(j, _mm256_set1_epi32(2));

  __m256i sign_bit = _mm256_slli_epi32(
      _mm256_andnot_si256(_mm256_set1_epi32(4),
                          _mm256_and_si256(j, _mm256_set1_epi32(4))),
      29);

  /* But actually for cos: negate when (j+2)&4 == 0? Let me redo.
   * For cos, the sign flip is: !(j&4) → negative, equivalently (j^4)&4 */
  sign_bit = _mm256_slli_epi32(
      _mm256_and_si256(_mm256_xor_si256(j, _mm256_set1_epi32(4)),
                       _mm256_set1_epi32(4)),
      29);

  __m256 poly_mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(
      _mm256_and_si256(j, _mm256_set1_epi32(2)), _mm256_set1_epi32(2)));

  xa = _mm256_fmadd_ps(y, minus_cephes_DP1, xa);
  xa = _mm256_fmadd_ps(y, minus_cephes_DP2, xa);
  xa = _mm256_fmadd_ps(y, minus_cephes_DP3, xa);

  __m256 z = _mm256_mul_ps(xa, xa);

  __m256 yc = coscof_p0;
  yc = _mm256_fmadd_ps(yc, z, coscof_p1);
  yc = _mm256_fmadd_ps(yc, z, coscof_p2);
  yc = _mm256_mul_ps(yc, _mm256_mul_ps(z, z));
  yc = _mm256_fnmadd_ps(half, z, yc);
  yc = _mm256_add_ps(yc, one);

  __m256 ys = sincof_p0;
  ys = _mm256_fmadd_ps(ys, z, sincof_p1);
  ys = _mm256_fmadd_ps(ys, z, sincof_p2);
  ys = _mm256_mul_ps(ys, _mm256_mul_ps(z, xa));
  ys = _mm256_add_ps(ys, xa);

  y = _mm256_blendv_ps(ys, yc, poly_mask);
  y = _mm256_xor_ps(y, _mm256_castsi256_ps(sign_bit));
  return y;
}

/* Combined sincos — computes sin and cos simultaneously, sharing
 * range reduction work. Returns sin in *s, cos in *c. */
static inline void _mm256_sincos_ps(__m256 x, __m256 *s, __m256 *c) {
  const __m256 minus_cephes_DP1 = _mm256_set1_ps(-0.78515625f);
  const __m256 minus_cephes_DP2 = _mm256_set1_ps(-2.4187564849853515625e-4f);
  const __m256 minus_cephes_DP3 = _mm256_set1_ps(-3.77489497744594108e-8f);
  const __m256 sincof_p0 = _mm256_set1_ps(-1.9515295891e-4f);
  const __m256 sincof_p1 = _mm256_set1_ps(8.3321608736e-3f);
  const __m256 sincof_p2 = _mm256_set1_ps(-1.6666654611e-1f);
  const __m256 coscof_p0 = _mm256_set1_ps(2.443315711809948e-5f);
  const __m256 coscof_p1 = _mm256_set1_ps(-1.388731625493765e-3f);
  const __m256 coscof_p2 = _mm256_set1_ps(4.166664568298827e-2f);
  const __m256 fopi = _mm256_set1_ps(1.27323954473516f);
  const __m256 half = _mm256_set1_ps(0.5f);
  const __m256 one = _mm256_set1_ps(1.0f);

  __m256i sign_sin = _mm256_and_si256(_mm256_castps_si256(x),
                                      _mm256_set1_epi32((int)0x80000000));
  __m256 xa = _mm256_andnot_ps(
      _mm256_castsi256_ps(_mm256_set1_epi32((int)0x80000000)), x);

  __m256 y = _mm256_mul_ps(xa, fopi);
  __m256i j = _mm256_cvttps_epi32(y);
  j = _mm256_add_epi32(j, _mm256_and_si256(j, _mm256_set1_epi32(1)));
  y = _mm256_cvtepi32_ps(j);

  __m256i j4 = _mm256_and_si256(j, _mm256_set1_epi32(4));
  sign_sin = _mm256_xor_si256(sign_sin, _mm256_slli_epi32(j4, 29));

  /* cos sign: negate when ((j-2)^4)&4 is set, i.e. j&4 */
  __m256i sign_cos = _mm256_slli_epi32(
      _mm256_and_si256(
          _mm256_xor_si256(_mm256_sub_epi32(j, _mm256_set1_epi32(2)),
                           _mm256_set1_epi32(4)),
          _mm256_set1_epi32(4)),
      29);

  __m256 poly_mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(
      _mm256_and_si256(j, _mm256_set1_epi32(2)), _mm256_set1_epi32(2)));

  xa = _mm256_fmadd_ps(y, minus_cephes_DP1, xa);
  xa = _mm256_fmadd_ps(y, minus_cephes_DP2, xa);
  xa = _mm256_fmadd_ps(y, minus_cephes_DP3, xa);

  __m256 z = _mm256_mul_ps(xa, xa);

  __m256 yc = coscof_p0;
  yc = _mm256_fmadd_ps(yc, z, coscof_p1);
  yc = _mm256_fmadd_ps(yc, z, coscof_p2);
  yc = _mm256_mul_ps(yc, _mm256_mul_ps(z, z));
  yc = _mm256_fnmadd_ps(half, z, yc);
  yc = _mm256_add_ps(yc, one);

  __m256 yz = sincof_p0;
  yz = _mm256_fmadd_ps(yz, z, sincof_p1);
  yz = _mm256_fmadd_ps(yz, z, sincof_p2);
  yz = _mm256_mul_ps(yz, _mm256_mul_ps(z, xa));
  yz = _mm256_add_ps(yz, xa);

  /* sin: select based on octant, then apply sign */
  *s = _mm256_xor_ps(_mm256_blendv_ps(yz, yc, poly_mask),
                     _mm256_castsi256_ps(sign_sin));

  /* cos: opposite polynomial selection from sin */
  __m256 cos_poly_mask = _mm256_xor_ps(
      poly_mask, _mm256_castsi256_ps(_mm256_set1_epi32((int)0xFFFFFFFF)));
  *c = _mm256_xor_ps(_mm256_blendv_ps(yz, yc, cos_poly_mask),
                     _mm256_castsi256_ps(sign_cos));
}

#endif /* NUMC_MATH_AVX2_H */
