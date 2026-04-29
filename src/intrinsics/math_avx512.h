/**
 * @file math_avx512.h
 * @brief Vectorized math intrinsics for AVX-512.
 *
 * Mirrors math_avx2.h with 16-wide float32 and 8-wide float64 processing.
 */
#ifndef NUMC_MATH_AVX512_H
#define NUMC_MATH_AVX512_H

#include "math_helpers.h"
#include <immintrin.h>
#include <math.h>
#include <stdint.h>

static inline __m512 _mm512_log_ps(__m512 x) {
  const __m512 ln2 = _mm512_set1_ps(6.9314718056e-01f);
  const __m512 lg1 = _mm512_set1_ps(6.6666668653e-01f);
  const __m512 lg2 = _mm512_set1_ps(4.0000004172e-01f);
  const __m512 lg3 = _mm512_set1_ps(2.8571429849e-01f);
  const __m512 lg4 = _mm512_set1_ps(2.2222198009e-01f);
  const __m512 half = _mm512_set1_ps(0.5f);
  const __m512 one = _mm512_set1_ps(1.0f);
  const __m512 two = _mm512_set1_ps(2.0f);
  const __m512 sqrt2 = _mm512_set1_ps(1.41421356f);

  __m512i ix = _mm512_castps_si512(x);
  __m512i exp_bits =
      _mm512_and_si512(_mm512_srli_epi32(ix, 23), _mm512_set1_epi32(0xFF));
  __m512i k_i = _mm512_sub_epi32(exp_bits, _mm512_set1_epi32(127));
  __m512i mantissa =
      _mm512_or_si512(_mm512_and_si512(ix, _mm512_set1_epi32(0x007FFFFF)),
                      _mm512_set1_epi32(0x3F800000));
  __m512 m = _mm512_castsi512_ps(mantissa);

  __mmask16 mask = _mm512_cmp_ps_mask(m, sqrt2, _CMP_GT_OS);
  m = _mm512_mask_blend_ps(mask, m, _mm512_mul_ps(m, half));
  k_i = _mm512_mask_add_epi32(k_i, mask, k_i, _mm512_set1_epi32(1));
  __m512 k = _mm512_cvtepi32_ps(k_i);

  __m512 f = _mm512_sub_ps(m, one);
  __m512 s = _mm512_div_ps(f, _mm512_add_ps(two, f));
  __m512 z = _mm512_mul_ps(s, s);
  __m512 w = _mm512_mul_ps(z, z);
  __m512 t1 = _mm512_mul_ps(w, _mm512_fmadd_ps(w, lg4, lg2));
  __m512 t2 = _mm512_mul_ps(z, _mm512_fmadd_ps(w, lg3, lg1));
  __m512 r = _mm512_add_ps(t1, t2);
  __m512 hfsq = _mm512_mul_ps(half, _mm512_mul_ps(f, f));
  __m512 result =
      _mm512_fmadd_ps(k, ln2,
                      _mm512_add_ps(_mm512_sub_ps(f, hfsq),
                                    _mm512_mul_ps(s, _mm512_add_ps(hfsq, r))));

  __mmask16 pos = _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_GT_OS);
  return _mm512_maskz_mov_ps(pos, result);
}

static inline __m512 _mm512_exp_ps(__m512 x) {
  const __m512 log2e = _mm512_set1_ps(1.44269504088896341f);
  const __m512 ln2hi = _mm512_set1_ps(6.93359375000000000e-1f);
  const __m512 ln2lo = _mm512_set1_ps(-2.12194440e-4f);
  const __m512 p0 = _mm512_set1_ps(1.9875691500e-4f);
  const __m512 p1 = _mm512_set1_ps(1.3981999507e-3f);
  const __m512 p2 = _mm512_set1_ps(8.3334519073e-3f);
  const __m512 p3 = _mm512_set1_ps(4.1665795894e-2f);
  const __m512 p4 = _mm512_set1_ps(1.6666665459e-1f);
  const __m512 p5 = _mm512_set1_ps(5.0000001201e-1f);
  const __m512 one = _mm512_set1_ps(1.0f);
  const __m512 exp_hi = _mm512_set1_ps(88.3762626647949f);
  const __m512 exp_lo = _mm512_set1_ps(-87.33654475f);
  const __m512 x_orig = x;

  x = _mm512_max_ps(x, exp_lo);
  x = _mm512_min_ps(x, exp_hi);

  __m512 n = _mm512_roundscale_ps(
      _mm512_mul_ps(x, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m512 r = _mm512_fnmadd_ps(n, ln2hi, x);
  r = _mm512_fnmadd_ps(n, ln2lo, r);

  __m512 p = p0;
  p = _mm512_fmadd_ps(p, r, p1);
  p = _mm512_fmadd_ps(p, r, p2);
  p = _mm512_fmadd_ps(p, r, p3);
  p = _mm512_fmadd_ps(p, r, p4);
  p = _mm512_fmadd_ps(p, r, p5);
  p = _mm512_fmadd_ps(p, _mm512_mul_ps(r, r), _mm512_add_ps(r, one));

  __m512i ni = _mm512_slli_epi32(_mm512_cvtps_epi32(n), 23);
  __m512 result =
      _mm512_castsi512_ps(_mm512_add_epi32(_mm512_castps_si512(p), ni));

  __mmask16 underflow = _mm512_cmp_ps_mask(x_orig, exp_lo, _CMP_LE_OS);
  __mmask16 overflow = _mm512_cmp_ps_mask(x_orig, exp_hi, _CMP_GT_OS);
  result = _mm512_mask_mov_ps(result, underflow, _mm512_setzero_ps());
  return _mm512_mask_mov_ps(result, overflow, _mm512_set1_ps(1.0f / 0.0f));
}

static inline __m512d _mm512_log_pd(__m512d x) {
  double in[8], out[8];
  _mm512_storeu_pd(in, x);
  for (int i = 0; i < 8; i++)
    out[i] = _log_f64(in[i]);
  return _mm512_loadu_pd(out);
}

static inline __m512d _mm512_exp_pd(__m512d x) {
  double in[8], out[8];
  _mm512_storeu_pd(in, x);
  for (int i = 0; i < 8; i++)
    out[i] = _exp_f64(in[i]);
  return _mm512_loadu_pd(out);
}

static inline __m512 _mm512_tanh_ps(__m512 x) {
  const __m512 two = _mm512_set1_ps(2.0f);
  const __m512 one = _mm512_set1_ps(1.0f);
  __m512 e = _mm512_exp_ps(_mm512_mul_ps(_mm512_set1_ps(-2.0f), x));
  return _mm512_sub_ps(_mm512_div_ps(two, _mm512_add_ps(one, e)), one);
}

static inline __m512d _mm512_tanh_pd(__m512d x) {
  const __m512d two = _mm512_set1_pd(2.0);
  const __m512d one = _mm512_set1_pd(1.0);
  __m512d e = _mm512_exp_pd(_mm512_mul_pd(_mm512_set1_pd(-2.0), x));
  return _mm512_sub_pd(_mm512_div_pd(two, _mm512_add_pd(one, e)), one);
}

static inline __m512 _mm512_sigmoid_ps(__m512 x) {
  const __m512 zero = _mm512_setzero_ps();
  const __m512 one = _mm512_set1_ps(1.0f);
  __mmask16 pos = _mm512_cmp_ps_mask(x, zero, _CMP_GE_OS);
  __m512 z_pos = _mm512_exp_ps(_mm512_sub_ps(zero, x));
  __m512 y_pos = _mm512_div_ps(one, _mm512_add_ps(one, z_pos));
  __m512 z_neg = _mm512_exp_ps(x);
  __m512 y_neg = _mm512_div_ps(z_neg, _mm512_add_ps(one, z_neg));
  return _mm512_mask_blend_ps(pos, y_neg, y_pos);
}

static inline __m512d _mm512_sigmoid_pd(__m512d x) {
  const __m512d zero = _mm512_setzero_pd();
  const __m512d one = _mm512_set1_pd(1.0);
  __mmask8 pos = _mm512_cmp_pd_mask(x, zero, _CMP_GE_OS);
  __m512d z_pos = _mm512_exp_pd(_mm512_sub_pd(zero, x));
  __m512d y_pos = _mm512_div_pd(one, _mm512_add_pd(one, z_pos));
  __m512d z_neg = _mm512_exp_pd(x);
  __m512d y_neg = _mm512_div_pd(z_neg, _mm512_add_pd(one, z_neg));
  return _mm512_mask_blend_pd(pos, y_neg, y_pos);
}

static inline void _fast_tanh_f32_avx512(const void *restrict ap,
                                         void *restrict op, size_t n) {
  const float *a = (const float *)ap;
  float *out = (float *)op;
  size_t i = 0;
  for (; i + 16 <= n; i += 16)
    _mm512_storeu_ps(out + i, _mm512_tanh_ps(_mm512_loadu_ps(a + i)));
  for (; i < n; i++)
    out[i] = tanhf(a[i]);
}

static inline void _fast_tanh_f64_avx512(const void *restrict ap,
                                         void *restrict op, size_t n) {
  const double *a = (const double *)ap;
  double *out = (double *)op;
  size_t i = 0;
  for (; i + 8 <= n; i += 8)
    _mm512_storeu_pd(out + i, _mm512_tanh_pd(_mm512_loadu_pd(a + i)));
  for (; i < n; i++)
    out[i] = tanh(a[i]);
}

static inline void _sigmoid_f32_avx512(const float x, float *out) {
  float z = 0.0f;
  if (x >= 0.0f) {
    z = _exp_f32(-x);
    *out = 1.0f / (1.0f + z);
  } else {
    z = _exp_f32(x);
    *out = z / (1.0f + z);
  }
}

static inline void _sigmoid_f64_avx512(const double x, double *out) {
  double z = 0.0;
  if (x >= 0.0) {
    z = _exp_f64(-x);
    *out = 1.0 / (1.0 + z);
  } else {
    z = _exp_f64(x);
    *out = z / (1.0 + z);
  }
}

static inline void _fast_sigmoid_f32_avx512(const void *restrict ap,
                                            void *restrict op, size_t n) {
  const float *a = (const float *)ap;
  float *out = (float *)op;
  size_t i = 0;
  for (; i + 16 <= n; i += 16)
    _mm512_storeu_ps(out + i, _mm512_sigmoid_ps(_mm512_loadu_ps(a + i)));
  for (; i < n; i++)
    _sigmoid_f32_avx512(a[i], out + i);
}

static inline void _fast_sigmoid_f64_avx512(const void *restrict ap,
                                            void *restrict op, size_t n) {
  const double *a = (const double *)ap;
  double *out = (double *)op;
  size_t i = 0;
  for (; i + 8 <= n; i += 8)
    _mm512_storeu_pd(out + i, _mm512_sigmoid_pd(_mm512_loadu_pd(a + i)));
  for (; i < n; i++)
    _sigmoid_f64_avx512(a[i], out + i);
}

static inline __m512 _mm512_sin_ps(__m512 x) {
  float in[16], out[16];
  _mm512_storeu_ps(in, x);
  for (int i = 0; i < 16; i++)
    out[i] = _sin_f32(in[i]);
  return _mm512_loadu_ps(out);
}

static inline __m512 _mm512_cos_ps(__m512 x) {
  float in[16], out[16];
  _mm512_storeu_ps(in, x);
  for (int i = 0; i < 16; i++)
    out[i] = _cos_f32(in[i]);
  return _mm512_loadu_ps(out);
}

static inline void _mm512_sincos_ps(__m512 x, __m512 *s, __m512 *c) {
  float in[16], sout[16], cout[16];
  _mm512_storeu_ps(in, x);
  for (int i = 0; i < 16; i++) {
    sout[i] = _sin_f32(in[i]);
    cout[i] = _cos_f32(in[i]);
  }
  *s = _mm512_loadu_ps(sout);
  *c = _mm512_loadu_ps(cout);
}

#endif /* NUMC_MATH_AVX512_H */
