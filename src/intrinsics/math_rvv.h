/**
 * @file math_rvv.h
 * @brief Vectorized log/exp/sin/cos intrinsics for RISC-V RVV.
 *
 * Same fdlibm/Cephes algorithms as math_avx2.h, ported to RVV
 * scalable-width intrinsics with LMUL=m4 for math kernels.
 * All operations take explicit vl parameter.
 */
#ifndef NUMC_MATH_RVV_H
#define NUMC_MATH_RVV_H

#include "math_helpers.h"
#include <riscv_vector.h>

/* ===================================================================
   Vectorized log — float32 (VL-wide, fdlibm algorithm)
   =================================================================== */

static inline vfloat32m4_t _rvv_log_f32(vfloat32m4_t x, size_t vl) {
  vfloat32m4_t ln2 = __riscv_vfmv_v_f_f32m4(6.9314718056e-01f, vl);
  vfloat32m4_t lg1 = __riscv_vfmv_v_f_f32m4(6.6666668653e-01f, vl);
  vfloat32m4_t lg2 = __riscv_vfmv_v_f_f32m4(4.0000004172e-01f, vl);
  vfloat32m4_t lg3 = __riscv_vfmv_v_f_f32m4(2.8571429849e-01f, vl);
  vfloat32m4_t lg4 = __riscv_vfmv_v_f_f32m4(2.2222198009e-01f, vl);
  vfloat32m4_t vhalf = __riscv_vfmv_v_f_f32m4(0.5f, vl);
  vfloat32m4_t vone = __riscv_vfmv_v_f_f32m4(1.0f, vl);
  vfloat32m4_t vtwo = __riscv_vfmv_v_f_f32m4(2.0f, vl);

  /* Step 1: Argument reduction */
  vuint32m4_t ix = __riscv_vreinterpret_v_f32m4_u32m4(x);

  /* k = exponent - 127 */
  vuint32m4_t exp_bits = __riscv_vsrl_vx_u32m4(ix, 23, vl);
  exp_bits = __riscv_vand_vx_u32m4(exp_bits, 0xFF, vl);
  vint32m4_t k_i = __riscv_vsub_vx_i32m4(
      __riscv_vreinterpret_v_u32m4_i32m4(exp_bits), 127, vl);

  /* m = mantissa with exponent = 0 (biased 127) */
  vuint32m4_t mantissa = __riscv_vor_vx_u32m4(
      __riscv_vand_vx_u32m4(ix, 0x007FFFFF, vl), 0x3F800000, vl);
  vfloat32m4_t m = __riscv_vreinterpret_v_u32m4_f32m4(mantissa);

  /* If m > sqrt(2), halve m and increment k */
  vbool8_t mask = __riscv_vmfgt_vf_f32m4_b8(m, 1.41421356f, vl);
  m = __riscv_vfmul_vv_f32m4_mu(mask, m, m, vhalf, vl);
  k_i = __riscv_vadd_vx_i32m4_mu(mask, k_i, k_i, 1, vl);

  vfloat32m4_t k = __riscv_vfcvt_f_x_v_f32m4(k_i, vl);

  /* f = m - 1 */
  vfloat32m4_t f = __riscv_vfsub_vv_f32m4(m, vone, vl);

  /* s = f / (2 + f) */
  vfloat32m4_t s =
      __riscv_vfdiv_vv_f32m4(f, __riscv_vfadd_vv_f32m4(vtwo, f, vl), vl);
  vfloat32m4_t z = __riscv_vfmul_vv_f32m4(s, s, vl);
  vfloat32m4_t w = __riscv_vfmul_vv_f32m4(z, z, vl);

  /* Horner polynomial */
  vfloat32m4_t t1 = __riscv_vfmadd_vv_f32m4(w, lg4, lg2, vl);
  t1 = __riscv_vfmul_vv_f32m4(w, t1, vl);
  vfloat32m4_t t2 = __riscv_vfmadd_vv_f32m4(w, lg3, lg1, vl);
  t2 = __riscv_vfmul_vv_f32m4(z, t2, vl);
  vfloat32m4_t r = __riscv_vfadd_vv_f32m4(t1, t2, vl);

  /* result = k*ln2 + f - hfsq + s*(hfsq + r) */
  vfloat32m4_t hfsq =
      __riscv_vfmul_vv_f32m4(vhalf, __riscv_vfmul_vv_f32m4(f, f, vl), vl);
  vfloat32m4_t result = __riscv_vfmadd_vv_f32m4(
      k, ln2,
      __riscv_vfadd_vv_f32m4(
          __riscv_vfsub_vv_f32m4(f, hfsq, vl),
          __riscv_vfmul_vv_f32m4(s, __riscv_vfadd_vv_f32m4(hfsq, r, vl), vl),
          vl),
      vl);

  /* Handle x <= 0 */
  vbool8_t pos_mask = __riscv_vmfgt_vf_f32m4_b8(x, 0.0f, vl);
  result = __riscv_vfmerge_vfm_f32m4(result, 0.0f,
                                     __riscv_vmnot_m_b8(pos_mask, vl), vl);

  return result;
}

/* ===================================================================
   Vectorized exp — float32 (VL-wide, Cephes algorithm)
   =================================================================== */

static inline vfloat32m4_t _rvv_exp_f32(vfloat32m4_t x, size_t vl) {
  vfloat32m4_t p0c = __riscv_vfmv_v_f_f32m4(1.9875691500e-4f, vl);
  vfloat32m4_t p1c = __riscv_vfmv_v_f_f32m4(1.3981999507e-3f, vl);
  vfloat32m4_t p2c = __riscv_vfmv_v_f_f32m4(8.3334519073e-3f, vl);
  vfloat32m4_t p3c = __riscv_vfmv_v_f_f32m4(4.1665795894e-2f, vl);
  vfloat32m4_t p4c = __riscv_vfmv_v_f_f32m4(1.6666665459e-1f, vl);
  vfloat32m4_t p5c = __riscv_vfmv_v_f_f32m4(5.0000001201e-1f, vl);
  vfloat32m4_t vone = __riscv_vfmv_v_f_f32m4(1.0f, vl);

  /* Save original for underflow/overflow masking */
  vfloat32m4_t x_orig = x;

  /* Clamp */
  x = __riscv_vfmax_vf_f32m4(x, -87.33654475f, vl);
  x = __riscv_vfmin_vf_f32m4(x, 88.3762626647949f, vl);

  /* n = round(x * log2e) */
  vfloat32m4_t scaled = __riscv_vfmul_vf_f32m4(x, 1.44269504088896341f, vl);
  /* RVV: convert to int with rounding, then back */
  vint32m4_t n_i = __riscv_vfcvt_x_f_v_i32m4(scaled, vl);
  vfloat32m4_t n = __riscv_vfcvt_f_x_v_f32m4(n_i, vl);

  /* r = x - n * ln2 */
  vfloat32m4_t r = __riscv_vfnmsub_vf_f32m4(n, 6.93359375000000000e-1f, x, vl);
  r = __riscv_vfnmsac_vf_f32m4(r, -2.12194440e-4f, n, vl);

  /* Horner polynomial */
  vfloat32m4_t p = p0c;
  p = __riscv_vfmadd_vv_f32m4(p, r, p1c, vl);
  p = __riscv_vfmadd_vv_f32m4(p, r, p2c, vl);
  p = __riscv_vfmadd_vv_f32m4(p, r, p3c, vl);
  p = __riscv_vfmadd_vv_f32m4(p, r, p4c, vl);
  p = __riscv_vfmadd_vv_f32m4(p, r, p5c, vl);
  p = __riscv_vfmadd_vv_f32m4(p, __riscv_vfmul_vv_f32m4(r, r, vl),
                              __riscv_vfadd_vv_f32m4(r, vone, vl), vl);

  /* Scale by 2^n */
  n_i = __riscv_vsll_vx_i32m4(n_i, 23, vl);
  vint32m4_t result_i =
      __riscv_vadd_vv_i32m4(__riscv_vreinterpret_v_f32m4_i32m4(p), n_i, vl);

  vfloat32m4_t result = __riscv_vreinterpret_v_i32m4_f32m4(result_i);

  /* Underflow: x < -87.33654475 -> 0, Overflow: x > 88.376 -> +inf */
  vfloat32m4_t zero = __riscv_vfmv_v_f_f32m4(0.0f, vl);
  vfloat32m4_t inf = __riscv_vfmv_v_f_f32m4(1.0f / 0.0f, vl);
  vbool8_t underflow = __riscv_vmflt_vf_f32m4_b8(x_orig, -87.33654475f, vl);
  vbool8_t overflow = __riscv_vmfgt_vf_f32m4_b8(x_orig, 88.3762626647949f, vl);
  result = __riscv_vmerge_vvm_f32m4(result, zero, underflow, vl);
  result = __riscv_vmerge_vvm_f32m4(result, inf, overflow, vl);

  return result;
}

/* ===================================================================
   Vectorized log — float64 (VL-wide, fdlibm algorithm)
   =================================================================== */

static inline vfloat64m4_t _rvv_log_f64(vfloat64m4_t x, size_t vl) {
  vfloat64m4_t ln2 = __riscv_vfmv_v_f_f64m4(6.9314718055994530942e-01, vl);
  vfloat64m4_t lg1 = __riscv_vfmv_v_f_f64m4(6.6666666666666735130e-01, vl);
  vfloat64m4_t lg2 = __riscv_vfmv_v_f_f64m4(3.9999999999940941908e-01, vl);
  vfloat64m4_t lg3 = __riscv_vfmv_v_f_f64m4(2.8571428743662391490e-01, vl);
  vfloat64m4_t lg4 = __riscv_vfmv_v_f_f64m4(2.2221984321497839600e-01, vl);
  vfloat64m4_t lg5 = __riscv_vfmv_v_f_f64m4(1.8183572161618050120e-01, vl);
  vfloat64m4_t lg6 = __riscv_vfmv_v_f_f64m4(1.5313837699209373320e-01, vl);
  vfloat64m4_t lg7 = __riscv_vfmv_v_f_f64m4(1.4798198605116585910e-01, vl);
  vfloat64m4_t vhalf = __riscv_vfmv_v_f_f64m4(0.5, vl);
  vfloat64m4_t vone = __riscv_vfmv_v_f_f64m4(1.0, vl);
  vfloat64m4_t vtwo = __riscv_vfmv_v_f_f64m4(2.0, vl);

  vuint64m4_t ix = __riscv_vreinterpret_v_f64m4_u64m4(x);
  vuint64m4_t exp_bits = __riscv_vsrl_vx_u64m4(ix, 52, vl);
  exp_bits = __riscv_vand_vx_u64m4(exp_bits, 0x7FF, vl);
  vint64m4_t k_i = __riscv_vsub_vx_i64m4(
      __riscv_vreinterpret_v_u64m4_i64m4(exp_bits), 1023, vl);

  vuint64m4_t mantissa =
      __riscv_vor_vx_u64m4(__riscv_vand_vx_u64m4(ix, 0x000FFFFFFFFFFFFFULL, vl),
                           0x3FF0000000000000ULL, vl);
  vfloat64m4_t m = __riscv_vreinterpret_v_u64m4_f64m4(mantissa);

  vbool16_t mask = __riscv_vmfgt_vf_f64m4_b16(m, 1.4142135623730951, vl);
  m = __riscv_vfmul_vv_f64m4_mu(mask, m, m, vhalf, vl);
  k_i = __riscv_vadd_vx_i64m4_mu(mask, k_i, k_i, 1, vl);

  vfloat64m4_t k = __riscv_vfcvt_f_x_v_f64m4(k_i, vl);

  vfloat64m4_t f = __riscv_vfsub_vv_f64m4(m, vone, vl);
  vfloat64m4_t s =
      __riscv_vfdiv_vv_f64m4(f, __riscv_vfadd_vv_f64m4(vtwo, f, vl), vl);
  vfloat64m4_t z = __riscv_vfmul_vv_f64m4(s, s, vl);
  vfloat64m4_t w = __riscv_vfmul_vv_f64m4(z, z, vl);

  /* t1 = w*(lg2 + w*(lg4 + w*lg6)) */
  vfloat64m4_t t1 = __riscv_vfmadd_vv_f64m4(w, lg6, lg4, vl);
  t1 = __riscv_vfmadd_vv_f64m4(t1, w, lg2, vl);
  t1 = __riscv_vfmul_vv_f64m4(w, t1, vl);

  /* t2 = z*(lg1 + w*(lg3 + w*(lg5 + w*lg7))) */
  vfloat64m4_t t2 = __riscv_vfmadd_vv_f64m4(w, lg7, lg5, vl);
  t2 = __riscv_vfmadd_vv_f64m4(t2, w, lg3, vl);
  t2 = __riscv_vfmadd_vv_f64m4(t2, w, lg1, vl);
  t2 = __riscv_vfmul_vv_f64m4(z, t2, vl);

  vfloat64m4_t r = __riscv_vfadd_vv_f64m4(t1, t2, vl);

  vfloat64m4_t hfsq =
      __riscv_vfmul_vv_f64m4(vhalf, __riscv_vfmul_vv_f64m4(f, f, vl), vl);
  vfloat64m4_t result = __riscv_vfmadd_vv_f64m4(
      k, ln2,
      __riscv_vfadd_vv_f64m4(
          __riscv_vfsub_vv_f64m4(f, hfsq, vl),
          __riscv_vfmul_vv_f64m4(s, __riscv_vfadd_vv_f64m4(hfsq, r, vl), vl),
          vl),
      vl);

  vbool16_t pos_mask = __riscv_vmfgt_vf_f64m4_b16(x, 0.0, vl);
  result = __riscv_vfmerge_vfm_f64m4(result, 0.0,
                                     __riscv_vmnot_m_b16(pos_mask, vl), vl);

  return result;
}

/* ===================================================================
   Vectorized exp — float64 (VL-wide, Cephes algorithm)
   =================================================================== */

static inline vfloat64m4_t _rvv_exp_f64(vfloat64m4_t x, size_t vl) {
  vfloat64m4_t c2 = __riscv_vfmv_v_f_f64m4(5.00000000000000000000e-01, vl);
  vfloat64m4_t c3 = __riscv_vfmv_v_f_f64m4(1.66666666666666666667e-01, vl);
  vfloat64m4_t c4 = __riscv_vfmv_v_f_f64m4(4.16666666666666666667e-02, vl);
  vfloat64m4_t c5 = __riscv_vfmv_v_f_f64m4(8.33333333333333333333e-03, vl);
  vfloat64m4_t c6 = __riscv_vfmv_v_f_f64m4(1.38888888888888888889e-03, vl);
  vfloat64m4_t c7 = __riscv_vfmv_v_f_f64m4(1.98412698412698412698e-04, vl);
  vfloat64m4_t c8 = __riscv_vfmv_v_f_f64m4(2.48015873015873015873e-05, vl);
  vfloat64m4_t c9 = __riscv_vfmv_v_f_f64m4(2.75573192239858906526e-06, vl);
  vfloat64m4_t c10 = __riscv_vfmv_v_f_f64m4(2.75573192239858906526e-07, vl);
  vfloat64m4_t c11 = __riscv_vfmv_v_f_f64m4(2.50521083854417187751e-08, vl);
  vfloat64m4_t c12 = __riscv_vfmv_v_f_f64m4(2.08767569878680989792e-09, vl);
  vfloat64m4_t vone = __riscv_vfmv_v_f_f64m4(1.0, vl);

  /* Save original for underflow/overflow masking */
  vfloat64m4_t x_orig = x;

  /* Clamp */
  x = __riscv_vfmax_vf_f64m4(x, -708.3964185322641, vl);
  x = __riscv_vfmin_vf_f64m4(x, 709.782712893383996843, vl);

  /* n = round(x * log2e) */
  vfloat64m4_t scaled =
      __riscv_vfmul_vf_f64m4(x, 1.44269504088896338700e+00, vl);
  vint64m4_t n_i = __riscv_vfcvt_x_f_v_i64m4(scaled, vl);
  vfloat64m4_t n = __riscv_vfcvt_f_x_v_f64m4(n_i, vl);

  /* r = x - n * ln2 */
  vfloat64m4_t r =
      __riscv_vfnmsub_vf_f64m4(n, 6.93147180369123816490e-01, x, vl);
  r = __riscv_vfnmsac_vf_f64m4(r, 1.90821492927058770002e-10, n, vl);

  /* Horner polynomial */
  vfloat64m4_t p = c12;
  p = __riscv_vfmadd_vv_f64m4(p, r, c11, vl);
  p = __riscv_vfmadd_vv_f64m4(p, r, c10, vl);
  p = __riscv_vfmadd_vv_f64m4(p, r, c9, vl);
  p = __riscv_vfmadd_vv_f64m4(p, r, c8, vl);
  p = __riscv_vfmadd_vv_f64m4(p, r, c7, vl);
  p = __riscv_vfmadd_vv_f64m4(p, r, c6, vl);
  p = __riscv_vfmadd_vv_f64m4(p, r, c5, vl);
  p = __riscv_vfmadd_vv_f64m4(p, r, c4, vl);
  p = __riscv_vfmadd_vv_f64m4(p, r, c3, vl);
  p = __riscv_vfmadd_vv_f64m4(p, r, c2, vl);
  p = __riscv_vfmadd_vv_f64m4(p, __riscv_vfmul_vv_f64m4(r, r, vl),
                              __riscv_vfadd_vv_f64m4(r, vone, vl), vl);

  /* Scale by 2^n */
  n_i = __riscv_vsll_vx_i64m4(n_i, 52, vl);
  vint64m4_t result_i =
      __riscv_vadd_vv_i64m4(__riscv_vreinterpret_v_f64m4_i64m4(p), n_i, vl);

  vfloat64m4_t result = __riscv_vreinterpret_v_i64m4_f64m4(result_i);

  /* Underflow: x < -708.3964185322641 -> 0, Overflow: x > 709.78 -> +inf */
  vfloat64m4_t zero = __riscv_vfmv_v_f_f64m4(0.0, vl);
  vfloat64m4_t inf = __riscv_vfmv_v_f_f64m4(1.0 / 0.0, vl);
  vbool16_t underflow =
      __riscv_vmflt_vf_f64m4_b16(x_orig, -708.3964185322641, vl);
  vbool16_t overflow =
      __riscv_vmfgt_vf_f64m4_b16(x_orig, 709.782712893383996843, vl);
  result = __riscv_vmerge_vvm_f64m4(result, zero, underflow, vl);
  result = __riscv_vmerge_vvm_f64m4(result, inf, overflow, vl);

  return result;
}

/* ===================================================================
   Vectorized sin — float32 (VL-wide, Cephes algorithm)
   =================================================================== */

static inline vfloat32m4_t _rvv_sin_f32(vfloat32m4_t x, size_t vl) {
  /* Extract sign */
  vuint32m4_t sign_bit = __riscv_vand_vx_u32m4(
      __riscv_vreinterpret_v_f32m4_u32m4(x), 0x80000000u, vl);
  vfloat32m4_t xa = __riscv_vfabs_v_f32m4(x, vl);

  /* j = (int)(|x| * 4/pi), round up to even */
  vfloat32m4_t y = __riscv_vfmul_vf_f32m4(xa, 1.27323954473516f, vl);
  vint32m4_t j = __riscv_vfcvt_rtz_x_f_v_i32m4(y, vl);
  j = __riscv_vadd_vv_i32m4(j, __riscv_vand_vx_i32m4(j, 1, vl), vl);
  y = __riscv_vfcvt_f_x_v_f32m4(j, vl);

  /* j&4 -> sign swap */
  vuint32m4_t swap_sign = __riscv_vsll_vx_u32m4(
      __riscv_vreinterpret_v_i32m4_u32m4(__riscv_vand_vx_i32m4(j, 4, vl)), 29,
      vl);
  sign_bit = __riscv_vxor_vv_u32m4(sign_bit, swap_sign, vl);

  /* j&2 -> use cosine polynomial */
  vbool8_t poly_mask =
      __riscv_vmseq_vx_i32m4_b8(__riscv_vand_vx_i32m4(j, 2, vl), 2, vl);

  /* Range reduction */
  xa = __riscv_vfmadd_vf_f32m4(y, -0.78515625f, xa, vl);
  xa = __riscv_vfmadd_vf_f32m4(y, -2.4187564849853515625e-4f, xa, vl);
  xa = __riscv_vfmadd_vf_f32m4(y, -3.77489497744594108e-8f, xa, vl);

  vfloat32m4_t z = __riscv_vfmul_vv_f32m4(xa, xa, vl);

  /* cos polynomial */
  vfloat32m4_t yc = __riscv_vfmv_v_f_f32m4(2.443315711809948e-5f, vl);
  yc = __riscv_vfmadd_vv_f32m4(
      yc, z, __riscv_vfmv_v_f_f32m4(-1.388731625493765e-3f, vl), vl);
  yc = __riscv_vfmadd_vv_f32m4(
      yc, z, __riscv_vfmv_v_f_f32m4(4.166664568298827e-2f, vl), vl);
  yc = __riscv_vfmul_vv_f32m4(yc, __riscv_vfmul_vv_f32m4(z, z, vl), vl);
  yc = __riscv_vfnmsac_vf_f32m4(yc, 0.5f, z, vl);
  yc = __riscv_vfadd_vf_f32m4(yc, 1.0f, vl);

  /* sin polynomial */
  vfloat32m4_t ys = __riscv_vfmv_v_f_f32m4(-1.9515295891e-4f, vl);
  ys = __riscv_vfmadd_vv_f32m4(
      ys, z, __riscv_vfmv_v_f_f32m4(8.3321608736e-3f, vl), vl);
  ys = __riscv_vfmadd_vv_f32m4(
      ys, z, __riscv_vfmv_v_f_f32m4(-1.6666654611e-1f, vl), vl);
  ys = __riscv_vfmul_vv_f32m4(ys, __riscv_vfmul_vv_f32m4(z, xa, vl), vl);
  ys = __riscv_vfadd_vv_f32m4(ys, xa, vl);

  /* Select sin/cos based on octant */
  y = __riscv_vmerge_vvm_f32m4(ys, yc, poly_mask, vl);

  /* Apply sign */
  y = __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vxor_vv_u32m4(
      __riscv_vreinterpret_v_f32m4_u32m4(y), sign_bit, vl));
  return y;
}

/* ===================================================================
   Vectorized cos — float32 (VL-wide, Cephes algorithm)
   =================================================================== */

static inline vfloat32m4_t _rvv_cos_f32(vfloat32m4_t x, size_t vl) {
  vfloat32m4_t xa = __riscv_vfabs_v_f32m4(x, vl);

  vfloat32m4_t y = __riscv_vfmul_vf_f32m4(xa, 1.27323954473516f, vl);
  vint32m4_t j = __riscv_vfcvt_rtz_x_f_v_i32m4(y, vl);
  j = __riscv_vadd_vv_i32m4(j, __riscv_vand_vx_i32m4(j, 1, vl), vl);
  y = __riscv_vfcvt_f_x_v_f32m4(j, vl);

  /* cos: shift j by 2 */
  j = __riscv_vsub_vx_i32m4(j, 2, vl);

  /* sign: (j^4)&4 */
  vuint32m4_t sign_bit = __riscv_vsll_vx_u32m4(
      __riscv_vreinterpret_v_i32m4_u32m4(
          __riscv_vand_vx_i32m4(__riscv_vxor_vx_i32m4(j, 4, vl), 4, vl)),
      29, vl);

  vbool8_t poly_mask =
      __riscv_vmseq_vx_i32m4_b8(__riscv_vand_vx_i32m4(j, 2, vl), 2, vl);

  xa = __riscv_vfmadd_vf_f32m4(y, -0.78515625f, xa, vl);
  xa = __riscv_vfmadd_vf_f32m4(y, -2.4187564849853515625e-4f, xa, vl);
  xa = __riscv_vfmadd_vf_f32m4(y, -3.77489497744594108e-8f, xa, vl);

  vfloat32m4_t z = __riscv_vfmul_vv_f32m4(xa, xa, vl);

  vfloat32m4_t yc = __riscv_vfmv_v_f_f32m4(2.443315711809948e-5f, vl);
  yc = __riscv_vfmadd_vv_f32m4(
      yc, z, __riscv_vfmv_v_f_f32m4(-1.388731625493765e-3f, vl), vl);
  yc = __riscv_vfmadd_vv_f32m4(
      yc, z, __riscv_vfmv_v_f_f32m4(4.166664568298827e-2f, vl), vl);
  yc = __riscv_vfmul_vv_f32m4(yc, __riscv_vfmul_vv_f32m4(z, z, vl), vl);
  yc = __riscv_vfnmsac_vf_f32m4(yc, 0.5f, z, vl);
  yc = __riscv_vfadd_vf_f32m4(yc, 1.0f, vl);

  vfloat32m4_t ys = __riscv_vfmv_v_f_f32m4(-1.9515295891e-4f, vl);
  ys = __riscv_vfmadd_vv_f32m4(
      ys, z, __riscv_vfmv_v_f_f32m4(8.3321608736e-3f, vl), vl);
  ys = __riscv_vfmadd_vv_f32m4(
      ys, z, __riscv_vfmv_v_f_f32m4(-1.6666654611e-1f, vl), vl);
  ys = __riscv_vfmul_vv_f32m4(ys, __riscv_vfmul_vv_f32m4(z, xa, vl), vl);
  ys = __riscv_vfadd_vv_f32m4(ys, xa, vl);

  y = __riscv_vmerge_vvm_f32m4(ys, yc, poly_mask, vl);
  y = __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vxor_vv_u32m4(
      __riscv_vreinterpret_v_f32m4_u32m4(y), sign_bit, vl));
  return y;
}

/* ===================================================================
   Combined sincos — float32 (VL-wide)
   =================================================================== */

static inline void _rvv_sincos_f32(vfloat32m4_t x, vfloat32m4_t *s,
                                   vfloat32m4_t *c, size_t vl) {
  vuint32m4_t sign_sin = __riscv_vand_vx_u32m4(
      __riscv_vreinterpret_v_f32m4_u32m4(x), 0x80000000u, vl);
  vfloat32m4_t xa = __riscv_vfabs_v_f32m4(x, vl);

  vfloat32m4_t y = __riscv_vfmul_vf_f32m4(xa, 1.27323954473516f, vl);
  vint32m4_t j = __riscv_vfcvt_rtz_x_f_v_i32m4(y, vl);
  j = __riscv_vadd_vv_i32m4(j, __riscv_vand_vx_i32m4(j, 1, vl), vl);
  y = __riscv_vfcvt_f_x_v_f32m4(j, vl);

  vuint32m4_t j4 =
      __riscv_vreinterpret_v_i32m4_u32m4(__riscv_vand_vx_i32m4(j, 4, vl));
  sign_sin =
      __riscv_vxor_vv_u32m4(sign_sin, __riscv_vsll_vx_u32m4(j4, 29, vl), vl);

  /* cos sign */
  vuint32m4_t sign_cos = __riscv_vsll_vx_u32m4(
      __riscv_vreinterpret_v_i32m4_u32m4(__riscv_vand_vx_i32m4(
          __riscv_vxor_vx_i32m4(__riscv_vsub_vx_i32m4(j, 2, vl), 4, vl), 4,
          vl)),
      29, vl);

  vbool8_t poly_mask =
      __riscv_vmseq_vx_i32m4_b8(__riscv_vand_vx_i32m4(j, 2, vl), 2, vl);

  xa = __riscv_vfmadd_vf_f32m4(y, -0.78515625f, xa, vl);
  xa = __riscv_vfmadd_vf_f32m4(y, -2.4187564849853515625e-4f, xa, vl);
  xa = __riscv_vfmadd_vf_f32m4(y, -3.77489497744594108e-8f, xa, vl);

  vfloat32m4_t z = __riscv_vfmul_vv_f32m4(xa, xa, vl);

  vfloat32m4_t yc = __riscv_vfmv_v_f_f32m4(2.443315711809948e-5f, vl);
  yc = __riscv_vfmadd_vv_f32m4(
      yc, z, __riscv_vfmv_v_f_f32m4(-1.388731625493765e-3f, vl), vl);
  yc = __riscv_vfmadd_vv_f32m4(
      yc, z, __riscv_vfmv_v_f_f32m4(4.166664568298827e-2f, vl), vl);
  yc = __riscv_vfmul_vv_f32m4(yc, __riscv_vfmul_vv_f32m4(z, z, vl), vl);
  yc = __riscv_vfnmsac_vf_f32m4(yc, 0.5f, z, vl);
  yc = __riscv_vfadd_vf_f32m4(yc, 1.0f, vl);

  vfloat32m4_t yz = __riscv_vfmv_v_f_f32m4(-1.9515295891e-4f, vl);
  yz = __riscv_vfmadd_vv_f32m4(
      yz, z, __riscv_vfmv_v_f_f32m4(8.3321608736e-3f, vl), vl);
  yz = __riscv_vfmadd_vv_f32m4(
      yz, z, __riscv_vfmv_v_f_f32m4(-1.6666654611e-1f, vl), vl);
  yz = __riscv_vfmul_vv_f32m4(yz, __riscv_vfmul_vv_f32m4(z, xa, vl), vl);
  yz = __riscv_vfadd_vv_f32m4(yz, xa, vl);

  /* sin */
  vfloat32m4_t sin_val = __riscv_vmerge_vvm_f32m4(yz, yc, poly_mask, vl);
  *s = __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vxor_vv_u32m4(
      __riscv_vreinterpret_v_f32m4_u32m4(sin_val), sign_sin, vl));

  /* cos: opposite selection */
  vbool8_t cos_poly_mask = __riscv_vmnot_m_b8(poly_mask, vl);
  vfloat32m4_t cos_val = __riscv_vmerge_vvm_f32m4(yz, yc, cos_poly_mask, vl);
  *c = __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vxor_vv_u32m4(
      __riscv_vreinterpret_v_f32m4_u32m4(cos_val), sign_cos, vl));
}

/* ===================================================================
    Vectorized tanh — float32/float64 (VLA, LMUL=4)
    Uses identity: tanh(x) = 2 / (1 + exp(-2x)) - 1
    =================================================================== */

static inline vfloat32m4_t _rvv_tanh_f32(vfloat32m4_t x, size_t vl) {
  vfloat32m4_t vtwo = __riscv_vfmv_v_f_f32m4(2.0f, vl);
  vfloat32m4_t vone = __riscv_vfmv_v_f_f32m4(1.0f, vl);
  vfloat32m4_t vneg2 = __riscv_vfmv_v_f_f32m4(-2.0f, vl);
  vfloat32m4_t e = _rvv_exp_f32(__riscv_vfmul_vv_f32m4(vneg2, x, vl), vl);
  return __riscv_vfsub_vv_f32m4(
      __riscv_vfdiv_vv_f32m4(vtwo, __riscv_vfadd_vv_f32m4(vone, e, vl), vl),
      vone, vl);
}

static inline vfloat64m4_t _rvv_tanh_f64(vfloat64m4_t x, size_t vl) {
  vfloat64m4_t vtwo = __riscv_vfmv_v_f_f64m4(2.0, vl);
  vfloat64m4_t vone = __riscv_vfmv_v_f_f64m4(1.0, vl);
  vfloat64m4_t vneg2 = __riscv_vfmv_v_f_f64m4(-2.0, vl);
  vfloat64m4_t e = _rvv_exp_f64(__riscv_vfmul_vv_f64m4(vneg2, x, vl), vl);
  return __riscv_vfsub_vv_f64m4(
      __riscv_vfdiv_vv_f64m4(vtwo, __riscv_vfadd_vv_f64m4(vone, e, vl), vl),
      vone, vl);
}

static inline void _fast_tanh_f32_rvv(const void *restrict ap,
                                      void *restrict op, size_t n) {
  const float *a = (const float *)ap;
  float *out = (float *)op;
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m4(n - i);
    vfloat32m4_t va = __riscv_vle32_v_f32m4(a + i, vl);
    __riscv_vse32_v_f32m4(out + i, _rvv_tanh_f32(va, vl), vl);
  }
}

static inline void _fast_tanh_f64_rvv(const void *restrict ap,
                                      void *restrict op, size_t n) {
  const double *a = (const double *)ap;
  double *out = (double *)op;
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e64m4(n - i);
    vfloat64m4_t va = __riscv_vle64_v_f64m4(a + i, vl);
    __riscv_vse64_v_f64m4(out + i, _rvv_tanh_f64(va, vl), vl);
  }
}

static inline vfloat32m4_t _rvv_sigmoid_f32(vfloat32m4_t x, size_t vl) {
  const vfloat32m4_t zero = __riscv_vfmv_v_f_f32m4(0.0f, vl);
  const vfloat32m4_t one = __riscv_vfmv_v_f_f32m4(1.0f, vl);
  vbool8_t pos = __riscv_vmfge_vv_f32m4_b8(x, zero, vl);
  vfloat32m4_t z_pos = _rvv_exp_f32(__riscv_vfneg_v_f32m4(x, vl), vl);
  vfloat32m4_t y_pos =
      __riscv_vfdiv_vv_f32m4(one, __riscv_vfadd_vv_f32m4(one, z_pos, vl), vl);
  vfloat32m4_t z_neg = _rvv_exp_f32(x, vl);
  vfloat32m4_t y_neg =
      __riscv_vfdiv_vv_f32m4(z_neg, __riscv_vfadd_vv_f32m4(one, z_neg, vl), vl);
  return __riscv_vmerge_vvm_f32m4(y_neg, y_pos, pos, vl);
}

static inline vfloat64m4_t _rvv_sigmoid_f64(vfloat64m4_t x, size_t vl) {
  const vfloat64m4_t zero = __riscv_vfmv_v_f_f64m4(0.0, vl);
  const vfloat64m4_t one = __riscv_vfmv_v_f_f64m4(1.0, vl);
  vbool16_t pos = __riscv_vmfge_vv_f64m4_b16(x, zero, vl);
  vfloat64m4_t z_pos = _rvv_exp_f64(__riscv_vfneg_v_f64m4(x, vl), vl);
  vfloat64m4_t y_pos =
      __riscv_vfdiv_vv_f64m4(one, __riscv_vfadd_vv_f64m4(one, z_pos, vl), vl);
  vfloat64m4_t z_neg = _rvv_exp_f64(x, vl);
  vfloat64m4_t y_neg =
      __riscv_vfdiv_vv_f64m4(z_neg, __riscv_vfadd_vv_f64m4(one, z_neg, vl), vl);
  return __riscv_vmerge_vvm_f64m4(y_neg, y_pos, pos, vl);
}

static inline void _sigmoid_f32_rvv(const float x, float *out) {
  float z = 0.0f;
  if (x >= 0.0f) {
    z = _exp_f32(-x);
    *out = 1.0f / (1.0f + z);
  } else {
    z = _exp_f32(x);
    *out = z / (1.0f + z);
  }
}

static inline void _sigmoid_f64_rvv(const double x, double *out) {
  double z = 0.0;
  if (x >= 0.0) {
    z = _exp_f64(-x);
    *out = 1.0 / (1.0 + z);
  } else {
    z = _exp_f64(x);
    *out = z / (1.0 + z);
  }
}

static inline void _fast_sigmoid_f32_rvv(const void *restrict ap,
                                         void *restrict op, size_t n) {
  const float *a = (const float *)ap;
  float *out = (float *)op;
  size_t i = 0;
  while (i < n) {
    size_t vl = __riscv_vsetvl_e32m4(n - i);
    vfloat32m4_t x = __riscv_vle32_v_f32m4(a + i, vl);
    __riscv_vse32_v_f32m4(out + i, _rvv_sigmoid_f32(x, vl), vl);
    i += vl;
  }
}

static inline void _fast_sigmoid_f64_rvv(const void *restrict ap,
                                         void *restrict op, size_t n) {
  const double *a = (const double *)ap;
  double *out = (double *)op;
  size_t i = 0;
  while (i < n) {
    size_t vl = __riscv_vsetvl_e64m4(n - i);
    vfloat64m4_t x = __riscv_vle64_v_f64m4(a + i, vl);
    __riscv_vse64_v_f64m4(out + i, _rvv_sigmoid_f64(x, vl), vl);
    i += vl;
  }
}

#endif /* NUMC_MATH_RVV_H */
