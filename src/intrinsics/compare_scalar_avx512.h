/**
 * @file compare_scalar_avx512.h
 * @brief AVX-512 scalar comparison kernels for all 10 types.
 *
 * AVX-512 comparisons return mask registers. We convert to 0/1 vectors
 * using _mm512_maskz_set1 (integers) or _mm512_mask_blend (floats).
 * Native unsigned and signed comparisons — no XOR bias needed.
 */
#ifndef NUMC_COMPARE_SCALAR_AVX512_H
#define NUMC_COMPARE_SCALAR_AVX512_H

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

/* ── Float: f32 (16 per vector) ─────────────────────────────────── */

#define CMPSC_F32_FUNC_AVX512(NAME, IMM, SCALAR_OP)                    \
  static inline void _cmpsc_##NAME##_f32_avx512(                       \
      const void *restrict ap, const void *restrict sp,                \
      void *restrict op, size_t n) {                                   \
    const float *a = (const float *)ap;                                \
    const float s = *(const float *)sp;                                \
    float *out = (float *)op;                                          \
    const __m512 vs = _mm512_set1_ps(s);                               \
    const __m512 zero = _mm512_setzero_ps();                           \
    const __m512 one = _mm512_set1_ps(1.0f);                           \
    size_t i = 0;                                                      \
    for (; i + 16 <= n; i += 16) {                                     \
      __m512 va = _mm512_loadu_ps(a + i);                              \
      __mmask16 m = _mm512_cmp_ps_mask(va, vs, (IMM));                 \
      _mm512_storeu_ps(out + i, _mm512_mask_blend_ps(m, zero, one));   \
    }                                                                  \
    for (; i < n; i++)                                                 \
      out[i] = (float)(a[i] SCALAR_OP s);                             \
  }

CMPSC_F32_FUNC_AVX512(eq, _CMP_EQ_OQ, ==)
CMPSC_F32_FUNC_AVX512(gt, _CMP_GT_OS, >)
CMPSC_F32_FUNC_AVX512(lt, _CMP_LT_OS, <)
CMPSC_F32_FUNC_AVX512(ge, _CMP_GE_OS, >=)
CMPSC_F32_FUNC_AVX512(le, _CMP_LE_OS, <=)
#undef CMPSC_F32_FUNC_AVX512

/* ── Float: f64 (8 per vector) ──────────────────────────────────── */

#define CMPSC_F64_FUNC_AVX512(NAME, IMM, SCALAR_OP)                    \
  static inline void _cmpsc_##NAME##_f64_avx512(                       \
      const void *restrict ap, const void *restrict sp,                \
      void *restrict op, size_t n) {                                   \
    const double *a = (const double *)ap;                              \
    const double s = *(const double *)sp;                              \
    double *out = (double *)op;                                        \
    const __m512d vs = _mm512_set1_pd(s);                              \
    const __m512d zero = _mm512_setzero_pd();                          \
    const __m512d one = _mm512_set1_pd(1.0);                           \
    size_t i = 0;                                                      \
    for (; i + 8 <= n; i += 8) {                                       \
      __m512d va = _mm512_loadu_pd(a + i);                             \
      __mmask8 m = _mm512_cmp_pd_mask(va, vs, (IMM));                  \
      _mm512_storeu_pd(out + i, _mm512_mask_blend_pd(m, zero, one));   \
    }                                                                  \
    for (; i < n; i++)                                                 \
      out[i] = (double)(a[i] SCALAR_OP s);                            \
  }

CMPSC_F64_FUNC_AVX512(eq, _CMP_EQ_OQ, ==)
CMPSC_F64_FUNC_AVX512(gt, _CMP_GT_OS, >)
CMPSC_F64_FUNC_AVX512(lt, _CMP_LT_OS, <)
CMPSC_F64_FUNC_AVX512(ge, _CMP_GE_OS, >=)
CMPSC_F64_FUNC_AVX512(le, _CMP_LE_OS, <=)
#undef CMPSC_F64_FUNC_AVX512

/* ── Signed integers ────────────────────────────────────────────── */

#define STAMP_CMPSC_SINT_AVX512(SFX, CT, VPV, SET1, CMP, MASKZ, MASKT) \
  static inline void _cmpsc_eq_##SFX##_avx512(                        \
      const void *restrict ap, const void *restrict sp,                \
      void *restrict op, size_t n) {                                   \
    const CT *a = (const CT *)ap;                                      \
    const CT s = *(const CT *)sp;                                      \
    CT *out = (CT *)op;                                                \
    const __m512i vs = SET1(s);                                        \
    size_t i = 0;                                                      \
    for (; i + (VPV) <= n; i += (VPV)) {                               \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));       \
      MASKT m = CMP(va, vs, _MM_CMPINT_EQ);                           \
      _mm512_storeu_si512((__m512i *)(out + i), MASKZ(m, 1));          \
    }                                                                  \
    for (; i < n; i++)                                                 \
      out[i] = (CT)(a[i] == s);                                       \
  }                                                                    \
  static inline void _cmpsc_gt_##SFX##_avx512(                        \
      const void *restrict ap, const void *restrict sp,                \
      void *restrict op, size_t n) {                                   \
    const CT *a = (const CT *)ap;                                      \
    const CT s = *(const CT *)sp;                                      \
    CT *out = (CT *)op;                                                \
    const __m512i vs = SET1(s);                                        \
    size_t i = 0;                                                      \
    for (; i + (VPV) <= n; i += (VPV)) {                               \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));       \
      MASKT m = CMP(va, vs, _MM_CMPINT_NLE);                          \
      _mm512_storeu_si512((__m512i *)(out + i), MASKZ(m, 1));          \
    }                                                                  \
    for (; i < n; i++)                                                 \
      out[i] = (CT)(a[i] > s);                                        \
  }                                                                    \
  static inline void _cmpsc_lt_##SFX##_avx512(                        \
      const void *restrict ap, const void *restrict sp,                \
      void *restrict op, size_t n) {                                   \
    const CT *a = (const CT *)ap;                                      \
    const CT s = *(const CT *)sp;                                      \
    CT *out = (CT *)op;                                                \
    const __m512i vs = SET1(s);                                        \
    size_t i = 0;                                                      \
    for (; i + (VPV) <= n; i += (VPV)) {                               \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));       \
      MASKT m = CMP(va, vs, _MM_CMPINT_LT);                           \
      _mm512_storeu_si512((__m512i *)(out + i), MASKZ(m, 1));          \
    }                                                                  \
    for (; i < n; i++)                                                 \
      out[i] = (CT)(a[i] < s);                                        \
  }                                                                    \
  static inline void _cmpsc_ge_##SFX##_avx512(                        \
      const void *restrict ap, const void *restrict sp,                \
      void *restrict op, size_t n) {                                   \
    const CT *a = (const CT *)ap;                                      \
    const CT s = *(const CT *)sp;                                      \
    CT *out = (CT *)op;                                                \
    const __m512i vs = SET1(s);                                        \
    size_t i = 0;                                                      \
    for (; i + (VPV) <= n; i += (VPV)) {                               \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));       \
      MASKT m = CMP(va, vs, _MM_CMPINT_NLT);                          \
      _mm512_storeu_si512((__m512i *)(out + i), MASKZ(m, 1));          \
    }                                                                  \
    for (; i < n; i++)                                                 \
      out[i] = (CT)(a[i] >= s);                                       \
  }                                                                    \
  static inline void _cmpsc_le_##SFX##_avx512(                        \
      const void *restrict ap, const void *restrict sp,                \
      void *restrict op, size_t n) {                                   \
    const CT *a = (const CT *)ap;                                      \
    const CT s = *(const CT *)sp;                                      \
    CT *out = (CT *)op;                                                \
    const __m512i vs = SET1(s);                                        \
    size_t i = 0;                                                      \
    for (; i + (VPV) <= n; i += (VPV)) {                               \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));       \
      MASKT m = CMP(va, vs, _MM_CMPINT_LE);                           \
      _mm512_storeu_si512((__m512i *)(out + i), MASKZ(m, 1));          \
    }                                                                  \
    for (; i < n; i++)                                                 \
      out[i] = (CT)(a[i] <= s);                                       \
  }

STAMP_CMPSC_SINT_AVX512(i8, int8_t, 64, _mm512_set1_epi8,
                         _mm512_cmp_epi8_mask, _mm512_maskz_set1_epi8,
                         __mmask64)
STAMP_CMPSC_SINT_AVX512(i16, int16_t, 32, _mm512_set1_epi16,
                         _mm512_cmp_epi16_mask, _mm512_maskz_set1_epi16,
                         __mmask32)
STAMP_CMPSC_SINT_AVX512(i32, int32_t, 16, _mm512_set1_epi32,
                         _mm512_cmp_epi32_mask, _mm512_maskz_set1_epi32,
                         __mmask16)
STAMP_CMPSC_SINT_AVX512(i64, int64_t, 8, _mm512_set1_epi64,
                         _mm512_cmp_epi64_mask, _mm512_maskz_set1_epi64,
                         __mmask8)
#undef STAMP_CMPSC_SINT_AVX512

/* ── Unsigned integers (native AVX-512 unsigned compare) ─────── */

#define STAMP_CMPSC_UINT_AVX512(SFX, CT, VPV, SET1, CMP, MASKZ, MASKT) \
  static inline void _cmpsc_eq_##SFX##_avx512(                        \
      const void *restrict ap, const void *restrict sp,                \
      void *restrict op, size_t n) {                                   \
    const CT *a = (const CT *)ap;                                      \
    const CT s = *(const CT *)sp;                                      \
    CT *out = (CT *)op;                                                \
    const __m512i vs = SET1((CT)s);                                    \
    size_t i = 0;                                                      \
    for (; i + (VPV) <= n; i += (VPV)) {                               \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));       \
      MASKT m = CMP(va, vs, _MM_CMPINT_EQ);                           \
      _mm512_storeu_si512((__m512i *)(out + i), MASKZ(m, 1));          \
    }                                                                  \
    for (; i < n; i++)                                                 \
      out[i] = (CT)(a[i] == s);                                       \
  }                                                                    \
  static inline void _cmpsc_gt_##SFX##_avx512(                        \
      const void *restrict ap, const void *restrict sp,                \
      void *restrict op, size_t n) {                                   \
    const CT *a = (const CT *)ap;                                      \
    const CT s = *(const CT *)sp;                                      \
    CT *out = (CT *)op;                                                \
    const __m512i vs = SET1((CT)s);                                    \
    size_t i = 0;                                                      \
    for (; i + (VPV) <= n; i += (VPV)) {                               \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));       \
      MASKT m = CMP(va, vs, _MM_CMPINT_NLE);                          \
      _mm512_storeu_si512((__m512i *)(out + i), MASKZ(m, 1));          \
    }                                                                  \
    for (; i < n; i++)                                                 \
      out[i] = (CT)(a[i] > s);                                        \
  }                                                                    \
  static inline void _cmpsc_lt_##SFX##_avx512(                        \
      const void *restrict ap, const void *restrict sp,                \
      void *restrict op, size_t n) {                                   \
    const CT *a = (const CT *)ap;                                      \
    const CT s = *(const CT *)sp;                                      \
    CT *out = (CT *)op;                                                \
    const __m512i vs = SET1((CT)s);                                    \
    size_t i = 0;                                                      \
    for (; i + (VPV) <= n; i += (VPV)) {                               \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));       \
      MASKT m = CMP(va, vs, _MM_CMPINT_LT);                           \
      _mm512_storeu_si512((__m512i *)(out + i), MASKZ(m, 1));          \
    }                                                                  \
    for (; i < n; i++)                                                 \
      out[i] = (CT)(a[i] < s);                                        \
  }                                                                    \
  static inline void _cmpsc_ge_##SFX##_avx512(                        \
      const void *restrict ap, const void *restrict sp,                \
      void *restrict op, size_t n) {                                   \
    const CT *a = (const CT *)ap;                                      \
    const CT s = *(const CT *)sp;                                      \
    CT *out = (CT *)op;                                                \
    const __m512i vs = SET1((CT)s);                                    \
    size_t i = 0;                                                      \
    for (; i + (VPV) <= n; i += (VPV)) {                               \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));       \
      MASKT m = CMP(va, vs, _MM_CMPINT_NLT);                          \
      _mm512_storeu_si512((__m512i *)(out + i), MASKZ(m, 1));          \
    }                                                                  \
    for (; i < n; i++)                                                 \
      out[i] = (CT)(a[i] >= s);                                       \
  }                                                                    \
  static inline void _cmpsc_le_##SFX##_avx512(                        \
      const void *restrict ap, const void *restrict sp,                \
      void *restrict op, size_t n) {                                   \
    const CT *a = (const CT *)ap;                                      \
    const CT s = *(const CT *)sp;                                      \
    CT *out = (CT *)op;                                                \
    const __m512i vs = SET1((CT)s);                                    \
    size_t i = 0;                                                      \
    for (; i + (VPV) <= n; i += (VPV)) {                               \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));       \
      MASKT m = CMP(va, vs, _MM_CMPINT_LE);                           \
      _mm512_storeu_si512((__m512i *)(out + i), MASKZ(m, 1));          \
    }                                                                  \
    for (; i < n; i++)                                                 \
      out[i] = (CT)(a[i] <= s);                                       \
  }

STAMP_CMPSC_UINT_AVX512(u8, uint8_t, 64, _mm512_set1_epi8,
                         _mm512_cmp_epu8_mask, _mm512_maskz_set1_epi8,
                         __mmask64)
STAMP_CMPSC_UINT_AVX512(u16, uint16_t, 32, _mm512_set1_epi16,
                         _mm512_cmp_epu16_mask, _mm512_maskz_set1_epi16,
                         __mmask32)
STAMP_CMPSC_UINT_AVX512(u32, uint32_t, 16, _mm512_set1_epi32,
                         _mm512_cmp_epu32_mask, _mm512_maskz_set1_epi32,
                         __mmask16)
STAMP_CMPSC_UINT_AVX512(u64, uint64_t, 8, _mm512_set1_epi64,
                         _mm512_cmp_epu64_mask, _mm512_maskz_set1_epi64,
                         __mmask8)
#undef STAMP_CMPSC_UINT_AVX512

#endif /* NUMC_COMPARE_SCALAR_AVX512_H */
