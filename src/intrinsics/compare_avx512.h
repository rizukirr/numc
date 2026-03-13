/**
 * @file compare_avx512.h
 * @brief AVX-512 binary comparison kernels for all 10 types.
 *
 * Produces 0 or 1 (same type as input) per element.
 * AVX-512 has native signed and unsigned integer comparisons via mask
 * registers — no XOR sign-bit bias needed.
 * Float: _mm512_cmp_ps_mask/_mm512_cmp_pd_mask with ordered predicates,
 * then _mm512_maskz_set1_ps/pd to produce 0.0/1.0.
 */
#ifndef NUMC_COMPARE_AVX512_H
#define NUMC_COMPARE_AVX512_H

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

/* ════════════════════════════════════════════════════════════════════
 * Signed integer comparisons (native _mm512_cmp_epi mask)
 * ════════════════════════════════════════════════════════════════ */

#define FAST_CMP_SINT_AVX512(SFX, CT, VPV, CMP, MASKZ_SET1, LOAD,       \
                              STORE)                                      \
  static inline void _fast_eq_##SFX##_avx512(                            \
      const void *restrict ap, const void *restrict bp,                   \
      void *restrict op, size_t n) {                                      \
    const CT *a = (const CT *)ap;                                         \
    const CT *b = (const CT *)bp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t i = 0;                                                         \
    for (; i + (VPV) <= n; i += (VPV)) {                                  \
      __m512i va = LOAD((const __m512i *)(a + i));                        \
      __m512i vb = LOAD((const __m512i *)(b + i));                        \
      __mmask64 m = CMP(va, vb, _MM_CMPINT_EQ);                          \
      STORE((__m512i *)(out + i), MASKZ_SET1(m, (CT)1));                  \
    }                                                                     \
    for (; i < n; i++)                                                    \
      out[i] = (CT)(a[i] == b[i]);                                       \
  }                                                                       \
  static inline void _fast_gt_##SFX##_avx512(                            \
      const void *restrict ap, const void *restrict bp,                   \
      void *restrict op, size_t n) {                                      \
    const CT *a = (const CT *)ap;                                         \
    const CT *b = (const CT *)bp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t i = 0;                                                         \
    for (; i + (VPV) <= n; i += (VPV)) {                                  \
      __m512i va = LOAD((const __m512i *)(a + i));                        \
      __m512i vb = LOAD((const __m512i *)(b + i));                        \
      __mmask64 m = CMP(va, vb, _MM_CMPINT_NLE);                         \
      STORE((__m512i *)(out + i), MASKZ_SET1(m, (CT)1));                  \
    }                                                                     \
    for (; i < n; i++)                                                    \
      out[i] = (CT)(a[i] > b[i]);                                        \
  }                                                                       \
  static inline void _fast_lt_##SFX##_avx512(                            \
      const void *restrict ap, const void *restrict bp,                   \
      void *restrict op, size_t n) {                                      \
    const CT *a = (const CT *)ap;                                         \
    const CT *b = (const CT *)bp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t i = 0;                                                         \
    for (; i + (VPV) <= n; i += (VPV)) {                                  \
      __m512i va = LOAD((const __m512i *)(a + i));                        \
      __m512i vb = LOAD((const __m512i *)(b + i));                        \
      __mmask64 m = CMP(va, vb, _MM_CMPINT_LT);                          \
      STORE((__m512i *)(out + i), MASKZ_SET1(m, (CT)1));                  \
    }                                                                     \
    for (; i < n; i++)                                                    \
      out[i] = (CT)(a[i] < b[i]);                                        \
  }                                                                       \
  static inline void _fast_ge_##SFX##_avx512(                            \
      const void *restrict ap, const void *restrict bp,                   \
      void *restrict op, size_t n) {                                      \
    const CT *a = (const CT *)ap;                                         \
    const CT *b = (const CT *)bp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t i = 0;                                                         \
    for (; i + (VPV) <= n; i += (VPV)) {                                  \
      __m512i va = LOAD((const __m512i *)(a + i));                        \
      __m512i vb = LOAD((const __m512i *)(b + i));                        \
      __mmask64 m = CMP(va, vb, _MM_CMPINT_NLT);                         \
      STORE((__m512i *)(out + i), MASKZ_SET1(m, (CT)1));                  \
    }                                                                     \
    for (; i < n; i++)                                                    \
      out[i] = (CT)(a[i] >= b[i]);                                       \
  }                                                                       \
  static inline void _fast_le_##SFX##_avx512(                            \
      const void *restrict ap, const void *restrict bp,                   \
      void *restrict op, size_t n) {                                      \
    const CT *a = (const CT *)ap;                                         \
    const CT *b = (const CT *)bp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t i = 0;                                                         \
    for (; i + (VPV) <= n; i += (VPV)) {                                  \
      __m512i va = LOAD((const __m512i *)(a + i));                        \
      __m512i vb = LOAD((const __m512i *)(b + i));                        \
      __mmask64 m = CMP(va, vb, _MM_CMPINT_LE);                          \
      STORE((__m512i *)(out + i), MASKZ_SET1(m, (CT)1));                  \
    }                                                                     \
    for (; i < n; i++)                                                    \
      out[i] = (CT)(a[i] <= b[i]);                                       \
  }

FAST_CMP_SINT_AVX512(i8, int8_t, 64, _mm512_cmp_epi8_mask,
                      _mm512_maskz_set1_epi8, _mm512_loadu_si512,
                      _mm512_storeu_si512)
FAST_CMP_SINT_AVX512(i16, int16_t, 32, _mm512_cmp_epi16_mask,
                      _mm512_maskz_set1_epi16, _mm512_loadu_si512,
                      _mm512_storeu_si512)
FAST_CMP_SINT_AVX512(i32, int32_t, 16, _mm512_cmp_epi32_mask,
                      _mm512_maskz_set1_epi32, _mm512_loadu_si512,
                      _mm512_storeu_si512)
FAST_CMP_SINT_AVX512(i64, int64_t, 8, _mm512_cmp_epi64_mask,
                      _mm512_maskz_set1_epi64, _mm512_loadu_si512,
                      _mm512_storeu_si512)
#undef FAST_CMP_SINT_AVX512

/* ════════════════════════════════════════════════════════════════════
 * Unsigned integer comparisons (native _mm512_cmp_epu mask)
 * ════════════════════════════════════════════════════════════════ */

#define FAST_CMP_UINT_AVX512(SFX, CT, VPV, CMP, MASKZ_SET1, LOAD,       \
                              STORE)                                      \
  static inline void _fast_eq_##SFX##_avx512(                            \
      const void *restrict ap, const void *restrict bp,                   \
      void *restrict op, size_t n) {                                      \
    const CT *a = (const CT *)ap;                                         \
    const CT *b = (const CT *)bp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t i = 0;                                                         \
    for (; i + (VPV) <= n; i += (VPV)) {                                  \
      __m512i va = LOAD((const __m512i *)(a + i));                        \
      __m512i vb = LOAD((const __m512i *)(b + i));                        \
      __mmask64 m = CMP(va, vb, _MM_CMPINT_EQ);                          \
      STORE((__m512i *)(out + i), MASKZ_SET1(m, (CT)1));                  \
    }                                                                     \
    for (; i < n; i++)                                                    \
      out[i] = (CT)(a[i] == b[i]);                                       \
  }                                                                       \
  static inline void _fast_gt_##SFX##_avx512(                            \
      const void *restrict ap, const void *restrict bp,                   \
      void *restrict op, size_t n) {                                      \
    const CT *a = (const CT *)ap;                                         \
    const CT *b = (const CT *)bp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t i = 0;                                                         \
    for (; i + (VPV) <= n; i += (VPV)) {                                  \
      __m512i va = LOAD((const __m512i *)(a + i));                        \
      __m512i vb = LOAD((const __m512i *)(b + i));                        \
      __mmask64 m = CMP(va, vb, _MM_CMPINT_NLE);                         \
      STORE((__m512i *)(out + i), MASKZ_SET1(m, (CT)1));                  \
    }                                                                     \
    for (; i < n; i++)                                                    \
      out[i] = (CT)(a[i] > b[i]);                                        \
  }                                                                       \
  static inline void _fast_lt_##SFX##_avx512(                            \
      const void *restrict ap, const void *restrict bp,                   \
      void *restrict op, size_t n) {                                      \
    const CT *a = (const CT *)ap;                                         \
    const CT *b = (const CT *)bp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t i = 0;                                                         \
    for (; i + (VPV) <= n; i += (VPV)) {                                  \
      __m512i va = LOAD((const __m512i *)(a + i));                        \
      __m512i vb = LOAD((const __m512i *)(b + i));                        \
      __mmask64 m = CMP(va, vb, _MM_CMPINT_LT);                          \
      STORE((__m512i *)(out + i), MASKZ_SET1(m, (CT)1));                  \
    }                                                                     \
    for (; i < n; i++)                                                    \
      out[i] = (CT)(a[i] < b[i]);                                        \
  }                                                                       \
  static inline void _fast_ge_##SFX##_avx512(                            \
      const void *restrict ap, const void *restrict bp,                   \
      void *restrict op, size_t n) {                                      \
    const CT *a = (const CT *)ap;                                         \
    const CT *b = (const CT *)bp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t i = 0;                                                         \
    for (; i + (VPV) <= n; i += (VPV)) {                                  \
      __m512i va = LOAD((const __m512i *)(a + i));                        \
      __m512i vb = LOAD((const __m512i *)(b + i));                        \
      __mmask64 m = CMP(va, vb, _MM_CMPINT_NLT);                         \
      STORE((__m512i *)(out + i), MASKZ_SET1(m, (CT)1));                  \
    }                                                                     \
    for (; i < n; i++)                                                    \
      out[i] = (CT)(a[i] >= b[i]);                                       \
  }                                                                       \
  static inline void _fast_le_##SFX##_avx512(                            \
      const void *restrict ap, const void *restrict bp,                   \
      void *restrict op, size_t n) {                                      \
    const CT *a = (const CT *)ap;                                         \
    const CT *b = (const CT *)bp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t i = 0;                                                         \
    for (; i + (VPV) <= n; i += (VPV)) {                                  \
      __m512i va = LOAD((const __m512i *)(a + i));                        \
      __m512i vb = LOAD((const __m512i *)(b + i));                        \
      __mmask64 m = CMP(va, vb, _MM_CMPINT_LE);                          \
      STORE((__m512i *)(out + i), MASKZ_SET1(m, (CT)1));                  \
    }                                                                     \
    for (; i < n; i++)                                                    \
      out[i] = (CT)(a[i] <= b[i]);                                       \
  }

FAST_CMP_UINT_AVX512(u8, uint8_t, 64, _mm512_cmp_epu8_mask,
                      _mm512_maskz_set1_epi8, _mm512_loadu_si512,
                      _mm512_storeu_si512)
FAST_CMP_UINT_AVX512(u16, uint16_t, 32, _mm512_cmp_epu16_mask,
                      _mm512_maskz_set1_epi16, _mm512_loadu_si512,
                      _mm512_storeu_si512)
FAST_CMP_UINT_AVX512(u32, uint32_t, 16, _mm512_cmp_epu32_mask,
                      _mm512_maskz_set1_epi32, _mm512_loadu_si512,
                      _mm512_storeu_si512)
FAST_CMP_UINT_AVX512(u64, uint64_t, 8, _mm512_cmp_epu64_mask,
                      _mm512_maskz_set1_epi64, _mm512_loadu_si512,
                      _mm512_storeu_si512)
#undef FAST_CMP_UINT_AVX512

/* ════════════════════════════════════════════════════════════════════
 * Float comparisons (f32 — 16 per vector)
 * ════════════════════════════════════════════════════════════════ */

#define FAST_CMP_F32_AVX512(OP, PRED, TAIL_OP)                           \
  static inline void _fast_##OP##_f32_avx512(                            \
      const void *restrict ap, const void *restrict bp,                   \
      void *restrict op, size_t n) {                                      \
    const float *a = (const float *)ap;                                   \
    const float *b = (const float *)bp;                                   \
    float *out = (float *)op;                                             \
    const __m512 one = _mm512_set1_ps(1.0f);                              \
    size_t i = 0;                                                         \
    for (; i + 16 <= n; i += 16) {                                        \
      __m512 va = _mm512_loadu_ps(a + i);                                 \
      __m512 vb = _mm512_loadu_ps(b + i);                                 \
      __mmask16 m = _mm512_cmp_ps_mask(va, vb, PRED);                    \
      _mm512_storeu_ps(out + i, _mm512_maskz_mov_ps(m, one));            \
    }                                                                     \
    for (; i < n; i++)                                                    \
      out[i] = (float)(a[i] TAIL_OP b[i]);                               \
  }

FAST_CMP_F32_AVX512(eq, _CMP_EQ_OQ, ==)
FAST_CMP_F32_AVX512(gt, _CMP_GT_OQ, >)
FAST_CMP_F32_AVX512(lt, _CMP_LT_OQ, <)
FAST_CMP_F32_AVX512(ge, _CMP_GE_OQ, >=)
FAST_CMP_F32_AVX512(le, _CMP_LE_OQ, <=)
#undef FAST_CMP_F32_AVX512

/* ════════════════════════════════════════════════════════════════════
 * Float comparisons (f64 — 8 per vector)
 * ════════════════════════════════════════════════════════════════ */

#define FAST_CMP_F64_AVX512(OP, PRED, TAIL_OP)                           \
  static inline void _fast_##OP##_f64_avx512(                            \
      const void *restrict ap, const void *restrict bp,                   \
      void *restrict op, size_t n) {                                      \
    const double *a = (const double *)ap;                                 \
    const double *b = (const double *)bp;                                 \
    double *out = (double *)op;                                           \
    const __m512d one = _mm512_set1_pd(1.0);                              \
    size_t i = 0;                                                         \
    for (; i + 8 <= n; i += 8) {                                          \
      __m512d va = _mm512_loadu_pd(a + i);                                \
      __m512d vb = _mm512_loadu_pd(b + i);                                \
      __mmask8 m = _mm512_cmp_pd_mask(va, vb, PRED);                     \
      _mm512_storeu_pd(out + i, _mm512_maskz_mov_pd(m, one));            \
    }                                                                     \
    for (; i < n; i++)                                                    \
      out[i] = (double)(a[i] TAIL_OP b[i]);                              \
  }

FAST_CMP_F64_AVX512(eq, _CMP_EQ_OQ, ==)
FAST_CMP_F64_AVX512(gt, _CMP_GT_OQ, >)
FAST_CMP_F64_AVX512(lt, _CMP_LT_OQ, <)
FAST_CMP_F64_AVX512(ge, _CMP_GE_OQ, >=)
FAST_CMP_F64_AVX512(le, _CMP_LE_OQ, <=)
#undef FAST_CMP_F64_AVX512

#endif /* NUMC_COMPARE_AVX512_H */
