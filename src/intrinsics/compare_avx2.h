/**
 * @file compare_avx2.h
 * @brief AVX2 binary comparison kernels for all 10 types.
 *
 * Produces 0 or 1 (same type as input) per element.
 * Signed int: native cmpeq/cmpgt.
 * Unsigned int: XOR sign-bit bias + signed compare.
 * Float: _mm256_cmp_ps/pd with ordered predicates, AND with 1.0.
 */
#ifndef NUMC_COMPARE_AVX2_H
#define NUMC_COMPARE_AVX2_H

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

/* ════════════════════════════════════════════════════════════════════
 * Signed integer comparisons (native cmpeq/cmpgt)
 * ════════════════════════════════════════════════════════════════ */

#define FAST_CMP_SINT_AVX2(SFX, CT, VPV, CMPEQ, CMPGT, SET1)         \
  static inline void _fast_eq_##SFX##_avx2(                           \
      const void *restrict ap, const void *restrict bp,                \
      void *restrict op, size_t n) {                                   \
    const CT *a = (const CT *)ap;                                      \
    const CT *b = (const CT *)bp;                                      \
    CT *out = (CT *)op;                                                \
    const __m256i one = SET1((CT)1);                                   \
    size_t i = 0;                                                      \
    for (; i + (VPV) <= n; i += (VPV)) {                               \
      __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));       \
      __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));       \
      _mm256_storeu_si256((__m256i *)(out + i),                        \
                          _mm256_and_si256(CMPEQ(va, vb), one));       \
    }                                                                  \
    for (; i < n; i++)                                                 \
      out[i] = (CT)(a[i] == b[i]);                                    \
  }                                                                    \
  static inline void _fast_gt_##SFX##_avx2(                           \
      const void *restrict ap, const void *restrict bp,                \
      void *restrict op, size_t n) {                                   \
    const CT *a = (const CT *)ap;                                      \
    const CT *b = (const CT *)bp;                                      \
    CT *out = (CT *)op;                                                \
    const __m256i one = SET1((CT)1);                                   \
    size_t i = 0;                                                      \
    for (; i + (VPV) <= n; i += (VPV)) {                               \
      __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));       \
      __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));       \
      _mm256_storeu_si256((__m256i *)(out + i),                        \
                          _mm256_and_si256(CMPGT(va, vb), one));       \
    }                                                                  \
    for (; i < n; i++)                                                 \
      out[i] = (CT)(a[i] > b[i]);                                     \
  }                                                                    \
  static inline void _fast_lt_##SFX##_avx2(                           \
      const void *restrict ap, const void *restrict bp,                \
      void *restrict op, size_t n) {                                   \
    _fast_gt_##SFX##_avx2(bp, ap, op, n);                              \
  }                                                                    \
  static inline void _fast_ge_##SFX##_avx2(                           \
      const void *restrict ap, const void *restrict bp,                \
      void *restrict op, size_t n) {                                   \
    const CT *a = (const CT *)ap;                                      \
    const CT *b = (const CT *)bp;                                      \
    CT *out = (CT *)op;                                                \
    const __m256i one = SET1((CT)1);                                   \
    size_t i = 0;                                                      \
    for (; i + (VPV) <= n; i += (VPV)) {                               \
      __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));       \
      __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));       \
      _mm256_storeu_si256(                                             \
          (__m256i *)(out + i),                                        \
          _mm256_andnot_si256(CMPGT(vb, va), one));                    \
    }                                                                  \
    for (; i < n; i++)                                                 \
      out[i] = (CT)(a[i] >= b[i]);                                    \
  }                                                                    \
  static inline void _fast_le_##SFX##_avx2(                           \
      const void *restrict ap, const void *restrict bp,                \
      void *restrict op, size_t n) {                                   \
    _fast_ge_##SFX##_avx2(bp, ap, op, n);                              \
  }

FAST_CMP_SINT_AVX2(i8, int8_t, 32, _mm256_cmpeq_epi8,
                    _mm256_cmpgt_epi8, _mm256_set1_epi8)
FAST_CMP_SINT_AVX2(i16, int16_t, 16, _mm256_cmpeq_epi16,
                    _mm256_cmpgt_epi16, _mm256_set1_epi16)
FAST_CMP_SINT_AVX2(i32, int32_t, 8, _mm256_cmpeq_epi32,
                    _mm256_cmpgt_epi32, _mm256_set1_epi32)
FAST_CMP_SINT_AVX2(i64, int64_t, 4, _mm256_cmpeq_epi64,
                    _mm256_cmpgt_epi64, _mm256_set1_epi64x)
#undef FAST_CMP_SINT_AVX2

/* ════════════════════════════════════════════════════════════════════
 * Unsigned integer comparisons (XOR sign-bit bias + signed compare)
 * ════════════════════════════════════════════════════════════════ */

#define FAST_CMP_UINT_AVX2(SFX, CT, VPV, CMPEQ, CMPGT, SET1, BIAS)   \
  static inline void _fast_eq_##SFX##_avx2(                           \
      const void *restrict ap, const void *restrict bp,                \
      void *restrict op, size_t n) {                                   \
    const CT *a = (const CT *)ap;                                      \
    const CT *b = (const CT *)bp;                                      \
    CT *out = (CT *)op;                                                \
    const __m256i one = SET1((CT)1);                                   \
    size_t i = 0;                                                      \
    for (; i + (VPV) <= n; i += (VPV)) {                               \
      __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));       \
      __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));       \
      _mm256_storeu_si256((__m256i *)(out + i),                        \
                          _mm256_and_si256(CMPEQ(va, vb), one));       \
    }                                                                  \
    for (; i < n; i++)                                                 \
      out[i] = (CT)(a[i] == b[i]);                                    \
  }                                                                    \
  static inline void _fast_gt_##SFX##_avx2(                           \
      const void *restrict ap, const void *restrict bp,                \
      void *restrict op, size_t n) {                                   \
    const CT *a = (const CT *)ap;                                      \
    const CT *b = (const CT *)bp;                                      \
    CT *out = (CT *)op;                                                \
    const __m256i bias = BIAS;                                         \
    const __m256i one = SET1((CT)1);                                   \
    size_t i = 0;                                                      \
    for (; i + (VPV) <= n; i += (VPV)) {                               \
      __m256i va = _mm256_xor_si256(                                   \
          _mm256_loadu_si256((const __m256i *)(a + i)), bias);         \
      __m256i vb = _mm256_xor_si256(                                   \
          _mm256_loadu_si256((const __m256i *)(b + i)), bias);         \
      _mm256_storeu_si256((__m256i *)(out + i),                        \
                          _mm256_and_si256(CMPGT(va, vb), one));       \
    }                                                                  \
    for (; i < n; i++)                                                 \
      out[i] = (CT)(a[i] > b[i]);                                     \
  }                                                                    \
  static inline void _fast_lt_##SFX##_avx2(                           \
      const void *restrict ap, const void *restrict bp,                \
      void *restrict op, size_t n) {                                   \
    _fast_gt_##SFX##_avx2(bp, ap, op, n);                              \
  }                                                                    \
  static inline void _fast_ge_##SFX##_avx2(                           \
      const void *restrict ap, const void *restrict bp,                \
      void *restrict op, size_t n) {                                   \
    const CT *a = (const CT *)ap;                                      \
    const CT *b = (const CT *)bp;                                      \
    CT *out = (CT *)op;                                                \
    const __m256i bias = BIAS;                                         \
    const __m256i one = SET1((CT)1);                                   \
    size_t i = 0;                                                      \
    for (; i + (VPV) <= n; i += (VPV)) {                               \
      __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));       \
      __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));       \
      __m256i ba = _mm256_xor_si256(va, bias);                         \
      __m256i bb = _mm256_xor_si256(vb, bias);                         \
      _mm256_storeu_si256(                                             \
          (__m256i *)(out + i),                                        \
          _mm256_andnot_si256(CMPGT(bb, ba), one));                    \
    }                                                                  \
    for (; i < n; i++)                                                 \
      out[i] = (CT)(a[i] >= b[i]);                                    \
  }                                                                    \
  static inline void _fast_le_##SFX##_avx2(                           \
      const void *restrict ap, const void *restrict bp,                \
      void *restrict op, size_t n) {                                   \
    _fast_ge_##SFX##_avx2(bp, ap, op, n);                              \
  }

FAST_CMP_UINT_AVX2(u8, uint8_t, 32, _mm256_cmpeq_epi8,
                    _mm256_cmpgt_epi8, _mm256_set1_epi8,
                    _mm256_set1_epi8((char)0x80))
FAST_CMP_UINT_AVX2(u16, uint16_t, 16, _mm256_cmpeq_epi16,
                    _mm256_cmpgt_epi16, _mm256_set1_epi16,
                    _mm256_set1_epi16((short)0x8000))
FAST_CMP_UINT_AVX2(u32, uint32_t, 8, _mm256_cmpeq_epi32,
                    _mm256_cmpgt_epi32, _mm256_set1_epi32,
                    _mm256_set1_epi32((int)0x80000000))
FAST_CMP_UINT_AVX2(u64, uint64_t, 4, _mm256_cmpeq_epi64,
                    _mm256_cmpgt_epi64, _mm256_set1_epi64x,
                    _mm256_set1_epi64x((long long)0x8000000000000000LL))
#undef FAST_CMP_UINT_AVX2

/* ════════════════════════════════════════════════════════════════════
 * Float comparisons
 * ════════════════════════════════════════════════════════════════ */

#define FAST_CMP_F32_AVX2(OP, PRED, TAIL_OP)                          \
  static inline void _fast_##OP##_f32_avx2(                           \
      const void *restrict ap, const void *restrict bp,                \
      void *restrict op, size_t n) {                                   \
    const float *a = (const float *)ap;                                \
    const float *b = (const float *)bp;                                \
    float *out = (float *)op;                                          \
    const __m256 one = _mm256_set1_ps(1.0f);                           \
    size_t i = 0;                                                      \
    for (; i + 8 <= n; i += 8) {                                       \
      __m256 va = _mm256_loadu_ps(a + i);                              \
      __m256 vb = _mm256_loadu_ps(b + i);                              \
      _mm256_storeu_ps(out + i,                                        \
                       _mm256_and_ps(_mm256_cmp_ps(va, vb, PRED),      \
                                    one));                             \
    }                                                                  \
    for (; i < n; i++)                                                 \
      out[i] = (float)(a[i] TAIL_OP b[i]);                            \
  }

FAST_CMP_F32_AVX2(eq, _CMP_EQ_OQ, ==)
FAST_CMP_F32_AVX2(gt, _CMP_GT_OQ, >)
FAST_CMP_F32_AVX2(lt, _CMP_LT_OQ, <)
FAST_CMP_F32_AVX2(ge, _CMP_GE_OQ, >=)
FAST_CMP_F32_AVX2(le, _CMP_LE_OQ, <=)
#undef FAST_CMP_F32_AVX2

#define FAST_CMP_F64_AVX2(OP, PRED, TAIL_OP)                          \
  static inline void _fast_##OP##_f64_avx2(                           \
      const void *restrict ap, const void *restrict bp,                \
      void *restrict op, size_t n) {                                   \
    const double *a = (const double *)ap;                              \
    const double *b = (const double *)bp;                              \
    double *out = (double *)op;                                        \
    const __m256d one = _mm256_set1_pd(1.0);                           \
    size_t i = 0;                                                      \
    for (; i + 4 <= n; i += 4) {                                       \
      __m256d va = _mm256_loadu_pd(a + i);                             \
      __m256d vb = _mm256_loadu_pd(b + i);                             \
      _mm256_storeu_pd(out + i,                                        \
                       _mm256_and_pd(_mm256_cmp_pd(va, vb, PRED),      \
                                    one));                             \
    }                                                                  \
    for (; i < n; i++)                                                 \
      out[i] = (double)(a[i] TAIL_OP b[i]);                           \
  }

FAST_CMP_F64_AVX2(eq, _CMP_EQ_OQ, ==)
FAST_CMP_F64_AVX2(gt, _CMP_GT_OQ, >)
FAST_CMP_F64_AVX2(lt, _CMP_LT_OQ, <)
FAST_CMP_F64_AVX2(ge, _CMP_GE_OQ, >=)
FAST_CMP_F64_AVX2(le, _CMP_LE_OQ, <=)
#undef FAST_CMP_F64_AVX2

#endif /* NUMC_COMPARE_AVX2_H */
