/**
 * @file compare_scalar_avx2.h
 * @brief AVX2 scalar comparison kernels for all 10 types.
 *
 * Each function compares array elements against a broadcast scalar,
 * storing 0 or 1 (in the element's own type) into the output.
 *
 * Signed integers: cmpeq/cmpgt directly.
 * Unsigned integers: XOR sign-bit bias, then signed compare.
 * Floats: _mm256_cmp_ps/pd with IMM predicates.
 */
#ifndef NUMC_COMPARE_SCALAR_AVX2_H
#define NUMC_COMPARE_SCALAR_AVX2_H

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

/* ── Float: f32 (8 per vector) ──────────────────────────────────── */

#define STAMP_CMPSC_F32_AVX2(NAME, IMM)                                      \
  static inline void _cmpsc_##NAME##_f32_avx2(const void *restrict ap,       \
                                              const void *restrict sp,       \
                                              void *restrict op, size_t n) { \
    const float *a = (const float *)ap;                                      \
    const float s = *(const float *)sp;                                      \
    float *out = (float *)op;                                                \
    const __m256 vs = _mm256_set1_ps(s);                                     \
    const __m256 one = _mm256_set1_ps(1.0f);                                 \
    size_t i = 0;                                                            \
    for (; i + 8 <= n; i += 8) {                                             \
      __m256 va = _mm256_loadu_ps(a + i);                                    \
      __m256 mask = _mm256_cmp_ps(va, vs, IMM);                              \
      _mm256_storeu_ps(out + i, _mm256_and_ps(mask, one));                   \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (float)(a[i] == s && (IMM) == 0) /* placeholder */;           \
  }

/* Can't use generic scalar tail with IMM, so write each explicitly: */
#undef STAMP_CMPSC_F32_AVX2

#define CMPSC_F32_FUNC(NAME, IMM, SCALAR_OP)                                 \
  static inline void _cmpsc_##NAME##_f32_avx2(const void *restrict ap,       \
                                              const void *restrict sp,       \
                                              void *restrict op, size_t n) { \
    const float *a = (const float *)ap;                                      \
    const float s = *(const float *)sp;                                      \
    float *out = (float *)op;                                                \
    const __m256 vs = _mm256_set1_ps(s);                                     \
    const __m256 one = _mm256_set1_ps(1.0f);                                 \
    size_t i = 0;                                                            \
    for (; i + 8 <= n; i += 8) {                                             \
      __m256 va = _mm256_loadu_ps(a + i);                                    \
      __m256 mask = _mm256_cmp_ps(va, vs, (IMM));                            \
      _mm256_storeu_ps(out + i, _mm256_and_ps(mask, one));                   \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (float)(a[i] SCALAR_OP s);                                    \
  }

CMPSC_F32_FUNC(eq, _CMP_EQ_OQ, ==)
CMPSC_F32_FUNC(gt, _CMP_GT_OS, >)
CMPSC_F32_FUNC(lt, _CMP_LT_OS, <)
CMPSC_F32_FUNC(ge, _CMP_GE_OS, >=)
CMPSC_F32_FUNC(le, _CMP_LE_OS, <=)
#undef CMPSC_F32_FUNC

/* ── Float: f64 (4 per vector) ──────────────────────────────────── */

#define CMPSC_F64_FUNC(NAME, IMM, SCALAR_OP)                                 \
  static inline void _cmpsc_##NAME##_f64_avx2(const void *restrict ap,       \
                                              const void *restrict sp,       \
                                              void *restrict op, size_t n) { \
    const double *a = (const double *)ap;                                    \
    const double s = *(const double *)sp;                                    \
    double *out = (double *)op;                                              \
    const __m256d vs = _mm256_set1_pd(s);                                    \
    const __m256d one = _mm256_set1_pd(1.0);                                 \
    size_t i = 0;                                                            \
    for (; i + 4 <= n; i += 4) {                                             \
      __m256d va = _mm256_loadu_pd(a + i);                                   \
      __m256d mask = _mm256_cmp_pd(va, vs, (IMM));                           \
      _mm256_storeu_pd(out + i, _mm256_and_pd(mask, one));                   \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (double)(a[i] SCALAR_OP s);                                   \
  }

CMPSC_F64_FUNC(eq, _CMP_EQ_OQ, ==)
CMPSC_F64_FUNC(gt, _CMP_GT_OS, >)
CMPSC_F64_FUNC(lt, _CMP_LT_OS, <)
CMPSC_F64_FUNC(ge, _CMP_GE_OS, >=)
CMPSC_F64_FUNC(le, _CMP_LE_OS, <=)
#undef CMPSC_F64_FUNC

/* ── Signed integers ────────────────────────────────────────────── */

#define STAMP_CMPSC_SINT_AVX2(SFX, CT, VPV, SET1, CMPEQ, CMPGT)            \
  static inline void _cmpsc_eq_##SFX##_avx2(const void *restrict ap,       \
                                            const void *restrict sp,       \
                                            void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                          \
    const CT s = *(const CT *)sp;                                          \
    CT *out = (CT *)op;                                                    \
    const __m256i vs = SET1(s);                                            \
    const __m256i one = SET1(1);                                           \
    size_t i = 0;                                                          \
    for (; i + (VPV) <= n; i += (VPV)) {                                   \
      __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));           \
      __m256i r = _mm256_and_si256(CMPEQ(va, vs), one);                    \
      _mm256_storeu_si256((__m256i *)(out + i), r);                        \
    }                                                                      \
    for (; i < n; i++)                                                     \
      out[i] = (CT)(a[i] == s);                                            \
  }                                                                        \
  static inline void _cmpsc_gt_##SFX##_avx2(const void *restrict ap,       \
                                            const void *restrict sp,       \
                                            void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                          \
    const CT s = *(const CT *)sp;                                          \
    CT *out = (CT *)op;                                                    \
    const __m256i vs = SET1(s);                                            \
    const __m256i one = SET1(1);                                           \
    size_t i = 0;                                                          \
    for (; i + (VPV) <= n; i += (VPV)) {                                   \
      __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));           \
      __m256i r = _mm256_and_si256(CMPGT(va, vs), one);                    \
      _mm256_storeu_si256((__m256i *)(out + i), r);                        \
    }                                                                      \
    for (; i < n; i++)                                                     \
      out[i] = (CT)(a[i] > s);                                             \
  }                                                                        \
  static inline void _cmpsc_lt_##SFX##_avx2(const void *restrict ap,       \
                                            const void *restrict sp,       \
                                            void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                          \
    const CT s = *(const CT *)sp;                                          \
    CT *out = (CT *)op;                                                    \
    const __m256i vs = SET1(s);                                            \
    const __m256i one = SET1(1);                                           \
    size_t i = 0;                                                          \
    for (; i + (VPV) <= n; i += (VPV)) {                                   \
      __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));           \
      __m256i r = _mm256_and_si256(CMPGT(vs, va), one);                    \
      _mm256_storeu_si256((__m256i *)(out + i), r);                        \
    }                                                                      \
    for (; i < n; i++)                                                     \
      out[i] = (CT)(a[i] < s);                                             \
  }                                                                        \
  static inline void _cmpsc_ge_##SFX##_avx2(const void *restrict ap,       \
                                            const void *restrict sp,       \
                                            void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                          \
    const CT s = *(const CT *)sp;                                          \
    CT *out = (CT *)op;                                                    \
    const __m256i vs = SET1(s);                                            \
    const __m256i one = SET1(1);                                           \
    size_t i = 0;                                                          \
    for (; i + (VPV) <= n; i += (VPV)) {                                   \
      __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));           \
      __m256i eq = CMPEQ(va, vs);                                          \
      __m256i gt = CMPGT(va, vs);                                          \
      __m256i r = _mm256_and_si256(_mm256_or_si256(eq, gt), one);          \
      _mm256_storeu_si256((__m256i *)(out + i), r);                        \
    }                                                                      \
    for (; i < n; i++)                                                     \
      out[i] = (CT)(a[i] >= s);                                            \
  }                                                                        \
  static inline void _cmpsc_le_##SFX##_avx2(const void *restrict ap,       \
                                            const void *restrict sp,       \
                                            void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                          \
    const CT s = *(const CT *)sp;                                          \
    CT *out = (CT *)op;                                                    \
    const __m256i vs = SET1(s);                                            \
    const __m256i one = SET1(1);                                           \
    size_t i = 0;                                                          \
    for (; i + (VPV) <= n; i += (VPV)) {                                   \
      __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));           \
      __m256i eq = CMPEQ(va, vs);                                          \
      __m256i lt = CMPGT(vs, va);                                          \
      __m256i r = _mm256_and_si256(_mm256_or_si256(eq, lt), one);          \
      _mm256_storeu_si256((__m256i *)(out + i), r);                        \
    }                                                                      \
    for (; i < n; i++)                                                     \
      out[i] = (CT)(a[i] <= s);                                            \
  }

STAMP_CMPSC_SINT_AVX2(i8, int8_t, 32, _mm256_set1_epi8, _mm256_cmpeq_epi8,
                      _mm256_cmpgt_epi8)
STAMP_CMPSC_SINT_AVX2(i16, int16_t, 16, _mm256_set1_epi16, _mm256_cmpeq_epi16,
                      _mm256_cmpgt_epi16)
STAMP_CMPSC_SINT_AVX2(i32, int32_t, 8, _mm256_set1_epi32, _mm256_cmpeq_epi32,
                      _mm256_cmpgt_epi32)
STAMP_CMPSC_SINT_AVX2(i64, int64_t, 4, _mm256_set1_epi64x, _mm256_cmpeq_epi64,
                      _mm256_cmpgt_epi64)
#undef STAMP_CMPSC_SINT_AVX2

/* ── Unsigned integers (XOR sign-bit bias for ordering compares) ─ */

#define STAMP_CMPSC_UINT_AVX2(SFX, CT, VPV, SET1, CMPEQ, CMPGT, BIAS)      \
  static inline void _cmpsc_eq_##SFX##_avx2(const void *restrict ap,       \
                                            const void *restrict sp,       \
                                            void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                          \
    const CT s = *(const CT *)sp;                                          \
    CT *out = (CT *)op;                                                    \
    const __m256i vs = SET1((CT)s);                                        \
    const __m256i one = SET1(1);                                           \
    size_t i = 0;                                                          \
    for (; i + (VPV) <= n; i += (VPV)) {                                   \
      __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));           \
      __m256i r = _mm256_and_si256(CMPEQ(va, vs), one);                    \
      _mm256_storeu_si256((__m256i *)(out + i), r);                        \
    }                                                                      \
    for (; i < n; i++)                                                     \
      out[i] = (CT)(a[i] == s);                                            \
  }                                                                        \
  static inline void _cmpsc_gt_##SFX##_avx2(const void *restrict ap,       \
                                            const void *restrict sp,       \
                                            void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                          \
    const CT s = *(const CT *)sp;                                          \
    CT *out = (CT *)op;                                                    \
    const __m256i bias = BIAS;                                             \
    const __m256i vs_b = _mm256_xor_si256(SET1((CT)s), bias);              \
    const __m256i one = SET1(1);                                           \
    size_t i = 0;                                                          \
    for (; i + (VPV) <= n; i += (VPV)) {                                   \
      __m256i va = _mm256_xor_si256(                                       \
          _mm256_loadu_si256((const __m256i *)(a + i)), bias);             \
      __m256i r = _mm256_and_si256(CMPGT(va, vs_b), one);                  \
      _mm256_storeu_si256((__m256i *)(out + i), r);                        \
    }                                                                      \
    for (; i < n; i++)                                                     \
      out[i] = (CT)(a[i] > s);                                             \
  }                                                                        \
  static inline void _cmpsc_lt_##SFX##_avx2(const void *restrict ap,       \
                                            const void *restrict sp,       \
                                            void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                          \
    const CT s = *(const CT *)sp;                                          \
    CT *out = (CT *)op;                                                    \
    const __m256i bias = BIAS;                                             \
    const __m256i vs_b = _mm256_xor_si256(SET1((CT)s), bias);              \
    const __m256i one = SET1(1);                                           \
    size_t i = 0;                                                          \
    for (; i + (VPV) <= n; i += (VPV)) {                                   \
      __m256i va = _mm256_xor_si256(                                       \
          _mm256_loadu_si256((const __m256i *)(a + i)), bias);             \
      __m256i r = _mm256_and_si256(CMPGT(vs_b, va), one);                  \
      _mm256_storeu_si256((__m256i *)(out + i), r);                        \
    }                                                                      \
    for (; i < n; i++)                                                     \
      out[i] = (CT)(a[i] < s);                                             \
  }                                                                        \
  static inline void _cmpsc_ge_##SFX##_avx2(const void *restrict ap,       \
                                            const void *restrict sp,       \
                                            void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                          \
    const CT s = *(const CT *)sp;                                          \
    CT *out = (CT *)op;                                                    \
    const __m256i bias = BIAS;                                             \
    const __m256i vs_raw = SET1((CT)s);                                    \
    const __m256i vs_b = _mm256_xor_si256(vs_raw, bias);                   \
    const __m256i one = SET1(1);                                           \
    size_t i = 0;                                                          \
    for (; i + (VPV) <= n; i += (VPV)) {                                   \
      __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));           \
      __m256i eq = CMPEQ(va, vs_raw);                                      \
      __m256i va_b = _mm256_xor_si256(va, bias);                           \
      __m256i gt = CMPGT(va_b, vs_b);                                      \
      __m256i r = _mm256_and_si256(_mm256_or_si256(eq, gt), one);          \
      _mm256_storeu_si256((__m256i *)(out + i), r);                        \
    }                                                                      \
    for (; i < n; i++)                                                     \
      out[i] = (CT)(a[i] >= s);                                            \
  }                                                                        \
  static inline void _cmpsc_le_##SFX##_avx2(const void *restrict ap,       \
                                            const void *restrict sp,       \
                                            void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                          \
    const CT s = *(const CT *)sp;                                          \
    CT *out = (CT *)op;                                                    \
    const __m256i bias = BIAS;                                             \
    const __m256i vs_raw = SET1((CT)s);                                    \
    const __m256i vs_b = _mm256_xor_si256(vs_raw, bias);                   \
    const __m256i one = SET1(1);                                           \
    size_t i = 0;                                                          \
    for (; i + (VPV) <= n; i += (VPV)) {                                   \
      __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));           \
      __m256i eq = CMPEQ(va, vs_raw);                                      \
      __m256i va_b = _mm256_xor_si256(va, bias);                           \
      __m256i lt = CMPGT(vs_b, va_b);                                      \
      __m256i r = _mm256_and_si256(_mm256_or_si256(eq, lt), one);          \
      _mm256_storeu_si256((__m256i *)(out + i), r);                        \
    }                                                                      \
    for (; i < n; i++)                                                     \
      out[i] = (CT)(a[i] <= s);                                            \
  }

STAMP_CMPSC_UINT_AVX2(u8, uint8_t, 32, _mm256_set1_epi8, _mm256_cmpeq_epi8,
                      _mm256_cmpgt_epi8, _mm256_set1_epi8((char)0x80))
STAMP_CMPSC_UINT_AVX2(u16, uint16_t, 16, _mm256_set1_epi16, _mm256_cmpeq_epi16,
                      _mm256_cmpgt_epi16, _mm256_set1_epi16((short)0x8000))
STAMP_CMPSC_UINT_AVX2(u32, uint32_t, 8, _mm256_set1_epi32, _mm256_cmpeq_epi32,
                      _mm256_cmpgt_epi32, _mm256_set1_epi32((int)0x80000000))
STAMP_CMPSC_UINT_AVX2(u64, uint64_t, 4, _mm256_set1_epi64x, _mm256_cmpeq_epi64,
                      _mm256_cmpgt_epi64,
                      _mm256_set1_epi64x((long long)0x8000000000000000LL))
#undef STAMP_CMPSC_UINT_AVX2

#endif /* NUMC_COMPARE_SCALAR_AVX2_H */
