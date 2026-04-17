/**
 * @file compare_scalar_avx512.h
 * @brief AVX-512 scalar comparison kernels — uint8 output (0/1).
 *
 * All comparison functions output uint8_t* (NumPy-compatible bool).
 * AVX-512 comparisons return mask registers; we convert to byte vectors:
 *   8-bit:  __mmask64 → _mm512_maskz_set1_epi8 → store 64 bytes
 *  16-bit:  __mmask32 → _mm256_maskz_set1_epi8 → store 32 bytes
 *  32-bit:  __mmask16 → _mm_maskz_set1_epi8    → store 16 bytes
 *  64-bit:  __mmask8  → _mm_maskz_set1_epi8((__mmask16)m) → storel 8 bytes
 * Float:   same as 32/64-bit integer mask widths.
 */
#ifndef NUMC_COMPARE_SCALAR_AVX512_H
#define NUMC_COMPARE_SCALAR_AVX512_H

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

/* -- Float: f32 (16 per vector → 16 uint8 output) ---------------- */

#define CMPSC_F32_FUNC_AVX512(NAME, IMM, SCALAR_OP)                            \
  static inline void _cmpsc_##NAME##_f32_avx512(const void *restrict ap,       \
                                                const void *restrict sp,       \
                                                void *restrict op, size_t n) { \
    const float *a = (const float *)ap;                                        \
    const float s = *(const float *)sp;                                        \
    uint8_t *out = (uint8_t *)op;                                              \
    const __m512 vs = _mm512_set1_ps(s);                                       \
    size_t i = 0;                                                              \
    for (; i + 16 <= n; i += 16) {                                             \
      __m512 va = _mm512_loadu_ps(a + i);                                      \
      __mmask16 m = _mm512_cmp_ps_mask(va, vs, (IMM));                         \
      _mm_storeu_si128((__m128i *)(out + i), _mm_maskz_set1_epi8(m, 1));       \
    }                                                                          \
    for (; i < n; i++)                                                         \
      out[i] = (uint8_t)(a[i] SCALAR_OP s);                                    \
  }

CMPSC_F32_FUNC_AVX512(eq, _CMP_EQ_OQ, ==)
CMPSC_F32_FUNC_AVX512(gt, _CMP_GT_OS, >)
CMPSC_F32_FUNC_AVX512(lt, _CMP_LT_OS, <)
CMPSC_F32_FUNC_AVX512(ge, _CMP_GE_OS, >=)
CMPSC_F32_FUNC_AVX512(le, _CMP_LE_OS, <=)
#undef CMPSC_F32_FUNC_AVX512

/* -- Float: f64 (8 per vector → 8 uint8 output) ------------------ */

#define CMPSC_F64_FUNC_AVX512(NAME, IMM, SCALAR_OP)                            \
  static inline void _cmpsc_##NAME##_f64_avx512(const void *restrict ap,       \
                                                const void *restrict sp,       \
                                                void *restrict op, size_t n) { \
    const double *a = (const double *)ap;                                      \
    const double s = *(const double *)sp;                                      \
    uint8_t *out = (uint8_t *)op;                                              \
    const __m512d vs = _mm512_set1_pd(s);                                      \
    size_t i = 0;                                                              \
    for (; i + 8 <= n; i += 8) {                                               \
      __m512d va = _mm512_loadu_pd(a + i);                                     \
      __mmask8 m = _mm512_cmp_pd_mask(va, vs, (IMM));                          \
      _mm_storel_epi64((__m128i *)(out + i),                                   \
                       _mm_maskz_set1_epi8((__mmask16)m, 1));                  \
    }                                                                          \
    for (; i < n; i++)                                                         \
      out[i] = (uint8_t)(a[i] SCALAR_OP s);                                    \
  }

CMPSC_F64_FUNC_AVX512(eq, _CMP_EQ_OQ, ==)
CMPSC_F64_FUNC_AVX512(gt, _CMP_GT_OS, >)
CMPSC_F64_FUNC_AVX512(lt, _CMP_LT_OS, <)
CMPSC_F64_FUNC_AVX512(ge, _CMP_GE_OS, >=)
CMPSC_F64_FUNC_AVX512(le, _CMP_LE_OS, <=)
#undef CMPSC_F64_FUNC_AVX512

/* -- 8-bit signed/unsigned integers (64 per vector → 64 uint8) -- */

#define STAMP_CMPSC_I8_AVX512(SFX, CT, SET1, CMP)                              \
  static inline void _cmpsc_eq_##SFX##_avx512(const void *restrict ap,         \
                                              const void *restrict sp,         \
                                              void *restrict op, size_t n) {   \
    const CT *a = (const CT *)ap;                                              \
    const CT s = *(const CT *)sp;                                              \
    uint8_t *out = (uint8_t *)op;                                              \
    const __m512i vs = SET1(s);                                                \
    size_t i = 0;                                                              \
    for (; i + 64 <= n; i += 64) {                                             \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));               \
      __mmask64 m = CMP(va, vs, _MM_CMPINT_EQ);                                \
      _mm512_storeu_si512((__m512i *)(out + i), _mm512_maskz_set1_epi8(m, 1)); \
    }                                                                          \
    for (; i < n; i++)                                                         \
      out[i] = (uint8_t)(a[i] == s);                                           \
  }                                                                            \
  static inline void _cmpsc_gt_##SFX##_avx512(const void *restrict ap,         \
                                              const void *restrict sp,         \
                                              void *restrict op, size_t n) {   \
    const CT *a = (const CT *)ap;                                              \
    const CT s = *(const CT *)sp;                                              \
    uint8_t *out = (uint8_t *)op;                                              \
    const __m512i vs = SET1(s);                                                \
    size_t i = 0;                                                              \
    for (; i + 64 <= n; i += 64) {                                             \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));               \
      __mmask64 m = CMP(va, vs, _MM_CMPINT_NLE);                               \
      _mm512_storeu_si512((__m512i *)(out + i), _mm512_maskz_set1_epi8(m, 1)); \
    }                                                                          \
    for (; i < n; i++)                                                         \
      out[i] = (uint8_t)(a[i] > s);                                            \
  }                                                                            \
  static inline void _cmpsc_lt_##SFX##_avx512(const void *restrict ap,         \
                                              const void *restrict sp,         \
                                              void *restrict op, size_t n) {   \
    const CT *a = (const CT *)ap;                                              \
    const CT s = *(const CT *)sp;                                              \
    uint8_t *out = (uint8_t *)op;                                              \
    const __m512i vs = SET1(s);                                                \
    size_t i = 0;                                                              \
    for (; i + 64 <= n; i += 64) {                                             \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));               \
      __mmask64 m = CMP(va, vs, _MM_CMPINT_LT);                                \
      _mm512_storeu_si512((__m512i *)(out + i), _mm512_maskz_set1_epi8(m, 1)); \
    }                                                                          \
    for (; i < n; i++)                                                         \
      out[i] = (uint8_t)(a[i] < s);                                            \
  }                                                                            \
  static inline void _cmpsc_ge_##SFX##_avx512(const void *restrict ap,         \
                                              const void *restrict sp,         \
                                              void *restrict op, size_t n) {   \
    const CT *a = (const CT *)ap;                                              \
    const CT s = *(const CT *)sp;                                              \
    uint8_t *out = (uint8_t *)op;                                              \
    const __m512i vs = SET1(s);                                                \
    size_t i = 0;                                                              \
    for (; i + 64 <= n; i += 64) {                                             \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));               \
      __mmask64 m = CMP(va, vs, _MM_CMPINT_NLT);                               \
      _mm512_storeu_si512((__m512i *)(out + i), _mm512_maskz_set1_epi8(m, 1)); \
    }                                                                          \
    for (; i < n; i++)                                                         \
      out[i] = (uint8_t)(a[i] >= s);                                           \
  }                                                                            \
  static inline void _cmpsc_le_##SFX##_avx512(const void *restrict ap,         \
                                              const void *restrict sp,         \
                                              void *restrict op, size_t n) {   \
    const CT *a = (const CT *)ap;                                              \
    const CT s = *(const CT *)sp;                                              \
    uint8_t *out = (uint8_t *)op;                                              \
    const __m512i vs = SET1(s);                                                \
    size_t i = 0;                                                              \
    for (; i + 64 <= n; i += 64) {                                             \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));               \
      __mmask64 m = CMP(va, vs, _MM_CMPINT_LE);                                \
      _mm512_storeu_si512((__m512i *)(out + i), _mm512_maskz_set1_epi8(m, 1)); \
    }                                                                          \
    for (; i < n; i++)                                                         \
      out[i] = (uint8_t)(a[i] <= s);                                           \
  }

STAMP_CMPSC_I8_AVX512(i8, int8_t, _mm512_set1_epi8, _mm512_cmp_epi8_mask)
STAMP_CMPSC_I8_AVX512(u8, uint8_t, _mm512_set1_epi8, _mm512_cmp_epu8_mask)
#undef STAMP_CMPSC_I8_AVX512

/* -- 16-bit signed/unsigned integers (32 per vector → 32 uint8) -- */

#define STAMP_CMPSC_I16_AVX512(SFX, CT, SET1, CMP)                             \
  static inline void _cmpsc_eq_##SFX##_avx512(const void *restrict ap,         \
                                              const void *restrict sp,         \
                                              void *restrict op, size_t n) {   \
    const CT *a = (const CT *)ap;                                              \
    const CT s = *(const CT *)sp;                                              \
    uint8_t *out = (uint8_t *)op;                                              \
    const __m512i vs = SET1(s);                                                \
    size_t i = 0;                                                              \
    for (; i + 32 <= n; i += 32) {                                             \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));               \
      __mmask32 m = CMP(va, vs, _MM_CMPINT_EQ);                                \
      _mm256_storeu_si256((__m256i *)(out + i), _mm256_maskz_set1_epi8(m, 1)); \
    }                                                                          \
    for (; i < n; i++)                                                         \
      out[i] = (uint8_t)(a[i] == s);                                           \
  }                                                                            \
  static inline void _cmpsc_gt_##SFX##_avx512(const void *restrict ap,         \
                                              const void *restrict sp,         \
                                              void *restrict op, size_t n) {   \
    const CT *a = (const CT *)ap;                                              \
    const CT s = *(const CT *)sp;                                              \
    uint8_t *out = (uint8_t *)op;                                              \
    const __m512i vs = SET1(s);                                                \
    size_t i = 0;                                                              \
    for (; i + 32 <= n; i += 32) {                                             \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));               \
      __mmask32 m = CMP(va, vs, _MM_CMPINT_NLE);                               \
      _mm256_storeu_si256((__m256i *)(out + i), _mm256_maskz_set1_epi8(m, 1)); \
    }                                                                          \
    for (; i < n; i++)                                                         \
      out[i] = (uint8_t)(a[i] > s);                                            \
  }                                                                            \
  static inline void _cmpsc_lt_##SFX##_avx512(const void *restrict ap,         \
                                              const void *restrict sp,         \
                                              void *restrict op, size_t n) {   \
    const CT *a = (const CT *)ap;                                              \
    const CT s = *(const CT *)sp;                                              \
    uint8_t *out = (uint8_t *)op;                                              \
    const __m512i vs = SET1(s);                                                \
    size_t i = 0;                                                              \
    for (; i + 32 <= n; i += 32) {                                             \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));               \
      __mmask32 m = CMP(va, vs, _MM_CMPINT_LT);                                \
      _mm256_storeu_si256((__m256i *)(out + i), _mm256_maskz_set1_epi8(m, 1)); \
    }                                                                          \
    for (; i < n; i++)                                                         \
      out[i] = (uint8_t)(a[i] < s);                                            \
  }                                                                            \
  static inline void _cmpsc_ge_##SFX##_avx512(const void *restrict ap,         \
                                              const void *restrict sp,         \
                                              void *restrict op, size_t n) {   \
    const CT *a = (const CT *)ap;                                              \
    const CT s = *(const CT *)sp;                                              \
    uint8_t *out = (uint8_t *)op;                                              \
    const __m512i vs = SET1(s);                                                \
    size_t i = 0;                                                              \
    for (; i + 32 <= n; i += 32) {                                             \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));               \
      __mmask32 m = CMP(va, vs, _MM_CMPINT_NLT);                               \
      _mm256_storeu_si256((__m256i *)(out + i), _mm256_maskz_set1_epi8(m, 1)); \
    }                                                                          \
    for (; i < n; i++)                                                         \
      out[i] = (uint8_t)(a[i] >= s);                                           \
  }                                                                            \
  static inline void _cmpsc_le_##SFX##_avx512(const void *restrict ap,         \
                                              const void *restrict sp,         \
                                              void *restrict op, size_t n) {   \
    const CT *a = (const CT *)ap;                                              \
    const CT s = *(const CT *)sp;                                              \
    uint8_t *out = (uint8_t *)op;                                              \
    const __m512i vs = SET1(s);                                                \
    size_t i = 0;                                                              \
    for (; i + 32 <= n; i += 32) {                                             \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));               \
      __mmask32 m = CMP(va, vs, _MM_CMPINT_LE);                                \
      _mm256_storeu_si256((__m256i *)(out + i), _mm256_maskz_set1_epi8(m, 1)); \
    }                                                                          \
    for (; i < n; i++)                                                         \
      out[i] = (uint8_t)(a[i] <= s);                                           \
  }

STAMP_CMPSC_I16_AVX512(i16, int16_t, _mm512_set1_epi16, _mm512_cmp_epi16_mask)
STAMP_CMPSC_I16_AVX512(u16, uint16_t, _mm512_set1_epi16, _mm512_cmp_epu16_mask)
#undef STAMP_CMPSC_I16_AVX512

/* -- 32-bit signed/unsigned integers (16 per vector → 16 uint8) -- */

#define STAMP_CMPSC_I32_AVX512(SFX, CT, SET1, CMP)                           \
  static inline void _cmpsc_eq_##SFX##_avx512(const void *restrict ap,       \
                                              const void *restrict sp,       \
                                              void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                            \
    const CT s = *(const CT *)sp;                                            \
    uint8_t *out = (uint8_t *)op;                                            \
    const __m512i vs = SET1(s);                                              \
    size_t i = 0;                                                            \
    for (; i + 16 <= n; i += 16) {                                           \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));             \
      __mmask16 m = CMP(va, vs, _MM_CMPINT_EQ);                              \
      _mm_storeu_si128((__m128i *)(out + i), _mm_maskz_set1_epi8(m, 1));     \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (uint8_t)(a[i] == s);                                         \
  }                                                                          \
  static inline void _cmpsc_gt_##SFX##_avx512(const void *restrict ap,       \
                                              const void *restrict sp,       \
                                              void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                            \
    const CT s = *(const CT *)sp;                                            \
    uint8_t *out = (uint8_t *)op;                                            \
    const __m512i vs = SET1(s);                                              \
    size_t i = 0;                                                            \
    for (; i + 16 <= n; i += 16) {                                           \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));             \
      __mmask16 m = CMP(va, vs, _MM_CMPINT_NLE);                             \
      _mm_storeu_si128((__m128i *)(out + i), _mm_maskz_set1_epi8(m, 1));     \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (uint8_t)(a[i] > s);                                          \
  }                                                                          \
  static inline void _cmpsc_lt_##SFX##_avx512(const void *restrict ap,       \
                                              const void *restrict sp,       \
                                              void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                            \
    const CT s = *(const CT *)sp;                                            \
    uint8_t *out = (uint8_t *)op;                                            \
    const __m512i vs = SET1(s);                                              \
    size_t i = 0;                                                            \
    for (; i + 16 <= n; i += 16) {                                           \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));             \
      __mmask16 m = CMP(va, vs, _MM_CMPINT_LT);                              \
      _mm_storeu_si128((__m128i *)(out + i), _mm_maskz_set1_epi8(m, 1));     \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (uint8_t)(a[i] < s);                                          \
  }                                                                          \
  static inline void _cmpsc_ge_##SFX##_avx512(const void *restrict ap,       \
                                              const void *restrict sp,       \
                                              void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                            \
    const CT s = *(const CT *)sp;                                            \
    uint8_t *out = (uint8_t *)op;                                            \
    const __m512i vs = SET1(s);                                              \
    size_t i = 0;                                                            \
    for (; i + 16 <= n; i += 16) {                                           \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));             \
      __mmask16 m = CMP(va, vs, _MM_CMPINT_NLT);                             \
      _mm_storeu_si128((__m128i *)(out + i), _mm_maskz_set1_epi8(m, 1));     \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (uint8_t)(a[i] >= s);                                         \
  }                                                                          \
  static inline void _cmpsc_le_##SFX##_avx512(const void *restrict ap,       \
                                              const void *restrict sp,       \
                                              void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                            \
    const CT s = *(const CT *)sp;                                            \
    uint8_t *out = (uint8_t *)op;                                            \
    const __m512i vs = SET1(s);                                              \
    size_t i = 0;                                                            \
    for (; i + 16 <= n; i += 16) {                                           \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));             \
      __mmask16 m = CMP(va, vs, _MM_CMPINT_LE);                              \
      _mm_storeu_si128((__m128i *)(out + i), _mm_maskz_set1_epi8(m, 1));     \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (uint8_t)(a[i] <= s);                                         \
  }

STAMP_CMPSC_I32_AVX512(i32, int32_t, _mm512_set1_epi32, _mm512_cmp_epi32_mask)
STAMP_CMPSC_I32_AVX512(u32, uint32_t, _mm512_set1_epi32, _mm512_cmp_epu32_mask)
#undef STAMP_CMPSC_I32_AVX512

/* -- 64-bit signed/unsigned integers (8 per vector → 8 uint8) --- */

#define STAMP_CMPSC_I64_AVX512(SFX, CT, SET1, CMP)                           \
  static inline void _cmpsc_eq_##SFX##_avx512(const void *restrict ap,       \
                                              const void *restrict sp,       \
                                              void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                            \
    const CT s = *(const CT *)sp;                                            \
    uint8_t *out = (uint8_t *)op;                                            \
    const __m512i vs = SET1(s);                                              \
    size_t i = 0;                                                            \
    for (; i + 8 <= n; i += 8) {                                             \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));             \
      __mmask8 m = CMP(va, vs, _MM_CMPINT_EQ);                               \
      _mm_storel_epi64((__m128i *)(out + i),                                 \
                       _mm_maskz_set1_epi8((__mmask16)m, 1));                \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (uint8_t)(a[i] == s);                                         \
  }                                                                          \
  static inline void _cmpsc_gt_##SFX##_avx512(const void *restrict ap,       \
                                              const void *restrict sp,       \
                                              void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                            \
    const CT s = *(const CT *)sp;                                            \
    uint8_t *out = (uint8_t *)op;                                            \
    const __m512i vs = SET1(s);                                              \
    size_t i = 0;                                                            \
    for (; i + 8 <= n; i += 8) {                                             \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));             \
      __mmask8 m = CMP(va, vs, _MM_CMPINT_NLE);                              \
      _mm_storel_epi64((__m128i *)(out + i),                                 \
                       _mm_maskz_set1_epi8((__mmask16)m, 1));                \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (uint8_t)(a[i] > s);                                          \
  }                                                                          \
  static inline void _cmpsc_lt_##SFX##_avx512(const void *restrict ap,       \
                                              const void *restrict sp,       \
                                              void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                            \
    const CT s = *(const CT *)sp;                                            \
    uint8_t *out = (uint8_t *)op;                                            \
    const __m512i vs = SET1(s);                                              \
    size_t i = 0;                                                            \
    for (; i + 8 <= n; i += 8) {                                             \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));             \
      __mmask8 m = CMP(va, vs, _MM_CMPINT_LT);                               \
      _mm_storel_epi64((__m128i *)(out + i),                                 \
                       _mm_maskz_set1_epi8((__mmask16)m, 1));                \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (uint8_t)(a[i] < s);                                          \
  }                                                                          \
  static inline void _cmpsc_ge_##SFX##_avx512(const void *restrict ap,       \
                                              const void *restrict sp,       \
                                              void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                            \
    const CT s = *(const CT *)sp;                                            \
    uint8_t *out = (uint8_t *)op;                                            \
    const __m512i vs = SET1(s);                                              \
    size_t i = 0;                                                            \
    for (; i + 8 <= n; i += 8) {                                             \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));             \
      __mmask8 m = CMP(va, vs, _MM_CMPINT_NLT);                              \
      _mm_storel_epi64((__m128i *)(out + i),                                 \
                       _mm_maskz_set1_epi8((__mmask16)m, 1));                \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (uint8_t)(a[i] >= s);                                         \
  }                                                                          \
  static inline void _cmpsc_le_##SFX##_avx512(const void *restrict ap,       \
                                              const void *restrict sp,       \
                                              void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                            \
    const CT s = *(const CT *)sp;                                            \
    uint8_t *out = (uint8_t *)op;                                            \
    const __m512i vs = SET1(s);                                              \
    size_t i = 0;                                                            \
    for (; i + 8 <= n; i += 8) {                                             \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));             \
      __mmask8 m = CMP(va, vs, _MM_CMPINT_LE);                               \
      _mm_storel_epi64((__m128i *)(out + i),                                 \
                       _mm_maskz_set1_epi8((__mmask16)m, 1));                \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (uint8_t)(a[i] <= s);                                         \
  }

STAMP_CMPSC_I64_AVX512(i64, int64_t, _mm512_set1_epi64, _mm512_cmp_epi64_mask)
STAMP_CMPSC_I64_AVX512(u64, uint64_t, _mm512_set1_epi64, _mm512_cmp_epu64_mask)
#undef STAMP_CMPSC_I64_AVX512

#endif /* NUMC_COMPARE_SCALAR_AVX512_H */
