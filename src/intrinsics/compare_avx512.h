/**
 * @file compare_avx512.h
 * @brief AVX-512 binary comparison kernels — uint8 output (0/1).
 *
 * All comparison functions output uint8_t* (NumPy-compatible bool).
 * AVX-512 comparisons return __mmask types; we convert to byte vectors:
 *   8-bit:  __mmask64 → _mm512_maskz_set1_epi8 → store 64 bytes
 *  16-bit:  __mmask32 → _mm256_maskz_set1_epi8 → store 32 bytes
 *  32-bit:  __mmask16 → _mm_maskz_set1_epi8    → store 16 bytes
 *  64-bit:  __mmask8  → _mm_maskz_set1_epi8((__mmask16)m) → storel 8 bytes
 */
#ifndef NUMC_COMPARE_AVX512_H
#define NUMC_COMPARE_AVX512_H

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

/* ====================================================================
 * 8-bit signed integers (64 elems/vector → 64 uint8 output)
 * mask is __mmask64, _mm512_maskz_set1_epi8 produces 64 bytes of 0/1
 * ================================================================ */

#define FAST_CMP_I8_AVX512(SFX, CT, CMP)                                       \
  static inline void _fast_eq_##SFX##_avx512(const void *restrict ap,          \
                                             const void *restrict bp,          \
                                             void *restrict op, size_t n) {    \
    const CT *a = (const CT *)ap;                                              \
    const CT *b = (const CT *)bp;                                              \
    uint8_t *out = (uint8_t *)op;                                              \
    size_t i = 0;                                                              \
    for (; i + 64 <= n; i += 64) {                                             \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));               \
      __m512i vb = _mm512_loadu_si512((const __m512i *)(b + i));               \
      __mmask64 m = CMP(va, vb, _MM_CMPINT_EQ);                                \
      _mm512_storeu_si512((__m512i *)(out + i), _mm512_maskz_set1_epi8(m, 1)); \
    }                                                                          \
    for (; i < n; i++)                                                         \
      out[i] = (uint8_t)(a[i] == b[i]);                                        \
  }                                                                            \
  static inline void _fast_gt_##SFX##_avx512(const void *restrict ap,          \
                                             const void *restrict bp,          \
                                             void *restrict op, size_t n) {    \
    const CT *a = (const CT *)ap;                                              \
    const CT *b = (const CT *)bp;                                              \
    uint8_t *out = (uint8_t *)op;                                              \
    size_t i = 0;                                                              \
    for (; i + 64 <= n; i += 64) {                                             \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));               \
      __m512i vb = _mm512_loadu_si512((const __m512i *)(b + i));               \
      __mmask64 m = CMP(va, vb, _MM_CMPINT_NLE);                               \
      _mm512_storeu_si512((__m512i *)(out + i), _mm512_maskz_set1_epi8(m, 1)); \
    }                                                                          \
    for (; i < n; i++)                                                         \
      out[i] = (uint8_t)(a[i] > b[i]);                                         \
  }                                                                            \
  static inline void _fast_lt_##SFX##_avx512(const void *restrict ap,          \
                                             const void *restrict bp,          \
                                             void *restrict op, size_t n) {    \
    const CT *a = (const CT *)ap;                                              \
    const CT *b = (const CT *)bp;                                              \
    uint8_t *out = (uint8_t *)op;                                              \
    size_t i = 0;                                                              \
    for (; i + 64 <= n; i += 64) {                                             \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));               \
      __m512i vb = _mm512_loadu_si512((const __m512i *)(b + i));               \
      __mmask64 m = CMP(va, vb, _MM_CMPINT_LT);                                \
      _mm512_storeu_si512((__m512i *)(out + i), _mm512_maskz_set1_epi8(m, 1)); \
    }                                                                          \
    for (; i < n; i++)                                                         \
      out[i] = (uint8_t)(a[i] < b[i]);                                         \
  }                                                                            \
  static inline void _fast_ge_##SFX##_avx512(const void *restrict ap,          \
                                             const void *restrict bp,          \
                                             void *restrict op, size_t n) {    \
    const CT *a = (const CT *)ap;                                              \
    const CT *b = (const CT *)bp;                                              \
    uint8_t *out = (uint8_t *)op;                                              \
    size_t i = 0;                                                              \
    for (; i + 64 <= n; i += 64) {                                             \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));               \
      __m512i vb = _mm512_loadu_si512((const __m512i *)(b + i));               \
      __mmask64 m = CMP(va, vb, _MM_CMPINT_NLT);                               \
      _mm512_storeu_si512((__m512i *)(out + i), _mm512_maskz_set1_epi8(m, 1)); \
    }                                                                          \
    for (; i < n; i++)                                                         \
      out[i] = (uint8_t)(a[i] >= b[i]);                                        \
  }                                                                            \
  static inline void _fast_le_##SFX##_avx512(const void *restrict ap,          \
                                             const void *restrict bp,          \
                                             void *restrict op, size_t n) {    \
    const CT *a = (const CT *)ap;                                              \
    const CT *b = (const CT *)bp;                                              \
    uint8_t *out = (uint8_t *)op;                                              \
    size_t i = 0;                                                              \
    for (; i + 64 <= n; i += 64) {                                             \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));               \
      __m512i vb = _mm512_loadu_si512((const __m512i *)(b + i));               \
      __mmask64 m = CMP(va, vb, _MM_CMPINT_LE);                                \
      _mm512_storeu_si512((__m512i *)(out + i), _mm512_maskz_set1_epi8(m, 1)); \
    }                                                                          \
    for (; i < n; i++)                                                         \
      out[i] = (uint8_t)(a[i] <= b[i]);                                        \
  }

FAST_CMP_I8_AVX512(i8, int8_t, _mm512_cmp_epi8_mask)
FAST_CMP_I8_AVX512(u8, uint8_t, _mm512_cmp_epu8_mask)
#undef FAST_CMP_I8_AVX512

/* ====================================================================
 * 16-bit integers (32 elems/vector → 32 uint8 output)
 * mask is __mmask32, _mm256_maskz_set1_epi8 produces 32 bytes
 * ================================================================ */

#define FAST_CMP_I16_AVX512(SFX, CT, CMP)                                      \
  static inline void _fast_eq_##SFX##_avx512(const void *restrict ap,          \
                                             const void *restrict bp,          \
                                             void *restrict op, size_t n) {    \
    const CT *a = (const CT *)ap;                                              \
    const CT *b = (const CT *)bp;                                              \
    uint8_t *out = (uint8_t *)op;                                              \
    size_t i = 0;                                                              \
    for (; i + 32 <= n; i += 32) {                                             \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));               \
      __m512i vb = _mm512_loadu_si512((const __m512i *)(b + i));               \
      __mmask32 m = CMP(va, vb, _MM_CMPINT_EQ);                                \
      _mm256_storeu_si256((__m256i *)(out + i), _mm256_maskz_set1_epi8(m, 1)); \
    }                                                                          \
    for (; i < n; i++)                                                         \
      out[i] = (uint8_t)(a[i] == b[i]);                                        \
  }                                                                            \
  static inline void _fast_gt_##SFX##_avx512(const void *restrict ap,          \
                                             const void *restrict bp,          \
                                             void *restrict op, size_t n) {    \
    const CT *a = (const CT *)ap;                                              \
    const CT *b = (const CT *)bp;                                              \
    uint8_t *out = (uint8_t *)op;                                              \
    size_t i = 0;                                                              \
    for (; i + 32 <= n; i += 32) {                                             \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));               \
      __m512i vb = _mm512_loadu_si512((const __m512i *)(b + i));               \
      __mmask32 m = CMP(va, vb, _MM_CMPINT_NLE);                               \
      _mm256_storeu_si256((__m256i *)(out + i), _mm256_maskz_set1_epi8(m, 1)); \
    }                                                                          \
    for (; i < n; i++)                                                         \
      out[i] = (uint8_t)(a[i] > b[i]);                                         \
  }                                                                            \
  static inline void _fast_lt_##SFX##_avx512(const void *restrict ap,          \
                                             const void *restrict bp,          \
                                             void *restrict op, size_t n) {    \
    const CT *a = (const CT *)ap;                                              \
    const CT *b = (const CT *)bp;                                              \
    uint8_t *out = (uint8_t *)op;                                              \
    size_t i = 0;                                                              \
    for (; i + 32 <= n; i += 32) {                                             \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));               \
      __m512i vb = _mm512_loadu_si512((const __m512i *)(b + i));               \
      __mmask32 m = CMP(va, vb, _MM_CMPINT_LT);                                \
      _mm256_storeu_si256((__m256i *)(out + i), _mm256_maskz_set1_epi8(m, 1)); \
    }                                                                          \
    for (; i < n; i++)                                                         \
      out[i] = (uint8_t)(a[i] < b[i]);                                         \
  }                                                                            \
  static inline void _fast_ge_##SFX##_avx512(const void *restrict ap,          \
                                             const void *restrict bp,          \
                                             void *restrict op, size_t n) {    \
    const CT *a = (const CT *)ap;                                              \
    const CT *b = (const CT *)bp;                                              \
    uint8_t *out = (uint8_t *)op;                                              \
    size_t i = 0;                                                              \
    for (; i + 32 <= n; i += 32) {                                             \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));               \
      __m512i vb = _mm512_loadu_si512((const __m512i *)(b + i));               \
      __mmask32 m = CMP(va, vb, _MM_CMPINT_NLT);                               \
      _mm256_storeu_si256((__m256i *)(out + i), _mm256_maskz_set1_epi8(m, 1)); \
    }                                                                          \
    for (; i < n; i++)                                                         \
      out[i] = (uint8_t)(a[i] >= b[i]);                                        \
  }                                                                            \
  static inline void _fast_le_##SFX##_avx512(const void *restrict ap,          \
                                             const void *restrict bp,          \
                                             void *restrict op, size_t n) {    \
    const CT *a = (const CT *)ap;                                              \
    const CT *b = (const CT *)bp;                                              \
    uint8_t *out = (uint8_t *)op;                                              \
    size_t i = 0;                                                              \
    for (; i + 32 <= n; i += 32) {                                             \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));               \
      __m512i vb = _mm512_loadu_si512((const __m512i *)(b + i));               \
      __mmask32 m = CMP(va, vb, _MM_CMPINT_LE);                                \
      _mm256_storeu_si256((__m256i *)(out + i), _mm256_maskz_set1_epi8(m, 1)); \
    }                                                                          \
    for (; i < n; i++)                                                         \
      out[i] = (uint8_t)(a[i] <= b[i]);                                        \
  }

FAST_CMP_I16_AVX512(i16, int16_t, _mm512_cmp_epi16_mask)
FAST_CMP_I16_AVX512(u16, uint16_t, _mm512_cmp_epu16_mask)
#undef FAST_CMP_I16_AVX512

/* ====================================================================
 * 32-bit integers (16 elems/vector → 16 uint8 output)
 * mask is __mmask16, _mm_maskz_set1_epi8 produces 16 bytes
 * ================================================================ */

#define FAST_CMP_I32_AVX512(SFX, CT, CMP)                                   \
  static inline void _fast_eq_##SFX##_avx512(const void *restrict ap,       \
                                             const void *restrict bp,       \
                                             void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                           \
    const CT *b = (const CT *)bp;                                           \
    uint8_t *out = (uint8_t *)op;                                           \
    size_t i = 0;                                                           \
    for (; i + 16 <= n; i += 16) {                                          \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));            \
      __m512i vb = _mm512_loadu_si512((const __m512i *)(b + i));            \
      __mmask16 m = CMP(va, vb, _MM_CMPINT_EQ);                             \
      _mm_storeu_si128((__m128i *)(out + i), _mm_maskz_set1_epi8(m, 1));    \
    }                                                                       \
    for (; i < n; i++)                                                      \
      out[i] = (uint8_t)(a[i] == b[i]);                                     \
  }                                                                         \
  static inline void _fast_gt_##SFX##_avx512(const void *restrict ap,       \
                                             const void *restrict bp,       \
                                             void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                           \
    const CT *b = (const CT *)bp;                                           \
    uint8_t *out = (uint8_t *)op;                                           \
    size_t i = 0;                                                           \
    for (; i + 16 <= n; i += 16) {                                          \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));            \
      __m512i vb = _mm512_loadu_si512((const __m512i *)(b + i));            \
      __mmask16 m = CMP(va, vb, _MM_CMPINT_NLE);                            \
      _mm_storeu_si128((__m128i *)(out + i), _mm_maskz_set1_epi8(m, 1));    \
    }                                                                       \
    for (; i < n; i++)                                                      \
      out[i] = (uint8_t)(a[i] > b[i]);                                      \
  }                                                                         \
  static inline void _fast_lt_##SFX##_avx512(const void *restrict ap,       \
                                             const void *restrict bp,       \
                                             void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                           \
    const CT *b = (const CT *)bp;                                           \
    uint8_t *out = (uint8_t *)op;                                           \
    size_t i = 0;                                                           \
    for (; i + 16 <= n; i += 16) {                                          \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));            \
      __m512i vb = _mm512_loadu_si512((const __m512i *)(b + i));            \
      __mmask16 m = CMP(va, vb, _MM_CMPINT_LT);                             \
      _mm_storeu_si128((__m128i *)(out + i), _mm_maskz_set1_epi8(m, 1));    \
    }                                                                       \
    for (; i < n; i++)                                                      \
      out[i] = (uint8_t)(a[i] < b[i]);                                      \
  }                                                                         \
  static inline void _fast_ge_##SFX##_avx512(const void *restrict ap,       \
                                             const void *restrict bp,       \
                                             void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                           \
    const CT *b = (const CT *)bp;                                           \
    uint8_t *out = (uint8_t *)op;                                           \
    size_t i = 0;                                                           \
    for (; i + 16 <= n; i += 16) {                                          \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));            \
      __m512i vb = _mm512_loadu_si512((const __m512i *)(b + i));            \
      __mmask16 m = CMP(va, vb, _MM_CMPINT_NLT);                            \
      _mm_storeu_si128((__m128i *)(out + i), _mm_maskz_set1_epi8(m, 1));    \
    }                                                                       \
    for (; i < n; i++)                                                      \
      out[i] = (uint8_t)(a[i] >= b[i]);                                     \
  }                                                                         \
  static inline void _fast_le_##SFX##_avx512(const void *restrict ap,       \
                                             const void *restrict bp,       \
                                             void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                           \
    const CT *b = (const CT *)bp;                                           \
    uint8_t *out = (uint8_t *)op;                                           \
    size_t i = 0;                                                           \
    for (; i + 16 <= n; i += 16) {                                          \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));            \
      __m512i vb = _mm512_loadu_si512((const __m512i *)(b + i));            \
      __mmask16 m = CMP(va, vb, _MM_CMPINT_LE);                             \
      _mm_storeu_si128((__m128i *)(out + i), _mm_maskz_set1_epi8(m, 1));    \
    }                                                                       \
    for (; i < n; i++)                                                      \
      out[i] = (uint8_t)(a[i] <= b[i]);                                     \
  }

FAST_CMP_I32_AVX512(i32, int32_t, _mm512_cmp_epi32_mask)
FAST_CMP_I32_AVX512(u32, uint32_t, _mm512_cmp_epu32_mask)
#undef FAST_CMP_I32_AVX512

/* ====================================================================
 * 64-bit integers (8 elems/vector → 8 uint8 output)
 * mask is __mmask8, widen to __mmask16 for _mm_maskz_set1_epi8,
 * then storel (low 8 bytes)
 * ================================================================ */

#define FAST_CMP_I64_AVX512(SFX, CT, CMP)                                   \
  static inline void _fast_eq_##SFX##_avx512(const void *restrict ap,       \
                                             const void *restrict bp,       \
                                             void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                           \
    const CT *b = (const CT *)bp;                                           \
    uint8_t *out = (uint8_t *)op;                                           \
    size_t i = 0;                                                           \
    for (; i + 8 <= n; i += 8) {                                            \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));            \
      __m512i vb = _mm512_loadu_si512((const __m512i *)(b + i));            \
      __mmask8 m = CMP(va, vb, _MM_CMPINT_EQ);                              \
      _mm_storel_epi64((__m128i *)(out + i),                                \
                       _mm_maskz_set1_epi8((__mmask16)m, 1));               \
    }                                                                       \
    for (; i < n; i++)                                                      \
      out[i] = (uint8_t)(a[i] == b[i]);                                     \
  }                                                                         \
  static inline void _fast_gt_##SFX##_avx512(const void *restrict ap,       \
                                             const void *restrict bp,       \
                                             void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                           \
    const CT *b = (const CT *)bp;                                           \
    uint8_t *out = (uint8_t *)op;                                           \
    size_t i = 0;                                                           \
    for (; i + 8 <= n; i += 8) {                                            \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));            \
      __m512i vb = _mm512_loadu_si512((const __m512i *)(b + i));            \
      __mmask8 m = CMP(va, vb, _MM_CMPINT_NLE);                             \
      _mm_storel_epi64((__m128i *)(out + i),                                \
                       _mm_maskz_set1_epi8((__mmask16)m, 1));               \
    }                                                                       \
    for (; i < n; i++)                                                      \
      out[i] = (uint8_t)(a[i] > b[i]);                                      \
  }                                                                         \
  static inline void _fast_lt_##SFX##_avx512(const void *restrict ap,       \
                                             const void *restrict bp,       \
                                             void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                           \
    const CT *b = (const CT *)bp;                                           \
    uint8_t *out = (uint8_t *)op;                                           \
    size_t i = 0;                                                           \
    for (; i + 8 <= n; i += 8) {                                            \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));            \
      __m512i vb = _mm512_loadu_si512((const __m512i *)(b + i));            \
      __mmask8 m = CMP(va, vb, _MM_CMPINT_LT);                              \
      _mm_storel_epi64((__m128i *)(out + i),                                \
                       _mm_maskz_set1_epi8((__mmask16)m, 1));               \
    }                                                                       \
    for (; i < n; i++)                                                      \
      out[i] = (uint8_t)(a[i] < b[i]);                                      \
  }                                                                         \
  static inline void _fast_ge_##SFX##_avx512(const void *restrict ap,       \
                                             const void *restrict bp,       \
                                             void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                           \
    const CT *b = (const CT *)bp;                                           \
    uint8_t *out = (uint8_t *)op;                                           \
    size_t i = 0;                                                           \
    for (; i + 8 <= n; i += 8) {                                            \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));            \
      __m512i vb = _mm512_loadu_si512((const __m512i *)(b + i));            \
      __mmask8 m = CMP(va, vb, _MM_CMPINT_NLT);                             \
      _mm_storel_epi64((__m128i *)(out + i),                                \
                       _mm_maskz_set1_epi8((__mmask16)m, 1));               \
    }                                                                       \
    for (; i < n; i++)                                                      \
      out[i] = (uint8_t)(a[i] >= b[i]);                                     \
  }                                                                         \
  static inline void _fast_le_##SFX##_avx512(const void *restrict ap,       \
                                             const void *restrict bp,       \
                                             void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                           \
    const CT *b = (const CT *)bp;                                           \
    uint8_t *out = (uint8_t *)op;                                           \
    size_t i = 0;                                                           \
    for (; i + 8 <= n; i += 8) {                                            \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));            \
      __m512i vb = _mm512_loadu_si512((const __m512i *)(b + i));            \
      __mmask8 m = CMP(va, vb, _MM_CMPINT_LE);                              \
      _mm_storel_epi64((__m128i *)(out + i),                                \
                       _mm_maskz_set1_epi8((__mmask16)m, 1));               \
    }                                                                       \
    for (; i < n; i++)                                                      \
      out[i] = (uint8_t)(a[i] <= b[i]);                                     \
  }

FAST_CMP_I64_AVX512(i64, int64_t, _mm512_cmp_epi64_mask)
FAST_CMP_I64_AVX512(u64, uint64_t, _mm512_cmp_epu64_mask)
#undef FAST_CMP_I64_AVX512

/* ====================================================================
 * Float comparisons (f32 — 16 per vector → 16 uint8 output)
 * mask is __mmask16, _mm_maskz_set1_epi8 produces 16 bytes
 * ================================================================ */

#define FAST_CMP_F32_AVX512(OP, PRED, TAIL_OP)                              \
  static inline void _fast_##OP##_f32_avx512(const void *restrict ap,       \
                                             const void *restrict bp,       \
                                             void *restrict op, size_t n) { \
    const float *a = (const float *)ap;                                     \
    const float *b = (const float *)bp;                                     \
    uint8_t *out = (uint8_t *)op;                                           \
    size_t i = 0;                                                           \
    for (; i + 16 <= n; i += 16) {                                          \
      __m512 va = _mm512_loadu_ps(a + i);                                   \
      __m512 vb = _mm512_loadu_ps(b + i);                                   \
      __mmask16 m = _mm512_cmp_ps_mask(va, vb, PRED);                       \
      _mm_storeu_si128((__m128i *)(out + i), _mm_maskz_set1_epi8(m, 1));    \
    }                                                                       \
    for (; i < n; i++)                                                      \
      out[i] = (uint8_t)(a[i] TAIL_OP b[i]);                                \
  }

FAST_CMP_F32_AVX512(eq, _CMP_EQ_OQ, ==)
FAST_CMP_F32_AVX512(gt, _CMP_GT_OQ, >)
FAST_CMP_F32_AVX512(lt, _CMP_LT_OQ, <)
FAST_CMP_F32_AVX512(ge, _CMP_GE_OQ, >=)
FAST_CMP_F32_AVX512(le, _CMP_LE_OQ, <=)
#undef FAST_CMP_F32_AVX512

/* ====================================================================
 * Float comparisons (f64 — 8 per vector → 8 uint8 output)
 * mask is __mmask8, widen to __mmask16 for _mm_maskz_set1_epi8,
 * then storel (low 8 bytes)
 * ================================================================ */

#define FAST_CMP_F64_AVX512(OP, PRED, TAIL_OP)                              \
  static inline void _fast_##OP##_f64_avx512(const void *restrict ap,       \
                                             const void *restrict bp,       \
                                             void *restrict op, size_t n) { \
    const double *a = (const double *)ap;                                   \
    const double *b = (const double *)bp;                                   \
    uint8_t *out = (uint8_t *)op;                                           \
    size_t i = 0;                                                           \
    for (; i + 8 <= n; i += 8) {                                            \
      __m512d va = _mm512_loadu_pd(a + i);                                  \
      __m512d vb = _mm512_loadu_pd(b + i);                                  \
      __mmask8 m = _mm512_cmp_pd_mask(va, vb, PRED);                        \
      _mm_storel_epi64((__m128i *)(out + i),                                \
                       _mm_maskz_set1_epi8((__mmask16)m, 1));               \
    }                                                                       \
    for (; i < n; i++)                                                      \
      out[i] = (uint8_t)(a[i] TAIL_OP b[i]);                                \
  }

FAST_CMP_F64_AVX512(eq, _CMP_EQ_OQ, ==)
FAST_CMP_F64_AVX512(gt, _CMP_GT_OQ, >)
FAST_CMP_F64_AVX512(lt, _CMP_LT_OQ, <)
FAST_CMP_F64_AVX512(ge, _CMP_GE_OQ, >=)
FAST_CMP_F64_AVX512(le, _CMP_LE_OQ, <=)
#undef FAST_CMP_F64_AVX512

#endif /* NUMC_COMPARE_AVX512_H */
