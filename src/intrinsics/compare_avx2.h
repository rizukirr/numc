/**
 * @file compare_avx2.h
 * @brief AVX2 binary comparison kernels — uint8 output (0/1).
 *
 * All comparison functions output uint8_t* (NumPy-compatible bool).
 * Uses pack-based narrowing for 16/32/64-bit types to minimize
 * output bandwidth.
 *
 * 8-bit:  compare → AND 1 → store 32 bytes per vector
 * 16-bit: compare 2 vectors → packs_epi16 → AND 1 → store 32 bytes
 * 32-bit: compare 4 vectors → packs_epi32 → packs_epi16 → permute → AND 1
 * 64-bit: compare 8 vectors → 3-stage pack → permute → AND 1
 */
#ifndef NUMC_COMPARE_AVX2_H
#define NUMC_COMPARE_AVX2_H

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

/* ====================================================================
 * 8-bit signed integer: 32 elems → 32 uint8 output per vector
 * ================================================================ */

#define FAST_CMP_8_AVX2(SUFFIX, CT, CMPEQ, CMPGT)                            \
  static inline void _fast_eq_##SUFFIX##_avx2(const void *restrict ap,       \
                                              const void *restrict bp,       \
                                              void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                            \
    const CT *b = (const CT *)bp;                                            \
    uint8_t *out = (uint8_t *)op;                                            \
    const __m256i one = _mm256_set1_epi8(1);                                 \
    size_t i = 0;                                                            \
    for (; i + 128 <= n; i += 128) {                                         \
      _mm_prefetch((const char *)(a + i + 256), _MM_HINT_T0);                \
      _mm_prefetch((const char *)(b + i + 256), _MM_HINT_T0);                \
      for (int k = 0; k < 4; k++) {                                          \
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i + k * 32));  \
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i + k * 32));  \
        _mm256_storeu_si256((__m256i *)(out + i + k * 32),                   \
                            _mm256_and_si256(CMPEQ(va, vb), one));           \
      }                                                                      \
    }                                                                        \
    for (; i + 32 <= n; i += 32) {                                           \
      __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));             \
      __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));             \
      _mm256_storeu_si256((__m256i *)(out + i),                              \
                          _mm256_and_si256(CMPEQ(va, vb), one));             \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (uint8_t)(a[i] == b[i]);                                      \
  }                                                                          \
  static inline void _fast_gt_##SUFFIX##_avx2(const void *restrict ap,       \
                                              const void *restrict bp,       \
                                              void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                            \
    const CT *b = (const CT *)bp;                                            \
    uint8_t *out = (uint8_t *)op;                                            \
    const __m256i one = _mm256_set1_epi8(1);                                 \
    size_t i = 0;                                                            \
    for (; i + 128 <= n; i += 128) {                                         \
      _mm_prefetch((const char *)(a + i + 256), _MM_HINT_T0);                \
      _mm_prefetch((const char *)(b + i + 256), _MM_HINT_T0);                \
      for (int k = 0; k < 4; k++) {                                          \
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i + k * 32));  \
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i + k * 32));  \
        _mm256_storeu_si256((__m256i *)(out + i + k * 32),                   \
                            _mm256_and_si256(CMPGT(va, vb), one));           \
      }                                                                      \
    }                                                                        \
    for (; i + 32 <= n; i += 32) {                                           \
      __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));             \
      __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));             \
      _mm256_storeu_si256((__m256i *)(out + i),                              \
                          _mm256_and_si256(CMPGT(va, vb), one));             \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (uint8_t)(a[i] > b[i]);                                       \
  }                                                                          \
  static inline void _fast_lt_##SUFFIX##_avx2(const void *restrict ap,       \
                                              const void *restrict bp,       \
                                              void *restrict op, size_t n) { \
    _fast_gt_##SUFFIX##_avx2(bp, ap, op, n);                                 \
  }                                                                          \
  static inline void _fast_ge_##SUFFIX##_avx2(const void *restrict ap,       \
                                              const void *restrict bp,       \
                                              void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                            \
    const CT *b = (const CT *)bp;                                            \
    uint8_t *out = (uint8_t *)op;                                            \
    const __m256i one = _mm256_set1_epi8(1);                                 \
    size_t i = 0;                                                            \
    for (; i + 128 <= n; i += 128) {                                         \
      _mm_prefetch((const char *)(a + i + 256), _MM_HINT_T0);                \
      _mm_prefetch((const char *)(b + i + 256), _MM_HINT_T0);                \
      for (int k = 0; k < 4; k++) {                                          \
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i + k * 32));  \
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i + k * 32));  \
        _mm256_storeu_si256((__m256i *)(out + i + k * 32),                   \
                            _mm256_andnot_si256(CMPGT(vb, va), one));        \
      }                                                                      \
    }                                                                        \
    for (; i + 32 <= n; i += 32) {                                           \
      __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));             \
      __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));             \
      _mm256_storeu_si256((__m256i *)(out + i),                              \
                          _mm256_andnot_si256(CMPGT(vb, va), one));          \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (uint8_t)(a[i] >= b[i]);                                      \
  }                                                                          \
  static inline void _fast_le_##SUFFIX##_avx2(const void *restrict ap,       \
                                              const void *restrict bp,       \
                                              void *restrict op, size_t n) { \
    _fast_ge_##SUFFIX##_avx2(bp, ap, op, n);                                 \
  }

FAST_CMP_8_AVX2(i8, int8_t, _mm256_cmpeq_epi8, _mm256_cmpgt_epi8)
#undef FAST_CMP_8_AVX2

/* 8-bit unsigned: eq uses cmpeq (no bias needed), gt/ge use XOR bias */
static inline void _fast_eq_u8_avx2(const void *restrict ap,
                                    const void *restrict bp, void *restrict op,
                                    size_t n) {
  const uint8_t *a = (const uint8_t *)ap;
  const uint8_t *b = (const uint8_t *)bp;
  uint8_t *out = (uint8_t *)op;
  const __m256i one = _mm256_set1_epi8(1);
  size_t i = 0;
  for (; i + 128 <= n; i += 128) {
    _mm_prefetch((const char *)(a + i + 256), _MM_HINT_T0);
    _mm_prefetch((const char *)(b + i + 256), _MM_HINT_T0);
    for (int k = 0; k < 4; k++) {
      __m256i va = _mm256_loadu_si256((const __m256i *)(a + i + k * 32));
      __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i + k * 32));
      _mm256_storeu_si256((__m256i *)(out + i + k * 32),
                          _mm256_and_si256(_mm256_cmpeq_epi8(va, vb), one));
    }
  }
  for (; i + 32 <= n; i += 32) {
    __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
    __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
    _mm256_storeu_si256((__m256i *)(out + i),
                        _mm256_and_si256(_mm256_cmpeq_epi8(va, vb), one));
  }
  for (; i < n; i++)
    out[i] = (uint8_t)(a[i] == b[i]);
}

static inline void _fast_gt_u8_avx2(const void *restrict ap,
                                    const void *restrict bp, void *restrict op,
                                    size_t n) {
  const uint8_t *a = (const uint8_t *)ap;
  const uint8_t *b = (const uint8_t *)bp;
  uint8_t *out = (uint8_t *)op;
  const __m256i bias = _mm256_set1_epi8((char)0x80);
  const __m256i one = _mm256_set1_epi8(1);
  size_t i = 0;
  for (; i + 128 <= n; i += 128) {
    _mm_prefetch((const char *)(a + i + 256), _MM_HINT_T0);
    _mm_prefetch((const char *)(b + i + 256), _MM_HINT_T0);
    for (int k = 0; k < 4; k++) {
      __m256i va = _mm256_xor_si256(
          _mm256_loadu_si256((const __m256i *)(a + i + k * 32)), bias);
      __m256i vb = _mm256_xor_si256(
          _mm256_loadu_si256((const __m256i *)(b + i + k * 32)), bias);
      _mm256_storeu_si256((__m256i *)(out + i + k * 32),
                          _mm256_and_si256(_mm256_cmpgt_epi8(va, vb), one));
    }
  }
  for (; i + 32 <= n; i += 32) {
    __m256i va =
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + i)), bias);
    __m256i vb =
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(b + i)), bias);
    _mm256_storeu_si256((__m256i *)(out + i),
                        _mm256_and_si256(_mm256_cmpgt_epi8(va, vb), one));
  }
  for (; i < n; i++)
    out[i] = (uint8_t)(a[i] > b[i]);
}

static inline void _fast_lt_u8_avx2(const void *restrict ap,
                                    const void *restrict bp, void *restrict op,
                                    size_t n) {
  _fast_gt_u8_avx2(bp, ap, op, n);
}

static inline void _fast_ge_u8_avx2(const void *restrict ap,
                                    const void *restrict bp, void *restrict op,
                                    size_t n) {
  const uint8_t *a = (const uint8_t *)ap;
  const uint8_t *b = (const uint8_t *)bp;
  uint8_t *out = (uint8_t *)op;
  const __m256i bias = _mm256_set1_epi8((char)0x80);
  const __m256i one = _mm256_set1_epi8(1);
  size_t i = 0;
  for (; i + 128 <= n; i += 128) {
    _mm_prefetch((const char *)(a + i + 256), _MM_HINT_T0);
    _mm_prefetch((const char *)(b + i + 256), _MM_HINT_T0);
    for (int k = 0; k < 4; k++) {
      __m256i va = _mm256_loadu_si256((const __m256i *)(a + i + k * 32));
      __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i + k * 32));
      __m256i ba = _mm256_xor_si256(vb, bias);
      __m256i bb = _mm256_xor_si256(va, bias);
      _mm256_storeu_si256((__m256i *)(out + i + k * 32),
                          _mm256_andnot_si256(_mm256_cmpgt_epi8(ba, bb), one));
    }
  }
  for (; i + 32 <= n; i += 32) {
    __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
    __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
    _mm256_storeu_si256(
        (__m256i *)(out + i),
        _mm256_andnot_si256(_mm256_cmpgt_epi8(_mm256_xor_si256(vb, bias),
                                              _mm256_xor_si256(va, bias)),
                            one));
  }
  for (; i < n; i++)
    out[i] = (uint8_t)(a[i] >= b[i]);
}

static inline void _fast_le_u8_avx2(const void *restrict ap,
                                    const void *restrict bp, void *restrict op,
                                    size_t n) {
  _fast_ge_u8_avx2(bp, ap, op, n);
}

/* ====================================================================
 * 16-bit: compare 2×16 elems → packs_epi16 → 32 uint8 output
 * ================================================================ */

#define FAST_CMP_16_AVX2(SUFFIX, CT, CMPEQ, CMPGT)                           \
  static inline void _fast_eq_##SUFFIX##_avx2(const void *restrict ap,       \
                                              const void *restrict bp,       \
                                              void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                            \
    const CT *b = (const CT *)bp;                                            \
    uint8_t *out = (uint8_t *)op;                                            \
    const __m256i one = _mm256_set1_epi8(1);                                 \
    size_t i = 0;                                                            \
    for (; i + 128 <= n; i += 128) {                                         \
      _mm_prefetch((const char *)(a + i + 256), _MM_HINT_T0);                \
      _mm_prefetch((const char *)(b + i + 256), _MM_HINT_T0);                \
      for (int k = 0; k < 4; k++) {                                          \
        size_t off = i + k * 32;                                             \
        __m256i a0 = _mm256_loadu_si256((const __m256i *)(a + off));         \
        __m256i b0 = _mm256_loadu_si256((const __m256i *)(b + off));         \
        __m256i a1 = _mm256_loadu_si256((const __m256i *)(a + off + 16));    \
        __m256i b1 = _mm256_loadu_si256((const __m256i *)(b + off + 16));    \
        __m256i c0 = CMPEQ(a0, b0);                                          \
        __m256i c1 = CMPEQ(a1, b1);                                          \
        __m256i packed = _mm256_packs_epi16(c0, c1);                         \
        packed = _mm256_permute4x64_epi64(packed, 0xD8);                     \
        _mm256_storeu_si256((__m256i *)(out + off),                          \
                            _mm256_and_si256(packed, one));                  \
      }                                                                      \
    }                                                                        \
    for (; i + 32 <= n; i += 32) {                                           \
      __m256i a0 = _mm256_loadu_si256((const __m256i *)(a + i));             \
      __m256i b0 = _mm256_loadu_si256((const __m256i *)(b + i));             \
      __m256i a1 = _mm256_loadu_si256((const __m256i *)(a + i + 16));        \
      __m256i b1 = _mm256_loadu_si256((const __m256i *)(b + i + 16));        \
      __m256i c0 = CMPEQ(a0, b0);                                            \
      __m256i c1 = CMPEQ(a1, b1);                                            \
      __m256i packed = _mm256_packs_epi16(c0, c1);                           \
      packed = _mm256_permute4x64_epi64(packed, 0xD8);                       \
      _mm256_storeu_si256((__m256i *)(out + i),                              \
                          _mm256_and_si256(packed, one));                    \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (uint8_t)(a[i] == b[i]);                                      \
  }                                                                          \
  static inline void _fast_gt_##SUFFIX##_avx2(const void *restrict ap,       \
                                              const void *restrict bp,       \
                                              void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                            \
    const CT *b = (const CT *)bp;                                            \
    uint8_t *out = (uint8_t *)op;                                            \
    const __m256i one = _mm256_set1_epi8(1);                                 \
    size_t i = 0;                                                            \
    for (; i + 128 <= n; i += 128) {                                         \
      _mm_prefetch((const char *)(a + i + 256), _MM_HINT_T0);                \
      _mm_prefetch((const char *)(b + i + 256), _MM_HINT_T0);                \
      for (int k = 0; k < 4; k++) {                                          \
        size_t off = i + k * 32;                                             \
        __m256i a0 = _mm256_loadu_si256((const __m256i *)(a + off));         \
        __m256i b0 = _mm256_loadu_si256((const __m256i *)(b + off));         \
        __m256i a1 = _mm256_loadu_si256((const __m256i *)(a + off + 16));    \
        __m256i b1 = _mm256_loadu_si256((const __m256i *)(b + off + 16));    \
        __m256i c0 = CMPGT(a0, b0);                                          \
        __m256i c1 = CMPGT(a1, b1);                                          \
        __m256i packed = _mm256_packs_epi16(c0, c1);                         \
        packed = _mm256_permute4x64_epi64(packed, 0xD8);                     \
        _mm256_storeu_si256((__m256i *)(out + off),                          \
                            _mm256_and_si256(packed, one));                  \
      }                                                                      \
    }                                                                        \
    for (; i + 32 <= n; i += 32) {                                           \
      __m256i a0 = _mm256_loadu_si256((const __m256i *)(a + i));             \
      __m256i b0 = _mm256_loadu_si256((const __m256i *)(b + i));             \
      __m256i a1 = _mm256_loadu_si256((const __m256i *)(a + i + 16));        \
      __m256i b1 = _mm256_loadu_si256((const __m256i *)(b + i + 16));        \
      __m256i c0 = CMPGT(a0, b0);                                            \
      __m256i c1 = CMPGT(a1, b1);                                            \
      __m256i packed = _mm256_packs_epi16(c0, c1);                           \
      packed = _mm256_permute4x64_epi64(packed, 0xD8);                       \
      _mm256_storeu_si256((__m256i *)(out + i),                              \
                          _mm256_and_si256(packed, one));                    \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (uint8_t)(a[i] > b[i]);                                       \
  }                                                                          \
  static inline void _fast_lt_##SUFFIX##_avx2(const void *restrict ap,       \
                                              const void *restrict bp,       \
                                              void *restrict op, size_t n) { \
    _fast_gt_##SUFFIX##_avx2(bp, ap, op, n);                                 \
  }                                                                          \
  static inline void _fast_ge_##SUFFIX##_avx2(const void *restrict ap,       \
                                              const void *restrict bp,       \
                                              void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                            \
    const CT *b = (const CT *)bp;                                            \
    uint8_t *out = (uint8_t *)op;                                            \
    const __m256i one = _mm256_set1_epi8(1);                                 \
    size_t i = 0;                                                            \
    for (; i + 128 <= n; i += 128) {                                         \
      _mm_prefetch((const char *)(a + i + 256), _MM_HINT_T0);                \
      _mm_prefetch((const char *)(b + i + 256), _MM_HINT_T0);                \
      for (int k = 0; k < 4; k++) {                                          \
        size_t off = i + k * 32;                                             \
        __m256i a0 = _mm256_loadu_si256((const __m256i *)(a + off));         \
        __m256i b0 = _mm256_loadu_si256((const __m256i *)(b + off));         \
        __m256i a1 = _mm256_loadu_si256((const __m256i *)(a + off + 16));    \
        __m256i b1 = _mm256_loadu_si256((const __m256i *)(b + off + 16));    \
        __m256i c0 = CMPGT(b0, a0);                                          \
        __m256i c1 = CMPGT(b1, a1);                                          \
        __m256i packed = _mm256_packs_epi16(c0, c1);                         \
        packed = _mm256_permute4x64_epi64(packed, 0xD8);                     \
        _mm256_storeu_si256((__m256i *)(out + off),                          \
                            _mm256_andnot_si256(packed, one));               \
      }                                                                      \
    }                                                                        \
    for (; i + 32 <= n; i += 32) {                                           \
      __m256i a0 = _mm256_loadu_si256((const __m256i *)(a + i));             \
      __m256i b0 = _mm256_loadu_si256((const __m256i *)(b + i));             \
      __m256i a1 = _mm256_loadu_si256((const __m256i *)(a + i + 16));        \
      __m256i b1 = _mm256_loadu_si256((const __m256i *)(b + i + 16));        \
      __m256i c0 = CMPGT(b0, a0);                                            \
      __m256i c1 = CMPGT(b1, a1);                                            \
      __m256i packed = _mm256_packs_epi16(c0, c1);                           \
      packed = _mm256_permute4x64_epi64(packed, 0xD8);                       \
      _mm256_storeu_si256((__m256i *)(out + i),                              \
                          _mm256_andnot_si256(packed, one));                 \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (uint8_t)(a[i] >= b[i]);                                      \
  }                                                                          \
  static inline void _fast_le_##SUFFIX##_avx2(const void *restrict ap,       \
                                              const void *restrict bp,       \
                                              void *restrict op, size_t n) { \
    _fast_ge_##SUFFIX##_avx2(bp, ap, op, n);                                 \
  }

FAST_CMP_16_AVX2(i16, int16_t, _mm256_cmpeq_epi16, _mm256_cmpgt_epi16)
#undef FAST_CMP_16_AVX2

/* 16-bit unsigned: eq no bias, gt/ge with XOR bias */
#define FAST_CMP_U16_AVX2(SUFFIX, CT, CMPEQ, CMPGT, BIAS)                    \
  static inline void _fast_eq_##SUFFIX##_avx2(const void *restrict ap,       \
                                              const void *restrict bp,       \
                                              void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                            \
    const CT *b = (const CT *)bp;                                            \
    uint8_t *out = (uint8_t *)op;                                            \
    const __m256i one = _mm256_set1_epi8(1);                                 \
    size_t i = 0;                                                            \
    for (; i + 128 <= n; i += 128) {                                         \
      _mm_prefetch((const char *)(a + i + 256), _MM_HINT_T0);                \
      _mm_prefetch((const char *)(b + i + 256), _MM_HINT_T0);                \
      for (int k = 0; k < 4; k++) {                                          \
        size_t off = i + k * 32;                                             \
        __m256i a0 = _mm256_loadu_si256((const __m256i *)(a + off));         \
        __m256i b0 = _mm256_loadu_si256((const __m256i *)(b + off));         \
        __m256i a1 = _mm256_loadu_si256((const __m256i *)(a + off + 16));    \
        __m256i b1 = _mm256_loadu_si256((const __m256i *)(b + off + 16));    \
        __m256i c0 = CMPEQ(a0, b0);                                          \
        __m256i c1 = CMPEQ(a1, b1);                                          \
        __m256i packed = _mm256_packs_epi16(c0, c1);                         \
        packed = _mm256_permute4x64_epi64(packed, 0xD8);                     \
        _mm256_storeu_si256((__m256i *)(out + off),                          \
                            _mm256_and_si256(packed, one));                  \
      }                                                                      \
    }                                                                        \
    for (; i + 32 <= n; i += 32) {                                           \
      __m256i a0 = _mm256_loadu_si256((const __m256i *)(a + i));             \
      __m256i b0 = _mm256_loadu_si256((const __m256i *)(b + i));             \
      __m256i a1 = _mm256_loadu_si256((const __m256i *)(a + i + 16));        \
      __m256i b1 = _mm256_loadu_si256((const __m256i *)(b + i + 16));        \
      __m256i c0 = CMPEQ(a0, b0);                                            \
      __m256i c1 = CMPEQ(a1, b1);                                            \
      __m256i packed = _mm256_packs_epi16(c0, c1);                           \
      packed = _mm256_permute4x64_epi64(packed, 0xD8);                       \
      _mm256_storeu_si256((__m256i *)(out + i),                              \
                          _mm256_and_si256(packed, one));                    \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (uint8_t)(a[i] == b[i]);                                      \
  }                                                                          \
  static inline void _fast_gt_##SUFFIX##_avx2(const void *restrict ap,       \
                                              const void *restrict bp,       \
                                              void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                            \
    const CT *b = (const CT *)bp;                                            \
    uint8_t *out = (uint8_t *)op;                                            \
    const __m256i bias = BIAS;                                               \
    const __m256i one = _mm256_set1_epi8(1);                                 \
    size_t i = 0;                                                            \
    for (; i + 128 <= n; i += 128) {                                         \
      _mm_prefetch((const char *)(a + i + 256), _MM_HINT_T0);                \
      _mm_prefetch((const char *)(b + i + 256), _MM_HINT_T0);                \
      for (int k = 0; k < 4; k++) {                                          \
        size_t off = i + k * 32;                                             \
        __m256i a0 = _mm256_xor_si256(                                       \
            _mm256_loadu_si256((const __m256i *)(a + off)), bias);           \
        __m256i b0 = _mm256_xor_si256(                                       \
            _mm256_loadu_si256((const __m256i *)(b + off)), bias);           \
        __m256i a1 = _mm256_xor_si256(                                       \
            _mm256_loadu_si256((const __m256i *)(a + off + 16)), bias);      \
        __m256i b1 = _mm256_xor_si256(                                       \
            _mm256_loadu_si256((const __m256i *)(b + off + 16)), bias);      \
        __m256i c0 = CMPGT(a0, b0);                                          \
        __m256i c1 = CMPGT(a1, b1);                                          \
        __m256i packed = _mm256_packs_epi16(c0, c1);                         \
        packed = _mm256_permute4x64_epi64(packed, 0xD8);                     \
        _mm256_storeu_si256((__m256i *)(out + off),                          \
                            _mm256_and_si256(packed, one));                  \
      }                                                                      \
    }                                                                        \
    for (; i + 32 <= n; i += 32) {                                           \
      __m256i a0 = _mm256_xor_si256(                                         \
          _mm256_loadu_si256((const __m256i *)(a + i)), bias);               \
      __m256i b0 = _mm256_xor_si256(                                         \
          _mm256_loadu_si256((const __m256i *)(b + i)), bias);               \
      __m256i a1 = _mm256_xor_si256(                                         \
          _mm256_loadu_si256((const __m256i *)(a + i + 16)), bias);          \
      __m256i b1 = _mm256_xor_si256(                                         \
          _mm256_loadu_si256((const __m256i *)(b + i + 16)), bias);          \
      __m256i c0 = CMPGT(a0, b0);                                            \
      __m256i c1 = CMPGT(a1, b1);                                            \
      __m256i packed = _mm256_packs_epi16(c0, c1);                           \
      packed = _mm256_permute4x64_epi64(packed, 0xD8);                       \
      _mm256_storeu_si256((__m256i *)(out + i),                              \
                          _mm256_and_si256(packed, one));                    \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (uint8_t)(a[i] > b[i]);                                       \
  }                                                                          \
  static inline void _fast_lt_##SUFFIX##_avx2(const void *restrict ap,       \
                                              const void *restrict bp,       \
                                              void *restrict op, size_t n) { \
    _fast_gt_##SUFFIX##_avx2(bp, ap, op, n);                                 \
  }                                                                          \
  static inline void _fast_ge_##SUFFIX##_avx2(const void *restrict ap,       \
                                              const void *restrict bp,       \
                                              void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                            \
    const CT *b = (const CT *)bp;                                            \
    uint8_t *out = (uint8_t *)op;                                            \
    const __m256i bias = BIAS;                                               \
    const __m256i one = _mm256_set1_epi8(1);                                 \
    size_t i = 0;                                                            \
    for (; i + 128 <= n; i += 128) {                                         \
      _mm_prefetch((const char *)(a + i + 256), _MM_HINT_T0);                \
      _mm_prefetch((const char *)(b + i + 256), _MM_HINT_T0);                \
      for (int k = 0; k < 4; k++) {                                          \
        size_t off = i + k * 32;                                             \
        __m256i a0 = _mm256_loadu_si256((const __m256i *)(a + off));         \
        __m256i b0 = _mm256_loadu_si256((const __m256i *)(b + off));         \
        __m256i a1 = _mm256_loadu_si256((const __m256i *)(a + off + 16));    \
        __m256i b1 = _mm256_loadu_si256((const __m256i *)(b + off + 16));    \
        __m256i c0 =                                                         \
            CMPGT(_mm256_xor_si256(b0, bias), _mm256_xor_si256(a0, bias));   \
        __m256i c1 =                                                         \
            CMPGT(_mm256_xor_si256(b1, bias), _mm256_xor_si256(a1, bias));   \
        __m256i packed = _mm256_packs_epi16(c0, c1);                         \
        packed = _mm256_permute4x64_epi64(packed, 0xD8);                     \
        _mm256_storeu_si256((__m256i *)(out + off),                          \
                            _mm256_andnot_si256(packed, one));               \
      }                                                                      \
    }                                                                        \
    for (; i + 32 <= n; i += 32) {                                           \
      __m256i a0 = _mm256_loadu_si256((const __m256i *)(a + i));             \
      __m256i b0 = _mm256_loadu_si256((const __m256i *)(b + i));             \
      __m256i a1 = _mm256_loadu_si256((const __m256i *)(a + i + 16));        \
      __m256i b1 = _mm256_loadu_si256((const __m256i *)(b + i + 16));        \
      __m256i c0 =                                                           \
          CMPGT(_mm256_xor_si256(b0, bias), _mm256_xor_si256(a0, bias));     \
      __m256i c1 =                                                           \
          CMPGT(_mm256_xor_si256(b1, bias), _mm256_xor_si256(a1, bias));     \
      __m256i packed = _mm256_packs_epi16(c0, c1);                           \
      packed = _mm256_permute4x64_epi64(packed, 0xD8);                       \
      _mm256_storeu_si256((__m256i *)(out + i),                              \
                          _mm256_andnot_si256(packed, one));                 \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (uint8_t)(a[i] >= b[i]);                                      \
  }                                                                          \
  static inline void _fast_le_##SUFFIX##_avx2(const void *restrict ap,       \
                                              const void *restrict bp,       \
                                              void *restrict op, size_t n) { \
    _fast_ge_##SUFFIX##_avx2(bp, ap, op, n);                                 \
  }

FAST_CMP_U16_AVX2(u16, uint16_t, _mm256_cmpeq_epi16, _mm256_cmpgt_epi16,
                  _mm256_set1_epi16((short)0x8000))
#undef FAST_CMP_U16_AVX2

/* ====================================================================
 * 32-bit: compare 4×8 elems → packs_epi32 → packs_epi16 → 32 uint8
 * ================================================================ */

/* Helper: pack 4 comparison results (32-bit masks) into 32 bytes of uint8 */
static inline __m256i _pack32_avx2(__m256i c0, __m256i c1, __m256i c2,
                                   __m256i c3, __m256i one) {
  __m256i p01 = _mm256_packs_epi32(c0, c1);
  __m256i p23 = _mm256_packs_epi32(c2, c3);
  __m256i p = _mm256_packs_epi16(p01, p23);
  p = _mm256_permute4x64_epi64(p, 0xD8);
  /* Fix interleaved order from nested packs: need additional shuffle */
  p = _mm256_shuffle_epi32(p, 0xD8);
  p = _mm256_shufflehi_epi16(p, 0xD8);
  p = _mm256_shufflelo_epi16(p, 0xD8);
  return _mm256_and_si256(p, one);
}

#define FAST_CMP_32_AVX2(SUFFIX, CT, CMPEQ, CMPGT, LOAD, TAIL_EQ, TAIL_GT,    \
                         TAIL_GE)                                             \
  static inline void _fast_eq_##SUFFIX##_avx2(const void *restrict ap,        \
                                              const void *restrict bp,        \
                                              void *restrict op, size_t n) {  \
    const CT *a = (const CT *)ap;                                             \
    const CT *b = (const CT *)bp;                                             \
    uint8_t *out = (uint8_t *)op;                                             \
    const __m256i one = _mm256_set1_epi8(1);                                  \
    size_t i = 0;                                                             \
    for (; i + 128 <= n; i += 128) {                                          \
      _mm_prefetch((const char *)(a + i + 256), _MM_HINT_T0);                 \
      _mm_prefetch((const char *)(b + i + 256), _MM_HINT_T0);                 \
      for (int g = 0; g < 4; g++) {                                           \
        size_t off = i + g * 32;                                              \
        __m256i c0 = CMPEQ(LOAD(a + off), LOAD(b + off));                     \
        __m256i c1 = CMPEQ(LOAD(a + off + 8), LOAD(b + off + 8));             \
        __m256i c2 = CMPEQ(LOAD(a + off + 16), LOAD(b + off + 16));           \
        __m256i c3 = CMPEQ(LOAD(a + off + 24), LOAD(b + off + 24));           \
        _mm256_storeu_si256((__m256i *)(out + off),                           \
                            _pack32_avx2(c0, c1, c2, c3, one));               \
      }                                                                       \
    }                                                                         \
    for (; i + 32 <= n; i += 32) {                                            \
      __m256i c0 = CMPEQ(LOAD(a + i), LOAD(b + i));                           \
      __m256i c1 = CMPEQ(LOAD(a + i + 8), LOAD(b + i + 8));                   \
      __m256i c2 = CMPEQ(LOAD(a + i + 16), LOAD(b + i + 16));                 \
      __m256i c3 = CMPEQ(LOAD(a + i + 24), LOAD(b + i + 24));                 \
      _mm256_storeu_si256((__m256i *)(out + i),                               \
                          _pack32_avx2(c0, c1, c2, c3, one));                 \
    }                                                                         \
    for (; i < n; i++)                                                        \
      out[i] = (uint8_t)TAIL_EQ;                                              \
  }                                                                           \
  static inline void _fast_gt_##SUFFIX##_avx2(const void *restrict ap,        \
                                              const void *restrict bp,        \
                                              void *restrict op, size_t n) {  \
    const CT *a = (const CT *)ap;                                             \
    const CT *b = (const CT *)bp;                                             \
    uint8_t *out = (uint8_t *)op;                                             \
    const __m256i one = _mm256_set1_epi8(1);                                  \
    size_t i = 0;                                                             \
    for (; i + 128 <= n; i += 128) {                                          \
      _mm_prefetch((const char *)(a + i + 256), _MM_HINT_T0);                 \
      _mm_prefetch((const char *)(b + i + 256), _MM_HINT_T0);                 \
      for (int g = 0; g < 4; g++) {                                           \
        size_t off = i + g * 32;                                              \
        __m256i c0 = CMPGT(LOAD(a + off), LOAD(b + off));                     \
        __m256i c1 = CMPGT(LOAD(a + off + 8), LOAD(b + off + 8));             \
        __m256i c2 = CMPGT(LOAD(a + off + 16), LOAD(b + off + 16));           \
        __m256i c3 = CMPGT(LOAD(a + off + 24), LOAD(b + off + 24));           \
        _mm256_storeu_si256((__m256i *)(out + off),                           \
                            _pack32_avx2(c0, c1, c2, c3, one));               \
      }                                                                       \
    }                                                                         \
    for (; i + 32 <= n; i += 32) {                                            \
      __m256i c0 = CMPGT(LOAD(a + i), LOAD(b + i));                           \
      __m256i c1 = CMPGT(LOAD(a + i + 8), LOAD(b + i + 8));                   \
      __m256i c2 = CMPGT(LOAD(a + i + 16), LOAD(b + i + 16));                 \
      __m256i c3 = CMPGT(LOAD(a + i + 24), LOAD(b + i + 24));                 \
      _mm256_storeu_si256((__m256i *)(out + i),                               \
                          _pack32_avx2(c0, c1, c2, c3, one));                 \
    }                                                                         \
    for (; i < n; i++)                                                        \
      out[i] = (uint8_t)TAIL_GT;                                              \
  }                                                                           \
  static inline void _fast_lt_##SUFFIX##_avx2(const void *restrict ap,        \
                                              const void *restrict bp,        \
                                              void *restrict op, size_t n) {  \
    _fast_gt_##SUFFIX##_avx2(bp, ap, op, n);                                  \
  }                                                                           \
  static inline void _fast_ge_##SUFFIX##_avx2(const void *restrict ap,        \
                                              const void *restrict bp,        \
                                              void *restrict op, size_t n) {  \
    const CT *a = (const CT *)ap;                                             \
    const CT *b = (const CT *)bp;                                             \
    uint8_t *out = (uint8_t *)op;                                             \
    const __m256i one = _mm256_set1_epi8(1);                                  \
    size_t i = 0;                                                             \
    for (; i + 128 <= n; i += 128) {                                          \
      _mm_prefetch((const char *)(a + i + 256), _MM_HINT_T0);                 \
      _mm_prefetch((const char *)(b + i + 256), _MM_HINT_T0);                 \
      for (int g = 0; g < 4; g++) {                                           \
        size_t off = i + g * 32;                                              \
        __m256i c0 = CMPGT(LOAD(b + off), LOAD(a + off));                     \
        __m256i c1 = CMPGT(LOAD(b + off + 8), LOAD(a + off + 8));             \
        __m256i c2 = CMPGT(LOAD(b + off + 16), LOAD(a + off + 16));           \
        __m256i c3 = CMPGT(LOAD(b + off + 24), LOAD(a + off + 24));           \
        __m256i p = _pack32_avx2(c0, c1, c2, c3, one);                        \
        _mm256_storeu_si256((__m256i *)(out + off),                           \
                            _mm256_andnot_si256(p, one));                     \
      }                                                                       \
    }                                                                         \
    for (; i + 32 <= n; i += 32) {                                            \
      __m256i c0 = CMPGT(LOAD(b + i), LOAD(a + i));                           \
      __m256i c1 = CMPGT(LOAD(b + i + 8), LOAD(a + i + 8));                   \
      __m256i c2 = CMPGT(LOAD(b + i + 16), LOAD(a + i + 16));                 \
      __m256i c3 = CMPGT(LOAD(b + i + 24), LOAD(a + i + 24));                 \
      __m256i p = _pack32_avx2(c0, c1, c2, c3, one);                          \
      _mm256_storeu_si256((__m256i *)(out + i), _mm256_andnot_si256(p, one)); \
    }                                                                         \
    for (; i < n; i++)                                                        \
      out[i] = (uint8_t)TAIL_GE;                                              \
  }                                                                           \
  static inline void _fast_le_##SUFFIX##_avx2(const void *restrict ap,        \
                                              const void *restrict bp,        \
                                              void *restrict op, size_t n) {  \
    _fast_ge_##SUFFIX##_avx2(bp, ap, op, n);                                  \
  }

/* i32 helpers */
static inline __m256i _loadi32(const int32_t *p) {
  return _mm256_loadu_si256((const __m256i *)p);
}
#define _CMPEQ_I32(a, b) _mm256_cmpeq_epi32(a, b)
#define _CMPGT_I32(a, b) _mm256_cmpgt_epi32(a, b)
FAST_CMP_32_AVX2(i32, int32_t, _CMPEQ_I32, _CMPGT_I32, _loadi32, (a[i] == b[i]),
                 (a[i] > b[i]), (a[i] >= b[i]))
#undef _CMPEQ_I32
#undef _CMPGT_I32

/* u32: eq uses cmpeq, gt/ge use XOR bias */
static inline __m256i _loadu32(const uint32_t *p) {
  return _mm256_loadu_si256((const __m256i *)p);
}
static inline __m256i _cmpeq_u32(const uint32_t *a, const uint32_t *b) {
  return _mm256_cmpeq_epi32(_mm256_loadu_si256((const __m256i *)a),
                            _mm256_loadu_si256((const __m256i *)b));
}

static inline void _fast_eq_u32_avx2(const void *restrict ap,
                                     const void *restrict bp, void *restrict op,
                                     size_t n) {
  const uint32_t *a = (const uint32_t *)ap;
  const uint32_t *b = (const uint32_t *)bp;
  uint8_t *out = (uint8_t *)op;
  const __m256i one = _mm256_set1_epi8(1);
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    __m256i c0 = _mm256_cmpeq_epi32(_loadu32(a + i), _loadu32(b + i));
    __m256i c1 = _mm256_cmpeq_epi32(_loadu32(a + i + 8), _loadu32(b + i + 8));
    __m256i c2 = _mm256_cmpeq_epi32(_loadu32(a + i + 16), _loadu32(b + i + 16));
    __m256i c3 = _mm256_cmpeq_epi32(_loadu32(a + i + 24), _loadu32(b + i + 24));
    _mm256_storeu_si256((__m256i *)(out + i),
                        _pack32_avx2(c0, c1, c2, c3, one));
  }
  for (; i < n; i++)
    out[i] = (uint8_t)(a[i] == b[i]);
}

static inline void _fast_gt_u32_avx2(const void *restrict ap,
                                     const void *restrict bp, void *restrict op,
                                     size_t n) {
  const uint32_t *a = (const uint32_t *)ap;
  const uint32_t *b = (const uint32_t *)bp;
  uint8_t *out = (uint8_t *)op;
  const __m256i bias = _mm256_set1_epi32((int)0x80000000);
  const __m256i one = _mm256_set1_epi8(1);
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    __m256i c0 = _mm256_cmpgt_epi32(_mm256_xor_si256(_loadu32(a + i), bias),
                                    _mm256_xor_si256(_loadu32(b + i), bias));
    __m256i c1 =
        _mm256_cmpgt_epi32(_mm256_xor_si256(_loadu32(a + i + 8), bias),
                           _mm256_xor_si256(_loadu32(b + i + 8), bias));
    __m256i c2 =
        _mm256_cmpgt_epi32(_mm256_xor_si256(_loadu32(a + i + 16), bias),
                           _mm256_xor_si256(_loadu32(b + i + 16), bias));
    __m256i c3 =
        _mm256_cmpgt_epi32(_mm256_xor_si256(_loadu32(a + i + 24), bias),
                           _mm256_xor_si256(_loadu32(b + i + 24), bias));
    _mm256_storeu_si256((__m256i *)(out + i),
                        _pack32_avx2(c0, c1, c2, c3, one));
  }
  for (; i < n; i++)
    out[i] = (uint8_t)(a[i] > b[i]);
}

static inline void _fast_lt_u32_avx2(const void *restrict ap,
                                     const void *restrict bp, void *restrict op,
                                     size_t n) {
  _fast_gt_u32_avx2(bp, ap, op, n);
}

static inline void _fast_ge_u32_avx2(const void *restrict ap,
                                     const void *restrict bp, void *restrict op,
                                     size_t n) {
  const uint32_t *a = (const uint32_t *)ap;
  const uint32_t *b = (const uint32_t *)bp;
  uint8_t *out = (uint8_t *)op;
  const __m256i bias = _mm256_set1_epi32((int)0x80000000);
  const __m256i one = _mm256_set1_epi8(1);
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    __m256i c0 = _mm256_cmpgt_epi32(_mm256_xor_si256(_loadu32(b + i), bias),
                                    _mm256_xor_si256(_loadu32(a + i), bias));
    __m256i c1 =
        _mm256_cmpgt_epi32(_mm256_xor_si256(_loadu32(b + i + 8), bias),
                           _mm256_xor_si256(_loadu32(a + i + 8), bias));
    __m256i c2 =
        _mm256_cmpgt_epi32(_mm256_xor_si256(_loadu32(b + i + 16), bias),
                           _mm256_xor_si256(_loadu32(a + i + 16), bias));
    __m256i c3 =
        _mm256_cmpgt_epi32(_mm256_xor_si256(_loadu32(b + i + 24), bias),
                           _mm256_xor_si256(_loadu32(a + i + 24), bias));
    __m256i p = _pack32_avx2(c0, c1, c2, c3, one);
    _mm256_storeu_si256((__m256i *)(out + i), _mm256_andnot_si256(p, one));
  }
  for (; i < n; i++)
    out[i] = (uint8_t)(a[i] >= b[i]);
}

static inline void _fast_le_u32_avx2(const void *restrict ap,
                                     const void *restrict bp, void *restrict op,
                                     size_t n) {
  _fast_ge_u32_avx2(bp, ap, op, n);
}

/* f32: cmp_ps returns __m256 → cast to __m256i for packing */
#define _CMP_F32(a, b, pred) \
  _mm256_castps_si256(       \
      _mm256_cmp_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b), pred))

#define FAST_CMP_F32_AVX2(OP, PRED, TAIL_OP)                              \
  static inline void _fast_##OP##_f32_avx2(const void *restrict ap,       \
                                           const void *restrict bp,       \
                                           void *restrict op, size_t n) { \
    const float *a = (const float *)ap;                                   \
    const float *b = (const float *)bp;                                   \
    uint8_t *out = (uint8_t *)op;                                         \
    const __m256i one = _mm256_set1_epi8(1);                              \
    size_t i = 0;                                                         \
    for (; i + 128 <= n; i += 128) {                                      \
      _mm_prefetch((const char *)(a + i + 256), _MM_HINT_T0);             \
      _mm_prefetch((const char *)(b + i + 256), _MM_HINT_T0);             \
      for (int g = 0; g < 4; g++) {                                       \
        size_t off = i + g * 32;                                          \
        __m256i c0 = _CMP_F32(a + off, b + off, PRED);                    \
        __m256i c1 = _CMP_F32(a + off + 8, b + off + 8, PRED);            \
        __m256i c2 = _CMP_F32(a + off + 16, b + off + 16, PRED);          \
        __m256i c3 = _CMP_F32(a + off + 24, b + off + 24, PRED);          \
        _mm256_storeu_si256((__m256i *)(out + off),                       \
                            _pack32_avx2(c0, c1, c2, c3, one));           \
      }                                                                   \
    }                                                                     \
    for (; i + 32 <= n; i += 32) {                                        \
      __m256i c0 = _CMP_F32(a + i, b + i, PRED);                          \
      __m256i c1 = _CMP_F32(a + i + 8, b + i + 8, PRED);                  \
      __m256i c2 = _CMP_F32(a + i + 16, b + i + 16, PRED);                \
      __m256i c3 = _CMP_F32(a + i + 24, b + i + 24, PRED);                \
      _mm256_storeu_si256((__m256i *)(out + i),                           \
                          _pack32_avx2(c0, c1, c2, c3, one));             \
    }                                                                     \
    for (; i < n; i++)                                                    \
      out[i] = (uint8_t)(a[i] TAIL_OP b[i]);                              \
  }

FAST_CMP_F32_AVX2(eq, _CMP_EQ_OQ, ==)
FAST_CMP_F32_AVX2(gt, _CMP_GT_OQ, >)
FAST_CMP_F32_AVX2(lt, _CMP_LT_OQ, <)
FAST_CMP_F32_AVX2(ge, _CMP_GE_OQ, >=)
FAST_CMP_F32_AVX2(le, _CMP_LE_OQ, <=)
#undef FAST_CMP_F32_AVX2
#undef _CMP_F32

/* ====================================================================
 * 64-bit: compare 8×4 elems → 3-stage pack → 32 uint8 output
 * ================================================================ */

/* 64-bit types use movemask + LUT for simplicity (packing 64→8 is complex) */
static const uint32_t _cmp_lut4_avx2[16] = {
    0x00000000, 0x00000001, 0x00000100, 0x00000101, 0x00010000, 0x00010001,
    0x00010100, 0x00010101, 0x01000000, 0x01000001, 0x01000100, 0x01000101,
    0x01010000, 0x01010001, 0x01010100, 0x01010101,
};

/* clang-format off */
/* i64 */
#define FAST_CMP_64_AVX2(SUFFIX, CT, CMPEQ, CMPGT, LOAD, MASK, TAIL_EQ,      \
                         TAIL_GT, TAIL_GE)                                   \
  static inline void _fast_eq_##SUFFIX##_avx2(const void *restrict ap,       \
                                              const void *restrict bp,       \
                                              void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                            \
    const CT *b = (const CT *)bp;                                            \
    uint8_t *out = (uint8_t *)op;                                            \
    size_t i = 0;                                                            \
    for (; i + 32 <= n; i += 32) {                                           \
      _mm_prefetch((const char *)(a + i + 32), _MM_HINT_T0);                 \
      _mm_prefetch((const char *)(b + i + 32), _MM_HINT_T0);                 \
      for (int k = 0; k < 8; k++) {                                          \
        int m = MASK(CMPEQ(LOAD(a + i + k * 4), LOAD(b + i + k * 4)));       \
        *(uint32_t *)(out + i + k * 4) = _cmp_lut4_avx2[m & 0xF];            \
      }                                                                      \
    }                                                                        \
    for (; i + 4 <= n; i += 4) {                                             \
      int m = MASK(CMPEQ(LOAD(a + i), LOAD(b + i)));                         \
      *(uint32_t *)(out + i) = _cmp_lut4_avx2[m & 0xF];                      \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (uint8_t)TAIL_EQ;                                             \
  }                                                                          \
  static inline void _fast_gt_##SUFFIX##_avx2(const void *restrict ap,       \
                                              const void *restrict bp,       \
                                              void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                            \
    const CT *b = (const CT *)bp;                                            \
    uint8_t *out = (uint8_t *)op;                                            \
    size_t i = 0;                                                            \
    for (; i + 32 <= n; i += 32) {                                           \
      _mm_prefetch((const char *)(a + i + 32), _MM_HINT_T0);                 \
      _mm_prefetch((const char *)(b + i + 32), _MM_HINT_T0);                 \
      for (int k = 0; k < 8; k++) {                                          \
        int m = MASK(CMPGT(LOAD(a + i + k * 4), LOAD(b + i + k * 4)));       \
        *(uint32_t *)(out + i + k * 4) = _cmp_lut4_avx2[m & 0xF];            \
      }                                                                      \
    }                                                                        \
    for (; i + 4 <= n; i += 4) {                                             \
      int m = MASK(CMPGT(LOAD(a + i), LOAD(b + i)));                         \
      *(uint32_t *)(out + i) = _cmp_lut4_avx2[m & 0xF];                      \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (uint8_t)TAIL_GT;                                             \
  }                                                                          \
  static inline void _fast_lt_##SUFFIX##_avx2(const void *restrict ap,       \
                                              const void *restrict bp,       \
                                              void *restrict op, size_t n) { \
    _fast_gt_##SUFFIX##_avx2(bp, ap, op, n);                                 \
  }                                                                          \
  static inline void _fast_ge_##SUFFIX##_avx2(const void *restrict ap,       \
                                              const void *restrict bp,       \
                                              void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                            \
    const CT *b = (const CT *)bp;                                            \
    uint8_t *out = (uint8_t *)op;                                            \
    size_t i = 0;                                                            \
    for (; i + 32 <= n; i += 32) {                                           \
      _mm_prefetch((const char *)(a + i + 32), _MM_HINT_T0);                 \
      _mm_prefetch((const char *)(b + i + 32), _MM_HINT_T0);                 \
      for (int k = 0; k < 8; k++) {                                          \
        int m = MASK(CMPGT(LOAD(b + i + k * 4), LOAD(a + i + k * 4)));       \
        *(uint32_t *)(out + i + k * 4) =                                     \
            _cmp_lut4_avx2[m & 0xF] ^ 0x01010101;                            \
      }                                                                      \
    }                                                                        \
    for (; i + 4 <= n; i += 4) {                                             \
      int m = MASK(CMPGT(LOAD(b + i), LOAD(a + i)));                         \
      *(uint32_t *)(out + i) = _cmp_lut4_avx2[m & 0xF] ^ 0x01010101;         \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (uint8_t)TAIL_GE;                                             \
  }                                                                          \
  static inline void _fast_le_##SUFFIX##_avx2(const void *restrict ap,       \
                                              const void *restrict bp,       \
                                              void *restrict op, size_t n) { \
    _fast_ge_##SUFFIX##_avx2(bp, ap, op, n);                                 \
  }
/* clang-format on */

static inline __m256i _loadi64(const int64_t *p) {
  return _mm256_loadu_si256((const __m256i *)p);
}
#define _MASKI64(v) _mm256_movemask_pd(_mm256_castsi256_pd(v))
FAST_CMP_64_AVX2(i64, int64_t, _mm256_cmpeq_epi64, _mm256_cmpgt_epi64, _loadi64,
                 _MASKI64, (a[i] == b[i]), (a[i] > b[i]), (a[i] >= b[i]))

/* u64: eq uses cmpeq, gt/ge use XOR bias */
static inline __m256i _loadu64(const uint64_t *p) {
  return _mm256_loadu_si256((const __m256i *)p);
}
static inline void _fast_eq_u64_avx2(const void *restrict ap,
                                     const void *restrict bp, void *restrict op,
                                     size_t n) {
  const uint64_t *a = (const uint64_t *)ap;
  const uint64_t *b = (const uint64_t *)bp;
  uint8_t *out = (uint8_t *)op;
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    _mm_prefetch((const char *)(a + i + 32), _MM_HINT_T0);
    _mm_prefetch((const char *)(b + i + 32), _MM_HINT_T0);
    for (int k = 0; k < 8; k++) {
      int m = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(
          _loadu64(a + i + k * 4), _loadu64(b + i + k * 4))));
      *(uint32_t *)(out + i + k * 4) = _cmp_lut4_avx2[m & 0xF];
    }
  }
  for (; i + 4 <= n; i += 4) {
    int m = _mm256_movemask_pd(_mm256_castsi256_pd(
        _mm256_cmpeq_epi64(_loadu64(a + i), _loadu64(b + i))));
    *(uint32_t *)(out + i) = _cmp_lut4_avx2[m & 0xF];
  }
  for (; i < n; i++)
    out[i] = (uint8_t)(a[i] == b[i]);
}

static inline void _fast_gt_u64_avx2(const void *restrict ap,
                                     const void *restrict bp, void *restrict op,
                                     size_t n) {
  const uint64_t *a = (const uint64_t *)ap;
  const uint64_t *b = (const uint64_t *)bp;
  uint8_t *out = (uint8_t *)op;
  const __m256i bias = _mm256_set1_epi64x((long long)0x8000000000000000LL);
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    _mm_prefetch((const char *)(a + i + 32), _MM_HINT_T0);
    _mm_prefetch((const char *)(b + i + 32), _MM_HINT_T0);
    for (int k = 0; k < 8; k++) {
      int m = _mm256_movemask_pd(_mm256_castsi256_pd(
          _mm256_cmpgt_epi64(_mm256_xor_si256(_loadu64(a + i + k * 4), bias),
                             _mm256_xor_si256(_loadu64(b + i + k * 4), bias))));
      *(uint32_t *)(out + i + k * 4) = _cmp_lut4_avx2[m & 0xF];
    }
  }
  for (; i + 4 <= n; i += 4) {
    int m = _mm256_movemask_pd(_mm256_castsi256_pd(
        _mm256_cmpgt_epi64(_mm256_xor_si256(_loadu64(a + i), bias),
                           _mm256_xor_si256(_loadu64(b + i), bias))));
    *(uint32_t *)(out + i) = _cmp_lut4_avx2[m & 0xF];
  }
  for (; i < n; i++)
    out[i] = (uint8_t)(a[i] > b[i]);
}

static inline void _fast_lt_u64_avx2(const void *restrict ap,
                                     const void *restrict bp, void *restrict op,
                                     size_t n) {
  _fast_gt_u64_avx2(bp, ap, op, n);
}

static inline void _fast_ge_u64_avx2(const void *restrict ap,
                                     const void *restrict bp, void *restrict op,
                                     size_t n) {
  const uint64_t *a = (const uint64_t *)ap;
  const uint64_t *b = (const uint64_t *)bp;
  uint8_t *out = (uint8_t *)op;
  const __m256i bias = _mm256_set1_epi64x((long long)0x8000000000000000LL);
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    _mm_prefetch((const char *)(a + i + 32), _MM_HINT_T0);
    _mm_prefetch((const char *)(b + i + 32), _MM_HINT_T0);
    for (int k = 0; k < 8; k++) {
      int m = _mm256_movemask_pd(_mm256_castsi256_pd(
          _mm256_cmpgt_epi64(_mm256_xor_si256(_loadu64(b + i + k * 4), bias),
                             _mm256_xor_si256(_loadu64(a + i + k * 4), bias))));
      *(uint32_t *)(out + i + k * 4) = _cmp_lut4_avx2[m & 0xF] ^ 0x01010101;
    }
  }
  for (; i + 4 <= n; i += 4) {
    int m = _mm256_movemask_pd(_mm256_castsi256_pd(
        _mm256_cmpgt_epi64(_mm256_xor_si256(_loadu64(b + i), bias),
                           _mm256_xor_si256(_loadu64(a + i), bias))));
    *(uint32_t *)(out + i) = _cmp_lut4_avx2[m & 0xF] ^ 0x01010101;
  }
  for (; i < n; i++)
    out[i] = (uint8_t)(a[i] >= b[i]);
}

static inline void _fast_le_u64_avx2(const void *restrict ap,
                                     const void *restrict bp, void *restrict op,
                                     size_t n) {
  _fast_ge_u64_avx2(bp, ap, op, n);
}

/* f64 */
#define FAST_CMP_F64_AVX2(OP, PRED, TAIL_OP)                                   \
  static inline void _fast_##OP##_f64_avx2(const void *restrict ap,            \
                                           const void *restrict bp,            \
                                           void *restrict op, size_t n) {      \
    const double *a = (const double *)ap;                                      \
    const double *b = (const double *)bp;                                      \
    uint8_t *out = (uint8_t *)op;                                              \
    size_t i = 0;                                                              \
    for (; i + 32 <= n; i += 32) {                                             \
      _mm_prefetch((const char *)(a + i + 32), _MM_HINT_T0);                   \
      _mm_prefetch((const char *)(b + i + 32), _MM_HINT_T0);                   \
      for (int k = 0; k < 8; k++) {                                            \
        int m = _mm256_movemask_pd(                                            \
            _mm256_cmp_pd(_mm256_loadu_pd(a + i + k * 4),                      \
                          _mm256_loadu_pd(b + i + k * 4), PRED));              \
        *(uint32_t *)(out + i + k * 4) = _cmp_lut4_avx2[m & 0xF];              \
      }                                                                        \
    }                                                                          \
    for (; i + 4 <= n; i += 4) {                                               \
      int m = _mm256_movemask_pd(_mm256_cmp_pd(_mm256_loadu_pd(a + i),         \
                                               _mm256_loadu_pd(b + i), PRED)); \
      *(uint32_t *)(out + i) = _cmp_lut4_avx2[m & 0xF];                        \
    }                                                                          \
    for (; i < n; i++)                                                         \
      out[i] = (uint8_t)(a[i] TAIL_OP b[i]);                                   \
  }

FAST_CMP_F64_AVX2(eq, _CMP_EQ_OQ, ==)
FAST_CMP_F64_AVX2(gt, _CMP_GT_OQ, >)
FAST_CMP_F64_AVX2(lt, _CMP_LT_OQ, <)
FAST_CMP_F64_AVX2(ge, _CMP_GE_OQ, >=)
FAST_CMP_F64_AVX2(le, _CMP_LE_OQ, <=)
#undef FAST_CMP_F64_AVX2

#undef FAST_CMP_64_AVX2
#undef _MASKI64
#undef FAST_CMP_32_AVX2

#endif /* NUMC_COMPARE_AVX2_H */
