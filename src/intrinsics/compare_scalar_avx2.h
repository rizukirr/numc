/**
 * @file compare_scalar_avx2.h
 * @brief AVX2 scalar comparison kernels — uint8 output (0/1).
 *
 * Each function compares array elements against a broadcast scalar,
 * storing 0 or 1 as uint8_t into the output (NumPy-compatible bool).
 *
 * 8-bit:  compare → AND 1 → store 32 bytes per vector (same width)
 * 16-bit: compare 2×16 elems → packs_epi16 → permute → AND 1 → 32 uint8
 * 32-bit: compare 4×8 elems → packs_epi32 → packs_epi16 → permute → 32 uint8
 * 64-bit: movemask + LUT table → 4 uint8 per vector
 * Floats: same narrowing as their integer width counterparts
 *
 * Scalar is pre-broadcast into a vector register before the loop.
 * 8x unrolled main loop + 1x cleanup + scalar tail.
 */
#ifndef NUMC_COMPARE_SCALAR_AVX2_H
#define NUMC_COMPARE_SCALAR_AVX2_H

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* -- Helpers ------------------------------------------------------- */

/* LUT: 4-bit movemask → 4 bytes of 0/1 (for 64-bit types) */
static const uint32_t _cmp_lut4_sc_avx2[16] = {
    0x00000000, 0x00000001, 0x00000100, 0x00000101, 0x00010000, 0x00010001,
    0x00010100, 0x00010101, 0x01000000, 0x01000001, 0x01000100, 0x01000101,
    0x01010000, 0x01010001, 0x01010100, 0x01010101,
};

/* Pack 4×8-element 32-bit comparison masks into 32 bytes of uint8 0/1 */
static inline __m256i _pack32_sc_avx2(__m256i c0, __m256i c1, __m256i c2,
                                      __m256i c3, __m256i one) {
  __m256i p01 = _mm256_packs_epi32(c0, c1);
  __m256i p23 = _mm256_packs_epi32(c2, c3);
  __m256i p = _mm256_packs_epi16(p01, p23);
  p = _mm256_permute4x64_epi64(p, 0xD8);
  p = _mm256_shuffle_epi32(p, 0xD8);
  p = _mm256_shufflehi_epi16(p, 0xD8);
  p = _mm256_shufflelo_epi16(p, 0xD8);
  return _mm256_and_si256(p, one);
}

/* ====================================================================
 * Float: f32 (8 per vector) → 32-bit packing path
 * ================================================================ */

#define CMPSC_F32_FUNC(NAME, IMM, SCALAR_OP)                                 \
  static inline void _cmpsc_##NAME##_f32_avx2(const void *restrict ap,       \
                                              const void *restrict sp,       \
                                              void *restrict op, size_t n) { \
    const float *a = (const float *)ap;                                      \
    const float s = *(const float *)sp;                                      \
    uint8_t *out = (uint8_t *)op;                                            \
    const __m256 vs = _mm256_set1_ps(s);                                     \
    const __m256i one = _mm256_set1_epi8(1);                                 \
    size_t i = 0;                                                            \
    /* 4×32 = 128 elements per group → 4 groups of 32-elem packing */        \
    for (; i + 128 <= n; i += 128) {                                         \
      _mm_prefetch((const char *)(a + i + 256), _MM_HINT_T0);                \
      for (int g = 0; g < 4; g++) {                                          \
        size_t off = i + g * 32;                                             \
        __m256i c0 = _mm256_castps_si256(                                    \
            _mm256_cmp_ps(_mm256_loadu_ps(a + off), vs, (IMM)));             \
        __m256i c1 = _mm256_castps_si256(                                    \
            _mm256_cmp_ps(_mm256_loadu_ps(a + off + 8), vs, (IMM)));         \
        __m256i c2 = _mm256_castps_si256(                                    \
            _mm256_cmp_ps(_mm256_loadu_ps(a + off + 16), vs, (IMM)));        \
        __m256i c3 = _mm256_castps_si256(                                    \
            _mm256_cmp_ps(_mm256_loadu_ps(a + off + 24), vs, (IMM)));        \
        _mm256_storeu_si256((__m256i *)(out + off),                          \
                            _pack32_sc_avx2(c0, c1, c2, c3, one));           \
      }                                                                      \
    }                                                                        \
    for (; i + 32 <= n; i += 32) {                                           \
      __m256i c0 = _mm256_castps_si256(                                      \
          _mm256_cmp_ps(_mm256_loadu_ps(a + i), vs, (IMM)));                 \
      __m256i c1 = _mm256_castps_si256(                                      \
          _mm256_cmp_ps(_mm256_loadu_ps(a + i + 8), vs, (IMM)));             \
      __m256i c2 = _mm256_castps_si256(                                      \
          _mm256_cmp_ps(_mm256_loadu_ps(a + i + 16), vs, (IMM)));            \
      __m256i c3 = _mm256_castps_si256(                                      \
          _mm256_cmp_ps(_mm256_loadu_ps(a + i + 24), vs, (IMM)));            \
      _mm256_storeu_si256((__m256i *)(out + i),                              \
                          _pack32_sc_avx2(c0, c1, c2, c3, one));             \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (uint8_t)(a[i] SCALAR_OP s);                                  \
  }

CMPSC_F32_FUNC(eq, _CMP_EQ_OQ, ==)
CMPSC_F32_FUNC(gt, _CMP_GT_OS, >)
CMPSC_F32_FUNC(lt, _CMP_LT_OS, <)
CMPSC_F32_FUNC(ge, _CMP_GE_OS, >=)
CMPSC_F32_FUNC(le, _CMP_LE_OS, <=)
#undef CMPSC_F32_FUNC

/* ====================================================================
 * Float: f64 (4 per vector) → movemask + LUT path
 * ================================================================ */

#define CMPSC_F64_FUNC(NAME, IMM, SCALAR_OP)                                 \
  static inline void _cmpsc_##NAME##_f64_avx2(const void *restrict ap,       \
                                              const void *restrict sp,       \
                                              void *restrict op, size_t n) { \
    const double *a = (const double *)ap;                                    \
    const double s = *(const double *)sp;                                    \
    uint8_t *out = (uint8_t *)op;                                            \
    const __m256d vs = _mm256_set1_pd(s);                                    \
    size_t i = 0;                                                            \
    for (; i + 32 <= n; i += 32) {                                           \
      _mm_prefetch((const char *)(a + i + 64), _MM_HINT_T0);                 \
      for (int k = 0; k < 8; k++) {                                          \
        int m = _mm256_movemask_pd(                                          \
            _mm256_cmp_pd(_mm256_loadu_pd(a + i + k * 4), vs, (IMM)));       \
        *(uint32_t *)(out + i + k * 4) = _cmp_lut4_sc_avx2[m & 0xF];         \
      }                                                                      \
    }                                                                        \
    for (; i + 4 <= n; i += 4) {                                             \
      int m = _mm256_movemask_pd(                                            \
          _mm256_cmp_pd(_mm256_loadu_pd(a + i), vs, (IMM)));                 \
      *(uint32_t *)(out + i) = _cmp_lut4_sc_avx2[m & 0xF];                   \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (uint8_t)(a[i] SCALAR_OP s);                                  \
  }

CMPSC_F64_FUNC(eq, _CMP_EQ_OQ, ==)
CMPSC_F64_FUNC(gt, _CMP_GT_OS, >)
CMPSC_F64_FUNC(lt, _CMP_LT_OS, <)
CMPSC_F64_FUNC(ge, _CMP_GE_OS, >=)
CMPSC_F64_FUNC(le, _CMP_LE_OS, <=)
#undef CMPSC_F64_FUNC

/* ====================================================================
 * 8-bit signed: i8 — 32 elems per vector, output is also 8-bit
 * ================================================================ */

#define CMPSC_I8_FUNC(NAME, CMP_EXPR, SCALAR_OP)                            \
  static inline void _cmpsc_##NAME##_i8_avx2(const void *restrict ap,       \
                                             const void *restrict sp,       \
                                             void *restrict op, size_t n) { \
    const int8_t *a = (const int8_t *)ap;                                   \
    const int8_t s = *(const int8_t *)sp;                                   \
    uint8_t *out = (uint8_t *)op;                                           \
    const __m256i vs = _mm256_set1_epi8(s);                                 \
    const __m256i one = _mm256_set1_epi8(1);                                \
    size_t i = 0;                                                           \
    for (; i + 256 <= n; i += 256) {                                        \
      _mm_prefetch((const char *)(a + i + 512), _MM_HINT_T0);               \
      for (int k = 0; k < 8; k++) {                                         \
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i + k * 32)); \
        _mm256_storeu_si256((__m256i *)(out + i + k * 32),                  \
                            _mm256_and_si256(CMP_EXPR, one));               \
      }                                                                     \
    }                                                                       \
    for (; i + 32 <= n; i += 32) {                                          \
      __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));            \
      _mm256_storeu_si256((__m256i *)(out + i),                             \
                          _mm256_and_si256(CMP_EXPR, one));                 \
    }                                                                       \
    for (; i < n; i++)                                                      \
      out[i] = (uint8_t)(a[i] SCALAR_OP s);                                 \
  }

CMPSC_I8_FUNC(eq, _mm256_cmpeq_epi8(va, vs), ==)
CMPSC_I8_FUNC(gt, _mm256_cmpgt_epi8(va, vs), >)
CMPSC_I8_FUNC(lt, _mm256_cmpgt_epi8(vs, va), <)
CMPSC_I8_FUNC(ge,
              _mm256_andnot_si256(_mm256_cmpgt_epi8(vs, va),
                                  _mm256_set1_epi8(-1)),
              >=)
CMPSC_I8_FUNC(le,
              _mm256_andnot_si256(_mm256_cmpgt_epi8(va, vs),
                                  _mm256_set1_epi8(-1)),
              <=)
#undef CMPSC_I8_FUNC

/* ====================================================================
 * 8-bit unsigned: u8 — eq uses cmpeq, gt/ge use XOR sign-bit bias
 * ================================================================ */

static inline void _cmpsc_eq_u8_avx2(const void *restrict ap,
                                     const void *restrict sp, void *restrict op,
                                     size_t n) {
  const uint8_t *a = (const uint8_t *)ap;
  const uint8_t s = *(const uint8_t *)sp;
  uint8_t *out = (uint8_t *)op;
  const __m256i vs = _mm256_set1_epi8((char)s);
  const __m256i one = _mm256_set1_epi8(1);
  size_t i = 0;
  for (; i + 256 <= n; i += 256) {
    _mm_prefetch((const char *)(a + i + 512), _MM_HINT_T0);
    for (int k = 0; k < 8; k++) {
      __m256i va = _mm256_loadu_si256((const __m256i *)(a + i + k * 32));
      _mm256_storeu_si256((__m256i *)(out + i + k * 32),
                          _mm256_and_si256(_mm256_cmpeq_epi8(va, vs), one));
    }
  }
  for (; i + 32 <= n; i += 32) {
    __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
    _mm256_storeu_si256((__m256i *)(out + i),
                        _mm256_and_si256(_mm256_cmpeq_epi8(va, vs), one));
  }
  for (; i < n; i++)
    out[i] = (uint8_t)(a[i] == s);
}

static inline void _cmpsc_gt_u8_avx2(const void *restrict ap,
                                     const void *restrict sp, void *restrict op,
                                     size_t n) {
  const uint8_t *a = (const uint8_t *)ap;
  const uint8_t s = *(const uint8_t *)sp;
  uint8_t *out = (uint8_t *)op;
  const __m256i bias = _mm256_set1_epi8((char)0x80);
  const __m256i vs_b = _mm256_xor_si256(_mm256_set1_epi8((char)s), bias);
  const __m256i one = _mm256_set1_epi8(1);
  size_t i = 0;
  for (; i + 256 <= n; i += 256) {
    _mm_prefetch((const char *)(a + i + 512), _MM_HINT_T0);
    for (int k = 0; k < 8; k++) {
      __m256i va = _mm256_xor_si256(
          _mm256_loadu_si256((const __m256i *)(a + i + k * 32)), bias);
      _mm256_storeu_si256((__m256i *)(out + i + k * 32),
                          _mm256_and_si256(_mm256_cmpgt_epi8(va, vs_b), one));
    }
  }
  for (; i + 32 <= n; i += 32) {
    __m256i va =
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + i)), bias);
    _mm256_storeu_si256((__m256i *)(out + i),
                        _mm256_and_si256(_mm256_cmpgt_epi8(va, vs_b), one));
  }
  for (; i < n; i++)
    out[i] = (uint8_t)(a[i] > s);
}

static inline void _cmpsc_lt_u8_avx2(const void *restrict ap,
                                     const void *restrict sp, void *restrict op,
                                     size_t n) {
  const uint8_t *a = (const uint8_t *)ap;
  const uint8_t s = *(const uint8_t *)sp;
  uint8_t *out = (uint8_t *)op;
  const __m256i bias = _mm256_set1_epi8((char)0x80);
  const __m256i vs_b = _mm256_xor_si256(_mm256_set1_epi8((char)s), bias);
  const __m256i one = _mm256_set1_epi8(1);
  size_t i = 0;
  for (; i + 256 <= n; i += 256) {
    _mm_prefetch((const char *)(a + i + 512), _MM_HINT_T0);
    for (int k = 0; k < 8; k++) {
      __m256i va = _mm256_xor_si256(
          _mm256_loadu_si256((const __m256i *)(a + i + k * 32)), bias);
      _mm256_storeu_si256((__m256i *)(out + i + k * 32),
                          _mm256_and_si256(_mm256_cmpgt_epi8(vs_b, va), one));
    }
  }
  for (; i + 32 <= n; i += 32) {
    __m256i va =
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + i)), bias);
    _mm256_storeu_si256((__m256i *)(out + i),
                        _mm256_and_si256(_mm256_cmpgt_epi8(vs_b, va), one));
  }
  for (; i < n; i++)
    out[i] = (uint8_t)(a[i] < s);
}

static inline void _cmpsc_ge_u8_avx2(const void *restrict ap,
                                     const void *restrict sp, void *restrict op,
                                     size_t n) {
  const uint8_t *a = (const uint8_t *)ap;
  const uint8_t s = *(const uint8_t *)sp;
  uint8_t *out = (uint8_t *)op;
  const __m256i bias = _mm256_set1_epi8((char)0x80);
  const __m256i vs_b = _mm256_xor_si256(_mm256_set1_epi8((char)s), bias);
  const __m256i one = _mm256_set1_epi8(1);
  size_t i = 0;
  for (; i + 256 <= n; i += 256) {
    _mm_prefetch((const char *)(a + i + 512), _MM_HINT_T0);
    for (int k = 0; k < 8; k++) {
      __m256i va = _mm256_xor_si256(
          _mm256_loadu_si256((const __m256i *)(a + i + k * 32)), bias);
      _mm256_storeu_si256(
          (__m256i *)(out + i + k * 32),
          _mm256_andnot_si256(_mm256_cmpgt_epi8(vs_b, va), one));
    }
  }
  for (; i + 32 <= n; i += 32) {
    __m256i va =
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + i)), bias);
    _mm256_storeu_si256((__m256i *)(out + i),
                        _mm256_andnot_si256(_mm256_cmpgt_epi8(vs_b, va), one));
  }
  for (; i < n; i++)
    out[i] = (uint8_t)(a[i] >= s);
}

static inline void _cmpsc_le_u8_avx2(const void *restrict ap,
                                     const void *restrict sp, void *restrict op,
                                     size_t n) {
  const uint8_t *a = (const uint8_t *)ap;
  const uint8_t s = *(const uint8_t *)sp;
  uint8_t *out = (uint8_t *)op;
  const __m256i bias = _mm256_set1_epi8((char)0x80);
  const __m256i vs_b = _mm256_xor_si256(_mm256_set1_epi8((char)s), bias);
  const __m256i one = _mm256_set1_epi8(1);
  size_t i = 0;
  for (; i + 256 <= n; i += 256) {
    _mm_prefetch((const char *)(a + i + 512), _MM_HINT_T0);
    for (int k = 0; k < 8; k++) {
      __m256i va = _mm256_xor_si256(
          _mm256_loadu_si256((const __m256i *)(a + i + k * 32)), bias);
      _mm256_storeu_si256(
          (__m256i *)(out + i + k * 32),
          _mm256_andnot_si256(_mm256_cmpgt_epi8(va, vs_b), one));
    }
  }
  for (; i + 32 <= n; i += 32) {
    __m256i va =
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + i)), bias);
    _mm256_storeu_si256((__m256i *)(out + i),
                        _mm256_andnot_si256(_mm256_cmpgt_epi8(va, vs_b), one));
  }
  for (; i < n; i++)
    out[i] = (uint8_t)(a[i] <= s);
}

/* ====================================================================
 * 16-bit signed: i16 — 16 per vector, pack 2→1 for uint8 output
 * ================================================================ */

#define CMPSC_I16_EQ_GT(NAME, CMP_EXPR, SCALAR_OP)                           \
  static inline void _cmpsc_##NAME##_i16_avx2(const void *restrict ap,       \
                                              const void *restrict sp,       \
                                              void *restrict op, size_t n) { \
    const int16_t *a = (const int16_t *)ap;                                  \
    const int16_t s = *(const int16_t *)sp;                                  \
    uint8_t *out = (uint8_t *)op;                                            \
    const __m256i vs = _mm256_set1_epi16(s);                                 \
    const __m256i one = _mm256_set1_epi8(1);                                 \
    size_t i = 0;                                                            \
    for (; i + 128 <= n; i += 128) {                                         \
      _mm_prefetch((const char *)(a + i + 256), _MM_HINT_T0);                \
      for (int k = 0; k < 4; k++) {                                          \
        size_t off = i + k * 32;                                             \
        __m256i a0 = _mm256_loadu_si256((const __m256i *)(a + off));         \
        __m256i a1 = _mm256_loadu_si256((const __m256i *)(a + off + 16));    \
        __m256i c0, c1;                                                      \
        {                                                                    \
          __m256i va = a0;                                                   \
          c0 = CMP_EXPR;                                                     \
        }                                                                    \
        {                                                                    \
          __m256i va = a1;                                                   \
          c1 = CMP_EXPR;                                                     \
        }                                                                    \
        __m256i packed = _mm256_packs_epi16(c0, c1);                         \
        packed = _mm256_permute4x64_epi64(packed, 0xD8);                     \
        _mm256_storeu_si256((__m256i *)(out + off),                          \
                            _mm256_and_si256(packed, one));                  \
      }                                                                      \
    }                                                                        \
    for (; i + 32 <= n; i += 32) {                                           \
      __m256i a0 = _mm256_loadu_si256((const __m256i *)(a + i));             \
      __m256i a1 = _mm256_loadu_si256((const __m256i *)(a + i + 16));        \
      __m256i c0, c1;                                                        \
      {                                                                      \
        __m256i va = a0;                                                     \
        c0 = CMP_EXPR;                                                       \
      }                                                                      \
      {                                                                      \
        __m256i va = a1;                                                     \
        c1 = CMP_EXPR;                                                       \
      }                                                                      \
      __m256i packed = _mm256_packs_epi16(c0, c1);                           \
      packed = _mm256_permute4x64_epi64(packed, 0xD8);                       \
      _mm256_storeu_si256((__m256i *)(out + i),                              \
                          _mm256_and_si256(packed, one));                    \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (uint8_t)(a[i] SCALAR_OP s);                                  \
  }

CMPSC_I16_EQ_GT(eq, _mm256_cmpeq_epi16(va, vs), ==)
CMPSC_I16_EQ_GT(gt, _mm256_cmpgt_epi16(va, vs), >)
CMPSC_I16_EQ_GT(lt, _mm256_cmpgt_epi16(vs, va), <)
#undef CMPSC_I16_EQ_GT

/* ge: NOT(s > a), le: NOT(a > s) — need andnot after packing */
#define CMPSC_I16_GE_LE(NAME, CMP_EXPR, SCALAR_OP)                           \
  static inline void _cmpsc_##NAME##_i16_avx2(const void *restrict ap,       \
                                              const void *restrict sp,       \
                                              void *restrict op, size_t n) { \
    const int16_t *a = (const int16_t *)ap;                                  \
    const int16_t s = *(const int16_t *)sp;                                  \
    uint8_t *out = (uint8_t *)op;                                            \
    const __m256i vs = _mm256_set1_epi16(s);                                 \
    const __m256i one = _mm256_set1_epi8(1);                                 \
    size_t i = 0;                                                            \
    for (; i + 128 <= n; i += 128) {                                         \
      _mm_prefetch((const char *)(a + i + 256), _MM_HINT_T0);                \
      for (int k = 0; k < 4; k++) {                                          \
        size_t off = i + k * 32;                                             \
        __m256i a0 = _mm256_loadu_si256((const __m256i *)(a + off));         \
        __m256i a1 = _mm256_loadu_si256((const __m256i *)(a + off + 16));    \
        __m256i c0, c1;                                                      \
        {                                                                    \
          __m256i va = a0;                                                   \
          c0 = CMP_EXPR;                                                     \
        }                                                                    \
        {                                                                    \
          __m256i va = a1;                                                   \
          c1 = CMP_EXPR;                                                     \
        }                                                                    \
        __m256i packed = _mm256_packs_epi16(c0, c1);                         \
        packed = _mm256_permute4x64_epi64(packed, 0xD8);                     \
        _mm256_storeu_si256((__m256i *)(out + off),                          \
                            _mm256_andnot_si256(packed, one));               \
      }                                                                      \
    }                                                                        \
    for (; i + 32 <= n; i += 32) {                                           \
      __m256i a0 = _mm256_loadu_si256((const __m256i *)(a + i));             \
      __m256i a1 = _mm256_loadu_si256((const __m256i *)(a + i + 16));        \
      __m256i c0, c1;                                                        \
      {                                                                      \
        __m256i va = a0;                                                     \
        c0 = CMP_EXPR;                                                       \
      }                                                                      \
      {                                                                      \
        __m256i va = a1;                                                     \
        c1 = CMP_EXPR;                                                       \
      }                                                                      \
      __m256i packed = _mm256_packs_epi16(c0, c1);                           \
      packed = _mm256_permute4x64_epi64(packed, 0xD8);                       \
      _mm256_storeu_si256((__m256i *)(out + i),                              \
                          _mm256_andnot_si256(packed, one));                 \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (uint8_t)(a[i] SCALAR_OP s);                                  \
  }

CMPSC_I16_GE_LE(ge, _mm256_cmpgt_epi16(vs, va), >=)
CMPSC_I16_GE_LE(le, _mm256_cmpgt_epi16(va, vs), <=)
#undef CMPSC_I16_GE_LE

/* ====================================================================
 * 16-bit unsigned: u16 — eq uses cmpeq, gt/ge use XOR sign-bit bias
 * ================================================================ */

static inline void _cmpsc_eq_u16_avx2(const void *restrict ap,
                                      const void *restrict sp,
                                      void *restrict op, size_t n) {
  const uint16_t *a = (const uint16_t *)ap;
  const uint16_t s = *(const uint16_t *)sp;
  uint8_t *out = (uint8_t *)op;
  const __m256i vs = _mm256_set1_epi16((short)s);
  const __m256i one = _mm256_set1_epi8(1);
  size_t i = 0;
  for (; i + 128 <= n; i += 128) {
    _mm_prefetch((const char *)(a + i + 256), _MM_HINT_T0);
    for (int k = 0; k < 4; k++) {
      size_t off = i + k * 32;
      __m256i a0 = _mm256_loadu_si256((const __m256i *)(a + off));
      __m256i a1 = _mm256_loadu_si256((const __m256i *)(a + off + 16));
      __m256i c0 = _mm256_cmpeq_epi16(a0, vs);
      __m256i c1 = _mm256_cmpeq_epi16(a1, vs);
      __m256i packed = _mm256_packs_epi16(c0, c1);
      packed = _mm256_permute4x64_epi64(packed, 0xD8);
      _mm256_storeu_si256((__m256i *)(out + off),
                          _mm256_and_si256(packed, one));
    }
  }
  for (; i + 32 <= n; i += 32) {
    __m256i a0 = _mm256_loadu_si256((const __m256i *)(a + i));
    __m256i a1 = _mm256_loadu_si256((const __m256i *)(a + i + 16));
    __m256i c0 = _mm256_cmpeq_epi16(a0, vs);
    __m256i c1 = _mm256_cmpeq_epi16(a1, vs);
    __m256i packed = _mm256_packs_epi16(c0, c1);
    packed = _mm256_permute4x64_epi64(packed, 0xD8);
    _mm256_storeu_si256((__m256i *)(out + i), _mm256_and_si256(packed, one));
  }
  for (; i < n; i++)
    out[i] = (uint8_t)(a[i] == s);
}

static inline void _cmpsc_gt_u16_avx2(const void *restrict ap,
                                      const void *restrict sp,
                                      void *restrict op, size_t n) {
  const uint16_t *a = (const uint16_t *)ap;
  const uint16_t s = *(const uint16_t *)sp;
  uint8_t *out = (uint8_t *)op;
  const __m256i bias = _mm256_set1_epi16((short)0x8000);
  const __m256i vs_b = _mm256_xor_si256(_mm256_set1_epi16((short)s), bias);
  const __m256i one = _mm256_set1_epi8(1);
  size_t i = 0;
  for (; i + 128 <= n; i += 128) {
    _mm_prefetch((const char *)(a + i + 256), _MM_HINT_T0);
    for (int k = 0; k < 4; k++) {
      size_t off = i + k * 32;
      __m256i a0 = _mm256_xor_si256(
          _mm256_loadu_si256((const __m256i *)(a + off)), bias);
      __m256i a1 = _mm256_xor_si256(
          _mm256_loadu_si256((const __m256i *)(a + off + 16)), bias);
      __m256i c0 = _mm256_cmpgt_epi16(a0, vs_b);
      __m256i c1 = _mm256_cmpgt_epi16(a1, vs_b);
      __m256i packed = _mm256_packs_epi16(c0, c1);
      packed = _mm256_permute4x64_epi64(packed, 0xD8);
      _mm256_storeu_si256((__m256i *)(out + off),
                          _mm256_and_si256(packed, one));
    }
  }
  for (; i + 32 <= n; i += 32) {
    __m256i a0 =
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + i)), bias);
    __m256i a1 = _mm256_xor_si256(
        _mm256_loadu_si256((const __m256i *)(a + i + 16)), bias);
    __m256i c0 = _mm256_cmpgt_epi16(a0, vs_b);
    __m256i c1 = _mm256_cmpgt_epi16(a1, vs_b);
    __m256i packed = _mm256_packs_epi16(c0, c1);
    packed = _mm256_permute4x64_epi64(packed, 0xD8);
    _mm256_storeu_si256((__m256i *)(out + i), _mm256_and_si256(packed, one));
  }
  for (; i < n; i++)
    out[i] = (uint8_t)(a[i] > s);
}

static inline void _cmpsc_lt_u16_avx2(const void *restrict ap,
                                      const void *restrict sp,
                                      void *restrict op, size_t n) {
  const uint16_t *a = (const uint16_t *)ap;
  const uint16_t s = *(const uint16_t *)sp;
  uint8_t *out = (uint8_t *)op;
  const __m256i bias = _mm256_set1_epi16((short)0x8000);
  const __m256i vs_b = _mm256_xor_si256(_mm256_set1_epi16((short)s), bias);
  const __m256i one = _mm256_set1_epi8(1);
  size_t i = 0;
  for (; i + 128 <= n; i += 128) {
    _mm_prefetch((const char *)(a + i + 256), _MM_HINT_T0);
    for (int k = 0; k < 4; k++) {
      size_t off = i + k * 32;
      __m256i a0 = _mm256_xor_si256(
          _mm256_loadu_si256((const __m256i *)(a + off)), bias);
      __m256i a1 = _mm256_xor_si256(
          _mm256_loadu_si256((const __m256i *)(a + off + 16)), bias);
      __m256i c0 = _mm256_cmpgt_epi16(vs_b, a0);
      __m256i c1 = _mm256_cmpgt_epi16(vs_b, a1);
      __m256i packed = _mm256_packs_epi16(c0, c1);
      packed = _mm256_permute4x64_epi64(packed, 0xD8);
      _mm256_storeu_si256((__m256i *)(out + off),
                          _mm256_and_si256(packed, one));
    }
  }
  for (; i + 32 <= n; i += 32) {
    __m256i a0 =
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + i)), bias);
    __m256i a1 = _mm256_xor_si256(
        _mm256_loadu_si256((const __m256i *)(a + i + 16)), bias);
    __m256i c0 = _mm256_cmpgt_epi16(vs_b, a0);
    __m256i c1 = _mm256_cmpgt_epi16(vs_b, a1);
    __m256i packed = _mm256_packs_epi16(c0, c1);
    packed = _mm256_permute4x64_epi64(packed, 0xD8);
    _mm256_storeu_si256((__m256i *)(out + i), _mm256_and_si256(packed, one));
  }
  for (; i < n; i++)
    out[i] = (uint8_t)(a[i] < s);
}

static inline void _cmpsc_ge_u16_avx2(const void *restrict ap,
                                      const void *restrict sp,
                                      void *restrict op, size_t n) {
  const uint16_t *a = (const uint16_t *)ap;
  const uint16_t s = *(const uint16_t *)sp;
  uint8_t *out = (uint8_t *)op;
  const __m256i bias = _mm256_set1_epi16((short)0x8000);
  const __m256i vs_b = _mm256_xor_si256(_mm256_set1_epi16((short)s), bias);
  const __m256i one = _mm256_set1_epi8(1);
  size_t i = 0;
  for (; i + 128 <= n; i += 128) {
    _mm_prefetch((const char *)(a + i + 256), _MM_HINT_T0);
    for (int k = 0; k < 4; k++) {
      size_t off = i + k * 32;
      __m256i a0 = _mm256_xor_si256(
          _mm256_loadu_si256((const __m256i *)(a + off)), bias);
      __m256i a1 = _mm256_xor_si256(
          _mm256_loadu_si256((const __m256i *)(a + off + 16)), bias);
      __m256i c0 = _mm256_cmpgt_epi16(vs_b, a0);
      __m256i c1 = _mm256_cmpgt_epi16(vs_b, a1);
      __m256i packed = _mm256_packs_epi16(c0, c1);
      packed = _mm256_permute4x64_epi64(packed, 0xD8);
      _mm256_storeu_si256((__m256i *)(out + off),
                          _mm256_andnot_si256(packed, one));
    }
  }
  for (; i + 32 <= n; i += 32) {
    __m256i a0 =
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + i)), bias);
    __m256i a1 = _mm256_xor_si256(
        _mm256_loadu_si256((const __m256i *)(a + i + 16)), bias);
    __m256i c0 = _mm256_cmpgt_epi16(vs_b, a0);
    __m256i c1 = _mm256_cmpgt_epi16(vs_b, a1);
    __m256i packed = _mm256_packs_epi16(c0, c1);
    packed = _mm256_permute4x64_epi64(packed, 0xD8);
    _mm256_storeu_si256((__m256i *)(out + i), _mm256_andnot_si256(packed, one));
  }
  for (; i < n; i++)
    out[i] = (uint8_t)(a[i] >= s);
}

static inline void _cmpsc_le_u16_avx2(const void *restrict ap,
                                      const void *restrict sp,
                                      void *restrict op, size_t n) {
  const uint16_t *a = (const uint16_t *)ap;
  const uint16_t s = *(const uint16_t *)sp;
  uint8_t *out = (uint8_t *)op;
  const __m256i bias = _mm256_set1_epi16((short)0x8000);
  const __m256i vs_b = _mm256_xor_si256(_mm256_set1_epi16((short)s), bias);
  const __m256i one = _mm256_set1_epi8(1);
  size_t i = 0;
  for (; i + 128 <= n; i += 128) {
    _mm_prefetch((const char *)(a + i + 256), _MM_HINT_T0);
    for (int k = 0; k < 4; k++) {
      size_t off = i + k * 32;
      __m256i a0 = _mm256_xor_si256(
          _mm256_loadu_si256((const __m256i *)(a + off)), bias);
      __m256i a1 = _mm256_xor_si256(
          _mm256_loadu_si256((const __m256i *)(a + off + 16)), bias);
      __m256i c0 = _mm256_cmpgt_epi16(a0, vs_b);
      __m256i c1 = _mm256_cmpgt_epi16(a1, vs_b);
      __m256i packed = _mm256_packs_epi16(c0, c1);
      packed = _mm256_permute4x64_epi64(packed, 0xD8);
      _mm256_storeu_si256((__m256i *)(out + off),
                          _mm256_andnot_si256(packed, one));
    }
  }
  for (; i + 32 <= n; i += 32) {
    __m256i a0 =
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + i)), bias);
    __m256i a1 = _mm256_xor_si256(
        _mm256_loadu_si256((const __m256i *)(a + i + 16)), bias);
    __m256i c0 = _mm256_cmpgt_epi16(a0, vs_b);
    __m256i c1 = _mm256_cmpgt_epi16(a1, vs_b);
    __m256i packed = _mm256_packs_epi16(c0, c1);
    packed = _mm256_permute4x64_epi64(packed, 0xD8);
    _mm256_storeu_si256((__m256i *)(out + i), _mm256_andnot_si256(packed, one));
  }
  for (; i < n; i++)
    out[i] = (uint8_t)(a[i] <= s);
}

/* ====================================================================
 * 32-bit signed: i32 — 8 per vector, pack 4→1 for uint8 output
 * ================================================================ */

#define CMPSC_I32_FUNC(NAME, CMP_CALL, SCALAR_OP)                              \
  static inline void _cmpsc_##NAME##_i32_avx2(const void *restrict ap,         \
                                              const void *restrict sp,         \
                                              void *restrict op, size_t n) {   \
    const int32_t *a = (const int32_t *)ap;                                    \
    const int32_t s = *(const int32_t *)sp;                                    \
    uint8_t *out = (uint8_t *)op;                                              \
    const __m256i vs = _mm256_set1_epi32(s);                                   \
    const __m256i one = _mm256_set1_epi8(1);                                   \
    size_t i = 0;                                                              \
    for (; i + 128 <= n; i += 128) {                                           \
      _mm_prefetch((const char *)(a + i + 256), _MM_HINT_T0);                  \
      for (int g = 0; g < 4; g++) {                                            \
        size_t off = i + g * 32;                                               \
        __m256i c0 =                                                           \
            CMP_CALL(_mm256_loadu_si256((const __m256i *)(a + off)), vs);      \
        __m256i c1 =                                                           \
            CMP_CALL(_mm256_loadu_si256((const __m256i *)(a + off + 8)), vs);  \
        __m256i c2 =                                                           \
            CMP_CALL(_mm256_loadu_si256((const __m256i *)(a + off + 16)), vs); \
        __m256i c3 =                                                           \
            CMP_CALL(_mm256_loadu_si256((const __m256i *)(a + off + 24)), vs); \
        _mm256_storeu_si256((__m256i *)(out + off),                            \
                            _pack32_sc_avx2(c0, c1, c2, c3, one));             \
      }                                                                        \
    }                                                                          \
    for (; i + 32 <= n; i += 32) {                                             \
      __m256i c0 = CMP_CALL(_mm256_loadu_si256((const __m256i *)(a + i)), vs); \
      __m256i c1 =                                                             \
          CMP_CALL(_mm256_loadu_si256((const __m256i *)(a + i + 8)), vs);      \
      __m256i c2 =                                                             \
          CMP_CALL(_mm256_loadu_si256((const __m256i *)(a + i + 16)), vs);     \
      __m256i c3 =                                                             \
          CMP_CALL(_mm256_loadu_si256((const __m256i *)(a + i + 24)), vs);     \
      _mm256_storeu_si256((__m256i *)(out + i),                                \
                          _pack32_sc_avx2(c0, c1, c2, c3, one));               \
    }                                                                          \
    for (; i < n; i++)                                                         \
      out[i] = (uint8_t)(a[i] SCALAR_OP s);                                    \
  }

#define _SC_CMPEQ_I32(va, vs) _mm256_cmpeq_epi32(va, vs)
#define _SC_CMPGT_I32(va, vs) _mm256_cmpgt_epi32(va, vs)
#define _SC_CMPLT_I32(va, vs) _mm256_cmpgt_epi32(vs, va)

CMPSC_I32_FUNC(eq, _SC_CMPEQ_I32, ==)
CMPSC_I32_FUNC(gt, _SC_CMPGT_I32, >)
CMPSC_I32_FUNC(lt, _SC_CMPLT_I32, <)
#undef CMPSC_I32_FUNC

/* ge/le: need andnot after packing */
#define CMPSC_I32_GE_LE(NAME, CMP_CALL, SCALAR_OP)                             \
  static inline void _cmpsc_##NAME##_i32_avx2(const void *restrict ap,         \
                                              const void *restrict sp,         \
                                              void *restrict op, size_t n) {   \
    const int32_t *a = (const int32_t *)ap;                                    \
    const int32_t s = *(const int32_t *)sp;                                    \
    uint8_t *out = (uint8_t *)op;                                              \
    const __m256i vs = _mm256_set1_epi32(s);                                   \
    const __m256i one = _mm256_set1_epi8(1);                                   \
    size_t i = 0;                                                              \
    for (; i + 128 <= n; i += 128) {                                           \
      _mm_prefetch((const char *)(a + i + 256), _MM_HINT_T0);                  \
      for (int g = 0; g < 4; g++) {                                            \
        size_t off = i + g * 32;                                               \
        __m256i c0 =                                                           \
            CMP_CALL(_mm256_loadu_si256((const __m256i *)(a + off)), vs);      \
        __m256i c1 =                                                           \
            CMP_CALL(_mm256_loadu_si256((const __m256i *)(a + off + 8)), vs);  \
        __m256i c2 =                                                           \
            CMP_CALL(_mm256_loadu_si256((const __m256i *)(a + off + 16)), vs); \
        __m256i c3 =                                                           \
            CMP_CALL(_mm256_loadu_si256((const __m256i *)(a + off + 24)), vs); \
        __m256i p = _pack32_sc_avx2(c0, c1, c2, c3, one);                      \
        _mm256_storeu_si256((__m256i *)(out + off),                            \
                            _mm256_andnot_si256(p, one));                      \
      }                                                                        \
    }                                                                          \
    for (; i + 32 <= n; i += 32) {                                             \
      __m256i c0 = CMP_CALL(_mm256_loadu_si256((const __m256i *)(a + i)), vs); \
      __m256i c1 =                                                             \
          CMP_CALL(_mm256_loadu_si256((const __m256i *)(a + i + 8)), vs);      \
      __m256i c2 =                                                             \
          CMP_CALL(_mm256_loadu_si256((const __m256i *)(a + i + 16)), vs);     \
      __m256i c3 =                                                             \
          CMP_CALL(_mm256_loadu_si256((const __m256i *)(a + i + 24)), vs);     \
      __m256i p = _pack32_sc_avx2(c0, c1, c2, c3, one);                        \
      _mm256_storeu_si256((__m256i *)(out + i), _mm256_andnot_si256(p, one));  \
    }                                                                          \
    for (; i < n; i++)                                                         \
      out[i] = (uint8_t)(a[i] SCALAR_OP s);                                    \
  }

CMPSC_I32_GE_LE(ge, _SC_CMPLT_I32, >=)
CMPSC_I32_GE_LE(le, _SC_CMPGT_I32, <=)
#undef CMPSC_I32_GE_LE
#undef _SC_CMPEQ_I32
#undef _SC_CMPGT_I32
#undef _SC_CMPLT_I32

/* ====================================================================
 * 32-bit unsigned: u32 — eq uses cmpeq, gt/ge use XOR sign-bit bias
 * ================================================================ */

static inline void _cmpsc_eq_u32_avx2(const void *restrict ap,
                                      const void *restrict sp,
                                      void *restrict op, size_t n) {
  const uint32_t *a = (const uint32_t *)ap;
  const uint32_t s = *(const uint32_t *)sp;
  uint8_t *out = (uint8_t *)op;
  const __m256i vs = _mm256_set1_epi32((int)s);
  const __m256i one = _mm256_set1_epi8(1);
  size_t i = 0;
  for (; i + 128 <= n; i += 128) {
    _mm_prefetch((const char *)(a + i + 256), _MM_HINT_T0);
    for (int g = 0; g < 4; g++) {
      size_t off = i + g * 32;
      __m256i c0 = _mm256_cmpeq_epi32(
          _mm256_loadu_si256((const __m256i *)(a + off)), vs);
      __m256i c1 = _mm256_cmpeq_epi32(
          _mm256_loadu_si256((const __m256i *)(a + off + 8)), vs);
      __m256i c2 = _mm256_cmpeq_epi32(
          _mm256_loadu_si256((const __m256i *)(a + off + 16)), vs);
      __m256i c3 = _mm256_cmpeq_epi32(
          _mm256_loadu_si256((const __m256i *)(a + off + 24)), vs);
      _mm256_storeu_si256((__m256i *)(out + off),
                          _pack32_sc_avx2(c0, c1, c2, c3, one));
    }
  }
  for (; i + 32 <= n; i += 32) {
    __m256i c0 =
        _mm256_cmpeq_epi32(_mm256_loadu_si256((const __m256i *)(a + i)), vs);
    __m256i c1 = _mm256_cmpeq_epi32(
        _mm256_loadu_si256((const __m256i *)(a + i + 8)), vs);
    __m256i c2 = _mm256_cmpeq_epi32(
        _mm256_loadu_si256((const __m256i *)(a + i + 16)), vs);
    __m256i c3 = _mm256_cmpeq_epi32(
        _mm256_loadu_si256((const __m256i *)(a + i + 24)), vs);
    _mm256_storeu_si256((__m256i *)(out + i),
                        _pack32_sc_avx2(c0, c1, c2, c3, one));
  }
  for (; i < n; i++)
    out[i] = (uint8_t)(a[i] == s);
}

static inline void _cmpsc_gt_u32_avx2(const void *restrict ap,
                                      const void *restrict sp,
                                      void *restrict op, size_t n) {
  const uint32_t *a = (const uint32_t *)ap;
  const uint32_t s = *(const uint32_t *)sp;
  uint8_t *out = (uint8_t *)op;
  const __m256i bias = _mm256_set1_epi32((int)0x80000000);
  const __m256i vs_b = _mm256_xor_si256(_mm256_set1_epi32((int)s), bias);
  const __m256i one = _mm256_set1_epi8(1);
  size_t i = 0;
  for (; i + 128 <= n; i += 128) {
    _mm_prefetch((const char *)(a + i + 256), _MM_HINT_T0);
    for (int g = 0; g < 4; g++) {
      size_t off = i + g * 32;
      __m256i c0 = _mm256_cmpgt_epi32(
          _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + off)),
                           bias),
          vs_b);
      __m256i c1 = _mm256_cmpgt_epi32(
          _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + off + 8)),
                           bias),
          vs_b);
      __m256i c2 = _mm256_cmpgt_epi32(
          _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + off + 16)),
                           bias),
          vs_b);
      __m256i c3 = _mm256_cmpgt_epi32(
          _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + off + 24)),
                           bias),
          vs_b);
      _mm256_storeu_si256((__m256i *)(out + off),
                          _pack32_sc_avx2(c0, c1, c2, c3, one));
    }
  }
  for (; i + 32 <= n; i += 32) {
    __m256i c0 = _mm256_cmpgt_epi32(
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + i)), bias),
        vs_b);
    __m256i c1 = _mm256_cmpgt_epi32(
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + i + 8)),
                         bias),
        vs_b);
    __m256i c2 = _mm256_cmpgt_epi32(
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + i + 16)),
                         bias),
        vs_b);
    __m256i c3 = _mm256_cmpgt_epi32(
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + i + 24)),
                         bias),
        vs_b);
    _mm256_storeu_si256((__m256i *)(out + i),
                        _pack32_sc_avx2(c0, c1, c2, c3, one));
  }
  for (; i < n; i++)
    out[i] = (uint8_t)(a[i] > s);
}

static inline void _cmpsc_lt_u32_avx2(const void *restrict ap,
                                      const void *restrict sp,
                                      void *restrict op, size_t n) {
  const uint32_t *a = (const uint32_t *)ap;
  const uint32_t s = *(const uint32_t *)sp;
  uint8_t *out = (uint8_t *)op;
  const __m256i bias = _mm256_set1_epi32((int)0x80000000);
  const __m256i vs_b = _mm256_xor_si256(_mm256_set1_epi32((int)s), bias);
  const __m256i one = _mm256_set1_epi8(1);
  size_t i = 0;
  for (; i + 128 <= n; i += 128) {
    _mm_prefetch((const char *)(a + i + 256), _MM_HINT_T0);
    for (int g = 0; g < 4; g++) {
      size_t off = i + g * 32;
      __m256i c0 = _mm256_cmpgt_epi32(
          vs_b, _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + off)),
                                 bias));
      __m256i c1 = _mm256_cmpgt_epi32(
          vs_b, _mm256_xor_si256(
                    _mm256_loadu_si256((const __m256i *)(a + off + 8)), bias));
      __m256i c2 = _mm256_cmpgt_epi32(
          vs_b, _mm256_xor_si256(
                    _mm256_loadu_si256((const __m256i *)(a + off + 16)), bias));
      __m256i c3 = _mm256_cmpgt_epi32(
          vs_b, _mm256_xor_si256(
                    _mm256_loadu_si256((const __m256i *)(a + off + 24)), bias));
      _mm256_storeu_si256((__m256i *)(out + off),
                          _pack32_sc_avx2(c0, c1, c2, c3, one));
    }
  }
  for (; i + 32 <= n; i += 32) {
    __m256i c0 = _mm256_cmpgt_epi32(
        vs_b,
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + i)), bias));
    __m256i c1 = _mm256_cmpgt_epi32(
        vs_b, _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + i + 8)),
                               bias));
    __m256i c2 = _mm256_cmpgt_epi32(
        vs_b, _mm256_xor_si256(
                  _mm256_loadu_si256((const __m256i *)(a + i + 16)), bias));
    __m256i c3 = _mm256_cmpgt_epi32(
        vs_b, _mm256_xor_si256(
                  _mm256_loadu_si256((const __m256i *)(a + i + 24)), bias));
    _mm256_storeu_si256((__m256i *)(out + i),
                        _pack32_sc_avx2(c0, c1, c2, c3, one));
  }
  for (; i < n; i++)
    out[i] = (uint8_t)(a[i] < s);
}

static inline void _cmpsc_ge_u32_avx2(const void *restrict ap,
                                      const void *restrict sp,
                                      void *restrict op, size_t n) {
  const uint32_t *a = (const uint32_t *)ap;
  const uint32_t s = *(const uint32_t *)sp;
  uint8_t *out = (uint8_t *)op;
  const __m256i bias = _mm256_set1_epi32((int)0x80000000);
  const __m256i vs_b = _mm256_xor_si256(_mm256_set1_epi32((int)s), bias);
  const __m256i one = _mm256_set1_epi8(1);
  size_t i = 0;
  for (; i + 128 <= n; i += 128) {
    _mm_prefetch((const char *)(a + i + 256), _MM_HINT_T0);
    for (int g = 0; g < 4; g++) {
      size_t off = i + g * 32;
      __m256i c0 = _mm256_cmpgt_epi32(
          vs_b, _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + off)),
                                 bias));
      __m256i c1 = _mm256_cmpgt_epi32(
          vs_b, _mm256_xor_si256(
                    _mm256_loadu_si256((const __m256i *)(a + off + 8)), bias));
      __m256i c2 = _mm256_cmpgt_epi32(
          vs_b, _mm256_xor_si256(
                    _mm256_loadu_si256((const __m256i *)(a + off + 16)), bias));
      __m256i c3 = _mm256_cmpgt_epi32(
          vs_b, _mm256_xor_si256(
                    _mm256_loadu_si256((const __m256i *)(a + off + 24)), bias));
      __m256i p = _pack32_sc_avx2(c0, c1, c2, c3, one);
      _mm256_storeu_si256((__m256i *)(out + off), _mm256_andnot_si256(p, one));
    }
  }
  for (; i + 32 <= n; i += 32) {
    __m256i c0 = _mm256_cmpgt_epi32(
        vs_b,
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + i)), bias));
    __m256i c1 = _mm256_cmpgt_epi32(
        vs_b, _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + i + 8)),
                               bias));
    __m256i c2 = _mm256_cmpgt_epi32(
        vs_b, _mm256_xor_si256(
                  _mm256_loadu_si256((const __m256i *)(a + i + 16)), bias));
    __m256i c3 = _mm256_cmpgt_epi32(
        vs_b, _mm256_xor_si256(
                  _mm256_loadu_si256((const __m256i *)(a + i + 24)), bias));
    __m256i p = _pack32_sc_avx2(c0, c1, c2, c3, one);
    _mm256_storeu_si256((__m256i *)(out + i), _mm256_andnot_si256(p, one));
  }
  for (; i < n; i++)
    out[i] = (uint8_t)(a[i] >= s);
}

static inline void _cmpsc_le_u32_avx2(const void *restrict ap,
                                      const void *restrict sp,
                                      void *restrict op, size_t n) {
  const uint32_t *a = (const uint32_t *)ap;
  const uint32_t s = *(const uint32_t *)sp;
  uint8_t *out = (uint8_t *)op;
  const __m256i bias = _mm256_set1_epi32((int)0x80000000);
  const __m256i vs_b = _mm256_xor_si256(_mm256_set1_epi32((int)s), bias);
  const __m256i one = _mm256_set1_epi8(1);
  size_t i = 0;
  for (; i + 128 <= n; i += 128) {
    _mm_prefetch((const char *)(a + i + 256), _MM_HINT_T0);
    for (int g = 0; g < 4; g++) {
      size_t off = i + g * 32;
      __m256i c0 = _mm256_cmpgt_epi32(
          _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + off)),
                           bias),
          vs_b);
      __m256i c1 = _mm256_cmpgt_epi32(
          _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + off + 8)),
                           bias),
          vs_b);
      __m256i c2 = _mm256_cmpgt_epi32(
          _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + off + 16)),
                           bias),
          vs_b);
      __m256i c3 = _mm256_cmpgt_epi32(
          _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + off + 24)),
                           bias),
          vs_b);
      __m256i p = _pack32_sc_avx2(c0, c1, c2, c3, one);
      _mm256_storeu_si256((__m256i *)(out + off), _mm256_andnot_si256(p, one));
    }
  }
  for (; i + 32 <= n; i += 32) {
    __m256i c0 = _mm256_cmpgt_epi32(
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + i)), bias),
        vs_b);
    __m256i c1 = _mm256_cmpgt_epi32(
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + i + 8)),
                         bias),
        vs_b);
    __m256i c2 = _mm256_cmpgt_epi32(
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + i + 16)),
                         bias),
        vs_b);
    __m256i c3 = _mm256_cmpgt_epi32(
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + i + 24)),
                         bias),
        vs_b);
    __m256i p = _pack32_sc_avx2(c0, c1, c2, c3, one);
    _mm256_storeu_si256((__m256i *)(out + i), _mm256_andnot_si256(p, one));
  }
  for (; i < n; i++)
    out[i] = (uint8_t)(a[i] <= s);
}

/* ====================================================================
 * 64-bit signed: i64 — movemask + LUT path (4 elems per vector)
 * ================================================================ */

#define _SC_MASKI64(v) _mm256_movemask_pd(_mm256_castsi256_pd(v))

static inline void _cmpsc_eq_i64_avx2(const void *restrict ap,
                                      const void *restrict sp,
                                      void *restrict op, size_t n) {
  const int64_t *a = (const int64_t *)ap;
  const int64_t s = *(const int64_t *)sp;
  uint8_t *out = (uint8_t *)op;
  const __m256i vs = _mm256_set1_epi64x(s);
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    _mm_prefetch((const char *)(a + i + 64), _MM_HINT_T0);
    for (int k = 0; k < 8; k++) {
      int m = _SC_MASKI64(_mm256_cmpeq_epi64(
          _mm256_loadu_si256((const __m256i *)(a + i + k * 4)), vs));
      *(uint32_t *)(out + i + k * 4) = _cmp_lut4_sc_avx2[m & 0xF];
    }
  }
  for (; i + 4 <= n; i += 4) {
    int m = _SC_MASKI64(
        _mm256_cmpeq_epi64(_mm256_loadu_si256((const __m256i *)(a + i)), vs));
    *(uint32_t *)(out + i) = _cmp_lut4_sc_avx2[m & 0xF];
  }
  for (; i < n; i++)
    out[i] = (uint8_t)(a[i] == s);
}

static inline void _cmpsc_gt_i64_avx2(const void *restrict ap,
                                      const void *restrict sp,
                                      void *restrict op, size_t n) {
  const int64_t *a = (const int64_t *)ap;
  const int64_t s = *(const int64_t *)sp;
  uint8_t *out = (uint8_t *)op;
  const __m256i vs = _mm256_set1_epi64x(s);
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    _mm_prefetch((const char *)(a + i + 64), _MM_HINT_T0);
    for (int k = 0; k < 8; k++) {
      int m = _SC_MASKI64(_mm256_cmpgt_epi64(
          _mm256_loadu_si256((const __m256i *)(a + i + k * 4)), vs));
      *(uint32_t *)(out + i + k * 4) = _cmp_lut4_sc_avx2[m & 0xF];
    }
  }
  for (; i + 4 <= n; i += 4) {
    int m = _SC_MASKI64(
        _mm256_cmpgt_epi64(_mm256_loadu_si256((const __m256i *)(a + i)), vs));
    *(uint32_t *)(out + i) = _cmp_lut4_sc_avx2[m & 0xF];
  }
  for (; i < n; i++)
    out[i] = (uint8_t)(a[i] > s);
}

static inline void _cmpsc_lt_i64_avx2(const void *restrict ap,
                                      const void *restrict sp,
                                      void *restrict op, size_t n) {
  const int64_t *a = (const int64_t *)ap;
  const int64_t s = *(const int64_t *)sp;
  uint8_t *out = (uint8_t *)op;
  const __m256i vs = _mm256_set1_epi64x(s);
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    _mm_prefetch((const char *)(a + i + 64), _MM_HINT_T0);
    for (int k = 0; k < 8; k++) {
      int m = _SC_MASKI64(_mm256_cmpgt_epi64(
          vs, _mm256_loadu_si256((const __m256i *)(a + i + k * 4))));
      *(uint32_t *)(out + i + k * 4) = _cmp_lut4_sc_avx2[m & 0xF];
    }
  }
  for (; i + 4 <= n; i += 4) {
    int m = _SC_MASKI64(
        _mm256_cmpgt_epi64(vs, _mm256_loadu_si256((const __m256i *)(a + i))));
    *(uint32_t *)(out + i) = _cmp_lut4_sc_avx2[m & 0xF];
  }
  for (; i < n; i++)
    out[i] = (uint8_t)(a[i] < s);
}

static inline void _cmpsc_ge_i64_avx2(const void *restrict ap,
                                      const void *restrict sp,
                                      void *restrict op, size_t n) {
  const int64_t *a = (const int64_t *)ap;
  const int64_t s = *(const int64_t *)sp;
  uint8_t *out = (uint8_t *)op;
  const __m256i vs = _mm256_set1_epi64x(s);
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    _mm_prefetch((const char *)(a + i + 64), _MM_HINT_T0);
    for (int k = 0; k < 8; k++) {
      int m = _SC_MASKI64(_mm256_cmpgt_epi64(
          vs, _mm256_loadu_si256((const __m256i *)(a + i + k * 4))));
      *(uint32_t *)(out + i + k * 4) = _cmp_lut4_sc_avx2[m & 0xF] ^ 0x01010101;
    }
  }
  for (; i + 4 <= n; i += 4) {
    int m = _SC_MASKI64(
        _mm256_cmpgt_epi64(vs, _mm256_loadu_si256((const __m256i *)(a + i))));
    *(uint32_t *)(out + i) = _cmp_lut4_sc_avx2[m & 0xF] ^ 0x01010101;
  }
  for (; i < n; i++)
    out[i] = (uint8_t)(a[i] >= s);
}

static inline void _cmpsc_le_i64_avx2(const void *restrict ap,
                                      const void *restrict sp,
                                      void *restrict op, size_t n) {
  const int64_t *a = (const int64_t *)ap;
  const int64_t s = *(const int64_t *)sp;
  uint8_t *out = (uint8_t *)op;
  const __m256i vs = _mm256_set1_epi64x(s);
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    _mm_prefetch((const char *)(a + i + 64), _MM_HINT_T0);
    for (int k = 0; k < 8; k++) {
      int m = _SC_MASKI64(_mm256_cmpgt_epi64(
          _mm256_loadu_si256((const __m256i *)(a + i + k * 4)), vs));
      *(uint32_t *)(out + i + k * 4) = _cmp_lut4_sc_avx2[m & 0xF] ^ 0x01010101;
    }
  }
  for (; i + 4 <= n; i += 4) {
    int m = _SC_MASKI64(
        _mm256_cmpgt_epi64(_mm256_loadu_si256((const __m256i *)(a + i)), vs));
    *(uint32_t *)(out + i) = _cmp_lut4_sc_avx2[m & 0xF] ^ 0x01010101;
  }
  for (; i < n; i++)
    out[i] = (uint8_t)(a[i] <= s);
}

#undef _SC_MASKI64

/* ====================================================================
 * 64-bit unsigned: u64 — eq uses cmpeq, gt/ge use XOR sign-bit bias
 * ================================================================ */

static inline void _cmpsc_eq_u64_avx2(const void *restrict ap,
                                      const void *restrict sp,
                                      void *restrict op, size_t n) {
  const uint64_t *a = (const uint64_t *)ap;
  const uint64_t s = *(const uint64_t *)sp;
  uint8_t *out = (uint8_t *)op;
  const __m256i vs = _mm256_set1_epi64x((long long)s);
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    _mm_prefetch((const char *)(a + i + 64), _MM_HINT_T0);
    for (int k = 0; k < 8; k++) {
      int m = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(
          _mm256_loadu_si256((const __m256i *)(a + i + k * 4)), vs)));
      *(uint32_t *)(out + i + k * 4) = _cmp_lut4_sc_avx2[m & 0xF];
    }
  }
  for (; i + 4 <= n; i += 4) {
    int m = _mm256_movemask_pd(_mm256_castsi256_pd(
        _mm256_cmpeq_epi64(_mm256_loadu_si256((const __m256i *)(a + i)), vs)));
    *(uint32_t *)(out + i) = _cmp_lut4_sc_avx2[m & 0xF];
  }
  for (; i < n; i++)
    out[i] = (uint8_t)(a[i] == s);
}

static inline void _cmpsc_gt_u64_avx2(const void *restrict ap,
                                      const void *restrict sp,
                                      void *restrict op, size_t n) {
  const uint64_t *a = (const uint64_t *)ap;
  const uint64_t s = *(const uint64_t *)sp;
  uint8_t *out = (uint8_t *)op;
  const __m256i bias = _mm256_set1_epi64x((long long)0x8000000000000000LL);
  const __m256i vs_b = _mm256_xor_si256(_mm256_set1_epi64x((long long)s), bias);
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    _mm_prefetch((const char *)(a + i + 64), _MM_HINT_T0);
    for (int k = 0; k < 8; k++) {
      int m = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpgt_epi64(
          _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + i + k * 4)),
                           bias),
          vs_b)));
      *(uint32_t *)(out + i + k * 4) = _cmp_lut4_sc_avx2[m & 0xF];
    }
  }
  for (; i + 4 <= n; i += 4) {
    int m = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpgt_epi64(
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + i)), bias),
        vs_b)));
    *(uint32_t *)(out + i) = _cmp_lut4_sc_avx2[m & 0xF];
  }
  for (; i < n; i++)
    out[i] = (uint8_t)(a[i] > s);
}

static inline void _cmpsc_lt_u64_avx2(const void *restrict ap,
                                      const void *restrict sp,
                                      void *restrict op, size_t n) {
  const uint64_t *a = (const uint64_t *)ap;
  const uint64_t s = *(const uint64_t *)sp;
  uint8_t *out = (uint8_t *)op;
  const __m256i bias = _mm256_set1_epi64x((long long)0x8000000000000000LL);
  const __m256i vs_b = _mm256_xor_si256(_mm256_set1_epi64x((long long)s), bias);
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    _mm_prefetch((const char *)(a + i + 64), _MM_HINT_T0);
    for (int k = 0; k < 8; k++) {
      int m = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpgt_epi64(
          vs_b,
          _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + i + k * 4)),
                           bias))));
      *(uint32_t *)(out + i + k * 4) = _cmp_lut4_sc_avx2[m & 0xF];
    }
  }
  for (; i + 4 <= n; i += 4) {
    int m = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpgt_epi64(
        vs_b,
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + i)), bias))));
    *(uint32_t *)(out + i) = _cmp_lut4_sc_avx2[m & 0xF];
  }
  for (; i < n; i++)
    out[i] = (uint8_t)(a[i] < s);
}

static inline void _cmpsc_ge_u64_avx2(const void *restrict ap,
                                      const void *restrict sp,
                                      void *restrict op, size_t n) {
  const uint64_t *a = (const uint64_t *)ap;
  const uint64_t s = *(const uint64_t *)sp;
  uint8_t *out = (uint8_t *)op;
  const __m256i bias = _mm256_set1_epi64x((long long)0x8000000000000000LL);
  const __m256i vs_b = _mm256_xor_si256(_mm256_set1_epi64x((long long)s), bias);
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    _mm_prefetch((const char *)(a + i + 64), _MM_HINT_T0);
    for (int k = 0; k < 8; k++) {
      int m = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpgt_epi64(
          vs_b,
          _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + i + k * 4)),
                           bias))));
      *(uint32_t *)(out + i + k * 4) = _cmp_lut4_sc_avx2[m & 0xF] ^ 0x01010101;
    }
  }
  for (; i + 4 <= n; i += 4) {
    int m = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpgt_epi64(
        vs_b,
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + i)), bias))));
    *(uint32_t *)(out + i) = _cmp_lut4_sc_avx2[m & 0xF] ^ 0x01010101;
  }
  for (; i < n; i++)
    out[i] = (uint8_t)(a[i] >= s);
}

static inline void _cmpsc_le_u64_avx2(const void *restrict ap,
                                      const void *restrict sp,
                                      void *restrict op, size_t n) {
  const uint64_t *a = (const uint64_t *)ap;
  const uint64_t s = *(const uint64_t *)sp;
  uint8_t *out = (uint8_t *)op;
  const __m256i bias = _mm256_set1_epi64x((long long)0x8000000000000000LL);
  const __m256i vs_b = _mm256_xor_si256(_mm256_set1_epi64x((long long)s), bias);
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    _mm_prefetch((const char *)(a + i + 64), _MM_HINT_T0);
    for (int k = 0; k < 8; k++) {
      int m = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpgt_epi64(
          _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + i + k * 4)),
                           bias),
          vs_b)));
      *(uint32_t *)(out + i + k * 4) = _cmp_lut4_sc_avx2[m & 0xF] ^ 0x01010101;
    }
  }
  for (; i + 4 <= n; i += 4) {
    int m = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpgt_epi64(
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + i)), bias),
        vs_b)));
    *(uint32_t *)(out + i) = _cmp_lut4_sc_avx2[m & 0xF] ^ 0x01010101;
  }
  for (; i < n; i++)
    out[i] = (uint8_t)(a[i] <= s);
}

#endif /* NUMC_COMPARE_SCALAR_AVX2_H */
