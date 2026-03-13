/**
 * @file elemwise_avx512.h
 * @brief AVX-512 element-wise binary/unary kernels for all 10 types.
 *
 * Binary: sub, mul, maximum, minimum
 * Unary: neg, abs
 *
 * Special cases: i8/u8 mul (widening, no native 8-bit multiply),
 * i64/u64 mul (scalar fallback — _mm512_mullo_epi64 requires AVX512DQ).
 */
#ifndef NUMC_ELEMWISE_AVX512_H
#define NUMC_ELEMWISE_AVX512_H

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

/* ════════════════════════════════════════════════════════════════════
 * Binary: generic integer macro (for ops with native AVX-512 support)
 * ════════════════════════════════════════════════════════════════ */

#define FAST_BIN_INT_AVX512(OP, SFX, CT, VPV, VEC_OP, TAIL_EXPR)      \
  static inline void _fast_##OP##_##SFX##_avx512(                     \
      const void *restrict ap, const void *restrict bp,                \
      void *restrict op, size_t n) {                                   \
    const CT *a = (const CT *)ap;                                      \
    const CT *b = (const CT *)bp;                                      \
    CT *out = (CT *)op;                                                \
    size_t i = 0;                                                      \
    for (; i + (VPV) <= n; i += (VPV)) {                               \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));       \
      __m512i vb = _mm512_loadu_si512((const __m512i *)(b + i));       \
      _mm512_storeu_si512((__m512i *)(out + i), VEC_OP(va, vb));       \
    }                                                                  \
    for (; i < n; i++)                                                 \
      out[i] = (CT)(TAIL_EXPR);                                       \
  }

#define FAST_BIN_F32_AVX512(OP, VEC_OP, TAIL_EXPR)                    \
  static inline void _fast_##OP##_f32_avx512(                         \
      const void *restrict ap, const void *restrict bp,                \
      void *restrict op, size_t n) {                                   \
    const float *a = (const float *)ap;                                \
    const float *b = (const float *)bp;                                \
    float *out = (float *)op;                                          \
    size_t i = 0;                                                      \
    for (; i + 16 <= n; i += 16) {                                     \
      __m512 va = _mm512_loadu_ps(a + i);                              \
      __m512 vb = _mm512_loadu_ps(b + i);                              \
      _mm512_storeu_ps(out + i, VEC_OP(va, vb));                       \
    }                                                                  \
    for (; i < n; i++)                                                 \
      out[i] = (float)(TAIL_EXPR);                                    \
  }

#define FAST_BIN_F64_AVX512(OP, VEC_OP, TAIL_EXPR)                    \
  static inline void _fast_##OP##_f64_avx512(                         \
      const void *restrict ap, const void *restrict bp,                \
      void *restrict op, size_t n) {                                   \
    const double *a = (const double *)ap;                              \
    const double *b = (const double *)bp;                              \
    double *out = (double *)op;                                        \
    size_t i = 0;                                                      \
    for (; i + 8 <= n; i += 8) {                                       \
      __m512d va = _mm512_loadu_pd(a + i);                             \
      __m512d vb = _mm512_loadu_pd(b + i);                             \
      _mm512_storeu_pd(out + i, VEC_OP(va, vb));                       \
    }                                                                  \
    for (; i < n; i++)                                                 \
      out[i] = (double)(TAIL_EXPR);                                   \
  }

/* ── Add ─────────────────────────────────────────────────────────── */

FAST_BIN_INT_AVX512(add, i8, int8_t, 64, _mm512_add_epi8, a[i] + b[i])
FAST_BIN_INT_AVX512(add, i16, int16_t, 32, _mm512_add_epi16,
                    a[i] + b[i])
FAST_BIN_INT_AVX512(add, i32, int32_t, 16, _mm512_add_epi32,
                    a[i] + b[i])
FAST_BIN_INT_AVX512(add, i64, int64_t, 8, _mm512_add_epi64,
                    a[i] + b[i])
FAST_BIN_INT_AVX512(add, u8, uint8_t, 64, _mm512_add_epi8, a[i] + b[i])
FAST_BIN_INT_AVX512(add, u16, uint16_t, 32, _mm512_add_epi16,
                    a[i] + b[i])
FAST_BIN_INT_AVX512(add, u32, uint32_t, 16, _mm512_add_epi32,
                    a[i] + b[i])
FAST_BIN_INT_AVX512(add, u64, uint64_t, 8, _mm512_add_epi64,
                    a[i] + b[i])
FAST_BIN_F32_AVX512(add, _mm512_add_ps, a[i] + b[i])
FAST_BIN_F64_AVX512(add, _mm512_add_pd, a[i] + b[i])

/* ── Sub ─────────────────────────────────────────────────────────── */

FAST_BIN_INT_AVX512(sub, i8, int8_t, 64, _mm512_sub_epi8, a[i] - b[i])
FAST_BIN_INT_AVX512(sub, i16, int16_t, 32, _mm512_sub_epi16,
                    a[i] - b[i])
FAST_BIN_INT_AVX512(sub, i32, int32_t, 16, _mm512_sub_epi32,
                    a[i] - b[i])
FAST_BIN_INT_AVX512(sub, i64, int64_t, 8, _mm512_sub_epi64,
                    a[i] - b[i])
FAST_BIN_INT_AVX512(sub, u8, uint8_t, 64, _mm512_sub_epi8, a[i] - b[i])
FAST_BIN_INT_AVX512(sub, u16, uint16_t, 32, _mm512_sub_epi16,
                    a[i] - b[i])
FAST_BIN_INT_AVX512(sub, u32, uint32_t, 16, _mm512_sub_epi32,
                    a[i] - b[i])
FAST_BIN_INT_AVX512(sub, u64, uint64_t, 8, _mm512_sub_epi64,
                    a[i] - b[i])
FAST_BIN_F32_AVX512(sub, _mm512_sub_ps, a[i] - b[i])
FAST_BIN_F64_AVX512(sub, _mm512_sub_pd, a[i] - b[i])

/* ── Maximum (signed) ────────────────────────────────────────────── */

FAST_BIN_INT_AVX512(maximum, i8, int8_t, 64, _mm512_max_epi8,
                    a[i] > b[i] ? a[i] : b[i])
FAST_BIN_INT_AVX512(maximum, i16, int16_t, 32, _mm512_max_epi16,
                    a[i] > b[i] ? a[i] : b[i])
FAST_BIN_INT_AVX512(maximum, i32, int32_t, 16, _mm512_max_epi32,
                    a[i] > b[i] ? a[i] : b[i])
FAST_BIN_INT_AVX512(maximum, i64, int64_t, 8, _mm512_max_epi64,
                    a[i] > b[i] ? a[i] : b[i])

/* ── Maximum (unsigned) ──────────────────────────────────────────── */

FAST_BIN_INT_AVX512(maximum, u8, uint8_t, 64, _mm512_max_epu8,
                    a[i] > b[i] ? a[i] : b[i])
FAST_BIN_INT_AVX512(maximum, u16, uint16_t, 32, _mm512_max_epu16,
                    a[i] > b[i] ? a[i] : b[i])
FAST_BIN_INT_AVX512(maximum, u32, uint32_t, 16, _mm512_max_epu32,
                    a[i] > b[i] ? a[i] : b[i])
FAST_BIN_INT_AVX512(maximum, u64, uint64_t, 8, _mm512_max_epu64,
                    a[i] > b[i] ? a[i] : b[i])

/* ── Maximum (float) ─────────────────────────────────────────────── */

FAST_BIN_F32_AVX512(maximum, _mm512_max_ps, a[i] > b[i] ? a[i] : b[i])
FAST_BIN_F64_AVX512(maximum, _mm512_max_pd, a[i] > b[i] ? a[i] : b[i])

/* ── Minimum (signed) ────────────────────────────────────────────── */

FAST_BIN_INT_AVX512(minimum, i8, int8_t, 64, _mm512_min_epi8,
                    a[i] < b[i] ? a[i] : b[i])
FAST_BIN_INT_AVX512(minimum, i16, int16_t, 32, _mm512_min_epi16,
                    a[i] < b[i] ? a[i] : b[i])
FAST_BIN_INT_AVX512(minimum, i32, int32_t, 16, _mm512_min_epi32,
                    a[i] < b[i] ? a[i] : b[i])
FAST_BIN_INT_AVX512(minimum, i64, int64_t, 8, _mm512_min_epi64,
                    a[i] < b[i] ? a[i] : b[i])

/* ── Minimum (unsigned) ──────────────────────────────────────────── */

FAST_BIN_INT_AVX512(minimum, u8, uint8_t, 64, _mm512_min_epu8,
                    a[i] < b[i] ? a[i] : b[i])
FAST_BIN_INT_AVX512(minimum, u16, uint16_t, 32, _mm512_min_epu16,
                    a[i] < b[i] ? a[i] : b[i])
FAST_BIN_INT_AVX512(minimum, u32, uint32_t, 16, _mm512_min_epu32,
                    a[i] < b[i] ? a[i] : b[i])
FAST_BIN_INT_AVX512(minimum, u64, uint64_t, 8, _mm512_min_epu64,
                    a[i] < b[i] ? a[i] : b[i])

/* ── Minimum (float) ─────────────────────────────────────────────── */

FAST_BIN_F32_AVX512(minimum, _mm512_min_ps, a[i] < b[i] ? a[i] : b[i])
FAST_BIN_F64_AVX512(minimum, _mm512_min_pd, a[i] < b[i] ? a[i] : b[i])

/* ── Mul (16/32-bit: native mullo) ───────────────────────────────── */

FAST_BIN_INT_AVX512(mul, i16, int16_t, 32, _mm512_mullo_epi16,
                    a[i] * b[i])
FAST_BIN_INT_AVX512(mul, i32, int32_t, 16, _mm512_mullo_epi32,
                    a[i] * b[i])
FAST_BIN_INT_AVX512(mul, u16, uint16_t, 32, _mm512_mullo_epi16,
                    a[i] * b[i])
FAST_BIN_INT_AVX512(mul, u32, uint32_t, 16, _mm512_mullo_epi32,
                    a[i] * b[i])

/* ── Mul i8/u8: widening trick (no native 8-bit multiply) ──────── */

static inline void _fast_mul_i8_avx512(const void *restrict ap,
                                       const void *restrict bp,
                                       void *restrict op, size_t n) {
  const int8_t *a = (const int8_t *)ap;
  const int8_t *b = (const int8_t *)bp;
  int8_t *out = (int8_t *)op;
  const __m512i mask = _mm512_set1_epi16(0x00FF);
  size_t i = 0;
  for (; i + 64 <= n; i += 64) {
    __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));
    __m512i vb = _mm512_loadu_si512((const __m512i *)(b + i));
    /* Even bytes: mask to 16-bit, multiply, mask result */
    __m512i lo = _mm512_and_si512(
        _mm512_mullo_epi16(_mm512_and_si512(va, mask),
                           _mm512_and_si512(vb, mask)),
        mask);
    /* Odd bytes: shift right 8, multiply, shift left 8 */
    __m512i hi = _mm512_slli_epi16(
        _mm512_mullo_epi16(_mm512_srli_epi16(va, 8),
                           _mm512_srli_epi16(vb, 8)),
        8);
    _mm512_storeu_si512((__m512i *)(out + i), _mm512_or_si512(lo, hi));
  }
  for (; i < n; i++)
    out[i] = (int8_t)(a[i] * b[i]);
}

static inline void _fast_mul_u8_avx512(const void *restrict ap,
                                       const void *restrict bp,
                                       void *restrict op, size_t n) {
  const uint8_t *a = (const uint8_t *)ap;
  const uint8_t *b = (const uint8_t *)bp;
  uint8_t *out = (uint8_t *)op;
  const __m512i mask = _mm512_set1_epi16(0x00FF);
  size_t i = 0;
  for (; i + 64 <= n; i += 64) {
    __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));
    __m512i vb = _mm512_loadu_si512((const __m512i *)(b + i));
    __m512i lo = _mm512_and_si512(
        _mm512_mullo_epi16(_mm512_and_si512(va, mask),
                           _mm512_and_si512(vb, mask)),
        mask);
    __m512i hi = _mm512_slli_epi16(
        _mm512_mullo_epi16(_mm512_srli_epi16(va, 8),
                           _mm512_srli_epi16(vb, 8)),
        8);
    _mm512_storeu_si512((__m512i *)(out + i), _mm512_or_si512(lo, hi));
  }
  for (; i < n; i++)
    out[i] = (uint8_t)(a[i] * b[i]);
}

/* ── Mul i64/u64: scalar (AVX512DQ may not be available) ─────────── */

static inline void _fast_mul_i64_avx512(const void *restrict ap,
                                        const void *restrict bp,
                                        void *restrict op, size_t n) {
  const int64_t *a = (const int64_t *)ap;
  const int64_t *b = (const int64_t *)bp;
  int64_t *out = (int64_t *)op;
  for (size_t i = 0; i < n; i++)
    out[i] = a[i] * b[i];
}

static inline void _fast_mul_u64_avx512(const void *restrict ap,
                                        const void *restrict bp,
                                        void *restrict op, size_t n) {
  const uint64_t *a = (const uint64_t *)ap;
  const uint64_t *b = (const uint64_t *)bp;
  uint64_t *out = (uint64_t *)op;
  for (size_t i = 0; i < n; i++)
    out[i] = a[i] * b[i];
}

FAST_BIN_F32_AVX512(mul, _mm512_mul_ps, a[i] * b[i])
FAST_BIN_F64_AVX512(mul, _mm512_mul_pd, a[i] * b[i])

#undef FAST_BIN_INT_AVX512
#undef FAST_BIN_F32_AVX512
#undef FAST_BIN_F64_AVX512

/* ════════════════════════════════════════════════════════════════════
 * Unary operations
 * ════════════════════════════════════════════════════════════════ */

#define FAST_UN_INT_AVX512(OP, SFX, CT, VPV, VEC_OP, TAIL_EXPR)       \
  static inline void _fast_##OP##_##SFX##_avx512(                     \
      const void *restrict ap, void *restrict op, size_t n) {         \
    const CT *a = (const CT *)ap;                                      \
    CT *out = (CT *)op;                                                \
    size_t i = 0;                                                      \
    for (; i + (VPV) <= n; i += (VPV)) {                               \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));       \
      _mm512_storeu_si512((__m512i *)(out + i), VEC_OP);               \
    }                                                                  \
    for (; i < n; i++)                                                 \
      out[i] = (CT)(TAIL_EXPR);                                       \
  }

/* ── Neg ─────────────────────────────────────────────────────────── */

#define NEG_VEC512(W) _mm512_sub_epi##W(_mm512_setzero_si512(), va)

FAST_UN_INT_AVX512(neg, i8, int8_t, 64, NEG_VEC512(8), -a[i])
FAST_UN_INT_AVX512(neg, i16, int16_t, 32, NEG_VEC512(16), -a[i])
FAST_UN_INT_AVX512(neg, i32, int32_t, 16, NEG_VEC512(32), -a[i])
FAST_UN_INT_AVX512(neg, i64, int64_t, 8, NEG_VEC512(64), -a[i])
FAST_UN_INT_AVX512(neg, u8, uint8_t, 64, NEG_VEC512(8),
                   (uint8_t)(-(int8_t)a[i]))
FAST_UN_INT_AVX512(neg, u16, uint16_t, 32, NEG_VEC512(16),
                   (uint16_t)(-(int16_t)a[i]))
FAST_UN_INT_AVX512(neg, u32, uint32_t, 16, NEG_VEC512(32),
                   (uint32_t)(-(int32_t)a[i]))
FAST_UN_INT_AVX512(neg, u64, uint64_t, 8, NEG_VEC512(64),
                   (uint64_t)(-(int64_t)a[i]))
#undef NEG_VEC512

static inline void _fast_neg_f32_avx512(const void *restrict ap,
                                        void *restrict op, size_t n) {
  const float *a = (const float *)ap;
  float *out = (float *)op;
  const __m512 sign = _mm512_set1_ps(-0.0f);
  size_t i = 0;
  for (; i + 16 <= n; i += 16)
    _mm512_storeu_ps(out + i,
                     _mm512_xor_ps(_mm512_loadu_ps(a + i), sign));
  for (; i < n; i++)
    out[i] = -a[i];
}

static inline void _fast_neg_f64_avx512(const void *restrict ap,
                                        void *restrict op, size_t n) {
  const double *a = (const double *)ap;
  double *out = (double *)op;
  const __m512d sign = _mm512_set1_pd(-0.0);
  size_t i = 0;
  for (; i + 8 <= n; i += 8)
    _mm512_storeu_pd(out + i,
                     _mm512_xor_pd(_mm512_loadu_pd(a + i), sign));
  for (; i < n; i++)
    out[i] = -a[i];
}

/* ── Abs (signed integers) ───────────────────────────────────────── */

FAST_UN_INT_AVX512(abs, i8, int8_t, 64, _mm512_abs_epi8(va),
                   (int8_t)(a[i] < 0 ? -a[i] : a[i]))
FAST_UN_INT_AVX512(abs, i16, int16_t, 32, _mm512_abs_epi16(va),
                   (int16_t)(a[i] < 0 ? -a[i] : a[i]))
FAST_UN_INT_AVX512(abs, i32, int32_t, 16, _mm512_abs_epi32(va),
                   (int32_t)(a[i] < 0 ? -a[i] : a[i]))
FAST_UN_INT_AVX512(abs, i64, int64_t, 8, _mm512_abs_epi64(va),
                   a[i] < 0 ? -a[i] : a[i])

static inline void _fast_abs_f32_avx512(const void *restrict ap,
                                        void *restrict op, size_t n) {
  const float *a = (const float *)ap;
  float *out = (float *)op;
  const __m512 mask =
      _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));
  size_t i = 0;
  for (; i + 16 <= n; i += 16)
    _mm512_storeu_ps(out + i,
                     _mm512_and_ps(_mm512_loadu_ps(a + i), mask));
  for (; i < n; i++)
    out[i] = a[i] < 0 ? -a[i] : a[i];
}

static inline void _fast_abs_f64_avx512(const void *restrict ap,
                                        void *restrict op, size_t n) {
  const double *a = (const double *)ap;
  double *out = (double *)op;
  const __m512d mask =
      _mm512_castsi512_pd(_mm512_set1_epi64(0x7FFFFFFFFFFFFFFFLL));
  size_t i = 0;
  for (; i + 8 <= n; i += 8)
    _mm512_storeu_pd(out + i,
                     _mm512_and_pd(_mm512_loadu_pd(a + i), mask));
  for (; i < n; i++)
    out[i] = a[i] < 0 ? -a[i] : a[i];
}

#undef FAST_UN_INT_AVX512

#endif /* NUMC_ELEMWISE_AVX512_H */
