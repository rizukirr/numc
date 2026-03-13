/**
 * @file elemwise_scalar_avx2.h
 * @brief AVX2 scalar arithmetic kernels for all 10 types.
 *
 * Operations: add_scalar, sub_scalar, mul_scalar
 * Types: i8, i16, i32, i64, u8, u16, u32, u64, f32, f64
 *
 * Each function applies: out[i] = a[i] OP scalar
 *
 * Special cases: i8/u8 mul_scalar (widening, no native 8-bit multiply),
 * i64/u64 mul_scalar (scalar fallback, no efficient AVX2 64-bit multiply).
 */
#ifndef NUMC_ELEMWISE_SCALAR_AVX2_H
#define NUMC_ELEMWISE_SCALAR_AVX2_H

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

/* ════════════════════════════════════════════════════════════════════
 * Generic macros for scalar operations
 * ════════════════════════════════════════════════════════════════ */

#define FAST_SCAL_INT_AVX2(OP, SFX, CT, VPV, SET1, VEC_OP, TAIL_EXPR) \
  static inline void _fast_##OP##_scalar_##SFX##_avx2(                \
      const void *restrict ap, const void *restrict sp,                \
      void *restrict op, size_t n) {                                   \
    const CT *a = (const CT *)ap;                                      \
    const CT s = *(const CT *)sp;                                      \
    CT *out = (CT *)op;                                                \
    __m256i vs = SET1(s);                                              \
    size_t i = 0;                                                      \
    for (; i + (VPV) <= n; i += (VPV)) {                               \
      __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));       \
      _mm256_storeu_si256((__m256i *)(out + i), VEC_OP(va, vs));       \
    }                                                                  \
    for (; i < n; i++)                                                 \
      out[i] = (CT)(TAIL_EXPR);                                       \
  }

#define FAST_SCAL_F32_AVX2(OP, VEC_OP, TAIL_EXPR)                     \
  static inline void _fast_##OP##_scalar_f32_avx2(                    \
      const void *restrict ap, const void *restrict sp,                \
      void *restrict op, size_t n) {                                   \
    const float *a = (const float *)ap;                                \
    const float s = *(const float *)sp;                                \
    float *out = (float *)op;                                          \
    __m256 vs = _mm256_set1_ps(s);                                     \
    size_t i = 0;                                                      \
    for (; i + 8 <= n; i += 8) {                                       \
      __m256 va = _mm256_loadu_ps(a + i);                              \
      _mm256_storeu_ps(out + i, VEC_OP(va, vs));                       \
    }                                                                  \
    for (; i < n; i++)                                                 \
      out[i] = (float)(TAIL_EXPR);                                    \
  }

#define FAST_SCAL_F64_AVX2(OP, VEC_OP, TAIL_EXPR)                     \
  static inline void _fast_##OP##_scalar_f64_avx2(                    \
      const void *restrict ap, const void *restrict sp,                \
      void *restrict op, size_t n) {                                   \
    const double *a = (const double *)ap;                              \
    const double s = *(const double *)sp;                              \
    double *out = (double *)op;                                        \
    __m256d vs = _mm256_set1_pd(s);                                    \
    size_t i = 0;                                                      \
    for (; i + 4 <= n; i += 4) {                                       \
      __m256d va = _mm256_loadu_pd(a + i);                             \
      _mm256_storeu_pd(out + i, VEC_OP(va, vs));                       \
    }                                                                  \
    for (; i < n; i++)                                                 \
      out[i] = (double)(TAIL_EXPR);                                   \
  }

/* ── Add scalar ──────────────────────────────────────────────────── */

FAST_SCAL_INT_AVX2(add, i8, int8_t, 32,
                   _mm256_set1_epi8, _mm256_add_epi8, a[i] + s)
FAST_SCAL_INT_AVX2(add, i16, int16_t, 16,
                   _mm256_set1_epi16, _mm256_add_epi16, a[i] + s)
FAST_SCAL_INT_AVX2(add, i32, int32_t, 8,
                   _mm256_set1_epi32, _mm256_add_epi32, a[i] + s)
FAST_SCAL_INT_AVX2(add, i64, int64_t, 4,
                   _mm256_set1_epi64x, _mm256_add_epi64, a[i] + s)
FAST_SCAL_INT_AVX2(add, u8, uint8_t, 32,
                   _mm256_set1_epi8, _mm256_add_epi8, a[i] + s)
FAST_SCAL_INT_AVX2(add, u16, uint16_t, 16,
                   _mm256_set1_epi16, _mm256_add_epi16, a[i] + s)
FAST_SCAL_INT_AVX2(add, u32, uint32_t, 8,
                   _mm256_set1_epi32, _mm256_add_epi32, a[i] + s)
FAST_SCAL_INT_AVX2(add, u64, uint64_t, 4,
                   _mm256_set1_epi64x, _mm256_add_epi64, a[i] + s)
FAST_SCAL_F32_AVX2(add, _mm256_add_ps, a[i] + s)
FAST_SCAL_F64_AVX2(add, _mm256_add_pd, a[i] + s)

/* ── Sub scalar ──────────────────────────────────────────────────── */

FAST_SCAL_INT_AVX2(sub, i8, int8_t, 32,
                   _mm256_set1_epi8, _mm256_sub_epi8, a[i] - s)
FAST_SCAL_INT_AVX2(sub, i16, int16_t, 16,
                   _mm256_set1_epi16, _mm256_sub_epi16, a[i] - s)
FAST_SCAL_INT_AVX2(sub, i32, int32_t, 8,
                   _mm256_set1_epi32, _mm256_sub_epi32, a[i] - s)
FAST_SCAL_INT_AVX2(sub, i64, int64_t, 4,
                   _mm256_set1_epi64x, _mm256_sub_epi64, a[i] - s)
FAST_SCAL_INT_AVX2(sub, u8, uint8_t, 32,
                   _mm256_set1_epi8, _mm256_sub_epi8, a[i] - s)
FAST_SCAL_INT_AVX2(sub, u16, uint16_t, 16,
                   _mm256_set1_epi16, _mm256_sub_epi16, a[i] - s)
FAST_SCAL_INT_AVX2(sub, u32, uint32_t, 8,
                   _mm256_set1_epi32, _mm256_sub_epi32, a[i] - s)
FAST_SCAL_INT_AVX2(sub, u64, uint64_t, 4,
                   _mm256_set1_epi64x, _mm256_sub_epi64, a[i] - s)
FAST_SCAL_F32_AVX2(sub, _mm256_sub_ps, a[i] - s)
FAST_SCAL_F64_AVX2(sub, _mm256_sub_pd, a[i] - s)

/* ── Mul scalar (16/32-bit: native mullo) ────────────────────────── */

FAST_SCAL_INT_AVX2(mul, i16, int16_t, 16,
                   _mm256_set1_epi16, _mm256_mullo_epi16, a[i] * s)
FAST_SCAL_INT_AVX2(mul, i32, int32_t, 8,
                   _mm256_set1_epi32, _mm256_mullo_epi32, a[i] * s)
FAST_SCAL_INT_AVX2(mul, u16, uint16_t, 16,
                   _mm256_set1_epi16, _mm256_mullo_epi16, a[i] * s)
FAST_SCAL_INT_AVX2(mul, u32, uint32_t, 8,
                   _mm256_set1_epi32, _mm256_mullo_epi32, a[i] * s)

/* ── Mul scalar i8: widening trick (no native 8-bit multiply) ──── */

static inline void _fast_mul_scalar_i8_avx2(const void *restrict ap,
                                             const void *restrict sp,
                                             void *restrict op,
                                             size_t n) {
  const int8_t *a = (const int8_t *)ap;
  const int8_t s = *(const int8_t *)sp;
  int8_t *out = (int8_t *)op;
  const __m256i mask = _mm256_set1_epi16(0x00FF);
  const __m256i vs_lo = _mm256_and_si256(_mm256_set1_epi8(s), mask);
  const __m256i vs_hi = _mm256_srli_epi16(_mm256_set1_epi8(s), 8);
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
    /* Even bytes: mask to 16-bit, multiply, mask result */
    __m256i lo = _mm256_and_si256(
        _mm256_mullo_epi16(_mm256_and_si256(va, mask), vs_lo), mask);
    /* Odd bytes: shift right 8, multiply, shift left 8 */
    __m256i hi = _mm256_slli_epi16(
        _mm256_mullo_epi16(_mm256_srli_epi16(va, 8), vs_hi), 8);
    _mm256_storeu_si256((__m256i *)(out + i), _mm256_or_si256(lo, hi));
  }
  for (; i < n; i++)
    out[i] = (int8_t)(a[i] * s);
}

static inline void _fast_mul_scalar_u8_avx2(const void *restrict ap,
                                             const void *restrict sp,
                                             void *restrict op,
                                             size_t n) {
  const uint8_t *a = (const uint8_t *)ap;
  const uint8_t s = *(const uint8_t *)sp;
  uint8_t *out = (uint8_t *)op;
  const __m256i mask = _mm256_set1_epi16(0x00FF);
  const __m256i vs_lo = _mm256_and_si256(_mm256_set1_epi8((char)s), mask);
  const __m256i vs_hi =
      _mm256_srli_epi16(_mm256_set1_epi8((char)s), 8);
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
    __m256i lo = _mm256_and_si256(
        _mm256_mullo_epi16(_mm256_and_si256(va, mask), vs_lo), mask);
    __m256i hi = _mm256_slli_epi16(
        _mm256_mullo_epi16(_mm256_srli_epi16(va, 8), vs_hi), 8);
    _mm256_storeu_si256((__m256i *)(out + i), _mm256_or_si256(lo, hi));
  }
  for (; i < n; i++)
    out[i] = (uint8_t)(a[i] * s);
}

/* ── Mul scalar i64/u64: scalar fallback (no efficient AVX2) ───── */

static inline void _fast_mul_scalar_i64_avx2(const void *restrict ap,
                                              const void *restrict sp,
                                              void *restrict op,
                                              size_t n) {
  const int64_t *a = (const int64_t *)ap;
  const int64_t s = *(const int64_t *)sp;
  int64_t *out = (int64_t *)op;
  for (size_t i = 0; i < n; i++)
    out[i] = a[i] * s;
}

static inline void _fast_mul_scalar_u64_avx2(const void *restrict ap,
                                              const void *restrict sp,
                                              void *restrict op,
                                              size_t n) {
  const uint64_t *a = (const uint64_t *)ap;
  const uint64_t s = *(const uint64_t *)sp;
  uint64_t *out = (uint64_t *)op;
  for (size_t i = 0; i < n; i++)
    out[i] = a[i] * s;
}

/* ── Mul scalar (float) ──────────────────────────────────────────── */

FAST_SCAL_F32_AVX2(mul, _mm256_mul_ps, a[i] * s)
FAST_SCAL_F64_AVX2(mul, _mm256_mul_pd, a[i] * s)

/* ── Clean up macros ─────────────────────────────────────────────── */

#undef FAST_SCAL_INT_AVX2
#undef FAST_SCAL_F32_AVX2
#undef FAST_SCAL_F64_AVX2

#endif /* NUMC_ELEMWISE_SCALAR_AVX2_H */
