/**
 * @file elemwise_avx2.h
 * @brief AVX2 element-wise binary/unary kernels for all 10 types.
 *
 * Binary: sub, mul, maximum, minimum
 * Unary: neg, abs
 *
 * Special cases: i8/u8 mul (widening, no native 8-bit multiply),
 * i64/u64 max/min (emulated via cmpgt+blendv), i64 abs (conditional neg).
 */
#ifndef NUMC_ELEMWISE_AVX2_H
#define NUMC_ELEMWISE_AVX2_H

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

/* ════════════════════════════════════════════════════════════════════
 * Binary: generic integer macro (for ops with native AVX2 support)
 * ════════════════════════════════════════════════════════════════ */

#define FAST_BIN_INT_AVX2(OP, SFX, CT, VPV, VEC_OP, TAIL_EXPR)        \
  static inline void _fast_##OP##_##SFX##_avx2(                       \
      const void *restrict ap, const void *restrict bp,                \
      void *restrict op, size_t n) {                                   \
    const CT *a = (const CT *)ap;                                      \
    const CT *b = (const CT *)bp;                                      \
    CT *out = (CT *)op;                                                \
    size_t i = 0;                                                      \
    for (; i + (VPV) <= n; i += (VPV)) {                               \
      __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));       \
      __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));       \
      _mm256_storeu_si256((__m256i *)(out + i), VEC_OP(va, vb));       \
    }                                                                  \
    for (; i < n; i++)                                                 \
      out[i] = (CT)(TAIL_EXPR);                                       \
  }

#define FAST_BIN_F32_AVX2(OP, VEC_OP, TAIL_EXPR)                      \
  static inline void _fast_##OP##_f32_avx2(                           \
      const void *restrict ap, const void *restrict bp,                \
      void *restrict op, size_t n) {                                   \
    const float *a = (const float *)ap;                                \
    const float *b = (const float *)bp;                                \
    float *out = (float *)op;                                          \
    size_t i = 0;                                                      \
    for (; i + 8 <= n; i += 8) {                                       \
      __m256 va = _mm256_loadu_ps(a + i);                              \
      __m256 vb = _mm256_loadu_ps(b + i);                              \
      _mm256_storeu_ps(out + i, VEC_OP(va, vb));                       \
    }                                                                  \
    for (; i < n; i++)                                                 \
      out[i] = (float)(TAIL_EXPR);                                    \
  }

#define FAST_BIN_F64_AVX2(OP, VEC_OP, TAIL_EXPR)                      \
  static inline void _fast_##OP##_f64_avx2(                           \
      const void *restrict ap, const void *restrict bp,                \
      void *restrict op, size_t n) {                                   \
    const double *a = (const double *)ap;                              \
    const double *b = (const double *)bp;                              \
    double *out = (double *)op;                                        \
    size_t i = 0;                                                      \
    for (; i + 4 <= n; i += 4) {                                       \
      __m256d va = _mm256_loadu_pd(a + i);                             \
      __m256d vb = _mm256_loadu_pd(b + i);                             \
      _mm256_storeu_pd(out + i, VEC_OP(va, vb));                       \
    }                                                                  \
    for (; i < n; i++)                                                 \
      out[i] = (double)(TAIL_EXPR);                                   \
  }

/* ── Add ─────────────────────────────────────────────────────────── */

FAST_BIN_INT_AVX2(add, i8, int8_t, 32, _mm256_add_epi8, a[i] + b[i])
FAST_BIN_INT_AVX2(add, i16, int16_t, 16, _mm256_add_epi16, a[i] + b[i])
FAST_BIN_INT_AVX2(add, i32, int32_t, 8, _mm256_add_epi32, a[i] + b[i])
FAST_BIN_INT_AVX2(add, i64, int64_t, 4, _mm256_add_epi64, a[i] + b[i])
FAST_BIN_INT_AVX2(add, u8, uint8_t, 32, _mm256_add_epi8, a[i] + b[i])
FAST_BIN_INT_AVX2(add, u16, uint16_t, 16, _mm256_add_epi16, a[i] + b[i])
FAST_BIN_INT_AVX2(add, u32, uint32_t, 8, _mm256_add_epi32, a[i] + b[i])
FAST_BIN_INT_AVX2(add, u64, uint64_t, 4, _mm256_add_epi64, a[i] + b[i])
FAST_BIN_F32_AVX2(add, _mm256_add_ps, a[i] + b[i])
FAST_BIN_F64_AVX2(add, _mm256_add_pd, a[i] + b[i])

/* ── Sub ─────────────────────────────────────────────────────────── */

FAST_BIN_INT_AVX2(sub, i8, int8_t, 32, _mm256_sub_epi8, a[i] - b[i])
FAST_BIN_INT_AVX2(sub, i16, int16_t, 16, _mm256_sub_epi16, a[i] - b[i])
FAST_BIN_INT_AVX2(sub, i32, int32_t, 8, _mm256_sub_epi32, a[i] - b[i])
FAST_BIN_INT_AVX2(sub, i64, int64_t, 4, _mm256_sub_epi64, a[i] - b[i])
FAST_BIN_INT_AVX2(sub, u8, uint8_t, 32, _mm256_sub_epi8, a[i] - b[i])
FAST_BIN_INT_AVX2(sub, u16, uint16_t, 16, _mm256_sub_epi16, a[i] - b[i])
FAST_BIN_INT_AVX2(sub, u32, uint32_t, 8, _mm256_sub_epi32, a[i] - b[i])
FAST_BIN_INT_AVX2(sub, u64, uint64_t, 4, _mm256_sub_epi64, a[i] - b[i])
FAST_BIN_F32_AVX2(sub, _mm256_sub_ps, a[i] - b[i])
FAST_BIN_F64_AVX2(sub, _mm256_sub_pd, a[i] - b[i])

/* ── Maximum (signed) ────────────────────────────────────────────── */

FAST_BIN_INT_AVX2(maximum, i8, int8_t, 32, _mm256_max_epi8,
                  a[i] > b[i] ? a[i] : b[i])
FAST_BIN_INT_AVX2(maximum, i16, int16_t, 16, _mm256_max_epi16,
                  a[i] > b[i] ? a[i] : b[i])
FAST_BIN_INT_AVX2(maximum, i32, int32_t, 8, _mm256_max_epi32,
                  a[i] > b[i] ? a[i] : b[i])

/* i64 maximum: emulated via cmpgt + blendv */
static inline void _fast_maximum_i64_avx2(const void *restrict ap,
                                          const void *restrict bp,
                                          void *restrict op, size_t n) {
  const int64_t *a = (const int64_t *)ap;
  const int64_t *b = (const int64_t *)bp;
  int64_t *out = (int64_t *)op;
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
    __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
    __m256i gt = _mm256_cmpgt_epi64(va, vb);
    _mm256_storeu_si256((__m256i *)(out + i),
                        _mm256_blendv_epi8(vb, va, gt));
  }
  for (; i < n; i++)
    out[i] = a[i] > b[i] ? a[i] : b[i];
}

/* ── Maximum (unsigned) ──────────────────────────────────────────── */

FAST_BIN_INT_AVX2(maximum, u8, uint8_t, 32, _mm256_max_epu8,
                  a[i] > b[i] ? a[i] : b[i])
FAST_BIN_INT_AVX2(maximum, u16, uint16_t, 16, _mm256_max_epu16,
                  a[i] > b[i] ? a[i] : b[i])
FAST_BIN_INT_AVX2(maximum, u32, uint32_t, 8, _mm256_max_epu32,
                  a[i] > b[i] ? a[i] : b[i])

/* u64 maximum: XOR sign-bit bias + signed cmpgt */
static inline void _fast_maximum_u64_avx2(const void *restrict ap,
                                          const void *restrict bp,
                                          void *restrict op, size_t n) {
  const uint64_t *a = (const uint64_t *)ap;
  const uint64_t *b = (const uint64_t *)bp;
  uint64_t *out = (uint64_t *)op;
  const __m256i bias = _mm256_set1_epi64x((long long)0x8000000000000000LL);
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
    __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
    __m256i gt = _mm256_cmpgt_epi64(_mm256_xor_si256(va, bias),
                                     _mm256_xor_si256(vb, bias));
    _mm256_storeu_si256((__m256i *)(out + i),
                        _mm256_blendv_epi8(vb, va, gt));
  }
  for (; i < n; i++)
    out[i] = a[i] > b[i] ? a[i] : b[i];
}

/* ── Maximum (float) ─────────────────────────────────────────────── */

FAST_BIN_F32_AVX2(maximum, _mm256_max_ps, a[i] > b[i] ? a[i] : b[i])
FAST_BIN_F64_AVX2(maximum, _mm256_max_pd, a[i] > b[i] ? a[i] : b[i])

/* ── Minimum (signed) ────────────────────────────────────────────── */

FAST_BIN_INT_AVX2(minimum, i8, int8_t, 32, _mm256_min_epi8,
                  a[i] < b[i] ? a[i] : b[i])
FAST_BIN_INT_AVX2(minimum, i16, int16_t, 16, _mm256_min_epi16,
                  a[i] < b[i] ? a[i] : b[i])
FAST_BIN_INT_AVX2(minimum, i32, int32_t, 8, _mm256_min_epi32,
                  a[i] < b[i] ? a[i] : b[i])

static inline void _fast_minimum_i64_avx2(const void *restrict ap,
                                          const void *restrict bp,
                                          void *restrict op, size_t n) {
  const int64_t *a = (const int64_t *)ap;
  const int64_t *b = (const int64_t *)bp;
  int64_t *out = (int64_t *)op;
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
    __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
    __m256i gt = _mm256_cmpgt_epi64(va, vb);
    _mm256_storeu_si256((__m256i *)(out + i),
                        _mm256_blendv_epi8(va, vb, gt));
  }
  for (; i < n; i++)
    out[i] = a[i] < b[i] ? a[i] : b[i];
}

/* ── Minimum (unsigned) ──────────────────────────────────────────── */

FAST_BIN_INT_AVX2(minimum, u8, uint8_t, 32, _mm256_min_epu8,
                  a[i] < b[i] ? a[i] : b[i])
FAST_BIN_INT_AVX2(minimum, u16, uint16_t, 16, _mm256_min_epu16,
                  a[i] < b[i] ? a[i] : b[i])
FAST_BIN_INT_AVX2(minimum, u32, uint32_t, 8, _mm256_min_epu32,
                  a[i] < b[i] ? a[i] : b[i])

static inline void _fast_minimum_u64_avx2(const void *restrict ap,
                                          const void *restrict bp,
                                          void *restrict op, size_t n) {
  const uint64_t *a = (const uint64_t *)ap;
  const uint64_t *b = (const uint64_t *)bp;
  uint64_t *out = (uint64_t *)op;
  const __m256i bias = _mm256_set1_epi64x((long long)0x8000000000000000LL);
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
    __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
    __m256i gt = _mm256_cmpgt_epi64(_mm256_xor_si256(va, bias),
                                     _mm256_xor_si256(vb, bias));
    _mm256_storeu_si256((__m256i *)(out + i),
                        _mm256_blendv_epi8(va, vb, gt));
  }
  for (; i < n; i++)
    out[i] = a[i] < b[i] ? a[i] : b[i];
}

FAST_BIN_F32_AVX2(minimum, _mm256_min_ps, a[i] < b[i] ? a[i] : b[i])
FAST_BIN_F64_AVX2(minimum, _mm256_min_pd, a[i] < b[i] ? a[i] : b[i])

/* ── Mul (16/32-bit: native mullo) ───────────────────────────────── */

FAST_BIN_INT_AVX2(mul, i16, int16_t, 16, _mm256_mullo_epi16,
                  a[i] * b[i])
FAST_BIN_INT_AVX2(mul, i32, int32_t, 8, _mm256_mullo_epi32,
                  a[i] * b[i])
FAST_BIN_INT_AVX2(mul, u16, uint16_t, 16, _mm256_mullo_epi16,
                  a[i] * b[i])
FAST_BIN_INT_AVX2(mul, u32, uint32_t, 8, _mm256_mullo_epi32,
                  a[i] * b[i])

/* ── Mul i8/u8: widening trick (no native 8-bit multiply) ──────── */

static inline void _fast_mul_i8_avx2(const void *restrict ap,
                                     const void *restrict bp,
                                     void *restrict op, size_t n) {
  const int8_t *a = (const int8_t *)ap;
  const int8_t *b = (const int8_t *)bp;
  int8_t *out = (int8_t *)op;
  const __m256i mask = _mm256_set1_epi16(0x00FF);
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
    __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
    /* Even bytes: mask to 16-bit, multiply, mask result */
    __m256i lo = _mm256_and_si256(
        _mm256_mullo_epi16(_mm256_and_si256(va, mask),
                           _mm256_and_si256(vb, mask)),
        mask);
    /* Odd bytes: shift right 8, multiply, shift left 8 */
    __m256i hi = _mm256_slli_epi16(
        _mm256_mullo_epi16(_mm256_srli_epi16(va, 8),
                           _mm256_srli_epi16(vb, 8)),
        8);
    _mm256_storeu_si256((__m256i *)(out + i), _mm256_or_si256(lo, hi));
  }
  for (; i < n; i++)
    out[i] = (int8_t)(a[i] * b[i]);
}

static inline void _fast_mul_u8_avx2(const void *restrict ap,
                                     const void *restrict bp,
                                     void *restrict op, size_t n) {
  const uint8_t *a = (const uint8_t *)ap;
  const uint8_t *b = (const uint8_t *)bp;
  uint8_t *out = (uint8_t *)op;
  const __m256i mask = _mm256_set1_epi16(0x00FF);
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
    __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
    __m256i lo = _mm256_and_si256(
        _mm256_mullo_epi16(_mm256_and_si256(va, mask),
                           _mm256_and_si256(vb, mask)),
        mask);
    __m256i hi = _mm256_slli_epi16(
        _mm256_mullo_epi16(_mm256_srli_epi16(va, 8),
                           _mm256_srli_epi16(vb, 8)),
        8);
    _mm256_storeu_si256((__m256i *)(out + i), _mm256_or_si256(lo, hi));
  }
  for (; i < n; i++)
    out[i] = (uint8_t)(a[i] * b[i]);
}

/* ── Mul i64/u64: scalar (no efficient AVX2 emulation) ───────────── */

static inline void _fast_mul_i64_avx2(const void *restrict ap,
                                      const void *restrict bp,
                                      void *restrict op, size_t n) {
  const int64_t *a = (const int64_t *)ap;
  const int64_t *b = (const int64_t *)bp;
  int64_t *out = (int64_t *)op;
  for (size_t i = 0; i < n; i++)
    out[i] = a[i] * b[i];
}

static inline void _fast_mul_u64_avx2(const void *restrict ap,
                                      const void *restrict bp,
                                      void *restrict op, size_t n) {
  const uint64_t *a = (const uint64_t *)ap;
  const uint64_t *b = (const uint64_t *)bp;
  uint64_t *out = (uint64_t *)op;
  for (size_t i = 0; i < n; i++)
    out[i] = a[i] * b[i];
}

FAST_BIN_F32_AVX2(mul, _mm256_mul_ps, a[i] * b[i])
FAST_BIN_F64_AVX2(mul, _mm256_mul_pd, a[i] * b[i])

#undef FAST_BIN_INT_AVX2
#undef FAST_BIN_F32_AVX2
#undef FAST_BIN_F64_AVX2

/* ════════════════════════════════════════════════════════════════════
 * Unary operations
 * ════════════════════════════════════════════════════════════════ */

#define FAST_UN_INT_AVX2(OP, SFX, CT, VPV, VEC_OP, TAIL_EXPR)         \
  static inline void _fast_##OP##_##SFX##_avx2(                       \
      const void *restrict ap, void *restrict op, size_t n) {         \
    const CT *a = (const CT *)ap;                                      \
    CT *out = (CT *)op;                                                \
    size_t i = 0;                                                      \
    for (; i + (VPV) <= n; i += (VPV)) {                               \
      __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));       \
      _mm256_storeu_si256((__m256i *)(out + i), VEC_OP);               \
    }                                                                  \
    for (; i < n; i++)                                                 \
      out[i] = (CT)(TAIL_EXPR);                                       \
  }

/* ── Neg ─────────────────────────────────────────────────────────── */

#define NEG_VEC(W) _mm256_sub_epi##W(_mm256_setzero_si256(), va)

FAST_UN_INT_AVX2(neg, i8, int8_t, 32, NEG_VEC(8), -a[i])
FAST_UN_INT_AVX2(neg, i16, int16_t, 16, NEG_VEC(16), -a[i])
FAST_UN_INT_AVX2(neg, i32, int32_t, 8, NEG_VEC(32), -a[i])
FAST_UN_INT_AVX2(neg, i64, int64_t, 4, NEG_VEC(64), -a[i])
FAST_UN_INT_AVX2(neg, u8, uint8_t, 32, NEG_VEC(8), (uint8_t)(-(int8_t)a[i]))
FAST_UN_INT_AVX2(neg, u16, uint16_t, 16, NEG_VEC(16),
                 (uint16_t)(-(int16_t)a[i]))
FAST_UN_INT_AVX2(neg, u32, uint32_t, 8, NEG_VEC(32),
                 (uint32_t)(-(int32_t)a[i]))
FAST_UN_INT_AVX2(neg, u64, uint64_t, 4, NEG_VEC(64),
                 (uint64_t)(-(int64_t)a[i]))
#undef NEG_VEC

static inline void _fast_neg_f32_avx2(const void *restrict ap,
                                      void *restrict op, size_t n) {
  const float *a = (const float *)ap;
  float *out = (float *)op;
  const __m256 sign = _mm256_set1_ps(-0.0f);
  size_t i = 0;
  for (; i + 8 <= n; i += 8)
    _mm256_storeu_ps(out + i,
                     _mm256_xor_ps(_mm256_loadu_ps(a + i), sign));
  for (; i < n; i++)
    out[i] = -a[i];
}

static inline void _fast_neg_f64_avx2(const void *restrict ap,
                                      void *restrict op, size_t n) {
  const double *a = (const double *)ap;
  double *out = (double *)op;
  const __m256d sign = _mm256_set1_pd(-0.0);
  size_t i = 0;
  for (; i + 4 <= n; i += 4)
    _mm256_storeu_pd(out + i,
                     _mm256_xor_pd(_mm256_loadu_pd(a + i), sign));
  for (; i < n; i++)
    out[i] = -a[i];
}

/* ── Abs (signed integers) ───────────────────────────────────────── */

FAST_UN_INT_AVX2(abs, i8, int8_t, 32, _mm256_abs_epi8(va),
                 (int8_t)(a[i] < 0 ? -a[i] : a[i]))
FAST_UN_INT_AVX2(abs, i16, int16_t, 16, _mm256_abs_epi16(va),
                 (int16_t)(a[i] < 0 ? -a[i] : a[i]))
FAST_UN_INT_AVX2(abs, i32, int32_t, 8, _mm256_abs_epi32(va),
                 (int32_t)(a[i] < 0 ? -a[i] : a[i]))

/* i64 abs: max(val, -val) */
static inline void _fast_abs_i64_avx2(const void *restrict ap,
                                      void *restrict op, size_t n) {
  const int64_t *a = (const int64_t *)ap;
  int64_t *out = (int64_t *)op;
  const __m256i zero = _mm256_setzero_si256();
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
    __m256i neg = _mm256_sub_epi64(zero, va);
    __m256i pos = _mm256_cmpgt_epi64(va, zero);
    _mm256_storeu_si256((__m256i *)(out + i),
                        _mm256_blendv_epi8(neg, va, pos));
  }
  for (; i < n; i++)
    out[i] = a[i] < 0 ? -a[i] : a[i];
}

static inline void _fast_abs_f32_avx2(const void *restrict ap,
                                      void *restrict op, size_t n) {
  const float *a = (const float *)ap;
  float *out = (float *)op;
  const __m256 mask =
      _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
  size_t i = 0;
  for (; i + 8 <= n; i += 8)
    _mm256_storeu_ps(out + i,
                     _mm256_and_ps(_mm256_loadu_ps(a + i), mask));
  for (; i < n; i++)
    out[i] = a[i] < 0 ? -a[i] : a[i];
}

static inline void _fast_abs_f64_avx2(const void *restrict ap,
                                      void *restrict op, size_t n) {
  const double *a = (const double *)ap;
  double *out = (double *)op;
  const __m256d mask =
      _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL));
  size_t i = 0;
  for (; i + 4 <= n; i += 4)
    _mm256_storeu_pd(out + i,
                     _mm256_and_pd(_mm256_loadu_pd(a + i), mask));
  for (; i < n; i++)
    out[i] = a[i] < 0 ? -a[i] : a[i];
}

#undef FAST_UN_INT_AVX2

#endif /* NUMC_ELEMWISE_AVX2_H */
