/**
 * @file elemwise_scalar_avx512.h
 * @brief AVX-512 scalar arithmetic kernels for all 10 types.
 *
 * Operations: add_scalar, sub_scalar, mul_scalar
 * Types: i8, i16, i32, i64, u8, u16, u32, u64, f32, f64
 *
 * Special cases: i8/u8 mul (widening trick, no native 8-bit multiply),
 * i64/u64 mul (scalar fallback — _mm512_mullo_epi64 requires AVX512DQ).
 */
#ifndef NUMC_ELEMWISE_SCALAR_AVX512_H
#define NUMC_ELEMWISE_SCALAR_AVX512_H

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

/* ════════════════════════════════════════════════════════════════════
 * Generic integer scalar macro (for ops with native AVX-512 support)
 * ════════════════════════════════════════════════════════════════ */

#define FAST_SC_INT_AVX512(OP, SFX, CT, VPV, BCAST, VEC_OP, TAIL_OP)       \
  static inline void _fast_##OP##_scalar_##SFX##_avx512(                   \
      const void *restrict ap, const void *restrict sp, void *restrict op, \
      size_t n) {                                                          \
    const CT *a = (const CT *)ap;                                          \
    const CT s = *(const CT *)sp;                                          \
    CT *out = (CT *)op;                                                    \
    const __m512i vs = BCAST(s);                                           \
    size_t i = 0;                                                          \
    for (; i + (VPV) <= n; i += (VPV)) {                                   \
      __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));           \
      _mm512_storeu_si512((__m512i *)(out + i), VEC_OP(va, vs));           \
    }                                                                      \
    for (; i < n; i++)                                                     \
      out[i] = (CT)(a[i] TAIL_OP s);                                       \
  }

#define FAST_SC_F32_AVX512(OP, VEC_OP, TAIL_OP)                            \
  static inline void _fast_##OP##_scalar_f32_avx512(                       \
      const void *restrict ap, const void *restrict sp, void *restrict op, \
      size_t n) {                                                          \
    const float *a = (const float *)ap;                                    \
    const float s = *(const float *)sp;                                    \
    float *out = (float *)op;                                              \
    const __m512 vs = _mm512_set1_ps(s);                                   \
    size_t i = 0;                                                          \
    for (; i + 16 <= n; i += 16) {                                         \
      __m512 va = _mm512_loadu_ps(a + i);                                  \
      _mm512_storeu_ps(out + i, VEC_OP(va, vs));                           \
    }                                                                      \
    for (; i < n; i++)                                                     \
      out[i] = (float)(a[i] TAIL_OP s);                                    \
  }

#define FAST_SC_F64_AVX512(OP, VEC_OP, TAIL_OP)                            \
  static inline void _fast_##OP##_scalar_f64_avx512(                       \
      const void *restrict ap, const void *restrict sp, void *restrict op, \
      size_t n) {                                                          \
    const double *a = (const double *)ap;                                  \
    const double s = *(const double *)sp;                                  \
    double *out = (double *)op;                                            \
    const __m512d vs = _mm512_set1_pd(s);                                  \
    size_t i = 0;                                                          \
    for (; i + 8 <= n; i += 8) {                                           \
      __m512d va = _mm512_loadu_pd(a + i);                                 \
      _mm512_storeu_pd(out + i, VEC_OP(va, vs));                           \
    }                                                                      \
    for (; i < n; i++)                                                     \
      out[i] = (double)(a[i] TAIL_OP s);                                   \
  }

/* ── Add scalar ─────────────────────────────────────────────────── */

FAST_SC_INT_AVX512(add, i8, int8_t, 64, _mm512_set1_epi8, _mm512_add_epi8, +)
FAST_SC_INT_AVX512(add, i16, int16_t, 32, _mm512_set1_epi16, _mm512_add_epi16,
                   +)
FAST_SC_INT_AVX512(add, i32, int32_t, 16, _mm512_set1_epi32, _mm512_add_epi32,
                   +)
FAST_SC_INT_AVX512(add, i64, int64_t, 8, _mm512_set1_epi64, _mm512_add_epi64, +)
FAST_SC_INT_AVX512(add, u8, uint8_t, 64, _mm512_set1_epi8, _mm512_add_epi8, +)
FAST_SC_INT_AVX512(add, u16, uint16_t, 32, _mm512_set1_epi16, _mm512_add_epi16,
                   +)
FAST_SC_INT_AVX512(add, u32, uint32_t, 16, _mm512_set1_epi32, _mm512_add_epi32,
                   +)
FAST_SC_INT_AVX512(add, u64, uint64_t, 8, _mm512_set1_epi64, _mm512_add_epi64,
                   +)
FAST_SC_F32_AVX512(add, _mm512_add_ps, +)
FAST_SC_F64_AVX512(add, _mm512_add_pd, +)

/* ── Sub scalar ─────────────────────────────────────────────────── */

FAST_SC_INT_AVX512(sub, i8, int8_t, 64, _mm512_set1_epi8, _mm512_sub_epi8, -)
FAST_SC_INT_AVX512(sub, i16, int16_t, 32, _mm512_set1_epi16, _mm512_sub_epi16,
                   -)
FAST_SC_INT_AVX512(sub, i32, int32_t, 16, _mm512_set1_epi32, _mm512_sub_epi32,
                   -)
FAST_SC_INT_AVX512(sub, i64, int64_t, 8, _mm512_set1_epi64, _mm512_sub_epi64, -)
FAST_SC_INT_AVX512(sub, u8, uint8_t, 64, _mm512_set1_epi8, _mm512_sub_epi8, -)
FAST_SC_INT_AVX512(sub, u16, uint16_t, 32, _mm512_set1_epi16, _mm512_sub_epi16,
                   -)
FAST_SC_INT_AVX512(sub, u32, uint32_t, 16, _mm512_set1_epi32, _mm512_sub_epi32,
                   -)
FAST_SC_INT_AVX512(sub, u64, uint64_t, 8, _mm512_set1_epi64, _mm512_sub_epi64,
                   -)
FAST_SC_F32_AVX512(sub, _mm512_sub_ps, -)
FAST_SC_F64_AVX512(sub, _mm512_sub_pd, -)

/* ── Mul scalar (16/32-bit: native mullo) ───────────────────────── */

FAST_SC_INT_AVX512(mul, i16, int16_t, 32, _mm512_set1_epi16,
                   _mm512_mullo_epi16, *)
FAST_SC_INT_AVX512(mul, i32, int32_t, 16, _mm512_set1_epi32,
                   _mm512_mullo_epi32, *)
FAST_SC_INT_AVX512(mul, u16, uint16_t, 32, _mm512_set1_epi16,
                   _mm512_mullo_epi16, *)
FAST_SC_INT_AVX512(mul, u32, uint32_t, 16, _mm512_set1_epi32,
                   _mm512_mullo_epi32, *)

/* ── Mul scalar i8: widening trick (no native 8-bit multiply) ──── */

static inline void _fast_mul_scalar_i8_avx512(const void *restrict ap,
                                              const void *restrict sp,
                                              void *restrict op, size_t n) {
  const int8_t *a = (const int8_t *)ap;
  const int8_t s = *(const int8_t *)sp;
  int8_t *out = (int8_t *)op;
  const __m512i vs = _mm512_set1_epi8(s);
  const __m512i mask = _mm512_set1_epi16(0x00FF);
  size_t i = 0;
  for (; i + 64 <= n; i += 64) {
    __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));
    /* Even bytes: mask to 16-bit, multiply, mask result */
    __m512i lo =
        _mm512_and_si512(_mm512_mullo_epi16(_mm512_and_si512(va, mask),
                                            _mm512_and_si512(vs, mask)),
                         mask);
    /* Odd bytes: shift right 8, multiply, shift left 8 */
    __m512i hi = _mm512_slli_epi16(
        _mm512_mullo_epi16(_mm512_srli_epi16(va, 8), _mm512_srli_epi16(vs, 8)),
        8);
    _mm512_storeu_si512((__m512i *)(out + i), _mm512_or_si512(lo, hi));
  }
  for (; i < n; i++)
    out[i] = (int8_t)(a[i] * s);
}

static inline void _fast_mul_scalar_u8_avx512(const void *restrict ap,
                                              const void *restrict sp,
                                              void *restrict op, size_t n) {
  const uint8_t *a = (const uint8_t *)ap;
  const uint8_t s = *(const uint8_t *)sp;
  uint8_t *out = (uint8_t *)op;
  const __m512i vs = _mm512_set1_epi8((char)s);
  const __m512i mask = _mm512_set1_epi16(0x00FF);
  size_t i = 0;
  for (; i + 64 <= n; i += 64) {
    __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));
    __m512i lo =
        _mm512_and_si512(_mm512_mullo_epi16(_mm512_and_si512(va, mask),
                                            _mm512_and_si512(vs, mask)),
                         mask);
    __m512i hi = _mm512_slli_epi16(
        _mm512_mullo_epi16(_mm512_srli_epi16(va, 8), _mm512_srli_epi16(vs, 8)),
        8);
    _mm512_storeu_si512((__m512i *)(out + i), _mm512_or_si512(lo, hi));
  }
  for (; i < n; i++)
    out[i] = (uint8_t)(a[i] * s);
}

/* ── Mul scalar i64/u64: scalar fallback (AVX512DQ not guaranteed) ─ */

static inline void _fast_mul_scalar_i64_avx512(const void *restrict ap,
                                               const void *restrict sp,
                                               void *restrict op, size_t n) {
  const int64_t *a = (const int64_t *)ap;
  const int64_t s = *(const int64_t *)sp;
  int64_t *out = (int64_t *)op;
  for (size_t i = 0; i < n; i++)
    out[i] = a[i] * s;
}

static inline void _fast_mul_scalar_u64_avx512(const void *restrict ap,
                                               const void *restrict sp,
                                               void *restrict op, size_t n) {
  const uint64_t *a = (const uint64_t *)ap;
  const uint64_t s = *(const uint64_t *)sp;
  uint64_t *out = (uint64_t *)op;
  for (size_t i = 0; i < n; i++)
    out[i] = a[i] * s;
}

FAST_SC_F32_AVX512(mul, _mm512_mul_ps, *)
FAST_SC_F64_AVX512(mul, _mm512_mul_pd, *)

#undef FAST_SC_INT_AVX512
#undef FAST_SC_F32_AVX512
#undef FAST_SC_F64_AVX512

#endif /* NUMC_ELEMWISE_SCALAR_AVX512_H */
