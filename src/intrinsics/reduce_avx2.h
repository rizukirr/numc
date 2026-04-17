#ifndef NUMC_REDUCE_AVX2_H
#define NUMC_REDUCE_AVX2_H

#include <immintrin.h>
#include <stdint.h>

// clang-format off

/* -- load/store helpers ---------------------------------------------- */

#define RLOADI(p) _mm256_loadu_si256((const __m256i *)(p))
#define RSTOREI(p, v) _mm256_storeu_si256((__m256i *)(p), v)

#define _RSMIN(a, b) ((a) < (b) ? (a) : (b))
#define _RSMAX(a, b) ((a) > (b) ? (a) : (b))

/* -- horizontal reduction helpers ------------------------------------ */

/* 8-bit: 256→128→64→32→16→8 bits */
#define DEFINE_HREDUCE_8(NAME, CT, CMP128)              \
  static inline CT NAME(__m256i v) {                    \
    __m128i lo = _mm256_castsi256_si128(v);             \
    __m128i hi = _mm256_extracti128_si256(v, 1);        \
    __m128i m = CMP128(lo, hi);                         \
    m = CMP128(m, _mm_srli_si128(m, 8));                \
    m = CMP128(m, _mm_srli_si128(m, 4));                \
    m = CMP128(m, _mm_srli_si128(m, 2));                \
    m = CMP128(m, _mm_srli_si128(m, 1));                \
    return (CT)_mm_extract_epi8(m, 0);                  \
  }

/* 16-bit: 256→128→64→32→16 bits */
#define DEFINE_HREDUCE_16(NAME, CT, CMP128)             \
  static inline CT NAME(__m256i v) {                    \
    __m128i lo = _mm256_castsi256_si128(v);             \
    __m128i hi = _mm256_extracti128_si256(v, 1);        \
    __m128i m = CMP128(lo, hi);                         \
    m = CMP128(m, _mm_srli_si128(m, 8));                \
    m = CMP128(m, _mm_srli_si128(m, 4));                \
    m = CMP128(m, _mm_srli_si128(m, 2));                \
    return (CT)_mm_extract_epi16(m, 0);                 \
  }

/* 32-bit: 256→128→64→32 bits */
#define DEFINE_HREDUCE_32(NAME, CT, CMP128)             \
  static inline CT NAME(__m256i v) {                    \
    __m128i lo = _mm256_castsi256_si128(v);             \
    __m128i hi = _mm256_extracti128_si256(v, 1);        \
    __m128i m = CMP128(lo, hi);                         \
    m = CMP128(m, _mm_srli_si128(m, 8));                \
    m = CMP128(m, _mm_srli_si128(m, 4));                \
    return (CT)_mm_cvtsi128_si32(m);                    \
  }

DEFINE_HREDUCE_8(_hmin_epi8_avx2,   int8_t,   _mm_min_epi8)
DEFINE_HREDUCE_8(_hmax_epi8_avx2,   int8_t,   _mm_max_epi8)
DEFINE_HREDUCE_8(_hmin_epu8_avx2,   uint8_t,  _mm_min_epu8)
DEFINE_HREDUCE_8(_hmax_epu8_avx2,   uint8_t,  _mm_max_epu8)

DEFINE_HREDUCE_16(_hmin_epi16_avx2, int16_t,  _mm_min_epi16)
DEFINE_HREDUCE_16(_hmax_epi16_avx2, int16_t,  _mm_max_epi16)
DEFINE_HREDUCE_16(_hmin_epu16_avx2, uint16_t, _mm_min_epu16)
DEFINE_HREDUCE_16(_hmax_epu16_avx2, uint16_t, _mm_max_epu16)

DEFINE_HREDUCE_32(_hmin_epi32_avx2, int32_t,  _mm_min_epi32)
DEFINE_HREDUCE_32(_hmax_epi32_avx2, int32_t,  _mm_max_epi32)
DEFINE_HREDUCE_32(_hmin_epu32_avx2, uint32_t, _mm_min_epu32)
DEFINE_HREDUCE_32(_hmax_epu32_avx2, uint32_t, _mm_max_epu32)

#undef DEFINE_HREDUCE_8
#undef DEFINE_HREDUCE_16
#undef DEFINE_HREDUCE_32

/* -- full array min/max reduction ------------------------------------ *
 *
 * 4 vector accumulators, 128 bytes/iteration for 8-bit types.
 * Single-vector cleanup loop, then scalar tail.                      */

#define DEFINE_REDUCE_FULL_AVX2(NAME, CT, INIT_VEC, CMP256, HREDUCE, SCMP) \
  static inline CT NAME(const CT *restrict a, size_t n) {                  \
    const size_t EPV = 32 / sizeof(CT);                                    \
    __m256i a0 = INIT_VEC, a1 = INIT_VEC;                                  \
    __m256i a2 = INIT_VEC, a3 = INIT_VEC;                                  \
    size_t i = 0;                                                          \
    for (; i + 4 * EPV <= n; i += 4 * EPV) {                               \
      a0 = CMP256(a0, RLOADI(a + i));                                      \
      a1 = CMP256(a1, RLOADI(a + i + EPV));                                \
      a2 = CMP256(a2, RLOADI(a + i + 2 * EPV));                            \
      a3 = CMP256(a3, RLOADI(a + i + 3 * EPV));                            \
    }                                                                      \
    a0 = CMP256(CMP256(a0, a1), CMP256(a2, a3));                           \
    for (; i + EPV <= n; i += EPV)                                         \
      a0 = CMP256(a0, RLOADI(a + i));                                      \
    CT result = HREDUCE(a0);                                               \
    for (; i < n; i++)                                                     \
      result = SCMP(a[i], result);                                         \
    return result;                                                         \
  }

/* min reductions */
DEFINE_REDUCE_FULL_AVX2(reduce_min_i8_avx2,  int8_t,
  _mm256_set1_epi8(INT8_MAX),
  _mm256_min_epi8,  _hmin_epi8_avx2,  _RSMIN)
DEFINE_REDUCE_FULL_AVX2(reduce_min_u8_avx2,  uint8_t,
  _mm256_set1_epi8(-1),
  _mm256_min_epu8,  _hmin_epu8_avx2,  _RSMIN)
DEFINE_REDUCE_FULL_AVX2(reduce_min_i16_avx2, int16_t,
  _mm256_set1_epi16(INT16_MAX),
  _mm256_min_epi16, _hmin_epi16_avx2, _RSMIN)
DEFINE_REDUCE_FULL_AVX2(reduce_min_u16_avx2, uint16_t,
  _mm256_set1_epi16(-1),
  _mm256_min_epu16, _hmin_epu16_avx2, _RSMIN)
DEFINE_REDUCE_FULL_AVX2(reduce_min_i32_avx2, int32_t,
  _mm256_set1_epi32(INT32_MAX),
  _mm256_min_epi32, _hmin_epi32_avx2, _RSMIN)
DEFINE_REDUCE_FULL_AVX2(reduce_min_u32_avx2, uint32_t,
  _mm256_set1_epi32(-1),
  _mm256_min_epu32, _hmin_epu32_avx2, _RSMIN)

/* max reductions */
DEFINE_REDUCE_FULL_AVX2(reduce_max_i8_avx2,  int8_t,
  _mm256_set1_epi8(INT8_MIN),
  _mm256_max_epi8,  _hmax_epi8_avx2,  _RSMAX)
DEFINE_REDUCE_FULL_AVX2(reduce_max_u8_avx2,  uint8_t,
  _mm256_setzero_si256(),
  _mm256_max_epu8,  _hmax_epu8_avx2,  _RSMAX)
DEFINE_REDUCE_FULL_AVX2(reduce_max_i16_avx2, int16_t,
  _mm256_set1_epi16(INT16_MIN),
  _mm256_max_epi16, _hmax_epi16_avx2, _RSMAX)
DEFINE_REDUCE_FULL_AVX2(reduce_max_u16_avx2, uint16_t,
  _mm256_setzero_si256(),
  _mm256_max_epu16, _hmax_epu16_avx2, _RSMAX)
DEFINE_REDUCE_FULL_AVX2(reduce_max_i32_avx2, int32_t,
  _mm256_set1_epi32(INT32_MIN),
  _mm256_max_epi32, _hmax_epi32_avx2, _RSMAX)
DEFINE_REDUCE_FULL_AVX2(reduce_max_u32_avx2, uint32_t,
  _mm256_setzero_si256(),
  _mm256_max_epu32, _hmax_epu32_avx2, _RSMAX)

#undef DEFINE_REDUCE_FULL_AVX2

/* -- fused row-reduce (axis-1 reduction) ----------------------------- *
 *
 * Processes 4 rows at a time, vectorizing the inner column loop.
 * For int8: 32 columns per SIMD iteration (vs 1 scalar).            */

#define DEFINE_FUSED_REDUCE_AVX2(NAME, CT, CMP256, SCMP)                     \
  static inline void NAME(const char *restrict base, intptr_t row_stride,    \
                           size_t nrows, char *restrict dst,                 \
                           size_t ncols) {                                   \
    CT *restrict d = (CT *)dst;                                              \
    const size_t EPV = 32 / sizeof(CT);                                      \
    size_t r = 0;                                                            \
    for (; r + 4 <= nrows; r += 4) {                                         \
      const CT *restrict s0 = (const CT *)(base + r * row_stride);           \
      const CT *restrict s1 =                                                \
          (const CT *)(base + (r + 1) * row_stride);                         \
      const CT *restrict s2 =                                                \
          (const CT *)(base + (r + 2) * row_stride);                         \
      const CT *restrict s3 =                                                \
          (const CT *)(base + (r + 3) * row_stride);                         \
      size_t i = 0;                                                          \
      for (; i + EPV <= ncols; i += EPV) {                                   \
        __m256i dv  = RLOADI(d + i);                                         \
        __m256i v01 = CMP256(RLOADI(s0 + i), RLOADI(s1 + i));               \
        __m256i v23 = CMP256(RLOADI(s2 + i), RLOADI(s3 + i));               \
        RSTOREI(d + i, CMP256(dv, CMP256(v01, v23)));                       \
      }                                                                      \
      for (; i < ncols; i++) {                                               \
        CT v = SCMP(s0[i], s1[i]);                                           \
        v = SCMP(v, s2[i]);                                                  \
        v = SCMP(v, s3[i]);                                                  \
        d[i] = SCMP(v, d[i]);                                               \
      }                                                                      \
    }                                                                        \
    for (; r < nrows; r++) {                                                  \
      const CT *restrict s =                                                 \
          (const CT *)(base + r * row_stride);                               \
      size_t i = 0;                                                          \
      for (; i + EPV <= ncols; i += EPV)                                     \
        RSTOREI(d + i, CMP256(RLOADI(d + i), RLOADI(s + i)));               \
      for (; i < ncols; i++)                                                  \
        d[i] = SCMP(s[i], d[i]);                                            \
    }                                                                        \
  }

/* min fused */
DEFINE_FUSED_REDUCE_AVX2(_min_fused_i8_avx2,  int8_t,
  _mm256_min_epi8,  _RSMIN)
DEFINE_FUSED_REDUCE_AVX2(_min_fused_u8_avx2,  uint8_t,
  _mm256_min_epu8,  _RSMIN)
DEFINE_FUSED_REDUCE_AVX2(_min_fused_i16_avx2, int16_t,
  _mm256_min_epi16, _RSMIN)
DEFINE_FUSED_REDUCE_AVX2(_min_fused_u16_avx2, uint16_t,
  _mm256_min_epu16, _RSMIN)
DEFINE_FUSED_REDUCE_AVX2(_min_fused_i32_avx2, int32_t,
  _mm256_min_epi32, _RSMIN)
DEFINE_FUSED_REDUCE_AVX2(_min_fused_u32_avx2, uint32_t,
  _mm256_min_epu32, _RSMIN)

/* max fused */
DEFINE_FUSED_REDUCE_AVX2(_max_fused_i8_avx2,  int8_t,
  _mm256_max_epi8,  _RSMAX)
DEFINE_FUSED_REDUCE_AVX2(_max_fused_u8_avx2,  uint8_t,
  _mm256_max_epu8,  _RSMAX)
DEFINE_FUSED_REDUCE_AVX2(_max_fused_i16_avx2, int16_t,
  _mm256_max_epi16, _RSMAX)
DEFINE_FUSED_REDUCE_AVX2(_max_fused_u16_avx2, uint16_t,
  _mm256_max_epu16, _RSMAX)
DEFINE_FUSED_REDUCE_AVX2(_max_fused_i32_avx2, int32_t,
  _mm256_max_epi32, _RSMAX)
DEFINE_FUSED_REDUCE_AVX2(_max_fused_u32_avx2, uint32_t,
  _mm256_max_epu32, _RSMAX)

#undef DEFINE_FUSED_REDUCE_AVX2

/* -- 64-bit integer min/max (emulated, no native AVX2 support) ---- *
 *
 * AVX2 lacks vpminsq/vpmaxsq/vpminuq/vpmaxuq.  We emulate using:
 *   signed:   cmp = vpcmpgtq(a, b);  result = blendvpd(b, a, cmp)
 *   unsigned: XOR with sign-bit to reduce to signed, cmpgt, blendvpd */

static inline __m256i _mm256_max_epi64_emu(__m256i a, __m256i b) {
  __m256i gt = _mm256_cmpgt_epi64(a, b);
  return _mm256_blendv_epi8(b, a, gt);
}
static inline __m256i _mm256_min_epi64_emu(__m256i a, __m256i b) {
  __m256i gt = _mm256_cmpgt_epi64(a, b);
  return _mm256_blendv_epi8(a, b, gt);
}
static inline __m256i _mm256_max_epu64_emu(__m256i a, __m256i b) {
  __m256i sign = _mm256_set1_epi64x((int64_t)0x8000000000000000ULL);
  __m256i gt = _mm256_cmpgt_epi64(_mm256_xor_si256(a, sign),
                                  _mm256_xor_si256(b, sign));
  return _mm256_blendv_epi8(b, a, gt);
}
static inline __m256i _mm256_min_epu64_emu(__m256i a, __m256i b) {
  __m256i sign = _mm256_set1_epi64x((int64_t)0x8000000000000000ULL);
  __m256i gt = _mm256_cmpgt_epi64(_mm256_xor_si256(a, sign),
                                  _mm256_xor_si256(b, sign));
  return _mm256_blendv_epi8(a, b, gt);
}

/* 64-bit horizontal reduction: 256→128→64 bits */
static inline int64_t _hmax_epi64_avx2(__m256i v) {
  __m128i lo = _mm256_castsi256_si128(v);
  __m128i hi = _mm256_extracti128_si256(v, 1);
  /* max of lo and hi 128-bit halves (2 x i64 each) */
  __m128i gt = _mm_cmpgt_epi64(lo, hi);
  __m128i m = _mm_blendv_epi8(hi, lo, gt);
  /* max of two 64-bit lanes */
  __m128i shuf = _mm_srli_si128(m, 8);
  __m128i gt2 = _mm_cmpgt_epi64(m, shuf);
  m = _mm_blendv_epi8(shuf, m, gt2);
  return _mm_cvtsi128_si64(m);
}
static inline int64_t _hmin_epi64_avx2(__m256i v) {
  __m128i lo = _mm256_castsi256_si128(v);
  __m128i hi = _mm256_extracti128_si256(v, 1);
  __m128i gt = _mm_cmpgt_epi64(lo, hi);
  __m128i m = _mm_blendv_epi8(lo, hi, gt);
  __m128i shuf = _mm_srli_si128(m, 8);
  __m128i gt2 = _mm_cmpgt_epi64(m, shuf);
  m = _mm_blendv_epi8(m, shuf, gt2);
  return _mm_cvtsi128_si64(m);
}
static inline uint64_t _hmax_epu64_avx2(__m256i v) {
  __m128i sign = _mm_set1_epi64x((int64_t)0x8000000000000000ULL);
  __m128i lo = _mm256_castsi256_si128(v);
  __m128i hi = _mm256_extracti128_si256(v, 1);
  __m128i gt = _mm_cmpgt_epi64(_mm_xor_si128(lo, sign),
                               _mm_xor_si128(hi, sign));
  __m128i m = _mm_blendv_epi8(hi, lo, gt);
  __m128i shuf = _mm_srli_si128(m, 8);
  __m128i gt2 = _mm_cmpgt_epi64(_mm_xor_si128(m, sign),
                                _mm_xor_si128(shuf, sign));
  m = _mm_blendv_epi8(shuf, m, gt2);
  return (uint64_t)_mm_cvtsi128_si64(m);
}
static inline uint64_t _hmin_epu64_avx2(__m256i v) {
  __m128i sign = _mm_set1_epi64x((int64_t)0x8000000000000000ULL);
  __m128i lo = _mm256_castsi256_si128(v);
  __m128i hi = _mm256_extracti128_si256(v, 1);
  __m128i gt = _mm_cmpgt_epi64(_mm_xor_si128(lo, sign),
                               _mm_xor_si128(hi, sign));
  __m128i m = _mm_blendv_epi8(lo, hi, gt);
  __m128i shuf = _mm_srli_si128(m, 8);
  __m128i gt2 = _mm_cmpgt_epi64(_mm_xor_si128(m, sign),
                                _mm_xor_si128(shuf, sign));
  m = _mm_blendv_epi8(m, shuf, gt2);
  return (uint64_t)_mm_cvtsi128_si64(m);
}

/* -- 64-bit full array reductions ------------------------------------ */

#define DEFINE_REDUCE_FULL_64_AVX2(NAME, CT, INIT_VEC, CMP256, HREDUCE, \
                                   SCMP)                                 \
  static inline CT NAME(const CT *restrict a, size_t n) {                \
    const size_t EPV = 4; /* 256 / 64 */                                 \
    __m256i a0 = INIT_VEC, a1 = INIT_VEC;                                \
    __m256i a2 = INIT_VEC, a3 = INIT_VEC;                                \
    size_t i = 0;                                                        \
    for (; i + 4 * EPV <= n; i += 4 * EPV) {                             \
      a0 = CMP256(a0, RLOADI(a + i));                                    \
      a1 = CMP256(a1, RLOADI(a + i + EPV));                              \
      a2 = CMP256(a2, RLOADI(a + i + 2 * EPV));                          \
      a3 = CMP256(a3, RLOADI(a + i + 3 * EPV));                          \
    }                                                                    \
    a0 = CMP256(CMP256(a0, a1), CMP256(a2, a3));                         \
    for (; i + EPV <= n; i += EPV)                                       \
      a0 = CMP256(a0, RLOADI(a + i));                                    \
    CT result = HREDUCE(a0);                                             \
    for (; i < n; i++)                                                   \
      result = SCMP(a[i], result);                                       \
    return result;                                                       \
  }

DEFINE_REDUCE_FULL_64_AVX2(reduce_max_i64_avx2, int64_t,
  _mm256_set1_epi64x(INT64_MIN),
  _mm256_max_epi64_emu, _hmax_epi64_avx2, _RSMAX)
DEFINE_REDUCE_FULL_64_AVX2(reduce_max_u64_avx2, uint64_t,
  _mm256_setzero_si256(),
  _mm256_max_epu64_emu, _hmax_epu64_avx2, _RSMAX)
DEFINE_REDUCE_FULL_64_AVX2(reduce_min_i64_avx2, int64_t,
  _mm256_set1_epi64x(INT64_MAX),
  _mm256_min_epi64_emu, _hmin_epi64_avx2, _RSMIN)
DEFINE_REDUCE_FULL_64_AVX2(reduce_min_u64_avx2, uint64_t,
  _mm256_set1_epi64x(-1),
  _mm256_min_epu64_emu, _hmin_epu64_avx2, _RSMIN)

#undef DEFINE_REDUCE_FULL_64_AVX2

/* -- 64-bit fused row-reduce (axis-1) -------------------------------- */

#define DEFINE_FUSED_64_AVX2(NAME, CT, CMP256, SCMP)                          \
  static inline void NAME(const char *restrict base, intptr_t row_stride,     \
                           size_t nrows, char *restrict dst,                   \
                           size_t ncols) {                                     \
    CT *restrict d = (CT *)dst;                                               \
    const size_t EPV = 4;                                                     \
    size_t r = 0;                                                             \
    for (; r + 4 <= nrows; r += 4) {                                          \
      const CT *restrict s0 = (const CT *)(base + r * row_stride);            \
      const CT *restrict s1 =                                                 \
          (const CT *)(base + (r + 1) * row_stride);                          \
      const CT *restrict s2 =                                                 \
          (const CT *)(base + (r + 2) * row_stride);                          \
      const CT *restrict s3 =                                                 \
          (const CT *)(base + (r + 3) * row_stride);                          \
      size_t i = 0;                                                           \
      for (; i + EPV <= ncols; i += EPV) {                                    \
        __m256i dv  = RLOADI(d + i);                                          \
        __m256i v01 = CMP256(RLOADI(s0 + i), RLOADI(s1 + i));                \
        __m256i v23 = CMP256(RLOADI(s2 + i), RLOADI(s3 + i));                \
        RSTOREI(d + i, CMP256(dv, CMP256(v01, v23)));                        \
      }                                                                       \
      for (; i < ncols; i++) {                                                \
        CT v = SCMP(s0[i], s1[i]);                                            \
        v = SCMP(v, s2[i]);                                                   \
        v = SCMP(v, s3[i]);                                                   \
        d[i] = SCMP(v, d[i]);                                                \
      }                                                                       \
    }                                                                         \
    for (; r < nrows; r++) {                                                  \
      const CT *restrict s =                                                  \
          (const CT *)(base + r * row_stride);                                \
      size_t i = 0;                                                           \
      for (; i + EPV <= ncols; i += EPV)                                      \
        RSTOREI(d + i, CMP256(RLOADI(d + i), RLOADI(s + i)));                \
      for (; i < ncols; i++)                                                  \
        d[i] = SCMP(s[i], d[i]);                                             \
    }                                                                         \
  }

DEFINE_FUSED_64_AVX2(_max_fused_i64_avx2, int64_t,
  _mm256_max_epi64_emu, _RSMAX)
DEFINE_FUSED_64_AVX2(_max_fused_u64_avx2, uint64_t,
  _mm256_max_epu64_emu, _RSMAX)
DEFINE_FUSED_64_AVX2(_min_fused_i64_avx2, int64_t,
  _mm256_min_epi64_emu, _RSMIN)
DEFINE_FUSED_64_AVX2(_min_fused_u64_avx2, uint64_t,
  _mm256_min_epu64_emu, _RSMIN)

#undef DEFINE_FUSED_64_AVX2

// clang-format on

#undef RLOADI
#undef RSTOREI
#undef _RSMIN
#undef _RSMAX

#endif
