#ifndef NUMC_REDUCE_AVX512_H
#define NUMC_REDUCE_AVX512_H

#include <immintrin.h>
#include <limits.h>
#include <stdint.h>

// clang-format off

/* -- load/store helpers ---------------------------------------------- */

#define R512LOAD(p) _mm512_loadu_si512((const void *)(p))
#define R512STORE(p, v) _mm512_storeu_si512((void *)(p), v)

#define _RSMIN(a, b) ((a) < (b) ? (a) : (b))
#define _RSMAX(a, b) ((a) > (b) ? (a) : (b))

/* Some clang toolchains do not provide _mm512_reduce_* for 8/16-bit integer
 * types. Keep AVX-512 vector accumulation, then do portable horizontal folds. */
static inline int8_t _hmin_i8_avx512(__m512i v) {
  int8_t tmp[64];
  _mm512_storeu_si512((void *)tmp, v);
  int8_t r = tmp[0];
  for (size_t i = 1; i < 64; i++)
    r = _RSMIN(r, tmp[i]);
  return r;
}

static inline uint8_t _hmin_u8_avx512(__m512i v) {
  uint8_t tmp[64];
  _mm512_storeu_si512((void *)tmp, v);
  uint8_t r = tmp[0];
  for (size_t i = 1; i < 64; i++)
    r = _RSMIN(r, tmp[i]);
  return r;
}

static inline int16_t _hmin_i16_avx512(__m512i v) {
  int16_t tmp[32];
  _mm512_storeu_si512((void *)tmp, v);
  int16_t r = tmp[0];
  for (size_t i = 1; i < 32; i++)
    r = _RSMIN(r, tmp[i]);
  return r;
}

static inline uint16_t _hmin_u16_avx512(__m512i v) {
  uint16_t tmp[32];
  _mm512_storeu_si512((void *)tmp, v);
  uint16_t r = tmp[0];
  for (size_t i = 1; i < 32; i++)
    r = _RSMIN(r, tmp[i]);
  return r;
}

static inline int8_t _hmax_i8_avx512(__m512i v) {
  int8_t tmp[64];
  _mm512_storeu_si512((void *)tmp, v);
  int8_t r = tmp[0];
  for (size_t i = 1; i < 64; i++)
    r = _RSMAX(r, tmp[i]);
  return r;
}

static inline uint8_t _hmax_u8_avx512(__m512i v) {
  uint8_t tmp[64];
  _mm512_storeu_si512((void *)tmp, v);
  uint8_t r = tmp[0];
  for (size_t i = 1; i < 64; i++)
    r = _RSMAX(r, tmp[i]);
  return r;
}

static inline int16_t _hmax_i16_avx512(__m512i v) {
  int16_t tmp[32];
  _mm512_storeu_si512((void *)tmp, v);
  int16_t r = tmp[0];
  for (size_t i = 1; i < 32; i++)
    r = _RSMAX(r, tmp[i]);
  return r;
}

static inline uint16_t _hmax_u16_avx512(__m512i v) {
  uint16_t tmp[32];
  _mm512_storeu_si512((void *)tmp, v);
  uint16_t r = tmp[0];
  for (size_t i = 1; i < 32; i++)
    r = _RSMAX(r, tmp[i]);
  return r;
}

/* -- full array min/max reduction ------------------------------------ *
 *
 * 4 vector accumulators, 256 bytes/iteration for 8-bit types.
 * Single-vector cleanup loop, then scalar tail.
 * AVX-512 has native horizontal reductions for all integer widths.  */

#define DEFINE_REDUCE_FULL_AVX512(NAME, CT, EPV, INIT_VEC, CMP512,     \
                                  HREDUCE, SCMP)                       \
  static inline CT NAME(const CT *restrict a, size_t n) {              \
    __m512i a0 = INIT_VEC, a1 = INIT_VEC;                              \
    __m512i a2 = INIT_VEC, a3 = INIT_VEC;                              \
    size_t i = 0;                                                      \
    for (; i + 4 * EPV <= n; i += 4 * EPV) {                           \
      a0 = CMP512(a0, R512LOAD(a + i));                                \
      a1 = CMP512(a1, R512LOAD(a + i + EPV));                          \
      a2 = CMP512(a2, R512LOAD(a + i + 2 * EPV));                      \
      a3 = CMP512(a3, R512LOAD(a + i + 3 * EPV));                      \
    }                                                                  \
    a0 = CMP512(CMP512(a0, a1), CMP512(a2, a3));                      \
    for (; i + EPV <= n; i += EPV)                                     \
      a0 = CMP512(a0, R512LOAD(a + i));                                \
    CT result = (CT)HREDUCE(a0);                                       \
    for (; i < n; i++)                                                 \
      result = SCMP(a[i], result);                                     \
    return result;                                                     \
  }

/* min reductions — init to type MAX */
DEFINE_REDUCE_FULL_AVX512(reduce_min_i8_avx512, int8_t, 64,
  _mm512_set1_epi8(INT8_MAX),
  _mm512_min_epi8, _hmin_i8_avx512, _RSMIN)
DEFINE_REDUCE_FULL_AVX512(reduce_min_u8_avx512, uint8_t, 64,
  _mm512_set1_epi8((char)0xFF),
  _mm512_min_epu8, _hmin_u8_avx512, _RSMIN)
DEFINE_REDUCE_FULL_AVX512(reduce_min_i16_avx512, int16_t, 32,
  _mm512_set1_epi16(INT16_MAX),
  _mm512_min_epi16, _hmin_i16_avx512, _RSMIN)
DEFINE_REDUCE_FULL_AVX512(reduce_min_u16_avx512, uint16_t, 32,
  _mm512_set1_epi16((short)-1),
  _mm512_min_epu16, _hmin_u16_avx512, _RSMIN)
DEFINE_REDUCE_FULL_AVX512(reduce_min_i32_avx512, int32_t, 16,
  _mm512_set1_epi32(INT32_MAX),
  _mm512_min_epi32, _mm512_reduce_min_epi32, _RSMIN)
DEFINE_REDUCE_FULL_AVX512(reduce_min_u32_avx512, uint32_t, 16,
  _mm512_set1_epi32(-1),
  _mm512_min_epu32, _mm512_reduce_min_epu32, _RSMIN)
DEFINE_REDUCE_FULL_AVX512(reduce_min_i64_avx512, int64_t, 8,
  _mm512_set1_epi64(INT64_MAX),
  _mm512_min_epi64, _mm512_reduce_min_epi64, _RSMIN)
DEFINE_REDUCE_FULL_AVX512(reduce_min_u64_avx512, uint64_t, 8,
  _mm512_set1_epi64(-1),
  _mm512_min_epu64, _mm512_reduce_min_epu64, _RSMIN)

/* max reductions — init to type MIN */
DEFINE_REDUCE_FULL_AVX512(reduce_max_i8_avx512, int8_t, 64,
  _mm512_set1_epi8(INT8_MIN),
  _mm512_max_epi8, _hmax_i8_avx512, _RSMAX)
DEFINE_REDUCE_FULL_AVX512(reduce_max_u8_avx512, uint8_t, 64,
  _mm512_setzero_si512(),
  _mm512_max_epu8, _hmax_u8_avx512, _RSMAX)
DEFINE_REDUCE_FULL_AVX512(reduce_max_i16_avx512, int16_t, 32,
  _mm512_set1_epi16(INT16_MIN),
  _mm512_max_epi16, _hmax_i16_avx512, _RSMAX)
DEFINE_REDUCE_FULL_AVX512(reduce_max_u16_avx512, uint16_t, 32,
  _mm512_setzero_si512(),
  _mm512_max_epu16, _hmax_u16_avx512, _RSMAX)
DEFINE_REDUCE_FULL_AVX512(reduce_max_i32_avx512, int32_t, 16,
  _mm512_set1_epi32(INT32_MIN),
  _mm512_max_epi32, _mm512_reduce_max_epi32, _RSMAX)
DEFINE_REDUCE_FULL_AVX512(reduce_max_u32_avx512, uint32_t, 16,
  _mm512_setzero_si512(),
  _mm512_max_epu32, _mm512_reduce_max_epu32, _RSMAX)
DEFINE_REDUCE_FULL_AVX512(reduce_max_i64_avx512, int64_t, 8,
  _mm512_set1_epi64(INT64_MIN),
  _mm512_max_epi64, _mm512_reduce_max_epi64, _RSMAX)
DEFINE_REDUCE_FULL_AVX512(reduce_max_u64_avx512, uint64_t, 8,
  _mm512_setzero_si512(),
  _mm512_max_epu64, _mm512_reduce_max_epu64, _RSMAX)

#undef DEFINE_REDUCE_FULL_AVX512

/* -- fused row-reduce (axis-1 reduction) ----------------------------- *
 *
 * Processes 4 rows at a time, vectorizing the inner column loop.
 * For int8: 64 columns per SIMD iteration (vs 1 scalar).            */

#define DEFINE_FUSED_REDUCE_AVX512(NAME, CT, EPV, CMP512, SCMP)            \
  static inline void NAME(const char *restrict base, intptr_t row_stride,  \
                           size_t nrows, char *restrict dst,               \
                           size_t ncols) {                                 \
    CT *restrict d = (CT *)dst;                                            \
    size_t r = 0;                                                          \
    for (; r + 4 <= nrows; r += 4) {                                       \
      const CT *restrict s0 = (const CT *)(base + r * row_stride);         \
      const CT *restrict s1 =                                              \
          (const CT *)(base + (r + 1) * row_stride);                       \
      const CT *restrict s2 =                                              \
          (const CT *)(base + (r + 2) * row_stride);                       \
      const CT *restrict s3 =                                              \
          (const CT *)(base + (r + 3) * row_stride);                       \
      size_t i = 0;                                                        \
      for (; i + EPV <= ncols; i += EPV) {                                 \
        __m512i dv  = R512LOAD(d + i);                                     \
        __m512i v01 = CMP512(R512LOAD(s0 + i), R512LOAD(s1 + i));         \
        __m512i v23 = CMP512(R512LOAD(s2 + i), R512LOAD(s3 + i));         \
        R512STORE(d + i, CMP512(dv, CMP512(v01, v23)));                   \
      }                                                                    \
      for (; i < ncols; i++) {                                             \
        CT v = SCMP(s0[i], s1[i]);                                         \
        v = SCMP(v, s2[i]);                                                \
        v = SCMP(v, s3[i]);                                                \
        d[i] = SCMP(v, d[i]);                                             \
      }                                                                    \
    }                                                                      \
    for (; r < nrows; r++) {                                               \
      const CT *restrict s =                                               \
          (const CT *)(base + r * row_stride);                             \
      size_t i = 0;                                                        \
      for (; i + EPV <= ncols; i += EPV)                                   \
        R512STORE(d + i, CMP512(R512LOAD(d + i), R512LOAD(s + i)));       \
      for (; i < ncols; i++)                                               \
        d[i] = SCMP(s[i], d[i]);                                          \
    }                                                                      \
  }

/* min fused */
DEFINE_FUSED_REDUCE_AVX512(_min_fused_i8_avx512, int8_t, 64,
  _mm512_min_epi8, _RSMIN)
DEFINE_FUSED_REDUCE_AVX512(_min_fused_u8_avx512, uint8_t, 64,
  _mm512_min_epu8, _RSMIN)
DEFINE_FUSED_REDUCE_AVX512(_min_fused_i16_avx512, int16_t, 32,
  _mm512_min_epi16, _RSMIN)
DEFINE_FUSED_REDUCE_AVX512(_min_fused_u16_avx512, uint16_t, 32,
  _mm512_min_epu16, _RSMIN)
DEFINE_FUSED_REDUCE_AVX512(_min_fused_i32_avx512, int32_t, 16,
  _mm512_min_epi32, _RSMIN)
DEFINE_FUSED_REDUCE_AVX512(_min_fused_u32_avx512, uint32_t, 16,
  _mm512_min_epu32, _RSMIN)
DEFINE_FUSED_REDUCE_AVX512(_min_fused_i64_avx512, int64_t, 8,
  _mm512_min_epi64, _RSMIN)
DEFINE_FUSED_REDUCE_AVX512(_min_fused_u64_avx512, uint64_t, 8,
  _mm512_min_epu64, _RSMIN)

/* max fused */
DEFINE_FUSED_REDUCE_AVX512(_max_fused_i8_avx512, int8_t, 64,
  _mm512_max_epi8, _RSMAX)
DEFINE_FUSED_REDUCE_AVX512(_max_fused_u8_avx512, uint8_t, 64,
  _mm512_max_epu8, _RSMAX)
DEFINE_FUSED_REDUCE_AVX512(_max_fused_i16_avx512, int16_t, 32,
  _mm512_max_epi16, _RSMAX)
DEFINE_FUSED_REDUCE_AVX512(_max_fused_u16_avx512, uint16_t, 32,
  _mm512_max_epu16, _RSMAX)
DEFINE_FUSED_REDUCE_AVX512(_max_fused_i32_avx512, int32_t, 16,
  _mm512_max_epi32, _RSMAX)
DEFINE_FUSED_REDUCE_AVX512(_max_fused_u32_avx512, uint32_t, 16,
  _mm512_max_epu32, _RSMAX)
DEFINE_FUSED_REDUCE_AVX512(_max_fused_i64_avx512, int64_t, 8,
  _mm512_max_epi64, _RSMAX)
DEFINE_FUSED_REDUCE_AVX512(_max_fused_u64_avx512, uint64_t, 8,
  _mm512_max_epu64, _RSMAX)

#undef DEFINE_FUSED_REDUCE_AVX512

// clang-format on

#undef R512LOAD
#undef R512STORE
#undef _RSMIN
#undef _RSMAX

#endif
