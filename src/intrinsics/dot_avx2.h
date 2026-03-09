#ifndef NUMC_DOT_AVX2_H
#define NUMC_DOT_AVX2_H

#include <immintrin.h>
#include <stdint.h>

// clang-format off

/* ── helpers ─────────────────────────────────────────────────────────── */

static inline __m256i _mullo_epi64_avx2(__m256i a, __m256i b) {
  __m256i lo    = _mm256_mul_epu32(a, b);
  __m256i a_hi  = _mm256_srli_epi64(a, 32);
  __m256i b_hi  = _mm256_srli_epi64(b, 32);
  __m256i cross = _mm256_add_epi64(_mm256_mul_epu32(a_hi, b),
                                   _mm256_mul_epu32(a, b_hi));
  return _mm256_add_epi64(lo, _mm256_slli_epi64(cross, 32));
}

static inline float _hsum_ps_avx2(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 s  = _mm_add_ps(lo, hi);
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_movehdup_ps(s));
  return _mm_cvtss_f32(s);
}

static inline double _hsum_pd_avx2(__m256d v) {
  __m128d lo = _mm256_castpd256_pd128(v);
  __m128d hi = _mm256_extractf128_pd(v, 1);
  __m128d s  = _mm_add_pd(lo, hi);
  s = _mm_add_pd(s, _mm_unpackhi_pd(s, s));
  return _mm_cvtsd_f64(s);
}

static inline int32_t _hsum_epi32_avx2(__m256i v) {
  __m128i lo = _mm256_castsi256_si128(v);
  __m128i hi = _mm256_extracti128_si256(v, 1);
  __m128i s  = _mm_add_epi32(lo, hi);
  s = _mm_add_epi32(s, _mm_srli_si128(s, 8));
  s = _mm_add_epi32(s, _mm_srli_si128(s, 4));
  return _mm_cvtsi128_si32(s);
}

static inline int64_t _hsum_epi64_avx2(__m256i v) {
  __m128i lo = _mm256_castsi256_si128(v);
  __m128i hi = _mm256_extracti128_si256(v, 1);
  __m128i s  = _mm_add_epi64(lo, hi);
  s = _mm_add_epi64(s, _mm_unpackhi_epi64(s, s));
  return _mm_cvtsi128_si64(s);
}

/* tree-reduce 8 accumulators into acc0 */
#define REDUCE_ACC8(add_fn, a0, a1, a2, a3, a4, a5, a6, a7) \
  do {                                                        \
    a0 = add_fn(a0, a1);  a2 = add_fn(a2, a3);               \
    a4 = add_fn(a4, a5);  a6 = add_fn(a6, a7);               \
    a0 = add_fn(a0, a2);  a4 = add_fn(a4, a6);               \
    a0 = add_fn(a0, a4);                                      \
  } while (0)

#define PREFETCH2(a, b, off)                               \
  _mm_prefetch((const char *)((a) + (off)), _MM_HINT_T0);  \
  _mm_prefetch((const char *)((b) + (off)), _MM_HINT_T0)

#define LOADI(p) _mm256_loadu_si256((const __m256i *)(p))

/* widen 32 bytes → two madd_epi16 results (8×i32 each), add to acc */
#define WIDEN_MADD_I8(acc, pa, pb)                                          \
  do {                                                                      \
    __m256i _va = LOADI(pa), _vb = LOADI(pb);                               \
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(                          \
        _mm256_cvtepi8_epi16(_mm256_castsi256_si128(_va)),                  \
        _mm256_cvtepi8_epi16(_mm256_castsi256_si128(_vb))));                \
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(                          \
        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(_va, 1)),             \
        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(_vb, 1))));           \
  } while (0)

#define WIDEN_MADD_U8(acc, pa, pb)                                          \
  do {                                                                      \
    __m256i _va = LOADI(pa), _vb = LOADI(pb);                               \
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(                          \
        _mm256_cvtepu8_epi16(_mm256_castsi256_si128(_va)),                  \
        _mm256_cvtepu8_epi16(_mm256_castsi256_si128(_vb))));                \
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(                          \
        _mm256_cvtepu8_epi16(_mm256_extracti128_si256(_va, 1)),             \
        _mm256_cvtepu8_epi16(_mm256_extracti128_si256(_vb, 1))));           \
  } while (0)

/* widen 16 u16 → two mullo_epi32 results (8×i32 each), add to acc_lo/acc_hi */
#define WIDEN_MUL_U16(acc_lo, acc_hi, pa, pb)                               \
  do {                                                                      \
    __m256i _va = LOADI(pa), _vb = LOADI(pb);                               \
    acc_lo = _mm256_add_epi32(acc_lo, _mm256_mullo_epi32(                   \
        _mm256_cvtepu16_epi32(_mm256_castsi256_si128(_va)),                 \
        _mm256_cvtepu16_epi32(_mm256_castsi256_si128(_vb))));               \
    acc_hi = _mm256_add_epi32(acc_hi, _mm256_mullo_epi32(                   \
        _mm256_cvtepu16_epi32(_mm256_extracti128_si256(_va, 1)),            \
        _mm256_cvtepu16_epi32(_mm256_extracti128_si256(_vb, 1))));          \
  } while (0)

/* ── float dot products ──────────────────────────────────────────────── */

static inline void dot_f32u_avx2(const float *a, const float *b, size_t n,
                                 float *dest) {
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps(),
         acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps(),
         acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps(),
         acc6 = _mm256_setzero_ps(), acc7 = _mm256_setzero_ps();

  size_t i = 0;
  for (; i + 64 <= n; i += 64) {
    PREFETCH2(a, b, i + 256);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i),    _mm256_loadu_ps(b+i),    acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+8),  _mm256_loadu_ps(b+i+8),  acc1);
    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+16), _mm256_loadu_ps(b+i+16), acc2);
    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+24), _mm256_loadu_ps(b+i+24), acc3);
    acc4 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+32), _mm256_loadu_ps(b+i+32), acc4);
    acc5 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+40), _mm256_loadu_ps(b+i+40), acc5);
    acc6 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+48), _mm256_loadu_ps(b+i+48), acc6);
    acc7 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+56), _mm256_loadu_ps(b+i+56), acc7);
  }
  REDUCE_ACC8(_mm256_add_ps, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);

  float result = _hsum_ps_avx2(acc0);
  float tail = 0.0f;
  for (; i < n; i++) tail += a[i] * b[i];
  *dest = result + tail;
}

static inline void dot_f64u_avx2(const double *a, const double *b, size_t n,
                                 double *dest) {
  __m256d acc0 = _mm256_setzero_pd(), acc1 = _mm256_setzero_pd(),
          acc2 = _mm256_setzero_pd(), acc3 = _mm256_setzero_pd(),
          acc4 = _mm256_setzero_pd(), acc5 = _mm256_setzero_pd(),
          acc6 = _mm256_setzero_pd(), acc7 = _mm256_setzero_pd();

  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    PREFETCH2(a, b, i + 256);
    acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(a+i),    _mm256_loadu_pd(b+i),    acc0);
    acc1 = _mm256_fmadd_pd(_mm256_loadu_pd(a+i+4),  _mm256_loadu_pd(b+i+4),  acc1);
    acc2 = _mm256_fmadd_pd(_mm256_loadu_pd(a+i+8),  _mm256_loadu_pd(b+i+8),  acc2);
    acc3 = _mm256_fmadd_pd(_mm256_loadu_pd(a+i+12), _mm256_loadu_pd(b+i+12), acc3);
    acc4 = _mm256_fmadd_pd(_mm256_loadu_pd(a+i+16), _mm256_loadu_pd(b+i+16), acc4);
    acc5 = _mm256_fmadd_pd(_mm256_loadu_pd(a+i+20), _mm256_loadu_pd(b+i+20), acc5);
    acc6 = _mm256_fmadd_pd(_mm256_loadu_pd(a+i+24), _mm256_loadu_pd(b+i+24), acc6);
    acc7 = _mm256_fmadd_pd(_mm256_loadu_pd(a+i+28), _mm256_loadu_pd(b+i+28), acc7);
  }
  REDUCE_ACC8(_mm256_add_pd, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);

  double result = _hsum_pd_avx2(acc0);
  double tail = 0.0;
  for (; i < n; i++) tail += a[i] * b[i];
  *dest = result + tail;
}

/* ── 32-bit integer dot products ─────────────────────────────────────── */

static inline void dot_i32_avx2(const int32_t *a, const int32_t *b, size_t n,
                                int32_t *dest) {
  __m256i acc0 = _mm256_setzero_si256(), acc1 = _mm256_setzero_si256(),
          acc2 = _mm256_setzero_si256(), acc3 = _mm256_setzero_si256(),
          acc4 = _mm256_setzero_si256(), acc5 = _mm256_setzero_si256(),
          acc6 = _mm256_setzero_si256(), acc7 = _mm256_setzero_si256();

  size_t i = 0;
  for (; i + 64 <= n; i += 64) {
    PREFETCH2(a, b, i + 256);
    acc0 = _mm256_add_epi32(_mm256_mullo_epi32(LOADI(a+i),    LOADI(b+i)),    acc0);
    acc1 = _mm256_add_epi32(_mm256_mullo_epi32(LOADI(a+i+8),  LOADI(b+i+8)),  acc1);
    acc2 = _mm256_add_epi32(_mm256_mullo_epi32(LOADI(a+i+16), LOADI(b+i+16)), acc2);
    acc3 = _mm256_add_epi32(_mm256_mullo_epi32(LOADI(a+i+24), LOADI(b+i+24)), acc3);
    acc4 = _mm256_add_epi32(_mm256_mullo_epi32(LOADI(a+i+32), LOADI(b+i+32)), acc4);
    acc5 = _mm256_add_epi32(_mm256_mullo_epi32(LOADI(a+i+40), LOADI(b+i+40)), acc5);
    acc6 = _mm256_add_epi32(_mm256_mullo_epi32(LOADI(a+i+48), LOADI(b+i+48)), acc6);
    acc7 = _mm256_add_epi32(_mm256_mullo_epi32(LOADI(a+i+56), LOADI(b+i+56)), acc7);
  }
  REDUCE_ACC8(_mm256_add_epi32, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);

  int32_t result = _hsum_epi32_avx2(acc0);
  int32_t tail = 0;
  for (; i < n; i++) tail += a[i] * b[i];
  *dest = result + tail;
}

static inline void dot_u32_avx2(const uint32_t *a, const uint32_t *b, size_t n,
                                uint32_t *dest) {
  __m256i acc0 = _mm256_setzero_si256(), acc1 = _mm256_setzero_si256(),
          acc2 = _mm256_setzero_si256(), acc3 = _mm256_setzero_si256(),
          acc4 = _mm256_setzero_si256(), acc5 = _mm256_setzero_si256(),
          acc6 = _mm256_setzero_si256(), acc7 = _mm256_setzero_si256();

  size_t i = 0;
  for (; i + 64 <= n; i += 64) {
    PREFETCH2(a, b, i + 256);
    acc0 = _mm256_add_epi32(_mm256_mullo_epi32(LOADI(a+i),    LOADI(b+i)),    acc0);
    acc1 = _mm256_add_epi32(_mm256_mullo_epi32(LOADI(a+i+8),  LOADI(b+i+8)),  acc1);
    acc2 = _mm256_add_epi32(_mm256_mullo_epi32(LOADI(a+i+16), LOADI(b+i+16)), acc2);
    acc3 = _mm256_add_epi32(_mm256_mullo_epi32(LOADI(a+i+24), LOADI(b+i+24)), acc3);
    acc4 = _mm256_add_epi32(_mm256_mullo_epi32(LOADI(a+i+32), LOADI(b+i+32)), acc4);
    acc5 = _mm256_add_epi32(_mm256_mullo_epi32(LOADI(a+i+40), LOADI(b+i+40)), acc5);
    acc6 = _mm256_add_epi32(_mm256_mullo_epi32(LOADI(a+i+48), LOADI(b+i+48)), acc6);
    acc7 = _mm256_add_epi32(_mm256_mullo_epi32(LOADI(a+i+56), LOADI(b+i+56)), acc7);
  }
  REDUCE_ACC8(_mm256_add_epi32, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);

  uint32_t result = (uint32_t)_hsum_epi32_avx2(acc0);
  uint32_t tail = 0;
  for (; i < n; i++) tail += a[i] * b[i];
  *dest = result + tail;
}

/* ── 64-bit integer dot products ─────────────────────────────────────── */

static inline void dot_i64_avx2(const int64_t *a, const int64_t *b, size_t n,
                                int64_t *dest) {
  __m256i acc0 = _mm256_setzero_si256(), acc1 = _mm256_setzero_si256(),
          acc2 = _mm256_setzero_si256(), acc3 = _mm256_setzero_si256(),
          acc4 = _mm256_setzero_si256(), acc5 = _mm256_setzero_si256(),
          acc6 = _mm256_setzero_si256(), acc7 = _mm256_setzero_si256();

  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    PREFETCH2(a, b, i + 256);
    acc0 = _mm256_add_epi64(_mullo_epi64_avx2(LOADI(a+i),    LOADI(b+i)),    acc0);
    acc1 = _mm256_add_epi64(_mullo_epi64_avx2(LOADI(a+i+4),  LOADI(b+i+4)),  acc1);
    acc2 = _mm256_add_epi64(_mullo_epi64_avx2(LOADI(a+i+8),  LOADI(b+i+8)),  acc2);
    acc3 = _mm256_add_epi64(_mullo_epi64_avx2(LOADI(a+i+12), LOADI(b+i+12)), acc3);
    acc4 = _mm256_add_epi64(_mullo_epi64_avx2(LOADI(a+i+16), LOADI(b+i+16)), acc4);
    acc5 = _mm256_add_epi64(_mullo_epi64_avx2(LOADI(a+i+20), LOADI(b+i+20)), acc5);
    acc6 = _mm256_add_epi64(_mullo_epi64_avx2(LOADI(a+i+24), LOADI(b+i+24)), acc6);
    acc7 = _mm256_add_epi64(_mullo_epi64_avx2(LOADI(a+i+28), LOADI(b+i+28)), acc7);
  }
  REDUCE_ACC8(_mm256_add_epi64, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);

  int64_t result = _hsum_epi64_avx2(acc0);
  int64_t tail = 0;
  for (; i < n; i++) tail += a[i] * b[i];
  *dest = result + tail;
}

static inline void dot_u64_avx2(const uint64_t *a, const uint64_t *b, size_t n,
                                uint64_t *dest) {
  __m256i acc0 = _mm256_setzero_si256(), acc1 = _mm256_setzero_si256(),
          acc2 = _mm256_setzero_si256(), acc3 = _mm256_setzero_si256(),
          acc4 = _mm256_setzero_si256(), acc5 = _mm256_setzero_si256(),
          acc6 = _mm256_setzero_si256(), acc7 = _mm256_setzero_si256();

  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    PREFETCH2(a, b, i + 256);
    acc0 = _mm256_add_epi64(_mullo_epi64_avx2(LOADI(a+i),    LOADI(b+i)),    acc0);
    acc1 = _mm256_add_epi64(_mullo_epi64_avx2(LOADI(a+i+4),  LOADI(b+i+4)),  acc1);
    acc2 = _mm256_add_epi64(_mullo_epi64_avx2(LOADI(a+i+8),  LOADI(b+i+8)),  acc2);
    acc3 = _mm256_add_epi64(_mullo_epi64_avx2(LOADI(a+i+12), LOADI(b+i+12)), acc3);
    acc4 = _mm256_add_epi64(_mullo_epi64_avx2(LOADI(a+i+16), LOADI(b+i+16)), acc4);
    acc5 = _mm256_add_epi64(_mullo_epi64_avx2(LOADI(a+i+20), LOADI(b+i+20)), acc5);
    acc6 = _mm256_add_epi64(_mullo_epi64_avx2(LOADI(a+i+24), LOADI(b+i+24)), acc6);
    acc7 = _mm256_add_epi64(_mullo_epi64_avx2(LOADI(a+i+28), LOADI(b+i+28)), acc7);
  }
  REDUCE_ACC8(_mm256_add_epi64, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);

  uint64_t result = (uint64_t)_hsum_epi64_avx2(acc0);
  uint64_t tail = 0;
  for (; i < n; i++) tail += a[i] * b[i];
  *dest = result + tail;
}

/* ── 8-bit dot (widen to i16, madd_epi16 → i32 accumulators) ────────── */

static inline void dot_i8_avx2(const int8_t *a, const int8_t *b, size_t n,
                                int8_t *dest) {
  __m256i acc0 = _mm256_setzero_si256(), acc1 = _mm256_setzero_si256(),
          acc2 = _mm256_setzero_si256(), acc3 = _mm256_setzero_si256(),
          acc4 = _mm256_setzero_si256(), acc5 = _mm256_setzero_si256(),
          acc6 = _mm256_setzero_si256(), acc7 = _mm256_setzero_si256();

  size_t i = 0;
  for (; i + 256 <= n; i += 256) {
    PREFETCH2(a, b, i + 512);
    WIDEN_MADD_I8(acc0, a+i,     b+i);
    WIDEN_MADD_I8(acc1, a+i+32,  b+i+32);
    WIDEN_MADD_I8(acc2, a+i+64,  b+i+64);
    WIDEN_MADD_I8(acc3, a+i+96,  b+i+96);
    WIDEN_MADD_I8(acc4, a+i+128, b+i+128);
    WIDEN_MADD_I8(acc5, a+i+160, b+i+160);
    WIDEN_MADD_I8(acc6, a+i+192, b+i+192);
    WIDEN_MADD_I8(acc7, a+i+224, b+i+224);
  }
  REDUCE_ACC8(_mm256_add_epi32, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);

  int8_t result = (int8_t)_hsum_epi32_avx2(acc0);
  int8_t tail = 0;
  for (; i < n; i++) tail += (int8_t)(a[i] * b[i]);
  *dest = (int8_t)(result + tail);
}

static inline void dot_u8_avx2(const uint8_t *a, const uint8_t *b, size_t n,
                                uint8_t *dest) {
  __m256i acc0 = _mm256_setzero_si256(), acc1 = _mm256_setzero_si256(),
          acc2 = _mm256_setzero_si256(), acc3 = _mm256_setzero_si256(),
          acc4 = _mm256_setzero_si256(), acc5 = _mm256_setzero_si256(),
          acc6 = _mm256_setzero_si256(), acc7 = _mm256_setzero_si256();

  size_t i = 0;
  for (; i + 256 <= n; i += 256) {
    PREFETCH2(a, b, i + 512);
    WIDEN_MADD_U8(acc0, a+i,     b+i);
    WIDEN_MADD_U8(acc1, a+i+32,  b+i+32);
    WIDEN_MADD_U8(acc2, a+i+64,  b+i+64);
    WIDEN_MADD_U8(acc3, a+i+96,  b+i+96);
    WIDEN_MADD_U8(acc4, a+i+128, b+i+128);
    WIDEN_MADD_U8(acc5, a+i+160, b+i+160);
    WIDEN_MADD_U8(acc6, a+i+192, b+i+192);
    WIDEN_MADD_U8(acc7, a+i+224, b+i+224);
  }
  REDUCE_ACC8(_mm256_add_epi32, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);

  uint8_t result = (uint8_t)_hsum_epi32_avx2(acc0);
  uint8_t tail = 0;
  for (; i < n; i++) tail += (uint8_t)(a[i] * b[i]);
  *dest = (uint8_t)(result + tail);
}

/* ── 16-bit dot (madd_epi16 → i32 accumulators) ─────────────────────── */

static inline void dot_i16_avx2(const int16_t *a, const int16_t *b, size_t n,
                                int16_t *dest) {
  __m256i acc0 = _mm256_setzero_si256(), acc1 = _mm256_setzero_si256(),
          acc2 = _mm256_setzero_si256(), acc3 = _mm256_setzero_si256(),
          acc4 = _mm256_setzero_si256(), acc5 = _mm256_setzero_si256(),
          acc6 = _mm256_setzero_si256(), acc7 = _mm256_setzero_si256();

  size_t i = 0;
  for (; i + 128 <= n; i += 128) {
    PREFETCH2(a, b, i + 256);
    acc0 = _mm256_add_epi32(_mm256_madd_epi16(LOADI(a+i),     LOADI(b+i)),     acc0);
    acc1 = _mm256_add_epi32(_mm256_madd_epi16(LOADI(a+i+16),  LOADI(b+i+16)),  acc1);
    acc2 = _mm256_add_epi32(_mm256_madd_epi16(LOADI(a+i+32),  LOADI(b+i+32)),  acc2);
    acc3 = _mm256_add_epi32(_mm256_madd_epi16(LOADI(a+i+48),  LOADI(b+i+48)),  acc3);
    acc4 = _mm256_add_epi32(_mm256_madd_epi16(LOADI(a+i+64),  LOADI(b+i+64)),  acc4);
    acc5 = _mm256_add_epi32(_mm256_madd_epi16(LOADI(a+i+80),  LOADI(b+i+80)),  acc5);
    acc6 = _mm256_add_epi32(_mm256_madd_epi16(LOADI(a+i+96),  LOADI(b+i+96)),  acc6);
    acc7 = _mm256_add_epi32(_mm256_madd_epi16(LOADI(a+i+112), LOADI(b+i+112)), acc7);
  }
  REDUCE_ACC8(_mm256_add_epi32, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);

  int16_t result = (int16_t)_hsum_epi32_avx2(acc0);
  int16_t tail = 0;
  for (; i < n; i++) tail += (int16_t)(a[i] * b[i]);
  *dest = (int16_t)(result + tail);
}

static inline void dot_u16_avx2(const uint16_t *a, const uint16_t *b, size_t n,
                                uint16_t *dest) {
  __m256i acc0 = _mm256_setzero_si256(), acc1 = _mm256_setzero_si256(),
          acc2 = _mm256_setzero_si256(), acc3 = _mm256_setzero_si256(),
          acc4 = _mm256_setzero_si256(), acc5 = _mm256_setzero_si256(),
          acc6 = _mm256_setzero_si256(), acc7 = _mm256_setzero_si256();

  size_t i = 0;
  for (; i + 64 <= n; i += 64) {
    PREFETCH2(a, b, i + 256);
    WIDEN_MUL_U16(acc0, acc1, a+i,    b+i);
    WIDEN_MUL_U16(acc2, acc3, a+i+16, b+i+16);
    WIDEN_MUL_U16(acc4, acc5, a+i+32, b+i+32);
    WIDEN_MUL_U16(acc6, acc7, a+i+48, b+i+48);
  }
  REDUCE_ACC8(_mm256_add_epi32, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);

  uint16_t result = (uint16_t)_hsum_epi32_avx2(acc0);
  uint16_t tail = 0;
  for (; i < n; i++) tail += (uint16_t)(a[i] * b[i]);
  *dest = (uint16_t)(result + tail);
}

// clang-format on

#undef REDUCE_ACC8
#undef PREFETCH2
#undef LOADI
#undef WIDEN_MADD_I8
#undef WIDEN_MADD_U8
#undef WIDEN_MUL_U16

#endif
