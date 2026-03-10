#ifndef NUMC_DOT_AVX512_H
#define NUMC_DOT_AVX512_H

#include <immintrin.h>
#include <stdint.h>

// clang-format off

/* ── helpers ─────────────────────────────────────────────────────────── */

/* tree-reduce 8 accumulators into acc0 */
#define REDUCE_ACC8_512(add_fn, a0, a1, a2, a3, a4, a5, a6, a7) \
  do {                                                            \
    a0 = add_fn(a0, a1);  a2 = add_fn(a2, a3);                   \
    a4 = add_fn(a4, a5);  a6 = add_fn(a6, a7);                   \
    a0 = add_fn(a0, a2);  a4 = add_fn(a4, a6);                   \
    a0 = add_fn(a0, a4);                                          \
  } while (0)

#define PREFETCH2_512(a, b, off)                               \
  _mm_prefetch((const char *)((a) + (off)), _MM_HINT_T0);      \
  _mm_prefetch((const char *)((b) + (off)), _MM_HINT_T0)

#define LOADI512(p) _mm512_loadu_si512((const void *)(p))

/* widen 64 bytes → two madd_epi16 results (16×i32 each), add to acc */
#define WIDEN_MADD_I8_512(acc, pa, pb)                                          \
  do {                                                                          \
    __m512i _va = LOADI512(pa), _vb = LOADI512(pb);                             \
    acc = _mm512_add_epi32(acc, _mm512_madd_epi16(                              \
        _mm512_cvtepi8_epi16(_mm512_castsi512_si256(_va)),                      \
        _mm512_cvtepi8_epi16(_mm512_castsi512_si256(_vb))));                    \
    acc = _mm512_add_epi32(acc, _mm512_madd_epi16(                              \
        _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_va, 1)),               \
        _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_vb, 1))));             \
  } while (0)

#define WIDEN_MADD_U8_512(acc, pa, pb)                                          \
  do {                                                                          \
    __m512i _va = LOADI512(pa), _vb = LOADI512(pb);                             \
    acc = _mm512_add_epi32(acc, _mm512_madd_epi16(                              \
        _mm512_cvtepu8_epi16(_mm512_castsi512_si256(_va)),                      \
        _mm512_cvtepu8_epi16(_mm512_castsi512_si256(_vb))));                    \
    acc = _mm512_add_epi32(acc, _mm512_madd_epi16(                              \
        _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(_va, 1)),               \
        _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(_vb, 1))));             \
  } while (0)

/* widen 32 u16 → two mullo_epi32 results (16×i32 each), add to acc_lo/acc_hi */
#define WIDEN_MUL_U16_512(acc_lo, acc_hi, pa, pb)                               \
  do {                                                                          \
    __m512i _va = LOADI512(pa), _vb = LOADI512(pb);                             \
    acc_lo = _mm512_add_epi32(acc_lo, _mm512_mullo_epi32(                       \
        _mm512_cvtepu16_epi32(_mm512_castsi512_si256(_va)),                     \
        _mm512_cvtepu16_epi32(_mm512_castsi512_si256(_vb))));                   \
    acc_hi = _mm512_add_epi32(acc_hi, _mm512_mullo_epi32(                       \
        _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(_va, 1)),              \
        _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(_vb, 1))));            \
  } while (0)

/* ── float dot products ──────────────────────────────────────────────── */

static inline void dot_f32u_avx512(const float *a, const float *b, size_t n,
                                   float *dest) {
  __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps(),
         acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps(),
         acc4 = _mm512_setzero_ps(), acc5 = _mm512_setzero_ps(),
         acc6 = _mm512_setzero_ps(), acc7 = _mm512_setzero_ps();

  size_t i = 0;
  for (; i + 128 <= n; i += 128) {
    PREFETCH2_512(a, b, i + 512);
    acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i),     _mm512_loadu_ps(b+i),     acc0);
    acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+16),  _mm512_loadu_ps(b+i+16),  acc1);
    acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+32),  _mm512_loadu_ps(b+i+32),  acc2);
    acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+48),  _mm512_loadu_ps(b+i+48),  acc3);
    acc4 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+64),  _mm512_loadu_ps(b+i+64),  acc4);
    acc5 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+80),  _mm512_loadu_ps(b+i+80),  acc5);
    acc6 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+96),  _mm512_loadu_ps(b+i+96),  acc6);
    acc7 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+112), _mm512_loadu_ps(b+i+112), acc7);
  }
  REDUCE_ACC8_512(_mm512_add_ps, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);

  float result = _mm512_reduce_add_ps(acc0);
  float tail = 0.0f;
  for (; i < n; i++) tail += a[i] * b[i];
  *dest = result + tail;
}

static inline void dot_f64u_avx512(const double *a, const double *b, size_t n,
                                   double *dest) {
  __m512d acc0 = _mm512_setzero_pd(), acc1 = _mm512_setzero_pd(),
          acc2 = _mm512_setzero_pd(), acc3 = _mm512_setzero_pd(),
          acc4 = _mm512_setzero_pd(), acc5 = _mm512_setzero_pd(),
          acc6 = _mm512_setzero_pd(), acc7 = _mm512_setzero_pd();

  size_t i = 0;
  for (; i + 64 <= n; i += 64) {
    PREFETCH2_512(a, b, i + 512);
    acc0 = _mm512_fmadd_pd(_mm512_loadu_pd(a+i),    _mm512_loadu_pd(b+i),    acc0);
    acc1 = _mm512_fmadd_pd(_mm512_loadu_pd(a+i+8),  _mm512_loadu_pd(b+i+8),  acc1);
    acc2 = _mm512_fmadd_pd(_mm512_loadu_pd(a+i+16), _mm512_loadu_pd(b+i+16), acc2);
    acc3 = _mm512_fmadd_pd(_mm512_loadu_pd(a+i+24), _mm512_loadu_pd(b+i+24), acc3);
    acc4 = _mm512_fmadd_pd(_mm512_loadu_pd(a+i+32), _mm512_loadu_pd(b+i+32), acc4);
    acc5 = _mm512_fmadd_pd(_mm512_loadu_pd(a+i+40), _mm512_loadu_pd(b+i+40), acc5);
    acc6 = _mm512_fmadd_pd(_mm512_loadu_pd(a+i+48), _mm512_loadu_pd(b+i+48), acc6);
    acc7 = _mm512_fmadd_pd(_mm512_loadu_pd(a+i+56), _mm512_loadu_pd(b+i+56), acc7);
  }
  REDUCE_ACC8_512(_mm512_add_pd, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);

  double result = _mm512_reduce_add_pd(acc0);
  double tail = 0.0;
  for (; i < n; i++) tail += a[i] * b[i];
  *dest = result + tail;
}

/* ── 32-bit integer dot products ─────────────────────────────────────── */

static inline void dot_i32_avx512(const int32_t *a, const int32_t *b, size_t n,
                                  int32_t *dest) {
  __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512(),
          acc2 = _mm512_setzero_si512(), acc3 = _mm512_setzero_si512(),
          acc4 = _mm512_setzero_si512(), acc5 = _mm512_setzero_si512(),
          acc6 = _mm512_setzero_si512(), acc7 = _mm512_setzero_si512();

  size_t i = 0;
  for (; i + 128 <= n; i += 128) {
    PREFETCH2_512(a, b, i + 512);
    acc0 = _mm512_add_epi32(_mm512_mullo_epi32(LOADI512(a+i),     LOADI512(b+i)),     acc0);
    acc1 = _mm512_add_epi32(_mm512_mullo_epi32(LOADI512(a+i+16),  LOADI512(b+i+16)),  acc1);
    acc2 = _mm512_add_epi32(_mm512_mullo_epi32(LOADI512(a+i+32),  LOADI512(b+i+32)),  acc2);
    acc3 = _mm512_add_epi32(_mm512_mullo_epi32(LOADI512(a+i+48),  LOADI512(b+i+48)),  acc3);
    acc4 = _mm512_add_epi32(_mm512_mullo_epi32(LOADI512(a+i+64),  LOADI512(b+i+64)),  acc4);
    acc5 = _mm512_add_epi32(_mm512_mullo_epi32(LOADI512(a+i+80),  LOADI512(b+i+80)),  acc5);
    acc6 = _mm512_add_epi32(_mm512_mullo_epi32(LOADI512(a+i+96),  LOADI512(b+i+96)),  acc6);
    acc7 = _mm512_add_epi32(_mm512_mullo_epi32(LOADI512(a+i+112), LOADI512(b+i+112)), acc7);
  }
  REDUCE_ACC8_512(_mm512_add_epi32, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);

  int32_t result = _mm512_reduce_add_epi32(acc0);
  int32_t tail = 0;
  for (; i < n; i++) tail += a[i] * b[i];
  *dest = result + tail;
}

static inline void dot_u32_avx512(const uint32_t *a, const uint32_t *b, size_t n,
                                  uint32_t *dest) {
  __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512(),
          acc2 = _mm512_setzero_si512(), acc3 = _mm512_setzero_si512(),
          acc4 = _mm512_setzero_si512(), acc5 = _mm512_setzero_si512(),
          acc6 = _mm512_setzero_si512(), acc7 = _mm512_setzero_si512();

  size_t i = 0;
  for (; i + 128 <= n; i += 128) {
    PREFETCH2_512(a, b, i + 512);
    acc0 = _mm512_add_epi32(_mm512_mullo_epi32(LOADI512(a+i),     LOADI512(b+i)),     acc0);
    acc1 = _mm512_add_epi32(_mm512_mullo_epi32(LOADI512(a+i+16),  LOADI512(b+i+16)),  acc1);
    acc2 = _mm512_add_epi32(_mm512_mullo_epi32(LOADI512(a+i+32),  LOADI512(b+i+32)),  acc2);
    acc3 = _mm512_add_epi32(_mm512_mullo_epi32(LOADI512(a+i+48),  LOADI512(b+i+48)),  acc3);
    acc4 = _mm512_add_epi32(_mm512_mullo_epi32(LOADI512(a+i+64),  LOADI512(b+i+64)),  acc4);
    acc5 = _mm512_add_epi32(_mm512_mullo_epi32(LOADI512(a+i+80),  LOADI512(b+i+80)),  acc5);
    acc6 = _mm512_add_epi32(_mm512_mullo_epi32(LOADI512(a+i+96),  LOADI512(b+i+96)),  acc6);
    acc7 = _mm512_add_epi32(_mm512_mullo_epi32(LOADI512(a+i+112), LOADI512(b+i+112)), acc7);
  }
  REDUCE_ACC8_512(_mm512_add_epi32, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);

  uint32_t result = (uint32_t)_mm512_reduce_add_epi32(acc0);
  uint32_t tail = 0;
  for (; i < n; i++) tail += a[i] * b[i];
  *dest = result + tail;
}

/* ── 64-bit integer dot products ─────────────────────────────────────── */

static inline void dot_i64_avx512(const int64_t *a, const int64_t *b, size_t n,
                                  int64_t *dest) {
  __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512(),
          acc2 = _mm512_setzero_si512(), acc3 = _mm512_setzero_si512(),
          acc4 = _mm512_setzero_si512(), acc5 = _mm512_setzero_si512(),
          acc6 = _mm512_setzero_si512(), acc7 = _mm512_setzero_si512();

  size_t i = 0;
  for (; i + 64 <= n; i += 64) {
    PREFETCH2_512(a, b, i + 512);
    acc0 = _mm512_add_epi64(_mm512_mullo_epi64(LOADI512(a+i),    LOADI512(b+i)),    acc0);
    acc1 = _mm512_add_epi64(_mm512_mullo_epi64(LOADI512(a+i+8),  LOADI512(b+i+8)),  acc1);
    acc2 = _mm512_add_epi64(_mm512_mullo_epi64(LOADI512(a+i+16), LOADI512(b+i+16)), acc2);
    acc3 = _mm512_add_epi64(_mm512_mullo_epi64(LOADI512(a+i+24), LOADI512(b+i+24)), acc3);
    acc4 = _mm512_add_epi64(_mm512_mullo_epi64(LOADI512(a+i+32), LOADI512(b+i+32)), acc4);
    acc5 = _mm512_add_epi64(_mm512_mullo_epi64(LOADI512(a+i+40), LOADI512(b+i+40)), acc5);
    acc6 = _mm512_add_epi64(_mm512_mullo_epi64(LOADI512(a+i+48), LOADI512(b+i+48)), acc6);
    acc7 = _mm512_add_epi64(_mm512_mullo_epi64(LOADI512(a+i+56), LOADI512(b+i+56)), acc7);
  }
  REDUCE_ACC8_512(_mm512_add_epi64, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);

  int64_t result = _mm512_reduce_add_epi64(acc0);
  int64_t tail = 0;
  for (; i < n; i++) tail += a[i] * b[i];
  *dest = result + tail;
}

static inline void dot_u64_avx512(const uint64_t *a, const uint64_t *b, size_t n,
                                  uint64_t *dest) {
  __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512(),
          acc2 = _mm512_setzero_si512(), acc3 = _mm512_setzero_si512(),
          acc4 = _mm512_setzero_si512(), acc5 = _mm512_setzero_si512(),
          acc6 = _mm512_setzero_si512(), acc7 = _mm512_setzero_si512();

  size_t i = 0;
  for (; i + 64 <= n; i += 64) {
    PREFETCH2_512(a, b, i + 512);
    acc0 = _mm512_add_epi64(_mm512_mullo_epi64(LOADI512(a+i),    LOADI512(b+i)),    acc0);
    acc1 = _mm512_add_epi64(_mm512_mullo_epi64(LOADI512(a+i+8),  LOADI512(b+i+8)),  acc1);
    acc2 = _mm512_add_epi64(_mm512_mullo_epi64(LOADI512(a+i+16), LOADI512(b+i+16)), acc2);
    acc3 = _mm512_add_epi64(_mm512_mullo_epi64(LOADI512(a+i+24), LOADI512(b+i+24)), acc3);
    acc4 = _mm512_add_epi64(_mm512_mullo_epi64(LOADI512(a+i+32), LOADI512(b+i+32)), acc4);
    acc5 = _mm512_add_epi64(_mm512_mullo_epi64(LOADI512(a+i+40), LOADI512(b+i+40)), acc5);
    acc6 = _mm512_add_epi64(_mm512_mullo_epi64(LOADI512(a+i+48), LOADI512(b+i+48)), acc6);
    acc7 = _mm512_add_epi64(_mm512_mullo_epi64(LOADI512(a+i+56), LOADI512(b+i+56)), acc7);
  }
  REDUCE_ACC8_512(_mm512_add_epi64, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);

  uint64_t result = (uint64_t)_mm512_reduce_add_epi64(acc0);
  uint64_t tail = 0;
  for (; i < n; i++) tail += a[i] * b[i];
  *dest = result + tail;
}

/* ── 8-bit dot (widen to i16, madd_epi16 → i32 accumulators) ────────── */

static inline void dot_i8_avx512(const int8_t *a, const int8_t *b, size_t n,
                                  int8_t *dest) {
  __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512(),
          acc2 = _mm512_setzero_si512(), acc3 = _mm512_setzero_si512(),
          acc4 = _mm512_setzero_si512(), acc5 = _mm512_setzero_si512(),
          acc6 = _mm512_setzero_si512(), acc7 = _mm512_setzero_si512();

  size_t i = 0;
  for (; i + 512 <= n; i += 512) {
    PREFETCH2_512(a, b, i + 1024);
    WIDEN_MADD_I8_512(acc0, a+i,     b+i);
    WIDEN_MADD_I8_512(acc1, a+i+64,  b+i+64);
    WIDEN_MADD_I8_512(acc2, a+i+128, b+i+128);
    WIDEN_MADD_I8_512(acc3, a+i+192, b+i+192);
    WIDEN_MADD_I8_512(acc4, a+i+256, b+i+256);
    WIDEN_MADD_I8_512(acc5, a+i+320, b+i+320);
    WIDEN_MADD_I8_512(acc6, a+i+384, b+i+384);
    WIDEN_MADD_I8_512(acc7, a+i+448, b+i+448);
  }
  REDUCE_ACC8_512(_mm512_add_epi32, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);

  int result = _mm512_reduce_add_epi32(acc0);
  int tail = 0;
  for (; i < n; i++) tail += a[i] * b[i];
  *dest = (int8_t)(result + tail);
}

static inline void dot_u8_avx512(const uint8_t *a, const uint8_t *b, size_t n,
                                  uint8_t *dest) {
  __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512(),
          acc2 = _mm512_setzero_si512(), acc3 = _mm512_setzero_si512(),
          acc4 = _mm512_setzero_si512(), acc5 = _mm512_setzero_si512(),
          acc6 = _mm512_setzero_si512(), acc7 = _mm512_setzero_si512();

  size_t i = 0;
  for (; i + 512 <= n; i += 512) {
    PREFETCH2_512(a, b, i + 1024);
    WIDEN_MADD_U8_512(acc0, a+i,     b+i);
    WIDEN_MADD_U8_512(acc1, a+i+64,  b+i+64);
    WIDEN_MADD_U8_512(acc2, a+i+128, b+i+128);
    WIDEN_MADD_U8_512(acc3, a+i+192, b+i+192);
    WIDEN_MADD_U8_512(acc4, a+i+256, b+i+256);
    WIDEN_MADD_U8_512(acc5, a+i+320, b+i+320);
    WIDEN_MADD_U8_512(acc6, a+i+384, b+i+384);
    WIDEN_MADD_U8_512(acc7, a+i+448, b+i+448);
  }
  REDUCE_ACC8_512(_mm512_add_epi32, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);

  uint8_t result = (uint8_t)_mm512_reduce_add_epi32(acc0);
  uint8_t tail = 0;
  for (; i < n; i++) tail += (uint8_t)(a[i] * b[i]);
  *dest = (uint8_t)(result + tail);
}

/* ── 16-bit dot (madd_epi16 → i32 accumulators) ─────────────────────── */

static inline void dot_i16_avx512(const int16_t *a, const int16_t *b, size_t n,
                                  int16_t *dest) {
  __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512(),
          acc2 = _mm512_setzero_si512(), acc3 = _mm512_setzero_si512(),
          acc4 = _mm512_setzero_si512(), acc5 = _mm512_setzero_si512(),
          acc6 = _mm512_setzero_si512(), acc7 = _mm512_setzero_si512();

  size_t i = 0;
  for (; i + 256 <= n; i += 256) {
    PREFETCH2_512(a, b, i + 512);
    acc0 = _mm512_add_epi32(_mm512_madd_epi16(LOADI512(a+i),     LOADI512(b+i)),     acc0);
    acc1 = _mm512_add_epi32(_mm512_madd_epi16(LOADI512(a+i+32),  LOADI512(b+i+32)),  acc1);
    acc2 = _mm512_add_epi32(_mm512_madd_epi16(LOADI512(a+i+64),  LOADI512(b+i+64)),  acc2);
    acc3 = _mm512_add_epi32(_mm512_madd_epi16(LOADI512(a+i+96),  LOADI512(b+i+96)),  acc3);
    acc4 = _mm512_add_epi32(_mm512_madd_epi16(LOADI512(a+i+128), LOADI512(b+i+128)), acc4);
    acc5 = _mm512_add_epi32(_mm512_madd_epi16(LOADI512(a+i+160), LOADI512(b+i+160)), acc5);
    acc6 = _mm512_add_epi32(_mm512_madd_epi16(LOADI512(a+i+192), LOADI512(b+i+192)), acc6);
    acc7 = _mm512_add_epi32(_mm512_madd_epi16(LOADI512(a+i+224), LOADI512(b+i+224)), acc7);
  }
  REDUCE_ACC8_512(_mm512_add_epi32, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);

  int result = _mm512_reduce_add_epi32(acc0);
  int tail = 0;
  for (; i < n; i++) tail += a[i] * b[i];
  *dest = (int16_t)(result + tail);
}

static inline void dot_u16_avx512(const uint16_t *a, const uint16_t *b, size_t n,
                                  uint16_t *dest) {
  __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512(),
          acc2 = _mm512_setzero_si512(), acc3 = _mm512_setzero_si512(),
          acc4 = _mm512_setzero_si512(), acc5 = _mm512_setzero_si512(),
          acc6 = _mm512_setzero_si512(), acc7 = _mm512_setzero_si512();

  size_t i = 0;
  for (; i + 128 <= n; i += 128) {
    PREFETCH2_512(a, b, i + 512);
    WIDEN_MUL_U16_512(acc0, acc1, a+i,    b+i);
    WIDEN_MUL_U16_512(acc2, acc3, a+i+32, b+i+32);
    WIDEN_MUL_U16_512(acc4, acc5, a+i+64, b+i+64);
    WIDEN_MUL_U16_512(acc6, acc7, a+i+96, b+i+96);
  }
  REDUCE_ACC8_512(_mm512_add_epi32, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);

  uint16_t result = (uint16_t)_mm512_reduce_add_epi32(acc0);
  uint16_t tail = 0;
  for (; i < n; i++) tail += (uint16_t)(a[i] * b[i]);
  *dest = (uint16_t)(result + tail);
}

// clang-format on

#undef REDUCE_ACC8_512
#undef PREFETCH2_512
#undef LOADI512
#undef WIDEN_MADD_I8_512
#undef WIDEN_MADD_U8_512
#undef WIDEN_MUL_U16_512

#endif
