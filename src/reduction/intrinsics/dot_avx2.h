#ifndef NUMC_DOT_AVX2_H
#define NUMC_DOT_AVX2_H

#include <immintrin.h>

static inline void dot_f32u_avx2(const float *a, const float *b, size_t n,
                                 float *dest) {
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();
  __m256 acc4 = _mm256_setzero_ps();
  __m256 acc5 = _mm256_setzero_ps();
  __m256 acc6 = _mm256_setzero_ps();
  __m256 acc7 = _mm256_setzero_ps();

  size_t i = 0;
  for (; i + 64 <= n; i += 64) {
    acc0 =
        _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),
                           _mm256_loadu_ps(b + i + 8), acc1);
    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16),
                           _mm256_loadu_ps(b + i + 16), acc2);
    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24),
                           _mm256_loadu_ps(b + i + 24), acc3);
    acc4 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 32),
                           _mm256_loadu_ps(b + i + 32), acc4);
    acc5 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 40),
                           _mm256_loadu_ps(b + i + 40), acc5);
    acc6 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 48),
                           _mm256_loadu_ps(b + i + 48), acc6);
    acc7 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 56),
                           _mm256_loadu_ps(b + i + 56), acc7);
  }
  acc0 = _mm256_add_ps(acc0, acc1);
  acc2 = _mm256_add_ps(acc2, acc3);
  acc4 = _mm256_add_ps(acc4, acc5);
  acc6 = _mm256_add_ps(acc6, acc7);
  acc0 = _mm256_add_ps(acc0, acc2);
  acc4 = _mm256_add_ps(acc4, acc6);
  acc0 = _mm256_add_ps(acc0, acc4);

  __m128 lo = _mm256_castps256_ps128(acc0);
  __m128 hi = _mm256_extractf128_ps(acc0, 1);
  __m128 sum128 = _mm_add_ps(lo, hi);
  sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
  sum128 = _mm_add_ss(sum128, _mm_movehdup_ps(sum128));
  float result = _mm_cvtss_f32(sum128);

  float tail = 0.0f;
  for (; i < n; i++)
    tail += a[i] * b[i];

  *dest = result + tail;
}


static inline void dot_f32_avx2(const float *a, const float *b, size_t n,
                                float *dest) {
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();
  __m256 acc4 = _mm256_setzero_ps();
  __m256 acc5 = _mm256_setzero_ps();
  __m256 acc6 = _mm256_setzero_ps();
  __m256 acc7 = _mm256_setzero_ps();

  size_t i = 0;
  for (; i + 64 <= n; i += 64) {
    acc0 = _mm256_fmadd_ps(_mm256_load_ps(a + i), _mm256_load_ps(b + i), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_load_ps(a + i + 8), _mm256_load_ps(b + i + 8),
                           acc1);
    acc2 = _mm256_fmadd_ps(_mm256_load_ps(a + i + 16),
                           _mm256_load_ps(b + i + 16), acc2);
    acc3 = _mm256_fmadd_ps(_mm256_load_ps(a + i + 24),
                           _mm256_load_ps(b + i + 24), acc3);
    acc4 = _mm256_fmadd_ps(_mm256_load_ps(a + i + 32),
                           _mm256_load_ps(b + i + 32), acc4);
    acc5 = _mm256_fmadd_ps(_mm256_load_ps(a + i + 40),
                           _mm256_load_ps(b + i + 40), acc5);
    acc6 = _mm256_fmadd_ps(_mm256_load_ps(a + i + 48),
                           _mm256_load_ps(b + i + 48), acc6);
    acc7 = _mm256_fmadd_ps(_mm256_load_ps(a + i + 56),
                           _mm256_load_ps(b + i + 56), acc7);
  }
  acc0 = _mm256_add_ps(acc0, acc1);
  acc2 = _mm256_add_ps(acc2, acc3);
  acc4 = _mm256_add_ps(acc4, acc5);
  acc6 = _mm256_add_ps(acc6, acc7);
  acc0 = _mm256_add_ps(acc0, acc2);
  acc4 = _mm256_add_ps(acc4, acc6);
  acc0 = _mm256_add_ps(acc0, acc4);

  __m128 lo = _mm256_castps256_ps128(acc0);
  __m128 hi = _mm256_extractf128_ps(acc0, 1);
  __m128 sum128 = _mm_add_ps(lo, hi);
  sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
  sum128 = _mm_add_ss(sum128, _mm_movehdup_ps(sum128));
  float result = _mm_cvtss_f32(sum128);

  float tail = 0.0f;
  for (; i < n; i++)
    tail += a[i] * b[i];

  *dest = result + tail;
}

static inline void dot_f64u_avx2(const double *a, const double *b, size_t n,
                                 double *dest) {

  __m256d acc0 = _mm256_setzero_pd();
  __m256d acc1 = _mm256_setzero_pd();
  __m256d acc2 = _mm256_setzero_pd();
  __m256d acc3 = _mm256_setzero_pd();
  __m256d acc4 = _mm256_setzero_pd();
  __m256d acc5 = _mm256_setzero_pd();
  __m256d acc6 = _mm256_setzero_pd();
  __m256d acc7 = _mm256_setzero_pd();

  size_t i = 0;

  for (; i + 32 <= n; i += 32) {
    acc0 =
        _mm256_fmadd_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i), acc0);
    acc1 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i + 4),
                           _mm256_loadu_pd(b + i + 4), acc1);
    acc2 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i + 8),
                           _mm256_loadu_pd(b + i + 8), acc2);
    acc3 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i + 12),
                           _mm256_loadu_pd(b + i + 12), acc3);
    acc4 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i + 16),
                           _mm256_loadu_pd(b + i + 16), acc4);

    acc5 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i + 20),
                           _mm256_loadu_pd(b + i + 20), acc5);
    acc6 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i + 24),
                           _mm256_loadu_pd(b + i + 24), acc6);

    acc7 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i + 28),
                           _mm256_loadu_pd(b + i + 28), acc7);
  }

  acc0 = _mm256_add_pd(acc0, acc1);
  acc2 = _mm256_add_pd(acc2, acc3);
  acc4 = _mm256_add_pd(acc4, acc5);
  acc6 = _mm256_add_pd(acc6, acc7);
  acc0 = _mm256_add_pd(acc0, acc2);
  acc4 = _mm256_add_pd(acc4, acc6);
  acc0 = _mm256_add_pd(acc0, acc4);

  __m128d lo = _mm256_castpd256_pd128(acc0);
  __m128d hi = _mm256_extractf128_pd(acc0, 1);
  __m128d sum = _mm_add_pd(lo, hi);
  sum = _mm_add_pd(sum, _mm_unpackhi_pd(sum, sum));
  double result = _mm_cvtsd_f64(sum);

  double tail = 0.0;
  for (; i < n; i++)
    tail += a[i] * b[i];


  *dest = result + tail;
}

static inline void dot_f64_avx2(const double *a, const double *b, size_t n,
                                double *dest) {
  __m256d acc0 = _mm256_setzero_pd();
  __m256d acc1 = _mm256_setzero_pd();
  __m256d acc2 = _mm256_setzero_pd();
  __m256d acc3 = _mm256_setzero_pd();
  __m256d acc4 = _mm256_setzero_pd();
  __m256d acc5 = _mm256_setzero_pd();
  __m256d acc6 = _mm256_setzero_pd();
  __m256d acc7 = _mm256_setzero_pd();

  size_t i = 0;

  for (; i + 32 <= n; i += 32) {
    acc0 = _mm256_fmadd_pd(_mm256_load_pd(a + i), _mm256_load_pd(b + i), acc0);
    acc1 = _mm256_fmadd_pd(_mm256_load_pd(a + i + 4), _mm256_load_pd(b + i + 4),
                           acc1);
    acc2 = _mm256_fmadd_pd(_mm256_load_pd(a + i + 8), _mm256_load_pd(b + i + 8),
                           acc2);
    acc3 = _mm256_fmadd_pd(_mm256_load_pd(a + i + 12),
                           _mm256_load_pd(b + i + 12), acc3);
    acc4 = _mm256_fmadd_pd(_mm256_load_pd(a + i + 16),
                           _mm256_load_pd(b + i + 16), acc4);

    acc5 = _mm256_fmadd_pd(_mm256_load_pd(a + i + 20),
                           _mm256_load_pd(b + i + 20), acc5);
    acc6 = _mm256_fmadd_pd(_mm256_load_pd(a + i + 24),
                           _mm256_load_pd(b + i + 24), acc6);

    acc7 = _mm256_fmadd_pd(_mm256_load_pd(a + i + 28),
                           _mm256_load_pd(b + i + 28), acc7);
  }

  acc0 = _mm256_add_pd(acc0, acc1);
  acc2 = _mm256_add_pd(acc2, acc3);
  acc4 = _mm256_add_pd(acc4, acc5);
  acc6 = _mm256_add_pd(acc6, acc7);
  acc0 = _mm256_add_pd(acc0, acc2);
  acc4 = _mm256_add_pd(acc4, acc6);
  acc0 = _mm256_add_pd(acc0, acc4);

  __m128d lo = _mm256_castpd256_pd128(acc0);
  __m128d hi = _mm256_extractf128_pd(acc0, 1);
  __m128d sum = _mm_add_pd(lo, hi);
  sum = _mm_add_pd(sum, _mm_unpackhi_pd(sum, sum));
  double result = _mm_cvtsd_f64(sum);

  double tail = 0.0;
  for (; i < n; i++) {
    tail += a[i] * b[i];
  }

  *dest = result + tail;
}


#endif
