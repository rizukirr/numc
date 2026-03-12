#ifndef NUMC_GEMMSUP_AVX2_H
#define NUMC_GEMMSUP_AVX2_H

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define GEMMSUP_MIN(a, b) ((a) < (b) ? (a) : (b))

/* Threshold: M*K*N below which we use unpacked kernels.
 * Packing overhead dominates for matrices <= ~128x128. */
#define GEMMSUP_FLOPS_THRESHOLD (128UL * 128UL * 128UL)

/* =================================================================
   Float32 unpacked 6x16 micro-kernel
   Reads directly from strided A and B -- no packing needed.
   Same register tile as packed kernel (12 YMM accumulators +
   B loads + A bcast).
   ================================================================= */

static inline void gemmsup_ukernel_f32_6x16(const float *a, const float *b,
                                            float *c, size_t kc, intptr_t rsa,
                                            intptr_t csa, intptr_t rsb,
                                            intptr_t rso) {
  __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
  __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
  __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
  __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
  __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
  __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();

  /* 4× K-loop unroll to reduce loop overhead */
#define GEMMSUP_F32_K_BODY(p_off)                                              \
  do {                                                                         \
    const float *bp_ = b + (p_off)*rsb;                                        \
    __m256 b0_ = _mm256_loadu_ps(bp_);                                         \
    __m256 b1_ = _mm256_loadu_ps(bp_ + 8);                                     \
    __m256 av_;                                                                \
    av_ = _mm256_broadcast_ss(a + 0 * rsa + (p_off)*csa);                      \
    c00 = _mm256_fmadd_ps(av_, b0_, c00);                                      \
    c01 = _mm256_fmadd_ps(av_, b1_, c01);                                      \
    av_ = _mm256_broadcast_ss(a + 1 * rsa + (p_off)*csa);                      \
    c10 = _mm256_fmadd_ps(av_, b0_, c10);                                      \
    c11 = _mm256_fmadd_ps(av_, b1_, c11);                                      \
    av_ = _mm256_broadcast_ss(a + 2 * rsa + (p_off)*csa);                      \
    c20 = _mm256_fmadd_ps(av_, b0_, c20);                                      \
    c21 = _mm256_fmadd_ps(av_, b1_, c21);                                      \
    av_ = _mm256_broadcast_ss(a + 3 * rsa + (p_off)*csa);                      \
    c30 = _mm256_fmadd_ps(av_, b0_, c30);                                      \
    c31 = _mm256_fmadd_ps(av_, b1_, c31);                                      \
    av_ = _mm256_broadcast_ss(a + 4 * rsa + (p_off)*csa);                      \
    c40 = _mm256_fmadd_ps(av_, b0_, c40);                                      \
    c41 = _mm256_fmadd_ps(av_, b1_, c41);                                      \
    av_ = _mm256_broadcast_ss(a + 5 * rsa + (p_off)*csa);                      \
    c50 = _mm256_fmadd_ps(av_, b0_, c50);                                      \
    c51 = _mm256_fmadd_ps(av_, b1_, c51);                                      \
  } while (0)

  size_t k_iter = kc / 4;
  size_t k_left = kc % 4;
  size_t p = 0;
  for (size_t ki = 0; ki < k_iter; ki++, p += 4) {
    GEMMSUP_F32_K_BODY(p);
    GEMMSUP_F32_K_BODY(p + 1);
    GEMMSUP_F32_K_BODY(p + 2);
    GEMMSUP_F32_K_BODY(p + 3);
  }
  for (size_t pi = 0; pi < k_left; pi++, p++) {
    GEMMSUP_F32_K_BODY(p);
  }
#undef GEMMSUP_F32_K_BODY

  _mm256_storeu_ps(c, c00);
  _mm256_storeu_ps(c + 8, c01);
  _mm256_storeu_ps(c + rso, c10);
  _mm256_storeu_ps(c + rso + 8, c11);
  _mm256_storeu_ps(c + 2 * rso, c20);
  _mm256_storeu_ps(c + 2 * rso + 8, c21);
  _mm256_storeu_ps(c + 3 * rso, c30);
  _mm256_storeu_ps(c + 3 * rso + 8, c31);
  _mm256_storeu_ps(c + 4 * rso, c40);
  _mm256_storeu_ps(c + 4 * rso + 8, c41);
  _mm256_storeu_ps(c + 5 * rso, c50);
  _mm256_storeu_ps(c + 5 * rso + 8, c51);
}

/* Scalar edge kernel for remainder tiles */
static inline void gemmsup_edge_f32(const float *a, const float *b, float *c,
                                    size_t mr, size_t nr, size_t kc,
                                    intptr_t rsa, intptr_t csa, intptr_t rsb,
                                    intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
    for (size_t j = 0; j < nr; j++)
      c[i * rso + j] = 0.0f;
    for (size_t p = 0; p < kc; p++) {
      float aip = a[i * rsa + p * csa];
      __m256 va = _mm256_set1_ps(aip);
      const float *brow = b + p * rsb;
      float *crow = c + i * rso;
      size_t j = 0;
      for (; j + 8 <= nr; j += 8) {
        __m256 vo = _mm256_loadu_ps(crow + j);
        vo = _mm256_fmadd_ps(va, _mm256_loadu_ps(brow + j), vo);
        _mm256_storeu_ps(crow + j, vo);
      }
      for (; j < nr; j++)
        crow[j] += aip * brow[j];
    }
  }
}

/* Full gemmsup dispatch: tile MxN in 6x16 blocks, no packing */
static inline void gemmsup_f32_avx2(const float *a, const float *b, float *out,
                                    size_t M, size_t K, size_t N, intptr_t rsa,
                                    intptr_t csa, intptr_t rsb, intptr_t rso) {
  const size_t MR = 6, NR = 16;
  for (size_t i = 0; i < M; i += MR) {
    size_t mr = GEMMSUP_MIN(MR, M - i);
    for (size_t j = 0; j < N; j += NR) {
      size_t nr = GEMMSUP_MIN(NR, N - j);
      if (mr == MR && nr == NR) {
        gemmsup_ukernel_f32_6x16(a + i * rsa, b + j, out + i * rso + j, K, rsa,
                                 csa, rsb, rso);
      } else {
        gemmsup_edge_f32(a + i * rsa, b + j, out + i * rso + j, mr, nr, K, rsa,
                         csa, rsb, rso);
      }
    }
  }
}

/* =================================================================
   Float64 unpacked 6x8 micro-kernel
   ================================================================= */

static inline void gemmsup_ukernel_f64_6x8(const double *a, const double *b,
                                           double *c, size_t kc, intptr_t rsa,
                                           intptr_t csa, intptr_t rsb,
                                           intptr_t rso) {
  __m256d c00 = _mm256_setzero_pd(), c01 = _mm256_setzero_pd();
  __m256d c10 = _mm256_setzero_pd(), c11 = _mm256_setzero_pd();
  __m256d c20 = _mm256_setzero_pd(), c21 = _mm256_setzero_pd();
  __m256d c30 = _mm256_setzero_pd(), c31 = _mm256_setzero_pd();
  __m256d c40 = _mm256_setzero_pd(), c41 = _mm256_setzero_pd();
  __m256d c50 = _mm256_setzero_pd(), c51 = _mm256_setzero_pd();

  /* 4× K-loop unroll to reduce loop overhead */
#define GEMMSUP_F64_K_BODY(p_off)                                              \
  do {                                                                         \
    const double *bp_ = b + (p_off)*rsb;                                       \
    __m256d b0_ = _mm256_loadu_pd(bp_);                                        \
    __m256d b1_ = _mm256_loadu_pd(bp_ + 4);                                    \
    __m256d av_;                                                               \
    av_ = _mm256_broadcast_sd(a + 0 * rsa + (p_off)*csa);                      \
    c00 = _mm256_fmadd_pd(av_, b0_, c00);                                      \
    c01 = _mm256_fmadd_pd(av_, b1_, c01);                                      \
    av_ = _mm256_broadcast_sd(a + 1 * rsa + (p_off)*csa);                      \
    c10 = _mm256_fmadd_pd(av_, b0_, c10);                                      \
    c11 = _mm256_fmadd_pd(av_, b1_, c11);                                      \
    av_ = _mm256_broadcast_sd(a + 2 * rsa + (p_off)*csa);                      \
    c20 = _mm256_fmadd_pd(av_, b0_, c20);                                      \
    c21 = _mm256_fmadd_pd(av_, b1_, c21);                                      \
    av_ = _mm256_broadcast_sd(a + 3 * rsa + (p_off)*csa);                      \
    c30 = _mm256_fmadd_pd(av_, b0_, c30);                                      \
    c31 = _mm256_fmadd_pd(av_, b1_, c31);                                      \
    av_ = _mm256_broadcast_sd(a + 4 * rsa + (p_off)*csa);                      \
    c40 = _mm256_fmadd_pd(av_, b0_, c40);                                      \
    c41 = _mm256_fmadd_pd(av_, b1_, c41);                                      \
    av_ = _mm256_broadcast_sd(a + 5 * rsa + (p_off)*csa);                      \
    c50 = _mm256_fmadd_pd(av_, b0_, c50);                                      \
    c51 = _mm256_fmadd_pd(av_, b1_, c51);                                      \
  } while (0)

  size_t k_iter = kc / 4;
  size_t k_left = kc % 4;
  size_t p = 0;
  for (size_t ki = 0; ki < k_iter; ki++, p += 4) {
    GEMMSUP_F64_K_BODY(p);
    GEMMSUP_F64_K_BODY(p + 1);
    GEMMSUP_F64_K_BODY(p + 2);
    GEMMSUP_F64_K_BODY(p + 3);
  }
  for (size_t pi = 0; pi < k_left; pi++, p++) {
    GEMMSUP_F64_K_BODY(p);
  }
#undef GEMMSUP_F64_K_BODY

  _mm256_storeu_pd(c, c00);
  _mm256_storeu_pd(c + 4, c01);
  _mm256_storeu_pd(c + rso, c10);
  _mm256_storeu_pd(c + rso + 4, c11);
  _mm256_storeu_pd(c + 2 * rso, c20);
  _mm256_storeu_pd(c + 2 * rso + 4, c21);
  _mm256_storeu_pd(c + 3 * rso, c30);
  _mm256_storeu_pd(c + 3 * rso + 4, c31);
  _mm256_storeu_pd(c + 4 * rso, c40);
  _mm256_storeu_pd(c + 4 * rso + 4, c41);
  _mm256_storeu_pd(c + 5 * rso, c50);
  _mm256_storeu_pd(c + 5 * rso + 4, c51);
}

static inline void gemmsup_edge_f64(const double *a, const double *b, double *c,
                                    size_t mr, size_t nr, size_t kc,
                                    intptr_t rsa, intptr_t csa, intptr_t rsb,
                                    intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
    for (size_t j = 0; j < nr; j++)
      c[i * rso + j] = 0.0;
    for (size_t p = 0; p < kc; p++) {
      double aip = a[i * rsa + p * csa];
      __m256d va = _mm256_set1_pd(aip);
      const double *brow = b + p * rsb;
      double *crow = c + i * rso;
      size_t j = 0;
      for (; j + 4 <= nr; j += 4) {
        __m256d vo = _mm256_loadu_pd(crow + j);
        vo = _mm256_fmadd_pd(va, _mm256_loadu_pd(brow + j), vo);
        _mm256_storeu_pd(crow + j, vo);
      }
      for (; j < nr; j++)
        crow[j] += aip * brow[j];
    }
  }
}

static inline void gemmsup_f64_avx2(const double *a, const double *b,
                                    double *out, size_t M, size_t K, size_t N,
                                    intptr_t rsa, intptr_t csa, intptr_t rsb,
                                    intptr_t rso) {
  const size_t MR = 6, NR = 8;
  for (size_t i = 0; i < M; i += MR) {
    size_t mr = GEMMSUP_MIN(MR, M - i);
    for (size_t j = 0; j < N; j += NR) {
      size_t nr = GEMMSUP_MIN(NR, N - j);
      if (mr == MR && nr == NR) {
        gemmsup_ukernel_f64_6x8(a + i * rsa, b + j, out + i * rso + j, K, rsa,
                                csa, rsb, rso);
      } else {
        gemmsup_edge_f64(a + i * rsa, b + j, out + i * rso + j, mr, nr, K, rsa,
                         csa, rsb, rso);
      }
    }
  }
}

#undef GEMMSUP_MIN

#endif
