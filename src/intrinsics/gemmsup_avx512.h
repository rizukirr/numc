#ifndef NUMC_GEMMSUP_AVX512_H
#define NUMC_GEMMSUP_AVX512_H

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define GEMMSUP512_MIN(a, b) ((a) < (b) ? (a) : (b))

/* OMP gate for gemmsup: parallelize from ~128^3 and up. */
#define GEMMSUP512_OMP_THRESHOLD (1ULL << 20)

/* Threshold: M*K*N below which we use unpacked kernels.
 * Packing overhead dominates for matrices <= ~128x128.
 * AVX-512 packed GEMM has higher packing cost (larger MR/NR),
 * so the crossover may be slightly higher than AVX2. */
#define GEMMSUP512_FLOPS_THRESHOLD (128UL * 128UL * 128UL)

/* =================================================================
   Float32 unpacked 12x32 micro-kernel (AVX-512)
   Reads directly from strided A and B -- no packing needed.
   24 ZMM accumulators (12 rows × 2 ZMM) + 2 B loads + 2 A bcast
   = 28 of 32 ZMM registers.
   ================================================================= */

static inline void gemmsup_ukernel_f32_12x32(const float *a, const float *b,
                                             float *c, size_t kc, intptr_t rsa,
                                             intptr_t csa, intptr_t rsb,
                                             intptr_t rso) {
  __m512 c00 = _mm512_setzero_ps(), c01 = _mm512_setzero_ps();
  __m512 c10 = _mm512_setzero_ps(), c11 = _mm512_setzero_ps();
  __m512 c20 = _mm512_setzero_ps(), c21 = _mm512_setzero_ps();
  __m512 c30 = _mm512_setzero_ps(), c31 = _mm512_setzero_ps();
  __m512 c40 = _mm512_setzero_ps(), c41 = _mm512_setzero_ps();
  __m512 c50 = _mm512_setzero_ps(), c51 = _mm512_setzero_ps();
  __m512 c60 = _mm512_setzero_ps(), c61 = _mm512_setzero_ps();
  __m512 c70 = _mm512_setzero_ps(), c71 = _mm512_setzero_ps();
  __m512 c80 = _mm512_setzero_ps(), c81 = _mm512_setzero_ps();
  __m512 c90 = _mm512_setzero_ps(), c91 = _mm512_setzero_ps();
  __m512 cA0 = _mm512_setzero_ps(), cA1 = _mm512_setzero_ps();
  __m512 cB0 = _mm512_setzero_ps(), cB1 = _mm512_setzero_ps();

  /* 2 alternating A broadcast variables (a0_/a1_) for ILP.
   * 12 rows unrolled as 6 pairs. */
#define GEMMSUP512_F32_K_BODY(p_off)                   \
  do {                                                 \
    const float *bp_ = b + (p_off) * rsb;              \
    __m512 b0_ = _mm512_loadu_ps(bp_);                 \
    __m512 b1_ = _mm512_loadu_ps(bp_ + 16);            \
    __m512 a0_, a1_;                                   \
    a0_ = _mm512_set1_ps(a[0 * rsa + (p_off) * csa]);  \
    a1_ = _mm512_set1_ps(a[1 * rsa + (p_off) * csa]);  \
    c00 = _mm512_fmadd_ps(a0_, b0_, c00);              \
    c01 = _mm512_fmadd_ps(a0_, b1_, c01);              \
    c10 = _mm512_fmadd_ps(a1_, b0_, c10);              \
    c11 = _mm512_fmadd_ps(a1_, b1_, c11);              \
    a0_ = _mm512_set1_ps(a[2 * rsa + (p_off) * csa]);  \
    a1_ = _mm512_set1_ps(a[3 * rsa + (p_off) * csa]);  \
    c20 = _mm512_fmadd_ps(a0_, b0_, c20);              \
    c21 = _mm512_fmadd_ps(a0_, b1_, c21);              \
    c30 = _mm512_fmadd_ps(a1_, b0_, c30);              \
    c31 = _mm512_fmadd_ps(a1_, b1_, c31);              \
    a0_ = _mm512_set1_ps(a[4 * rsa + (p_off) * csa]);  \
    a1_ = _mm512_set1_ps(a[5 * rsa + (p_off) * csa]);  \
    c40 = _mm512_fmadd_ps(a0_, b0_, c40);              \
    c41 = _mm512_fmadd_ps(a0_, b1_, c41);              \
    c50 = _mm512_fmadd_ps(a1_, b0_, c50);              \
    c51 = _mm512_fmadd_ps(a1_, b1_, c51);              \
    a0_ = _mm512_set1_ps(a[6 * rsa + (p_off) * csa]);  \
    a1_ = _mm512_set1_ps(a[7 * rsa + (p_off) * csa]);  \
    c60 = _mm512_fmadd_ps(a0_, b0_, c60);              \
    c61 = _mm512_fmadd_ps(a0_, b1_, c61);              \
    c70 = _mm512_fmadd_ps(a1_, b0_, c70);              \
    c71 = _mm512_fmadd_ps(a1_, b1_, c71);              \
    a0_ = _mm512_set1_ps(a[8 * rsa + (p_off) * csa]);  \
    a1_ = _mm512_set1_ps(a[9 * rsa + (p_off) * csa]);  \
    c80 = _mm512_fmadd_ps(a0_, b0_, c80);              \
    c81 = _mm512_fmadd_ps(a0_, b1_, c81);              \
    c90 = _mm512_fmadd_ps(a1_, b0_, c90);              \
    c91 = _mm512_fmadd_ps(a1_, b1_, c91);              \
    a0_ = _mm512_set1_ps(a[10 * rsa + (p_off) * csa]); \
    a1_ = _mm512_set1_ps(a[11 * rsa + (p_off) * csa]); \
    cA0 = _mm512_fmadd_ps(a0_, b0_, cA0);              \
    cA1 = _mm512_fmadd_ps(a0_, b1_, cA1);              \
    cB0 = _mm512_fmadd_ps(a1_, b0_, cB0);              \
    cB1 = _mm512_fmadd_ps(a1_, b1_, cB1);              \
  } while (0)

  /* 8x K-loop unroll */
  size_t k_iter = kc / 8;
  size_t k_left = kc % 8;
  size_t p = 0;
  for (size_t ki = 0; ki < k_iter; ki++, p += 8) {
    GEMMSUP512_F32_K_BODY(p);
    GEMMSUP512_F32_K_BODY(p + 1);
    GEMMSUP512_F32_K_BODY(p + 2);
    GEMMSUP512_F32_K_BODY(p + 3);
    GEMMSUP512_F32_K_BODY(p + 4);
    GEMMSUP512_F32_K_BODY(p + 5);
    GEMMSUP512_F32_K_BODY(p + 6);
    GEMMSUP512_F32_K_BODY(p + 7);
  }
  for (size_t pi = 0; pi < k_left; pi++, p++) {
    GEMMSUP512_F32_K_BODY(p);
  }
#undef GEMMSUP512_F32_K_BODY

  _mm512_storeu_ps(c, c00);
  _mm512_storeu_ps(c + 16, c01);
  _mm512_storeu_ps(c + rso, c10);
  _mm512_storeu_ps(c + rso + 16, c11);
  _mm512_storeu_ps(c + 2 * rso, c20);
  _mm512_storeu_ps(c + 2 * rso + 16, c21);
  _mm512_storeu_ps(c + 3 * rso, c30);
  _mm512_storeu_ps(c + 3 * rso + 16, c31);
  _mm512_storeu_ps(c + 4 * rso, c40);
  _mm512_storeu_ps(c + 4 * rso + 16, c41);
  _mm512_storeu_ps(c + 5 * rso, c50);
  _mm512_storeu_ps(c + 5 * rso + 16, c51);
  _mm512_storeu_ps(c + 6 * rso, c60);
  _mm512_storeu_ps(c + 6 * rso + 16, c61);
  _mm512_storeu_ps(c + 7 * rso, c70);
  _mm512_storeu_ps(c + 7 * rso + 16, c71);
  _mm512_storeu_ps(c + 8 * rso, c80);
  _mm512_storeu_ps(c + 8 * rso + 16, c81);
  _mm512_storeu_ps(c + 9 * rso, c90);
  _mm512_storeu_ps(c + 9 * rso + 16, c91);
  _mm512_storeu_ps(c + 10 * rso, cA0);
  _mm512_storeu_ps(c + 10 * rso + 16, cA1);
  _mm512_storeu_ps(c + 11 * rso, cB0);
  _mm512_storeu_ps(c + 11 * rso + 16, cB1);
}

/* Scalar edge kernel for remainder tiles (AVX-512) */
static inline void gemmsup_edge_f32_avx512(const float *a, const float *b,
                                           float *c, size_t mr, size_t nr,
                                           size_t kc, intptr_t rsa,
                                           intptr_t csa, intptr_t rsb,
                                           intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
    for (size_t j = 0; j < nr; j++)
      c[i * rso + j] = 0.0f;
    for (size_t p = 0; p < kc; p++) {
      float aip = a[i * rsa + p * csa];
      __m512 va = _mm512_set1_ps(aip);
      const float *brow = b + p * rsb;
      float *crow = c + i * rso;
      size_t j = 0;
      for (; j + 16 <= nr; j += 16) {
        __m512 vo = _mm512_loadu_ps(crow + j);
        vo = _mm512_fmadd_ps(va, _mm512_loadu_ps(brow + j), vo);
        _mm512_storeu_ps(crow + j, vo);
      }
      for (; j < nr; j++)
        crow[j] += aip * brow[j];
    }
  }
}

/* Full gemmsup dispatch: tile MxN in 12x32 blocks, no packing.
 * OMP-parallel over MR-row tiles when compute volume warrants it. */
static inline void gemmsup_f32_avx512(const float *a, const float *b,
                                      float *out, size_t M, size_t K, size_t N,
                                      intptr_t rsa, intptr_t csa, intptr_t rsb,
                                      intptr_t rso) {
  const size_t MR = 12, NR = 32;
  size_t n_ir = (M + MR - 1) / MR;
#pragma omp parallel for schedule(static) if ((uint64_t)M * K * N > \
                                                  GEMMSUP512_OMP_THRESHOLD)
  for (size_t ir = 0; ir < n_ir; ir++) {
    size_t i = ir * MR;
    size_t mr = GEMMSUP512_MIN(MR, M - i);
    for (size_t j = 0; j < N; j += NR) {
      size_t nr = GEMMSUP512_MIN(NR, N - j);
      if (mr == MR && nr == NR) {
        gemmsup_ukernel_f32_12x32(a + i * rsa, b + j, out + i * rso + j, K, rsa,
                                  csa, rsb, rso);
      } else {
        gemmsup_edge_f32_avx512(a + i * rsa, b + j, out + i * rso + j, mr, nr,
                                K, rsa, csa, rsb, rso);
      }
    }
  }
}

/* =================================================================
   Float64 unpacked 14x16 micro-kernel (AVX-512)
   28 ZMM accumulators (14 rows × 2 ZMM) + 2 B loads + 2 A bcast
   = 32 ZMM registers (all used).
   ================================================================= */

static inline void gemmsup_ukernel_f64_14x16(const double *a, const double *b,
                                             double *c, size_t kc, intptr_t rsa,
                                             intptr_t csa, intptr_t rsb,
                                             intptr_t rso) {
  __m512d c00 = _mm512_setzero_pd(), c01 = _mm512_setzero_pd();
  __m512d c10 = _mm512_setzero_pd(), c11 = _mm512_setzero_pd();
  __m512d c20 = _mm512_setzero_pd(), c21 = _mm512_setzero_pd();
  __m512d c30 = _mm512_setzero_pd(), c31 = _mm512_setzero_pd();
  __m512d c40 = _mm512_setzero_pd(), c41 = _mm512_setzero_pd();
  __m512d c50 = _mm512_setzero_pd(), c51 = _mm512_setzero_pd();
  __m512d c60 = _mm512_setzero_pd(), c61 = _mm512_setzero_pd();
  __m512d c70 = _mm512_setzero_pd(), c71 = _mm512_setzero_pd();
  __m512d c80 = _mm512_setzero_pd(), c81 = _mm512_setzero_pd();
  __m512d c90 = _mm512_setzero_pd(), c91 = _mm512_setzero_pd();
  __m512d cA0 = _mm512_setzero_pd(), cA1 = _mm512_setzero_pd();
  __m512d cB0 = _mm512_setzero_pd(), cB1 = _mm512_setzero_pd();
  __m512d cC0 = _mm512_setzero_pd(), cC1 = _mm512_setzero_pd();
  __m512d cD0 = _mm512_setzero_pd(), cD1 = _mm512_setzero_pd();

  /* 2 alternating A broadcast variables (a0_/a1_) for ILP.
   * 14 rows unrolled as 7 pairs. */
#define GEMMSUP512_F64_K_BODY(p_off)                   \
  do {                                                 \
    const double *bp_ = b + (p_off) * rsb;             \
    __m512d b0_ = _mm512_loadu_pd(bp_);                \
    __m512d b1_ = _mm512_loadu_pd(bp_ + 8);            \
    __m512d a0_, a1_;                                  \
    a0_ = _mm512_set1_pd(a[0 * rsa + (p_off) * csa]);  \
    a1_ = _mm512_set1_pd(a[1 * rsa + (p_off) * csa]);  \
    c00 = _mm512_fmadd_pd(a0_, b0_, c00);              \
    c01 = _mm512_fmadd_pd(a0_, b1_, c01);              \
    c10 = _mm512_fmadd_pd(a1_, b0_, c10);              \
    c11 = _mm512_fmadd_pd(a1_, b1_, c11);              \
    a0_ = _mm512_set1_pd(a[2 * rsa + (p_off) * csa]);  \
    a1_ = _mm512_set1_pd(a[3 * rsa + (p_off) * csa]);  \
    c20 = _mm512_fmadd_pd(a0_, b0_, c20);              \
    c21 = _mm512_fmadd_pd(a0_, b1_, c21);              \
    c30 = _mm512_fmadd_pd(a1_, b0_, c30);              \
    c31 = _mm512_fmadd_pd(a1_, b1_, c31);              \
    a0_ = _mm512_set1_pd(a[4 * rsa + (p_off) * csa]);  \
    a1_ = _mm512_set1_pd(a[5 * rsa + (p_off) * csa]);  \
    c40 = _mm512_fmadd_pd(a0_, b0_, c40);              \
    c41 = _mm512_fmadd_pd(a0_, b1_, c41);              \
    c50 = _mm512_fmadd_pd(a1_, b0_, c50);              \
    c51 = _mm512_fmadd_pd(a1_, b1_, c51);              \
    a0_ = _mm512_set1_pd(a[6 * rsa + (p_off) * csa]);  \
    a1_ = _mm512_set1_pd(a[7 * rsa + (p_off) * csa]);  \
    c60 = _mm512_fmadd_pd(a0_, b0_, c60);              \
    c61 = _mm512_fmadd_pd(a0_, b1_, c61);              \
    c70 = _mm512_fmadd_pd(a1_, b0_, c70);              \
    c71 = _mm512_fmadd_pd(a1_, b1_, c71);              \
    a0_ = _mm512_set1_pd(a[8 * rsa + (p_off) * csa]);  \
    a1_ = _mm512_set1_pd(a[9 * rsa + (p_off) * csa]);  \
    c80 = _mm512_fmadd_pd(a0_, b0_, c80);              \
    c81 = _mm512_fmadd_pd(a0_, b1_, c81);              \
    c90 = _mm512_fmadd_pd(a1_, b0_, c90);              \
    c91 = _mm512_fmadd_pd(a1_, b1_, c91);              \
    a0_ = _mm512_set1_pd(a[10 * rsa + (p_off) * csa]); \
    a1_ = _mm512_set1_pd(a[11 * rsa + (p_off) * csa]); \
    cA0 = _mm512_fmadd_pd(a0_, b0_, cA0);              \
    cA1 = _mm512_fmadd_pd(a0_, b1_, cA1);              \
    cB0 = _mm512_fmadd_pd(a1_, b0_, cB0);              \
    cB1 = _mm512_fmadd_pd(a1_, b1_, cB1);              \
    a0_ = _mm512_set1_pd(a[12 * rsa + (p_off) * csa]); \
    a1_ = _mm512_set1_pd(a[13 * rsa + (p_off) * csa]); \
    cC0 = _mm512_fmadd_pd(a0_, b0_, cC0);              \
    cC1 = _mm512_fmadd_pd(a0_, b1_, cC1);              \
    cD0 = _mm512_fmadd_pd(a1_, b0_, cD0);              \
    cD1 = _mm512_fmadd_pd(a1_, b1_, cD1);              \
  } while (0)

  /* 8x K-loop unroll */
  size_t k_iter = kc / 8;
  size_t k_left = kc % 8;
  size_t p = 0;
  for (size_t ki = 0; ki < k_iter; ki++, p += 8) {
    GEMMSUP512_F64_K_BODY(p);
    GEMMSUP512_F64_K_BODY(p + 1);
    GEMMSUP512_F64_K_BODY(p + 2);
    GEMMSUP512_F64_K_BODY(p + 3);
    GEMMSUP512_F64_K_BODY(p + 4);
    GEMMSUP512_F64_K_BODY(p + 5);
    GEMMSUP512_F64_K_BODY(p + 6);
    GEMMSUP512_F64_K_BODY(p + 7);
  }
  for (size_t pi = 0; pi < k_left; pi++, p++) {
    GEMMSUP512_F64_K_BODY(p);
  }
#undef GEMMSUP512_F64_K_BODY

  _mm512_storeu_pd(c, c00);
  _mm512_storeu_pd(c + 8, c01);
  _mm512_storeu_pd(c + rso, c10);
  _mm512_storeu_pd(c + rso + 8, c11);
  _mm512_storeu_pd(c + 2 * rso, c20);
  _mm512_storeu_pd(c + 2 * rso + 8, c21);
  _mm512_storeu_pd(c + 3 * rso, c30);
  _mm512_storeu_pd(c + 3 * rso + 8, c31);
  _mm512_storeu_pd(c + 4 * rso, c40);
  _mm512_storeu_pd(c + 4 * rso + 8, c41);
  _mm512_storeu_pd(c + 5 * rso, c50);
  _mm512_storeu_pd(c + 5 * rso + 8, c51);
  _mm512_storeu_pd(c + 6 * rso, c60);
  _mm512_storeu_pd(c + 6 * rso + 8, c61);
  _mm512_storeu_pd(c + 7 * rso, c70);
  _mm512_storeu_pd(c + 7 * rso + 8, c71);
  _mm512_storeu_pd(c + 8 * rso, c80);
  _mm512_storeu_pd(c + 8 * rso + 8, c81);
  _mm512_storeu_pd(c + 9 * rso, c90);
  _mm512_storeu_pd(c + 9 * rso + 8, c91);
  _mm512_storeu_pd(c + 10 * rso, cA0);
  _mm512_storeu_pd(c + 10 * rso + 8, cA1);
  _mm512_storeu_pd(c + 11 * rso, cB0);
  _mm512_storeu_pd(c + 11 * rso + 8, cB1);
  _mm512_storeu_pd(c + 12 * rso, cC0);
  _mm512_storeu_pd(c + 12 * rso + 8, cC1);
  _mm512_storeu_pd(c + 13 * rso, cD0);
  _mm512_storeu_pd(c + 13 * rso + 8, cD1);
}

static inline void gemmsup_edge_f64_avx512(const double *a, const double *b,
                                           double *c, size_t mr, size_t nr,
                                           size_t kc, intptr_t rsa,
                                           intptr_t csa, intptr_t rsb,
                                           intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
    for (size_t j = 0; j < nr; j++)
      c[i * rso + j] = 0.0;
    for (size_t p = 0; p < kc; p++) {
      double aip = a[i * rsa + p * csa];
      __m512d va = _mm512_set1_pd(aip);
      const double *brow = b + p * rsb;
      double *crow = c + i * rso;
      size_t j = 0;
      for (; j + 8 <= nr; j += 8) {
        __m512d vo = _mm512_loadu_pd(crow + j);
        vo = _mm512_fmadd_pd(va, _mm512_loadu_pd(brow + j), vo);
        _mm512_storeu_pd(crow + j, vo);
      }
      for (; j < nr; j++)
        crow[j] += aip * brow[j];
    }
  }
}

/* OMP-parallel over MR-row tiles when compute volume warrants it. */
static inline void gemmsup_f64_avx512(const double *a, const double *b,
                                      double *out, size_t M, size_t K, size_t N,
                                      intptr_t rsa, intptr_t csa, intptr_t rsb,
                                      intptr_t rso) {
  const size_t MR = 14, NR = 16;
  size_t n_ir = (M + MR - 1) / MR;
#pragma omp parallel for schedule(static) if ((uint64_t)M * K * N > \
                                                  GEMMSUP512_OMP_THRESHOLD)
  for (size_t ir = 0; ir < n_ir; ir++) {
    size_t i = ir * MR;
    size_t mr = GEMMSUP512_MIN(MR, M - i);
    for (size_t j = 0; j < N; j += NR) {
      size_t nr = GEMMSUP512_MIN(NR, N - j);
      if (mr == MR && nr == NR) {
        gemmsup_ukernel_f64_14x16(a + i * rsa, b + j, out + i * rso + j, K, rsa,
                                  csa, rsb, rso);
      } else {
        gemmsup_edge_f64_avx512(a + i * rsa, b + j, out + i * rso + j, mr, nr,
                                K, rsa, csa, rsb, rso);
      }
    }
  }
}

#undef GEMMSUP512_MIN

#endif
