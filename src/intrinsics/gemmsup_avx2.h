#ifndef NUMC_GEMMSUP_AVX2_H
#define NUMC_GEMMSUP_AVX2_H

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define GEMMSUP_MIN(a, b) ((a) < (b) ? (a) : (b))

/* OMP gate for gemmsup: parallelize from ~128^3 and up. */
#define GEMMSUP_OMP_THRESHOLD (1ULL << 20)

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

  /* 2 alternating A broadcast variables (a0_/a1_) for ILP. */
#define GEMMSUP_F32_K_BODY(p_off)                           \
  do {                                                      \
    const float *bp_ = b + (p_off) * rsb;                   \
    __m256 b0_ = _mm256_loadu_ps(bp_);                      \
    __m256 b1_ = _mm256_loadu_ps(bp_ + 8);                  \
    __m256 a0_, a1_;                                        \
    a0_ = _mm256_broadcast_ss(a + 0 * rsa + (p_off) * csa); \
    a1_ = _mm256_broadcast_ss(a + 1 * rsa + (p_off) * csa); \
    c00 = _mm256_fmadd_ps(a0_, b0_, c00);                   \
    c01 = _mm256_fmadd_ps(a0_, b1_, c01);                   \
    c10 = _mm256_fmadd_ps(a1_, b0_, c10);                   \
    c11 = _mm256_fmadd_ps(a1_, b1_, c11);                   \
    a0_ = _mm256_broadcast_ss(a + 2 * rsa + (p_off) * csa); \
    a1_ = _mm256_broadcast_ss(a + 3 * rsa + (p_off) * csa); \
    c20 = _mm256_fmadd_ps(a0_, b0_, c20);                   \
    c21 = _mm256_fmadd_ps(a0_, b1_, c21);                   \
    c30 = _mm256_fmadd_ps(a1_, b0_, c30);                   \
    c31 = _mm256_fmadd_ps(a1_, b1_, c31);                   \
    a0_ = _mm256_broadcast_ss(a + 4 * rsa + (p_off) * csa); \
    a1_ = _mm256_broadcast_ss(a + 5 * rsa + (p_off) * csa); \
    c40 = _mm256_fmadd_ps(a0_, b0_, c40);                   \
    c41 = _mm256_fmadd_ps(a0_, b1_, c41);                   \
    c50 = _mm256_fmadd_ps(a1_, b0_, c50);                   \
    c51 = _mm256_fmadd_ps(a1_, b1_, c51);                   \
  } while (0)

  /* 8× K-loop unroll */
  size_t k_iter = kc / 8;
  size_t k_left = kc % 8;
  size_t p = 0;
  for (size_t ki = 0; ki < k_iter; ki++, p += 8) {
    GEMMSUP_F32_K_BODY(p);
    GEMMSUP_F32_K_BODY(p + 1);
    GEMMSUP_F32_K_BODY(p + 2);
    GEMMSUP_F32_K_BODY(p + 3);
    GEMMSUP_F32_K_BODY(p + 4);
    GEMMSUP_F32_K_BODY(p + 5);
    GEMMSUP_F32_K_BODY(p + 6);
    GEMMSUP_F32_K_BODY(p + 7);
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

/* Full gemmsup dispatch: tile MxN in 6x16 blocks, no packing.
 * OMP-parallel over MR-row tiles when compute volume warrants it. */
static inline void gemmsup_f32_avx2(const float *a, const float *b, float *out,
                                    size_t M, size_t K, size_t N, intptr_t rsa,
                                    intptr_t csa, intptr_t rsb, intptr_t rso) {
  const size_t MR = 6, NR = 16;
  size_t n_ir = (M + MR - 1) / MR;
#pragma omp parallel for schedule(static) if ((uint64_t)M * K * N > \
                                                  GEMMSUP_OMP_THRESHOLD)
  for (size_t ir = 0; ir < n_ir; ir++) {
    size_t i = ir * MR;
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

  /* 2 alternating A broadcast variables (a0_/a1_) for ILP. */
#define GEMMSUP_F64_K_BODY(p_off)                           \
  do {                                                      \
    const double *bp_ = b + (p_off) * rsb;                  \
    __m256d b0_ = _mm256_loadu_pd(bp_);                     \
    __m256d b1_ = _mm256_loadu_pd(bp_ + 4);                 \
    __m256d a0_, a1_;                                       \
    a0_ = _mm256_broadcast_sd(a + 0 * rsa + (p_off) * csa); \
    a1_ = _mm256_broadcast_sd(a + 1 * rsa + (p_off) * csa); \
    c00 = _mm256_fmadd_pd(a0_, b0_, c00);                   \
    c01 = _mm256_fmadd_pd(a0_, b1_, c01);                   \
    c10 = _mm256_fmadd_pd(a1_, b0_, c10);                   \
    c11 = _mm256_fmadd_pd(a1_, b1_, c11);                   \
    a0_ = _mm256_broadcast_sd(a + 2 * rsa + (p_off) * csa); \
    a1_ = _mm256_broadcast_sd(a + 3 * rsa + (p_off) * csa); \
    c20 = _mm256_fmadd_pd(a0_, b0_, c20);                   \
    c21 = _mm256_fmadd_pd(a0_, b1_, c21);                   \
    c30 = _mm256_fmadd_pd(a1_, b0_, c30);                   \
    c31 = _mm256_fmadd_pd(a1_, b1_, c31);                   \
    a0_ = _mm256_broadcast_sd(a + 4 * rsa + (p_off) * csa); \
    a1_ = _mm256_broadcast_sd(a + 5 * rsa + (p_off) * csa); \
    c40 = _mm256_fmadd_pd(a0_, b0_, c40);                   \
    c41 = _mm256_fmadd_pd(a0_, b1_, c41);                   \
    c50 = _mm256_fmadd_pd(a1_, b0_, c50);                   \
    c51 = _mm256_fmadd_pd(a1_, b1_, c51);                   \
  } while (0)

  /* 8× K-loop unroll */
  size_t k_iter = kc / 8;
  size_t k_left = kc % 8;
  size_t p = 0;
  for (size_t ki = 0; ki < k_iter; ki++, p += 8) {
    GEMMSUP_F64_K_BODY(p);
    GEMMSUP_F64_K_BODY(p + 1);
    GEMMSUP_F64_K_BODY(p + 2);
    GEMMSUP_F64_K_BODY(p + 3);
    GEMMSUP_F64_K_BODY(p + 4);
    GEMMSUP_F64_K_BODY(p + 5);
    GEMMSUP_F64_K_BODY(p + 6);
    GEMMSUP_F64_K_BODY(p + 7);
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

/* OMP-parallel over MR-row tiles when compute volume warrants it. */
static inline void gemmsup_f64_avx2(const double *a, const double *b,
                                    double *out, size_t M, size_t K, size_t N,
                                    intptr_t rsa, intptr_t csa, intptr_t rsb,
                                    intptr_t rso) {
  const size_t MR = 6, NR = 8;
  size_t n_ir = (M + MR - 1) / MR;
#pragma omp parallel for schedule(static) if ((uint64_t)M * K * N > \
                                                  GEMMSUP_OMP_THRESHOLD)
  for (size_t ir = 0; ir < n_ir; ir++) {
    size_t i = ir * MR;
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

/* =================================================================
   Int32 unpacked 6x16 micro-kernel
   Same tile as f32 but uses vpmulld + vpaddd instead of FMA.
   ================================================================= */

static inline void gemmsup_ukernel_i32_6x16(const int32_t *a, const int32_t *b,
                                            int32_t *c, size_t kc, intptr_t rsa,
                                            intptr_t csa, intptr_t rsb,
                                            intptr_t rso) {
  __m256i c00 = _mm256_setzero_si256(), c01 = _mm256_setzero_si256();
  __m256i c10 = _mm256_setzero_si256(), c11 = _mm256_setzero_si256();
  __m256i c20 = _mm256_setzero_si256(), c21 = _mm256_setzero_si256();
  __m256i c30 = _mm256_setzero_si256(), c31 = _mm256_setzero_si256();
  __m256i c40 = _mm256_setzero_si256(), c41 = _mm256_setzero_si256();
  __m256i c50 = _mm256_setzero_si256(), c51 = _mm256_setzero_si256();

#define GEMMSUP_I32_K_BODY(p_off)                                             \
  do {                                                                         \
    const int32_t *bp_ = b + (p_off) * rsb;                                    \
    __m256i b0_ = _mm256_loadu_si256((const __m256i *)bp_);                    \
    __m256i b1_ = _mm256_loadu_si256((const __m256i *)(bp_ + 8));              \
    __m256i a0_, a1_;                                                          \
    a0_ = _mm256_set1_epi32(a[0 * rsa + (p_off) * csa]);                      \
    a1_ = _mm256_set1_epi32(a[1 * rsa + (p_off) * csa]);                      \
    c00 = _mm256_add_epi32(c00, _mm256_mullo_epi32(a0_, b0_));                \
    c01 = _mm256_add_epi32(c01, _mm256_mullo_epi32(a0_, b1_));                \
    c10 = _mm256_add_epi32(c10, _mm256_mullo_epi32(a1_, b0_));                \
    c11 = _mm256_add_epi32(c11, _mm256_mullo_epi32(a1_, b1_));                \
    a0_ = _mm256_set1_epi32(a[2 * rsa + (p_off) * csa]);                      \
    a1_ = _mm256_set1_epi32(a[3 * rsa + (p_off) * csa]);                      \
    c20 = _mm256_add_epi32(c20, _mm256_mullo_epi32(a0_, b0_));                \
    c21 = _mm256_add_epi32(c21, _mm256_mullo_epi32(a0_, b1_));                \
    c30 = _mm256_add_epi32(c30, _mm256_mullo_epi32(a1_, b0_));                \
    c31 = _mm256_add_epi32(c31, _mm256_mullo_epi32(a1_, b1_));                \
    a0_ = _mm256_set1_epi32(a[4 * rsa + (p_off) * csa]);                      \
    a1_ = _mm256_set1_epi32(a[5 * rsa + (p_off) * csa]);                      \
    c40 = _mm256_add_epi32(c40, _mm256_mullo_epi32(a0_, b0_));                \
    c41 = _mm256_add_epi32(c41, _mm256_mullo_epi32(a0_, b1_));                \
    c50 = _mm256_add_epi32(c50, _mm256_mullo_epi32(a1_, b0_));                \
    c51 = _mm256_add_epi32(c51, _mm256_mullo_epi32(a1_, b1_));                \
  } while (0)

  /* 4x K-loop unroll */
  size_t k_iter = kc / 4;
  size_t k_left = kc % 4;
  size_t p = 0;
  for (size_t ki = 0; ki < k_iter; ki++, p += 4) {
    GEMMSUP_I32_K_BODY(p);
    GEMMSUP_I32_K_BODY(p + 1);
    GEMMSUP_I32_K_BODY(p + 2);
    GEMMSUP_I32_K_BODY(p + 3);
  }
  for (size_t pi = 0; pi < k_left; pi++, p++) {
    GEMMSUP_I32_K_BODY(p);
  }
#undef GEMMSUP_I32_K_BODY

  _mm256_storeu_si256((__m256i *)c, c00);
  _mm256_storeu_si256((__m256i *)(c + 8), c01);
  _mm256_storeu_si256((__m256i *)(c + rso), c10);
  _mm256_storeu_si256((__m256i *)(c + rso + 8), c11);
  _mm256_storeu_si256((__m256i *)(c + 2 * rso), c20);
  _mm256_storeu_si256((__m256i *)(c + 2 * rso + 8), c21);
  _mm256_storeu_si256((__m256i *)(c + 3 * rso), c30);
  _mm256_storeu_si256((__m256i *)(c + 3 * rso + 8), c31);
  _mm256_storeu_si256((__m256i *)(c + 4 * rso), c40);
  _mm256_storeu_si256((__m256i *)(c + 4 * rso + 8), c41);
  _mm256_storeu_si256((__m256i *)(c + 5 * rso), c50);
  _mm256_storeu_si256((__m256i *)(c + 5 * rso + 8), c51);
}

/* Scalar edge kernel for remainder tiles */
static inline void gemmsup_edge_i32(const int32_t *a, const int32_t *b,
                                    int32_t *c, size_t mr, size_t nr, size_t kc,
                                    intptr_t rsa, intptr_t csa, intptr_t rsb,
                                    intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
    for (size_t j = 0; j < nr; j++)
      c[i * rso + j] = 0;
    for (size_t p = 0; p < kc; p++) {
      int32_t aip = a[i * rsa + p * csa];
      __m256i va = _mm256_set1_epi32(aip);
      const int32_t *brow = b + p * rsb;
      int32_t *crow = c + i * rso;
      size_t j = 0;
      for (; j + 8 <= nr; j += 8) {
        __m256i vo = _mm256_loadu_si256((const __m256i *)(crow + j));
        vo = _mm256_add_epi32(vo, _mm256_mullo_epi32(
                                      va, _mm256_loadu_si256(
                                              (const __m256i *)(brow + j))));
        _mm256_storeu_si256((__m256i *)(crow + j), vo);
      }
      for (; j < nr; j++)
        crow[j] += aip * brow[j];
    }
  }
}

/* OMP-parallel over MR-row tiles when compute volume warrants it. */
static inline void gemmsup_i32_avx2(const int32_t *a, const int32_t *b,
                                    int32_t *out, size_t M, size_t K, size_t N,
                                    intptr_t rsa, intptr_t csa, intptr_t rsb,
                                    intptr_t rso) {
  const size_t MR = 6, NR = 16;
  size_t n_ir = (M + MR - 1) / MR;
#pragma omp parallel for schedule(static) if ((uint64_t)M * K * N > \
                                                  GEMMSUP_OMP_THRESHOLD)
  for (size_t ir = 0; ir < n_ir; ir++) {
    size_t i = ir * MR;
    size_t mr = GEMMSUP_MIN(MR, M - i);
    for (size_t j = 0; j < N; j += NR) {
      size_t nr = GEMMSUP_MIN(NR, N - j);
      if (mr == MR && nr == NR) {
        gemmsup_ukernel_i32_6x16(a + i * rsa, b + j, out + i * rso + j, K, rsa,
                                 csa, rsb, rso);
      } else {
        gemmsup_edge_i32(a + i * rsa, b + j, out + i * rso + j, mr, nr, K, rsa,
                         csa, rsb, rso);
      }
    }
  }
}

/* u32 wrapper: vpmulld is sign-agnostic for truncated multiply. */
static inline void gemmsup_u32_avx2(const uint32_t *a, const uint32_t *b,
                                    uint32_t *out, size_t M, size_t K, size_t N,
                                    intptr_t rsa, intptr_t csa, intptr_t rsb,
                                    intptr_t rso) {
  gemmsup_i32_avx2((const int32_t *)a, (const int32_t *)b, (int32_t *)out, M, K,
                   N, rsa, csa, rsb, rso);
}

/* =================================================================
   Int16 unpacked 6x32 micro-kernel
   Each YMM holds 16 i16 elements, so NR=32 needs 2 YMM loads per
   B-row.  12 accumulators + 2 B loads + 2 A broadcasts = 16 YMM.
   Uses vpmullw (sign-agnostic truncated multiply) + vpaddw.
   ================================================================= */

static inline void gemmsup_ukernel_i16_6x32(const int16_t *a, const int16_t *b,
                                            int16_t *c, size_t kc, intptr_t rsa,
                                            intptr_t csa, intptr_t rsb,
                                            intptr_t rso) {
  __m256i c00 = _mm256_setzero_si256(), c01 = _mm256_setzero_si256();
  __m256i c10 = _mm256_setzero_si256(), c11 = _mm256_setzero_si256();
  __m256i c20 = _mm256_setzero_si256(), c21 = _mm256_setzero_si256();
  __m256i c30 = _mm256_setzero_si256(), c31 = _mm256_setzero_si256();
  __m256i c40 = _mm256_setzero_si256(), c41 = _mm256_setzero_si256();
  __m256i c50 = _mm256_setzero_si256(), c51 = _mm256_setzero_si256();

#define GEMMSUP_I16_K_BODY(p_off)                                             \
  do {                                                                         \
    const int16_t *bp_ = b + (p_off) * rsb;                                    \
    __m256i b0_ = _mm256_loadu_si256((const __m256i *)bp_);                    \
    __m256i b1_ = _mm256_loadu_si256((const __m256i *)(bp_ + 16));             \
    __m256i a0_, a1_;                                                          \
    a0_ = _mm256_set1_epi16(a[0 * rsa + (p_off) * csa]);                      \
    a1_ = _mm256_set1_epi16(a[1 * rsa + (p_off) * csa]);                      \
    c00 = _mm256_add_epi16(c00, _mm256_mullo_epi16(a0_, b0_));                \
    c01 = _mm256_add_epi16(c01, _mm256_mullo_epi16(a0_, b1_));                \
    c10 = _mm256_add_epi16(c10, _mm256_mullo_epi16(a1_, b0_));                \
    c11 = _mm256_add_epi16(c11, _mm256_mullo_epi16(a1_, b1_));                \
    a0_ = _mm256_set1_epi16(a[2 * rsa + (p_off) * csa]);                      \
    a1_ = _mm256_set1_epi16(a[3 * rsa + (p_off) * csa]);                      \
    c20 = _mm256_add_epi16(c20, _mm256_mullo_epi16(a0_, b0_));                \
    c21 = _mm256_add_epi16(c21, _mm256_mullo_epi16(a0_, b1_));                \
    c30 = _mm256_add_epi16(c30, _mm256_mullo_epi16(a1_, b0_));                \
    c31 = _mm256_add_epi16(c31, _mm256_mullo_epi16(a1_, b1_));                \
    a0_ = _mm256_set1_epi16(a[4 * rsa + (p_off) * csa]);                      \
    a1_ = _mm256_set1_epi16(a[5 * rsa + (p_off) * csa]);                      \
    c40 = _mm256_add_epi16(c40, _mm256_mullo_epi16(a0_, b0_));                \
    c41 = _mm256_add_epi16(c41, _mm256_mullo_epi16(a0_, b1_));                \
    c50 = _mm256_add_epi16(c50, _mm256_mullo_epi16(a1_, b0_));                \
    c51 = _mm256_add_epi16(c51, _mm256_mullo_epi16(a1_, b1_));                \
  } while (0)

  /* 4x K-loop unroll */
  size_t k_iter = kc / 4;
  size_t k_left = kc % 4;
  size_t p = 0;
  for (size_t ki = 0; ki < k_iter; ki++, p += 4) {
    GEMMSUP_I16_K_BODY(p);
    GEMMSUP_I16_K_BODY(p + 1);
    GEMMSUP_I16_K_BODY(p + 2);
    GEMMSUP_I16_K_BODY(p + 3);
  }
  for (size_t pi = 0; pi < k_left; pi++, p++) {
    GEMMSUP_I16_K_BODY(p);
  }
#undef GEMMSUP_I16_K_BODY

  _mm256_storeu_si256((__m256i *)c, c00);
  _mm256_storeu_si256((__m256i *)(c + 16), c01);
  _mm256_storeu_si256((__m256i *)(c + rso), c10);
  _mm256_storeu_si256((__m256i *)(c + rso + 16), c11);
  _mm256_storeu_si256((__m256i *)(c + 2 * rso), c20);
  _mm256_storeu_si256((__m256i *)(c + 2 * rso + 16), c21);
  _mm256_storeu_si256((__m256i *)(c + 3 * rso), c30);
  _mm256_storeu_si256((__m256i *)(c + 3 * rso + 16), c31);
  _mm256_storeu_si256((__m256i *)(c + 4 * rso), c40);
  _mm256_storeu_si256((__m256i *)(c + 4 * rso + 16), c41);
  _mm256_storeu_si256((__m256i *)(c + 5 * rso), c50);
  _mm256_storeu_si256((__m256i *)(c + 5 * rso + 16), c51);
}

/* Scalar edge kernel for remainder tiles */
static inline void gemmsup_edge_i16(const int16_t *a, const int16_t *b,
                                    int16_t *c, size_t mr, size_t nr, size_t kc,
                                    intptr_t rsa, intptr_t csa, intptr_t rsb,
                                    intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
    for (size_t j = 0; j < nr; j++)
      c[i * rso + j] = 0;
    for (size_t p = 0; p < kc; p++) {
      int16_t aip = a[i * rsa + p * csa];
      __m256i va = _mm256_set1_epi16(aip);
      const int16_t *brow = b + p * rsb;
      int16_t *crow = c + i * rso;
      size_t j = 0;
      for (; j + 16 <= nr; j += 16) {
        __m256i vo = _mm256_loadu_si256((const __m256i *)(crow + j));
        vo = _mm256_add_epi16(vo, _mm256_mullo_epi16(
                                      va, _mm256_loadu_si256(
                                              (const __m256i *)(brow + j))));
        _mm256_storeu_si256((__m256i *)(crow + j), vo);
      }
      for (; j < nr; j++)
        crow[j] = (int16_t)(crow[j] + aip * brow[j]);
    }
  }
}

/* OMP-parallel over MR-row tiles when compute volume warrants it. */
static inline void gemmsup_i16_avx2(const int16_t *a, const int16_t *b,
                                    int16_t *out, size_t M, size_t K, size_t N,
                                    intptr_t rsa, intptr_t csa, intptr_t rsb,
                                    intptr_t rso) {
  const size_t MR = 6, NR = 32;
  size_t n_ir = (M + MR - 1) / MR;
#pragma omp parallel for schedule(static) if ((uint64_t)M * K * N > \
                                                  GEMMSUP_OMP_THRESHOLD)
  for (size_t ir = 0; ir < n_ir; ir++) {
    size_t i = ir * MR;
    size_t mr = GEMMSUP_MIN(MR, M - i);
    for (size_t j = 0; j < N; j += NR) {
      size_t nr = GEMMSUP_MIN(NR, N - j);
      if (mr == MR && nr == NR) {
        gemmsup_ukernel_i16_6x32(a + i * rsa, b + j, out + i * rso + j, K, rsa,
                                 csa, rsb, rso);
      } else {
        gemmsup_edge_i16(a + i * rsa, b + j, out + i * rso + j, mr, nr, K, rsa,
                         csa, rsb, rso);
      }
    }
  }
}

/* u16 wrapper: vpmullw is sign-agnostic for truncated multiply. */
static inline void gemmsup_u16_avx2(const uint16_t *a, const uint16_t *b,
                                    uint16_t *out, size_t M, size_t K, size_t N,
                                    intptr_t rsa, intptr_t csa, intptr_t rsb,
                                    intptr_t rso) {
  gemmsup_i16_avx2((const int16_t *)a, (const int16_t *)b, (int16_t *)out, M, K,
                   N, rsa, csa, rsb, rso);
}

/* =================================================================
   Int64 unpacked 6x8 micro-kernel
   Same tile as f64 (MR=6, NR=8).  AVX2 lacks vpmullq, so we use a
   widening-multiply helper that computes lower 64 bits via three
   vpmuludq operations.  Works for both signed and unsigned
   (two's complement truncation).
   ================================================================= */

/* 64-bit multiply helper (no vpmullq on AVX2).
 * Computes lower 64 bits of a*b using three vpmuludq operations.
 * Works for both signed and unsigned (two's complement truncation). */
static inline __m256i _mm256_mullo_epi64_avx2(__m256i a, __m256i b) {
  __m256i lo = _mm256_mul_epu32(a, b);
  __m256i a_hi = _mm256_srli_epi64(a, 32);
  __m256i b_hi = _mm256_srli_epi64(b, 32);
  __m256i hi1 = _mm256_mul_epu32(a_hi, b);
  __m256i hi2 = _mm256_mul_epu32(a, b_hi);
  __m256i hi_sum = _mm256_add_epi64(hi1, hi2);
  return _mm256_add_epi64(lo, _mm256_slli_epi64(hi_sum, 32));
}

static inline void gemmsup_ukernel_i64_6x8(const int64_t *a, const int64_t *b,
                                            int64_t *c, size_t kc, intptr_t rsa,
                                            intptr_t csa, intptr_t rsb,
                                            intptr_t rso) {
  __m256i c00 = _mm256_setzero_si256(), c01 = _mm256_setzero_si256();
  __m256i c10 = _mm256_setzero_si256(), c11 = _mm256_setzero_si256();
  __m256i c20 = _mm256_setzero_si256(), c21 = _mm256_setzero_si256();
  __m256i c30 = _mm256_setzero_si256(), c31 = _mm256_setzero_si256();
  __m256i c40 = _mm256_setzero_si256(), c41 = _mm256_setzero_si256();
  __m256i c50 = _mm256_setzero_si256(), c51 = _mm256_setzero_si256();

#define GEMMSUP_I64_K_BODY(p_off)                                             \
  do {                                                                         \
    const int64_t *bp_ = b + (p_off) * rsb;                                    \
    __m256i b0_ = _mm256_loadu_si256((const __m256i *)bp_);                    \
    __m256i b1_ = _mm256_loadu_si256((const __m256i *)(bp_ + 4));              \
    __m256i a0_, a1_;                                                          \
    a0_ = _mm256_set1_epi64x(a[0 * rsa + (p_off) * csa]);                     \
    a1_ = _mm256_set1_epi64x(a[1 * rsa + (p_off) * csa]);                     \
    c00 = _mm256_add_epi64(c00, _mm256_mullo_epi64_avx2(a0_, b0_));           \
    c01 = _mm256_add_epi64(c01, _mm256_mullo_epi64_avx2(a0_, b1_));           \
    c10 = _mm256_add_epi64(c10, _mm256_mullo_epi64_avx2(a1_, b0_));           \
    c11 = _mm256_add_epi64(c11, _mm256_mullo_epi64_avx2(a1_, b1_));           \
    a0_ = _mm256_set1_epi64x(a[2 * rsa + (p_off) * csa]);                     \
    a1_ = _mm256_set1_epi64x(a[3 * rsa + (p_off) * csa]);                     \
    c20 = _mm256_add_epi64(c20, _mm256_mullo_epi64_avx2(a0_, b0_));           \
    c21 = _mm256_add_epi64(c21, _mm256_mullo_epi64_avx2(a0_, b1_));           \
    c30 = _mm256_add_epi64(c30, _mm256_mullo_epi64_avx2(a1_, b0_));           \
    c31 = _mm256_add_epi64(c31, _mm256_mullo_epi64_avx2(a1_, b1_));           \
    a0_ = _mm256_set1_epi64x(a[4 * rsa + (p_off) * csa]);                     \
    a1_ = _mm256_set1_epi64x(a[5 * rsa + (p_off) * csa]);                     \
    c40 = _mm256_add_epi64(c40, _mm256_mullo_epi64_avx2(a0_, b0_));           \
    c41 = _mm256_add_epi64(c41, _mm256_mullo_epi64_avx2(a0_, b1_));           \
    c50 = _mm256_add_epi64(c50, _mm256_mullo_epi64_avx2(a1_, b0_));           \
    c51 = _mm256_add_epi64(c51, _mm256_mullo_epi64_avx2(a1_, b1_));           \
  } while (0)

  /* 2x K-loop unroll (each K-step is ~7 instructions due to widening mul) */
  size_t k_iter = kc / 2;
  size_t k_left = kc % 2;
  size_t p = 0;
  for (size_t ki = 0; ki < k_iter; ki++, p += 2) {
    GEMMSUP_I64_K_BODY(p);
    GEMMSUP_I64_K_BODY(p + 1);
  }
  for (size_t pi = 0; pi < k_left; pi++, p++) {
    GEMMSUP_I64_K_BODY(p);
  }
#undef GEMMSUP_I64_K_BODY

  _mm256_storeu_si256((__m256i *)c, c00);
  _mm256_storeu_si256((__m256i *)(c + 4), c01);
  _mm256_storeu_si256((__m256i *)(c + rso), c10);
  _mm256_storeu_si256((__m256i *)(c + rso + 4), c11);
  _mm256_storeu_si256((__m256i *)(c + 2 * rso), c20);
  _mm256_storeu_si256((__m256i *)(c + 2 * rso + 4), c21);
  _mm256_storeu_si256((__m256i *)(c + 3 * rso), c30);
  _mm256_storeu_si256((__m256i *)(c + 3 * rso + 4), c31);
  _mm256_storeu_si256((__m256i *)(c + 4 * rso), c40);
  _mm256_storeu_si256((__m256i *)(c + 4 * rso + 4), c41);
  _mm256_storeu_si256((__m256i *)(c + 5 * rso), c50);
  _mm256_storeu_si256((__m256i *)(c + 5 * rso + 4), c51);
}

/* Scalar edge kernel for remainder tiles */
static inline void gemmsup_edge_i64(const int64_t *a, const int64_t *b,
                                    int64_t *c, size_t mr, size_t nr, size_t kc,
                                    intptr_t rsa, intptr_t csa, intptr_t rsb,
                                    intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
    for (size_t j = 0; j < nr; j++)
      c[i * rso + j] = 0;
    for (size_t p = 0; p < kc; p++) {
      int64_t aip = a[i * rsa + p * csa];
      const int64_t *brow = b + p * rsb;
      int64_t *crow = c + i * rso;
      for (size_t j = 0; j < nr; j++)
        crow[j] += aip * brow[j];
    }
  }
}

/* OMP-parallel over MR-row tiles when compute volume warrants it. */
static inline void gemmsup_i64_avx2(const int64_t *a, const int64_t *b,
                                    int64_t *out, size_t M, size_t K, size_t N,
                                    intptr_t rsa, intptr_t csa, intptr_t rsb,
                                    intptr_t rso) {
  const size_t MR = 6, NR = 8;
  size_t n_ir = (M + MR - 1) / MR;
#pragma omp parallel for schedule(static) if ((uint64_t)M * K * N > \
                                                  GEMMSUP_OMP_THRESHOLD)
  for (size_t ir = 0; ir < n_ir; ir++) {
    size_t i = ir * MR;
    size_t mr = GEMMSUP_MIN(MR, M - i);
    for (size_t j = 0; j < N; j += NR) {
      size_t nr = GEMMSUP_MIN(NR, N - j);
      if (mr == MR && nr == NR) {
        gemmsup_ukernel_i64_6x8(a + i * rsa, b + j, out + i * rso + j, K, rsa,
                                csa, rsb, rso);
      } else {
        gemmsup_edge_i64(a + i * rsa, b + j, out + i * rso + j, mr, nr, K, rsa,
                         csa, rsb, rso);
      }
    }
  }
}

/* u64 wrapper: widening mul helper is sign-agnostic for truncated multiply. */
static inline void gemmsup_u64_avx2(const uint64_t *a, const uint64_t *b,
                                    uint64_t *out, size_t M, size_t K, size_t N,
                                    intptr_t rsa, intptr_t csa, intptr_t rsb,
                                    intptr_t rso) {
  gemmsup_i64_avx2((const int64_t *)a, (const int64_t *)b, (int64_t *)out, M, K,
                   N, rsa, csa, rsb, rso);
}

/* =================================================================
   Int8 unpacked 6x16 micro-kernel
   Accumulates in i32 to avoid overflow, stores back as (int8_t).
   Each B load converts 8 i8 -> 8 i32 via _mm256_cvtepi8_epi32.
   NR=16 output columns = 2 YMM vectors of 8 i32 each.
   12 accumulators (6 rows x 2 halves) + 2 B loads + 2 A bcast = 16 YMM.
   ================================================================= */

static inline __m256i _gemmsup_load_i8_to_i32(const int8_t *ptr) {
  return _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)ptr));
}

static inline void gemmsup_ukernel_i8_6x16(const int8_t *a, const int8_t *b,
                                            int8_t *c, size_t kc, intptr_t rsa,
                                            intptr_t csa, intptr_t rsb,
                                            intptr_t rso) {
  __m256i c00 = _mm256_setzero_si256(), c01 = _mm256_setzero_si256();
  __m256i c10 = _mm256_setzero_si256(), c11 = _mm256_setzero_si256();
  __m256i c20 = _mm256_setzero_si256(), c21 = _mm256_setzero_si256();
  __m256i c30 = _mm256_setzero_si256(), c31 = _mm256_setzero_si256();
  __m256i c40 = _mm256_setzero_si256(), c41 = _mm256_setzero_si256();
  __m256i c50 = _mm256_setzero_si256(), c51 = _mm256_setzero_si256();

#define GEMMSUP_I8_K_BODY(p_off)                                               \
  do {                                                                          \
    const int8_t *bp_ = b + (p_off) * rsb;                                     \
    __m256i b0_ = _gemmsup_load_i8_to_i32(bp_);                                \
    __m256i b1_ = _gemmsup_load_i8_to_i32(bp_ + 8);                            \
    __m256i a0_, a1_;                                                           \
    a0_ = _mm256_set1_epi32((int32_t)a[0 * rsa + (p_off) * csa]);              \
    a1_ = _mm256_set1_epi32((int32_t)a[1 * rsa + (p_off) * csa]);              \
    c00 = _mm256_add_epi32(c00, _mm256_mullo_epi32(a0_, b0_));                 \
    c01 = _mm256_add_epi32(c01, _mm256_mullo_epi32(a0_, b1_));                 \
    c10 = _mm256_add_epi32(c10, _mm256_mullo_epi32(a1_, b0_));                 \
    c11 = _mm256_add_epi32(c11, _mm256_mullo_epi32(a1_, b1_));                 \
    a0_ = _mm256_set1_epi32((int32_t)a[2 * rsa + (p_off) * csa]);              \
    a1_ = _mm256_set1_epi32((int32_t)a[3 * rsa + (p_off) * csa]);              \
    c20 = _mm256_add_epi32(c20, _mm256_mullo_epi32(a0_, b0_));                 \
    c21 = _mm256_add_epi32(c21, _mm256_mullo_epi32(a0_, b1_));                 \
    c30 = _mm256_add_epi32(c30, _mm256_mullo_epi32(a1_, b0_));                 \
    c31 = _mm256_add_epi32(c31, _mm256_mullo_epi32(a1_, b1_));                 \
    a0_ = _mm256_set1_epi32((int32_t)a[4 * rsa + (p_off) * csa]);              \
    a1_ = _mm256_set1_epi32((int32_t)a[5 * rsa + (p_off) * csa]);              \
    c40 = _mm256_add_epi32(c40, _mm256_mullo_epi32(a0_, b0_));                 \
    c41 = _mm256_add_epi32(c41, _mm256_mullo_epi32(a0_, b1_));                 \
    c50 = _mm256_add_epi32(c50, _mm256_mullo_epi32(a1_, b0_));                 \
    c51 = _mm256_add_epi32(c51, _mm256_mullo_epi32(a1_, b1_));                 \
  } while (0)

  /* 4x K-loop unroll */
  size_t k_iter = kc / 4;
  size_t k_left = kc % 4;
  size_t p = 0;
  for (size_t ki = 0; ki < k_iter; ki++, p += 4) {
    GEMMSUP_I8_K_BODY(p);
    GEMMSUP_I8_K_BODY(p + 1);
    GEMMSUP_I8_K_BODY(p + 2);
    GEMMSUP_I8_K_BODY(p + 3);
  }
  for (size_t pi = 0; pi < k_left; pi++, p++) {
    GEMMSUP_I8_K_BODY(p);
  }
#undef GEMMSUP_I8_K_BODY

  /* Store i32 accumulators to i8 output via temp buffer (truncation) */
  int32_t tmp[16];
#define GEMMSUP_I8_STORE_ROW(row, lo, hi)                                      \
  do {                                                                          \
    _mm256_storeu_si256((__m256i *)tmp, lo);                                    \
    _mm256_storeu_si256((__m256i *)(tmp + 8), hi);                             \
    for (size_t j = 0; j < 16; j++)                                            \
      c[(row) * rso + j] = (int8_t)tmp[j];                                    \
  } while (0)

  GEMMSUP_I8_STORE_ROW(0, c00, c01);
  GEMMSUP_I8_STORE_ROW(1, c10, c11);
  GEMMSUP_I8_STORE_ROW(2, c20, c21);
  GEMMSUP_I8_STORE_ROW(3, c30, c31);
  GEMMSUP_I8_STORE_ROW(4, c40, c41);
  GEMMSUP_I8_STORE_ROW(5, c50, c51);
#undef GEMMSUP_I8_STORE_ROW
}

/* Scalar edge kernel for i8 remainder tiles (accumulate in i32) */
static inline void gemmsup_edge_i8(const int8_t *a, const int8_t *b, int8_t *c,
                                   size_t mr, size_t nr, size_t kc,
                                   intptr_t rsa, intptr_t csa, intptr_t rsb,
                                   intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
    for (size_t j = 0; j < nr; j++) {
      int32_t acc = 0;
      for (size_t p = 0; p < kc; p++)
        acc += (int32_t)a[i * rsa + p * csa] * (int32_t)b[p * rsb + j];
      c[i * rso + j] = (int8_t)acc;
    }
  }
}

/* OMP-parallel over MR-row tiles when compute volume warrants it. */
static inline void gemmsup_i8_avx2(const int8_t *a, const int8_t *b,
                                   int8_t *out, size_t M, size_t K, size_t N,
                                   intptr_t rsa, intptr_t csa, intptr_t rsb,
                                   intptr_t rso) {
  const size_t MR = 6, NR = 16;
  size_t n_ir = (M + MR - 1) / MR;
#pragma omp parallel for schedule(static) if ((uint64_t)M * K * N > \
                                                  GEMMSUP_OMP_THRESHOLD)
  for (size_t ir = 0; ir < n_ir; ir++) {
    size_t i = ir * MR;
    size_t mr = GEMMSUP_MIN(MR, M - i);
    for (size_t j = 0; j < N; j += NR) {
      size_t nr = GEMMSUP_MIN(NR, N - j);
      if (mr == MR && nr == NR) {
        gemmsup_ukernel_i8_6x16(a + i * rsa, b + j, out + i * rso + j, K, rsa,
                                csa, rsb, rso);
      } else {
        gemmsup_edge_i8(a + i * rsa, b + j, out + i * rso + j, mr, nr, K, rsa,
                        csa, rsb, rso);
      }
    }
  }
}

/* =================================================================
   UInt8 unpacked 6x16 micro-kernel
   Same structure as i8 but uses _mm256_cvtepu8_epi32 for unsigned
   extension, and stores as (uint8_t).
   ================================================================= */

static inline __m256i _gemmsup_load_u8_to_i32(const uint8_t *ptr) {
  return _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)ptr));
}

static inline void gemmsup_ukernel_u8_6x16(const uint8_t *a, const uint8_t *b,
                                            uint8_t *c, size_t kc, intptr_t rsa,
                                            intptr_t csa, intptr_t rsb,
                                            intptr_t rso) {
  __m256i c00 = _mm256_setzero_si256(), c01 = _mm256_setzero_si256();
  __m256i c10 = _mm256_setzero_si256(), c11 = _mm256_setzero_si256();
  __m256i c20 = _mm256_setzero_si256(), c21 = _mm256_setzero_si256();
  __m256i c30 = _mm256_setzero_si256(), c31 = _mm256_setzero_si256();
  __m256i c40 = _mm256_setzero_si256(), c41 = _mm256_setzero_si256();
  __m256i c50 = _mm256_setzero_si256(), c51 = _mm256_setzero_si256();

#define GEMMSUP_U8_K_BODY(p_off)                                               \
  do {                                                                          \
    const uint8_t *bp_ = b + (p_off) * rsb;                                    \
    __m256i b0_ = _gemmsup_load_u8_to_i32(bp_);                                \
    __m256i b1_ = _gemmsup_load_u8_to_i32(bp_ + 8);                            \
    __m256i a0_, a1_;                                                           \
    a0_ = _mm256_set1_epi32((int32_t)(uint32_t)a[0 * rsa + (p_off) * csa]);    \
    a1_ = _mm256_set1_epi32((int32_t)(uint32_t)a[1 * rsa + (p_off) * csa]);    \
    c00 = _mm256_add_epi32(c00, _mm256_mullo_epi32(a0_, b0_));                 \
    c01 = _mm256_add_epi32(c01, _mm256_mullo_epi32(a0_, b1_));                 \
    c10 = _mm256_add_epi32(c10, _mm256_mullo_epi32(a1_, b0_));                 \
    c11 = _mm256_add_epi32(c11, _mm256_mullo_epi32(a1_, b1_));                 \
    a0_ = _mm256_set1_epi32((int32_t)(uint32_t)a[2 * rsa + (p_off) * csa]);    \
    a1_ = _mm256_set1_epi32((int32_t)(uint32_t)a[3 * rsa + (p_off) * csa]);    \
    c20 = _mm256_add_epi32(c20, _mm256_mullo_epi32(a0_, b0_));                 \
    c21 = _mm256_add_epi32(c21, _mm256_mullo_epi32(a0_, b1_));                 \
    c30 = _mm256_add_epi32(c30, _mm256_mullo_epi32(a1_, b0_));                 \
    c31 = _mm256_add_epi32(c31, _mm256_mullo_epi32(a1_, b1_));                 \
    a0_ = _mm256_set1_epi32((int32_t)(uint32_t)a[4 * rsa + (p_off) * csa]);    \
    a1_ = _mm256_set1_epi32((int32_t)(uint32_t)a[5 * rsa + (p_off) * csa]);    \
    c40 = _mm256_add_epi32(c40, _mm256_mullo_epi32(a0_, b0_));                 \
    c41 = _mm256_add_epi32(c41, _mm256_mullo_epi32(a0_, b1_));                 \
    c50 = _mm256_add_epi32(c50, _mm256_mullo_epi32(a1_, b0_));                 \
    c51 = _mm256_add_epi32(c51, _mm256_mullo_epi32(a1_, b1_));                 \
  } while (0)

  /* 4x K-loop unroll */
  size_t k_iter = kc / 4;
  size_t k_left = kc % 4;
  size_t p = 0;
  for (size_t ki = 0; ki < k_iter; ki++, p += 4) {
    GEMMSUP_U8_K_BODY(p);
    GEMMSUP_U8_K_BODY(p + 1);
    GEMMSUP_U8_K_BODY(p + 2);
    GEMMSUP_U8_K_BODY(p + 3);
  }
  for (size_t pi = 0; pi < k_left; pi++, p++) {
    GEMMSUP_U8_K_BODY(p);
  }
#undef GEMMSUP_U8_K_BODY

  /* Store i32 accumulators to u8 output via temp buffer (truncation) */
  int32_t tmp[16];
#define GEMMSUP_U8_STORE_ROW(row, lo, hi)                                      \
  do {                                                                          \
    _mm256_storeu_si256((__m256i *)tmp, lo);                                    \
    _mm256_storeu_si256((__m256i *)(tmp + 8), hi);                             \
    for (size_t j = 0; j < 16; j++)                                            \
      c[(row) * rso + j] = (uint8_t)tmp[j];                                   \
  } while (0)

  GEMMSUP_U8_STORE_ROW(0, c00, c01);
  GEMMSUP_U8_STORE_ROW(1, c10, c11);
  GEMMSUP_U8_STORE_ROW(2, c20, c21);
  GEMMSUP_U8_STORE_ROW(3, c30, c31);
  GEMMSUP_U8_STORE_ROW(4, c40, c41);
  GEMMSUP_U8_STORE_ROW(5, c50, c51);
#undef GEMMSUP_U8_STORE_ROW
}

/* Scalar edge kernel for u8 remainder tiles (accumulate in i32) */
static inline void gemmsup_edge_u8(const uint8_t *a, const uint8_t *b,
                                   uint8_t *c, size_t mr, size_t nr, size_t kc,
                                   intptr_t rsa, intptr_t csa, intptr_t rsb,
                                   intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
    for (size_t j = 0; j < nr; j++) {
      int32_t acc = 0;
      for (size_t p = 0; p < kc; p++)
        acc += (int32_t)(uint32_t)a[i * rsa + p * csa] *
               (int32_t)(uint32_t)b[p * rsb + j];
      c[i * rso + j] = (uint8_t)acc;
    }
  }
}

/* OMP-parallel over MR-row tiles when compute volume warrants it. */
static inline void gemmsup_u8_avx2(const uint8_t *a, const uint8_t *b,
                                   uint8_t *out, size_t M, size_t K, size_t N,
                                   intptr_t rsa, intptr_t csa, intptr_t rsb,
                                   intptr_t rso) {
  const size_t MR = 6, NR = 16;
  size_t n_ir = (M + MR - 1) / MR;
#pragma omp parallel for schedule(static) if ((uint64_t)M * K * N > \
                                                  GEMMSUP_OMP_THRESHOLD)
  for (size_t ir = 0; ir < n_ir; ir++) {
    size_t i = ir * MR;
    size_t mr = GEMMSUP_MIN(MR, M - i);
    for (size_t j = 0; j < N; j += NR) {
      size_t nr = GEMMSUP_MIN(NR, N - j);
      if (mr == MR && nr == NR) {
        gemmsup_ukernel_u8_6x16(a + i * rsa, b + j, out + i * rso + j, K, rsa,
                                csa, rsb, rso);
      } else {
        gemmsup_edge_u8(a + i * rsa, b + j, out + i * rso + j, mr, nr, K, rsa,
                        csa, rsb, rso);
      }
    }
  }
}

#undef GEMMSUP_MIN

#endif
