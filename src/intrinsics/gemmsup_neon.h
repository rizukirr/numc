#ifndef NUMC_GEMMSUP_NEON_H
#define NUMC_GEMMSUP_NEON_H

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifndef GEMMSUP_MIN
#define GEMMSUP_MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef GEMMSUP_FLOPS_THRESHOLD
#define GEMMSUP_FLOPS_THRESHOLD (128UL * 128UL * 128UL)
#endif

/* OMP gate for gemmsup: parallelize from ~128^3 and up. */
#ifndef GEMMSUP_OMP_THRESHOLD
#define GEMMSUP_OMP_THRESHOLD (1ULL << 20)
#endif

/* =================================================================
   Float32 unpacked 6x12 micro-kernel (NEON 128-bit)
   6 rows x 12 cols = 18 accumulators (float32x4_t) + 3 B + 2 A = 23 regs
   8x K-loop unroll with 2 alternating A broadcast registers.
   ================================================================= */

static inline void gemmsup_ukernel_f32_6x12(const float *a, const float *b,
                                            float *c, size_t kc, intptr_t rsa,
                                            intptr_t csa, intptr_t rsb,
                                            intptr_t rso) {
  float32x4_t c00 = vdupq_n_f32(0), c01 = vdupq_n_f32(0), c02 = vdupq_n_f32(0);
  float32x4_t c10 = vdupq_n_f32(0), c11 = vdupq_n_f32(0), c12 = vdupq_n_f32(0);
  float32x4_t c20 = vdupq_n_f32(0), c21 = vdupq_n_f32(0), c22 = vdupq_n_f32(0);
  float32x4_t c30 = vdupq_n_f32(0), c31 = vdupq_n_f32(0), c32 = vdupq_n_f32(0);
  float32x4_t c40 = vdupq_n_f32(0), c41 = vdupq_n_f32(0), c42 = vdupq_n_f32(0);
  float32x4_t c50 = vdupq_n_f32(0), c51 = vdupq_n_f32(0), c52 = vdupq_n_f32(0);

  /* 2 alternating A broadcast registers for ILP. */
#define GEMMSUP_NEON_F32_K_BODY(p_off)             \
  do {                                             \
    const float *bp_ = b + (p_off) * rsb;          \
    float32x4_t b0_ = vld1q_f32(bp_);              \
    float32x4_t b1_ = vld1q_f32(bp_ + 4);          \
    float32x4_t b2_ = vld1q_f32(bp_ + 8);          \
    float32x4_t a0_, a1_;                          \
    a0_ = vdupq_n_f32(a[0 * rsa + (p_off) * csa]); \
    a1_ = vdupq_n_f32(a[1 * rsa + (p_off) * csa]); \
    c00 = vfmaq_f32(c00, a0_, b0_);                \
    c01 = vfmaq_f32(c01, a0_, b1_);                \
    c02 = vfmaq_f32(c02, a0_, b2_);                \
    c10 = vfmaq_f32(c10, a1_, b0_);                \
    c11 = vfmaq_f32(c11, a1_, b1_);                \
    c12 = vfmaq_f32(c12, a1_, b2_);                \
    a0_ = vdupq_n_f32(a[2 * rsa + (p_off) * csa]); \
    a1_ = vdupq_n_f32(a[3 * rsa + (p_off) * csa]); \
    c20 = vfmaq_f32(c20, a0_, b0_);                \
    c21 = vfmaq_f32(c21, a0_, b1_);                \
    c22 = vfmaq_f32(c22, a0_, b2_);                \
    c30 = vfmaq_f32(c30, a1_, b0_);                \
    c31 = vfmaq_f32(c31, a1_, b1_);                \
    c32 = vfmaq_f32(c32, a1_, b2_);                \
    a0_ = vdupq_n_f32(a[4 * rsa + (p_off) * csa]); \
    a1_ = vdupq_n_f32(a[5 * rsa + (p_off) * csa]); \
    c40 = vfmaq_f32(c40, a0_, b0_);                \
    c41 = vfmaq_f32(c41, a0_, b1_);                \
    c42 = vfmaq_f32(c42, a0_, b2_);                \
    c50 = vfmaq_f32(c50, a1_, b0_);                \
    c51 = vfmaq_f32(c51, a1_, b1_);                \
    c52 = vfmaq_f32(c52, a1_, b2_);                \
  } while (0)

  /* 8x K-loop unroll */
  size_t k_iter = kc / 8;
  size_t k_left = kc % 8;
  size_t p = 0;
  for (size_t ki = 0; ki < k_iter; ki++, p += 8) {
    GEMMSUP_NEON_F32_K_BODY(p);
    GEMMSUP_NEON_F32_K_BODY(p + 1);
    GEMMSUP_NEON_F32_K_BODY(p + 2);
    GEMMSUP_NEON_F32_K_BODY(p + 3);
    GEMMSUP_NEON_F32_K_BODY(p + 4);
    GEMMSUP_NEON_F32_K_BODY(p + 5);
    GEMMSUP_NEON_F32_K_BODY(p + 6);
    GEMMSUP_NEON_F32_K_BODY(p + 7);
  }
  for (size_t pi = 0; pi < k_left; pi++, p++) {
    GEMMSUP_NEON_F32_K_BODY(p);
  }
#undef GEMMSUP_NEON_F32_K_BODY

  vst1q_f32(c, c00);
  vst1q_f32(c + 4, c01);
  vst1q_f32(c + 8, c02);
  vst1q_f32(c + rso, c10);
  vst1q_f32(c + rso + 4, c11);
  vst1q_f32(c + rso + 8, c12);
  vst1q_f32(c + 2 * rso, c20);
  vst1q_f32(c + 2 * rso + 4, c21);
  vst1q_f32(c + 2 * rso + 8, c22);
  vst1q_f32(c + 3 * rso, c30);
  vst1q_f32(c + 3 * rso + 4, c31);
  vst1q_f32(c + 3 * rso + 8, c32);
  vst1q_f32(c + 4 * rso, c40);
  vst1q_f32(c + 4 * rso + 4, c41);
  vst1q_f32(c + 4 * rso + 8, c42);
  vst1q_f32(c + 5 * rso, c50);
  vst1q_f32(c + 5 * rso + 4, c51);
  vst1q_f32(c + 5 * rso + 8, c52);
}

/* Scalar edge kernel for remainder tiles */
static inline void gemmsup_edge_f32_neon(const float *a, const float *b,
                                         float *c, size_t mr, size_t nr,
                                         size_t kc, intptr_t rsa, intptr_t csa,
                                         intptr_t rsb, intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
    for (size_t j = 0; j < nr; j++)
      c[i * rso + j] = 0.0f;
    for (size_t p = 0; p < kc; p++) {
      float aip = a[i * rsa + p * csa];
      float32x4_t va = vdupq_n_f32(aip);
      const float *brow = b + p * rsb;
      float *crow = c + i * rso;
      size_t j = 0;
      for (; j + 4 <= nr; j += 4) {
        float32x4_t vo = vld1q_f32(crow + j);
        vo = vfmaq_f32(vo, va, vld1q_f32(brow + j));
        vst1q_f32(crow + j, vo);
      }
      for (; j < nr; j++)
        crow[j] += aip * brow[j];
    }
  }
}

/* Full gemmsup dispatch: tile MxN in 6x12 blocks, no packing.
 * OMP-parallel over MR-row tiles when compute volume warrants it. */
static inline void gemmsup_f32_neon(const float *a, const float *b, float *out,
                                    size_t M, size_t K, size_t N, intptr_t rsa,
                                    intptr_t csa, intptr_t rsb, intptr_t rso) {
  const size_t MR = 6, NR = 12;
  size_t n_ir = (M + MR - 1) / MR;
#pragma omp parallel for schedule(static) if ((uint64_t)M * K * N > \
                                                  GEMMSUP_OMP_THRESHOLD)
  for (size_t ir = 0; ir < n_ir; ir++) {
    size_t i = ir * MR;
    size_t mr = GEMMSUP_MIN(MR, M - i);
    for (size_t j = 0; j < N; j += NR) {
      size_t nr = GEMMSUP_MIN(NR, N - j);
      if (mr == MR && nr == NR) {
        gemmsup_ukernel_f32_6x12(a + i * rsa, b + j, out + i * rso + j, K, rsa,
                                 csa, rsb, rso);
      } else {
        gemmsup_edge_f32_neon(a + i * rsa, b + j, out + i * rso + j, mr, nr, K,
                              rsa, csa, rsb, rso);
      }
    }
  }
}

/* =================================================================
   Float64 unpacked 6x8 micro-kernel (NEON 128-bit)
   6 rows x 8 cols = 24 accumulators (float64x2_t) + 4 B + 2 A = 30 regs
   Bumped from MR=4 to MR=6 to better utilize register file.
   8x K-loop unroll with 2 alternating A broadcast registers.
   ================================================================= */

static inline void gemmsup_ukernel_f64_6x8(const double *a, const double *b,
                                           double *c, size_t kc, intptr_t rsa,
                                           intptr_t csa, intptr_t rsb,
                                           intptr_t rso) {
  float64x2_t c00 = vdupq_n_f64(0), c01 = vdupq_n_f64(0), c02 = vdupq_n_f64(0),
              c03 = vdupq_n_f64(0);
  float64x2_t c10 = vdupq_n_f64(0), c11 = vdupq_n_f64(0), c12 = vdupq_n_f64(0),
              c13 = vdupq_n_f64(0);
  float64x2_t c20 = vdupq_n_f64(0), c21 = vdupq_n_f64(0), c22 = vdupq_n_f64(0),
              c23 = vdupq_n_f64(0);
  float64x2_t c30 = vdupq_n_f64(0), c31 = vdupq_n_f64(0), c32 = vdupq_n_f64(0),
              c33 = vdupq_n_f64(0);
  float64x2_t c40 = vdupq_n_f64(0), c41 = vdupq_n_f64(0), c42 = vdupq_n_f64(0),
              c43 = vdupq_n_f64(0);
  float64x2_t c50 = vdupq_n_f64(0), c51 = vdupq_n_f64(0), c52 = vdupq_n_f64(0),
              c53 = vdupq_n_f64(0);

  /* 2 alternating A broadcast registers for ILP. */
#define GEMMSUP_NEON_F64_K_BODY(p_off)             \
  do {                                             \
    const double *bp_ = b + (p_off) * rsb;         \
    float64x2_t b0_ = vld1q_f64(bp_);              \
    float64x2_t b1_ = vld1q_f64(bp_ + 2);          \
    float64x2_t b2_ = vld1q_f64(bp_ + 4);          \
    float64x2_t b3_ = vld1q_f64(bp_ + 6);          \
    float64x2_t a0_, a1_;                          \
    a0_ = vdupq_n_f64(a[0 * rsa + (p_off) * csa]); \
    a1_ = vdupq_n_f64(a[1 * rsa + (p_off) * csa]); \
    c00 = vfmaq_f64(c00, a0_, b0_);                \
    c01 = vfmaq_f64(c01, a0_, b1_);                \
    c02 = vfmaq_f64(c02, a0_, b2_);                \
    c03 = vfmaq_f64(c03, a0_, b3_);                \
    c10 = vfmaq_f64(c10, a1_, b0_);                \
    c11 = vfmaq_f64(c11, a1_, b1_);                \
    c12 = vfmaq_f64(c12, a1_, b2_);                \
    c13 = vfmaq_f64(c13, a1_, b3_);                \
    a0_ = vdupq_n_f64(a[2 * rsa + (p_off) * csa]); \
    a1_ = vdupq_n_f64(a[3 * rsa + (p_off) * csa]); \
    c20 = vfmaq_f64(c20, a0_, b0_);                \
    c21 = vfmaq_f64(c21, a0_, b1_);                \
    c22 = vfmaq_f64(c22, a0_, b2_);                \
    c23 = vfmaq_f64(c23, a0_, b3_);                \
    c30 = vfmaq_f64(c30, a1_, b0_);                \
    c31 = vfmaq_f64(c31, a1_, b1_);                \
    c32 = vfmaq_f64(c32, a1_, b2_);                \
    c33 = vfmaq_f64(c33, a1_, b3_);                \
    a0_ = vdupq_n_f64(a[4 * rsa + (p_off) * csa]); \
    a1_ = vdupq_n_f64(a[5 * rsa + (p_off) * csa]); \
    c40 = vfmaq_f64(c40, a0_, b0_);                \
    c41 = vfmaq_f64(c41, a0_, b1_);                \
    c42 = vfmaq_f64(c42, a0_, b2_);                \
    c43 = vfmaq_f64(c43, a0_, b3_);                \
    c50 = vfmaq_f64(c50, a1_, b0_);                \
    c51 = vfmaq_f64(c51, a1_, b1_);                \
    c52 = vfmaq_f64(c52, a1_, b2_);                \
    c53 = vfmaq_f64(c53, a1_, b3_);                \
  } while (0)

  /* 8x K-loop unroll */
  size_t k_iter = kc / 8;
  size_t k_left = kc % 8;
  size_t p = 0;
  for (size_t ki = 0; ki < k_iter; ki++, p += 8) {
    GEMMSUP_NEON_F64_K_BODY(p);
    GEMMSUP_NEON_F64_K_BODY(p + 1);
    GEMMSUP_NEON_F64_K_BODY(p + 2);
    GEMMSUP_NEON_F64_K_BODY(p + 3);
    GEMMSUP_NEON_F64_K_BODY(p + 4);
    GEMMSUP_NEON_F64_K_BODY(p + 5);
    GEMMSUP_NEON_F64_K_BODY(p + 6);
    GEMMSUP_NEON_F64_K_BODY(p + 7);
  }
  for (size_t pi = 0; pi < k_left; pi++, p++) {
    GEMMSUP_NEON_F64_K_BODY(p);
  }
#undef GEMMSUP_NEON_F64_K_BODY

  vst1q_f64(c, c00);
  vst1q_f64(c + 2, c01);
  vst1q_f64(c + 4, c02);
  vst1q_f64(c + 6, c03);
  vst1q_f64(c + rso, c10);
  vst1q_f64(c + rso + 2, c11);
  vst1q_f64(c + rso + 4, c12);
  vst1q_f64(c + rso + 6, c13);
  vst1q_f64(c + 2 * rso, c20);
  vst1q_f64(c + 2 * rso + 2, c21);
  vst1q_f64(c + 2 * rso + 4, c22);
  vst1q_f64(c + 2 * rso + 6, c23);
  vst1q_f64(c + 3 * rso, c30);
  vst1q_f64(c + 3 * rso + 2, c31);
  vst1q_f64(c + 3 * rso + 4, c32);
  vst1q_f64(c + 3 * rso + 6, c33);
  vst1q_f64(c + 4 * rso, c40);
  vst1q_f64(c + 4 * rso + 2, c41);
  vst1q_f64(c + 4 * rso + 4, c42);
  vst1q_f64(c + 4 * rso + 6, c43);
  vst1q_f64(c + 5 * rso, c50);
  vst1q_f64(c + 5 * rso + 2, c51);
  vst1q_f64(c + 5 * rso + 4, c52);
  vst1q_f64(c + 5 * rso + 6, c53);
}

static inline void gemmsup_edge_f64_neon(const double *a, const double *b,
                                         double *c, size_t mr, size_t nr,
                                         size_t kc, intptr_t rsa, intptr_t csa,
                                         intptr_t rsb, intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
    for (size_t j = 0; j < nr; j++)
      c[i * rso + j] = 0.0;
    for (size_t p = 0; p < kc; p++) {
      double aip = a[i * rsa + p * csa];
      float64x2_t va = vdupq_n_f64(aip);
      const double *brow = b + p * rsb;
      double *crow = c + i * rso;
      size_t j = 0;
      for (; j + 2 <= nr; j += 2) {
        float64x2_t vo = vld1q_f64(crow + j);
        vo = vfmaq_f64(vo, va, vld1q_f64(brow + j));
        vst1q_f64(crow + j, vo);
      }
      for (; j < nr; j++)
        crow[j] += aip * brow[j];
    }
  }
}

/* OMP-parallel over MR-row tiles when compute volume warrants it. */
static inline void gemmsup_f64_neon(const double *a, const double *b,
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
        gemmsup_edge_f64_neon(a + i * rsa, b + j, out + i * rso + j, mr, nr, K,
                              rsa, csa, rsb, rso);
      }
    }
  }
}

#endif /* NUMC_GEMMSUP_NEON_H */
