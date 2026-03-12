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

/* =================================================================
   Float32 unpacked 6x12 micro-kernel (NEON 128-bit)
   6 rows x 12 cols = 18 accumulators (float32x4_t) + 3 B + 6 A = 27 regs
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

  for (size_t p = 0; p < kc; p++) {
    const float *bp = b + p * rsb;
    float32x4_t b0 = vld1q_f32(bp);
    float32x4_t b1 = vld1q_f32(bp + 4);
    float32x4_t b2 = vld1q_f32(bp + 8);

    float32x4_t av;
    av = vdupq_n_f32(a[0 * rsa + p * csa]);
    c00 = vfmaq_f32(c00, av, b0);
    c01 = vfmaq_f32(c01, av, b1);
    c02 = vfmaq_f32(c02, av, b2);
    av = vdupq_n_f32(a[1 * rsa + p * csa]);
    c10 = vfmaq_f32(c10, av, b0);
    c11 = vfmaq_f32(c11, av, b1);
    c12 = vfmaq_f32(c12, av, b2);
    av = vdupq_n_f32(a[2 * rsa + p * csa]);
    c20 = vfmaq_f32(c20, av, b0);
    c21 = vfmaq_f32(c21, av, b1);
    c22 = vfmaq_f32(c22, av, b2);
    av = vdupq_n_f32(a[3 * rsa + p * csa]);
    c30 = vfmaq_f32(c30, av, b0);
    c31 = vfmaq_f32(c31, av, b1);
    c32 = vfmaq_f32(c32, av, b2);
    av = vdupq_n_f32(a[4 * rsa + p * csa]);
    c40 = vfmaq_f32(c40, av, b0);
    c41 = vfmaq_f32(c41, av, b1);
    c42 = vfmaq_f32(c42, av, b2);
    av = vdupq_n_f32(a[5 * rsa + p * csa]);
    c50 = vfmaq_f32(c50, av, b0);
    c51 = vfmaq_f32(c51, av, b1);
    c52 = vfmaq_f32(c52, av, b2);
  }

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

static inline void gemmsup_f32_neon(const float *a, const float *b, float *out,
                                    size_t M, size_t K, size_t N, intptr_t rsa,
                                    intptr_t csa, intptr_t rsb, intptr_t rso) {
  const size_t MR = 6, NR = 12;
  for (size_t i = 0; i < M; i += MR) {
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
   Float64 unpacked 4x8 micro-kernel (NEON 128-bit)
   4 rows x 8 cols = 16 accumulators (float64x2_t) + 4 B + 4 A = 24 regs
   ================================================================= */

static inline void gemmsup_ukernel_f64_4x8(const double *a, const double *b,
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

  for (size_t p = 0; p < kc; p++) {
    const double *bp = b + p * rsb;
    float64x2_t b0 = vld1q_f64(bp);
    float64x2_t b1 = vld1q_f64(bp + 2);
    float64x2_t b2 = vld1q_f64(bp + 4);
    float64x2_t b3 = vld1q_f64(bp + 6);

    float64x2_t av;
    av = vdupq_n_f64(a[0 * rsa + p * csa]);
    c00 = vfmaq_f64(c00, av, b0);
    c01 = vfmaq_f64(c01, av, b1);
    c02 = vfmaq_f64(c02, av, b2);
    c03 = vfmaq_f64(c03, av, b3);
    av = vdupq_n_f64(a[1 * rsa + p * csa]);
    c10 = vfmaq_f64(c10, av, b0);
    c11 = vfmaq_f64(c11, av, b1);
    c12 = vfmaq_f64(c12, av, b2);
    c13 = vfmaq_f64(c13, av, b3);
    av = vdupq_n_f64(a[2 * rsa + p * csa]);
    c20 = vfmaq_f64(c20, av, b0);
    c21 = vfmaq_f64(c21, av, b1);
    c22 = vfmaq_f64(c22, av, b2);
    c23 = vfmaq_f64(c23, av, b3);
    av = vdupq_n_f64(a[3 * rsa + p * csa]);
    c30 = vfmaq_f64(c30, av, b0);
    c31 = vfmaq_f64(c31, av, b1);
    c32 = vfmaq_f64(c32, av, b2);
    c33 = vfmaq_f64(c33, av, b3);
  }

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

static inline void gemmsup_f64_neon(const double *a, const double *b,
                                    double *out, size_t M, size_t K, size_t N,
                                    intptr_t rsa, intptr_t csa, intptr_t rsb,
                                    intptr_t rso) {
  const size_t MR = 4, NR = 8;
  for (size_t i = 0; i < M; i += MR) {
    size_t mr = GEMMSUP_MIN(MR, M - i);
    for (size_t j = 0; j < N; j += NR) {
      size_t nr = GEMMSUP_MIN(NR, N - j);
      if (mr == MR && nr == NR) {
        gemmsup_ukernel_f64_4x8(a + i * rsa, b + j, out + i * rso + j, K, rsa,
                                csa, rsb, rso);
      } else {
        gemmsup_edge_f64_neon(a + i * rsa, b + j, out + i * rso + j, mr, nr, K,
                              rsa, csa, rsb, rso);
      }
    }
  }
}

#endif /* NUMC_GEMMSUP_NEON_H */
