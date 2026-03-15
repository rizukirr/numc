#ifndef NUMC_GEMMSUP_SVE_H
#define NUMC_GEMMSUP_SVE_H

#include <arm_sve.h>
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
   Float32 unpacked scalable micro-kernel (SVE)
   MR=6, NR=2*svcntw() (2 SVE vectors wide), predicated tail on N.
   8x K-loop unroll with 2 alternating A broadcast registers.
   ================================================================= */

static inline void gemmsup_ukernel_f32_6xVL2(const float *a, const float *b,
                                             float *c, size_t kc, size_t nr,
                                             intptr_t rsa, intptr_t csa,
                                             intptr_t rsb, intptr_t rso) {
  svbool_t ptrue = svptrue_b32();
  size_t vl = svcntw();

  /* Predication for second vector (nr may be < 2*vl) */
  svbool_t pg0 = ptrue;
  svbool_t pg1 = svwhilelt_b32((uint32_t)vl, (uint32_t)nr);

  svfloat32_t c00 = svdup_f32(0), c01 = svdup_f32(0);
  svfloat32_t c10 = svdup_f32(0), c11 = svdup_f32(0);
  svfloat32_t c20 = svdup_f32(0), c21 = svdup_f32(0);
  svfloat32_t c30 = svdup_f32(0), c31 = svdup_f32(0);
  svfloat32_t c40 = svdup_f32(0), c41 = svdup_f32(0);
  svfloat32_t c50 = svdup_f32(0), c51 = svdup_f32(0);

  /* 2 alternating A broadcast registers for ILP. */
#define GEMMSUP_SVE_F32_K_BODY(p_off)            \
  do {                                           \
    const float *bp_ = b + (p_off) * rsb;        \
    svfloat32_t b0_ = svld1_f32(pg0, bp_);       \
    svfloat32_t b1_ = svld1_f32(pg1, bp_ + vl);  \
    svfloat32_t a0_, a1_;                        \
    a0_ = svdup_f32(a[0 * rsa + (p_off) * csa]); \
    a1_ = svdup_f32(a[1 * rsa + (p_off) * csa]); \
    c00 = svmla_f32_x(ptrue, c00, a0_, b0_);     \
    c01 = svmla_f32_x(ptrue, c01, a0_, b1_);     \
    c10 = svmla_f32_x(ptrue, c10, a1_, b0_);     \
    c11 = svmla_f32_x(ptrue, c11, a1_, b1_);     \
    a0_ = svdup_f32(a[2 * rsa + (p_off) * csa]); \
    a1_ = svdup_f32(a[3 * rsa + (p_off) * csa]); \
    c20 = svmla_f32_x(ptrue, c20, a0_, b0_);     \
    c21 = svmla_f32_x(ptrue, c21, a0_, b1_);     \
    c30 = svmla_f32_x(ptrue, c30, a1_, b0_);     \
    c31 = svmla_f32_x(ptrue, c31, a1_, b1_);     \
    a0_ = svdup_f32(a[4 * rsa + (p_off) * csa]); \
    a1_ = svdup_f32(a[5 * rsa + (p_off) * csa]); \
    c40 = svmla_f32_x(ptrue, c40, a0_, b0_);     \
    c41 = svmla_f32_x(ptrue, c41, a0_, b1_);     \
    c50 = svmla_f32_x(ptrue, c50, a1_, b0_);     \
    c51 = svmla_f32_x(ptrue, c51, a1_, b1_);     \
  } while (0)

  /* 8x K-loop unroll */
  size_t k_iter = kc / 8;
  size_t k_left = kc % 8;
  size_t p = 0;
  for (size_t ki = 0; ki < k_iter; ki++, p += 8) {
    GEMMSUP_SVE_F32_K_BODY(p);
    GEMMSUP_SVE_F32_K_BODY(p + 1);
    GEMMSUP_SVE_F32_K_BODY(p + 2);
    GEMMSUP_SVE_F32_K_BODY(p + 3);
    GEMMSUP_SVE_F32_K_BODY(p + 4);
    GEMMSUP_SVE_F32_K_BODY(p + 5);
    GEMMSUP_SVE_F32_K_BODY(p + 6);
    GEMMSUP_SVE_F32_K_BODY(p + 7);
  }
  for (size_t pi = 0; pi < k_left; pi++, p++) {
    GEMMSUP_SVE_F32_K_BODY(p);
  }
#undef GEMMSUP_SVE_F32_K_BODY

  svst1_f32(pg0, c, c00);
  svst1_f32(pg1, c + vl, c01);
  svst1_f32(pg0, c + rso, c10);
  svst1_f32(pg1, c + rso + vl, c11);
  svst1_f32(pg0, c + 2 * rso, c20);
  svst1_f32(pg1, c + 2 * rso + vl, c21);
  svst1_f32(pg0, c + 3 * rso, c30);
  svst1_f32(pg1, c + 3 * rso + vl, c31);
  svst1_f32(pg0, c + 4 * rso, c40);
  svst1_f32(pg1, c + 4 * rso + vl, c41);
  svst1_f32(pg0, c + 5 * rso, c50);
  svst1_f32(pg1, c + 5 * rso + vl, c51);
}

/* Scalar edge kernel for M-remainder tiles */
static inline void gemmsup_edge_f32_sve(const float *a, const float *b,
                                        float *c, size_t mr, size_t nr,
                                        size_t kc, intptr_t rsa, intptr_t csa,
                                        intptr_t rsb, intptr_t rso) {
  size_t vl = svcntw();
  for (size_t i = 0; i < mr; i++) {
    /* Zero output row */
    for (size_t j = 0; j < nr; j += vl) {
      svbool_t pg = svwhilelt_b32((uint32_t)j, (uint32_t)nr);
      svst1_f32(pg, c + i * rso + j, svdup_f32(0));
    }
    for (size_t p = 0; p < kc; p++) {
      svfloat32_t va = svdup_f32(a[i * rsa + p * csa]);
      const float *brow = b + p * rsb;
      float *crow = c + i * rso;
      for (size_t j = 0; j < nr; j += vl) {
        svbool_t pg = svwhilelt_b32((uint32_t)j, (uint32_t)nr);
        svfloat32_t vo = svld1_f32(pg, crow + j);
        vo = svmla_f32_x(pg, vo, va, svld1_f32(pg, brow + j));
        svst1_f32(pg, crow + j, vo);
      }
    }
  }
}

/* OMP-parallel over MR-row tiles when compute volume warrants it. */
static inline void gemmsup_f32_sve(const float *a, const float *b, float *out,
                                   size_t M, size_t K, size_t N, intptr_t rsa,
                                   intptr_t csa, intptr_t rsb, intptr_t rso) {
  const size_t MR = 6;
  size_t NR = svcntw() * 2;
  size_t n_ir = (M + MR - 1) / MR;
#pragma omp parallel for schedule(static) if ((uint64_t)M * K * N > \
                                                  GEMMSUP_OMP_THRESHOLD)
  for (size_t ir = 0; ir < n_ir; ir++) {
    size_t i = ir * MR;
    size_t mr = GEMMSUP_MIN(MR, M - i);
    for (size_t j = 0; j < N; j += NR) {
      size_t nr = GEMMSUP_MIN(NR, N - j);
      if (mr == MR) {
        gemmsup_ukernel_f32_6xVL2(a + i * rsa, b + j, out + i * rso + j, K, nr,
                                  rsa, csa, rsb, rso);
      } else {
        gemmsup_edge_f32_sve(a + i * rsa, b + j, out + i * rso + j, mr, nr, K,
                             rsa, csa, rsb, rso);
      }
    }
  }
}

/* =================================================================
   Float64 unpacked scalable micro-kernel (SVE)
   MR=6 (bumped from 4), NR=2*svcntd()
   6 rows x 2 vectors = 12 accumulators + 2 B + 2 A = 16 SVE regs
   8x K-loop unroll with 2 alternating A broadcast registers.
   ================================================================= */

static inline void gemmsup_ukernel_f64_6xVL2(const double *a, const double *b,
                                             double *c, size_t kc, size_t nr,
                                             intptr_t rsa, intptr_t csa,
                                             intptr_t rsb, intptr_t rso) {
  svbool_t ptrue = svptrue_b64();
  size_t vl = svcntd();
  svbool_t pg0 = ptrue;
  svbool_t pg1 = svwhilelt_b64((uint32_t)vl, (uint32_t)nr);

  svfloat64_t c00 = svdup_f64(0), c01 = svdup_f64(0);
  svfloat64_t c10 = svdup_f64(0), c11 = svdup_f64(0);
  svfloat64_t c20 = svdup_f64(0), c21 = svdup_f64(0);
  svfloat64_t c30 = svdup_f64(0), c31 = svdup_f64(0);
  svfloat64_t c40 = svdup_f64(0), c41 = svdup_f64(0);
  svfloat64_t c50 = svdup_f64(0), c51 = svdup_f64(0);

  /* 2 alternating A broadcast registers for ILP. */
#define GEMMSUP_SVE_F64_K_BODY(p_off)            \
  do {                                           \
    const double *bp_ = b + (p_off) * rsb;       \
    svfloat64_t b0_ = svld1_f64(pg0, bp_);       \
    svfloat64_t b1_ = svld1_f64(pg1, bp_ + vl);  \
    svfloat64_t a0_, a1_;                        \
    a0_ = svdup_f64(a[0 * rsa + (p_off) * csa]); \
    a1_ = svdup_f64(a[1 * rsa + (p_off) * csa]); \
    c00 = svmla_f64_x(ptrue, c00, a0_, b0_);     \
    c01 = svmla_f64_x(ptrue, c01, a0_, b1_);     \
    c10 = svmla_f64_x(ptrue, c10, a1_, b0_);     \
    c11 = svmla_f64_x(ptrue, c11, a1_, b1_);     \
    a0_ = svdup_f64(a[2 * rsa + (p_off) * csa]); \
    a1_ = svdup_f64(a[3 * rsa + (p_off) * csa]); \
    c20 = svmla_f64_x(ptrue, c20, a0_, b0_);     \
    c21 = svmla_f64_x(ptrue, c21, a0_, b1_);     \
    c30 = svmla_f64_x(ptrue, c30, a1_, b0_);     \
    c31 = svmla_f64_x(ptrue, c31, a1_, b1_);     \
    a0_ = svdup_f64(a[4 * rsa + (p_off) * csa]); \
    a1_ = svdup_f64(a[5 * rsa + (p_off) * csa]); \
    c40 = svmla_f64_x(ptrue, c40, a0_, b0_);     \
    c41 = svmla_f64_x(ptrue, c41, a0_, b1_);     \
    c50 = svmla_f64_x(ptrue, c50, a1_, b0_);     \
    c51 = svmla_f64_x(ptrue, c51, a1_, b1_);     \
  } while (0)

  /* 8x K-loop unroll */
  size_t k_iter = kc / 8;
  size_t k_left = kc % 8;
  size_t p = 0;
  for (size_t ki = 0; ki < k_iter; ki++, p += 8) {
    GEMMSUP_SVE_F64_K_BODY(p);
    GEMMSUP_SVE_F64_K_BODY(p + 1);
    GEMMSUP_SVE_F64_K_BODY(p + 2);
    GEMMSUP_SVE_F64_K_BODY(p + 3);
    GEMMSUP_SVE_F64_K_BODY(p + 4);
    GEMMSUP_SVE_F64_K_BODY(p + 5);
    GEMMSUP_SVE_F64_K_BODY(p + 6);
    GEMMSUP_SVE_F64_K_BODY(p + 7);
  }
  for (size_t pi = 0; pi < k_left; pi++, p++) {
    GEMMSUP_SVE_F64_K_BODY(p);
  }
#undef GEMMSUP_SVE_F64_K_BODY

  svst1_f64(pg0, c, c00);
  svst1_f64(pg1, c + vl, c01);
  svst1_f64(pg0, c + rso, c10);
  svst1_f64(pg1, c + rso + vl, c11);
  svst1_f64(pg0, c + 2 * rso, c20);
  svst1_f64(pg1, c + 2 * rso + vl, c21);
  svst1_f64(pg0, c + 3 * rso, c30);
  svst1_f64(pg1, c + 3 * rso + vl, c31);
  svst1_f64(pg0, c + 4 * rso, c40);
  svst1_f64(pg1, c + 4 * rso + vl, c41);
  svst1_f64(pg0, c + 5 * rso, c50);
  svst1_f64(pg1, c + 5 * rso + vl, c51);
}

static inline void gemmsup_edge_f64_sve(const double *a, const double *b,
                                        double *c, size_t mr, size_t nr,
                                        size_t kc, intptr_t rsa, intptr_t csa,
                                        intptr_t rsb, intptr_t rso) {
  size_t vl = svcntd();
  for (size_t i = 0; i < mr; i++) {
    for (size_t j = 0; j < nr; j += vl) {
      svbool_t pg = svwhilelt_b64((uint32_t)j, (uint32_t)nr);
      svst1_f64(pg, c + i * rso + j, svdup_f64(0));
    }
    for (size_t p = 0; p < kc; p++) {
      svfloat64_t va = svdup_f64(a[i * rsa + p * csa]);
      const double *brow = b + p * rsb;
      double *crow = c + i * rso;
      for (size_t j = 0; j < nr; j += vl) {
        svbool_t pg = svwhilelt_b64((uint32_t)j, (uint32_t)nr);
        svfloat64_t vo = svld1_f64(pg, crow + j);
        vo = svmla_f64_x(pg, vo, va, svld1_f64(pg, brow + j));
        svst1_f64(pg, crow + j, vo);
      }
    }
  }
}

/* OMP-parallel over MR-row tiles when compute volume warrants it. */
static inline void gemmsup_f64_sve(const double *a, const double *b,
                                   double *out, size_t M, size_t K, size_t N,
                                   intptr_t rsa, intptr_t csa, intptr_t rsb,
                                   intptr_t rso) {
  const size_t MR = 6;
  size_t NR = svcntd() * 2;
  size_t n_ir = (M + MR - 1) / MR;
#pragma omp parallel for schedule(static) if ((uint64_t)M * K * N > \
                                                  GEMMSUP_OMP_THRESHOLD)
  for (size_t ir = 0; ir < n_ir; ir++) {
    size_t i = ir * MR;
    size_t mr = GEMMSUP_MIN(MR, M - i);
    for (size_t j = 0; j < N; j += NR) {
      size_t nr = GEMMSUP_MIN(NR, N - j);
      if (mr == MR) {
        gemmsup_ukernel_f64_6xVL2(a + i * rsa, b + j, out + i * rso + j, K, nr,
                                  rsa, csa, rsb, rso);
      } else {
        gemmsup_edge_f64_sve(a + i * rsa, b + j, out + i * rso + j, mr, nr, K,
                             rsa, csa, rsb, rso);
      }
    }
  }
}

#endif /* NUMC_GEMMSUP_SVE_H */
