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

/* =================================================================
   Float32 unpacked scalable micro-kernel (SVE)
   MR=6, NR=2*svcntw() (2 SVE vectors wide), predicated tail on N
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

  for (size_t p = 0; p < kc; p++) {
    const float *bp = b + p * rsb;
    svfloat32_t b0 = svld1_f32(pg0, bp);
    svfloat32_t b1 = svld1_f32(pg1, bp + vl);

    svfloat32_t av;
    av = svdup_f32(a[0 * rsa + p * csa]);
    c00 = svmla_f32_x(ptrue, c00, av, b0);
    c01 = svmla_f32_x(ptrue, c01, av, b1);
    av = svdup_f32(a[1 * rsa + p * csa]);
    c10 = svmla_f32_x(ptrue, c10, av, b0);
    c11 = svmla_f32_x(ptrue, c11, av, b1);
    av = svdup_f32(a[2 * rsa + p * csa]);
    c20 = svmla_f32_x(ptrue, c20, av, b0);
    c21 = svmla_f32_x(ptrue, c21, av, b1);
    av = svdup_f32(a[3 * rsa + p * csa]);
    c30 = svmla_f32_x(ptrue, c30, av, b0);
    c31 = svmla_f32_x(ptrue, c31, av, b1);
    av = svdup_f32(a[4 * rsa + p * csa]);
    c40 = svmla_f32_x(ptrue, c40, av, b0);
    c41 = svmla_f32_x(ptrue, c41, av, b1);
    av = svdup_f32(a[5 * rsa + p * csa]);
    c50 = svmla_f32_x(ptrue, c50, av, b0);
    c51 = svmla_f32_x(ptrue, c51, av, b1);
  }

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

static inline void gemmsup_f32_sve(const float *a, const float *b, float *out,
                                   size_t M, size_t K, size_t N, intptr_t rsa,
                                   intptr_t csa, intptr_t rsb, intptr_t rso) {
  const size_t MR = 6;
  size_t NR = svcntw() * 2;
  for (size_t i = 0; i < M; i += MR) {
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
   MR=4, NR=2*svcntd()
   ================================================================= */

static inline void gemmsup_ukernel_f64_4xVL2(const double *a, const double *b,
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

  for (size_t p = 0; p < kc; p++) {
    const double *bp = b + p * rsb;
    svfloat64_t b0 = svld1_f64(pg0, bp);
    svfloat64_t b1 = svld1_f64(pg1, bp + vl);

    svfloat64_t av;
    av = svdup_f64(a[0 * rsa + p * csa]);
    c00 = svmla_f64_x(ptrue, c00, av, b0);
    c01 = svmla_f64_x(ptrue, c01, av, b1);
    av = svdup_f64(a[1 * rsa + p * csa]);
    c10 = svmla_f64_x(ptrue, c10, av, b0);
    c11 = svmla_f64_x(ptrue, c11, av, b1);
    av = svdup_f64(a[2 * rsa + p * csa]);
    c20 = svmla_f64_x(ptrue, c20, av, b0);
    c21 = svmla_f64_x(ptrue, c21, av, b1);
    av = svdup_f64(a[3 * rsa + p * csa]);
    c30 = svmla_f64_x(ptrue, c30, av, b0);
    c31 = svmla_f64_x(ptrue, c31, av, b1);
  }

  svst1_f64(pg0, c, c00);
  svst1_f64(pg1, c + vl, c01);
  svst1_f64(pg0, c + rso, c10);
  svst1_f64(pg1, c + rso + vl, c11);
  svst1_f64(pg0, c + 2 * rso, c20);
  svst1_f64(pg1, c + 2 * rso + vl, c21);
  svst1_f64(pg0, c + 3 * rso, c30);
  svst1_f64(pg1, c + 3 * rso + vl, c31);
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

static inline void gemmsup_f64_sve(const double *a, const double *b,
                                   double *out, size_t M, size_t K, size_t N,
                                   intptr_t rsa, intptr_t csa, intptr_t rsb,
                                   intptr_t rso) {
  const size_t MR = 4;
  size_t NR = svcntd() * 2;
  for (size_t i = 0; i < M; i += MR) {
    size_t mr = GEMMSUP_MIN(MR, M - i);
    for (size_t j = 0; j < N; j += NR) {
      size_t nr = GEMMSUP_MIN(NR, N - j);
      if (mr == MR) {
        gemmsup_ukernel_f64_4xVL2(a + i * rsa, b + j, out + i * rso + j, K, nr,
                                  rsa, csa, rsb, rso);
      } else {
        gemmsup_edge_f64_sve(a + i * rsa, b + j, out + i * rso + j, mr, nr, K,
                             rsa, csa, rsb, rso);
      }
    }
  }
}

#endif /* NUMC_GEMMSUP_SVE_H */
