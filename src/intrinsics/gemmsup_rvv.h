#ifndef NUMC_GEMMSUP_RVV_H
#define NUMC_GEMMSUP_RVV_H

#include <riscv_vector.h>
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
   Float32 unpacked scalable micro-kernel (RVV)
   MR=6, NR=vsetvlmax_e32m2 (LMUL=m2), natural tail via vsetvl
   ================================================================= */

static inline void gemmsup_ukernel_f32_6xVL(const float *a, const float *b,
                                            float *c, size_t kc, size_t nr,
                                            intptr_t rsa, intptr_t csa,
                                            intptr_t rsb, intptr_t rso) {
  size_t vl = __riscv_vsetvl_e32m2(nr);

  vfloat32m2_t c00 = __riscv_vfmv_v_f_f32m2(0, vl);
  vfloat32m2_t c10 = __riscv_vfmv_v_f_f32m2(0, vl);
  vfloat32m2_t c20 = __riscv_vfmv_v_f_f32m2(0, vl);
  vfloat32m2_t c30 = __riscv_vfmv_v_f_f32m2(0, vl);
  vfloat32m2_t c40 = __riscv_vfmv_v_f_f32m2(0, vl);
  vfloat32m2_t c50 = __riscv_vfmv_v_f_f32m2(0, vl);

  for (size_t p = 0; p < kc; p++) {
    vfloat32m2_t bv = __riscv_vle32_v_f32m2(b + p * rsb, vl);

    c00 = __riscv_vfmacc_vf_f32m2(c00, a[0 * rsa + p * csa], bv, vl);
    c10 = __riscv_vfmacc_vf_f32m2(c10, a[1 * rsa + p * csa], bv, vl);
    c20 = __riscv_vfmacc_vf_f32m2(c20, a[2 * rsa + p * csa], bv, vl);
    c30 = __riscv_vfmacc_vf_f32m2(c30, a[3 * rsa + p * csa], bv, vl);
    c40 = __riscv_vfmacc_vf_f32m2(c40, a[4 * rsa + p * csa], bv, vl);
    c50 = __riscv_vfmacc_vf_f32m2(c50, a[5 * rsa + p * csa], bv, vl);
  }

  __riscv_vse32_v_f32m2(c, c00, vl);
  __riscv_vse32_v_f32m2(c + rso, c10, vl);
  __riscv_vse32_v_f32m2(c + 2 * rso, c20, vl);
  __riscv_vse32_v_f32m2(c + 3 * rso, c30, vl);
  __riscv_vse32_v_f32m2(c + 4 * rso, c40, vl);
  __riscv_vse32_v_f32m2(c + 5 * rso, c50, vl);
}

/* Edge kernel for remainder tiles */
static inline void gemmsup_edge_f32_rvv(const float *a, const float *b,
                                        float *c, size_t mr, size_t nr,
                                        size_t kc, intptr_t rsa, intptr_t csa,
                                        intptr_t rsb, intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
    size_t vl = __riscv_vsetvl_e32m2(nr);
    /* Zero output */
    for (size_t j = 0; j < nr; j += vl) {
      vl = __riscv_vsetvl_e32m2(nr - j);
      __riscv_vse32_v_f32m2(c + i * rso + j, __riscv_vfmv_v_f_f32m2(0, vl), vl);
    }
    for (size_t p = 0; p < kc; p++) {
      float aip = a[i * rsa + p * csa];
      const float *brow = b + p * rsb;
      float *crow = c + i * rso;
      for (size_t j = 0; j < nr; j += vl) {
        vl = __riscv_vsetvl_e32m2(nr - j);
        vfloat32m2_t vo = __riscv_vle32_v_f32m2(crow + j, vl);
        vo = __riscv_vfmacc_vf_f32m2(vo, aip,
                                     __riscv_vle32_v_f32m2(brow + j, vl), vl);
        __riscv_vse32_v_f32m2(crow + j, vo, vl);
      }
    }
  }
}

static inline void gemmsup_f32_rvv(const float *a, const float *b, float *out,
                                   size_t M, size_t K, size_t N, intptr_t rsa,
                                   intptr_t csa, intptr_t rsb, intptr_t rso) {
  const size_t MR = 6;
  size_t NR = __riscv_vsetvlmax_e32m2();
  for (size_t i = 0; i < M; i += MR) {
    size_t mr = GEMMSUP_MIN(MR, M - i);
    for (size_t j = 0; j < N; j += NR) {
      size_t nr = GEMMSUP_MIN(NR, N - j);
      if (mr == MR) {
        gemmsup_ukernel_f32_6xVL(a + i * rsa, b + j, out + i * rso + j, K, nr,
                                 rsa, csa, rsb, rso);
      } else {
        gemmsup_edge_f32_rvv(a + i * rsa, b + j, out + i * rso + j, mr, nr, K,
                             rsa, csa, rsb, rso);
      }
    }
  }
}

/* =================================================================
   Float64 unpacked scalable micro-kernel (RVV)
   MR=4, NR=vsetvlmax_e64m2 (LMUL=m2)
   ================================================================= */

static inline void gemmsup_ukernel_f64_4xVL(const double *a, const double *b,
                                            double *c, size_t kc, size_t nr,
                                            intptr_t rsa, intptr_t csa,
                                            intptr_t rsb, intptr_t rso) {
  size_t vl = __riscv_vsetvl_e64m2(nr);

  vfloat64m2_t c00 = __riscv_vfmv_v_f_f64m2(0, vl);
  vfloat64m2_t c10 = __riscv_vfmv_v_f_f64m2(0, vl);
  vfloat64m2_t c20 = __riscv_vfmv_v_f_f64m2(0, vl);
  vfloat64m2_t c30 = __riscv_vfmv_v_f_f64m2(0, vl);

  for (size_t p = 0; p < kc; p++) {
    vfloat64m2_t bv = __riscv_vle64_v_f64m2(b + p * rsb, vl);

    c00 = __riscv_vfmacc_vf_f64m2(c00, a[0 * rsa + p * csa], bv, vl);
    c10 = __riscv_vfmacc_vf_f64m2(c10, a[1 * rsa + p * csa], bv, vl);
    c20 = __riscv_vfmacc_vf_f64m2(c20, a[2 * rsa + p * csa], bv, vl);
    c30 = __riscv_vfmacc_vf_f64m2(c30, a[3 * rsa + p * csa], bv, vl);
  }

  __riscv_vse64_v_f64m2(c, c00, vl);
  __riscv_vse64_v_f64m2(c + rso, c10, vl);
  __riscv_vse64_v_f64m2(c + 2 * rso, c20, vl);
  __riscv_vse64_v_f64m2(c + 3 * rso, c30, vl);
}

static inline void gemmsup_edge_f64_rvv(const double *a, const double *b,
                                        double *c, size_t mr, size_t nr,
                                        size_t kc, intptr_t rsa, intptr_t csa,
                                        intptr_t rsb, intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
    size_t vl;
    for (size_t j = 0; j < nr; j += vl) {
      vl = __riscv_vsetvl_e64m2(nr - j);
      __riscv_vse64_v_f64m2(c + i * rso + j, __riscv_vfmv_v_f_f64m2(0, vl), vl);
    }
    for (size_t p = 0; p < kc; p++) {
      double aip = a[i * rsa + p * csa];
      const double *brow = b + p * rsb;
      double *crow = c + i * rso;
      for (size_t j = 0; j < nr; j += vl) {
        vl = __riscv_vsetvl_e64m2(nr - j);
        vfloat64m2_t vo = __riscv_vle64_v_f64m2(crow + j, vl);
        vo = __riscv_vfmacc_vf_f64m2(vo, aip,
                                     __riscv_vle64_v_f64m2(brow + j, vl), vl);
        __riscv_vse64_v_f64m2(crow + j, vo, vl);
      }
    }
  }
}

static inline void gemmsup_f64_rvv(const double *a, const double *b,
                                   double *out, size_t M, size_t K, size_t N,
                                   intptr_t rsa, intptr_t csa, intptr_t rsb,
                                   intptr_t rso) {
  const size_t MR = 4;
  size_t NR = __riscv_vsetvlmax_e64m2();
  for (size_t i = 0; i < M; i += MR) {
    size_t mr = GEMMSUP_MIN(MR, M - i);
    for (size_t j = 0; j < N; j += NR) {
      size_t nr = GEMMSUP_MIN(NR, N - j);
      if (mr == MR) {
        gemmsup_ukernel_f64_4xVL(a + i * rsa, b + j, out + i * rso + j, K, nr,
                                 rsa, csa, rsb, rso);
      } else {
        gemmsup_edge_f64_rvv(a + i * rsa, b + j, out + i * rso + j, mr, nr, K,
                             rsa, csa, rsb, rso);
      }
    }
  }
}

#endif /* NUMC_GEMMSUP_RVV_H */
