#ifndef NUMC_GEMM_RVV_H
#define NUMC_GEMM_RVV_H

#include <riscv_vector.h>
#include <stdint.h>
#include <string.h>

// clang-format off

#define GEMM_MIN(a, b) ((a) < (b) ? (a) : (b))

/*
 * Cache-blocking parameters for RVV packed GEMM.
 * RISC-V Vector has scalable VLEN; NR is determined at runtime via vsetvl.
 * With LMUL=2 (m2), one vector register group holds VLEN*2/elem_bits elements.
 *   VLEN=128: f32 NR=8,  f64 NR=4
 *   VLEN=256: f32 NR=16, f64 NR=8
 *   VLEN=512: f32 NR=32, f64 NR=16
 *
 * f32 8xVL: 8 acc + 1 B = 9 m2 groups out of 16 available.
 *   KC x NR sliver in L1: 256 x 16 x 4 = 16KB < 32KB (VLEN=256)
 *   MC x KC panel  in L2: 128 x 256 x 4 = 128KB < 256KB
 *
 * f64 6xVL: 6 acc + 1 B = 7 m2 groups out of 16 available.
 *   KC x NR sliver in L1: 256 x 8 x 8 = 16KB < 32KB (VLEN=256)
 *   MC x KC panel  in L2: 96 x 256 x 8 = 196KB < 256KB
 *
 * Max NR for stack buffers: VLEN=1024 with m2 gives 64 f32 or 32 f64 elements.
 * We use 64 as the upper bound for tmp buffer sizing.
 */
#define GEMM_F32_MR 8
#define GEMM_F32_MC 128
#define GEMM_F32_KC 256
#define GEMM_F32_NC 4080

#define GEMM_F64_MR 6
#define GEMM_F64_MC 96
#define GEMM_F64_KC 256
#define GEMM_F64_NC 2048

#define GEMM_I32_MR 6
#define GEMM_I32_MC 72
#define GEMM_I32_KC 256

#define GEMM_I16_MR 6
#define GEMM_I16_MC 72
#define GEMM_I16_KC 512

#define GEMM_I64_MR 6
#define GEMM_I64_MC 72
#define GEMM_I64_KC 64

#define GEMM_I8_MR 6
#define GEMM_I8_MC 72

#define GEMM_OMP_THRESHOLD (1 << 23)

/* Maximum NR for tmp buffer sizing (VLEN=1024, LMUL=2) */
#define GEMM_RVV_MAX_VL_F32 64
#define GEMM_RVV_MAX_VL_F64 32
#define GEMM_RVV_MAX_VL_I32 64
#define GEMM_RVV_MAX_VL_I16 128
#define GEMM_RVV_MAX_VL_I64 32
#define GEMM_RVV_MAX_VL_I8  64

/* ── Runtime NR query helpers ──────────────────────────────────────────── */

static inline size_t gemm_nr_f32_rvv(void) {
  return __riscv_vsetvlmax_e32m2();
}

static inline size_t gemm_nr_f64_rvv(void) {
  return __riscv_vsetvlmax_e64m2();
}

static inline size_t gemm_nr_i32_rvv(void) {
  return __riscv_vsetvlmax_e32m2();
}

static inline size_t gemm_nr_i16_rvv(void) {
  return __riscv_vsetvlmax_e16m2();
}

static inline size_t gemm_nr_i64_rvv(void) {
  return __riscv_vsetvlmax_e64m2();
}

static inline size_t gemm_nr_i8_rvv(void) {
  /* i8 uses i32 accumulators, so NR matches i32 VL */
  return __riscv_vsetvlmax_e32m2();
}

/* ── Float32 packing routines ──────────────────────────────────────────── */

static inline void gemm_pack_b_f32_rvv(const float *b, float *packed,
                                        size_t kc, size_t nc, size_t nr,
                                        intptr_t rsb) {
  size_t jr = 0;
  for (; jr + nr <= nc; jr += nr) {
    float *dest = packed + jr * kc;
    for (size_t p = 0; p < kc; p++) {
      const float *src = b + p * rsb + jr;
      size_t vl = __riscv_vsetvl_e32m2(nr);
      vfloat32m2_t v = __riscv_vle32_v_f32m2(src, vl);
      __riscv_vse32_v_f32m2(dest + p * nr, v, vl);
    }
  }
  if (jr < nc) {
    float *dest = packed + jr * kc;
    size_t rem = nc - jr;
    for (size_t p = 0; p < kc; p++) {
      const float *src = b + p * rsb + jr;
      size_t j = 0;
      for (; j < rem; j++)
        dest[p * nr + j] = src[j];
      for (; j < nr; j++)
        dest[p * nr + j] = 0.0f;
    }
  }
}

static inline void gemm_pack_a_f32_rvv(const float *a, float *packed,
                                        size_t mc, size_t kc, intptr_t rsa,
                                        intptr_t csa) {
  size_t ir = 0;
  for (; ir + GEMM_F32_MR <= mc; ir += GEMM_F32_MR) {
    float *dest = packed + ir * kc;
    for (size_t p = 0; p < kc; p++) {
      for (size_t i = 0; i < GEMM_F32_MR; i++)
        dest[p * GEMM_F32_MR + i] = a[(ir + i) * rsa + p * csa];
    }
  }
  if (ir < mc) {
    float *dest = packed + ir * kc;
    size_t rem = mc - ir;
    for (size_t p = 0; p < kc; p++) {
      size_t i = 0;
      for (; i < rem; i++)
        dest[p * GEMM_F32_MR + i] = a[(ir + i) * rsa + p * csa];
      for (; i < GEMM_F32_MR; i++)
        dest[p * GEMM_F32_MR + i] = 0.0f;
    }
  }
}

/* ── Float64 packing routines ──────────────────────────────────────────── */

static inline void gemm_pack_b_f64_rvv(const double *b, double *packed,
                                        size_t kc, size_t nc, size_t nr,
                                        intptr_t rsb) {
  size_t jr = 0;
  for (; jr + nr <= nc; jr += nr) {
    double *dest = packed + jr * kc;
    for (size_t p = 0; p < kc; p++) {
      const double *src = b + p * rsb + jr;
      size_t vl = __riscv_vsetvl_e64m2(nr);
      vfloat64m2_t v = __riscv_vle64_v_f64m2(src, vl);
      __riscv_vse64_v_f64m2(dest + p * nr, v, vl);
    }
  }
  if (jr < nc) {
    double *dest = packed + jr * kc;
    size_t rem = nc - jr;
    for (size_t p = 0; p < kc; p++) {
      const double *src = b + p * rsb + jr;
      size_t j = 0;
      for (; j < rem; j++)
        dest[p * nr + j] = src[j];
      for (; j < nr; j++)
        dest[p * nr + j] = 0.0;
    }
  }
}

static inline void gemm_pack_a_f64_rvv(const double *a, double *packed,
                                        size_t mc, size_t kc, intptr_t rsa,
                                        intptr_t csa) {
  size_t ir = 0;
  for (; ir + GEMM_F64_MR <= mc; ir += GEMM_F64_MR) {
    double *dest = packed + ir * kc;
    for (size_t p = 0; p < kc; p++) {
      for (size_t i = 0; i < GEMM_F64_MR; i++)
        dest[p * GEMM_F64_MR + i] = a[(ir + i) * rsa + p * csa];
    }
  }
  if (ir < mc) {
    double *dest = packed + ir * kc;
    size_t rem = mc - ir;
    for (size_t p = 0; p < kc; p++) {
      size_t i = 0;
      for (; i < rem; i++)
        dest[p * GEMM_F64_MR + i] = a[(ir + i) * rsa + p * csa];
      for (; i < GEMM_F64_MR; i++)
        dest[p * GEMM_F64_MR + i] = 0.0;
    }
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   Float32: 8xVL micro-kernel (vfmacc_vf — scalar-vector FMA, no broadcast)
   8 acc m2 groups + 1 B m2 group = 9 m2 groups out of 16 available.
   ═══════════════════════════════════════════════════════════════════════════
 */

/* One K-iteration: load 1 B vector (VL elements), scalar-FMA into 8 rows. */
#define GEMM_F32_K_ITER(ap, bp, nr)                                            \
  do {                                                                         \
    vfloat32m2_t b0 = __riscv_vle32_v_f32m2(bp, nr);                          \
    c0 = __riscv_vfmacc_vf_f32m2(c0, (ap)[0], b0, nr);                        \
    c1 = __riscv_vfmacc_vf_f32m2(c1, (ap)[1], b0, nr);                        \
    c2 = __riscv_vfmacc_vf_f32m2(c2, (ap)[2], b0, nr);                        \
    c3 = __riscv_vfmacc_vf_f32m2(c3, (ap)[3], b0, nr);                        \
    c4 = __riscv_vfmacc_vf_f32m2(c4, (ap)[4], b0, nr);                        \
    c5 = __riscv_vfmacc_vf_f32m2(c5, (ap)[5], b0, nr);                        \
    c6 = __riscv_vfmacc_vf_f32m2(c6, (ap)[6], b0, nr);                        \
    c7 = __riscv_vfmacc_vf_f32m2(c7, (ap)[7], b0, nr);                        \
  } while (0)

static inline void gemm_ukernel_f32_8xVL(const float *a, const float *b,
                                           float *c, size_t kc, intptr_t rsa,
                                           intptr_t csa, intptr_t rsb,
                                           size_t nr, intptr_t rso, int first) {
  size_t vl = __riscv_vsetvl_e32m2(nr);

  vfloat32m2_t c0, c1, c2, c3, c4, c5, c6, c7;

  if (first) {
    c0 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    c1 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    c2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    c3 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    c4 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    c5 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    c6 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    c7 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
  } else {
    c0 = __riscv_vle32_v_f32m2(c, vl);
    c1 = __riscv_vle32_v_f32m2(c + rso, vl);
    c2 = __riscv_vle32_v_f32m2(c + 2 * rso, vl);
    c3 = __riscv_vle32_v_f32m2(c + 3 * rso, vl);
    c4 = __riscv_vle32_v_f32m2(c + 4 * rso, vl);
    c5 = __riscv_vle32_v_f32m2(c + 5 * rso, vl);
    c6 = __riscv_vle32_v_f32m2(c + 6 * rso, vl);
    c7 = __riscv_vle32_v_f32m2(c + 7 * rso, vl);
  }

  const float *ap = a;
  const float *bp = b;
  size_t k_iter = kc / 8;
  size_t k_left = kc % 8;

  for (size_t ki = 0; ki < k_iter; ki++) {
    GEMM_F32_K_ITER(ap, bp, vl);
    ap += csa; bp += rsb;
    GEMM_F32_K_ITER(ap, bp, vl);
    ap += csa; bp += rsb;
    GEMM_F32_K_ITER(ap, bp, vl);
    ap += csa; bp += rsb;
    GEMM_F32_K_ITER(ap, bp, vl);
    ap += csa; bp += rsb;
    GEMM_F32_K_ITER(ap, bp, vl);
    ap += csa; bp += rsb;
    GEMM_F32_K_ITER(ap, bp, vl);
    ap += csa; bp += rsb;
    GEMM_F32_K_ITER(ap, bp, vl);
    ap += csa; bp += rsb;
    GEMM_F32_K_ITER(ap, bp, vl);
    ap += csa; bp += rsb;
  }
  for (size_t ki = 0; ki < k_left; ki++) {
    GEMM_F32_K_ITER(ap, bp, vl);
    ap += csa; bp += rsb;
  }

  __riscv_vse32_v_f32m2(c, c0, vl);
  __riscv_vse32_v_f32m2(c + rso, c1, vl);
  __riscv_vse32_v_f32m2(c + 2 * rso, c2, vl);
  __riscv_vse32_v_f32m2(c + 3 * rso, c3, vl);
  __riscv_vse32_v_f32m2(c + 4 * rso, c4, vl);
  __riscv_vse32_v_f32m2(c + 5 * rso, c5, vl);
  __riscv_vse32_v_f32m2(c + 6 * rso, c6, vl);
  __riscv_vse32_v_f32m2(c + 7 * rso, c7, vl);
}

#undef GEMM_F32_K_ITER

static inline void gemm_edge_f32_rvv(const float *a, const float *b, float *c,
                                      size_t mr, size_t nr, size_t kc,
                                      intptr_t rsa, intptr_t csa, intptr_t rsb,
                                      intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
    for (size_t p = 0; p < kc; p++) {
      float aip = a[i * rsa + p * csa];
      const float *brow = b + p * rsb;
      float *crow = c + i * rso;
      size_t avl = nr;
      size_t j = 0;
      while (avl > 0) {
        size_t vl = __riscv_vsetvl_e32m2(avl);
        vfloat32m2_t vo = __riscv_vle32_v_f32m2(crow + j, vl);
        vfloat32m2_t vb = __riscv_vle32_v_f32m2(brow + j, vl);
        vo = __riscv_vfmacc_vf_f32m2(vo, aip, vb, vl);
        __riscv_vse32_v_f32m2(crow + j, vo, vl);
        j += vl;
        avl -= vl;
      }
    }
  }
}

static inline void gemm_f32_rvv(const float *a, const float *b, float *out,
                                 size_t m_dim, size_t k_dim, size_t n_dim,
                                 intptr_t rsa, intptr_t csa, intptr_t rsb,
                                 intptr_t rso) {
  size_t nr = gemm_nr_f32_rvv();
  size_t nc_max = GEMM_MIN(GEMM_F32_NC, n_dim);
  float *packed_b = (float *)numc_malloc(
      16, GEMM_F32_KC * (nc_max + nr) * sizeof(float));
  if (!packed_b)
    return;

  for (size_t jc = 0; jc < n_dim; jc += GEMM_F32_NC) {
    size_t nc = GEMM_MIN(GEMM_F32_NC, n_dim - jc);

    for (size_t pc = 0; pc < k_dim; pc += GEMM_F32_KC) {
      size_t kc = GEMM_MIN(GEMM_F32_KC, k_dim - pc);
      int first = (pc == 0);

      gemm_pack_b_f32_rvv(b + pc * rsb + jc, packed_b, kc, nc, nr, rsb);

      size_t n_ic = (m_dim + GEMM_F32_MC - 1) / GEMM_F32_MC;
      size_t n_jr = (nc + nr - 1) / nr;
      size_t n_tasks = n_ic * n_jr;

#ifdef _OPENMP
#pragma omp parallel if ((uint64_t)m_dim * kc * nc > GEMM_OMP_THRESHOLD)
      {
        NUMC_ALIGNAS(16) float packed_a[GEMM_F32_MC * GEMM_F32_KC];
        size_t last_ic = (size_t)-1;

#pragma omp for schedule(static)
        for (size_t task = 0; task < n_tasks; task++) {
          size_t ic = (task / n_jr) * GEMM_F32_MC;
          size_t jr = (task % n_jr) * nr;
          size_t mc = GEMM_MIN(GEMM_F32_MC, m_dim - ic);
          size_t nr_cur = GEMM_MIN(nr, nc - jr);

          if (ic != last_ic) {
            gemm_pack_a_f32_rvv(a + ic * rsa + pc * csa, packed_a, mc, kc,
                                rsa, csa);
            last_ic = ic;
          }

          for (size_t ir = 0; ir < mc; ir += GEMM_F32_MR) {
            size_t mr_cur = GEMM_MIN(GEMM_F32_MR, mc - ir);
            if (mr_cur == GEMM_F32_MR && nr_cur == nr) {
              gemm_ukernel_f32_8xVL(
                  packed_a + ir * kc, packed_b + jr * kc,
                  out + (ic + ir) * rso + (jc + jr), kc, 1, GEMM_F32_MR,
                  nr, nr, rso, first);
            } else {
              NUMC_ALIGNAS(16) float tmp[GEMM_F32_MR * GEMM_RVV_MAX_VL_F32];
              gemm_ukernel_f32_8xVL(packed_a + ir * kc, packed_b + jr * kc,
                                    tmp, kc, 1, GEMM_F32_MR, nr, nr, nr, 1);
              float *dst = out + (ic + ir) * rso + (jc + jr);
              if (first) {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] = tmp[ii * nr + jj];
              } else {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] += tmp[ii * nr + jj];
              }
            }
          }
        }
      }
#else
      {
        NUMC_ALIGNAS(16) float packed_a[GEMM_F32_MC * GEMM_F32_KC];

        for (size_t task = 0; task < n_tasks; task++) {
          size_t ic = (task / n_jr) * GEMM_F32_MC;
          size_t jr = (task % n_jr) * nr;
          size_t mc = GEMM_MIN(GEMM_F32_MC, m_dim - ic);
          size_t nr_cur = GEMM_MIN(nr, nc - jr);

          if (task % n_jr == 0)
            gemm_pack_a_f32_rvv(a + ic * rsa + pc * csa, packed_a, mc, kc,
                                rsa, csa);

          for (size_t ir = 0; ir < mc; ir += GEMM_F32_MR) {
            size_t mr_cur = GEMM_MIN(GEMM_F32_MR, mc - ir);
            if (mr_cur == GEMM_F32_MR && nr_cur == nr) {
              gemm_ukernel_f32_8xVL(
                  packed_a + ir * kc, packed_b + jr * kc,
                  out + (ic + ir) * rso + (jc + jr), kc, 1, GEMM_F32_MR,
                  nr, nr, rso, first);
            } else {
              NUMC_ALIGNAS(16) float tmp[GEMM_F32_MR * GEMM_RVV_MAX_VL_F32];
              gemm_ukernel_f32_8xVL(packed_a + ir * kc, packed_b + jr * kc,
                                    tmp, kc, 1, GEMM_F32_MR, nr, nr, nr, 1);
              float *dst = out + (ic + ir) * rso + (jc + jr);
              if (first) {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] = tmp[ii * nr + jj];
              } else {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] += tmp[ii * nr + jj];
              }
            }
          }
        }
      }
#endif
    }
  }

  numc_free(packed_b);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Float64: 6xVL micro-kernel (vfmacc_vf — scalar-vector FMA)
   6 acc m2 groups + 1 B m2 group = 7 m2 groups out of 16 available.
   ═══════════════════════════════════════════════════════════════════════════
 */

#define GEMM_F64_K_ITER(ap, bp, nr)                                            \
  do {                                                                         \
    vfloat64m2_t b0 = __riscv_vle64_v_f64m2(bp, nr);                          \
    c0 = __riscv_vfmacc_vf_f64m2(c0, (ap)[0], b0, nr);                        \
    c1 = __riscv_vfmacc_vf_f64m2(c1, (ap)[1], b0, nr);                        \
    c2 = __riscv_vfmacc_vf_f64m2(c2, (ap)[2], b0, nr);                        \
    c3 = __riscv_vfmacc_vf_f64m2(c3, (ap)[3], b0, nr);                        \
    c4 = __riscv_vfmacc_vf_f64m2(c4, (ap)[4], b0, nr);                        \
    c5 = __riscv_vfmacc_vf_f64m2(c5, (ap)[5], b0, nr);                        \
  } while (0)

static inline void gemm_ukernel_f64_6xVL(const double *a, const double *b,
                                           double *c, size_t kc, intptr_t rsa,
                                           intptr_t csa, intptr_t rsb,
                                           size_t nr, intptr_t rso, int first) {
  size_t vl = __riscv_vsetvl_e64m2(nr);

  vfloat64m2_t c0, c1, c2, c3, c4, c5;

  if (first) {
    c0 = __riscv_vfmv_v_f_f64m2(0.0, vl);
    c1 = __riscv_vfmv_v_f_f64m2(0.0, vl);
    c2 = __riscv_vfmv_v_f_f64m2(0.0, vl);
    c3 = __riscv_vfmv_v_f_f64m2(0.0, vl);
    c4 = __riscv_vfmv_v_f_f64m2(0.0, vl);
    c5 = __riscv_vfmv_v_f_f64m2(0.0, vl);
  } else {
    c0 = __riscv_vle64_v_f64m2(c, vl);
    c1 = __riscv_vle64_v_f64m2(c + rso, vl);
    c2 = __riscv_vle64_v_f64m2(c + 2 * rso, vl);
    c3 = __riscv_vle64_v_f64m2(c + 3 * rso, vl);
    c4 = __riscv_vle64_v_f64m2(c + 4 * rso, vl);
    c5 = __riscv_vle64_v_f64m2(c + 5 * rso, vl);
  }

  const double *ap = a;
  const double *bp = b;
  size_t k_iter = kc / 8;
  size_t k_left = kc % 8;

  for (size_t ki = 0; ki < k_iter; ki++) {
    GEMM_F64_K_ITER(ap, bp, vl);
    ap += csa; bp += rsb;
    GEMM_F64_K_ITER(ap, bp, vl);
    ap += csa; bp += rsb;
    GEMM_F64_K_ITER(ap, bp, vl);
    ap += csa; bp += rsb;
    GEMM_F64_K_ITER(ap, bp, vl);
    ap += csa; bp += rsb;
    GEMM_F64_K_ITER(ap, bp, vl);
    ap += csa; bp += rsb;
    GEMM_F64_K_ITER(ap, bp, vl);
    ap += csa; bp += rsb;
    GEMM_F64_K_ITER(ap, bp, vl);
    ap += csa; bp += rsb;
    GEMM_F64_K_ITER(ap, bp, vl);
    ap += csa; bp += rsb;
  }
  for (size_t ki = 0; ki < k_left; ki++) {
    GEMM_F64_K_ITER(ap, bp, vl);
    ap += csa; bp += rsb;
  }

  __riscv_vse64_v_f64m2(c, c0, vl);
  __riscv_vse64_v_f64m2(c + rso, c1, vl);
  __riscv_vse64_v_f64m2(c + 2 * rso, c2, vl);
  __riscv_vse64_v_f64m2(c + 3 * rso, c3, vl);
  __riscv_vse64_v_f64m2(c + 4 * rso, c4, vl);
  __riscv_vse64_v_f64m2(c + 5 * rso, c5, vl);
}

#undef GEMM_F64_K_ITER

static inline void gemm_edge_f64_rvv(const double *a, const double *b,
                                      double *c, size_t mr, size_t nr,
                                      size_t kc, intptr_t rsa, intptr_t csa,
                                      intptr_t rsb, intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
    for (size_t p = 0; p < kc; p++) {
      double aip = a[i * rsa + p * csa];
      const double *brow = b + p * rsb;
      double *crow = c + i * rso;
      size_t avl = nr;
      size_t j = 0;
      while (avl > 0) {
        size_t vl = __riscv_vsetvl_e64m2(avl);
        vfloat64m2_t vo = __riscv_vle64_v_f64m2(crow + j, vl);
        vfloat64m2_t vb = __riscv_vle64_v_f64m2(brow + j, vl);
        vo = __riscv_vfmacc_vf_f64m2(vo, aip, vb, vl);
        __riscv_vse64_v_f64m2(crow + j, vo, vl);
        j += vl;
        avl -= vl;
      }
    }
  }
}

static inline void gemm_f64_rvv(const double *a, const double *b, double *out,
                                 size_t m_dim, size_t k_dim, size_t n_dim,
                                 intptr_t rsa, intptr_t csa, intptr_t rsb,
                                 intptr_t rso) {
  size_t nr = gemm_nr_f64_rvv();
  size_t nc_max = GEMM_MIN(GEMM_F64_NC, n_dim);
  double *packed_b = (double *)numc_malloc(
      16, GEMM_F64_KC * (nc_max + nr) * sizeof(double));
  if (!packed_b)
    return;

  for (size_t jc = 0; jc < n_dim; jc += GEMM_F64_NC) {
    size_t nc = GEMM_MIN(GEMM_F64_NC, n_dim - jc);

    for (size_t pc = 0; pc < k_dim; pc += GEMM_F64_KC) {
      size_t kc = GEMM_MIN(GEMM_F64_KC, k_dim - pc);
      int first = (pc == 0);

      gemm_pack_b_f64_rvv(b + pc * rsb + jc, packed_b, kc, nc, nr, rsb);

      size_t n_ic = (m_dim + GEMM_F64_MC - 1) / GEMM_F64_MC;
      size_t n_jr = (nc + nr - 1) / nr;
      size_t n_tasks = n_ic * n_jr;

#ifdef _OPENMP
#pragma omp parallel if ((uint64_t)m_dim * kc * nc > GEMM_OMP_THRESHOLD)
      {
        NUMC_ALIGNAS(16) double packed_a[GEMM_F64_MC * GEMM_F64_KC];
        size_t last_ic = (size_t)-1;

#pragma omp for schedule(static)
        for (size_t task = 0; task < n_tasks; task++) {
          size_t ic = (task / n_jr) * GEMM_F64_MC;
          size_t jr = (task % n_jr) * nr;
          size_t mc = GEMM_MIN(GEMM_F64_MC, m_dim - ic);
          size_t nr_cur = GEMM_MIN(nr, nc - jr);

          if (ic != last_ic) {
            gemm_pack_a_f64_rvv(a + ic * rsa + pc * csa, packed_a, mc, kc,
                                rsa, csa);
            last_ic = ic;
          }

          for (size_t ir = 0; ir < mc; ir += GEMM_F64_MR) {
            size_t mr_cur = GEMM_MIN(GEMM_F64_MR, mc - ir);
            if (mr_cur == GEMM_F64_MR && nr_cur == nr) {
              gemm_ukernel_f64_6xVL(
                  packed_a + ir * kc, packed_b + jr * kc,
                  out + (ic + ir) * rso + (jc + jr), kc, 1, GEMM_F64_MR,
                  nr, nr, rso, first);
            } else {
              NUMC_ALIGNAS(16) double tmp[GEMM_F64_MR * GEMM_RVV_MAX_VL_F64];
              gemm_ukernel_f64_6xVL(packed_a + ir * kc, packed_b + jr * kc,
                                    tmp, kc, 1, GEMM_F64_MR, nr, nr, nr, 1);
              double *dst = out + (ic + ir) * rso + (jc + jr);
              if (first) {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] = tmp[ii * nr + jj];
              } else {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] += tmp[ii * nr + jj];
              }
            }
          }
        }
      }
#else
      {
        NUMC_ALIGNAS(16) double packed_a[GEMM_F64_MC * GEMM_F64_KC];

        for (size_t task = 0; task < n_tasks; task++) {
          size_t ic = (task / n_jr) * GEMM_F64_MC;
          size_t jr = (task % n_jr) * nr;
          size_t mc = GEMM_MIN(GEMM_F64_MC, m_dim - ic);
          size_t nr_cur = GEMM_MIN(nr, nc - jr);

          if (task % n_jr == 0)
            gemm_pack_a_f64_rvv(a + ic * rsa + pc * csa, packed_a, mc, kc,
                                rsa, csa);

          for (size_t ir = 0; ir < mc; ir += GEMM_F64_MR) {
            size_t mr_cur = GEMM_MIN(GEMM_F64_MR, mc - ir);
            if (mr_cur == GEMM_F64_MR && nr_cur == nr) {
              gemm_ukernel_f64_6xVL(
                  packed_a + ir * kc, packed_b + jr * kc,
                  out + (ic + ir) * rso + (jc + jr), kc, 1, GEMM_F64_MR,
                  nr, nr, rso, first);
            } else {
              NUMC_ALIGNAS(16) double tmp[GEMM_F64_MR * GEMM_RVV_MAX_VL_F64];
              gemm_ukernel_f64_6xVL(packed_a + ir * kc, packed_b + jr * kc,
                                    tmp, kc, 1, GEMM_F64_MR, nr, nr, nr, 1);
              double *dst = out + (ic + ir) * rso + (jc + jr);
              if (first) {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] = tmp[ii * nr + jj];
              } else {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] += tmp[ii * nr + jj];
              }
            }
          }
        }
      }
#endif
    }
  }

  numc_free(packed_b);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Int32/Uint32: 6xVL unpacked micro-kernel (vmacc_vx_i32m2)
   vmacc_vx: acc += scalar * b_vec — identical low bits for signed/unsigned.
   ═══════════════════════════════════════════════════════════════════════════
 */

static inline void gemm_ukernel_i32_6xVL(const int32_t *a, const int32_t *b,
                                           int32_t *c, size_t kc, intptr_t rsa,
                                           intptr_t csa, intptr_t rsb,
                                           intptr_t rso) {
  size_t nr = gemm_nr_i32_rvv();
  size_t vl = __riscv_vsetvl_e32m2(nr);

  vint32m2_t c0 = __riscv_vle32_v_i32m2(c, vl);
  vint32m2_t c1 = __riscv_vle32_v_i32m2(c + rso, vl);
  vint32m2_t c2 = __riscv_vle32_v_i32m2(c + 2 * rso, vl);
  vint32m2_t c3 = __riscv_vle32_v_i32m2(c + 3 * rso, vl);
  vint32m2_t c4 = __riscv_vle32_v_i32m2(c + 4 * rso, vl);
  vint32m2_t c5 = __riscv_vle32_v_i32m2(c + 5 * rso, vl);

  for (size_t p = 0; p < kc; p++) {
    const int32_t *bp = b + p * rsb;
    vint32m2_t b0 = __riscv_vle32_v_i32m2(bp, vl);
    c0 = __riscv_vmacc_vx_i32m2(c0, a[0 * rsa + p * csa], b0, vl);
    c1 = __riscv_vmacc_vx_i32m2(c1, a[1 * rsa + p * csa], b0, vl);
    c2 = __riscv_vmacc_vx_i32m2(c2, a[2 * rsa + p * csa], b0, vl);
    c3 = __riscv_vmacc_vx_i32m2(c3, a[3 * rsa + p * csa], b0, vl);
    c4 = __riscv_vmacc_vx_i32m2(c4, a[4 * rsa + p * csa], b0, vl);
    c5 = __riscv_vmacc_vx_i32m2(c5, a[5 * rsa + p * csa], b0, vl);
  }

  __riscv_vse32_v_i32m2(c, c0, vl);
  __riscv_vse32_v_i32m2(c + rso, c1, vl);
  __riscv_vse32_v_i32m2(c + 2 * rso, c2, vl);
  __riscv_vse32_v_i32m2(c + 3 * rso, c3, vl);
  __riscv_vse32_v_i32m2(c + 4 * rso, c4, vl);
  __riscv_vse32_v_i32m2(c + 5 * rso, c5, vl);
}

static inline void gemm_edge_i32_rvv(const int32_t *a, const int32_t *b,
                                      int32_t *c, size_t mr, size_t nr,
                                      size_t kc, intptr_t rsa, intptr_t csa,
                                      intptr_t rsb, intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
    for (size_t p = 0; p < kc; p++) {
      int32_t aip = a[i * rsa + p * csa];
      const int32_t *brow = b + p * rsb;
      int32_t *crow = c + i * rso;
      size_t avl = nr;
      size_t j = 0;
      while (avl > 0) {
        size_t vl = __riscv_vsetvl_e32m2(avl);
        vint32m2_t vo = __riscv_vle32_v_i32m2(crow + j, vl);
        vint32m2_t vb = __riscv_vle32_v_i32m2(brow + j, vl);
        vo = __riscv_vmacc_vx_i32m2(vo, aip, vb, vl);
        __riscv_vse32_v_i32m2(crow + j, vo, vl);
        j += vl;
        avl -= vl;
      }
    }
  }
}

static inline void gemm_i32_rvv(const int32_t *a, const int32_t *b,
                                 int32_t *out, size_t m_dim, size_t k_dim,
                                 size_t n_dim, intptr_t rsa, intptr_t csa,
                                 intptr_t rsb, intptr_t rso) {
  size_t nr = gemm_nr_i32_rvv();

  for (size_t i = 0; i < m_dim; i++)
    memset(out + i * rso, 0, n_dim * sizeof(int32_t));

  for (size_t pc = 0; pc < k_dim; pc += GEMM_I32_KC) {
    size_t kc = GEMM_MIN(GEMM_I32_KC, k_dim - pc);
#ifdef _OPENMP
#pragma omp parallel for schedule(                                             \
        static) if (m_dim * n_dim * sizeof(int32_t) > GEMM_OMP_THRESHOLD)
#endif
    for (size_t ic = 0; ic < m_dim; ic += GEMM_I32_MC) {
      size_t mc = GEMM_MIN(GEMM_I32_MC, m_dim - ic);
      size_t jr = 0;
      for (; jr + nr <= n_dim; jr += nr) {
        size_t ir = 0;
        for (; ir + GEMM_I32_MR <= mc; ir += GEMM_I32_MR)
          gemm_ukernel_i32_6xVL(a + (ic + ir) * rsa + pc * csa,
                                b + pc * rsb + jr, out + (ic + ir) * rso + jr,
                                kc, rsa, csa, rsb, rso);
        if (ir < mc)
          gemm_edge_i32_rvv(a + (ic + ir) * rsa + pc * csa, b + pc * rsb + jr,
                            out + (ic + ir) * rso + jr, mc - ir, nr, kc,
                            rsa, csa, rsb, rso);
      }
      if (jr < n_dim)
        gemm_edge_i32_rvv(a + ic * rsa + pc * csa, b + pc * rsb + jr,
                          out + ic * rso + jr, mc, n_dim - jr, kc, rsa, csa,
                          rsb, rso);
    }
  }
}

/* Uint32: identical bit-level operations as int32 */
static inline void gemm_u32_rvv(const uint32_t *a, const uint32_t *b,
                                 uint32_t *out, size_t m_dim, size_t k_dim,
                                 size_t n_dim, intptr_t rsa, intptr_t csa,
                                 intptr_t rsb, intptr_t rso) {
  gemm_i32_rvv((const int32_t *)a, (const int32_t *)b, (int32_t *)out, m_dim,
               k_dim, n_dim, rsa, csa, rsb, rso);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Int16/Uint16: 6xVL unpacked micro-kernel (vmacc_vx_i16m2)
   Same-width accumulation — matches the i32 overflow trade-off.
   ═══════════════════════════════════════════════════════════════════════════
 */

static inline void gemm_ukernel_i16_6xVL(const int16_t *a, const int16_t *b,
                                           int16_t *c, size_t kc, intptr_t rsa,
                                           intptr_t csa, intptr_t rsb,
                                           intptr_t rso) {
  size_t nr = gemm_nr_i16_rvv();
  size_t vl = __riscv_vsetvl_e16m2(nr);

  vint16m2_t c0 = __riscv_vle16_v_i16m2(c, vl);
  vint16m2_t c1 = __riscv_vle16_v_i16m2(c + rso, vl);
  vint16m2_t c2 = __riscv_vle16_v_i16m2(c + 2 * rso, vl);
  vint16m2_t c3 = __riscv_vle16_v_i16m2(c + 3 * rso, vl);
  vint16m2_t c4 = __riscv_vle16_v_i16m2(c + 4 * rso, vl);
  vint16m2_t c5 = __riscv_vle16_v_i16m2(c + 5 * rso, vl);

  for (size_t p = 0; p < kc; p++) {
    const int16_t *bp = b + p * rsb;
    vint16m2_t b0 = __riscv_vle16_v_i16m2(bp, vl);
    c0 = __riscv_vmacc_vx_i16m2(c0, a[0 * rsa + p * csa], b0, vl);
    c1 = __riscv_vmacc_vx_i16m2(c1, a[1 * rsa + p * csa], b0, vl);
    c2 = __riscv_vmacc_vx_i16m2(c2, a[2 * rsa + p * csa], b0, vl);
    c3 = __riscv_vmacc_vx_i16m2(c3, a[3 * rsa + p * csa], b0, vl);
    c4 = __riscv_vmacc_vx_i16m2(c4, a[4 * rsa + p * csa], b0, vl);
    c5 = __riscv_vmacc_vx_i16m2(c5, a[5 * rsa + p * csa], b0, vl);
  }

  __riscv_vse16_v_i16m2(c, c0, vl);
  __riscv_vse16_v_i16m2(c + rso, c1, vl);
  __riscv_vse16_v_i16m2(c + 2 * rso, c2, vl);
  __riscv_vse16_v_i16m2(c + 3 * rso, c3, vl);
  __riscv_vse16_v_i16m2(c + 4 * rso, c4, vl);
  __riscv_vse16_v_i16m2(c + 5 * rso, c5, vl);
}

static inline void gemm_edge_i16_rvv(const int16_t *a, const int16_t *b,
                                      int16_t *c, size_t mr, size_t nr,
                                      size_t kc, intptr_t rsa, intptr_t csa,
                                      intptr_t rsb, intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
    for (size_t p = 0; p < kc; p++) {
      int16_t aip = a[i * rsa + p * csa];
      const int16_t *brow = b + p * rsb;
      int16_t *crow = c + i * rso;
      size_t avl = nr;
      size_t j = 0;
      while (avl > 0) {
        size_t vl = __riscv_vsetvl_e16m2(avl);
        vint16m2_t vo = __riscv_vle16_v_i16m2(crow + j, vl);
        vint16m2_t vb = __riscv_vle16_v_i16m2(brow + j, vl);
        vo = __riscv_vmacc_vx_i16m2(vo, aip, vb, vl);
        __riscv_vse16_v_i16m2(crow + j, vo, vl);
        j += vl;
        avl -= vl;
      }
    }
  }
}

static inline void gemm_i16_rvv(const int16_t *a, const int16_t *b,
                                 int16_t *out, size_t m_dim, size_t k_dim,
                                 size_t n_dim, intptr_t rsa, intptr_t csa,
                                 intptr_t rsb, intptr_t rso) {
  size_t nr = gemm_nr_i16_rvv();

  for (size_t i = 0; i < m_dim; i++)
    memset(out + i * rso, 0, n_dim * sizeof(int16_t));

  for (size_t pc = 0; pc < k_dim; pc += GEMM_I16_KC) {
    size_t kc = GEMM_MIN(GEMM_I16_KC, k_dim - pc);
#ifdef _OPENMP
#pragma omp parallel for schedule(                                             \
        static) if (m_dim * n_dim * sizeof(int16_t) > GEMM_OMP_THRESHOLD)
#endif
    for (size_t ic = 0; ic < m_dim; ic += GEMM_I16_MC) {
      size_t mc = GEMM_MIN(GEMM_I16_MC, m_dim - ic);
      size_t jr = 0;
      for (; jr + nr <= n_dim; jr += nr) {
        size_t ir = 0;
        for (; ir + GEMM_I16_MR <= mc; ir += GEMM_I16_MR)
          gemm_ukernel_i16_6xVL(a + (ic + ir) * rsa + pc * csa,
                                b + pc * rsb + jr, out + (ic + ir) * rso + jr,
                                kc, rsa, csa, rsb, rso);
        if (ir < mc)
          gemm_edge_i16_rvv(a + (ic + ir) * rsa + pc * csa, b + pc * rsb + jr,
                            out + (ic + ir) * rso + jr, mc - ir, nr, kc,
                            rsa, csa, rsb, rso);
      }
      if (jr < n_dim)
        gemm_edge_i16_rvv(a + ic * rsa + pc * csa, b + pc * rsb + jr,
                          out + ic * rso + jr, mc, n_dim - jr, kc, rsa, csa,
                          rsb, rso);
    }
  }
}

/* Uint16: identical bit-level operations as int16 */
static inline void gemm_u16_rvv(const uint16_t *a, const uint16_t *b,
                                 uint16_t *out, size_t m_dim, size_t k_dim,
                                 size_t n_dim, intptr_t rsa, intptr_t csa,
                                 intptr_t rsb, intptr_t rso) {
  gemm_i16_rvv((const int16_t *)a, (const int16_t *)b, (int16_t *)out, m_dim,
               k_dim, n_dim, rsa, csa, rsb, rso);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Int64/Uint64: 6xVL unpacked micro-kernel (vmacc_vx_i64m2)
   RVV has native 64-bit integer multiply — unlike NEON and AVX2!
   ═══════════════════════════════════════════════════════════════════════════
 */

static inline void gemm_ukernel_i64_6xVL(const int64_t *a, const int64_t *b,
                                           int64_t *c, size_t kc, intptr_t rsa,
                                           intptr_t csa, intptr_t rsb,
                                           intptr_t rso) {
  size_t nr = gemm_nr_i64_rvv();
  size_t vl = __riscv_vsetvl_e64m2(nr);

  vint64m2_t c0 = __riscv_vle64_v_i64m2(c, vl);
  vint64m2_t c1 = __riscv_vle64_v_i64m2(c + rso, vl);
  vint64m2_t c2 = __riscv_vle64_v_i64m2(c + 2 * rso, vl);
  vint64m2_t c3 = __riscv_vle64_v_i64m2(c + 3 * rso, vl);
  vint64m2_t c4 = __riscv_vle64_v_i64m2(c + 4 * rso, vl);
  vint64m2_t c5 = __riscv_vle64_v_i64m2(c + 5 * rso, vl);

  for (size_t p = 0; p < kc; p++) {
    const int64_t *bp = b + p * rsb;
    vint64m2_t b0 = __riscv_vle64_v_i64m2(bp, vl);
    c0 = __riscv_vmacc_vx_i64m2(c0, a[0 * rsa + p * csa], b0, vl);
    c1 = __riscv_vmacc_vx_i64m2(c1, a[1 * rsa + p * csa], b0, vl);
    c2 = __riscv_vmacc_vx_i64m2(c2, a[2 * rsa + p * csa], b0, vl);
    c3 = __riscv_vmacc_vx_i64m2(c3, a[3 * rsa + p * csa], b0, vl);
    c4 = __riscv_vmacc_vx_i64m2(c4, a[4 * rsa + p * csa], b0, vl);
    c5 = __riscv_vmacc_vx_i64m2(c5, a[5 * rsa + p * csa], b0, vl);
  }

  __riscv_vse64_v_i64m2(c, c0, vl);
  __riscv_vse64_v_i64m2(c + rso, c1, vl);
  __riscv_vse64_v_i64m2(c + 2 * rso, c2, vl);
  __riscv_vse64_v_i64m2(c + 3 * rso, c3, vl);
  __riscv_vse64_v_i64m2(c + 4 * rso, c4, vl);
  __riscv_vse64_v_i64m2(c + 5 * rso, c5, vl);
}

static inline void gemm_edge_i64_rvv(const int64_t *a, const int64_t *b,
                                      int64_t *c, size_t mr, size_t nr,
                                      size_t kc, intptr_t rsa, intptr_t csa,
                                      intptr_t rsb, intptr_t rso) {
  for (size_t i = 0; i < mr; i++)
    for (size_t p = 0; p < kc; p++) {
      int64_t aip = a[i * rsa + p * csa];
      for (size_t j = 0; j < nr; j++)
        c[i * rso + j] += aip * b[p * rsb + j];
    }
}

static inline void gemm_i64_rvv(const int64_t *a, const int64_t *b,
                                 int64_t *out, size_t m_dim, size_t k_dim,
                                 size_t n_dim, intptr_t rsa, intptr_t csa,
                                 intptr_t rsb, intptr_t rso) {
  size_t nr = gemm_nr_i64_rvv();

  for (size_t i = 0; i < m_dim; i++)
    memset(out + i * rso, 0, n_dim * sizeof(int64_t));

  for (size_t pc = 0; pc < k_dim; pc += GEMM_I64_KC) {
    size_t kc = GEMM_MIN(GEMM_I64_KC, k_dim - pc);
#ifdef _OPENMP
#pragma omp parallel for schedule(                                             \
        static) if (m_dim * n_dim * sizeof(int64_t) > GEMM_OMP_THRESHOLD)
#endif
    for (size_t ic = 0; ic < m_dim; ic += GEMM_I64_MC) {
      size_t mc = GEMM_MIN(GEMM_I64_MC, m_dim - ic);
      size_t jr = 0;
      for (; jr + nr <= n_dim; jr += nr) {
        size_t ir = 0;
        for (; ir + GEMM_I64_MR <= mc; ir += GEMM_I64_MR)
          gemm_ukernel_i64_6xVL(a + (ic + ir) * rsa + pc * csa,
                                b + pc * rsb + jr, out + (ic + ir) * rso + jr,
                                kc, rsa, csa, rsb, rso);
        if (ir < mc)
          gemm_edge_i64_rvv(a + (ic + ir) * rsa + pc * csa, b + pc * rsb + jr,
                            out + (ic + ir) * rso + jr, mc - ir, nr, kc,
                            rsa, csa, rsb, rso);
      }
      if (jr < n_dim)
        gemm_edge_i64_rvv(a + ic * rsa + pc * csa, b + pc * rsb + jr,
                          out + ic * rso + jr, mc, n_dim - jr, kc, rsa, csa,
                          rsb, rso);
    }
  }
}

/* Uint64: identical bit-level operations as int64 */
static inline void gemm_u64_rvv(const uint64_t *a, const uint64_t *b,
                                 uint64_t *out, size_t m_dim, size_t k_dim,
                                 size_t n_dim, intptr_t rsa, intptr_t csa,
                                 intptr_t rsb, intptr_t rso) {
  gemm_i64_rvv((const int64_t *)a, (const int64_t *)b, (int64_t *)out, m_dim,
               k_dim, n_dim, rsa, csa, rsb, rso);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Int8: 6xVL promoted micro-kernel (widen to i32, full-K accumulation)
   NR matches i32 VL (e32m2). Each B element is widened i8->i16->i32.
   Uses vwadd for widening, then vmacc_vx_i32m2 for accumulation.
   ═══════════════════════════════════════════════════════════════════════════
 */

static inline void gemm_ukernel_i8_6xVL(const int8_t *a, const int8_t *b,
                                          int8_t *c, size_t k_dim,
                                          intptr_t rsa, intptr_t csa,
                                          intptr_t rsb, intptr_t rso) {
  size_t nr = gemm_nr_i8_rvv();
  size_t vl = __riscv_vsetvl_e32m2(nr);

  vint32m2_t c0 = __riscv_vmv_v_x_i32m2(0, vl);
  vint32m2_t c1 = __riscv_vmv_v_x_i32m2(0, vl);
  vint32m2_t c2 = __riscv_vmv_v_x_i32m2(0, vl);
  vint32m2_t c3 = __riscv_vmv_v_x_i32m2(0, vl);
  vint32m2_t c4 = __riscv_vmv_v_x_i32m2(0, vl);
  vint32m2_t c5 = __riscv_vmv_v_x_i32m2(0, vl);

  for (size_t p = 0; p < k_dim; p++) {
    const int8_t *bp = b + p * rsb;
    /* Load NR i8 values and widen to i32:
     * i8 -> i16 (vwadd) -> i32 (vsext) */
    size_t vl8 = __riscv_vsetvl_e8mf2(nr);
    vint8mf2_t bv8 = __riscv_vle8_v_i8mf2(bp, vl8);
    vint16m1_t bv16 = __riscv_vwadd_vx_i16m1(bv8, 0, vl8);
    vint32m2_t b0 = __riscv_vsext_vf2_i32m2(bv16, vl);

    c0 = __riscv_vmacc_vx_i32m2(c0, (int32_t)a[0 * rsa + p * csa], b0, vl);
    c1 = __riscv_vmacc_vx_i32m2(c1, (int32_t)a[1 * rsa + p * csa], b0, vl);
    c2 = __riscv_vmacc_vx_i32m2(c2, (int32_t)a[2 * rsa + p * csa], b0, vl);
    c3 = __riscv_vmacc_vx_i32m2(c3, (int32_t)a[3 * rsa + p * csa], b0, vl);
    c4 = __riscv_vmacc_vx_i32m2(c4, (int32_t)a[4 * rsa + p * csa], b0, vl);
    c5 = __riscv_vmacc_vx_i32m2(c5, (int32_t)a[5 * rsa + p * csa], b0, vl);
  }

  /* Narrow i32 -> i16 -> i8, store NR bytes per row */
#define RVV_STORE_I8_ROW(cx, row)                                              \
  do {                                                                         \
    vint16m1_t n16 = __riscv_vnsra_wx_i16m1(cx, 0, vl);                       \
    size_t vl8s = __riscv_vsetvl_e8mf2(nr);                                   \
    vint8mf2_t n8 = __riscv_vnsra_wx_i8mf2(n16, 0, vl8s);                    \
    __riscv_vse8_v_i8mf2(c + (row) * rso, n8, vl8s);                          \
  } while (0)
  RVV_STORE_I8_ROW(c0, 0);
  RVV_STORE_I8_ROW(c1, 1);
  RVV_STORE_I8_ROW(c2, 2);
  RVV_STORE_I8_ROW(c3, 3);
  RVV_STORE_I8_ROW(c4, 4);
  RVV_STORE_I8_ROW(c5, 5);
#undef RVV_STORE_I8_ROW
}

static inline void gemm_edge_i8_rvv(const int8_t *a, const int8_t *b,
                                     int8_t *c, size_t mr, size_t nr,
                                     size_t k_dim, intptr_t rsa, intptr_t csa,
                                     intptr_t rsb, intptr_t rso) {
  for (size_t i = 0; i < mr; i++)
    for (size_t j = 0; j < nr; j++) {
      int32_t acc = 0;
      for (size_t p = 0; p < k_dim; p++)
        acc += (int32_t)a[i * rsa + p * csa] * (int32_t)b[p * rsb + j];
      c[i * rso + j] = (int8_t)acc;
    }
}

static inline void gemm_i8_rvv(const int8_t *a, const int8_t *b, int8_t *out,
                                size_t m_dim, size_t k_dim, size_t n_dim,
                                intptr_t rsa, intptr_t csa, intptr_t rsb,
                                intptr_t rso) {
  size_t nr = gemm_nr_i8_rvv();

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (m_dim * n_dim >                  \
                                                  GEMM_OMP_THRESHOLD)
#endif
  for (size_t ic = 0; ic < m_dim; ic += GEMM_I8_MC) {
    size_t mc = GEMM_MIN(GEMM_I8_MC, m_dim - ic);
    size_t jr = 0;
    for (; jr + nr <= n_dim; jr += nr) {
      size_t ir = 0;
      for (; ir + GEMM_I8_MR <= mc; ir += GEMM_I8_MR)
        gemm_ukernel_i8_6xVL(a + (ic + ir) * rsa, b + jr,
                              out + (ic + ir) * rso + jr, k_dim, rsa, csa,
                              rsb, rso);
      if (ir < mc)
        gemm_edge_i8_rvv(a + (ic + ir) * rsa, b + jr,
                         out + (ic + ir) * rso + jr, mc - ir, nr, k_dim,
                         rsa, csa, rsb, rso);
    }
    if (jr < n_dim)
      gemm_edge_i8_rvv(a + ic * rsa, b + jr, out + ic * rso + jr, mc,
                       n_dim - jr, k_dim, rsa, csa, rsb, rso);
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   Uint8: 6xVL promoted micro-kernel (widen to u32, full-K accumulation)
   ═══════════════════════════════════════════════════════════════════════════
 */

static inline void gemm_ukernel_u8_6xVL(const uint8_t *a, const uint8_t *b,
                                          uint8_t *c, size_t k_dim,
                                          intptr_t rsa, intptr_t csa,
                                          intptr_t rsb, intptr_t rso) {
  size_t nr = gemm_nr_i8_rvv();
  size_t vl = __riscv_vsetvl_e32m2(nr);

  vuint32m2_t c0 = __riscv_vmv_v_x_u32m2(0, vl);
  vuint32m2_t c1 = __riscv_vmv_v_x_u32m2(0, vl);
  vuint32m2_t c2 = __riscv_vmv_v_x_u32m2(0, vl);
  vuint32m2_t c3 = __riscv_vmv_v_x_u32m2(0, vl);
  vuint32m2_t c4 = __riscv_vmv_v_x_u32m2(0, vl);
  vuint32m2_t c5 = __riscv_vmv_v_x_u32m2(0, vl);

  for (size_t p = 0; p < k_dim; p++) {
    const uint8_t *bp = b + p * rsb;
    /* Load NR u8 values and widen to u32:
     * u8 -> u16 (vwaddu) -> u32 (vzext) */
    size_t vl8 = __riscv_vsetvl_e8mf2(nr);
    vuint8mf2_t bv8 = __riscv_vle8_v_u8mf2(bp, vl8);
    vuint16m1_t bv16 = __riscv_vwaddu_vx_u16m1(bv8, 0, vl8);
    vuint32m2_t b0 = __riscv_vzext_vf2_u32m2(bv16, vl);

    c0 = __riscv_vmacc_vx_u32m2(c0, (uint32_t)a[0 * rsa + p * csa], b0, vl);
    c1 = __riscv_vmacc_vx_u32m2(c1, (uint32_t)a[1 * rsa + p * csa], b0, vl);
    c2 = __riscv_vmacc_vx_u32m2(c2, (uint32_t)a[2 * rsa + p * csa], b0, vl);
    c3 = __riscv_vmacc_vx_u32m2(c3, (uint32_t)a[3 * rsa + p * csa], b0, vl);
    c4 = __riscv_vmacc_vx_u32m2(c4, (uint32_t)a[4 * rsa + p * csa], b0, vl);
    c5 = __riscv_vmacc_vx_u32m2(c5, (uint32_t)a[5 * rsa + p * csa], b0, vl);
  }

  /* Narrow u32 -> u16 -> u8, store NR bytes per row */
#define RVV_STORE_U8_ROW(cx, row)                                              \
  do {                                                                         \
    vuint16m1_t n16 = __riscv_vnsrl_wx_u16m1(cx, 0, vl);                      \
    size_t vl8s = __riscv_vsetvl_e8mf2(nr);                                   \
    vuint8mf2_t n8 = __riscv_vnsrl_wx_u8mf2(n16, 0, vl8s);                   \
    __riscv_vse8_v_u8mf2(c + (row) * rso, n8, vl8s);                          \
  } while (0)
  RVV_STORE_U8_ROW(c0, 0);
  RVV_STORE_U8_ROW(c1, 1);
  RVV_STORE_U8_ROW(c2, 2);
  RVV_STORE_U8_ROW(c3, 3);
  RVV_STORE_U8_ROW(c4, 4);
  RVV_STORE_U8_ROW(c5, 5);
#undef RVV_STORE_U8_ROW
}

static inline void gemm_edge_u8_rvv(const uint8_t *a, const uint8_t *b,
                                     uint8_t *c, size_t mr, size_t nr,
                                     size_t k_dim, intptr_t rsa, intptr_t csa,
                                     intptr_t rsb, intptr_t rso) {
  for (size_t i = 0; i < mr; i++)
    for (size_t j = 0; j < nr; j++) {
      uint32_t acc = 0;
      for (size_t p = 0; p < k_dim; p++)
        acc += (uint32_t)a[i * rsa + p * csa] * (uint32_t)b[p * rsb + j];
      c[i * rso + j] = (uint8_t)acc;
    }
}

static inline void gemm_u8_rvv(const uint8_t *a, const uint8_t *b,
                                uint8_t *out, size_t m_dim, size_t k_dim,
                                size_t n_dim, intptr_t rsa, intptr_t csa,
                                intptr_t rsb, intptr_t rso) {
  size_t nr = gemm_nr_i8_rvv();

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (m_dim * n_dim >                  \
                                                  GEMM_OMP_THRESHOLD)
#endif
  for (size_t ic = 0; ic < m_dim; ic += GEMM_I8_MC) {
    size_t mc = GEMM_MIN(GEMM_I8_MC, m_dim - ic);
    size_t jr = 0;
    for (; jr + nr <= n_dim; jr += nr) {
      size_t ir = 0;
      for (; ir + GEMM_I8_MR <= mc; ir += GEMM_I8_MR)
        gemm_ukernel_u8_6xVL(a + (ic + ir) * rsa, b + jr,
                              out + (ic + ir) * rso + jr, k_dim, rsa, csa,
                              rsb, rso);
      if (ir < mc)
        gemm_edge_u8_rvv(a + (ic + ir) * rsa, b + jr,
                         out + (ic + ir) * rso + jr, mc - ir, nr, k_dim,
                         rsa, csa, rsb, rso);
    }
    if (jr < n_dim)
      gemm_edge_u8_rvv(a + ic * rsa, b + jr, out + ic * rso + jr, mc,
                       n_dim - jr, k_dim, rsa, csa, rsb, rso);
  }
}

// clang-format on

#undef GEMM_MIN

#endif
