#ifndef NUMC_GEMM_SVE_H
#define NUMC_GEMM_SVE_H

#include <arm_sve.h>
#include <stdint.h>
#include <string.h>

#define GEMM_MIN(a, b) ((a) < (b) ? (a) : (b))

/*
 * Cache-blocking parameters for SVE packed GEMM.
 * SVE has scalable vector length (128-2048 bits, runtime-determined).
 * AArch64 SVE: 32 scalable vector registers (z0-z31) + 16 predicates (p0-p15).
 *
 * NR is runtime: NR = 2 * svcntw() for f32, NR = 2 * svcntd() for f64.
 * At 128-bit SVE: NR_f32 = 2*4 = 8,  NR_f64 = 2*2 = 4
 * At 256-bit SVE: NR_f32 = 2*8 = 16, NR_f64 = 2*4 = 8
 * At 512-bit SVE: NR_f32 = 2*16 = 32, NR_f64 = 2*8 = 16
 *
 * f32 8x(2*VL): 16 acc + 2 B + 8 A-broadcast = 26 regs (fits in 32).
 *   KC x NR sliver in L1: 256 x 32 x 4 = 32KB (worst case 512-bit)
 *   MC x KC panel  in L2: 128 x 256 x 4 = 128KB < 256KB
 *
 * f64 6x(2*VL): 12 acc + 2 B + 6 A-broadcast = 20 regs (fits in 32).
 *   KC x NR sliver in L1: 256 x 16 x 8 = 32KB (worst case 512-bit)
 *   MC x KC panel  in L2: 72 x 256 x 8 = 144KB < 256KB
 */
#define GEMM_F32_MR 8
#define GEMM_F32_MC 128
#define GEMM_F32_KC 256

#define GEMM_F64_MR 6
#define GEMM_F64_MC 72
#define GEMM_F64_KC 256

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

#define GEMM_F32_NC 4080
#define GEMM_F64_NC 4080

/* Maximum NR across all SVE implementations (2048-bit / 16-bit = 128 elems).
 * Used for stack-allocated temporary buffers. */
#define GEMM_SVE_MAX_NR 128

/* ── Float32 packing routines ──────────────────────────────────────────── */

static inline void gemm_pack_b_f32_sve(const float *b, float *packed, size_t kc,
                                       size_t nc, intptr_t rsb, size_t nr) {
  size_t vl = svcntw();
  svbool_t ptrue = svptrue_b32();
  size_t jr = 0;
  for (; jr + nr <= nc; jr += nr) {
    float *dest = packed + jr * kc;
    for (size_t p = 0; p < kc; p++) {
      const float *src = b + p * rsb + jr;
      svst1_f32(ptrue, dest + p * nr, svld1_f32(ptrue, src));
      svst1_f32(ptrue, dest + p * nr + vl, svld1_f32(ptrue, src + vl));
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

static inline void gemm_pack_a_f32_sve(const float *a, float *packed, size_t mc,
                                       size_t kc, intptr_t rsa, intptr_t csa) {
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

static inline void gemm_pack_b_f64_sve(const double *b, double *packed,
                                       size_t kc, size_t nc, intptr_t rsb,
                                       size_t nr) {
  size_t vl = svcntd();
  svbool_t ptrue = svptrue_b64();
  size_t jr = 0;
  for (; jr + nr <= nc; jr += nr) {
    double *dest = packed + jr * kc;
    for (size_t p = 0; p < kc; p++) {
      const double *src = b + p * rsb + jr;
      svst1_f64(ptrue, dest + p * nr, svld1_f64(ptrue, src));
      svst1_f64(ptrue, dest + p * nr + vl, svld1_f64(ptrue, src + vl));
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

static inline void gemm_pack_a_f64_sve(const double *a, double *packed,
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
   Float32: 8x(2*VL) micro-kernel
   Uses svdup_f32(scalar) for A broadcast + svmla_f32_x for FMA.
   16 accumulator regs (8 rows x 2 B-vectors), 2 B regs, 1 A broadcast.
   ═══════════════════════════════════════════════════════════════════════════
 */

#define GEMM_F32_SVE_K_ITER(ap, bp, vl, ptrue)      \
  do {                                              \
    svfloat32_t b0 = svld1_f32(ptrue, (bp));        \
    svfloat32_t b1 = svld1_f32(ptrue, (bp) + (vl)); \
    svfloat32_t av;                                 \
    av = svdup_f32((ap)[0]);                        \
    c00 = svmla_f32_x(ptrue, c00, av, b0);          \
    c01 = svmla_f32_x(ptrue, c01, av, b1);          \
    av = svdup_f32((ap)[1]);                        \
    c10 = svmla_f32_x(ptrue, c10, av, b0);          \
    c11 = svmla_f32_x(ptrue, c11, av, b1);          \
    av = svdup_f32((ap)[2]);                        \
    c20 = svmla_f32_x(ptrue, c20, av, b0);          \
    c21 = svmla_f32_x(ptrue, c21, av, b1);          \
    av = svdup_f32((ap)[3]);                        \
    c30 = svmla_f32_x(ptrue, c30, av, b0);          \
    c31 = svmla_f32_x(ptrue, c31, av, b1);          \
    av = svdup_f32((ap)[4]);                        \
    c40 = svmla_f32_x(ptrue, c40, av, b0);          \
    c41 = svmla_f32_x(ptrue, c41, av, b1);          \
    av = svdup_f32((ap)[5]);                        \
    c50 = svmla_f32_x(ptrue, c50, av, b0);          \
    c51 = svmla_f32_x(ptrue, c51, av, b1);          \
    av = svdup_f32((ap)[6]);                        \
    c60 = svmla_f32_x(ptrue, c60, av, b0);          \
    c61 = svmla_f32_x(ptrue, c61, av, b1);          \
    av = svdup_f32((ap)[7]);                        \
    c70 = svmla_f32_x(ptrue, c70, av, b0);          \
    c71 = svmla_f32_x(ptrue, c71, av, b1);          \
  } while (0)

static inline void gemm_ukernel_f32_sve(const float *a, const float *b,
                                        float *c, size_t kc, intptr_t rsa,
                                        intptr_t csa, intptr_t rsb,
                                        intptr_t rso, int first) {
  size_t vl = svcntw();
  svbool_t ptrue = svptrue_b32();

  svfloat32_t c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51, c60,
      c61, c70, c71;

  /* Prefetch 8 rows of C */
  __builtin_prefetch(c, 1, 3);
  __builtin_prefetch(c + rso, 1, 3);
  __builtin_prefetch(c + 2 * rso, 1, 3);
  __builtin_prefetch(c + 3 * rso, 1, 3);
  __builtin_prefetch(c + 4 * rso, 1, 3);
  __builtin_prefetch(c + 5 * rso, 1, 3);
  __builtin_prefetch(c + 6 * rso, 1, 3);
  __builtin_prefetch(c + 7 * rso, 1, 3);

  if (first) {
    c00 = svdup_f32(0);
    c01 = svdup_f32(0);
    c10 = svdup_f32(0);
    c11 = svdup_f32(0);
    c20 = svdup_f32(0);
    c21 = svdup_f32(0);
    c30 = svdup_f32(0);
    c31 = svdup_f32(0);
    c40 = svdup_f32(0);
    c41 = svdup_f32(0);
    c50 = svdup_f32(0);
    c51 = svdup_f32(0);
    c60 = svdup_f32(0);
    c61 = svdup_f32(0);
    c70 = svdup_f32(0);
    c71 = svdup_f32(0);
  } else {
    c00 = svld1_f32(ptrue, c);
    c01 = svld1_f32(ptrue, c + vl);
    c10 = svld1_f32(ptrue, c + rso);
    c11 = svld1_f32(ptrue, c + rso + vl);
    c20 = svld1_f32(ptrue, c + 2 * rso);
    c21 = svld1_f32(ptrue, c + 2 * rso + vl);
    c30 = svld1_f32(ptrue, c + 3 * rso);
    c31 = svld1_f32(ptrue, c + 3 * rso + vl);
    c40 = svld1_f32(ptrue, c + 4 * rso);
    c41 = svld1_f32(ptrue, c + 4 * rso + vl);
    c50 = svld1_f32(ptrue, c + 5 * rso);
    c51 = svld1_f32(ptrue, c + 5 * rso + vl);
    c60 = svld1_f32(ptrue, c + 6 * rso);
    c61 = svld1_f32(ptrue, c + 6 * rso + vl);
    c70 = svld1_f32(ptrue, c + 7 * rso);
    c71 = svld1_f32(ptrue, c + 7 * rso + vl);
  }

  const float *ap = a;
  const float *bp = b;
  size_t k_iter = kc / 4;
  size_t k_left = kc % 4;

  for (size_t ki = 0; ki < k_iter; ki++) {
    GEMM_F32_SVE_K_ITER(ap, bp, vl, ptrue);
    ap += csa;
    bp += rsb;
    GEMM_F32_SVE_K_ITER(ap, bp, vl, ptrue);
    ap += csa;
    bp += rsb;
    __builtin_prefetch(ap + 64, 0, 3);
    GEMM_F32_SVE_K_ITER(ap, bp, vl, ptrue);
    ap += csa;
    bp += rsb;
    GEMM_F32_SVE_K_ITER(ap, bp, vl, ptrue);
    ap += csa;
    bp += rsb;
  }
  for (size_t ki = 0; ki < k_left; ki++) {
    GEMM_F32_SVE_K_ITER(ap, bp, vl, ptrue);
    ap += csa;
    bp += rsb;
  }

  svst1_f32(ptrue, c, c00);
  svst1_f32(ptrue, c + vl, c01);
  svst1_f32(ptrue, c + rso, c10);
  svst1_f32(ptrue, c + rso + vl, c11);
  svst1_f32(ptrue, c + 2 * rso, c20);
  svst1_f32(ptrue, c + 2 * rso + vl, c21);
  svst1_f32(ptrue, c + 3 * rso, c30);
  svst1_f32(ptrue, c + 3 * rso + vl, c31);
  svst1_f32(ptrue, c + 4 * rso, c40);
  svst1_f32(ptrue, c + 4 * rso + vl, c41);
  svst1_f32(ptrue, c + 5 * rso, c50);
  svst1_f32(ptrue, c + 5 * rso + vl, c51);
  svst1_f32(ptrue, c + 6 * rso, c60);
  svst1_f32(ptrue, c + 6 * rso + vl, c61);
  svst1_f32(ptrue, c + 7 * rso, c70);
  svst1_f32(ptrue, c + 7 * rso + vl, c71);
}

#undef GEMM_F32_SVE_K_ITER

static inline void gemm_edge_f32_sve(const float *a, const float *b, float *c,
                                     size_t mr, size_t nr, size_t kc,
                                     intptr_t rsa, intptr_t csa, intptr_t rsb,
                                     intptr_t rso) {
  size_t vl = svcntw();
  for (size_t i = 0; i < mr; i++) {
    for (size_t p = 0; p < kc; p++) {
      float aip = a[i * rsa + p * csa];
      svfloat32_t va = svdup_f32(aip);
      const float *brow = b + p * rsb;
      float *crow = c + i * rso;
      size_t j = 0;
      for (; j + vl <= nr; j += vl) {
        svbool_t pt = svptrue_b32();
        svfloat32_t vo = svld1_f32(pt, crow + j);
        vo = svmla_f32_x(pt, vo, va, svld1_f32(pt, brow + j));
        svst1_f32(pt, crow + j, vo);
      }
      if (j < nr) {
        svbool_t pg = svwhilelt_b32((uint32_t)j, (uint32_t)nr);
        svfloat32_t vo = svld1_f32(pg, crow + j);
        vo = svmla_f32_m(pg, vo, va, svld1_f32(pg, brow + j));
        svst1_f32(pg, crow + j, vo);
      }
    }
  }
}

static inline void gemm_f32_sve(const float *a, const float *b, float *out,
                                size_t m_dim, size_t k_dim, size_t n_dim,
                                intptr_t rsa, intptr_t csa, intptr_t rsb,
                                intptr_t rso) {
  size_t vl = svcntw();
  size_t nr = 2 * vl;
  size_t nc_max = GEMM_MIN(GEMM_F32_NC, n_dim);
  float *packed_b =
      (float *)numc_malloc(16, GEMM_F32_KC * (nc_max + nr) * sizeof(float));
  if (!packed_b)
    return;

  for (size_t jc = 0; jc < n_dim; jc += GEMM_F32_NC) {
    size_t nc = GEMM_MIN(GEMM_F32_NC, n_dim - jc);

    for (size_t pc = 0; pc < k_dim; pc += GEMM_F32_KC) {
      size_t kc = GEMM_MIN(GEMM_F32_KC, k_dim - pc);
      int first = (pc == 0);

      gemm_pack_b_f32_sve(b + pc * rsb + jc, packed_b, kc, nc, rsb, nr);

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
            gemm_pack_a_f32_sve(a + ic * rsa + pc * csa, packed_a, mc, kc, rsa,
                                csa);
            last_ic = ic;
          }

          for (size_t ir = 0; ir < mc; ir += GEMM_F32_MR) {
            size_t mr_cur = GEMM_MIN(GEMM_F32_MR, mc - ir);
            if (mr_cur == GEMM_F32_MR && nr_cur == nr) {
              gemm_ukernel_f32_sve(packed_a + ir * kc, packed_b + jr * kc,
                                   out + (ic + ir) * rso + (jc + jr), kc, 1,
                                   GEMM_F32_MR, nr, rso, first);
            } else {
              float tmp[GEMM_F32_MR * GEMM_SVE_MAX_NR];
              gemm_ukernel_f32_sve(packed_a + ir * kc, packed_b + jr * kc, tmp,
                                   kc, 1, GEMM_F32_MR, nr, nr, 1);
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
            gemm_pack_a_f32_sve(a + ic * rsa + pc * csa, packed_a, mc, kc, rsa,
                                csa);

          for (size_t ir = 0; ir < mc; ir += GEMM_F32_MR) {
            size_t mr_cur = GEMM_MIN(GEMM_F32_MR, mc - ir);
            if (mr_cur == GEMM_F32_MR && nr_cur == nr) {
              gemm_ukernel_f32_sve(packed_a + ir * kc, packed_b + jr * kc,
                                   out + (ic + ir) * rso + (jc + jr), kc, 1,
                                   GEMM_F32_MR, nr, rso, first);
            } else {
              float tmp[GEMM_F32_MR * GEMM_SVE_MAX_NR];
              gemm_ukernel_f32_sve(packed_a + ir * kc, packed_b + jr * kc, tmp,
                                   kc, 1, GEMM_F32_MR, nr, nr, 1);
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
   Float64: 6x(2*VL) micro-kernel
   Uses svdup_f64(scalar) for A broadcast + svmla_f64_x for FMA.
   12 accumulator regs (6 rows x 2 B-vectors), 2 B regs, 1 A broadcast.
   ═══════════════════════════════════════════════════════════════════════════
 */

#define GEMM_F64_SVE_K_ITER(ap, bp, vl, ptrue)      \
  do {                                              \
    svfloat64_t b0 = svld1_f64(ptrue, (bp));        \
    svfloat64_t b1 = svld1_f64(ptrue, (bp) + (vl)); \
    svfloat64_t av;                                 \
    av = svdup_f64((ap)[0]);                        \
    c00 = svmla_f64_x(ptrue, c00, av, b0);          \
    c01 = svmla_f64_x(ptrue, c01, av, b1);          \
    av = svdup_f64((ap)[1]);                        \
    c10 = svmla_f64_x(ptrue, c10, av, b0);          \
    c11 = svmla_f64_x(ptrue, c11, av, b1);          \
    av = svdup_f64((ap)[2]);                        \
    c20 = svmla_f64_x(ptrue, c20, av, b0);          \
    c21 = svmla_f64_x(ptrue, c21, av, b1);          \
    av = svdup_f64((ap)[3]);                        \
    c30 = svmla_f64_x(ptrue, c30, av, b0);          \
    c31 = svmla_f64_x(ptrue, c31, av, b1);          \
    av = svdup_f64((ap)[4]);                        \
    c40 = svmla_f64_x(ptrue, c40, av, b0);          \
    c41 = svmla_f64_x(ptrue, c41, av, b1);          \
    av = svdup_f64((ap)[5]);                        \
    c50 = svmla_f64_x(ptrue, c50, av, b0);          \
    c51 = svmla_f64_x(ptrue, c51, av, b1);          \
  } while (0)

static inline void gemm_ukernel_f64_sve(const double *a, const double *b,
                                        double *c, size_t kc, intptr_t rsa,
                                        intptr_t csa, intptr_t rsb,
                                        intptr_t rso, int first) {
  size_t vl = svcntd();
  svbool_t ptrue = svptrue_b64();

  svfloat64_t c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;

  __builtin_prefetch(c, 1, 3);
  __builtin_prefetch(c + rso, 1, 3);
  __builtin_prefetch(c + 2 * rso, 1, 3);
  __builtin_prefetch(c + 3 * rso, 1, 3);
  __builtin_prefetch(c + 4 * rso, 1, 3);
  __builtin_prefetch(c + 5 * rso, 1, 3);

  if (first) {
    c00 = svdup_f64(0);
    c01 = svdup_f64(0);
    c10 = svdup_f64(0);
    c11 = svdup_f64(0);
    c20 = svdup_f64(0);
    c21 = svdup_f64(0);
    c30 = svdup_f64(0);
    c31 = svdup_f64(0);
    c40 = svdup_f64(0);
    c41 = svdup_f64(0);
    c50 = svdup_f64(0);
    c51 = svdup_f64(0);
  } else {
    c00 = svld1_f64(ptrue, c);
    c01 = svld1_f64(ptrue, c + vl);
    c10 = svld1_f64(ptrue, c + rso);
    c11 = svld1_f64(ptrue, c + rso + vl);
    c20 = svld1_f64(ptrue, c + 2 * rso);
    c21 = svld1_f64(ptrue, c + 2 * rso + vl);
    c30 = svld1_f64(ptrue, c + 3 * rso);
    c31 = svld1_f64(ptrue, c + 3 * rso + vl);
    c40 = svld1_f64(ptrue, c + 4 * rso);
    c41 = svld1_f64(ptrue, c + 4 * rso + vl);
    c50 = svld1_f64(ptrue, c + 5 * rso);
    c51 = svld1_f64(ptrue, c + 5 * rso + vl);
  }

  const double *ap = a;
  const double *bp = b;
  size_t k_iter = kc / 4;
  size_t k_left = kc % 4;

  for (size_t ki = 0; ki < k_iter; ki++) {
    GEMM_F64_SVE_K_ITER(ap, bp, vl, ptrue);
    ap += csa;
    bp += rsb;
    GEMM_F64_SVE_K_ITER(ap, bp, vl, ptrue);
    ap += csa;
    bp += rsb;
    __builtin_prefetch(ap + 48, 0, 3);
    GEMM_F64_SVE_K_ITER(ap, bp, vl, ptrue);
    ap += csa;
    bp += rsb;
    GEMM_F64_SVE_K_ITER(ap, bp, vl, ptrue);
    ap += csa;
    bp += rsb;
  }
  for (size_t ki = 0; ki < k_left; ki++) {
    GEMM_F64_SVE_K_ITER(ap, bp, vl, ptrue);
    ap += csa;
    bp += rsb;
  }

  svst1_f64(ptrue, c, c00);
  svst1_f64(ptrue, c + vl, c01);
  svst1_f64(ptrue, c + rso, c10);
  svst1_f64(ptrue, c + rso + vl, c11);
  svst1_f64(ptrue, c + 2 * rso, c20);
  svst1_f64(ptrue, c + 2 * rso + vl, c21);
  svst1_f64(ptrue, c + 3 * rso, c30);
  svst1_f64(ptrue, c + 3 * rso + vl, c31);
  svst1_f64(ptrue, c + 4 * rso, c40);
  svst1_f64(ptrue, c + 4 * rso + vl, c41);
  svst1_f64(ptrue, c + 5 * rso, c50);
  svst1_f64(ptrue, c + 5 * rso + vl, c51);
}

#undef GEMM_F64_SVE_K_ITER

static inline void gemm_edge_f64_sve(const double *a, const double *b,
                                     double *c, size_t mr, size_t nr, size_t kc,
                                     intptr_t rsa, intptr_t csa, intptr_t rsb,
                                     intptr_t rso) {
  size_t vl = svcntd();
  for (size_t i = 0; i < mr; i++) {
    for (size_t p = 0; p < kc; p++) {
      double aip = a[i * rsa + p * csa];
      svfloat64_t va = svdup_f64(aip);
      const double *brow = b + p * rsb;
      double *crow = c + i * rso;
      size_t j = 0;
      for (; j + vl <= nr; j += vl) {
        svbool_t pt = svptrue_b64();
        svfloat64_t vo = svld1_f64(pt, crow + j);
        vo = svmla_f64_x(pt, vo, va, svld1_f64(pt, brow + j));
        svst1_f64(pt, crow + j, vo);
      }
      if (j < nr) {
        svbool_t pg = svwhilelt_b64((uint32_t)j, (uint32_t)nr);
        svfloat64_t vo = svld1_f64(pg, crow + j);
        vo = svmla_f64_m(pg, vo, va, svld1_f64(pg, brow + j));
        svst1_f64(pg, crow + j, vo);
      }
    }
  }
}

static inline void gemm_f64_sve(const double *a, const double *b, double *out,
                                size_t m_dim, size_t k_dim, size_t n_dim,
                                intptr_t rsa, intptr_t csa, intptr_t rsb,
                                intptr_t rso) {
  size_t vl = svcntd();
  size_t nr = 2 * vl;
  size_t nc_max = GEMM_MIN(GEMM_F64_NC, n_dim);
  double *packed_b =
      (double *)numc_malloc(16, GEMM_F64_KC * (nc_max + nr) * sizeof(double));
  if (!packed_b)
    return;

  for (size_t jc = 0; jc < n_dim; jc += GEMM_F64_NC) {
    size_t nc = GEMM_MIN(GEMM_F64_NC, n_dim - jc);

    for (size_t pc = 0; pc < k_dim; pc += GEMM_F64_KC) {
      size_t kc = GEMM_MIN(GEMM_F64_KC, k_dim - pc);
      int first = (pc == 0);

      gemm_pack_b_f64_sve(b + pc * rsb + jc, packed_b, kc, nc, rsb, nr);

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
            gemm_pack_a_f64_sve(a + ic * rsa + pc * csa, packed_a, mc, kc, rsa,
                                csa);
            last_ic = ic;
          }

          for (size_t ir = 0; ir < mc; ir += GEMM_F64_MR) {
            size_t mr_cur = GEMM_MIN(GEMM_F64_MR, mc - ir);
            if (mr_cur == GEMM_F64_MR && nr_cur == nr) {
              gemm_ukernel_f64_sve(packed_a + ir * kc, packed_b + jr * kc,
                                   out + (ic + ir) * rso + (jc + jr), kc, 1,
                                   GEMM_F64_MR, nr, rso, first);
            } else {
              double tmp[GEMM_F64_MR * GEMM_SVE_MAX_NR];
              gemm_ukernel_f64_sve(packed_a + ir * kc, packed_b + jr * kc, tmp,
                                   kc, 1, GEMM_F64_MR, nr, nr, 1);
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
            gemm_pack_a_f64_sve(a + ic * rsa + pc * csa, packed_a, mc, kc, rsa,
                                csa);

          for (size_t ir = 0; ir < mc; ir += GEMM_F64_MR) {
            size_t mr_cur = GEMM_MIN(GEMM_F64_MR, mc - ir);
            if (mr_cur == GEMM_F64_MR && nr_cur == nr) {
              gemm_ukernel_f64_sve(packed_a + ir * kc, packed_b + jr * kc,
                                   out + (ic + ir) * rso + (jc + jr), kc, 1,
                                   GEMM_F64_MR, nr, rso, first);
            } else {
              double tmp[GEMM_F64_MR * GEMM_SVE_MAX_NR];
              gemm_ukernel_f64_sve(packed_a + ir * kc, packed_b + jr * kc, tmp,
                                   kc, 1, GEMM_F64_MR, nr, nr, 1);
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
   Int32/Uint32: 6x(2*VL) unpacked micro-kernel (svmla_s32_x)
   ═══════════════════════════════════════════════════════════════════════════
 */

static inline void gemm_ukernel_i32_sve(const int32_t *a, const int32_t *b,
                                        int32_t *c, size_t kc, intptr_t rsa,
                                        intptr_t csa, intptr_t rsb,
                                        intptr_t rso) {
  size_t vl = svcntw();
  svbool_t ptrue = svptrue_b32();

  svint32_t c00 = svld1_s32(ptrue, c);
  svint32_t c01 = svld1_s32(ptrue, c + vl);
  svint32_t c10 = svld1_s32(ptrue, c + rso);
  svint32_t c11 = svld1_s32(ptrue, c + rso + vl);
  svint32_t c20 = svld1_s32(ptrue, c + 2 * rso);
  svint32_t c21 = svld1_s32(ptrue, c + 2 * rso + vl);
  svint32_t c30 = svld1_s32(ptrue, c + 3 * rso);
  svint32_t c31 = svld1_s32(ptrue, c + 3 * rso + vl);
  svint32_t c40 = svld1_s32(ptrue, c + 4 * rso);
  svint32_t c41 = svld1_s32(ptrue, c + 4 * rso + vl);
  svint32_t c50 = svld1_s32(ptrue, c + 5 * rso);
  svint32_t c51 = svld1_s32(ptrue, c + 5 * rso + vl);

  for (size_t p = 0; p < kc; p++) {
    const int32_t *bp = b + p * rsb;
    svint32_t b0 = svld1_s32(ptrue, bp);
    svint32_t b1 = svld1_s32(ptrue, bp + vl);
    svint32_t av;
    av = svdup_s32(a[0 * rsa + p * csa]);
    c00 = svmla_s32_x(ptrue, c00, av, b0);
    c01 = svmla_s32_x(ptrue, c01, av, b1);
    av = svdup_s32(a[1 * rsa + p * csa]);
    c10 = svmla_s32_x(ptrue, c10, av, b0);
    c11 = svmla_s32_x(ptrue, c11, av, b1);
    av = svdup_s32(a[2 * rsa + p * csa]);
    c20 = svmla_s32_x(ptrue, c20, av, b0);
    c21 = svmla_s32_x(ptrue, c21, av, b1);
    av = svdup_s32(a[3 * rsa + p * csa]);
    c30 = svmla_s32_x(ptrue, c30, av, b0);
    c31 = svmla_s32_x(ptrue, c31, av, b1);
    av = svdup_s32(a[4 * rsa + p * csa]);
    c40 = svmla_s32_x(ptrue, c40, av, b0);
    c41 = svmla_s32_x(ptrue, c41, av, b1);
    av = svdup_s32(a[5 * rsa + p * csa]);
    c50 = svmla_s32_x(ptrue, c50, av, b0);
    c51 = svmla_s32_x(ptrue, c51, av, b1);
  }

  svst1_s32(ptrue, c, c00);
  svst1_s32(ptrue, c + vl, c01);
  svst1_s32(ptrue, c + rso, c10);
  svst1_s32(ptrue, c + rso + vl, c11);
  svst1_s32(ptrue, c + 2 * rso, c20);
  svst1_s32(ptrue, c + 2 * rso + vl, c21);
  svst1_s32(ptrue, c + 3 * rso, c30);
  svst1_s32(ptrue, c + 3 * rso + vl, c31);
  svst1_s32(ptrue, c + 4 * rso, c40);
  svst1_s32(ptrue, c + 4 * rso + vl, c41);
  svst1_s32(ptrue, c + 5 * rso, c50);
  svst1_s32(ptrue, c + 5 * rso + vl, c51);
}

static inline void gemm_edge_i32_sve(const int32_t *a, const int32_t *b,
                                     int32_t *c, size_t mr, size_t nr,
                                     size_t kc, intptr_t rsa, intptr_t csa,
                                     intptr_t rsb, intptr_t rso) {
  size_t vl = svcntw();
  for (size_t i = 0; i < mr; i++) {
    for (size_t p = 0; p < kc; p++) {
      int32_t aip = a[i * rsa + p * csa];
      svint32_t va = svdup_s32(aip);
      const int32_t *brow = b + p * rsb;
      int32_t *crow = c + i * rso;
      size_t j = 0;
      for (; j + vl <= nr; j += vl) {
        svbool_t pt = svptrue_b32();
        svint32_t vo = svld1_s32(pt, crow + j);
        vo = svmla_s32_x(pt, vo, va, svld1_s32(pt, brow + j));
        svst1_s32(pt, crow + j, vo);
      }
      if (j < nr) {
        svbool_t pg = svwhilelt_b32((uint32_t)j, (uint32_t)nr);
        svint32_t vo = svld1_s32(pg, crow + j);
        vo = svmla_s32_m(pg, vo, va, svld1_s32(pg, brow + j));
        svst1_s32(pg, crow + j, vo);
      }
    }
  }
}

static inline void gemm_i32_sve(const int32_t *a, const int32_t *b,
                                int32_t *out, size_t m_dim, size_t k_dim,
                                size_t n_dim, intptr_t rsa, intptr_t csa,
                                intptr_t rsb, intptr_t rso) {
  size_t vl = svcntw();
  size_t i32_nr = 2 * vl;

  for (size_t i = 0; i < m_dim; i++)
    memset(out + i * rso, 0, n_dim * sizeof(int32_t));

  for (size_t pc = 0; pc < k_dim; pc += GEMM_I32_KC) {
    size_t kc = GEMM_MIN(GEMM_I32_KC, k_dim - pc);
#ifdef _OPENMP
#pragma omp parallel for schedule( \
        static) if (m_dim * n_dim * sizeof(int32_t) > GEMM_OMP_THRESHOLD)
#endif
    for (size_t ic = 0; ic < m_dim; ic += GEMM_I32_MC) {
      size_t mc = GEMM_MIN(GEMM_I32_MC, m_dim - ic);
      size_t jr = 0;
      for (; jr + i32_nr <= n_dim; jr += i32_nr) {
        size_t ir = 0;
        for (; ir + GEMM_I32_MR <= mc; ir += GEMM_I32_MR)
          gemm_ukernel_i32_sve(a + (ic + ir) * rsa + pc * csa,
                               b + pc * rsb + jr, out + (ic + ir) * rso + jr,
                               kc, rsa, csa, rsb, rso);
        if (ir < mc)
          gemm_edge_i32_sve(a + (ic + ir) * rsa + pc * csa, b + pc * rsb + jr,
                            out + (ic + ir) * rso + jr, mc - ir, i32_nr, kc,
                            rsa, csa, rsb, rso);
      }
      if (jr < n_dim)
        gemm_edge_i32_sve(a + ic * rsa + pc * csa, b + pc * rsb + jr,
                          out + ic * rso + jr, mc, n_dim - jr, kc, rsa, csa,
                          rsb, rso);
    }
  }
}

static inline void gemm_u32_sve(const uint32_t *a, const uint32_t *b,
                                uint32_t *out, size_t m_dim, size_t k_dim,
                                size_t n_dim, intptr_t rsa, intptr_t csa,
                                intptr_t rsb, intptr_t rso) {
  gemm_i32_sve((const int32_t *)a, (const int32_t *)b, (int32_t *)out, m_dim,
               k_dim, n_dim, rsa, csa, rsb, rso);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Int16/Uint16: 6x(2*VL) unpacked micro-kernel (svmla_s16_x)
   ═══════════════════════════════════════════════════════════════════════════
 */

static inline void gemm_ukernel_i16_sve(const int16_t *a, const int16_t *b,
                                        int16_t *c, size_t kc, intptr_t rsa,
                                        intptr_t csa, intptr_t rsb,
                                        intptr_t rso) {
  size_t vl = svcnth();
  svbool_t ptrue = svptrue_b16();

  svint16_t c00 = svld1_s16(ptrue, c);
  svint16_t c01 = svld1_s16(ptrue, c + vl);
  svint16_t c10 = svld1_s16(ptrue, c + rso);
  svint16_t c11 = svld1_s16(ptrue, c + rso + vl);
  svint16_t c20 = svld1_s16(ptrue, c + 2 * rso);
  svint16_t c21 = svld1_s16(ptrue, c + 2 * rso + vl);
  svint16_t c30 = svld1_s16(ptrue, c + 3 * rso);
  svint16_t c31 = svld1_s16(ptrue, c + 3 * rso + vl);
  svint16_t c40 = svld1_s16(ptrue, c + 4 * rso);
  svint16_t c41 = svld1_s16(ptrue, c + 4 * rso + vl);
  svint16_t c50 = svld1_s16(ptrue, c + 5 * rso);
  svint16_t c51 = svld1_s16(ptrue, c + 5 * rso + vl);

  for (size_t p = 0; p < kc; p++) {
    const int16_t *bp = b + p * rsb;
    svint16_t b0 = svld1_s16(ptrue, bp);
    svint16_t b1 = svld1_s16(ptrue, bp + vl);
    svint16_t av;
    av = svdup_s16(a[0 * rsa + p * csa]);
    c00 = svmla_s16_x(ptrue, c00, av, b0);
    c01 = svmla_s16_x(ptrue, c01, av, b1);
    av = svdup_s16(a[1 * rsa + p * csa]);
    c10 = svmla_s16_x(ptrue, c10, av, b0);
    c11 = svmla_s16_x(ptrue, c11, av, b1);
    av = svdup_s16(a[2 * rsa + p * csa]);
    c20 = svmla_s16_x(ptrue, c20, av, b0);
    c21 = svmla_s16_x(ptrue, c21, av, b1);
    av = svdup_s16(a[3 * rsa + p * csa]);
    c30 = svmla_s16_x(ptrue, c30, av, b0);
    c31 = svmla_s16_x(ptrue, c31, av, b1);
    av = svdup_s16(a[4 * rsa + p * csa]);
    c40 = svmla_s16_x(ptrue, c40, av, b0);
    c41 = svmla_s16_x(ptrue, c41, av, b1);
    av = svdup_s16(a[5 * rsa + p * csa]);
    c50 = svmla_s16_x(ptrue, c50, av, b0);
    c51 = svmla_s16_x(ptrue, c51, av, b1);
  }

  svst1_s16(ptrue, c, c00);
  svst1_s16(ptrue, c + vl, c01);
  svst1_s16(ptrue, c + rso, c10);
  svst1_s16(ptrue, c + rso + vl, c11);
  svst1_s16(ptrue, c + 2 * rso, c20);
  svst1_s16(ptrue, c + 2 * rso + vl, c21);
  svst1_s16(ptrue, c + 3 * rso, c30);
  svst1_s16(ptrue, c + 3 * rso + vl, c31);
  svst1_s16(ptrue, c + 4 * rso, c40);
  svst1_s16(ptrue, c + 4 * rso + vl, c41);
  svst1_s16(ptrue, c + 5 * rso, c50);
  svst1_s16(ptrue, c + 5 * rso + vl, c51);
}

static inline void gemm_edge_i16_sve(const int16_t *a, const int16_t *b,
                                     int16_t *c, size_t mr, size_t nr,
                                     size_t kc, intptr_t rsa, intptr_t csa,
                                     intptr_t rsb, intptr_t rso) {
  size_t vl = svcnth();
  for (size_t i = 0; i < mr; i++) {
    for (size_t p = 0; p < kc; p++) {
      int16_t aip = a[i * rsa + p * csa];
      svint16_t va = svdup_s16(aip);
      const int16_t *brow = b + p * rsb;
      int16_t *crow = c + i * rso;
      size_t j = 0;
      for (; j + vl <= nr; j += vl) {
        svbool_t pt = svptrue_b16();
        svint16_t vo = svld1_s16(pt, crow + j);
        vo = svmla_s16_x(pt, vo, va, svld1_s16(pt, brow + j));
        svst1_s16(pt, crow + j, vo);
      }
      if (j < nr) {
        svbool_t pg = svwhilelt_b16((uint32_t)j, (uint32_t)nr);
        svint16_t vo = svld1_s16(pg, crow + j);
        vo = svmla_s16_m(pg, vo, va, svld1_s16(pg, brow + j));
        svst1_s16(pg, crow + j, vo);
      }
    }
  }
}

static inline void gemm_i16_sve(const int16_t *a, const int16_t *b,
                                int16_t *out, size_t m_dim, size_t k_dim,
                                size_t n_dim, intptr_t rsa, intptr_t csa,
                                intptr_t rsb, intptr_t rso) {
  size_t vl = svcnth();
  size_t i16_nr = 2 * vl;

  for (size_t i = 0; i < m_dim; i++)
    memset(out + i * rso, 0, n_dim * sizeof(int16_t));

  for (size_t pc = 0; pc < k_dim; pc += GEMM_I16_KC) {
    size_t kc = GEMM_MIN(GEMM_I16_KC, k_dim - pc);
#ifdef _OPENMP
#pragma omp parallel for schedule( \
        static) if (m_dim * n_dim * sizeof(int16_t) > GEMM_OMP_THRESHOLD)
#endif
    for (size_t ic = 0; ic < m_dim; ic += GEMM_I16_MC) {
      size_t mc = GEMM_MIN(GEMM_I16_MC, m_dim - ic);
      size_t jr = 0;
      for (; jr + i16_nr <= n_dim; jr += i16_nr) {
        size_t ir = 0;
        for (; ir + GEMM_I16_MR <= mc; ir += GEMM_I16_MR)
          gemm_ukernel_i16_sve(a + (ic + ir) * rsa + pc * csa,
                               b + pc * rsb + jr, out + (ic + ir) * rso + jr,
                               kc, rsa, csa, rsb, rso);
        if (ir < mc)
          gemm_edge_i16_sve(a + (ic + ir) * rsa + pc * csa, b + pc * rsb + jr,
                            out + (ic + ir) * rso + jr, mc - ir, i16_nr, kc,
                            rsa, csa, rsb, rso);
      }
      if (jr < n_dim)
        gemm_edge_i16_sve(a + ic * rsa + pc * csa, b + pc * rsb + jr,
                          out + ic * rso + jr, mc, n_dim - jr, kc, rsa, csa,
                          rsb, rso);
    }
  }
}

static inline void gemm_u16_sve(const uint16_t *a, const uint16_t *b,
                                uint16_t *out, size_t m_dim, size_t k_dim,
                                size_t n_dim, intptr_t rsa, intptr_t csa,
                                intptr_t rsb, intptr_t rso) {
  gemm_i16_sve((const int16_t *)a, (const int16_t *)b, (int16_t *)out, m_dim,
               k_dim, n_dim, rsa, csa, rsb, rso);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Int64/Uint64: scalar loop (no native 64-bit multiply in SVE base)
   ═══════════════════════════════════════════════════════════════════════════
 */

static inline void gemm_edge_i64_sve(const int64_t *a, const int64_t *b,
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

static inline void gemm_i64_sve(const int64_t *a, const int64_t *b,
                                int64_t *out, size_t m_dim, size_t k_dim,
                                size_t n_dim, intptr_t rsa, intptr_t csa,
                                intptr_t rsb, intptr_t rso) {
  for (size_t i = 0; i < m_dim; i++)
    memset(out + i * rso, 0, n_dim * sizeof(int64_t));

  for (size_t pc = 0; pc < k_dim; pc += GEMM_I64_KC) {
    size_t kc = GEMM_MIN(GEMM_I64_KC, k_dim - pc);
#ifdef _OPENMP
#pragma omp parallel for schedule( \
        static) if (m_dim * n_dim * sizeof(int64_t) > GEMM_OMP_THRESHOLD)
#endif
    for (size_t ic = 0; ic < m_dim; ic += GEMM_I64_MC) {
      size_t mc = GEMM_MIN(GEMM_I64_MC, m_dim - ic);
      gemm_edge_i64_sve(a + ic * rsa + pc * csa, b + pc * rsb, out + ic * rso,
                        mc, n_dim, kc, rsa, csa, rsb, rso);
    }
  }
}

static inline void gemm_u64_sve(const uint64_t *a, const uint64_t *b,
                                uint64_t *out, size_t m_dim, size_t k_dim,
                                size_t n_dim, intptr_t rsa, intptr_t csa,
                                intptr_t rsb, intptr_t rso) {
  gemm_i64_sve((const int64_t *)a, (const int64_t *)b, (int64_t *)out, m_dim,
               k_dim, n_dim, rsa, csa, rsb, rso);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Int8: 6x(2*VL32) promoted micro-kernel (widen to i32, full-K accumulation)
   ═══════════════════════════════════════════════════════════════════════════
 */

static inline void gemm_ukernel_i8_sve(const int8_t *a, const int8_t *b,
                                       int8_t *c, size_t k_dim, intptr_t rsa,
                                       intptr_t csa, intptr_t rsb,
                                       intptr_t rso) {
  size_t vl32 = svcntw();
  svbool_t pt32 = svptrue_b32();

  svint32_t c00 = svdup_s32(0), c01 = svdup_s32(0);
  svint32_t c10 = svdup_s32(0), c11 = svdup_s32(0);
  svint32_t c20 = svdup_s32(0), c21 = svdup_s32(0);
  svint32_t c30 = svdup_s32(0), c31 = svdup_s32(0);
  svint32_t c40 = svdup_s32(0), c41 = svdup_s32(0);
  svint32_t c50 = svdup_s32(0), c51 = svdup_s32(0);

  for (size_t p = 0; p < k_dim; p++) {
    const int8_t *bp = b + p * rsb;
    /* Load vl32 i8 values per half, widen i8 -> i16 -> i32.
     * Use svld1sb_s32 to load i8 and sign-extend directly to i32. */
    svint32_t b0 = svld1sb_s32(pt32, bp);
    svint32_t b1 = svld1sb_s32(pt32, bp + vl32);

    svint32_t av;
    av = svdup_s32((int32_t)a[0 * rsa + p * csa]);
    c00 = svmla_s32_x(pt32, c00, av, b0);
    c01 = svmla_s32_x(pt32, c01, av, b1);
    av = svdup_s32((int32_t)a[1 * rsa + p * csa]);
    c10 = svmla_s32_x(pt32, c10, av, b0);
    c11 = svmla_s32_x(pt32, c11, av, b1);
    av = svdup_s32((int32_t)a[2 * rsa + p * csa]);
    c20 = svmla_s32_x(pt32, c20, av, b0);
    c21 = svmla_s32_x(pt32, c21, av, b1);
    av = svdup_s32((int32_t)a[3 * rsa + p * csa]);
    c30 = svmla_s32_x(pt32, c30, av, b0);
    c31 = svmla_s32_x(pt32, c31, av, b1);
    av = svdup_s32((int32_t)a[4 * rsa + p * csa]);
    c40 = svmla_s32_x(pt32, c40, av, b0);
    c41 = svmla_s32_x(pt32, c41, av, b1);
    av = svdup_s32((int32_t)a[5 * rsa + p * csa]);
    c50 = svmla_s32_x(pt32, c50, av, b0);
    c51 = svmla_s32_x(pt32, c51, av, b1);
  }

  /* Narrow i32 -> i8 with saturation and store.
   * Use svst1b_s32 to narrow-store i32 as i8 (truncates). We clamp first. */
#define SVE_STORE_I8_ROW(cx0, cx1, row)                     \
  do {                                                      \
    svint32_t lo = svmax_s32_x(pt32, cx0, svdup_s32(-128)); \
    lo = svmin_s32_x(pt32, lo, svdup_s32(127));             \
    svint32_t hi = svmax_s32_x(pt32, cx1, svdup_s32(-128)); \
    hi = svmin_s32_x(pt32, hi, svdup_s32(127));             \
    svst1b_s32(pt32, c + (row) * rso, lo);                  \
    svst1b_s32(pt32, c + (row) * rso + vl32, hi);           \
  } while (0)
  SVE_STORE_I8_ROW(c00, c01, 0);
  SVE_STORE_I8_ROW(c10, c11, 1);
  SVE_STORE_I8_ROW(c20, c21, 2);
  SVE_STORE_I8_ROW(c30, c31, 3);
  SVE_STORE_I8_ROW(c40, c41, 4);
  SVE_STORE_I8_ROW(c50, c51, 5);
#undef SVE_STORE_I8_ROW
}

static inline void gemm_edge_i8_sve(const int8_t *a, const int8_t *b, int8_t *c,
                                    size_t mr, size_t nr, size_t k_dim,
                                    intptr_t rsa, intptr_t csa, intptr_t rsb,
                                    intptr_t rso) {
  for (size_t i = 0; i < mr; i++)
    for (size_t j = 0; j < nr; j++) {
      int32_t acc = 0;
      for (size_t p = 0; p < k_dim; p++)
        acc += (int32_t)a[i * rsa + p * csa] * (int32_t)b[p * rsb + j];
      c[i * rso + j] = (int8_t)acc;
    }
}

static inline void gemm_i8_sve(const int8_t *a, const int8_t *b, int8_t *out,
                               size_t m_dim, size_t k_dim, size_t n_dim,
                               intptr_t rsa, intptr_t csa, intptr_t rsb,
                               intptr_t rso) {
  size_t vl32 = svcntw();
  size_t i8_nr = 2 * vl32;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (m_dim * n_dim > \
                                                  GEMM_OMP_THRESHOLD)
#endif
  for (size_t ic = 0; ic < m_dim; ic += GEMM_I8_MC) {
    size_t mc = GEMM_MIN(GEMM_I8_MC, m_dim - ic);
    size_t jr = 0;
    for (; jr + i8_nr <= n_dim; jr += i8_nr) {
      size_t ir = 0;
      for (; ir + GEMM_I8_MR <= mc; ir += GEMM_I8_MR)
        gemm_ukernel_i8_sve(a + (ic + ir) * rsa, b + jr,
                            out + (ic + ir) * rso + jr, k_dim, rsa, csa, rsb,
                            rso);
      if (ir < mc)
        gemm_edge_i8_sve(a + (ic + ir) * rsa, b + jr,
                         out + (ic + ir) * rso + jr, mc - ir, i8_nr, k_dim, rsa,
                         csa, rsb, rso);
    }
    if (jr < n_dim)
      gemm_edge_i8_sve(a + ic * rsa, b + jr, out + ic * rso + jr, mc,
                       n_dim - jr, k_dim, rsa, csa, rsb, rso);
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   Uint8: 6x(2*VL32) promoted micro-kernel (widen to u32, full-K accumulation)
   ═══════════════════════════════════════════════════════════════════════════
 */

static inline void gemm_ukernel_u8_sve(const uint8_t *a, const uint8_t *b,
                                       uint8_t *c, size_t k_dim, intptr_t rsa,
                                       intptr_t csa, intptr_t rsb,
                                       intptr_t rso) {
  size_t vl32 = svcntw();
  svbool_t pt32 = svptrue_b32();

  svuint32_t c00 = svdup_u32(0), c01 = svdup_u32(0);
  svuint32_t c10 = svdup_u32(0), c11 = svdup_u32(0);
  svuint32_t c20 = svdup_u32(0), c21 = svdup_u32(0);
  svuint32_t c30 = svdup_u32(0), c31 = svdup_u32(0);
  svuint32_t c40 = svdup_u32(0), c41 = svdup_u32(0);
  svuint32_t c50 = svdup_u32(0), c51 = svdup_u32(0);

  for (size_t p = 0; p < k_dim; p++) {
    const uint8_t *bp = b + p * rsb;
    /* Use svld1ub_u32 to load u8 and zero-extend directly to u32. */
    svuint32_t b0 = svld1ub_u32(pt32, bp);
    svuint32_t b1 = svld1ub_u32(pt32, bp + vl32);

    svuint32_t av;
    av = svdup_u32((uint32_t)a[0 * rsa + p * csa]);
    c00 = svmla_u32_x(pt32, c00, av, b0);
    c01 = svmla_u32_x(pt32, c01, av, b1);
    av = svdup_u32((uint32_t)a[1 * rsa + p * csa]);
    c10 = svmla_u32_x(pt32, c10, av, b0);
    c11 = svmla_u32_x(pt32, c11, av, b1);
    av = svdup_u32((uint32_t)a[2 * rsa + p * csa]);
    c20 = svmla_u32_x(pt32, c20, av, b0);
    c21 = svmla_u32_x(pt32, c21, av, b1);
    av = svdup_u32((uint32_t)a[3 * rsa + p * csa]);
    c30 = svmla_u32_x(pt32, c30, av, b0);
    c31 = svmla_u32_x(pt32, c31, av, b1);
    av = svdup_u32((uint32_t)a[4 * rsa + p * csa]);
    c40 = svmla_u32_x(pt32, c40, av, b0);
    c41 = svmla_u32_x(pt32, c41, av, b1);
    av = svdup_u32((uint32_t)a[5 * rsa + p * csa]);
    c50 = svmla_u32_x(pt32, c50, av, b0);
    c51 = svmla_u32_x(pt32, c51, av, b1);
  }

  /* Narrow u32 -> u8 with saturation and store.
   * Clamp to [0, 255], then use svst1b_u32 to narrow-store. */
#define SVE_STORE_U8_ROW(cx0, cx1, row)                     \
  do {                                                      \
    svuint32_t lo = svmin_u32_x(pt32, cx0, svdup_u32(255)); \
    svuint32_t hi = svmin_u32_x(pt32, cx1, svdup_u32(255)); \
    svst1b_u32(pt32, c + (row) * rso, lo);                  \
    svst1b_u32(pt32, c + (row) * rso + vl32, hi);           \
  } while (0)
  SVE_STORE_U8_ROW(c00, c01, 0);
  SVE_STORE_U8_ROW(c10, c11, 1);
  SVE_STORE_U8_ROW(c20, c21, 2);
  SVE_STORE_U8_ROW(c30, c31, 3);
  SVE_STORE_U8_ROW(c40, c41, 4);
  SVE_STORE_U8_ROW(c50, c51, 5);
#undef SVE_STORE_U8_ROW
}

static inline void gemm_edge_u8_sve(const uint8_t *a, const uint8_t *b,
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

static inline void gemm_u8_sve(const uint8_t *a, const uint8_t *b, uint8_t *out,
                               size_t m_dim, size_t k_dim, size_t n_dim,
                               intptr_t rsa, intptr_t csa, intptr_t rsb,
                               intptr_t rso) {
  size_t vl32 = svcntw();
  size_t u8_nr = 2 * vl32;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (m_dim * n_dim > \
                                                  GEMM_OMP_THRESHOLD)
#endif
  for (size_t ic = 0; ic < m_dim; ic += GEMM_I8_MC) {
    size_t mc = GEMM_MIN(GEMM_I8_MC, m_dim - ic);
    size_t jr = 0;
    for (; jr + u8_nr <= n_dim; jr += u8_nr) {
      size_t ir = 0;
      for (; ir + GEMM_I8_MR <= mc; ir += GEMM_I8_MR)
        gemm_ukernel_u8_sve(a + (ic + ir) * rsa, b + jr,
                            out + (ic + ir) * rso + jr, k_dim, rsa, csa, rsb,
                            rso);
      if (ir < mc)
        gemm_edge_u8_sve(a + (ic + ir) * rsa, b + jr,
                         out + (ic + ir) * rso + jr, mc - ir, u8_nr, k_dim, rsa,
                         csa, rsb, rso);
    }
    if (jr < n_dim)
      gemm_edge_u8_sve(a + ic * rsa, b + jr, out + ic * rso + jr, mc,
                       n_dim - jr, k_dim, rsa, csa, rsb, rso);
  }
}

#undef GEMM_MIN

#endif
