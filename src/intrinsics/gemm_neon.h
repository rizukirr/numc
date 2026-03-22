#ifndef NUMC_GEMM_NEON_H
#define NUMC_GEMM_NEON_H

#include <arm_neon.h>
#include <stdint.h>
#include <string.h>

#define GEMM_MIN(a, b) ((a) < (b) ? (a) : (b))

/*
 * Cache-blocking parameters for NEON packed GEMM (BLIS Cortex-A57 derived).
 * AArch64 has 32 x 128-bit NEON registers (v0-v31).
 *
 * f32 8x12: 24 acc + 3 B + 2 A = 29/32 registers.
 *   KC x NR sliver in L1: 256 x 12 x 4 = 12KB < 32KB
 *   MC x KC panel  in L2: 128 x 256 x 4 = 128KB < 256KB
 *
 * f64 6x8: 24 acc + 4 B + 3 A = 31/32 registers.
 *   KC x NR sliver in L1: 256 x 8 x 8 = 16KB < 32KB
 *   MC x KC panel  in L2: 96 x 256 x 8 = 196KB < 256KB
 */
#define GEMM_F32_MR 8
#define GEMM_F32_NR 12
#define GEMM_F32_MC 128
#define GEMM_F32_KC 256

#define GEMM_F64_MR 6
#define GEMM_F64_NR 8
#define GEMM_F64_MC 96
#define GEMM_F64_KC 256

#define GEMM_I32_MR 6
#define GEMM_I32_NR 8
#define GEMM_I32_MC 72
#define GEMM_I32_KC 256

#define GEMM_I16_MR 6
#define GEMM_I16_NR 16
#define GEMM_I16_MC 72
#define GEMM_I16_KC 512

#define GEMM_I64_MR 6
#define GEMM_I64_NR 4
#define GEMM_I64_MC 72
#define GEMM_I64_KC 64

#define GEMM_I8_MR 6
#define GEMM_I8_NR 8
#define GEMM_I8_MC 72

#define GEMM_OMP_THRESHOLD (1 << 23)

#define GEMM_F32_NC 4080
#define GEMM_F64_NC 2048

/* ── Float32 packing routines ──────────────────────────────────────────── */

static inline void gemm_pack_b_f32(const float *b, float *packed, size_t kc,
                                   size_t nc, intptr_t rsb) {
  size_t jr = 0;
  for (; jr + GEMM_F32_NR <= nc; jr += GEMM_F32_NR) {
    float *dest = packed + jr * kc;
    for (size_t p = 0; p < kc; p++) {
      const float *src = b + p * rsb + jr;
      __builtin_prefetch(src + 4 * rsb, 0, 3);
      vst1q_f32(dest + p * GEMM_F32_NR, vld1q_f32(src));
      vst1q_f32(dest + p * GEMM_F32_NR + 4, vld1q_f32(src + 4));
      vst1q_f32(dest + p * GEMM_F32_NR + 8, vld1q_f32(src + 8));
    }
  }
  if (jr < nc) {
    float *dest = packed + jr * kc;
    size_t rem = nc - jr;
    for (size_t p = 0; p < kc; p++) {
      const float *src = b + p * rsb + jr;
      size_t j = 0;
      /* SIMD fast path for common remainder sizes */
      if (rem >= 8) {
        vst1q_f32(dest + p * GEMM_F32_NR, vld1q_f32(src));
        vst1q_f32(dest + p * GEMM_F32_NR + 4, vld1q_f32(src + 4));
        j = 8;
      } else if (rem >= 4) {
        vst1q_f32(dest + p * GEMM_F32_NR, vld1q_f32(src));
        j = 4;
      }
      for (; j < rem; j++)
        dest[p * GEMM_F32_NR + j] = src[j];
      for (; j < GEMM_F32_NR; j++)
        dest[p * GEMM_F32_NR + j] = 0.0f;
    }
  }
}

static inline void _gemm_pack_b_strip_f32(const float *b, float *dest,
                                          size_t kc, size_t nr_pack,
                                          intptr_t rsb) {
  if (nr_pack == GEMM_F32_NR) {
    for (size_t p = 0; p < kc; p++) {
      const float *src = b + p * rsb;
      __builtin_prefetch(src + 4 * rsb, 0, 3);
      vst1q_f32(dest + p * GEMM_F32_NR, vld1q_f32(src));
      vst1q_f32(dest + p * GEMM_F32_NR + 4, vld1q_f32(src + 4));
      vst1q_f32(dest + p * GEMM_F32_NR + 8, vld1q_f32(src + 8));
    }
  } else {
    for (size_t p = 0; p < kc; p++) {
      const float *src = b + p * rsb;
      size_t j = 0;
      for (; j < nr_pack; j++)
        dest[p * GEMM_F32_NR + j] = src[j];
      for (; j < GEMM_F32_NR; j++)
        dest[p * GEMM_F32_NR + j] = 0.0f;
    }
  }
}

static inline void gemm_pack_a_f32(const float *a, float *packed, size_t mc,
                                   size_t kc, intptr_t rsa, intptr_t csa) {
  if (csa == 1) {
    /* Fast path: row-major A — transpose 8 rows x 4 K-cols at a time */
    size_t ir = 0;
    for (; ir + GEMM_F32_MR <= mc; ir += GEMM_F32_MR) {
      float *dest = packed + ir * kc;
      const float *r0 = a + (ir + 0) * rsa;
      const float *r1 = a + (ir + 1) * rsa;
      const float *r2 = a + (ir + 2) * rsa;
      const float *r3 = a + (ir + 3) * rsa;
      const float *r4 = a + (ir + 4) * rsa;
      const float *r5 = a + (ir + 5) * rsa;
      const float *r6 = a + (ir + 6) * rsa;
      const float *r7 = a + (ir + 7) * rsa;
      /* Prefetch next MR-panel's rows */
      if (ir + 2 * GEMM_F32_MR <= mc) {
        for (size_t i = 0; i < GEMM_F32_MR; i++)
          __builtin_prefetch(a + (ir + GEMM_F32_MR + i) * rsa, 0, 3);
      }
      size_t p = 0;
      /* Process 4 K-columns at a time via 4x4 NEON transpose:
       * vzip1q_f32(a,b) = [a0,b0,a1,b1], vzip2q_f32(a,b) = [a2,b2,a3,b3]
       * Then vcombine low/high halves to get correct column layout. */
      for (; p + 4 <= kc; p += 4) {
        /* Load 4 consecutive floats from each of 8 rows */
        float32x4_t row0 = vld1q_f32(r0 + p);
        float32x4_t row1 = vld1q_f32(r1 + p);
        float32x4_t row2 = vld1q_f32(r2 + p);
        float32x4_t row3 = vld1q_f32(r3 + p);
        float32x4_t row4 = vld1q_f32(r4 + p);
        float32x4_t row5 = vld1q_f32(r5 + p);
        float32x4_t row6 = vld1q_f32(r6 + p);
        float32x4_t row7 = vld1q_f32(r7 + p);
        /* Transpose rows 0-3: step 1 — interleave pairs */
        float32x4_t t01a =
            vzip1q_f32(row0, row1); /* [r0[0],r1[0],r0[1],r1[1]] */
        float32x4_t t01b =
            vzip2q_f32(row0, row1); /* [r0[2],r1[2],r0[3],r1[3]] */
        float32x4_t t23a =
            vzip1q_f32(row2, row3); /* [r2[0],r3[0],r2[1],r3[1]] */
        float32x4_t t23b =
            vzip2q_f32(row2, row3); /* [r2[2],r3[2],r2[3],r3[3]] */
        /* col p+0: [r0[0],r1[0],r2[0],r3[0]] */
        float32x4_t col0_lo =
            vcombine_f32(vget_low_f32(t01a), vget_low_f32(t23a));
        /* col p+1: [r0[1],r1[1],r2[1],r3[1]] */
        float32x4_t col1_lo =
            vcombine_f32(vget_high_f32(t01a), vget_high_f32(t23a));
        /* col p+2: [r0[2],r1[2],r2[2],r3[2]] */
        float32x4_t col2_lo =
            vcombine_f32(vget_low_f32(t01b), vget_low_f32(t23b));
        /* col p+3: [r0[3],r1[3],r2[3],r3[3]] */
        float32x4_t col3_lo =
            vcombine_f32(vget_high_f32(t01b), vget_high_f32(t23b));
        /* Transpose rows 4-7: same pattern */
        float32x4_t t45a = vzip1q_f32(row4, row5);
        float32x4_t t45b = vzip2q_f32(row4, row5);
        float32x4_t t67a = vzip1q_f32(row6, row7);
        float32x4_t t67b = vzip2q_f32(row6, row7);
        float32x4_t col0_hi =
            vcombine_f32(vget_low_f32(t45a), vget_low_f32(t67a));
        float32x4_t col1_hi =
            vcombine_f32(vget_high_f32(t45a), vget_high_f32(t67a));
        float32x4_t col2_hi =
            vcombine_f32(vget_low_f32(t45b), vget_low_f32(t67b));
        float32x4_t col3_hi =
            vcombine_f32(vget_high_f32(t45b), vget_high_f32(t67b));
        /* Store: each K-column p becomes packed row of 8 = [rows0-3, rows4-7]
         */
        vst1q_f32(dest + (p + 0) * GEMM_F32_MR, col0_lo);
        vst1q_f32(dest + (p + 0) * GEMM_F32_MR + 4, col0_hi);
        vst1q_f32(dest + (p + 1) * GEMM_F32_MR, col1_lo);
        vst1q_f32(dest + (p + 1) * GEMM_F32_MR + 4, col1_hi);
        vst1q_f32(dest + (p + 2) * GEMM_F32_MR, col2_lo);
        vst1q_f32(dest + (p + 2) * GEMM_F32_MR + 4, col2_hi);
        vst1q_f32(dest + (p + 3) * GEMM_F32_MR, col3_lo);
        vst1q_f32(dest + (p + 3) * GEMM_F32_MR + 4, col3_hi);
      }
      /* Scalar cleanup for remaining K columns */
      for (; p < kc; p++) {
        dest[p * GEMM_F32_MR + 0] = r0[p];
        dest[p * GEMM_F32_MR + 1] = r1[p];
        dest[p * GEMM_F32_MR + 2] = r2[p];
        dest[p * GEMM_F32_MR + 3] = r3[p];
        dest[p * GEMM_F32_MR + 4] = r4[p];
        dest[p * GEMM_F32_MR + 5] = r5[p];
        dest[p * GEMM_F32_MR + 6] = r6[p];
        dest[p * GEMM_F32_MR + 7] = r7[p];
      }
    }
    if (ir < mc) {
      float *dest = packed + ir * kc;
      size_t rem = mc - ir;
      for (size_t p = 0; p < kc; p++) {
        size_t i = 0;
        for (; i < rem; i++)
          dest[p * GEMM_F32_MR + i] = a[(ir + i) * rsa + p];
        for (; i < GEMM_F32_MR; i++)
          dest[p * GEMM_F32_MR + i] = 0.0f;
      }
    }
  } else {
    /* Generic path: arbitrary stride */
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
}

/* ── Float64 packing routines ──────────────────────────────────────────── */

static inline void gemm_pack_b_f64(const double *b, double *packed, size_t kc,
                                   size_t nc, intptr_t rsb) {
  size_t jr = 0;
  for (; jr + GEMM_F64_NR <= nc; jr += GEMM_F64_NR) {
    double *dest = packed + jr * kc;
    for (size_t p = 0; p < kc; p++) {
      const double *src = b + p * rsb + jr;
      __builtin_prefetch(src + 4 * rsb, 0, 3);
      vst1q_f64(dest + p * GEMM_F64_NR, vld1q_f64(src));
      vst1q_f64(dest + p * GEMM_F64_NR + 2, vld1q_f64(src + 2));
      vst1q_f64(dest + p * GEMM_F64_NR + 4, vld1q_f64(src + 4));
      vst1q_f64(dest + p * GEMM_F64_NR + 6, vld1q_f64(src + 6));
    }
  }
  if (jr < nc) {
    double *dest = packed + jr * kc;
    size_t rem = nc - jr;
    for (size_t p = 0; p < kc; p++) {
      const double *src = b + p * rsb + jr;
      size_t j = 0;
      /* SIMD fast path for common remainder sizes */
      if (rem >= 4) {
        vst1q_f64(dest + p * GEMM_F64_NR, vld1q_f64(src));
        vst1q_f64(dest + p * GEMM_F64_NR + 2, vld1q_f64(src + 2));
        j = 4;
      } else if (rem >= 2) {
        vst1q_f64(dest + p * GEMM_F64_NR, vld1q_f64(src));
        j = 2;
      }
      for (; j < rem; j++)
        dest[p * GEMM_F64_NR + j] = src[j];
      for (; j < GEMM_F64_NR; j++)
        dest[p * GEMM_F64_NR + j] = 0.0;
    }
  }
}

static inline void _gemm_pack_b_strip_f64(const double *b, double *dest,
                                          size_t kc, size_t nr_pack,
                                          intptr_t rsb) {
  if (nr_pack == GEMM_F64_NR) {
    for (size_t p = 0; p < kc; p++) {
      const double *src = b + p * rsb;
      __builtin_prefetch(src + 4 * rsb, 0, 3);
      vst1q_f64(dest + p * GEMM_F64_NR, vld1q_f64(src));
      vst1q_f64(dest + p * GEMM_F64_NR + 2, vld1q_f64(src + 2));
      vst1q_f64(dest + p * GEMM_F64_NR + 4, vld1q_f64(src + 4));
      vst1q_f64(dest + p * GEMM_F64_NR + 6, vld1q_f64(src + 6));
    }
  } else {
    for (size_t p = 0; p < kc; p++) {
      const double *src = b + p * rsb;
      size_t j = 0;
      for (; j < nr_pack; j++)
        dest[p * GEMM_F64_NR + j] = src[j];
      for (; j < GEMM_F64_NR; j++)
        dest[p * GEMM_F64_NR + j] = 0.0;
    }
  }
}

static inline void gemm_pack_a_f64(const double *a, double *packed, size_t mc,
                                   size_t kc, intptr_t rsa, intptr_t csa) {
  if (csa == 1) {
    /* Fast path: row-major A — transpose 2 K-cols at a time (2-lane vectors) */
    size_t ir = 0;
    for (; ir + GEMM_F64_MR <= mc; ir += GEMM_F64_MR) {
      double *dest = packed + ir * kc;
      const double *r0 = a + (ir + 0) * rsa;
      const double *r1 = a + (ir + 1) * rsa;
      const double *r2 = a + (ir + 2) * rsa;
      const double *r3 = a + (ir + 3) * rsa;
      const double *r4 = a + (ir + 4) * rsa;
      const double *r5 = a + (ir + 5) * rsa;
      /* Prefetch next MR-panel's rows */
      if (ir + 2 * GEMM_F64_MR <= mc) {
        for (size_t i = 0; i < GEMM_F64_MR; i++)
          __builtin_prefetch(a + (ir + GEMM_F64_MR + i) * rsa, 0, 3);
      }
      size_t p = 0;
      /* Process 2 K-columns at a time: transpose 2x2 blocks */
      for (; p + 2 <= kc; p += 2) {
        float64x2_t row0 = vld1q_f64(r0 + p); /* [r0[p], r0[p+1]] */
        float64x2_t row1 = vld1q_f64(r1 + p); /* [r1[p], r1[p+1]] */
        float64x2_t row2 = vld1q_f64(r2 + p);
        float64x2_t row3 = vld1q_f64(r3 + p);
        float64x2_t row4 = vld1q_f64(r4 + p);
        float64x2_t row5 = vld1q_f64(r5 + p);
        /* col_p   = [r0[p], r1[p]] */
        /* col_p1  = [r0[p+1], r1[p+1]] */
        float64x2_t col_p_01 = vtrn1q_f64(row0, row1);
        float64x2_t col_p1_01 = vtrn2q_f64(row0, row1);
        float64x2_t col_p_23 = vtrn1q_f64(row2, row3);
        float64x2_t col_p1_23 = vtrn2q_f64(row2, row3);
        float64x2_t col_p_45 = vtrn1q_f64(row4, row5);
        float64x2_t col_p1_45 = vtrn2q_f64(row4, row5);
        /* Store 6 doubles per K-column */
        vst1q_f64(dest + (p + 0) * GEMM_F64_MR + 0, col_p_01);
        vst1q_f64(dest + (p + 0) * GEMM_F64_MR + 2, col_p_23);
        vst1q_f64(dest + (p + 0) * GEMM_F64_MR + 4, col_p_45);
        vst1q_f64(dest + (p + 1) * GEMM_F64_MR + 0, col_p1_01);
        vst1q_f64(dest + (p + 1) * GEMM_F64_MR + 2, col_p1_23);
        vst1q_f64(dest + (p + 1) * GEMM_F64_MR + 4, col_p1_45);
      }
      /* Scalar cleanup for remaining K column */
      for (; p < kc; p++) {
        dest[p * GEMM_F64_MR + 0] = r0[p];
        dest[p * GEMM_F64_MR + 1] = r1[p];
        dest[p * GEMM_F64_MR + 2] = r2[p];
        dest[p * GEMM_F64_MR + 3] = r3[p];
        dest[p * GEMM_F64_MR + 4] = r4[p];
        dest[p * GEMM_F64_MR + 5] = r5[p];
      }
    }
    if (ir < mc) {
      double *dest = packed + ir * kc;
      size_t rem = mc - ir;
      for (size_t p = 0; p < kc; p++) {
        size_t i = 0;
        for (; i < rem; i++)
          dest[p * GEMM_F64_MR + i] = a[(ir + i) * rsa + p];
        for (; i < GEMM_F64_MR; i++)
          dest[p * GEMM_F64_MR + i] = 0.0;
      }
    }
  } else {
    /* Generic path: arbitrary stride */
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
}

/* ═══════════════════════════════════════════════════════════════════════════
   Float32: 8x12 micro-kernel (24 acc + 3 B + 2 A = 29/32 NEON regs)
   Uses vfmaq_laneq_f32 for fused broadcast+FMA — avoids explicit broadcast.
   ═══════════════════════════════════════════════════════════════════════════
 */

/* One K-iteration: uses pre-loaded b0/b1/b2 from outer scope, loads A,
 * does 24 FMA, then loads NEXT B into b0/b1/b2 (BLIS-style pipelining).
 * bp must point to CURRENT iter; next B is loaded from bp+rsb. */
#define GEMM_F32_K_ITER(ap, bp)            \
  do {                                     \
    float32x4_t a0 = vld1q_f32(ap);        \
    float32x4_t a1 = vld1q_f32((ap) + 4);  \
    c00 = vfmaq_laneq_f32(c00, b0, a0, 0); \
    c01 = vfmaq_laneq_f32(c01, b1, a0, 0); \
    c02 = vfmaq_laneq_f32(c02, b2, a0, 0); \
    c10 = vfmaq_laneq_f32(c10, b0, a0, 1); \
    c11 = vfmaq_laneq_f32(c11, b1, a0, 1); \
    c12 = vfmaq_laneq_f32(c12, b2, a0, 1); \
    c20 = vfmaq_laneq_f32(c20, b0, a0, 2); \
    c21 = vfmaq_laneq_f32(c21, b1, a0, 2); \
    c22 = vfmaq_laneq_f32(c22, b2, a0, 2); \
    c30 = vfmaq_laneq_f32(c30, b0, a0, 3); \
    c31 = vfmaq_laneq_f32(c31, b1, a0, 3); \
    c32 = vfmaq_laneq_f32(c32, b2, a0, 3); \
    c40 = vfmaq_laneq_f32(c40, b0, a1, 0); \
    c41 = vfmaq_laneq_f32(c41, b1, a1, 0); \
    c42 = vfmaq_laneq_f32(c42, b2, a1, 0); \
    c50 = vfmaq_laneq_f32(c50, b0, a1, 1); \
    c51 = vfmaq_laneq_f32(c51, b1, a1, 1); \
    c52 = vfmaq_laneq_f32(c52, b2, a1, 1); \
    c60 = vfmaq_laneq_f32(c60, b0, a1, 2); \
    c61 = vfmaq_laneq_f32(c61, b1, a1, 2); \
    c62 = vfmaq_laneq_f32(c62, b2, a1, 2); \
    c70 = vfmaq_laneq_f32(c70, b0, a1, 3); \
    c71 = vfmaq_laneq_f32(c71, b1, a1, 3); \
    c72 = vfmaq_laneq_f32(c72, b2, a1, 3); \
    b0 = vld1q_f32((bp) + rsb);            \
    b1 = vld1q_f32((bp) + rsb + 4);        \
    b2 = vld1q_f32((bp) + rsb + 8);        \
  } while (0)

static inline void gemm_ukernel_f32_8x12(const float *a, const float *b,
                                         float *c, size_t kc, intptr_t rsa,
                                         intptr_t csa, intptr_t rsb,
                                         intptr_t rso, int first) {
  float32x4_t c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32, c40,
      c41, c42, c50, c51, c52, c60, c61, c62, c70, c71, c72;

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
    c00 = vdupq_n_f32(0);
    c01 = vdupq_n_f32(0);
    c02 = vdupq_n_f32(0);
    c10 = vdupq_n_f32(0);
    c11 = vdupq_n_f32(0);
    c12 = vdupq_n_f32(0);
    c20 = vdupq_n_f32(0);
    c21 = vdupq_n_f32(0);
    c22 = vdupq_n_f32(0);
    c30 = vdupq_n_f32(0);
    c31 = vdupq_n_f32(0);
    c32 = vdupq_n_f32(0);
    c40 = vdupq_n_f32(0);
    c41 = vdupq_n_f32(0);
    c42 = vdupq_n_f32(0);
    c50 = vdupq_n_f32(0);
    c51 = vdupq_n_f32(0);
    c52 = vdupq_n_f32(0);
    c60 = vdupq_n_f32(0);
    c61 = vdupq_n_f32(0);
    c62 = vdupq_n_f32(0);
    c70 = vdupq_n_f32(0);
    c71 = vdupq_n_f32(0);
    c72 = vdupq_n_f32(0);
  } else {
    c00 = vld1q_f32(c);
    c01 = vld1q_f32(c + 4);
    c02 = vld1q_f32(c + 8);
    c10 = vld1q_f32(c + rso);
    c11 = vld1q_f32(c + rso + 4);
    c12 = vld1q_f32(c + rso + 8);
    c20 = vld1q_f32(c + 2 * rso);
    c21 = vld1q_f32(c + 2 * rso + 4);
    c22 = vld1q_f32(c + 2 * rso + 8);
    c30 = vld1q_f32(c + 3 * rso);
    c31 = vld1q_f32(c + 3 * rso + 4);
    c32 = vld1q_f32(c + 3 * rso + 8);
    c40 = vld1q_f32(c + 4 * rso);
    c41 = vld1q_f32(c + 4 * rso + 4);
    c42 = vld1q_f32(c + 4 * rso + 8);
    c50 = vld1q_f32(c + 5 * rso);
    c51 = vld1q_f32(c + 5 * rso + 4);
    c52 = vld1q_f32(c + 5 * rso + 8);
    c60 = vld1q_f32(c + 6 * rso);
    c61 = vld1q_f32(c + 6 * rso + 4);
    c62 = vld1q_f32(c + 6 * rso + 8);
    c70 = vld1q_f32(c + 7 * rso);
    c71 = vld1q_f32(c + 7 * rso + 4);
    c72 = vld1q_f32(c + 7 * rso + 8);
  }

  const float *ap = a;
  const float *bp = b;
  size_t k_iter = kc / 8;
  size_t k_left = kc % 8;

  /* Pre-load first B tile (BLIS-style pipelining) */
  float32x4_t b0 = vld1q_f32(bp);
  float32x4_t b1 = vld1q_f32(bp + 4);
  float32x4_t b2 = vld1q_f32(bp + 8);

  for (size_t ki = 0; ki < k_iter; ki++) {
    GEMM_F32_K_ITER(ap, bp);
    ap += csa;
    bp += rsb;
    GEMM_F32_K_ITER(ap, bp);
    ap += csa;
    bp += rsb;
    __builtin_prefetch(ap + 64, 0, 3);
    GEMM_F32_K_ITER(ap, bp);
    ap += csa;
    bp += rsb;
    GEMM_F32_K_ITER(ap, bp);
    ap += csa;
    bp += rsb;
    __builtin_prefetch(bp + 96, 0, 3);
    GEMM_F32_K_ITER(ap, bp);
    ap += csa;
    bp += rsb;
    GEMM_F32_K_ITER(ap, bp);
    ap += csa;
    bp += rsb;
    __builtin_prefetch(ap + 128, 0, 3);
    GEMM_F32_K_ITER(ap, bp);
    ap += csa;
    bp += rsb;
    GEMM_F32_K_ITER(ap, bp);
    ap += csa;
    bp += rsb;
  }
  for (size_t ki = 0; ki < k_left; ki++) {
    GEMM_F32_K_ITER(ap, bp);
    ap += csa;
    bp += rsb;
  }

  /* C prefetch refresh before store */
  __builtin_prefetch(c, 1, 3);
  __builtin_prefetch(c + rso, 1, 3);
  __builtin_prefetch(c + 2 * rso, 1, 3);
  __builtin_prefetch(c + 3 * rso, 1, 3);
  __builtin_prefetch(c + 4 * rso, 1, 3);
  __builtin_prefetch(c + 5 * rso, 1, 3);
  __builtin_prefetch(c + 6 * rso, 1, 3);
  __builtin_prefetch(c + 7 * rso, 1, 3);

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
  vst1q_f32(c + 6 * rso, c60);
  vst1q_f32(c + 6 * rso + 4, c61);
  vst1q_f32(c + 6 * rso + 8, c62);
  vst1q_f32(c + 7 * rso, c70);
  vst1q_f32(c + 7 * rso + 4, c71);
  vst1q_f32(c + 7 * rso + 8, c72);
}

#undef GEMM_F32_K_ITER

static inline void gemm_edge_f32_neon(const float *a, const float *b, float *c,
                                      size_t mr, size_t nr, size_t kc,
                                      intptr_t rsa, intptr_t csa, intptr_t rsb,
                                      intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
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

static inline void gemm_f32_neon(const float *a, const float *b, float *out,
                                 size_t m_dim, size_t k_dim, size_t n_dim,
                                 intptr_t rsa, intptr_t csa, intptr_t rsb,
                                 intptr_t rso) {
  size_t nc_max = GEMM_MIN(GEMM_F32_NC, n_dim);
  float *packed_b = (float *)numc_malloc(
      16, GEMM_F32_KC * (nc_max + GEMM_F32_NR) * sizeof(float));
  if (!packed_b)
    return;

  for (size_t jc = 0; jc < n_dim; jc += GEMM_F32_NC) {
    size_t nc = GEMM_MIN(GEMM_F32_NC, n_dim - jc);
    size_t n_jr = (nc + GEMM_F32_NR - 1) / GEMM_F32_NR;

#ifdef _OPENMP
#pragma omp parallel if ((uint64_t)m_dim * k_dim * nc > GEMM_OMP_THRESHOLD)
    {
      NUMC_ALIGNAS(16) float packed_a[GEMM_F32_MC * GEMM_F32_KC];

      for (size_t pc = 0; pc < k_dim; pc += GEMM_F32_KC) {
        size_t kc = GEMM_MIN(GEMM_F32_KC, k_dim - pc);
        int first = (pc == 0);
        size_t last_ic = (size_t)-1;

#pragma omp for schedule(static)
        for (size_t jr_idx = 0; jr_idx < n_jr; jr_idx++) {
          size_t jj = jr_idx * GEMM_F32_NR;
          size_t nr_pack = GEMM_MIN(GEMM_F32_NR, nc - jj);
          _gemm_pack_b_strip_f32(b + pc * rsb + jc + jj, packed_b + jj * kc, kc,
                                 nr_pack, rsb);
        }

        size_t n_ic = (m_dim + GEMM_F32_MC - 1) / GEMM_F32_MC;
        size_t n_tasks = n_ic * n_jr;

#pragma omp for schedule(static)
        for (size_t task = 0; task < n_tasks; task++) {
          size_t ic = (task / n_jr) * GEMM_F32_MC;
          size_t jr = (task % n_jr) * GEMM_F32_NR;
          size_t mc = GEMM_MIN(GEMM_F32_MC, m_dim - ic);
          size_t nr_cur = GEMM_MIN(GEMM_F32_NR, nc - jr);

          if (ic != last_ic) {
            gemm_pack_a_f32(a + ic * rsa + pc * csa, packed_a, mc, kc, rsa,
                            csa);
            last_ic = ic;
          }

          for (size_t ir = 0; ir < mc; ir += GEMM_F32_MR) {
            size_t mr_cur = GEMM_MIN(GEMM_F32_MR, mc - ir);
            if (mr_cur == GEMM_F32_MR && nr_cur == GEMM_F32_NR) {
              gemm_ukernel_f32_8x12(packed_a + ir * kc, packed_b + jr * kc,
                                    out + (ic + ir) * rso + (jc + jr), kc, 1,
                                    GEMM_F32_MR, GEMM_F32_NR, rso, first);
            } else {
              NUMC_ALIGNAS(16) float tmp[GEMM_F32_MR * GEMM_F32_NR];
              gemm_ukernel_f32_8x12(packed_a + ir * kc, packed_b + jr * kc, tmp,
                                    kc, 1, GEMM_F32_MR, GEMM_F32_NR,
                                    GEMM_F32_NR, 1);
              float *dst = out + (ic + ir) * rso + (jc + jr);
              if (first) {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] = tmp[ii * GEMM_F32_NR + jj];
              } else {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] += tmp[ii * GEMM_F32_NR + jj];
              }
            }
          }
        }
      }
    }
#else
    {
      NUMC_ALIGNAS(16) float packed_a[GEMM_F32_MC * GEMM_F32_KC];

      for (size_t pc = 0; pc < k_dim; pc += GEMM_F32_KC) {
        size_t kc = GEMM_MIN(GEMM_F32_KC, k_dim - pc);
        int first = (pc == 0);

        gemm_pack_b_f32(b + pc * rsb + jc, packed_b, kc, nc, rsb);

        size_t n_ic = (m_dim + GEMM_F32_MC - 1) / GEMM_F32_MC;
        size_t n_tasks = n_ic * n_jr;

        for (size_t task = 0; task < n_tasks; task++) {
          size_t ic = (task / n_jr) * GEMM_F32_MC;
          size_t jr = (task % n_jr) * GEMM_F32_NR;
          size_t mc = GEMM_MIN(GEMM_F32_MC, m_dim - ic);
          size_t nr_cur = GEMM_MIN(GEMM_F32_NR, nc - jr);

          if (task % n_jr == 0)
            gemm_pack_a_f32(a + ic * rsa + pc * csa, packed_a, mc, kc, rsa,
                            csa);

          for (size_t ir = 0; ir < mc; ir += GEMM_F32_MR) {
            size_t mr_cur = GEMM_MIN(GEMM_F32_MR, mc - ir);
            if (mr_cur == GEMM_F32_MR && nr_cur == GEMM_F32_NR) {
              gemm_ukernel_f32_8x12(packed_a + ir * kc, packed_b + jr * kc,
                                    out + (ic + ir) * rso + (jc + jr), kc, 1,
                                    GEMM_F32_MR, GEMM_F32_NR, rso, first);
            } else {
              NUMC_ALIGNAS(16) float tmp[GEMM_F32_MR * GEMM_F32_NR];
              gemm_ukernel_f32_8x12(packed_a + ir * kc, packed_b + jr * kc, tmp,
                                    kc, 1, GEMM_F32_MR, GEMM_F32_NR,
                                    GEMM_F32_NR, 1);
              float *dst = out + (ic + ir) * rso + (jc + jr);
              if (first) {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] = tmp[ii * GEMM_F32_NR + jj];
              } else {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] += tmp[ii * GEMM_F32_NR + jj];
              }
            }
          }
        }
      }
    }
#endif
  }

  numc_free(packed_b);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Float64: 6x8 micro-kernel (24 acc + 4 B + 3 A = 31/32 NEON regs)
   ═══════════════════════════════════════════════════════════════════════════
 */

/* Uses pre-loaded b0/b1/b2/b3 from outer scope, loads A,
 * does 24 FMA, then loads NEXT B (BLIS-style pipelining). */
#define GEMM_F64_K_ITER(ap, bp)            \
  do {                                     \
    float64x2_t a0 = vld1q_f64(ap);        \
    float64x2_t a1 = vld1q_f64((ap) + 2);  \
    float64x2_t a2 = vld1q_f64((ap) + 4);  \
    c00 = vfmaq_laneq_f64(c00, b0, a0, 0); \
    c01 = vfmaq_laneq_f64(c01, b1, a0, 0); \
    c02 = vfmaq_laneq_f64(c02, b2, a0, 0); \
    c03 = vfmaq_laneq_f64(c03, b3, a0, 0); \
    c10 = vfmaq_laneq_f64(c10, b0, a0, 1); \
    c11 = vfmaq_laneq_f64(c11, b1, a0, 1); \
    c12 = vfmaq_laneq_f64(c12, b2, a0, 1); \
    c13 = vfmaq_laneq_f64(c13, b3, a0, 1); \
    c20 = vfmaq_laneq_f64(c20, b0, a1, 0); \
    c21 = vfmaq_laneq_f64(c21, b1, a1, 0); \
    c22 = vfmaq_laneq_f64(c22, b2, a1, 0); \
    c23 = vfmaq_laneq_f64(c23, b3, a1, 0); \
    c30 = vfmaq_laneq_f64(c30, b0, a1, 1); \
    c31 = vfmaq_laneq_f64(c31, b1, a1, 1); \
    c32 = vfmaq_laneq_f64(c32, b2, a1, 1); \
    c33 = vfmaq_laneq_f64(c33, b3, a1, 1); \
    c40 = vfmaq_laneq_f64(c40, b0, a2, 0); \
    c41 = vfmaq_laneq_f64(c41, b1, a2, 0); \
    c42 = vfmaq_laneq_f64(c42, b2, a2, 0); \
    c43 = vfmaq_laneq_f64(c43, b3, a2, 0); \
    c50 = vfmaq_laneq_f64(c50, b0, a2, 1); \
    c51 = vfmaq_laneq_f64(c51, b1, a2, 1); \
    c52 = vfmaq_laneq_f64(c52, b2, a2, 1); \
    c53 = vfmaq_laneq_f64(c53, b3, a2, 1); \
    b0 = vld1q_f64((bp) + rsb);            \
    b1 = vld1q_f64((bp) + rsb + 2);        \
    b2 = vld1q_f64((bp) + rsb + 4);        \
    b3 = vld1q_f64((bp) + rsb + 6);        \
  } while (0)

static inline void gemm_ukernel_f64_6x8(const double *a, const double *b,
                                        double *c, size_t kc, intptr_t rsa,
                                        intptr_t csa, intptr_t rsb,
                                        intptr_t rso, int first) {
  float64x2_t c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30,
      c31, c32, c33, c40, c41, c42, c43, c50, c51, c52, c53;

  __builtin_prefetch(c, 1, 3);
  __builtin_prefetch(c + rso, 1, 3);
  __builtin_prefetch(c + 2 * rso, 1, 3);
  __builtin_prefetch(c + 3 * rso, 1, 3);
  __builtin_prefetch(c + 4 * rso, 1, 3);
  __builtin_prefetch(c + 5 * rso, 1, 3);

  if (first) {
    c00 = vdupq_n_f64(0);
    c01 = vdupq_n_f64(0);
    c02 = vdupq_n_f64(0);
    c03 = vdupq_n_f64(0);
    c10 = vdupq_n_f64(0);
    c11 = vdupq_n_f64(0);
    c12 = vdupq_n_f64(0);
    c13 = vdupq_n_f64(0);
    c20 = vdupq_n_f64(0);
    c21 = vdupq_n_f64(0);
    c22 = vdupq_n_f64(0);
    c23 = vdupq_n_f64(0);
    c30 = vdupq_n_f64(0);
    c31 = vdupq_n_f64(0);
    c32 = vdupq_n_f64(0);
    c33 = vdupq_n_f64(0);
    c40 = vdupq_n_f64(0);
    c41 = vdupq_n_f64(0);
    c42 = vdupq_n_f64(0);
    c43 = vdupq_n_f64(0);
    c50 = vdupq_n_f64(0);
    c51 = vdupq_n_f64(0);
    c52 = vdupq_n_f64(0);
    c53 = vdupq_n_f64(0);
  } else {
    c00 = vld1q_f64(c);
    c01 = vld1q_f64(c + 2);
    c02 = vld1q_f64(c + 4);
    c03 = vld1q_f64(c + 6);
    c10 = vld1q_f64(c + rso);
    c11 = vld1q_f64(c + rso + 2);
    c12 = vld1q_f64(c + rso + 4);
    c13 = vld1q_f64(c + rso + 6);
    c20 = vld1q_f64(c + 2 * rso);
    c21 = vld1q_f64(c + 2 * rso + 2);
    c22 = vld1q_f64(c + 2 * rso + 4);
    c23 = vld1q_f64(c + 2 * rso + 6);
    c30 = vld1q_f64(c + 3 * rso);
    c31 = vld1q_f64(c + 3 * rso + 2);
    c32 = vld1q_f64(c + 3 * rso + 4);
    c33 = vld1q_f64(c + 3 * rso + 6);
    c40 = vld1q_f64(c + 4 * rso);
    c41 = vld1q_f64(c + 4 * rso + 2);
    c42 = vld1q_f64(c + 4 * rso + 4);
    c43 = vld1q_f64(c + 4 * rso + 6);
    c50 = vld1q_f64(c + 5 * rso);
    c51 = vld1q_f64(c + 5 * rso + 2);
    c52 = vld1q_f64(c + 5 * rso + 4);
    c53 = vld1q_f64(c + 5 * rso + 6);
  }

  const double *ap = a;
  const double *bp = b;
  size_t k_iter = kc / 8;
  size_t k_left = kc % 8;

  /* Pre-load first B tile (BLIS-style pipelining) */
  float64x2_t b0 = vld1q_f64(bp);
  float64x2_t b1 = vld1q_f64(bp + 2);
  float64x2_t b2 = vld1q_f64(bp + 4);
  float64x2_t b3 = vld1q_f64(bp + 6);

  for (size_t ki = 0; ki < k_iter; ki++) {
    GEMM_F64_K_ITER(ap, bp);
    ap += csa;
    bp += rsb;
    GEMM_F64_K_ITER(ap, bp);
    ap += csa;
    bp += rsb;
    __builtin_prefetch(ap + 96, 0, 3);
    GEMM_F64_K_ITER(ap, bp);
    ap += csa;
    bp += rsb;
    GEMM_F64_K_ITER(ap, bp);
    ap += csa;
    bp += rsb;
    __builtin_prefetch(bp + 128, 0, 3);
    GEMM_F64_K_ITER(ap, bp);
    ap += csa;
    bp += rsb;
    GEMM_F64_K_ITER(ap, bp);
    ap += csa;
    bp += rsb;
    __builtin_prefetch(ap + 192, 0, 3);
    GEMM_F64_K_ITER(ap, bp);
    ap += csa;
    bp += rsb;
    GEMM_F64_K_ITER(ap, bp);
    ap += csa;
    bp += rsb;
  }
  for (size_t ki = 0; ki < k_left; ki++) {
    GEMM_F64_K_ITER(ap, bp);
    ap += csa;
    bp += rsb;
  }

  /* C prefetch refresh before store */
  __builtin_prefetch(c, 1, 3);
  __builtin_prefetch(c + rso, 1, 3);
  __builtin_prefetch(c + 2 * rso, 1, 3);
  __builtin_prefetch(c + 3 * rso, 1, 3);
  __builtin_prefetch(c + 4 * rso, 1, 3);
  __builtin_prefetch(c + 5 * rso, 1, 3);

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

#undef GEMM_F64_K_ITER

static inline void gemm_edge_f64_neon(const double *a, const double *b,
                                      double *c, size_t mr, size_t nr,
                                      size_t kc, intptr_t rsa, intptr_t csa,
                                      intptr_t rsb, intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
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

static inline void gemm_f64_neon(const double *a, const double *b, double *out,
                                 size_t m_dim, size_t k_dim, size_t n_dim,
                                 intptr_t rsa, intptr_t csa, intptr_t rsb,
                                 intptr_t rso) {
  size_t nc_max = GEMM_MIN(GEMM_F64_NC, n_dim);
  double *packed_b = (double *)numc_malloc(
      16, GEMM_F64_KC * (nc_max + GEMM_F64_NR) * sizeof(double));
  if (!packed_b)
    return;

  for (size_t jc = 0; jc < n_dim; jc += GEMM_F64_NC) {
    size_t nc = GEMM_MIN(GEMM_F64_NC, n_dim - jc);
    size_t n_jr = (nc + GEMM_F64_NR - 1) / GEMM_F64_NR;

#ifdef _OPENMP
#pragma omp parallel if ((uint64_t)m_dim * k_dim * nc > GEMM_OMP_THRESHOLD)
    {
      NUMC_ALIGNAS(16) double packed_a[GEMM_F64_MC * GEMM_F64_KC];

      for (size_t pc = 0; pc < k_dim; pc += GEMM_F64_KC) {
        size_t kc = GEMM_MIN(GEMM_F64_KC, k_dim - pc);
        int first = (pc == 0);
        size_t last_ic = (size_t)-1;

#pragma omp for schedule(static)
        for (size_t jr_idx = 0; jr_idx < n_jr; jr_idx++) {
          size_t jj = jr_idx * GEMM_F64_NR;
          size_t nr_pack = GEMM_MIN(GEMM_F64_NR, nc - jj);
          _gemm_pack_b_strip_f64(b + pc * rsb + jc + jj, packed_b + jj * kc, kc,
                                 nr_pack, rsb);
        }

        size_t n_ic = (m_dim + GEMM_F64_MC - 1) / GEMM_F64_MC;
        size_t n_tasks = n_ic * n_jr;

#pragma omp for schedule(static)
        for (size_t task = 0; task < n_tasks; task++) {
          size_t ic = (task / n_jr) * GEMM_F64_MC;
          size_t jr = (task % n_jr) * GEMM_F64_NR;
          size_t mc = GEMM_MIN(GEMM_F64_MC, m_dim - ic);
          size_t nr_cur = GEMM_MIN(GEMM_F64_NR, nc - jr);

          if (ic != last_ic) {
            gemm_pack_a_f64(a + ic * rsa + pc * csa, packed_a, mc, kc, rsa,
                            csa);
            last_ic = ic;
          }

          for (size_t ir = 0; ir < mc; ir += GEMM_F64_MR) {
            size_t mr_cur = GEMM_MIN(GEMM_F64_MR, mc - ir);
            if (mr_cur == GEMM_F64_MR && nr_cur == GEMM_F64_NR) {
              gemm_ukernel_f64_6x8(packed_a + ir * kc, packed_b + jr * kc,
                                   out + (ic + ir) * rso + (jc + jr), kc, 1,
                                   GEMM_F64_MR, GEMM_F64_NR, rso, first);
            } else {
              NUMC_ALIGNAS(16) double tmp[GEMM_F64_MR * GEMM_F64_NR];
              gemm_ukernel_f64_6x8(packed_a + ir * kc, packed_b + jr * kc, tmp,
                                   kc, 1, GEMM_F64_MR, GEMM_F64_NR, GEMM_F64_NR,
                                   1);
              double *dst = out + (ic + ir) * rso + (jc + jr);
              if (first) {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] = tmp[ii * GEMM_F64_NR + jj];
              } else {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] += tmp[ii * GEMM_F64_NR + jj];
              }
            }
          }
        }
      }
    }
#else
    {
      NUMC_ALIGNAS(16) double packed_a[GEMM_F64_MC * GEMM_F64_KC];

      for (size_t pc = 0; pc < k_dim; pc += GEMM_F64_KC) {
        size_t kc = GEMM_MIN(GEMM_F64_KC, k_dim - pc);
        int first = (pc == 0);

        gemm_pack_b_f64(b + pc * rsb + jc, packed_b, kc, nc, rsb);

        size_t n_ic = (m_dim + GEMM_F64_MC - 1) / GEMM_F64_MC;
        size_t n_tasks = n_ic * n_jr;

        for (size_t task = 0; task < n_tasks; task++) {
          size_t ic = (task / n_jr) * GEMM_F64_MC;
          size_t jr = (task % n_jr) * GEMM_F64_NR;
          size_t mc = GEMM_MIN(GEMM_F64_MC, m_dim - ic);
          size_t nr_cur = GEMM_MIN(GEMM_F64_NR, nc - jr);

          if (task % n_jr == 0)
            gemm_pack_a_f64(a + ic * rsa + pc * csa, packed_a, mc, kc, rsa,
                            csa);

          for (size_t ir = 0; ir < mc; ir += GEMM_F64_MR) {
            size_t mr_cur = GEMM_MIN(GEMM_F64_MR, mc - ir);
            if (mr_cur == GEMM_F64_MR && nr_cur == GEMM_F64_NR) {
              gemm_ukernel_f64_6x8(packed_a + ir * kc, packed_b + jr * kc,
                                   out + (ic + ir) * rso + (jc + jr), kc, 1,
                                   GEMM_F64_MR, GEMM_F64_NR, rso, first);
            } else {
              NUMC_ALIGNAS(16) double tmp[GEMM_F64_MR * GEMM_F64_NR];
              gemm_ukernel_f64_6x8(packed_a + ir * kc, packed_b + jr * kc, tmp,
                                   kc, 1, GEMM_F64_MR, GEMM_F64_NR, GEMM_F64_NR,
                                   1);
              double *dst = out + (ic + ir) * rso + (jc + jr);
              if (first) {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] = tmp[ii * GEMM_F64_NR + jj];
              } else {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] += tmp[ii * GEMM_F64_NR + jj];
              }
            }
          }
        }
      }
    }
#endif
  }

  numc_free(packed_b);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Int32/Uint32: 6x8 unpacked micro-kernel (vmlaq_s32)
   ═══════════════════════════════════════════════════════════════════════════
 */

static inline void gemm_ukernel_i32_6x8(const int32_t *a, const int32_t *b,
                                        int32_t *c, size_t kc, intptr_t rsa,
                                        intptr_t csa, intptr_t rsb,
                                        intptr_t rso) {
  int32x4_t c00 = vld1q_s32(c), c01 = vld1q_s32(c + 4);
  int32x4_t c10 = vld1q_s32(c + rso), c11 = vld1q_s32(c + rso + 4);
  int32x4_t c20 = vld1q_s32(c + 2 * rso), c21 = vld1q_s32(c + 2 * rso + 4);
  int32x4_t c30 = vld1q_s32(c + 3 * rso), c31 = vld1q_s32(c + 3 * rso + 4);
  int32x4_t c40 = vld1q_s32(c + 4 * rso), c41 = vld1q_s32(c + 4 * rso + 4);
  int32x4_t c50 = vld1q_s32(c + 5 * rso), c51 = vld1q_s32(c + 5 * rso + 4);

  for (size_t p = 0; p < kc; p++) {
    const int32_t *bp = b + p * rsb;
    int32x4_t b0 = vld1q_s32(bp), b1 = vld1q_s32(bp + 4);
    int32x4_t av;
    av = vdupq_n_s32(a[0 * rsa + p * csa]);
    c00 = vmlaq_s32(c00, av, b0);
    c01 = vmlaq_s32(c01, av, b1);
    av = vdupq_n_s32(a[1 * rsa + p * csa]);
    c10 = vmlaq_s32(c10, av, b0);
    c11 = vmlaq_s32(c11, av, b1);
    av = vdupq_n_s32(a[2 * rsa + p * csa]);
    c20 = vmlaq_s32(c20, av, b0);
    c21 = vmlaq_s32(c21, av, b1);
    av = vdupq_n_s32(a[3 * rsa + p * csa]);
    c30 = vmlaq_s32(c30, av, b0);
    c31 = vmlaq_s32(c31, av, b1);
    av = vdupq_n_s32(a[4 * rsa + p * csa]);
    c40 = vmlaq_s32(c40, av, b0);
    c41 = vmlaq_s32(c41, av, b1);
    av = vdupq_n_s32(a[5 * rsa + p * csa]);
    c50 = vmlaq_s32(c50, av, b0);
    c51 = vmlaq_s32(c51, av, b1);
  }

  vst1q_s32(c, c00);
  vst1q_s32(c + 4, c01);
  vst1q_s32(c + rso, c10);
  vst1q_s32(c + rso + 4, c11);
  vst1q_s32(c + 2 * rso, c20);
  vst1q_s32(c + 2 * rso + 4, c21);
  vst1q_s32(c + 3 * rso, c30);
  vst1q_s32(c + 3 * rso + 4, c31);
  vst1q_s32(c + 4 * rso, c40);
  vst1q_s32(c + 4 * rso + 4, c41);
  vst1q_s32(c + 5 * rso, c50);
  vst1q_s32(c + 5 * rso + 4, c51);
}

static inline void gemm_edge_i32_neon(const int32_t *a, const int32_t *b,
                                      int32_t *c, size_t mr, size_t nr,
                                      size_t kc, intptr_t rsa, intptr_t csa,
                                      intptr_t rsb, intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
    for (size_t p = 0; p < kc; p++) {
      int32_t aip = a[i * rsa + p * csa];
      int32x4_t va = vdupq_n_s32(aip);
      const int32_t *brow = b + p * rsb;
      int32_t *crow = c + i * rso;
      size_t j = 0;
      for (; j + 4 <= nr; j += 4) {
        int32x4_t vo = vld1q_s32(crow + j);
        vo = vmlaq_s32(vo, va, vld1q_s32(brow + j));
        vst1q_s32(crow + j, vo);
      }
      for (; j < nr; j++)
        crow[j] += aip * brow[j];
    }
  }
}

static inline void gemm_i32_neon(const int32_t *a, const int32_t *b,
                                 int32_t *out, size_t m_dim, size_t k_dim,
                                 size_t n_dim, intptr_t rsa, intptr_t csa,
                                 intptr_t rsb, intptr_t rso) {
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
      for (; jr + GEMM_I32_NR <= n_dim; jr += GEMM_I32_NR) {
        size_t ir = 0;
        for (; ir + GEMM_I32_MR <= mc; ir += GEMM_I32_MR)
          gemm_ukernel_i32_6x8(a + (ic + ir) * rsa + pc * csa,
                               b + pc * rsb + jr, out + (ic + ir) * rso + jr,
                               kc, rsa, csa, rsb, rso);
        if (ir < mc)
          gemm_edge_i32_neon(a + (ic + ir) * rsa + pc * csa, b + pc * rsb + jr,
                             out + (ic + ir) * rso + jr, mc - ir, GEMM_I32_NR,
                             kc, rsa, csa, rsb, rso);
      }
      if (jr < n_dim)
        gemm_edge_i32_neon(a + ic * rsa + pc * csa, b + pc * rsb + jr,
                           out + ic * rso + jr, mc, n_dim - jr, kc, rsa, csa,
                           rsb, rso);
    }
  }
}

static inline void gemm_u32_neon(const uint32_t *a, const uint32_t *b,
                                 uint32_t *out, size_t m_dim, size_t k_dim,
                                 size_t n_dim, intptr_t rsa, intptr_t csa,
                                 intptr_t rsb, intptr_t rso) {
  gemm_i32_neon((const int32_t *)a, (const int32_t *)b, (int32_t *)out, m_dim,
                k_dim, n_dim, rsa, csa, rsb, rso);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Int16/Uint16: 6x16 unpacked micro-kernel (vmlaq_s16)
   ═══════════════════════════════════════════════════════════════════════════
 */

static inline void gemm_ukernel_i16_6x16(const int16_t *a, const int16_t *b,
                                         int16_t *c, size_t kc, intptr_t rsa,
                                         intptr_t csa, intptr_t rsb,
                                         intptr_t rso) {
  int16x8_t c00 = vld1q_s16(c), c01 = vld1q_s16(c + 8);
  int16x8_t c10 = vld1q_s16(c + rso), c11 = vld1q_s16(c + rso + 8);
  int16x8_t c20 = vld1q_s16(c + 2 * rso), c21 = vld1q_s16(c + 2 * rso + 8);
  int16x8_t c30 = vld1q_s16(c + 3 * rso), c31 = vld1q_s16(c + 3 * rso + 8);
  int16x8_t c40 = vld1q_s16(c + 4 * rso), c41 = vld1q_s16(c + 4 * rso + 8);
  int16x8_t c50 = vld1q_s16(c + 5 * rso), c51 = vld1q_s16(c + 5 * rso + 8);

  for (size_t p = 0; p < kc; p++) {
    const int16_t *bp = b + p * rsb;
    int16x8_t b0 = vld1q_s16(bp), b1 = vld1q_s16(bp + 8);
    int16x8_t av;
    av = vdupq_n_s16(a[0 * rsa + p * csa]);
    c00 = vmlaq_s16(c00, av, b0);
    c01 = vmlaq_s16(c01, av, b1);
    av = vdupq_n_s16(a[1 * rsa + p * csa]);
    c10 = vmlaq_s16(c10, av, b0);
    c11 = vmlaq_s16(c11, av, b1);
    av = vdupq_n_s16(a[2 * rsa + p * csa]);
    c20 = vmlaq_s16(c20, av, b0);
    c21 = vmlaq_s16(c21, av, b1);
    av = vdupq_n_s16(a[3 * rsa + p * csa]);
    c30 = vmlaq_s16(c30, av, b0);
    c31 = vmlaq_s16(c31, av, b1);
    av = vdupq_n_s16(a[4 * rsa + p * csa]);
    c40 = vmlaq_s16(c40, av, b0);
    c41 = vmlaq_s16(c41, av, b1);
    av = vdupq_n_s16(a[5 * rsa + p * csa]);
    c50 = vmlaq_s16(c50, av, b0);
    c51 = vmlaq_s16(c51, av, b1);
  }

  vst1q_s16(c, c00);
  vst1q_s16(c + 8, c01);
  vst1q_s16(c + rso, c10);
  vst1q_s16(c + rso + 8, c11);
  vst1q_s16(c + 2 * rso, c20);
  vst1q_s16(c + 2 * rso + 8, c21);
  vst1q_s16(c + 3 * rso, c30);
  vst1q_s16(c + 3 * rso + 8, c31);
  vst1q_s16(c + 4 * rso, c40);
  vst1q_s16(c + 4 * rso + 8, c41);
  vst1q_s16(c + 5 * rso, c50);
  vst1q_s16(c + 5 * rso + 8, c51);
}

static inline void gemm_edge_i16_neon(const int16_t *a, const int16_t *b,
                                      int16_t *c, size_t mr, size_t nr,
                                      size_t kc, intptr_t rsa, intptr_t csa,
                                      intptr_t rsb, intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
    for (size_t p = 0; p < kc; p++) {
      int16_t aip = a[i * rsa + p * csa];
      int16x8_t va = vdupq_n_s16(aip);
      const int16_t *brow = b + p * rsb;
      int16_t *crow = c + i * rso;
      size_t j = 0;
      for (; j + 8 <= nr; j += 8) {
        int16x8_t vo = vld1q_s16(crow + j);
        vo = vmlaq_s16(vo, va, vld1q_s16(brow + j));
        vst1q_s16(crow + j, vo);
      }
      for (; j < nr; j++)
        crow[j] = (int16_t)(crow[j] + aip * brow[j]);
    }
  }
}

static inline void gemm_i16_neon(const int16_t *a, const int16_t *b,
                                 int16_t *out, size_t m_dim, size_t k_dim,
                                 size_t n_dim, intptr_t rsa, intptr_t csa,
                                 intptr_t rsb, intptr_t rso) {
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
      for (; jr + GEMM_I16_NR <= n_dim; jr += GEMM_I16_NR) {
        size_t ir = 0;
        for (; ir + GEMM_I16_MR <= mc; ir += GEMM_I16_MR)
          gemm_ukernel_i16_6x16(a + (ic + ir) * rsa + pc * csa,
                                b + pc * rsb + jr, out + (ic + ir) * rso + jr,
                                kc, rsa, csa, rsb, rso);
        if (ir < mc)
          gemm_edge_i16_neon(a + (ic + ir) * rsa + pc * csa, b + pc * rsb + jr,
                             out + (ic + ir) * rso + jr, mc - ir, GEMM_I16_NR,
                             kc, rsa, csa, rsb, rso);
      }
      if (jr < n_dim)
        gemm_edge_i16_neon(a + ic * rsa + pc * csa, b + pc * rsb + jr,
                           out + ic * rso + jr, mc, n_dim - jr, kc, rsa, csa,
                           rsb, rso);
    }
  }
}

static inline void gemm_u16_neon(const uint16_t *a, const uint16_t *b,
                                 uint16_t *out, size_t m_dim, size_t k_dim,
                                 size_t n_dim, intptr_t rsa, intptr_t csa,
                                 intptr_t rsb, intptr_t rso) {
  gemm_i16_neon((const int16_t *)a, (const int16_t *)b, (int16_t *)out, m_dim,
                k_dim, n_dim, rsa, csa, rsb, rso);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Int64/Uint64: 6x4 unpacked (no native 64-bit multiply, scalar emulation)
   ═══════════════════════════════════════════════════════════════════════════
 */

static inline void gemm_ukernel_i64_6x4(const int64_t *a, const int64_t *b,
                                        int64_t *c, size_t kc, intptr_t rsa,
                                        intptr_t csa, intptr_t rsb,
                                        intptr_t rso) {
  /* NEON has no 64-bit multiply — use scalar with i64x2 add */
  int64x2_t c00 = vld1q_s64(c), c01 = vld1q_s64(c + 2);
  int64x2_t c10 = vld1q_s64(c + rso), c11 = vld1q_s64(c + rso + 2);
  int64x2_t c20 = vld1q_s64(c + 2 * rso), c21 = vld1q_s64(c + 2 * rso + 2);
  int64x2_t c30 = vld1q_s64(c + 3 * rso), c31 = vld1q_s64(c + 3 * rso + 2);
  int64x2_t c40 = vld1q_s64(c + 4 * rso), c41 = vld1q_s64(c + 4 * rso + 2);
  int64x2_t c50 = vld1q_s64(c + 5 * rso), c51 = vld1q_s64(c + 5 * rso + 2);

  for (size_t p = 0; p < kc; p++) {
    const int64_t *bp = b + p * rsb;
    int64x2_t b0 = vld1q_s64(bp), b1 = vld1q_s64(bp + 2);

#define NEON_I64_ROW(row, cx0, cx1)                                 \
  do {                                                              \
    int64_t aval = a[(row) * rsa + p * csa];                        \
    int64x2_t av = vdupq_n_s64(aval);                               \
    /* Emulate: acc += a * b via lane extract */                    \
    int64_t p0 = aval * vgetq_lane_s64(b0, 0);                      \
    int64_t p1 = aval * vgetq_lane_s64(b0, 1);                      \
    int64_t p2 = aval * vgetq_lane_s64(b1, 0);                      \
    int64_t p3 = aval * vgetq_lane_s64(b1, 1);                      \
    int64x2_t prod0 = vcombine_s64(vdup_n_s64(p0), vdup_n_s64(p1)); \
    int64x2_t prod1 = vcombine_s64(vdup_n_s64(p2), vdup_n_s64(p3)); \
    cx0 = vaddq_s64(cx0, prod0);                                    \
    cx1 = vaddq_s64(cx1, prod1);                                    \
    (void)av;                                                       \
  } while (0)

    NEON_I64_ROW(0, c00, c01);
    NEON_I64_ROW(1, c10, c11);
    NEON_I64_ROW(2, c20, c21);
    NEON_I64_ROW(3, c30, c31);
    NEON_I64_ROW(4, c40, c41);
    NEON_I64_ROW(5, c50, c51);
#undef NEON_I64_ROW
  }

  vst1q_s64(c, c00);
  vst1q_s64(c + 2, c01);
  vst1q_s64(c + rso, c10);
  vst1q_s64(c + rso + 2, c11);
  vst1q_s64(c + 2 * rso, c20);
  vst1q_s64(c + 2 * rso + 2, c21);
  vst1q_s64(c + 3 * rso, c30);
  vst1q_s64(c + 3 * rso + 2, c31);
  vst1q_s64(c + 4 * rso, c40);
  vst1q_s64(c + 4 * rso + 2, c41);
  vst1q_s64(c + 5 * rso, c50);
  vst1q_s64(c + 5 * rso + 2, c51);
}

static inline void gemm_edge_i64_neon(const int64_t *a, const int64_t *b,
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

static inline void gemm_i64_neon(const int64_t *a, const int64_t *b,
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
      size_t jr = 0;
      for (; jr + GEMM_I64_NR <= n_dim; jr += GEMM_I64_NR) {
        size_t ir = 0;
        for (; ir + GEMM_I64_MR <= mc; ir += GEMM_I64_MR)
          gemm_ukernel_i64_6x4(a + (ic + ir) * rsa + pc * csa,
                               b + pc * rsb + jr, out + (ic + ir) * rso + jr,
                               kc, rsa, csa, rsb, rso);
        if (ir < mc)
          gemm_edge_i64_neon(a + (ic + ir) * rsa + pc * csa, b + pc * rsb + jr,
                             out + (ic + ir) * rso + jr, mc - ir, GEMM_I64_NR,
                             kc, rsa, csa, rsb, rso);
      }
      if (jr < n_dim)
        gemm_edge_i64_neon(a + ic * rsa + pc * csa, b + pc * rsb + jr,
                           out + ic * rso + jr, mc, n_dim - jr, kc, rsa, csa,
                           rsb, rso);
    }
  }
}

static inline void gemm_u64_neon(const uint64_t *a, const uint64_t *b,
                                 uint64_t *out, size_t m_dim, size_t k_dim,
                                 size_t n_dim, intptr_t rsa, intptr_t csa,
                                 intptr_t rsb, intptr_t rso) {
  gemm_i64_neon((const int64_t *)a, (const int64_t *)b, (int64_t *)out, m_dim,
                k_dim, n_dim, rsa, csa, rsb, rso);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Int8/Uint8: 6x8 promoted micro-kernel (widen to i32, full-K accumulation)
   ═══════════════════════════════════════════════════════════════════════════
 */

static inline void gemm_ukernel_i8_6x8(const int8_t *a, const int8_t *b,
                                       int8_t *c, size_t k_dim, intptr_t rsa,
                                       intptr_t csa, intptr_t rsb,
                                       intptr_t rso) {
  int32x4_t c00 = vdupq_n_s32(0), c01 = vdupq_n_s32(0);
  int32x4_t c10 = vdupq_n_s32(0), c11 = vdupq_n_s32(0);
  int32x4_t c20 = vdupq_n_s32(0), c21 = vdupq_n_s32(0);
  int32x4_t c30 = vdupq_n_s32(0), c31 = vdupq_n_s32(0);
  int32x4_t c40 = vdupq_n_s32(0), c41 = vdupq_n_s32(0);
  int32x4_t c50 = vdupq_n_s32(0), c51 = vdupq_n_s32(0);

  for (size_t p = 0; p < k_dim; p++) {
    const int8_t *bp = b + p * rsb;
    /* Load 8 i8, widen to i16, then to i32 */
    int8x8_t bv = vld1_s8(bp);
    int16x8_t bw = vmovl_s8(bv);
    int32x4_t b0 = vmovl_s16(vget_low_s16(bw));
    int32x4_t b1 = vmovl_s16(vget_high_s16(bw));

    int32x4_t av;
    av = vdupq_n_s32((int32_t)a[0 * rsa + p * csa]);
    c00 = vmlaq_s32(c00, av, b0);
    c01 = vmlaq_s32(c01, av, b1);
    av = vdupq_n_s32((int32_t)a[1 * rsa + p * csa]);
    c10 = vmlaq_s32(c10, av, b0);
    c11 = vmlaq_s32(c11, av, b1);
    av = vdupq_n_s32((int32_t)a[2 * rsa + p * csa]);
    c20 = vmlaq_s32(c20, av, b0);
    c21 = vmlaq_s32(c21, av, b1);
    av = vdupq_n_s32((int32_t)a[3 * rsa + p * csa]);
    c30 = vmlaq_s32(c30, av, b0);
    c31 = vmlaq_s32(c31, av, b1);
    av = vdupq_n_s32((int32_t)a[4 * rsa + p * csa]);
    c40 = vmlaq_s32(c40, av, b0);
    c41 = vmlaq_s32(c41, av, b1);
    av = vdupq_n_s32((int32_t)a[5 * rsa + p * csa]);
    c50 = vmlaq_s32(c50, av, b0);
    c51 = vmlaq_s32(c51, av, b1);
  }

  /* Narrow i32 -> i16 -> i8 with saturation, store 8 bytes per row */
#define NEON_STORE_I8_ROW(cx0, cx1, row)  \
  do {                                    \
    int16x4_t lo = vqmovn_s32(cx0);       \
    int16x4_t hi = vqmovn_s32(cx1);       \
    int16x8_t p16 = vcombine_s16(lo, hi); \
    int8x8_t p8 = vqmovn_s16(p16);        \
    vst1_s8(c + (row) * rso, p8);         \
  } while (0)
  NEON_STORE_I8_ROW(c00, c01, 0);
  NEON_STORE_I8_ROW(c10, c11, 1);
  NEON_STORE_I8_ROW(c20, c21, 2);
  NEON_STORE_I8_ROW(c30, c31, 3);
  NEON_STORE_I8_ROW(c40, c41, 4);
  NEON_STORE_I8_ROW(c50, c51, 5);
#undef NEON_STORE_I8_ROW
}

static inline void gemm_edge_i8_neon(const int8_t *a, const int8_t *b,
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

static inline void gemm_i8_neon(const int8_t *a, const int8_t *b, int8_t *out,
                                size_t m_dim, size_t k_dim, size_t n_dim,
                                intptr_t rsa, intptr_t csa, intptr_t rsb,
                                intptr_t rso) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (m_dim * n_dim > \
                                                  GEMM_OMP_THRESHOLD)
#endif
  for (size_t ic = 0; ic < m_dim; ic += GEMM_I8_MC) {
    size_t mc = GEMM_MIN(GEMM_I8_MC, m_dim - ic);
    size_t jr = 0;
    for (; jr + GEMM_I8_NR <= n_dim; jr += GEMM_I8_NR) {
      size_t ir = 0;
      for (; ir + GEMM_I8_MR <= mc; ir += GEMM_I8_MR)
        gemm_ukernel_i8_6x8(a + (ic + ir) * rsa, b + jr,
                            out + (ic + ir) * rso + jr, k_dim, rsa, csa, rsb,
                            rso);
      if (ir < mc)
        gemm_edge_i8_neon(a + (ic + ir) * rsa, b + jr,
                          out + (ic + ir) * rso + jr, mc - ir, GEMM_I8_NR,
                          k_dim, rsa, csa, rsb, rso);
    }
    if (jr < n_dim)
      gemm_edge_i8_neon(a + ic * rsa, b + jr, out + ic * rso + jr, mc,
                        n_dim - jr, k_dim, rsa, csa, rsb, rso);
  }
}

static inline void gemm_ukernel_u8_6x8(const uint8_t *a, const uint8_t *b,
                                       uint8_t *c, size_t k_dim, intptr_t rsa,
                                       intptr_t csa, intptr_t rsb,
                                       intptr_t rso) {
  uint32x4_t c00 = vdupq_n_u32(0), c01 = vdupq_n_u32(0);
  uint32x4_t c10 = vdupq_n_u32(0), c11 = vdupq_n_u32(0);
  uint32x4_t c20 = vdupq_n_u32(0), c21 = vdupq_n_u32(0);
  uint32x4_t c30 = vdupq_n_u32(0), c31 = vdupq_n_u32(0);
  uint32x4_t c40 = vdupq_n_u32(0), c41 = vdupq_n_u32(0);
  uint32x4_t c50 = vdupq_n_u32(0), c51 = vdupq_n_u32(0);

  for (size_t p = 0; p < k_dim; p++) {
    const uint8_t *bp = b + p * rsb;
    uint8x8_t bv = vld1_u8(bp);
    uint16x8_t bw = vmovl_u8(bv);
    uint32x4_t b0 = vmovl_u16(vget_low_u16(bw));
    uint32x4_t b1 = vmovl_u16(vget_high_u16(bw));

    uint32x4_t av;
    av = vdupq_n_u32((uint32_t)a[0 * rsa + p * csa]);
    c00 = vmlaq_u32(c00, av, b0);
    c01 = vmlaq_u32(c01, av, b1);
    av = vdupq_n_u32((uint32_t)a[1 * rsa + p * csa]);
    c10 = vmlaq_u32(c10, av, b0);
    c11 = vmlaq_u32(c11, av, b1);
    av = vdupq_n_u32((uint32_t)a[2 * rsa + p * csa]);
    c20 = vmlaq_u32(c20, av, b0);
    c21 = vmlaq_u32(c21, av, b1);
    av = vdupq_n_u32((uint32_t)a[3 * rsa + p * csa]);
    c30 = vmlaq_u32(c30, av, b0);
    c31 = vmlaq_u32(c31, av, b1);
    av = vdupq_n_u32((uint32_t)a[4 * rsa + p * csa]);
    c40 = vmlaq_u32(c40, av, b0);
    c41 = vmlaq_u32(c41, av, b1);
    av = vdupq_n_u32((uint32_t)a[5 * rsa + p * csa]);
    c50 = vmlaq_u32(c50, av, b0);
    c51 = vmlaq_u32(c51, av, b1);
  }

#define NEON_STORE_U8_ROW(cx0, cx1, row)   \
  do {                                     \
    uint16x4_t lo = vqmovn_u32(cx0);       \
    uint16x4_t hi = vqmovn_u32(cx1);       \
    uint16x8_t p16 = vcombine_u16(lo, hi); \
    uint8x8_t p8 = vqmovn_u16(p16);        \
    vst1_u8(c + (row) * rso, p8);          \
  } while (0)
  NEON_STORE_U8_ROW(c00, c01, 0);
  NEON_STORE_U8_ROW(c10, c11, 1);
  NEON_STORE_U8_ROW(c20, c21, 2);
  NEON_STORE_U8_ROW(c30, c31, 3);
  NEON_STORE_U8_ROW(c40, c41, 4);
  NEON_STORE_U8_ROW(c50, c51, 5);
#undef NEON_STORE_U8_ROW
}

static inline void gemm_edge_u8_neon(const uint8_t *a, const uint8_t *b,
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

static inline void gemm_u8_neon(const uint8_t *a, const uint8_t *b,
                                uint8_t *out, size_t m_dim, size_t k_dim,
                                size_t n_dim, intptr_t rsa, intptr_t csa,
                                intptr_t rsb, intptr_t rso) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (m_dim * n_dim > \
                                                  GEMM_OMP_THRESHOLD)
#endif
  for (size_t ic = 0; ic < m_dim; ic += GEMM_I8_MC) {
    size_t mc = GEMM_MIN(GEMM_I8_MC, m_dim - ic);
    size_t jr = 0;
    for (; jr + GEMM_I8_NR <= n_dim; jr += GEMM_I8_NR) {
      size_t ir = 0;
      for (; ir + GEMM_I8_MR <= mc; ir += GEMM_I8_MR)
        gemm_ukernel_u8_6x8(a + (ic + ir) * rsa, b + jr,
                            out + (ic + ir) * rso + jr, k_dim, rsa, csa, rsb,
                            rso);
      if (ir < mc)
        gemm_edge_u8_neon(a + (ic + ir) * rsa, b + jr,
                          out + (ic + ir) * rso + jr, mc - ir, GEMM_I8_NR,
                          k_dim, rsa, csa, rsb, rso);
    }
    if (jr < n_dim)
      gemm_edge_u8_neon(a + ic * rsa, b + jr, out + ic * rso + jr, mc,
                        n_dim - jr, k_dim, rsa, csa, rsb, rso);
  }
}

#undef GEMM_MIN

#endif
