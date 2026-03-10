#ifndef NUMC_GEMM_AVX512_H
#define NUMC_GEMM_AVX512_H

#include <immintrin.h>
#include <stdint.h>
#include <string.h>

#define GEMM_MIN(a, b) ((a) < (b) ? (a) : (b))

/*
 * Cache-blocking parameters for AVX-512 packed GEMM (BLIS SKX-derived,
 * adapted to row-major micro-kernels).
 *   - MC x KC panel of A resides in L2
 *   - KC x NR sliver of B resides in L1
 *   - KC x NC panel of B resides in LLC
 * f32 micro-kernel computes 12x32 tiles (row-major equivalent of BLIS 32x12).
 * f64 micro-kernel computes 14x16 tiles (row-major equivalent of BLIS 16x14).
 * Thread utilization via 2D IC x JR parallelism.
 */
#define GEMM_F32_MR 12
#define GEMM_F32_NR 32
#define GEMM_F32_MC 480
#define GEMM_F32_KC 384

#define GEMM_F64_MR 14
#define GEMM_F64_NR 16
#define GEMM_F64_MC 240
#define GEMM_F64_KC 256

/* GEMM OMP threshold on compute volume (M x K x N operations). */
#define GEMM_OMP_THRESHOLD (1 << 23)

/* N-dimension blocking for L3 residency */
#define GEMM_F32_NC 3072
#define GEMM_F64_NC 3752

/* ── Float32 packing routines ──────────────────────────────────────────── */

static inline void gemm_pack_b_f32(const float *b, float *packed, size_t kc,
                                   size_t nc, intptr_t rsb) {
  size_t jr = 0;
  for (; jr + GEMM_F32_NR <= nc; jr += GEMM_F32_NR) {
    float *dest = packed + jr * kc;
    for (size_t p = 0; p < kc; p++) {
      const float *src = b + p * rsb + jr;
      _mm512_storeu_ps(dest + p * GEMM_F32_NR, _mm512_loadu_ps(src));
      _mm512_storeu_ps(dest + p * GEMM_F32_NR + 16, _mm512_loadu_ps(src + 16));
    }
  }
  if (jr < nc) {
    float *dest = packed + jr * kc;
    size_t rem = nc - jr;
    for (size_t p = 0; p < kc; p++) {
      const float *src = b + p * rsb + jr;
      size_t j = 0;
      for (; j < rem; j++)
        dest[p * GEMM_F32_NR + j] = src[j];
      for (; j < GEMM_F32_NR; j++)
        dest[p * GEMM_F32_NR + j] = 0.0f;
    }
  }
}

static inline void gemm_pack_a_f32(const float *a, float *packed, size_t mc,
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

static inline void gemm_pack_b_f64(const double *b, double *packed, size_t kc,
                                   size_t nc, intptr_t rsb) {
  size_t jr = 0;
  for (; jr + GEMM_F64_NR <= nc; jr += GEMM_F64_NR) {
    double *dest = packed + jr * kc;
    for (size_t p = 0; p < kc; p++) {
      const double *src = b + p * rsb + jr;
      _mm512_storeu_pd(dest + p * GEMM_F64_NR, _mm512_loadu_pd(src));
      _mm512_storeu_pd(dest + p * GEMM_F64_NR + 8, _mm512_loadu_pd(src + 8));
    }
  }
  if (jr < nc) {
    double *dest = packed + jr * kc;
    size_t rem = nc - jr;
    for (size_t p = 0; p < kc; p++) {
      const double *src = b + p * rsb + jr;
      size_t j = 0;
      for (; j < rem; j++)
        dest[p * GEMM_F64_NR + j] = src[j];
      for (; j < GEMM_F64_NR; j++)
        dest[p * GEMM_F64_NR + j] = 0.0;
    }
  }
}

static inline void gemm_pack_a_f64(const double *a, double *packed, size_t mc,
                                   size_t kc, intptr_t rsa, intptr_t csa) {
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
   Float32: 12x32 micro-kernel (24 acc + 1 A broadcast + 2 B loads = 27 ZMM)
   ═══════════════════════════════════════════════════════════════════════════
 */

/* One K-iteration of the 12x32 micro-kernel body.
 * ap points to 12 packed A values (MR stride), bp points to 32 packed B values.
 * 24 FMA instructions: 12 broadcasts x 2 ZMM B-vector halves. */
#define GEMM_F32_K_ITER(ap, bp)                   \
  do {                                            \
    __m512 b0 = _mm512_loadu_ps(bp);              \
    __m512 b1 = _mm512_loadu_ps((bp) + 16);       \
    for (size_t _i = 0; _i < GEMM_F32_MR; _i++) { \
      __m512 av = _mm512_set1_ps(*((ap) + _i));   \
      c0[_i] = _mm512_fmadd_ps(av, b0, c0[_i]);   \
      c1[_i] = _mm512_fmadd_ps(av, b1, c1[_i]);   \
    }                                             \
  } while (0)

static inline void gemm_ukernel_f32_12x32(const float *a, const float *b,
                                          float *c, size_t kc, intptr_t rsa,
                                          intptr_t csa, intptr_t rsb,
                                          intptr_t rso, int first) {
  __m512 c0[GEMM_F32_MR], c1[GEMM_F32_MR];

  for (size_t i = 0; i < GEMM_F32_MR; i++)
    _mm_prefetch((const char *)(c + i * rso), _MM_HINT_T0);

  if (first) {
    for (size_t i = 0; i < GEMM_F32_MR; i++) {
      c0[i] = _mm512_setzero_ps();
      c1[i] = _mm512_setzero_ps();
    }
  } else {
    for (size_t i = 0; i < GEMM_F32_MR; i++) {
      float *ci = c + i * rso;
      c0[i] = _mm512_loadu_ps(ci);
      c1[i] = _mm512_loadu_ps(ci + 16);
    }
  }

  /* Pointer-based iteration through packed A (stride MR=12) and B (stride
   * NR=32). 4x unrolled K-loop (BLIS pattern: k_iter = kc/4, k_left = kc%4). */
  const float *ap = a;
  const float *bp = b;
  size_t k_iter = kc / 4;
  size_t k_left = kc % 4;

  for (size_t ki = 0; ki < k_iter; ki++) {
    GEMM_F32_K_ITER(ap, bp);
    ap += csa;
    bp += rsb;
    GEMM_F32_K_ITER(ap, bp);
    ap += csa;
    bp += rsb;
    _mm_prefetch((const char *)(ap + 2 * GEMM_F32_MR), _MM_HINT_T0);
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

  for (size_t i = 0; i < GEMM_F32_MR; i++) {
    float *ci = c + i * rso;
    _mm512_storeu_ps(ci, c0[i]);
    _mm512_storeu_ps(ci + 16, c1[i]);
  }
}

#undef GEMM_F32_K_ITER

static inline void gemm_edge_f32_avx512(const float *a, const float *b,
                                        float *c, size_t mr, size_t nr,
                                        size_t kc, intptr_t rsa, intptr_t csa,
                                        intptr_t rsb, intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
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

static inline void gemm_f32_avx512(const float *a, const float *b, float *out,
                                   size_t m_dim, size_t k_dim, size_t n_dim,
                                   intptr_t rsa, intptr_t csa, intptr_t rsb,
                                   intptr_t rso) {
  /* B packing buffer -- shared across threads, fits L3 */
  size_t nc_max = GEMM_MIN(GEMM_F32_NC, n_dim);
  float *packed_b = (float *)numc_malloc(
      64, GEMM_F32_KC * (nc_max + GEMM_F32_NR) * sizeof(float));
  if (!packed_b)
    return;

  for (size_t jc = 0; jc < n_dim; jc += GEMM_F32_NC) {
    size_t nc = GEMM_MIN(GEMM_F32_NC, n_dim - jc);

    for (size_t pc = 0; pc < k_dim; pc += GEMM_F32_KC) {
      size_t kc = GEMM_MIN(GEMM_F32_KC, k_dim - pc);
      int first = (pc == 0);

      gemm_pack_b_f32(b + pc * rsb + jc, packed_b, kc, nc, rsb);

      /* 2D IC x JR parallelism: linearize (ic_idx, jr_idx) into flat tasks. */
      size_t n_ic = (m_dim + GEMM_F32_MC - 1) / GEMM_F32_MC;
      size_t n_jr = (nc + GEMM_F32_NR - 1) / GEMM_F32_NR;
      size_t n_tasks = n_ic * n_jr;

#ifdef _OPENMP
#pragma omp parallel if ((uint64_t)m_dim * kc * nc > GEMM_OMP_THRESHOLD)
      {
        NUMC_ALIGNAS(64) float packed_a[GEMM_F32_MC * GEMM_F32_KC];
        size_t last_ic = (size_t)-1;

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
              gemm_ukernel_f32_12x32(packed_a + ir * kc, packed_b + jr * kc,
                                     out + (ic + ir) * rso + (jc + jr), kc, 1,
                                     GEMM_F32_MR, GEMM_F32_NR, rso, first);
            } else {
              NUMC_ALIGNAS(64) float tmp[GEMM_F32_MR * GEMM_F32_NR];
              gemm_ukernel_f32_12x32(packed_a + ir * kc, packed_b + jr * kc,
                                     tmp, kc, 1, GEMM_F32_MR, GEMM_F32_NR,
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
#else
      {
        NUMC_ALIGNAS(64) float packed_a[GEMM_F32_MC * GEMM_F32_KC];

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
              gemm_ukernel_f32_12x32(packed_a + ir * kc, packed_b + jr * kc,
                                     out + (ic + ir) * rso + (jc + jr), kc, 1,
                                     GEMM_F32_MR, GEMM_F32_NR, rso, first);
            } else {
              NUMC_ALIGNAS(64) float tmp[GEMM_F32_MR * GEMM_F32_NR];
              gemm_ukernel_f32_12x32(packed_a + ir * kc, packed_b + jr * kc,
                                     tmp, kc, 1, GEMM_F32_MR, GEMM_F32_NR,
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
#endif
    }
  }

  numc_free(packed_b);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Float64: 14x16 micro-kernel (28 acc + 1 A broadcast + 2 B loads = 31 ZMM)
   ═══════════════════════════════════════════════════════════════════════════
 */

#define GEMM_F64_K_ITER(ap, bp)                   \
  do {                                            \
    __m512d b0 = _mm512_loadu_pd(bp);             \
    __m512d b1 = _mm512_loadu_pd((bp) + 8);       \
    for (size_t _i = 0; _i < GEMM_F64_MR; _i++) { \
      __m512d av = _mm512_set1_pd(*((ap) + _i));  \
      c0[_i] = _mm512_fmadd_pd(av, b0, c0[_i]);   \
      c1[_i] = _mm512_fmadd_pd(av, b1, c1[_i]);   \
    }                                             \
  } while (0)

static inline void gemm_ukernel_f64_14x16(const double *a, const double *b,
                                          double *c, size_t kc, intptr_t rsa,
                                          intptr_t csa, intptr_t rsb,
                                          intptr_t rso, int first) {
  __m512d c0[GEMM_F64_MR], c1[GEMM_F64_MR];

  for (size_t i = 0; i < GEMM_F64_MR; i++)
    _mm_prefetch((const char *)(c + i * rso), _MM_HINT_T0);

  if (first) {
    for (size_t i = 0; i < GEMM_F64_MR; i++) {
      c0[i] = _mm512_setzero_pd();
      c1[i] = _mm512_setzero_pd();
    }
  } else {
    for (size_t i = 0; i < GEMM_F64_MR; i++) {
      double *ci = c + i * rso;
      c0[i] = _mm512_loadu_pd(ci);
      c1[i] = _mm512_loadu_pd(ci + 8);
    }
  }

  const double *ap = a;
  const double *bp = b;
  size_t k_iter = kc / 4;
  size_t k_left = kc % 4;

  for (size_t ki = 0; ki < k_iter; ki++) {
    GEMM_F64_K_ITER(ap, bp);
    ap += csa;
    bp += rsb;
    GEMM_F64_K_ITER(ap, bp);
    ap += csa;
    bp += rsb;
    _mm_prefetch((const char *)(ap + 2 * GEMM_F64_MR), _MM_HINT_T0);
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

  for (size_t i = 0; i < GEMM_F64_MR; i++) {
    double *ci = c + i * rso;
    _mm512_storeu_pd(ci, c0[i]);
    _mm512_storeu_pd(ci + 8, c1[i]);
  }
}

#undef GEMM_F64_K_ITER

static inline void gemm_edge_f64_avx512(const double *a, const double *b,
                                        double *c, size_t mr, size_t nr,
                                        size_t kc, intptr_t rsa, intptr_t csa,
                                        intptr_t rsb, intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
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

static inline void gemm_f64_avx512(const double *a, const double *b,
                                   double *out, size_t m_dim, size_t k_dim,
                                   size_t n_dim, intptr_t rsa, intptr_t csa,
                                   intptr_t rsb, intptr_t rso) {
  size_t nc_max = GEMM_MIN(GEMM_F64_NC, n_dim);
  double *packed_b = (double *)numc_malloc(
      64, GEMM_F64_KC * (nc_max + GEMM_F64_NR) * sizeof(double));
  if (!packed_b)
    return;

  for (size_t jc = 0; jc < n_dim; jc += GEMM_F64_NC) {
    size_t nc = GEMM_MIN(GEMM_F64_NC, n_dim - jc);

    for (size_t pc = 0; pc < k_dim; pc += GEMM_F64_KC) {
      size_t kc = GEMM_MIN(GEMM_F64_KC, k_dim - pc);
      int first = (pc == 0);

      gemm_pack_b_f64(b + pc * rsb + jc, packed_b, kc, nc, rsb);

      size_t n_ic = (m_dim + GEMM_F64_MC - 1) / GEMM_F64_MC;
      size_t n_jr = (nc + GEMM_F64_NR - 1) / GEMM_F64_NR;
      size_t n_tasks = n_ic * n_jr;

#ifdef _OPENMP
#pragma omp parallel if ((uint64_t)m_dim * kc * nc > GEMM_OMP_THRESHOLD)
      {
        NUMC_ALIGNAS(64) double packed_a[GEMM_F64_MC * GEMM_F64_KC];
        size_t last_ic = (size_t)-1;

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
              gemm_ukernel_f64_14x16(packed_a + ir * kc, packed_b + jr * kc,
                                     out + (ic + ir) * rso + (jc + jr), kc, 1,
                                     GEMM_F64_MR, GEMM_F64_NR, rso, first);
            } else {
              NUMC_ALIGNAS(64) double tmp[GEMM_F64_MR * GEMM_F64_NR];
              gemm_ukernel_f64_14x16(packed_a + ir * kc, packed_b + jr * kc,
                                     tmp, kc, 1, GEMM_F64_MR, GEMM_F64_NR,
                                     GEMM_F64_NR, 1);
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
#else
      {
        NUMC_ALIGNAS(64) double packed_a[GEMM_F64_MC * GEMM_F64_KC];

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
              gemm_ukernel_f64_14x16(packed_a + ir * kc, packed_b + jr * kc,
                                     out + (ic + ir) * rso + (jc + jr), kc, 1,
                                     GEMM_F64_MR, GEMM_F64_NR, rso, first);
            } else {
              NUMC_ALIGNAS(64) double tmp[GEMM_F64_MR * GEMM_F64_NR];
              gemm_ukernel_f64_14x16(packed_a + ir * kc, packed_b + jr * kc,
                                     tmp, kc, 1, GEMM_F64_MR, GEMM_F64_NR,
                                     GEMM_F64_NR, 1);
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
#endif
    }
  }

  numc_free(packed_b);
}

#undef GEMM_MIN

#endif
