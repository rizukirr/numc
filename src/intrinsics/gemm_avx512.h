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
   * NR=32). 8x unrolled K-loop with A+B prefetch (BLIS SKX pattern). */
  const float *ap = a;
  const float *bp = b;
  size_t k_iter = kc / 8;
  size_t k_left = kc % 8;

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
    _mm_prefetch((const char *)(bp + 2 * GEMM_F32_NR), _MM_HINT_T0);
    GEMM_F32_K_ITER(ap, bp);
    ap += csa;
    bp += rsb;
    GEMM_F32_K_ITER(ap, bp);
    ap += csa;
    bp += rsb;
    _mm_prefetch((const char *)(ap + 4 * GEMM_F32_MR), _MM_HINT_T0);
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
  size_t k_iter = kc / 8;
  size_t k_left = kc % 8;

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
    _mm_prefetch((const char *)(bp + 2 * GEMM_F64_NR), _MM_HINT_T0);
    GEMM_F64_K_ITER(ap, bp);
    ap += csa;
    bp += rsb;
    GEMM_F64_K_ITER(ap, bp);
    ap += csa;
    bp += rsb;
    _mm_prefetch((const char *)(ap + 4 * GEMM_F64_MR), _MM_HINT_T0);
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

/* ═══════════════════════════════════════════════════════════════════════════
   Int32/Uint32: 12×32 micro-kernel (mullo_epi32 + add_epi32, ZMM)
   mullo_epi32 produces identical low 32 bits for signed and unsigned.
   Reuses f32 blocking parameters (same element size).
   ═══════════════════════════════════════════════════════════════════════════
 */

#undef GEMM_I32_MR
#undef GEMM_I32_NR
#undef GEMM_I32_MC
#undef GEMM_I32_KC
#define GEMM_I32_MR 12
#define GEMM_I32_NR 32
#define GEMM_I32_MC 480
#define GEMM_I32_KC 384
#define GEMM_I32_NC 3072

static inline void gemm_ukernel_i32_12x32(const int32_t *a, const int32_t *b,
                                          int32_t *c, size_t kc, intptr_t rsa,
                                          intptr_t csa, intptr_t rsb,
                                          intptr_t rso) {
  __m512i c0[GEMM_I32_MR], c1[GEMM_I32_MR];

  for (size_t i = 0; i < GEMM_I32_MR; i++) {
    c0[i] = _mm512_loadu_si512(c + i * rso);
    c1[i] = _mm512_loadu_si512(c + i * rso + 16);
  }

  const int32_t *ap = a;
  const int32_t *bp = b;
  for (size_t p = 0; p < kc; p++) {
    __m512i b0 = _mm512_loadu_si512(bp);
    __m512i b1 = _mm512_loadu_si512(bp + 16);
    for (size_t i = 0; i < GEMM_I32_MR; i++) {
      __m512i av = _mm512_set1_epi32(ap[i]);
      c0[i] = _mm512_add_epi32(c0[i], _mm512_mullo_epi32(av, b0));
      c1[i] = _mm512_add_epi32(c1[i], _mm512_mullo_epi32(av, b1));
    }
    ap += csa;
    bp += rsb;
  }

  for (size_t i = 0; i < GEMM_I32_MR; i++) {
    _mm512_storeu_si512(c + i * rso, c0[i]);
    _mm512_storeu_si512(c + i * rso + 16, c1[i]);
  }
}

static inline void gemm_edge_i32_avx512(const int32_t *a, const int32_t *b,
                                        int32_t *c, size_t mr, size_t nr,
                                        size_t kc, intptr_t rsa, intptr_t csa,
                                        intptr_t rsb, intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
    for (size_t p = 0; p < kc; p++) {
      int32_t aip = a[i * rsa + p * csa];
      __m512i va = _mm512_set1_epi32(aip);
      const int32_t *brow = b + p * rsb;
      int32_t *crow = c + i * rso;
      size_t j = 0;
      for (; j + 16 <= nr; j += 16) {
        __m512i vo = _mm512_loadu_si512(crow + j);
        vo = _mm512_add_epi32(
            vo, _mm512_mullo_epi32(va, _mm512_loadu_si512(brow + j)));
        _mm512_storeu_si512(crow + j, vo);
      }
      for (; j < nr; j++)
        crow[j] += aip * brow[j];
    }
  }
}

static inline void gemm_i32_avx512(const int32_t *a, const int32_t *b,
                                   int32_t *out, size_t m_dim, size_t k_dim,
                                   size_t n_dim, intptr_t rsa, intptr_t csa,
                                   intptr_t rsb, intptr_t rso) {
  for (size_t i = 0; i < m_dim; i++)
    memset(out + i * rso, 0, n_dim * sizeof(int32_t));

  for (size_t pc = 0; pc < k_dim; pc += GEMM_I32_KC) {
    size_t kc = GEMM_MIN(GEMM_I32_KC, k_dim - pc);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if ((uint64_t)m_dim * n_dim * \
                                                      sizeof(int32_t) > \
                                                  GEMM_OMP_THRESHOLD)
#endif
    for (size_t ic = 0; ic < m_dim; ic += GEMM_I32_MC) {
      size_t mc = GEMM_MIN(GEMM_I32_MC, m_dim - ic);
      size_t jr = 0;
      for (; jr + GEMM_I32_NR <= n_dim; jr += GEMM_I32_NR) {
        size_t ir = 0;
        for (; ir + GEMM_I32_MR <= mc; ir += GEMM_I32_MR)
          gemm_ukernel_i32_12x32(a + (ic + ir) * rsa + pc * csa,
                                 b + pc * rsb + jr, out + (ic + ir) * rso + jr,
                                 kc, rsa, csa, rsb, rso);
        if (ir < mc)
          gemm_edge_i32_avx512(a + (ic + ir) * rsa + pc * csa,
                               b + pc * rsb + jr, out + (ic + ir) * rso + jr,
                               mc - ir, GEMM_I32_NR, kc, rsa, csa, rsb, rso);
      }
      if (jr < n_dim)
        gemm_edge_i32_avx512(a + ic * rsa + pc * csa, b + pc * rsb + jr,
                             out + ic * rso + jr, mc, n_dim - jr, kc, rsa, csa,
                             rsb, rso);
    }
  }
}

/* Uint32: identical bit-level operations as int32 */
static inline void gemm_u32_avx512(const uint32_t *a, const uint32_t *b,
                                   uint32_t *out, size_t m_dim, size_t k_dim,
                                   size_t n_dim, intptr_t rsa, intptr_t csa,
                                   intptr_t rsb, intptr_t rso) {
  gemm_i32_avx512((const int32_t *)a, (const int32_t *)b, (int32_t *)out, m_dim,
                  k_dim, n_dim, rsa, csa, rsb, rso);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Int16/Uint16: 12×64 micro-kernel (mullo_epi16 + add_epi16, ZMM)
   32 i16 per ZMM, 2 ZMM per row = 64 elements.
   ═══════════════════════════════════════════════════════════════════════════
 */

#undef GEMM_I16_MR
#undef GEMM_I16_NR
#undef GEMM_I16_MC
#undef GEMM_I16_KC
#define GEMM_I16_MR 12
#define GEMM_I16_NR 64
#define GEMM_I16_MC 240
#define GEMM_I16_KC 512
#define GEMM_I16_NC 4096

static inline void gemm_ukernel_i16_12x64(const int16_t *a, const int16_t *b,
                                          int16_t *c, size_t kc, intptr_t rsa,
                                          intptr_t csa, intptr_t rsb,
                                          intptr_t rso) {
  __m512i c0[GEMM_I16_MR], c1[GEMM_I16_MR];

  for (size_t i = 0; i < GEMM_I16_MR; i++) {
    c0[i] = _mm512_loadu_si512(c + i * rso);
    c1[i] = _mm512_loadu_si512(c + i * rso + 32);
  }

  const int16_t *ap = a;
  const int16_t *bp = b;
  for (size_t p = 0; p < kc; p++) {
    __m512i b0 = _mm512_loadu_si512(bp);
    __m512i b1 = _mm512_loadu_si512(bp + 32);
    for (size_t i = 0; i < GEMM_I16_MR; i++) {
      __m512i av = _mm512_set1_epi16(ap[i]);
      c0[i] = _mm512_add_epi16(c0[i], _mm512_mullo_epi16(av, b0));
      c1[i] = _mm512_add_epi16(c1[i], _mm512_mullo_epi16(av, b1));
    }
    ap += csa;
    bp += rsb;
  }

  for (size_t i = 0; i < GEMM_I16_MR; i++) {
    _mm512_storeu_si512(c + i * rso, c0[i]);
    _mm512_storeu_si512(c + i * rso + 32, c1[i]);
  }
}

static inline void gemm_edge_i16_avx512(const int16_t *a, const int16_t *b,
                                        int16_t *c, size_t mr, size_t nr,
                                        size_t kc, intptr_t rsa, intptr_t csa,
                                        intptr_t rsb, intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
    for (size_t p = 0; p < kc; p++) {
      int16_t aip = a[i * rsa + p * csa];
      __m512i va = _mm512_set1_epi16(aip);
      const int16_t *brow = b + p * rsb;
      int16_t *crow = c + i * rso;
      size_t j = 0;
      for (; j + 32 <= nr; j += 32) {
        __m512i vo = _mm512_loadu_si512(crow + j);
        vo = _mm512_add_epi16(
            vo, _mm512_mullo_epi16(va, _mm512_loadu_si512(brow + j)));
        _mm512_storeu_si512(crow + j, vo);
      }
      for (; j < nr; j++)
        crow[j] = (int16_t)(crow[j] + aip * brow[j]);
    }
  }
}

static inline void gemm_i16_avx512(const int16_t *a, const int16_t *b,
                                   int16_t *out, size_t m_dim, size_t k_dim,
                                   size_t n_dim, intptr_t rsa, intptr_t csa,
                                   intptr_t rsb, intptr_t rso) {
  for (size_t i = 0; i < m_dim; i++)
    memset(out + i * rso, 0, n_dim * sizeof(int16_t));

  for (size_t pc = 0; pc < k_dim; pc += GEMM_I16_KC) {
    size_t kc = GEMM_MIN(GEMM_I16_KC, k_dim - pc);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if ((uint64_t)m_dim * n_dim * \
                                                      sizeof(int16_t) > \
                                                  GEMM_OMP_THRESHOLD)
#endif
    for (size_t ic = 0; ic < m_dim; ic += GEMM_I16_MC) {
      size_t mc = GEMM_MIN(GEMM_I16_MC, m_dim - ic);
      size_t jr = 0;
      for (; jr + GEMM_I16_NR <= n_dim; jr += GEMM_I16_NR) {
        size_t ir = 0;
        for (; ir + GEMM_I16_MR <= mc; ir += GEMM_I16_MR)
          gemm_ukernel_i16_12x64(a + (ic + ir) * rsa + pc * csa,
                                 b + pc * rsb + jr, out + (ic + ir) * rso + jr,
                                 kc, rsa, csa, rsb, rso);
        if (ir < mc)
          gemm_edge_i16_avx512(a + (ic + ir) * rsa + pc * csa,
                               b + pc * rsb + jr, out + (ic + ir) * rso + jr,
                               mc - ir, GEMM_I16_NR, kc, rsa, csa, rsb, rso);
      }
      if (jr < n_dim)
        gemm_edge_i16_avx512(a + ic * rsa + pc * csa, b + pc * rsb + jr,
                             out + ic * rso + jr, mc, n_dim - jr, kc, rsa, csa,
                             rsb, rso);
    }
  }
}

/* Uint16: identical bit-level operations as int16 */
static inline void gemm_u16_avx512(const uint16_t *a, const uint16_t *b,
                                   uint16_t *out, size_t m_dim, size_t k_dim,
                                   size_t n_dim, intptr_t rsa, intptr_t csa,
                                   intptr_t rsb, intptr_t rso) {
  gemm_i16_avx512((const int16_t *)a, (const int16_t *)b, (int16_t *)out, m_dim,
                  k_dim, n_dim, rsa, csa, rsb, rso);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Int64/Uint64: 14×16 micro-kernel (ZMM, 8 i64 per register)
   AVX-512DQ has native _mm512_mullo_epi64; fallback uses widening.
   ═══════════════════════════════════════════════════════════════════════════
 */

#undef GEMM_I64_MR
#undef GEMM_I64_NR
#undef GEMM_I64_MC
#undef GEMM_I64_KC
#define GEMM_I64_MR 14
#define GEMM_I64_NR 16
#define GEMM_I64_MC 240
#define GEMM_I64_KC 64
#define GEMM_I64_NC 3752

#ifndef __AVX512DQ__
static inline __m512i gemm_mullo_epi64_512(__m512i a, __m512i b) {
  __m512i a_hi = _mm512_srli_epi64(a, 32);
  __m512i b_hi = _mm512_srli_epi64(b, 32);
  __m512i lo_lo = _mm512_mul_epu32(a, b);
  __m512i cross =
      _mm512_add_epi64(_mm512_mul_epu32(a, b_hi), _mm512_mul_epu32(a_hi, b));
  return _mm512_add_epi64(lo_lo, _mm512_slli_epi64(cross, 32));
}
#endif

static inline void gemm_ukernel_i64_14x16(const int64_t *a, const int64_t *b,
                                          int64_t *c, size_t kc, intptr_t rsa,
                                          intptr_t csa, intptr_t rsb,
                                          intptr_t rso) {
  __m512i c0[GEMM_I64_MR], c1[GEMM_I64_MR];

  for (size_t i = 0; i < GEMM_I64_MR; i++) {
    c0[i] = _mm512_loadu_si512(c + i * rso);
    c1[i] = _mm512_loadu_si512(c + i * rso + 8);
  }

  const int64_t *ap = a;
  const int64_t *bp = b;
  for (size_t p = 0; p < kc; p++) {
    __m512i b0 = _mm512_loadu_si512(bp);
    __m512i b1 = _mm512_loadu_si512(bp + 8);
    for (size_t i = 0; i < GEMM_I64_MR; i++) {
      __m512i av = _mm512_set1_epi64(ap[i]);
#ifdef __AVX512DQ__
      c0[i] = _mm512_add_epi64(c0[i], _mm512_mullo_epi64(av, b0));
      c1[i] = _mm512_add_epi64(c1[i], _mm512_mullo_epi64(av, b1));
#else
      c0[i] = _mm512_add_epi64(c0[i], gemm_mullo_epi64_512(av, b0));
      c1[i] = _mm512_add_epi64(c1[i], gemm_mullo_epi64_512(av, b1));
#endif
    }
    ap += csa;
    bp += rsb;
  }

  for (size_t i = 0; i < GEMM_I64_MR; i++) {
    _mm512_storeu_si512(c + i * rso, c0[i]);
    _mm512_storeu_si512(c + i * rso + 8, c1[i]);
  }
}

static inline void gemm_edge_i64_avx512(const int64_t *a, const int64_t *b,
                                        int64_t *c, size_t mr, size_t nr,
                                        size_t kc, intptr_t rsa, intptr_t csa,
                                        intptr_t rsb, intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
    for (size_t p = 0; p < kc; p++) {
      int64_t aip = a[i * rsa + p * csa];
      const int64_t *brow = b + p * rsb;
      int64_t *crow = c + i * rso;
      for (size_t j = 0; j < nr; j++)
        crow[j] += aip * brow[j];
    }
  }
}

static inline void gemm_i64_avx512(const int64_t *a, const int64_t *b,
                                   int64_t *out, size_t m_dim, size_t k_dim,
                                   size_t n_dim, intptr_t rsa, intptr_t csa,
                                   intptr_t rsb, intptr_t rso) {
  for (size_t i = 0; i < m_dim; i++)
    memset(out + i * rso, 0, n_dim * sizeof(int64_t));

  for (size_t pc = 0; pc < k_dim; pc += GEMM_I64_KC) {
    size_t kc = GEMM_MIN(GEMM_I64_KC, k_dim - pc);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if ((uint64_t)m_dim * n_dim * \
                                                      sizeof(int64_t) > \
                                                  GEMM_OMP_THRESHOLD)
#endif
    for (size_t ic = 0; ic < m_dim; ic += GEMM_I64_MC) {
      size_t mc = GEMM_MIN(GEMM_I64_MC, m_dim - ic);
      size_t jr = 0;
      for (; jr + GEMM_I64_NR <= n_dim; jr += GEMM_I64_NR) {
        size_t ir = 0;
        for (; ir + GEMM_I64_MR <= mc; ir += GEMM_I64_MR)
          gemm_ukernel_i64_14x16(a + (ic + ir) * rsa + pc * csa,
                                 b + pc * rsb + jr, out + (ic + ir) * rso + jr,
                                 kc, rsa, csa, rsb, rso);
        if (ir < mc)
          gemm_edge_i64_avx512(a + (ic + ir) * rsa + pc * csa,
                               b + pc * rsb + jr, out + (ic + ir) * rso + jr,
                               mc - ir, GEMM_I64_NR, kc, rsa, csa, rsb, rso);
      }
      if (jr < n_dim)
        gemm_edge_i64_avx512(a + ic * rsa + pc * csa, b + pc * rsb + jr,
                             out + ic * rso + jr, mc, n_dim - jr, kc, rsa, csa,
                             rsb, rso);
    }
  }
}

/* Uint64: identical bit-level operations as int64 */
static inline void gemm_u64_avx512(const uint64_t *a, const uint64_t *b,
                                   uint64_t *out, size_t m_dim, size_t k_dim,
                                   size_t n_dim, intptr_t rsa, intptr_t csa,
                                   intptr_t rsb, intptr_t rso) {
  gemm_i64_avx512((const int64_t *)a, (const int64_t *)b, (int64_t *)out, m_dim,
                  k_dim, n_dim, rsa, csa, rsb, rso);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Int8: 12×16 promoted micro-kernel (widen to i32, full-K accumulation)
   Load 16 i8, sign-extend to 16 i32 via _mm512_cvtepi8_epi32,
   accumulate in i32, narrow back to i8 on store.
   ═══════════════════════════════════════════════════════════════════════════
 */

#undef GEMM_I8_MR
#undef GEMM_I8_NR
#undef GEMM_I8_MC
#define GEMM_I8_MR 12
#define GEMM_I8_NR 16
#define GEMM_I8_MC 240

static inline void gemm_ukernel_i8_12x16(const int8_t *a, const int8_t *b,
                                         int8_t *c, size_t k_dim, intptr_t rsa,
                                         intptr_t csa, intptr_t rsb,
                                         intptr_t rso) {
  __m512i acc[GEMM_I8_MR];
  for (size_t i = 0; i < GEMM_I8_MR; i++)
    acc[i] = _mm512_setzero_si512();

  for (size_t p = 0; p < k_dim; p++) {
    const int8_t *bp = b + p * rsb;
    __m512i b0 = _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i const *)bp));
    for (size_t i = 0; i < GEMM_I8_MR; i++) {
      __m512i av = _mm512_set1_epi32((int32_t)a[i * rsa + p * csa]);
      acc[i] = _mm512_add_epi32(acc[i], _mm512_mullo_epi32(av, b0));
    }
  }

  /* Narrow i32 accumulators back to i8 via _mm512_cvtsepi32_epi8 */
  for (size_t i = 0; i < GEMM_I8_MR; i++) {
    __m128i narrow = _mm512_cvtsepi32_epi8(acc[i]);
    _mm_storeu_si128((__m128i *)(c + i * rso), narrow);
  }
}

static inline void gemm_edge_i8_avx512(const int8_t *a, const int8_t *b,
                                       int8_t *c, size_t mr, size_t nr,
                                       size_t k_dim, intptr_t rsa, intptr_t csa,
                                       intptr_t rsb, intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
    for (size_t j = 0; j < nr; j++) {
      int32_t sum = 0;
      for (size_t p = 0; p < k_dim; p++)
        sum += (int32_t)a[i * rsa + p * csa] * (int32_t)b[p * rsb + j];
      c[i * rso + j] = (int8_t)sum;
    }
  }
}

static inline void gemm_i8_avx512(const int8_t *a, const int8_t *b, int8_t *out,
                                  size_t m_dim, size_t k_dim, size_t n_dim,
                                  intptr_t rsa, intptr_t csa, intptr_t rsb,
                                  intptr_t rso) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if ((uint64_t)m_dim * n_dim > \
                                                  GEMM_OMP_THRESHOLD)
#endif
  for (size_t ic = 0; ic < m_dim; ic += GEMM_I8_MC) {
    size_t mc = GEMM_MIN(GEMM_I8_MC, m_dim - ic);
    size_t jr = 0;
    for (; jr + GEMM_I8_NR <= n_dim; jr += GEMM_I8_NR) {
      size_t ir = 0;
      for (; ir + GEMM_I8_MR <= mc; ir += GEMM_I8_MR)
        gemm_ukernel_i8_12x16(a + (ic + ir) * rsa, b + jr,
                              out + (ic + ir) * rso + jr, k_dim, rsa, csa, rsb,
                              rso);
      if (ir < mc)
        gemm_edge_i8_avx512(a + (ic + ir) * rsa, b + jr,
                            out + (ic + ir) * rso + jr, mc - ir, GEMM_I8_NR,
                            k_dim, rsa, csa, rsb, rso);
    }
    if (jr < n_dim)
      gemm_edge_i8_avx512(a + ic * rsa, b + jr, out + ic * rso + jr, mc,
                          n_dim - jr, k_dim, rsa, csa, rsb, rso);
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   Uint8: 12×16 promoted micro-kernel (widen to u32, full-K accumulation)
   Same as i8 but uses zero-extension and unsigned truncation.
   ═══════════════════════════════════════════════════════════════════════════
 */

static inline void gemm_ukernel_u8_12x16(const uint8_t *a, const uint8_t *b,
                                         uint8_t *c, size_t k_dim, intptr_t rsa,
                                         intptr_t csa, intptr_t rsb,
                                         intptr_t rso) {
  __m512i acc[GEMM_I8_MR];
  for (size_t i = 0; i < GEMM_I8_MR; i++)
    acc[i] = _mm512_setzero_si512();

  for (size_t p = 0; p < k_dim; p++) {
    const uint8_t *bp = b + p * rsb;
    __m512i b0 = _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i const *)bp));
    for (size_t i = 0; i < GEMM_I8_MR; i++) {
      __m512i av = _mm512_set1_epi32((int32_t)(uint32_t)a[i * rsa + p * csa]);
      acc[i] = _mm512_add_epi32(acc[i], _mm512_mullo_epi32(av, b0));
    }
  }

  /* Narrow i32 accumulators back to u8 via _mm512_cvtusepi32_epi8 */
  for (size_t i = 0; i < GEMM_I8_MR; i++) {
    __m128i narrow = _mm512_cvtusepi32_epi8(acc[i]);
    _mm_storeu_si128((__m128i *)(c + i * rso), narrow);
  }
}

static inline void gemm_edge_u8_avx512(const uint8_t *a, const uint8_t *b,
                                       uint8_t *c, size_t mr, size_t nr,
                                       size_t k_dim, intptr_t rsa, intptr_t csa,
                                       intptr_t rsb, intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
    for (size_t j = 0; j < nr; j++) {
      uint32_t sum = 0;
      for (size_t p = 0; p < k_dim; p++)
        sum += (uint32_t)a[i * rsa + p * csa] * (uint32_t)b[p * rsb + j];
      c[i * rso + j] = (uint8_t)sum;
    }
  }
}

static inline void gemm_u8_avx512(const uint8_t *a, const uint8_t *b,
                                  uint8_t *out, size_t m_dim, size_t k_dim,
                                  size_t n_dim, intptr_t rsa, intptr_t csa,
                                  intptr_t rsb, intptr_t rso) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if ((uint64_t)m_dim * n_dim > \
                                                  GEMM_OMP_THRESHOLD)
#endif
  for (size_t ic = 0; ic < m_dim; ic += GEMM_I8_MC) {
    size_t mc = GEMM_MIN(GEMM_I8_MC, m_dim - ic);
    size_t jr = 0;
    for (; jr + GEMM_I8_NR <= n_dim; jr += GEMM_I8_NR) {
      size_t ir = 0;
      for (; ir + GEMM_I8_MR <= mc; ir += GEMM_I8_MR)
        gemm_ukernel_u8_12x16(a + (ic + ir) * rsa, b + jr,
                              out + (ic + ir) * rso + jr, k_dim, rsa, csa, rsb,
                              rso);
      if (ir < mc)
        gemm_edge_u8_avx512(a + (ic + ir) * rsa, b + jr,
                            out + (ic + ir) * rso + jr, mc - ir, GEMM_I8_NR,
                            k_dim, rsa, csa, rsb, rso);
    }
    if (jr < n_dim)
      gemm_edge_u8_avx512(a + ic * rsa, b + jr, out + ic * rso + jr, mc,
                          n_dim - jr, k_dim, rsa, csa, rsb, rso);
  }
}

#undef GEMM_MIN

#endif
