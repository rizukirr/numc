#ifndef NUMC_GEMM_AVX2_H
#define NUMC_GEMM_AVX2_H

#include <immintrin.h>
#include <stdint.h>
#include <string.h>

#define GEMM_MIN(a, b) ((a) < (b) ? (a) : (b))

/*
 * Cache-blocking parameters tuned for typical L1 (32KB) / L2 (256KB).
 *   - MC × KC panel of A resides in L2
 *   - KC × NR sliver of B resides in L1
 */
#define GEMM_F32_MR 6
#define GEMM_F32_NR 16
#define GEMM_F32_MC 72
#define GEMM_F32_KC 256

#define GEMM_F64_MR 6
#define GEMM_F64_NR 8
#define GEMM_F64_MC 72
#define GEMM_F64_KC 128

#define GEMM_I32_MR 6
#define GEMM_I32_NR 16
#define GEMM_I32_MC 72
#define GEMM_I32_KC 256

#define GEMM_I16_MR 6
#define GEMM_I16_NR 32
#define GEMM_I16_MC 72
#define GEMM_I16_KC 512

#define GEMM_I64_MR 6
#define GEMM_I64_NR 8
#define GEMM_I64_MC 72
#define GEMM_I64_KC 64

/* i8/u8: promoted to i32 accumulators, no K-blocking */
#define GEMM_I8_MR 6
#define GEMM_I8_NR 8
#define GEMM_I8_MC 72

#define GEMM_OMP_THRESHOLD (1 << 20)

/* ═══════════════════════════════════════════════════════════════════════════
   Float32: 6×16 micro-kernel  (12 acc + 1 A broadcast + 2 B loads = 15 YMM)
   ═══════════════════════════════════════════════════════════════════════════
 */

static inline void gemm_ukernel_f32_6x16(const float *a, const float *b,
                                         float *c, size_t kc, intptr_t rsa,
                                         intptr_t csa, intptr_t rsb,
                                         intptr_t rso) {
  __m256 c00 = _mm256_loadu_ps(c), c01 = _mm256_loadu_ps(c + 8);
  __m256 c10 = _mm256_loadu_ps(c + rso), c11 = _mm256_loadu_ps(c + rso + 8);
  __m256 c20 = _mm256_loadu_ps(c + 2 * rso),
         c21 = _mm256_loadu_ps(c + 2 * rso + 8);
  __m256 c30 = _mm256_loadu_ps(c + 3 * rso),
         c31 = _mm256_loadu_ps(c + 3 * rso + 8);
  __m256 c40 = _mm256_loadu_ps(c + 4 * rso),
         c41 = _mm256_loadu_ps(c + 4 * rso + 8);
  __m256 c50 = _mm256_loadu_ps(c + 5 * rso),
         c51 = _mm256_loadu_ps(c + 5 * rso + 8);

  for (size_t p = 0; p < kc; p++) {
    const float *bp = b + p * rsb;
    _mm_prefetch((const char *)(bp + rsb), _MM_HINT_T0);
    _mm_prefetch((const char *)(bp + rsb + 8), _MM_HINT_T0);
    __m256 b0 = _mm256_loadu_ps(bp);
    __m256 b1 = _mm256_loadu_ps(bp + 8);

    __m256 av;
    /* Interleave broadcasts with FMAs to maximize port utilization */
    av = _mm256_broadcast_ss(a + 0 * rsa + p * csa);
    c00 = _mm256_fmadd_ps(av, b0, c00);
    c01 = _mm256_fmadd_ps(av, b1, c01);

    av = _mm256_broadcast_ss(a + 1 * rsa + p * csa);
    c10 = _mm256_fmadd_ps(av, b0, c10);
    c11 = _mm256_fmadd_ps(av, b1, c11);

    av = _mm256_broadcast_ss(a + 2 * rsa + p * csa);
    c20 = _mm256_fmadd_ps(av, b0, c20);
    c21 = _mm256_fmadd_ps(av, b1, c21);

    av = _mm256_broadcast_ss(a + 3 * rsa + p * csa);
    c30 = _mm256_fmadd_ps(av, b0, c30);
    c31 = _mm256_fmadd_ps(av, b1, c31);

    av = _mm256_broadcast_ss(a + 4 * rsa + p * csa);
    c40 = _mm256_fmadd_ps(av, b0, c40);
    c41 = _mm256_fmadd_ps(av, b1, c41);

    av = _mm256_broadcast_ss(a + 5 * rsa + p * csa);
    c50 = _mm256_fmadd_ps(av, b0, c50);
    c51 = _mm256_fmadd_ps(av, b1, c51);
  }

  _mm256_storeu_ps(c, c00);
  _mm256_storeu_ps(c + 8, c01);
  _mm256_storeu_ps(c + rso, c10);
  _mm256_storeu_ps(c + rso + 8, c11);
  _mm256_storeu_ps(c + 2 * rso, c20);
  _mm256_storeu_ps(c + 2 * rso + 8, c21);
  _mm256_storeu_ps(c + 3 * rso, c30);
  _mm256_storeu_ps(c + 3 * rso + 8, c31);
  _mm256_storeu_ps(c + 4 * rso, c40);
  _mm256_storeu_ps(c + 4 * rso + 8, c41);
  _mm256_storeu_ps(c + 5 * rso, c50);
  _mm256_storeu_ps(c + 5 * rso + 8, c51);
}

static inline void gemm_edge_f32(const float *a, const float *b, float *c,
                                 size_t mr, size_t nr, size_t kc, intptr_t rsa,
                                 intptr_t csa, intptr_t rsb, intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
    for (size_t p = 0; p < kc; p++) {
      float aip = a[i * rsa + p * csa];
      __m256 va = _mm256_set1_ps(aip);
      const float *brow = b + p * rsb;
      float *crow = c + i * rso;
      size_t j = 0;
      for (; j + 8 <= nr; j += 8) {
        __m256 vo = _mm256_loadu_ps(crow + j);
        vo = _mm256_fmadd_ps(va, _mm256_loadu_ps(brow + j), vo);
        _mm256_storeu_ps(crow + j, vo);
      }
      for (; j < nr; j++)
        crow[j] += aip * brow[j];
    }
  }
}

static inline void gemm_f32_avx2(const float *a, const float *b, float *out,
                                 size_t m_dim, size_t k_dim, size_t n_dim,
                                 intptr_t rsa, intptr_t csa, intptr_t rsb,
                                 intptr_t rso) {
  for (size_t i = 0; i < m_dim; i++)
    memset(out + i * rso, 0, n_dim * sizeof(float));

  for (size_t pc = 0; pc < k_dim; pc += GEMM_F32_KC) {
    size_t kc = GEMM_MIN(GEMM_F32_KC, k_dim - pc);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (m_dim * n_dim * sizeof(float) > \
                                                  GEMM_OMP_THRESHOLD)
#endif
    for (size_t ic = 0; ic < m_dim; ic += GEMM_F32_MC) {
      size_t mc = GEMM_MIN(GEMM_F32_MC, m_dim - ic);
      size_t jr = 0;
      for (; jr + GEMM_F32_NR <= n_dim; jr += GEMM_F32_NR) {
        size_t ir = 0;
        for (; ir + GEMM_F32_MR <= mc; ir += GEMM_F32_MR)
          gemm_ukernel_f32_6x16(a + (ic + ir) * rsa + pc * csa,
                                b + pc * rsb + jr, out + (ic + ir) * rso + jr,
                                kc, rsa, csa, rsb, rso);
        if (ir < mc)
          gemm_edge_f32(a + (ic + ir) * rsa + pc * csa, b + pc * rsb + jr,
                        out + (ic + ir) * rso + jr, mc - ir, GEMM_F32_NR, kc,
                        rsa, csa, rsb, rso);
      }
      if (jr < n_dim)
        gemm_edge_f32(a + ic * rsa + pc * csa, b + pc * rsb + jr,
                      out + ic * rso + jr, mc, n_dim - jr, kc, rsa, csa, rsb,
                      rso);
    }
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   Float64: 6×8 micro-kernel  (12 acc + 1 A broadcast + 2 B loads = 15 YMM)
   ═══════════════════════════════════════════════════════════════════════════
 */

static inline void gemm_ukernel_f64_6x8(const double *a, const double *b,
                                        double *c, size_t kc, intptr_t rsa,
                                        intptr_t csa, intptr_t rsb,
                                        intptr_t rso) {
  __m256d c00 = _mm256_loadu_pd(c), c01 = _mm256_loadu_pd(c + 4);
  __m256d c10 = _mm256_loadu_pd(c + rso), c11 = _mm256_loadu_pd(c + rso + 4);
  __m256d c20 = _mm256_loadu_pd(c + 2 * rso),
          c21 = _mm256_loadu_pd(c + 2 * rso + 4);
  __m256d c30 = _mm256_loadu_pd(c + 3 * rso),
          c31 = _mm256_loadu_pd(c + 3 * rso + 4);
  __m256d c40 = _mm256_loadu_pd(c + 4 * rso),
          c41 = _mm256_loadu_pd(c + 4 * rso + 4);
  __m256d c50 = _mm256_loadu_pd(c + 5 * rso),
          c51 = _mm256_loadu_pd(c + 5 * rso + 4);

  for (size_t p = 0; p < kc; p++) {
    const double *bp = b + p * rsb;
    _mm_prefetch((const char *)(bp + rsb), _MM_HINT_T0);
    _mm_prefetch((const char *)(bp + rsb + 4), _MM_HINT_T0);
    __m256d b0 = _mm256_loadu_pd(bp);
    __m256d b1 = _mm256_loadu_pd(bp + 4);

    __m256d av;
    av = _mm256_broadcast_sd(a + 0 * rsa + p * csa);
    c00 = _mm256_fmadd_pd(av, b0, c00);
    c01 = _mm256_fmadd_pd(av, b1, c01);

    av = _mm256_broadcast_sd(a + 1 * rsa + p * csa);
    c10 = _mm256_fmadd_pd(av, b0, c10);
    c11 = _mm256_fmadd_pd(av, b1, c11);

    av = _mm256_broadcast_sd(a + 2 * rsa + p * csa);
    c20 = _mm256_fmadd_pd(av, b0, c20);
    c21 = _mm256_fmadd_pd(av, b1, c21);

    av = _mm256_broadcast_sd(a + 3 * rsa + p * csa);
    c30 = _mm256_fmadd_pd(av, b0, c30);
    c31 = _mm256_fmadd_pd(av, b1, c31);

    av = _mm256_broadcast_sd(a + 4 * rsa + p * csa);
    c40 = _mm256_fmadd_pd(av, b0, c40);
    c41 = _mm256_fmadd_pd(av, b1, c41);

    av = _mm256_broadcast_sd(a + 5 * rsa + p * csa);
    c50 = _mm256_fmadd_pd(av, b0, c50);
    c51 = _mm256_fmadd_pd(av, b1, c51);
  }

  _mm256_storeu_pd(c, c00);
  _mm256_storeu_pd(c + 4, c01);
  _mm256_storeu_pd(c + rso, c10);
  _mm256_storeu_pd(c + rso + 4, c11);
  _mm256_storeu_pd(c + 2 * rso, c20);
  _mm256_storeu_pd(c + 2 * rso + 4, c21);
  _mm256_storeu_pd(c + 3 * rso, c30);
  _mm256_storeu_pd(c + 3 * rso + 4, c31);
  _mm256_storeu_pd(c + 4 * rso, c40);
  _mm256_storeu_pd(c + 4 * rso + 4, c41);
  _mm256_storeu_pd(c + 5 * rso, c50);
  _mm256_storeu_pd(c + 5 * rso + 4, c51);
}

static inline void gemm_edge_f64(const double *a, const double *b, double *c,
                                 size_t mr, size_t nr, size_t kc, intptr_t rsa,
                                 intptr_t csa, intptr_t rsb, intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
    for (size_t p = 0; p < kc; p++) {
      double aip = a[i * rsa + p * csa];
      __m256d va = _mm256_set1_pd(aip);
      const double *brow = b + p * rsb;
      double *crow = c + i * rso;
      size_t j = 0;
      for (; j + 4 <= nr; j += 4) {
        __m256d vo = _mm256_loadu_pd(crow + j);
        vo = _mm256_fmadd_pd(va, _mm256_loadu_pd(brow + j), vo);
        _mm256_storeu_pd(crow + j, vo);
      }
      for (; j < nr; j++)
        crow[j] += aip * brow[j];
    }
  }
}

static inline void gemm_f64_avx2(const double *a, const double *b, double *out,
                                 size_t m_dim, size_t k_dim, size_t n_dim,
                                 intptr_t rsa, intptr_t csa, intptr_t rsb,
                                 intptr_t rso) {
  for (size_t i = 0; i < m_dim; i++)
    memset(out + i * rso, 0, n_dim * sizeof(double));

  for (size_t pc = 0; pc < k_dim; pc += GEMM_F64_KC) {
    size_t kc = GEMM_MIN(GEMM_F64_KC, k_dim - pc);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (m_dim * n_dim * sizeof(double) > \
                                                  GEMM_OMP_THRESHOLD)
#endif
    for (size_t ic = 0; ic < m_dim; ic += GEMM_F64_MC) {
      size_t mc = GEMM_MIN(GEMM_F64_MC, m_dim - ic);
      size_t jr = 0;
      for (; jr + GEMM_F64_NR <= n_dim; jr += GEMM_F64_NR) {
        size_t ir = 0;
        for (; ir + GEMM_F64_MR <= mc; ir += GEMM_F64_MR)
          gemm_ukernel_f64_6x8(a + (ic + ir) * rsa + pc * csa,
                               b + pc * rsb + jr, out + (ic + ir) * rso + jr,
                               kc, rsa, csa, rsb, rso);
        if (ir < mc)
          gemm_edge_f64(a + (ic + ir) * rsa + pc * csa, b + pc * rsb + jr,
                        out + (ic + ir) * rso + jr, mc - ir, GEMM_F64_NR, kc,
                        rsa, csa, rsb, rso);
      }
      if (jr < n_dim)
        gemm_edge_f64(a + ic * rsa + pc * csa, b + pc * rsb + jr,
                      out + ic * rso + jr, mc, n_dim - jr, kc, rsa, csa, rsb,
                      rso);
    }
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   Int32/Uint32: 6×16 micro-kernel  (mullo_epi32 + add_epi32)
   mullo_epi32 produces identical low 32 bits for signed and unsigned.
   ═══════════════════════════════════════════════════════════════════════════
 */

static inline void gemm_ukernel_i32_6x16(const int32_t *a, const int32_t *b,
                                         int32_t *c, size_t kc, intptr_t rsa,
                                         intptr_t csa, intptr_t rsb,
                                         intptr_t rso) {
  __m256i c00 = _mm256_loadu_si256((__m256i *)(c)),
          c01 = _mm256_loadu_si256((__m256i *)(c + 8));
  __m256i c10 = _mm256_loadu_si256((__m256i *)(c + rso)),
          c11 = _mm256_loadu_si256((__m256i *)(c + rso + 8));
  __m256i c20 = _mm256_loadu_si256((__m256i *)(c + 2 * rso)),
          c21 = _mm256_loadu_si256((__m256i *)(c + 2 * rso + 8));
  __m256i c30 = _mm256_loadu_si256((__m256i *)(c + 3 * rso)),
          c31 = _mm256_loadu_si256((__m256i *)(c + 3 * rso + 8));
  __m256i c40 = _mm256_loadu_si256((__m256i *)(c + 4 * rso)),
          c41 = _mm256_loadu_si256((__m256i *)(c + 4 * rso + 8));
  __m256i c50 = _mm256_loadu_si256((__m256i *)(c + 5 * rso)),
          c51 = _mm256_loadu_si256((__m256i *)(c + 5 * rso + 8));

  for (size_t p = 0; p < kc; p++) {
    const int32_t *bp = b + p * rsb;
    _mm_prefetch((const char *)(bp + rsb), _MM_HINT_T0);
    _mm_prefetch((const char *)(bp + rsb + 8), _MM_HINT_T0);
    __m256i b0 = _mm256_loadu_si256((__m256i *)bp),
            b1 = _mm256_loadu_si256((__m256i *)(bp + 8));
    __m256i av;
    av = _mm256_set1_epi32(a[0 * rsa + p * csa]);
    c00 = _mm256_add_epi32(c00, _mm256_mullo_epi32(av, b0));
    c01 = _mm256_add_epi32(c01, _mm256_mullo_epi32(av, b1));
    av = _mm256_set1_epi32(a[1 * rsa + p * csa]);
    c10 = _mm256_add_epi32(c10, _mm256_mullo_epi32(av, b0));
    c11 = _mm256_add_epi32(c11, _mm256_mullo_epi32(av, b1));
    av = _mm256_set1_epi32(a[2 * rsa + p * csa]);
    c20 = _mm256_add_epi32(c20, _mm256_mullo_epi32(av, b0));
    c21 = _mm256_add_epi32(c21, _mm256_mullo_epi32(av, b1));
    av = _mm256_set1_epi32(a[3 * rsa + p * csa]);
    c30 = _mm256_add_epi32(c30, _mm256_mullo_epi32(av, b0));
    c31 = _mm256_add_epi32(c31, _mm256_mullo_epi32(av, b1));
    av = _mm256_set1_epi32(a[4 * rsa + p * csa]);
    c40 = _mm256_add_epi32(c40, _mm256_mullo_epi32(av, b0));
    c41 = _mm256_add_epi32(c41, _mm256_mullo_epi32(av, b1));
    av = _mm256_set1_epi32(a[5 * rsa + p * csa]);
    c50 = _mm256_add_epi32(c50, _mm256_mullo_epi32(av, b0));
    c51 = _mm256_add_epi32(c51, _mm256_mullo_epi32(av, b1));
  }

  _mm256_storeu_si256((__m256i *)(c), c00);
  _mm256_storeu_si256((__m256i *)(c + 8), c01);
  _mm256_storeu_si256((__m256i *)(c + rso), c10);
  _mm256_storeu_si256((__m256i *)(c + rso + 8), c11);
  _mm256_storeu_si256((__m256i *)(c + 2 * rso), c20);
  _mm256_storeu_si256((__m256i *)(c + 2 * rso + 8), c21);
  _mm256_storeu_si256((__m256i *)(c + 3 * rso), c30);
  _mm256_storeu_si256((__m256i *)(c + 3 * rso + 8), c31);
  _mm256_storeu_si256((__m256i *)(c + 4 * rso), c40);
  _mm256_storeu_si256((__m256i *)(c + 4 * rso + 8), c41);
  _mm256_storeu_si256((__m256i *)(c + 5 * rso), c50);
  _mm256_storeu_si256((__m256i *)(c + 5 * rso + 8), c51);
}

static inline void gemm_edge_i32(const int32_t *a, const int32_t *b, int32_t *c,
                                 size_t mr, size_t nr, size_t kc, intptr_t rsa,
                                 intptr_t csa, intptr_t rsb, intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
    for (size_t p = 0; p < kc; p++) {
      int32_t aip = a[i * rsa + p * csa];
      __m256i va = _mm256_set1_epi32(aip);
      const int32_t *brow = b + p * rsb;
      int32_t *crow = c + i * rso;
      size_t j = 0;
      for (; j + 8 <= nr; j += 8) {
        __m256i vo = _mm256_loadu_si256((__m256i *)(crow + j));
        vo = _mm256_add_epi32(
            vo,
            _mm256_mullo_epi32(va, _mm256_loadu_si256((__m256i *)(brow + j))));
        _mm256_storeu_si256((__m256i *)(crow + j), vo);
      }
      for (; j < nr; j++)
        crow[j] += aip * brow[j];
    }
  }
}

static inline void gemm_i32_avx2(const int32_t *a, const int32_t *b,
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
          gemm_ukernel_i32_6x16(a + (ic + ir) * rsa + pc * csa,
                                b + pc * rsb + jr, out + (ic + ir) * rso + jr,
                                kc, rsa, csa, rsb, rso);
        if (ir < mc)
          gemm_edge_i32(a + (ic + ir) * rsa + pc * csa, b + pc * rsb + jr,
                        out + (ic + ir) * rso + jr, mc - ir, GEMM_I32_NR, kc,
                        rsa, csa, rsb, rso);
      }
      if (jr < n_dim)
        gemm_edge_i32(a + ic * rsa + pc * csa, b + pc * rsb + jr,
                      out + ic * rso + jr, mc, n_dim - jr, kc, rsa, csa, rsb,
                      rso);
    }
  }
}

/* Uint32: identical bit-level operations as int32 */
static inline void gemm_u32_avx2(const uint32_t *a, const uint32_t *b,
                                 uint32_t *out, size_t m_dim, size_t k_dim,
                                 size_t n_dim, intptr_t rsa, intptr_t csa,
                                 intptr_t rsb, intptr_t rso) {
  gemm_i32_avx2((const int32_t *)a, (const int32_t *)b, (int32_t *)out, m_dim,
                k_dim, n_dim, rsa, csa, rsb, rso);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Int16/Uint16: 6×32 micro-kernel  (mullo_epi16 + add_epi16, 16 elem/reg)
   Same-width accumulation — matches the i32 overflow trade-off.
   ═══════════════════════════════════════════════════════════════════════════
 */

static inline void gemm_ukernel_i16_6x32(const int16_t *a, const int16_t *b,
                                         int16_t *c, size_t kc, intptr_t rsa,
                                         intptr_t csa, intptr_t rsb,
                                         intptr_t rso) {
  __m256i c00 = _mm256_loadu_si256((__m256i *)(c)),
          c01 = _mm256_loadu_si256((__m256i *)(c + 16));
  __m256i c10 = _mm256_loadu_si256((__m256i *)(c + rso)),
          c11 = _mm256_loadu_si256((__m256i *)(c + rso + 16));
  __m256i c20 = _mm256_loadu_si256((__m256i *)(c + 2 * rso)),
          c21 = _mm256_loadu_si256((__m256i *)(c + 2 * rso + 16));
  __m256i c30 = _mm256_loadu_si256((__m256i *)(c + 3 * rso)),
          c31 = _mm256_loadu_si256((__m256i *)(c + 3 * rso + 16));
  __m256i c40 = _mm256_loadu_si256((__m256i *)(c + 4 * rso)),
          c41 = _mm256_loadu_si256((__m256i *)(c + 4 * rso + 16));
  __m256i c50 = _mm256_loadu_si256((__m256i *)(c + 5 * rso)),
          c51 = _mm256_loadu_si256((__m256i *)(c + 5 * rso + 16));

  for (size_t p = 0; p < kc; p++) {
    const int16_t *bp = b + p * rsb;
    _mm_prefetch((const char *)(bp + rsb), _MM_HINT_T0);
    _mm_prefetch((const char *)(bp + rsb + 16), _MM_HINT_T0);
    __m256i b0 = _mm256_loadu_si256((__m256i *)bp),
            b1 = _mm256_loadu_si256((__m256i *)(bp + 16));
    __m256i av;
    av = _mm256_set1_epi16(a[0 * rsa + p * csa]);
    c00 = _mm256_add_epi16(c00, _mm256_mullo_epi16(av, b0));
    c01 = _mm256_add_epi16(c01, _mm256_mullo_epi16(av, b1));
    av = _mm256_set1_epi16(a[1 * rsa + p * csa]);
    c10 = _mm256_add_epi16(c10, _mm256_mullo_epi16(av, b0));
    c11 = _mm256_add_epi16(c11, _mm256_mullo_epi16(av, b1));
    av = _mm256_set1_epi16(a[2 * rsa + p * csa]);
    c20 = _mm256_add_epi16(c20, _mm256_mullo_epi16(av, b0));
    c21 = _mm256_add_epi16(c21, _mm256_mullo_epi16(av, b1));
    av = _mm256_set1_epi16(a[3 * rsa + p * csa]);
    c30 = _mm256_add_epi16(c30, _mm256_mullo_epi16(av, b0));
    c31 = _mm256_add_epi16(c31, _mm256_mullo_epi16(av, b1));
    av = _mm256_set1_epi16(a[4 * rsa + p * csa]);
    c40 = _mm256_add_epi16(c40, _mm256_mullo_epi16(av, b0));
    c41 = _mm256_add_epi16(c41, _mm256_mullo_epi16(av, b1));
    av = _mm256_set1_epi16(a[5 * rsa + p * csa]);
    c50 = _mm256_add_epi16(c50, _mm256_mullo_epi16(av, b0));
    c51 = _mm256_add_epi16(c51, _mm256_mullo_epi16(av, b1));
  }

  _mm256_storeu_si256((__m256i *)(c), c00);
  _mm256_storeu_si256((__m256i *)(c + 16), c01);
  _mm256_storeu_si256((__m256i *)(c + rso), c10);
  _mm256_storeu_si256((__m256i *)(c + rso + 16), c11);
  _mm256_storeu_si256((__m256i *)(c + 2 * rso), c20);
  _mm256_storeu_si256((__m256i *)(c + 2 * rso + 16), c21);
  _mm256_storeu_si256((__m256i *)(c + 3 * rso), c30);
  _mm256_storeu_si256((__m256i *)(c + 3 * rso + 16), c31);
  _mm256_storeu_si256((__m256i *)(c + 4 * rso), c40);
  _mm256_storeu_si256((__m256i *)(c + 4 * rso + 16), c41);
  _mm256_storeu_si256((__m256i *)(c + 5 * rso), c50);
  _mm256_storeu_si256((__m256i *)(c + 5 * rso + 16), c51);
}

static inline void gemm_edge_i16(const int16_t *a, const int16_t *b, int16_t *c,
                                 size_t mr, size_t nr, size_t kc, intptr_t rsa,
                                 intptr_t csa, intptr_t rsb, intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
    for (size_t p = 0; p < kc; p++) {
      int16_t aip = a[i * rsa + p * csa];
      __m256i va = _mm256_set1_epi16(aip);
      const int16_t *brow = b + p * rsb;
      int16_t *crow = c + i * rso;
      size_t j = 0;
      for (; j + 16 <= nr; j += 16) {
        __m256i vo = _mm256_loadu_si256((__m256i *)(crow + j));
        vo = _mm256_add_epi16(
            vo,
            _mm256_mullo_epi16(va, _mm256_loadu_si256((__m256i *)(brow + j))));
        _mm256_storeu_si256((__m256i *)(crow + j), vo);
      }
      for (; j < nr; j++)
        crow[j] = (int16_t)(crow[j] + aip * brow[j]);
    }
  }
}

static inline void gemm_i16_avx2(const int16_t *a, const int16_t *b,
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
          gemm_ukernel_i16_6x32(a + (ic + ir) * rsa + pc * csa,
                                b + pc * rsb + jr, out + (ic + ir) * rso + jr,
                                kc, rsa, csa, rsb, rso);
        if (ir < mc)
          gemm_edge_i16(a + (ic + ir) * rsa + pc * csa, b + pc * rsb + jr,
                        out + (ic + ir) * rso + jr, mc - ir, GEMM_I16_NR, kc,
                        rsa, csa, rsb, rso);
      }
      if (jr < n_dim)
        gemm_edge_i16(a + ic * rsa + pc * csa, b + pc * rsb + jr,
                      out + ic * rso + jr, mc, n_dim - jr, kc, rsa, csa, rsb,
                      rso);
    }
  }
}

static inline void gemm_u16_avx2(const uint16_t *a, const uint16_t *b,
                                 uint16_t *out, size_t m_dim, size_t k_dim,
                                 size_t n_dim, intptr_t rsa, intptr_t csa,
                                 intptr_t rsb, intptr_t rso) {
  gemm_i16_avx2((const int16_t *)a, (const int16_t *)b, (int16_t *)out, m_dim,
                k_dim, n_dim, rsa, csa, rsb, rso);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Int64/Uint64: 6×8 micro-kernel  (emulated 64-bit multiply via mul_epu32)
   No native mullo_epi64 in AVX2 — emulate: a*b = a_lo*b_lo + cross<<32
   ═══════════════════════════════════════════════════════════════════════════
 */

static inline __m256i gemm_mullo_epi64(__m256i a, __m256i b) {
  __m256i a_hi = _mm256_srli_epi64(a, 32);
  __m256i b_hi = _mm256_srli_epi64(b, 32);
  __m256i lo_lo = _mm256_mul_epu32(a, b);
  __m256i cross =
      _mm256_add_epi64(_mm256_mul_epu32(a, b_hi), _mm256_mul_epu32(a_hi, b));
  return _mm256_add_epi64(lo_lo, _mm256_slli_epi64(cross, 32));
}

static inline void gemm_ukernel_i64_6x8(const int64_t *a, const int64_t *b,
                                        int64_t *c, size_t kc, intptr_t rsa,
                                        intptr_t csa, intptr_t rsb,
                                        intptr_t rso) {
  __m256i c00 = _mm256_loadu_si256((__m256i *)(c)),
          c01 = _mm256_loadu_si256((__m256i *)(c + 4));
  __m256i c10 = _mm256_loadu_si256((__m256i *)(c + rso)),
          c11 = _mm256_loadu_si256((__m256i *)(c + rso + 4));
  __m256i c20 = _mm256_loadu_si256((__m256i *)(c + 2 * rso)),
          c21 = _mm256_loadu_si256((__m256i *)(c + 2 * rso + 4));
  __m256i c30 = _mm256_loadu_si256((__m256i *)(c + 3 * rso)),
          c31 = _mm256_loadu_si256((__m256i *)(c + 3 * rso + 4));
  __m256i c40 = _mm256_loadu_si256((__m256i *)(c + 4 * rso)),
          c41 = _mm256_loadu_si256((__m256i *)(c + 4 * rso + 4));
  __m256i c50 = _mm256_loadu_si256((__m256i *)(c + 5 * rso)),
          c51 = _mm256_loadu_si256((__m256i *)(c + 5 * rso + 4));

  for (size_t p = 0; p < kc; p++) {
    const int64_t *bp = b + p * rsb;
    _mm_prefetch((const char *)(bp + rsb), _MM_HINT_T0);
    __m256i b0 = _mm256_loadu_si256((__m256i *)bp),
            b1 = _mm256_loadu_si256((__m256i *)(bp + 4));
    __m256i av;
    av = _mm256_set1_epi64x(a[0 * rsa + p * csa]);
    c00 = _mm256_add_epi64(c00, gemm_mullo_epi64(av, b0));
    c01 = _mm256_add_epi64(c01, gemm_mullo_epi64(av, b1));
    av = _mm256_set1_epi64x(a[1 * rsa + p * csa]);
    c10 = _mm256_add_epi64(c10, gemm_mullo_epi64(av, b0));
    c11 = _mm256_add_epi64(c11, gemm_mullo_epi64(av, b1));
    av = _mm256_set1_epi64x(a[2 * rsa + p * csa]);
    c20 = _mm256_add_epi64(c20, gemm_mullo_epi64(av, b0));
    c21 = _mm256_add_epi64(c21, gemm_mullo_epi64(av, b1));
    av = _mm256_set1_epi64x(a[3 * rsa + p * csa]);
    c30 = _mm256_add_epi64(c30, gemm_mullo_epi64(av, b0));
    c31 = _mm256_add_epi64(c31, gemm_mullo_epi64(av, b1));
    av = _mm256_set1_epi64x(a[4 * rsa + p * csa]);
    c40 = _mm256_add_epi64(c40, gemm_mullo_epi64(av, b0));
    c41 = _mm256_add_epi64(c41, gemm_mullo_epi64(av, b1));
    av = _mm256_set1_epi64x(a[5 * rsa + p * csa]);
    c50 = _mm256_add_epi64(c50, gemm_mullo_epi64(av, b0));
    c51 = _mm256_add_epi64(c51, gemm_mullo_epi64(av, b1));
  }

  _mm256_storeu_si256((__m256i *)(c), c00);
  _mm256_storeu_si256((__m256i *)(c + 4), c01);
  _mm256_storeu_si256((__m256i *)(c + rso), c10);
  _mm256_storeu_si256((__m256i *)(c + rso + 4), c11);
  _mm256_storeu_si256((__m256i *)(c + 2 * rso), c20);
  _mm256_storeu_si256((__m256i *)(c + 2 * rso + 4), c21);
  _mm256_storeu_si256((__m256i *)(c + 3 * rso), c30);
  _mm256_storeu_si256((__m256i *)(c + 3 * rso + 4), c31);
  _mm256_storeu_si256((__m256i *)(c + 4 * rso), c40);
  _mm256_storeu_si256((__m256i *)(c + 4 * rso + 4), c41);
  _mm256_storeu_si256((__m256i *)(c + 5 * rso), c50);
  _mm256_storeu_si256((__m256i *)(c + 5 * rso + 4), c51);
}

static inline void gemm_edge_i64(const int64_t *a, const int64_t *b, int64_t *c,
                                 size_t mr, size_t nr, size_t kc, intptr_t rsa,
                                 intptr_t csa, intptr_t rsb, intptr_t rso) {
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

static inline void gemm_i64_avx2(const int64_t *a, const int64_t *b,
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
          gemm_ukernel_i64_6x8(a + (ic + ir) * rsa + pc * csa,
                               b + pc * rsb + jr, out + (ic + ir) * rso + jr,
                               kc, rsa, csa, rsb, rso);
        if (ir < mc)
          gemm_edge_i64(a + (ic + ir) * rsa + pc * csa, b + pc * rsb + jr,
                        out + (ic + ir) * rso + jr, mc - ir, GEMM_I64_NR, kc,
                        rsa, csa, rsb, rso);
      }
      if (jr < n_dim)
        gemm_edge_i64(a + ic * rsa + pc * csa, b + pc * rsb + jr,
                      out + ic * rso + jr, mc, n_dim - jr, kc, rsa, csa, rsb,
                      rso);
    }
  }
}

static inline void gemm_u64_avx2(const uint64_t *a, const uint64_t *b,
                                 uint64_t *out, size_t m_dim, size_t k_dim,
                                 size_t n_dim, intptr_t rsa, intptr_t csa,
                                 intptr_t rsb, intptr_t rso) {
  gemm_i64_avx2((const int64_t *)a, (const int64_t *)b, (int64_t *)out, m_dim,
                k_dim, n_dim, rsa, csa, rsb, rso);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Int8/Uint8: 6×8 promoted micro-kernel  (widen to i32, full-K accumulation)
   Uses saturation packing for stores to match scalar behavior more efficiently.
   ═══════════════════════════════════════════════════════════════════════════
 */

static inline void gemm_ukernel_i8_6x8(const int8_t *a, const int8_t *b,
                                       int8_t *c, size_t k_dim, intptr_t rsa,
                                       intptr_t csa, intptr_t rsb,
                                       intptr_t rso) {
  __m256i c00 = _mm256_setzero_si256(), c10 = _mm256_setzero_si256(),
          c20 = _mm256_setzero_si256(), c30 = _mm256_setzero_si256(),
          c40 = _mm256_setzero_si256(), c50 = _mm256_setzero_si256();

  for (size_t p = 0; p < k_dim; p++) {
    const int8_t *bp = b + p * rsb;
    __m256i b0 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i const *)bp));
    __m256i av;
    av = _mm256_set1_epi32((int32_t)a[0 * rsa + p * csa]);
    c00 = _mm256_add_epi32(c00, _mm256_mullo_epi32(av, b0));
    av = _mm256_set1_epi32((int32_t)a[1 * rsa + p * csa]);
    c10 = _mm256_add_epi32(c10, _mm256_mullo_epi32(av, b0));
    av = _mm256_set1_epi32((int32_t)a[2 * rsa + p * csa]);
    c20 = _mm256_add_epi32(c20, _mm256_mullo_epi32(av, b0));
    av = _mm256_set1_epi32((int32_t)a[3 * rsa + p * csa]);
    c30 = _mm256_add_epi32(c30, _mm256_mullo_epi32(av, b0));
    av = _mm256_set1_epi32((int32_t)a[4 * rsa + p * csa]);
    c40 = _mm256_add_epi32(c40, _mm256_mullo_epi32(av, b0));
    av = _mm256_set1_epi32((int32_t)a[5 * rsa + p * csa]);
    c50 = _mm256_add_epi32(c50, _mm256_mullo_epi32(av, b0));
  }

#define GEMM_STORE_I8_ROW(acc, row)                          \
  do {                                                       \
    __m128i lo = _mm256_castsi256_si128(acc);                \
    __m128i hi = _mm256_extracti128_si256(acc, 1);           \
    __m128i p16 = _mm_packs_epi32(lo, hi);                   \
    __m128i p8 = _mm_packs_epi16(p16, p16);                  \
    *((int64_t *)(c + (row) * rso)) = _mm_cvtsi128_si64(p8); \
  } while (0)
  GEMM_STORE_I8_ROW(c00, 0);
  GEMM_STORE_I8_ROW(c10, 1);
  GEMM_STORE_I8_ROW(c20, 2);
  GEMM_STORE_I8_ROW(c30, 3);
  GEMM_STORE_I8_ROW(c40, 4);
  GEMM_STORE_I8_ROW(c50, 5);
#undef GEMM_STORE_I8_ROW
}

static inline void gemm_edge_i8(const int8_t *a, const int8_t *b, int8_t *c,
                                size_t mr, size_t nr, size_t k_dim,
                                intptr_t rsa, intptr_t csa, intptr_t rsb,
                                intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
    for (size_t j = 0; j < nr; j++) {
      int32_t acc = 0;
      for (size_t p = 0; p < k_dim; p++)
        acc += (int32_t)a[i * rsa + p * csa] * (int32_t)b[p * rsb + j];
      c[i * rso + j] = (int8_t)acc;
    }
  }
}

static inline void gemm_i8_avx2(const int8_t *a, const int8_t *b, int8_t *out,
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
        gemm_edge_i8(a + (ic + ir) * rsa, b + jr, out + (ic + ir) * rso + jr,
                     mc - ir, GEMM_I8_NR, k_dim, rsa, csa, rsb, rso);
    }
    if (jr < n_dim)
      gemm_edge_i8(a + ic * rsa, b + jr, out + ic * rso + jr, mc, n_dim - jr,
                   k_dim, rsa, csa, rsb, rso);
  }
}

static inline void gemm_ukernel_u8_6x8(const uint8_t *a, const uint8_t *b,
                                       uint8_t *c, size_t k_dim, intptr_t rsa,
                                       intptr_t csa, intptr_t rsb,
                                       intptr_t rso) {
  __m256i c00 = _mm256_setzero_si256(), c10 = _mm256_setzero_si256(),
          c20 = _mm256_setzero_si256(), c30 = _mm256_setzero_si256(),
          c40 = _mm256_setzero_si256(), c50 = _mm256_setzero_si256();

  for (size_t p = 0; p < k_dim; p++) {
    const uint8_t *bp = b + p * rsb;
    __m256i b0 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i const *)bp));
    __m256i av;
    av = _mm256_set1_epi32((int32_t)(uint32_t)a[0 * rsa + p * csa]);
    c00 = _mm256_add_epi32(c00, _mm256_mullo_epi32(av, b0));
    av = _mm256_set1_epi32((int32_t)(uint32_t)a[1 * rsa + p * csa]);
    c10 = _mm256_add_epi32(c10, _mm256_mullo_epi32(av, b0));
    av = _mm256_set1_epi32((int32_t)(uint32_t)a[2 * rsa + p * csa]);
    c20 = _mm256_add_epi32(c20, _mm256_mullo_epi32(av, b0));
    av = _mm256_set1_epi32((int32_t)(uint32_t)a[3 * rsa + p * csa]);
    c30 = _mm256_add_epi32(c30, _mm256_mullo_epi32(av, b0));
    av = _mm256_set1_epi32((int32_t)(uint32_t)a[4 * rsa + p * csa]);
    c40 = _mm256_add_epi32(c40, _mm256_mullo_epi32(av, b0));
    av = _mm256_set1_epi32((int32_t)(uint32_t)a[5 * rsa + p * csa]);
    c50 = _mm256_add_epi32(c50, _mm256_mullo_epi32(av, b0));
  }

#define GEMM_STORE_U8_ROW(acc, row)                           \
  do {                                                        \
    __m128i lo = _mm256_castsi256_si128(acc);                 \
    __m128i hi = _mm256_extracti128_si256(acc, 1);            \
    __m128i p16 = _mm_packus_epi32(lo, hi);                   \
    __m128i p8 = _mm_packus_epi16(p16, p16);                  \
    *((uint64_t *)(c + (row) * rso)) = _mm_cvtsi128_si64(p8); \
  } while (0)
  GEMM_STORE_U8_ROW(c00, 0);
  GEMM_STORE_U8_ROW(c10, 1);
  GEMM_STORE_U8_ROW(c20, 2);
  GEMM_STORE_U8_ROW(c30, 3);
  GEMM_STORE_U8_ROW(c40, 4);
  GEMM_STORE_U8_ROW(c50, 5);
#undef GEMM_STORE_U8_ROW
}

static inline void gemm_edge_u8(const uint8_t *a, const uint8_t *b, uint8_t *c,
                                size_t mr, size_t nr, size_t k_dim,
                                intptr_t rsa, intptr_t csa, intptr_t rsb,
                                intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
    for (size_t j = 0; j < nr; j++) {
      uint32_t acc = 0;
      for (size_t p = 0; p < k_dim; p++)
        acc += (uint32_t)a[i * rsa + p * csa] * (uint32_t)b[p * rsb + j];
      c[i * rso + j] = (uint8_t)acc;
    }
  }
}

static inline void gemm_u8_avx2(const uint8_t *a, const uint8_t *b,
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
        gemm_edge_u8(a + (ic + ir) * rsa, b + jr, out + (ic + ir) * rso + jr,
                     mc - ir, GEMM_I8_NR, k_dim, rsa, csa, rsb, rso);
    }
    if (jr < n_dim)
      gemm_edge_u8(a + ic * rsa, b + jr, out + ic * rso + jr, mc, n_dim - jr,
                   k_dim, rsa, csa, rsb, rso);
  }
}

#undef GEMM_MIN

#endif
