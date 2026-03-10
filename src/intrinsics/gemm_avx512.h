#ifndef NUMC_GEMM_AVX512_H
#define NUMC_GEMM_AVX512_H

#include <immintrin.h>
#include <stdint.h>
#include <string.h>

#define GEMM_MIN(a, b) ((a) < (b) ? (a) : (b))

#define GEMM_F32_MR 6
#define GEMM_F32_NR 32
#define GEMM_F32_MC 72
#define GEMM_F32_KC 256

#define GEMM_F64_MR 6
#define GEMM_F64_NR 16
#define GEMM_F64_MC 72
#define GEMM_F64_KC 128

#define GEMM_OMP_THRESHOLD (1 << 20)

static inline void gemm_ukernel_f32_6x32(const float *a, const float *b,
                                         float *c, size_t kc, intptr_t rsa,
                                         intptr_t csa, intptr_t rsb,
                                         intptr_t rso) {
  __m512 c00 = _mm512_loadu_ps(c), c01 = _mm512_loadu_ps(c + 16);
  __m512 c10 = _mm512_loadu_ps(c + rso), c11 = _mm512_loadu_ps(c + rso + 16);
  __m512 c20 = _mm512_loadu_ps(c + 2 * rso),
         c21 = _mm512_loadu_ps(c + 2 * rso + 16);
  __m512 c30 = _mm512_loadu_ps(c + 3 * rso),
         c31 = _mm512_loadu_ps(c + 3 * rso + 16);
  __m512 c40 = _mm512_loadu_ps(c + 4 * rso),
         c41 = _mm512_loadu_ps(c + 4 * rso + 16);
  __m512 c50 = _mm512_loadu_ps(c + 5 * rso),
         c51 = _mm512_loadu_ps(c + 5 * rso + 16);

  for (size_t p = 0; p < kc; p++) {
    const float *bp = b + p * rsb;
    __m512 b0 = _mm512_loadu_ps(bp);
    __m512 b1 = _mm512_loadu_ps(bp + 16);

    __m512 av;
    av = _mm512_set1_ps(a[0 * rsa + p * csa]);
    c00 = _mm512_fmadd_ps(av, b0, c00);
    c01 = _mm512_fmadd_ps(av, b1, c01);

    av = _mm512_set1_ps(a[1 * rsa + p * csa]);
    c10 = _mm512_fmadd_ps(av, b0, c10);
    c11 = _mm512_fmadd_ps(av, b1, c11);

    av = _mm512_set1_ps(a[2 * rsa + p * csa]);
    c20 = _mm512_fmadd_ps(av, b0, c20);
    c21 = _mm512_fmadd_ps(av, b1, c21);

    av = _mm512_set1_ps(a[3 * rsa + p * csa]);
    c30 = _mm512_fmadd_ps(av, b0, c30);
    c31 = _mm512_fmadd_ps(av, b1, c31);

    av = _mm512_set1_ps(a[4 * rsa + p * csa]);
    c40 = _mm512_fmadd_ps(av, b0, c40);
    c41 = _mm512_fmadd_ps(av, b1, c41);

    av = _mm512_set1_ps(a[5 * rsa + p * csa]);
    c50 = _mm512_fmadd_ps(av, b0, c50);
    c51 = _mm512_fmadd_ps(av, b1, c51);
  }

  _mm512_storeu_ps(c, c00);
  _mm512_storeu_ps(c + 16, c01);
  _mm512_storeu_ps(c + rso, c10);
  _mm512_storeu_ps(c + rso + 16, c11);
  _mm512_storeu_ps(c + 2 * rso, c20);
  _mm512_storeu_ps(c + 2 * rso + 16, c21);
  _mm512_storeu_ps(c + 3 * rso, c30);
  _mm512_storeu_ps(c + 3 * rso + 16, c31);
  _mm512_storeu_ps(c + 4 * rso, c40);
  _mm512_storeu_ps(c + 4 * rso + 16, c41);
  _mm512_storeu_ps(c + 5 * rso, c50);
  _mm512_storeu_ps(c + 5 * rso + 16, c51);
}

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
          gemm_ukernel_f32_6x32(a + (ic + ir) * rsa + pc * csa,
                                b + pc * rsb + jr, out + (ic + ir) * rso + jr,
                                kc, rsa, csa, rsb, rso);
        if (ir < mc)
          gemm_edge_f32_avx512(a + (ic + ir) * rsa + pc * csa,
                               b + pc * rsb + jr, out + (ic + ir) * rso + jr,
                               mc - ir, GEMM_F32_NR, kc, rsa, csa, rsb, rso);
      }
      if (jr < n_dim)
        gemm_edge_f32_avx512(a + ic * rsa + pc * csa, b + pc * rsb + jr,
                             out + ic * rso + jr, mc, n_dim - jr, kc, rsa, csa,
                             rsb, rso);
    }
  }
}

static inline void gemm_ukernel_f64_6x16(const double *a, const double *b,
                                         double *c, size_t kc, intptr_t rsa,
                                         intptr_t csa, intptr_t rsb,
                                         intptr_t rso) {
  __m512d c00 = _mm512_loadu_pd(c), c01 = _mm512_loadu_pd(c + 8);
  __m512d c10 = _mm512_loadu_pd(c + rso), c11 = _mm512_loadu_pd(c + rso + 8);
  __m512d c20 = _mm512_loadu_pd(c + 2 * rso),
          c21 = _mm512_loadu_pd(c + 2 * rso + 8);
  __m512d c30 = _mm512_loadu_pd(c + 3 * rso),
          c31 = _mm512_loadu_pd(c + 3 * rso + 8);
  __m512d c40 = _mm512_loadu_pd(c + 4 * rso),
          c41 = _mm512_loadu_pd(c + 4 * rso + 8);
  __m512d c50 = _mm512_loadu_pd(c + 5 * rso),
          c51 = _mm512_loadu_pd(c + 5 * rso + 8);

  for (size_t p = 0; p < kc; p++) {
    const double *bp = b + p * rsb;
    __m512d b0 = _mm512_loadu_pd(bp);
    __m512d b1 = _mm512_loadu_pd(bp + 8);

    __m512d av;
    av = _mm512_set1_pd(a[0 * rsa + p * csa]);
    c00 = _mm512_fmadd_pd(av, b0, c00);
    c01 = _mm512_fmadd_pd(av, b1, c01);

    av = _mm512_set1_pd(a[1 * rsa + p * csa]);
    c10 = _mm512_fmadd_pd(av, b0, c10);
    c11 = _mm512_fmadd_pd(av, b1, c11);

    av = _mm512_set1_pd(a[2 * rsa + p * csa]);
    c20 = _mm512_fmadd_pd(av, b0, c20);
    c21 = _mm512_fmadd_pd(av, b1, c21);

    av = _mm512_set1_pd(a[3 * rsa + p * csa]);
    c30 = _mm512_fmadd_pd(av, b0, c30);
    c31 = _mm512_fmadd_pd(av, b1, c31);

    av = _mm512_set1_pd(a[4 * rsa + p * csa]);
    c40 = _mm512_fmadd_pd(av, b0, c40);
    c41 = _mm512_fmadd_pd(av, b1, c41);

    av = _mm512_set1_pd(a[5 * rsa + p * csa]);
    c50 = _mm512_fmadd_pd(av, b0, c50);
    c51 = _mm512_fmadd_pd(av, b1, c51);
  }

  _mm512_storeu_pd(c, c00);
  _mm512_storeu_pd(c + 8, c01);
  _mm512_storeu_pd(c + rso, c10);
  _mm512_storeu_pd(c + rso + 8, c11);
  _mm512_storeu_pd(c + 2 * rso, c20);
  _mm512_storeu_pd(c + 2 * rso + 8, c21);
  _mm512_storeu_pd(c + 3 * rso, c30);
  _mm512_storeu_pd(c + 3 * rso + 8, c31);
  _mm512_storeu_pd(c + 4 * rso, c40);
  _mm512_storeu_pd(c + 4 * rso + 8, c41);
  _mm512_storeu_pd(c + 5 * rso, c50);
  _mm512_storeu_pd(c + 5 * rso + 8, c51);
}

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
          gemm_ukernel_f64_6x16(a + (ic + ir) * rsa + pc * csa,
                                b + pc * rsb + jr, out + (ic + ir) * rso + jr,
                                kc, rsa, csa, rsb, rso);
        if (ir < mc)
          gemm_edge_f64_avx512(a + (ic + ir) * rsa + pc * csa,
                               b + pc * rsb + jr, out + (ic + ir) * rso + jr,
                               mc - ir, GEMM_F64_NR, kc, rsa, csa, rsb, rso);
      }
      if (jr < n_dim)
        gemm_edge_f64_avx512(a + ic * rsa + pc * csa, b + pc * rsb + jr,
                             out + ic * rso + jr, mc, n_dim - jr, kc, rsa, csa,
                             rsb, rso);
    }
  }
}

#undef GEMM_MIN

#endif
