#ifndef NUMC_GEMM_AVX2_H
#define NUMC_GEMM_AVX2_H

#include <immintrin.h>
#include <stdint.h>
#include <string.h>

static inline void gemm_f32_avx2(const float *fa, const float *fb, float *fo,
                                 size_t m_dim, size_t k_dim, size_t n_dim,
                                 intptr_t rsa, intptr_t csa, intptr_t rsb,
                                 intptr_t rso) {
  for (size_t i = 0; i < m_dim; i++)
    memset(fo + i * rso, 0, n_dim * sizeof(float));
  for (size_t i = 0; i < m_dim; i++) {
    for (size_t k = 0; k < k_dim; k++) {
      float a_ik = fa[i * rsa + k * csa];
      const float *b_row = fb + k * rsb;
      float *o_row = fo + i * rso;
      __m256 va = _mm256_set1_ps(a_ik);
      size_t j = 0;
      for (; j + 8 <= n_dim; j += 8) {
        __m256 vb = _mm256_loadu_ps(b_row + j);
        __m256 vo = _mm256_loadu_ps(o_row + j);
        vo = _mm256_fmadd_ps(va, vb, vo);
        _mm256_storeu_ps(o_row + j, vo);
      }
      for (; j < n_dim; j++)
        o_row[j] += a_ik * b_row[j];
    }
  }
}

static inline void gemm_f64_avx2(const double *da, const double *db,
                                  double *do_, size_t m_dim, size_t k_dim,
                                  size_t n_dim, intptr_t rsa, intptr_t csa,
                                  intptr_t rsb, intptr_t rso) {
  for (size_t i = 0; i < m_dim; i++)
    memset(do_ + i * rso, 0, n_dim * sizeof(double));
  for (size_t i = 0; i < m_dim; i++) {
    for (size_t k = 0; k < k_dim; k++) {
      double a_ik = da[i * rsa + k * csa];
      const double *b_row = db + k * rsb;
      double *o_row = do_ + i * rso;
      __m256d va = _mm256_set1_pd(a_ik);
      size_t j = 0;
      for (; j + 4 <= n_dim; j += 4) {
        __m256d vb = _mm256_loadu_pd(b_row + j);
        __m256d vo = _mm256_loadu_pd(o_row + j);
        vo = _mm256_fmadd_pd(va, vb, vo);
        _mm256_storeu_pd(o_row + j, vo);
      }
      for (; j < n_dim; j++)
        o_row[j] += a_ik * b_row[j];
    }
  }
}

#endif
