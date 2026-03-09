/**
 * @file blis_kernels.h
 * @brief BLIS wrappers for dot product, gemm, and batched gemm.
 *
 * Wraps BLIS sdotv/ddotv and sgemm/dgemm with stride-aware calling
 * conventions, plus a batched gemm helper for ND dot products.
 */
#ifndef NUMC_BLIS_KERNELS_H
#define NUMC_BLIS_KERNELS_H

#ifdef HAVE_BLAS

#include <blis.h>
#include <pthread.h>

/* ── Type definitions ────────────────────────────────────────────────── */

typedef void (*BlasDotKernel)(const NumcArray *a, const NumcArray *b,
                              NumcArray *out);
typedef void (*BlasGemmKernel)(const NumcArray *a, const NumcArray *b,
                               NumcArray *out, size_t M, size_t K, size_t N);
typedef void (*BlasBatchGemmKernel)(const char *a_data, const char *b_data,
                                    char *out_data, size_t M, size_t K,
                                    size_t N, intptr_t rsa, intptr_t csa,
                                    intptr_t rsb, intptr_t csb, intptr_t rso,
                                    intptr_t cso);

/* ── Dot kernels ─────────────────────────────────────────────────────── */

static void _blas_dot_f32(const NumcArray *a, const NumcArray *b,
                          NumcArray *out) {
  float rho = 0.0f;
  inc_t inca = (inc_t)(a->strides[0] / sizeof(float));
  inc_t incb = (inc_t)(b->strides[0] / sizeof(float));
  bli_sdotv(BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE, (dim_t)a->size,
            (float *)a->data, inca, (float *)b->data, incb, &rho);
  *(float *)out->data = rho;
}

static void _blas_dot_f64(const NumcArray *a, const NumcArray *b,
                          NumcArray *out) {
  double rho = 0.0;
  inc_t inca = (inc_t)(a->strides[0] / sizeof(double));
  inc_t incb = (inc_t)(b->strides[0] / sizeof(double));
  bli_ddotv(BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE, (dim_t)a->size,
            (double *)a->data, inca, (double *)b->data, incb, &rho);
  *(double *)out->data = rho;
}

/* ── GEMM kernels ────────────────────────────────────────────────────── */

static void _blas_gemm_f32(const NumcArray *a, const NumcArray *b,
                           NumcArray *out, size_t M, size_t K, size_t N) {
  float alpha = 1.0f, beta = 0.0f;

  inc_t rs_a =
      (inc_t)(a->dim > 1 ? a->strides[a->dim - 2] / sizeof(float) : (inc_t)K);
  inc_t cs_a = (inc_t)(a->strides[a->dim - 1] / sizeof(float));

  inc_t rs_b =
      (inc_t)(b->dim > 1 ? b->strides[b->dim - 2] / sizeof(float) : (inc_t)N);
  inc_t cs_b = (inc_t)(b->strides[b->dim - 1] / sizeof(float));

  inc_t rs_c = (inc_t)(out->dim > 1 ? out->strides[out->dim - 2] / sizeof(float)
                                    : (inc_t)N);
  inc_t cs_c = (inc_t)(out->strides[out->dim - 1] / sizeof(float));

  bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, (dim_t)M, (dim_t)N, (dim_t)K,
            &alpha, (float *)a->data, rs_a, cs_a, (float *)b->data, rs_b, cs_b,
            &beta, (float *)out->data, rs_c, cs_c);
}

static void _blas_gemm_f64(const NumcArray *a, const NumcArray *b,
                           NumcArray *out, size_t M, size_t K, size_t N) {
  double alpha = 1.0, beta = 0.0;

  inc_t rs_a =
      (inc_t)(a->dim > 1 ? a->strides[a->dim - 2] / sizeof(double) : (inc_t)K);
  inc_t cs_a = (inc_t)(a->strides[a->dim - 1] / sizeof(double));

  inc_t rs_b =
      (inc_t)(b->dim > 1 ? b->strides[b->dim - 2] / sizeof(double) : (inc_t)N);
  inc_t cs_b = (inc_t)(b->strides[b->dim - 1] / sizeof(double));

  inc_t rs_c =
      (inc_t)(out->dim > 1 ? out->strides[out->dim - 2] / sizeof(double)
                           : (inc_t)N);
  inc_t cs_c = (inc_t)(out->strides[out->dim - 1] / sizeof(double));

  bli_dgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, (dim_t)M, (dim_t)N, (dim_t)K,
            &alpha, (double *)a->data, rs_a, cs_a, (double *)b->data, rs_b,
            cs_b, &beta, (double *)out->data, rs_c, cs_c);
}

/* ── Batch GEMM kernels ──────────────────────────────────────────────── */

static void _blas_batch_gemm_f32(const char *a_data, const char *b_data,
                                 char *out_data, size_t M, size_t K, size_t N,
                                 intptr_t rsa, intptr_t csa, intptr_t rsb,
                                 intptr_t csb, intptr_t rso, intptr_t cso) {
  float alpha = 1.0f, beta = 0.0f;
  bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, (dim_t)M, (dim_t)N, (dim_t)K,
            &alpha, (float *)a_data, (inc_t)rsa, (inc_t)csa, (float *)b_data,
            (inc_t)rsb, (inc_t)csb, &beta, (float *)out_data, (inc_t)rso,
            (inc_t)cso);
}

static void _blas_batch_gemm_f64(const char *a_data, const char *b_data,
                                 char *out_data, size_t M, size_t K, size_t N,
                                 intptr_t rsa, intptr_t csa, intptr_t rsb,
                                 intptr_t csb, intptr_t rso, intptr_t cso) {
  double alpha = 1.0, beta = 0.0;
  bli_dgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, (dim_t)M, (dim_t)N, (dim_t)K,
            &alpha, (double *)a_data, (inc_t)rsa, (inc_t)csa, (double *)b_data,
            (inc_t)rsb, (inc_t)csb, &beta, (double *)out_data, (inc_t)rso,
            (inc_t)cso);
}

/* ── Dispatch tables ─────────────────────────────────────────────────── */

static const BlasDotKernel blas_dot_table[NUMC_DTYPE_COUNT] = {
    [NUMC_DTYPE_FLOAT32] = _blas_dot_f32,
    [NUMC_DTYPE_FLOAT64] = _blas_dot_f64,
};

static const BlasGemmKernel blas_gemm_table[NUMC_DTYPE_COUNT] = {
    [NUMC_DTYPE_FLOAT32] = _blas_gemm_f32,
    [NUMC_DTYPE_FLOAT64] = _blas_gemm_f64,
};

static const BlasBatchGemmKernel blas_batch_gemm_table[NUMC_DTYPE_COUNT] = {
    [NUMC_DTYPE_FLOAT32] = _blas_batch_gemm_f32,
    [NUMC_DTYPE_FLOAT64] = _blas_batch_gemm_f64,
};

#endif /* HAVE_BLAS */

#endif /* NUMC_BLIS_KERNELS_H */
