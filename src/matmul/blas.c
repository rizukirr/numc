#ifdef HAVE_BLAS

#include "internal.h"
#include <blis/blis.h>

void _matmul_blas_f32(const struct NumcArray *a, const struct NumcArray *b,
                      struct NumcArray *out) {
  blasint M = (blasint)a->shape[0];
  blasint N = (blasint)a->shape[1];
  blasint K = (blasint)b->shape[1];
  blasint lda = (blasint)a->stride[0] / sizeof(float);
  blasint ldb = (blasint)b->stride[0] / sizeof(float);
  blasint ldc = (blasint)N;

  cblas_sgemm(CblasRowMajor, CBlasNoTrans, CBlasNoTrans, M, N, K, 1.0f,
              (const float *)a->data, lda, (const float *)b->data, ldb, 0.0f,
              (float *)out->data, ldc);
}

#endif
