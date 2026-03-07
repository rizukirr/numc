#include "../arch_dispatch.h"
#include "dispatch.h"
#include "numc/dtype.h"
#include <numc/math.h>
#include <string.h>

#if NUMC_HAVE_AVX2
#include "intrinsics/dot_avx2.h"
#include "intrinsics/gemm_avx2.h"
#endif

/* ── Dot product reduction kernels ─────────────────────────────────── */

#define STAMP_DOT(TE, CT) \
  DEFINE_BINARY_REDUCTION_KERNEL(dot, TE, CT, 0, acc + (val_a * val_b), +)
GENERATE_INT8_INT16_NUMC_TYPES(STAMP_DOT)
GENERATE_INT32_NUMC_TYPES(STAMP_DOT)
STAMP_DOT(NUMC_DTYPE_INT64, NUMC_INT64)
STAMP_DOT(NUMC_DTYPE_UINT64, NUMC_UINT64)
STAMP_DOT(NUMC_DTYPE_FLOAT32, NUMC_FLOAT32)
STAMP_DOT(NUMC_DTYPE_FLOAT64, NUMC_FLOAT64)
#undef STAMP_DOT

/* ── Dispatch table ──────────────────────────────────────────────── */

#define R(OP, TE) [TE] = _kern_##OP##_##TE

static const NumcBinaryReductionKernel dot_table[] = {
    R(dot, NUMC_DTYPE_INT8),    R(dot, NUMC_DTYPE_INT16),
    R(dot, NUMC_DTYPE_INT32),   R(dot, NUMC_DTYPE_INT64),
    R(dot, NUMC_DTYPE_UINT8),   R(dot, NUMC_DTYPE_UINT16),
    R(dot, NUMC_DTYPE_UINT32),  R(dot, NUMC_DTYPE_UINT64),
    R(dot, NUMC_DTYPE_FLOAT32), R(dot, NUMC_DTYPE_FLOAT64),
};

#undef R

/* ── BLIS wrappers ─────────────────────────────────────────────────── */

#ifdef HAVE_BLAS
#include <blis.h>
#include <pthread.h>

static void _dot_blis_f32(const NumcArray *a, const NumcArray *b,
                          NumcArray *out) {
  float rho = 0.0f;
  inc_t inca = (inc_t)(a->strides[0] / sizeof(float));
  inc_t incb = (inc_t)(b->strides[0] / sizeof(float));
  bli_sdotv(BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE, (dim_t)a->size,
            (float *)a->data, inca, (float *)b->data, incb, &rho);
  *(float *)out->data = rho;
}

static void _dot_blis_f64(const NumcArray *a, const NumcArray *b,
                          NumcArray *out) {
  double rho = 0.0;
  inc_t inca = (inc_t)(a->strides[0] / sizeof(double));
  inc_t incb = (inc_t)(b->strides[0] / sizeof(double));
  bli_ddotv(BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE, (dim_t)a->size,
            (double *)a->data, inca, (double *)b->data, incb, &rho);
  *(double *)out->data = rho;
}

static void _dot_blis_gemm_f32(const NumcArray *a, const NumcArray *b,
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

static void _dot_blis_gemm_f64(const NumcArray *a, const NumcArray *b,
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
#endif

/* ── Naive matmul kernels ────────────────────────────────────────── */

#define STAMP_DOT_NAIVE(TE, CT, ACC_CT)                                        \
  static void _dot_naive_##TE(const char *pa, const char *pb, char *po,        \
                              size_t M, size_t K, size_t N, intptr_t rsa,      \
                              intptr_t csa, intptr_t rsb, intptr_t csb,        \
                              intptr_t rso, intptr_t cso) {                    \
    const CT *a = (const CT *)pa;                                              \
    const CT *b = (const CT *)pb;                                              \
    CT *out = (CT *)po;                                                        \
    NUMC_OMP_FOR(                                                              \
        M * N, sizeof(CT), for (size_t i = 0; i < M; i++) {                    \
          for (size_t j = 0; j < N; j++) {                                     \
            ACC_CT acc = 0;                                                    \
            for (size_t k = 0; k < K; k++) {                                   \
              acc +=                                                           \
                  (ACC_CT)a[i * rsa + k * csa] * (ACC_CT)b[k * rsb + j * csb]; \
            }                                                                  \
            out[i * rso + j * cso] = (CT)acc;                                  \
          }                                                                    \
        });                                                                    \
  }

STAMP_DOT_NAIVE(NUMC_DTYPE_INT8, int8_t, int32_t)
STAMP_DOT_NAIVE(NUMC_DTYPE_INT16, int16_t, int64_t)
STAMP_DOT_NAIVE(NUMC_DTYPE_INT32, int32_t, int64_t)
STAMP_DOT_NAIVE(NUMC_DTYPE_INT64, int64_t, int64_t)
STAMP_DOT_NAIVE(NUMC_DTYPE_UINT8, uint8_t, uint32_t)
STAMP_DOT_NAIVE(NUMC_DTYPE_UINT16, uint16_t, uint64_t)
STAMP_DOT_NAIVE(NUMC_DTYPE_UINT32, uint32_t, uint64_t)
STAMP_DOT_NAIVE(NUMC_DTYPE_UINT64, uint64_t, uint64_t)
STAMP_DOT_NAIVE(NUMC_DTYPE_FLOAT32, float, float)
STAMP_DOT_NAIVE(NUMC_DTYPE_FLOAT64, double, double)

typedef void (*DotNaiveKernel)(const char *pa, const char *pb, char *po,
                               size_t M, size_t K, size_t N, intptr_t rsa,
                               intptr_t csa, intptr_t rsb, intptr_t csb,
                               intptr_t rso, intptr_t cso);

static const DotNaiveKernel dot_naive_table[] = {
    [NUMC_DTYPE_INT8] = _dot_naive_NUMC_DTYPE_INT8,
    [NUMC_DTYPE_INT16] = _dot_naive_NUMC_DTYPE_INT16,
    [NUMC_DTYPE_INT32] = _dot_naive_NUMC_DTYPE_INT32,
    [NUMC_DTYPE_INT64] = _dot_naive_NUMC_DTYPE_INT64,
    [NUMC_DTYPE_UINT8] = _dot_naive_NUMC_DTYPE_UINT8,
    [NUMC_DTYPE_UINT16] = _dot_naive_NUMC_DTYPE_UINT16,
    [NUMC_DTYPE_UINT32] = _dot_naive_NUMC_DTYPE_UINT32,
    [NUMC_DTYPE_UINT64] = _dot_naive_NUMC_DTYPE_UINT64,
    [NUMC_DTYPE_FLOAT32] = _dot_naive_NUMC_DTYPE_FLOAT32,
    [NUMC_DTYPE_FLOAT64] = _dot_naive_NUMC_DTYPE_FLOAT64,
};

/* ── Helpers ─────────────────────────────────────────────────────── */

static double _to_double(const void *ptr, NumcDType dt) {
  switch (dt) {
  case NUMC_DTYPE_INT8:
    return (double)*(const int8_t *)ptr;
  case NUMC_DTYPE_INT16:
    return (double)*(const int16_t *)ptr;
  case NUMC_DTYPE_INT32:
    return (double)*(const int32_t *)ptr;
  case NUMC_DTYPE_INT64:
    return (double)*(const int64_t *)ptr;
  case NUMC_DTYPE_UINT8:
    return (double)*(const uint8_t *)ptr;
  case NUMC_DTYPE_UINT16:
    return (double)*(const uint16_t *)ptr;
  case NUMC_DTYPE_UINT32:
    return (double)*(const uint32_t *)ptr;
  case NUMC_DTYPE_UINT64:
    return (double)*(const uint64_t *)ptr;
  case NUMC_DTYPE_FLOAT32:
    return (double)*(const float *)ptr;
  case NUMC_DTYPE_FLOAT64:
    return *(const double *)ptr;
  }
  return 0.0;
}

/* ── Core dot dispatch ───────────────────────────────────────────── */

static inline void _reduce_dot_op(const struct NumcArray *a,
                                  const struct NumcArray *b,
                                  struct NumcArray *out,
                                  const NumcBinaryReductionKernel *table) {
  /* Case 3: Either is 0-D (scalar) */
  if (a->dim == 0 || b->dim == 0) {
    const struct NumcArray *scalar_arr = (a->dim == 0) ? a : b;
    const struct NumcArray *other_arr = (a->dim == 0) ? b : a;
    double val = _to_double(scalar_arr->data, scalar_arr->dtype);
    numc_mul_scalar(other_arr, val, out);
    return;
  }

  /* Case 1: Both are 1-D (vector dot product) */
  if (a->dim == 1 && b->dim == 1) {

#if NUMC_HAVE_AVX2
    intptr_t sa = (intptr_t)a->strides[0];
    intptr_t sb = (intptr_t)b->strides[0];

    if (sa == (intptr_t)sizeof(float) && sb == (intptr_t)sizeof(float) &&
        a->dtype == NUMC_DTYPE_FLOAT32) {
      dot_f32_avx2((const float *)a->data, (const float *)b->data, a->size,
                   (float *)out->data);
      return;
    }

    if (sa == (intptr_t)sizeof(double) && sb == (intptr_t)sizeof(double) &&
        a->dtype == NUMC_DTYPE_FLOAT64) {
      dot_f64_avx2((const double *)a->data, (const double *)b->data, a->size,
                   (double *)out->data);
      return;
    }
#endif

#ifdef HAVE_BLAS
    if (a->dtype == NUMC_DTYPE_FLOAT32) {
      _dot_blis_f32(a, b, out);
      return;
    }
    if (a->dtype == NUMC_DTYPE_FLOAT64) {
      _dot_blis_f64(a, b, out);
      return;
    }
#endif
    NumcBinaryReductionKernel kern = table[a->dtype];
    kern((const char *)a->data, (const char *)b->data, (char *)out->data,
         a->size, (intptr_t)a->strides[0], (intptr_t)b->strides[0]);
    return;
  }

  /* Unified ND support: Collapse to (M, K) @ (P, K, N)
   * Results in out(M, P, N) where P is the batch of b.
   */
  size_t k_dim = a->shape[a->dim - 1];
  size_t m_dim = a->size / k_dim;
  size_t n_dim = (b->dim == 1) ? 1 : b->shape[b->dim - 1];
  size_t p_batch = b->size / (k_dim * n_dim);

  intptr_t rsa =
      (intptr_t)(a->dim > 0 ? a->strides[0] : 0) / (intptr_t)a->elem_size;
  intptr_t csa = (intptr_t)a->strides[a->dim - 1] / (intptr_t)a->elem_size;

  intptr_t rsb = (intptr_t)(b->dim > 1 ? b->strides[b->dim - 2]
                                       : (b->dim > 0 ? b->strides[0] : 0)) /
                 (intptr_t)b->elem_size;
  intptr_t csb = (intptr_t)(b->dim > 0 ? b->strides[b->dim - 1] : 0) /
                 (intptr_t)b->elem_size;

  intptr_t rso =
      (intptr_t)(out->dim > 0 ? out->strides[0] : 0) / (intptr_t)out->elem_size;
  intptr_t cso = (intptr_t)(out->dim > 0 ? out->strides[out->dim - 1] : 0) /
                 (intptr_t)out->elem_size;

  if (p_batch == 1) {
#if NUMC_HAVE_AVX2
    if (csb == 1 && cso == 1 && a->dtype == NUMC_DTYPE_FLOAT32) {
      gemm_f32_avx2((const float *)a->data, (const float *)b->data,
                    (float *)out->data, m_dim, k_dim, n_dim, rsa, csa, rsb,
                    rso);
      return;
    }
    if (csb == 1 && cso == 1 && a->dtype == NUMC_DTYPE_FLOAT64) {
      gemm_f64_avx2((const double *)a->data, (const double *)b->data,
                    (double *)out->data, m_dim, k_dim, n_dim, rsa, csa, rsb,
                    rso);
      return;
    }
#endif

#ifdef HAVE_BLAS
    if (a->dtype == NUMC_DTYPE_FLOAT32 && a->is_contiguous &&
        b->is_contiguous) {
      _dot_blis_gemm_f32(a, b, out, m_dim, k_dim, n_dim);
      return;
    }
    if (a->dtype == NUMC_DTYPE_FLOAT64 && a->is_contiguous &&
        b->is_contiguous) {
      _dot_blis_gemm_f64(a, b, out, m_dim, k_dim, n_dim);
      return;
    }
#endif
    dot_naive_table[a->dtype]((const char *)a->data, (const char *)b->data,
                              (char *)out->data, m_dim, k_dim, n_dim, rsa, csa,
                              rsb, csb, rso, cso);
    return;
  }

  /* For P > 1, loop over batches of b and slices of out. */
  for (size_t p_idx = 0; p_idx < p_batch; p_idx++) {
    const char *bp =
        (const char *)b->data + p_idx * k_dim * n_dim * b->elem_size;
    char *op = (char *)out->data + p_idx * n_dim * out->elem_size;

    if (out->dim == 3) {
      op = (char *)out->data + p_idx * out->strides[1];
    }
    if (b->dim == 3) {
      bp = (const char *)b->data + p_idx * b->strides[0];
    }

#ifdef HAVE_BLAS
    if (a->dtype == NUMC_DTYPE_FLOAT32 && a->is_contiguous &&
        b->is_contiguous && out->is_contiguous) {
      float alpha = 1.0f, beta = 0.0f;
      bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, (dim_t)m_dim,
                (dim_t)n_dim, (dim_t)k_dim, &alpha, (float *)a->data,
                (inc_t)rsa, (inc_t)csa, (float *)bp, (inc_t)rsb, (inc_t)csb,
                &beta, (float *)op, (inc_t)rso, (inc_t)cso);
      continue;
    }
#endif
    dot_naive_table[a->dtype]((const char *)a->data, (const char *)bp,
                              (char *)op, m_dim, k_dim, n_dim, rsa, csa, rsb,
                              csb, rso, cso);
  }
}

/* ── Public API ──────────────────────────────────────────────────── */

int numc_dot(const NumcArray *a, const NumcArray *b, NumcArray *out) {
  int err = _check_dot(a, b, out);
  if (err)
    return err;
  _reduce_dot_op(a, b, out, dot_table);
  return 0;
}
