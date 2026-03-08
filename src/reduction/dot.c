#include "../arch_dispatch.h"
#include "dispatch.h"
#include "numc/dtype.h"
#include <numc/math.h>
#include <string.h>

#if NUMC_HAVE_AVX2
#include "intrinsics/dot_avx2.h"
#include "intrinsics/gemm_avx2.h"
#endif

#define IS_ALIGNED(ptr, align) \
  (((uintptr_t)(ptr) & ((uintptr_t)(align) - 1)) == 0)
#define NUMC_DTYPE_COUNT (NUMC_DTYPE_FLOAT64 + 1)

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

/* ── BLAS wrappers + tables ───────────────────────────────────────── */

#ifdef HAVE_BLAS
#include <blis.h>
#include <pthread.h>

typedef void (*BlasDotKernel)(const NumcArray *a, const NumcArray *b,
                              NumcArray *out);
typedef void (*BlasGemmKernel)(const NumcArray *a, const NumcArray *b,
                               NumcArray *out, size_t M, size_t K, size_t N);
typedef void (*BlasBatchGemmKernel)(const char *a_data, const char *b_data,
                                    char *out_data, size_t M, size_t K,
                                    size_t N, intptr_t rsa, intptr_t csa,
                                    intptr_t rsb, intptr_t csb, intptr_t rso,
                                    intptr_t cso);

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

/* ── SIMD wrappers + dispatch tables ──────────────────────────────── */

typedef void (*DotSimdKernel)(const void *a, const void *b, size_t n,
                              void *out);
typedef void (*GemmSimdKernel)(const void *a, const void *b, void *out,
                               size_t M, size_t K, size_t N, intptr_t rsa,
                               intptr_t csa, intptr_t rsb, intptr_t rso);

#if NUMC_HAVE_AVX2
static void _dot_f32_avx2_a(const void *a, const void *b, size_t n, void *out) {
  dot_f32_avx2((const float *)a, (const float *)b, n, (float *)out);
}
static void _dot_f32_avx2_u(const void *a, const void *b, size_t n, void *out) {
  dot_f32u_avx2((const float *)a, (const float *)b, n, (float *)out);
}
static void _dot_f64_avx2_a(const void *a, const void *b, size_t n, void *out) {
  dot_f64_avx2((const double *)a, (const double *)b, n, (double *)out);
}
static void _dot_f64_avx2_u(const void *a, const void *b, size_t n, void *out) {
  dot_f64u_avx2((const double *)a, (const double *)b, n, (double *)out);
}
static void _dot_i32_avx2_a(const void *a, const void *b, size_t n, void *out) {
  dot_i32_avx2((const int32_t *)a, (const int32_t *)b, n, (int32_t *)out);
}
static void _gemm_f32_avx2_w(const void *a, const void *b, void *out, size_t M,
                             size_t K, size_t N, intptr_t rsa, intptr_t csa,
                             intptr_t rsb, intptr_t rso) {
  gemm_f32_avx2((const float *)a, (const float *)b, (float *)out, M, K, N, rsa,
                csa, rsb, rso);
}
static void _gemm_f64_avx2_w(const void *a, const void *b, void *out, size_t M,
                             size_t K, size_t N, intptr_t rsa, intptr_t csa,
                             intptr_t rsb, intptr_t rso) {
  gemm_f64_avx2((const double *)a, (const double *)b, (double *)out, M, K, N,
                rsa, csa, rsb, rso);
}
#endif

static const DotSimdKernel dot_simd_aligned[NUMC_DTYPE_COUNT] = {
#if NUMC_HAVE_AVX2
    [NUMC_DTYPE_FLOAT32] = _dot_f32_avx2_a,
    [NUMC_DTYPE_FLOAT64] = _dot_f64_avx2_a,
    [NUMC_DTYPE_INT32] = _dot_i32_avx2_a,
#endif
};

static const DotSimdKernel dot_simd_unaligned[NUMC_DTYPE_COUNT] = {
#if NUMC_HAVE_AVX2
    [NUMC_DTYPE_FLOAT32] = _dot_f32_avx2_u,
    [NUMC_DTYPE_FLOAT64] = _dot_f64_avx2_u,
/* no i32 unaligned variant — NULL falls through to naive */
#endif
};

static const GemmSimdKernel gemm_simd_table[NUMC_DTYPE_COUNT] = {
#if NUMC_HAVE_AVX2
    [NUMC_DTYPE_FLOAT32] = _gemm_f32_avx2_w,
    [NUMC_DTYPE_FLOAT64] = _gemm_f64_avx2_w,
#endif
};

/* ── OMP-parallel SIMD dot (per-dtype, matches DEFINE_FLOAT_REDUCTION_KERNEL
 *    pattern: each thread calls SIMD kernel on its chunk, OMP reduces) ── */

#define DEFINE_DOT_SIMD_OMP(TE, CT)                                           \
  static inline void _dot_simd_omp_##TE(DotSimdKernel omp_kern,               \
                                        DotSimdKernel st_kern, const void *a, \
                                        const void *b, size_t n, void *out) { \
    const CT *pa = (const CT *)a;                                             \
    const CT *pb = (const CT *)b;                                             \
    size_t total_bytes = n * sizeof(CT) * 2;                                  \
    int nt = (int)(total_bytes / NUMC_OMP_BYTES_PER_THREAD);                  \
    if (total_bytes > NUMC_OMP_BYTE_THRESHOLD && nt >= 2) {                   \
      CT global = 0;                                                          \
      NUMC_PRAGMA(omp parallel for reduction(+ : global)                       \
                      schedule(static) num_threads(nt))                       \
      for (int t = 0; t < nt; t++) {                                          \
        size_t start = (size_t)t * (n / (size_t)nt);                          \
        size_t end = (t == nt - 1) ? n : start + n / (size_t)nt;              \
        CT local;                                                             \
        omp_kern(pa + start, pb + start, end - start, &local);                \
        global += local;                                                      \
      }                                                                       \
      *(CT *)out = global;                                                    \
    } else {                                                                  \
      st_kern(a, b, n, out);                                                  \
    }                                                                         \
  }

DEFINE_DOT_SIMD_OMP(NUMC_DTYPE_INT32, NUMC_INT32)
GENERATE_FLOAT_NUMC_TYPES(DEFINE_DOT_SIMD_OMP)
#undef DEFINE_DOT_SIMD_OMP

typedef void (*DotSimdOmpFn)(DotSimdKernel omp_kern, DotSimdKernel st_kern,
                             const void *a, const void *b, size_t n, void *out);

static const DotSimdOmpFn dot_simd_omp_table[NUMC_DTYPE_COUNT] = {
    [NUMC_DTYPE_INT32] = _dot_simd_omp_NUMC_DTYPE_INT32,
    [NUMC_DTYPE_FLOAT32] = _dot_simd_omp_NUMC_DTYPE_FLOAT32,
    [NUMC_DTYPE_FLOAT64] = _dot_simd_omp_NUMC_DTYPE_FLOAT64,
};

/* ── Case functions ───────────────────────────────────────────────── */

static inline void _dot_scalar_case(const NumcArray *a, const NumcArray *b,
                                    NumcArray *out) {
  const NumcArray *scalar_arr = (a->dim == 0) ? a : b;
  const NumcArray *other_arr = (a->dim == 0) ? b : a;
  double val = _to_double(scalar_arr->data, scalar_arr->dtype);
  numc_mul_scalar(other_arr, val, out);
}

static inline void _dot_1d_case(const NumcArray *a, const NumcArray *b,
                                NumcArray *out,
                                const NumcBinaryReductionKernel *table) {
  NumcDType dt = a->dtype;

  if (a->is_contiguous && b->is_contiguous) {
    /* OMP chunks may not be aligned, so always use unaligned for OMP.
     * Single-threaded path can use aligned if pointers qualify. */
    DotSimdKernel kern_u = dot_simd_unaligned[dt];
    if (kern_u) {
      DotSimdOmpFn omp_fn = dot_simd_omp_table[dt];
      bool aligned = IS_ALIGNED(a->data, NUMC_SIMD_ALIGN) &&
                     IS_ALIGNED(b->data, NUMC_SIMD_ALIGN);
      DotSimdKernel kern_st = aligned ? dot_simd_aligned[dt] : kern_u;
      omp_fn(kern_u, kern_st, a->data, b->data, a->size, out->data);
      return;
    }
    /* No unaligned variant — try aligned-only (single-threaded) */
    DotSimdKernel kern_a = dot_simd_aligned[dt];
    if (kern_a) {
      kern_a(a->data, b->data, a->size, out->data);
      return;
    }
  }

  /* Try BLAS */
#ifdef HAVE_BLAS
  {
    BlasDotKernel blas_kern = blas_dot_table[dt];
    if (blas_kern) {
      blas_kern(a, b, out);
      return;
    }
  }
#endif

  /* Naive fallback */
  table[dt]((const char *)a->data, (const char *)b->data, (char *)out->data,
            a->size, (intptr_t)a->strides[0], (intptr_t)b->strides[0]);
}

static inline void _dot_nd_case(const NumcArray *a, const NumcArray *b,
                                NumcArray *out) {
  NumcDType dt = a->dtype;

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
    /* Try SIMD GEMM (requires column-contiguous b and out) */
    if (csb == 1 && cso == 1) {
      GemmSimdKernel kern = gemm_simd_table[dt];
      if (kern) {
        kern(a->data, b->data, out->data, m_dim, k_dim, n_dim, rsa, csa, rsb,
             rso);
        return;
      }
    }

    /* Try BLAS GEMM */
#ifdef HAVE_BLAS
    if (a->is_contiguous && b->is_contiguous) {
      BlasGemmKernel blas_kern = blas_gemm_table[dt];
      if (blas_kern) {
        blas_kern(a, b, out, m_dim, k_dim, n_dim);
        return;
      }
    }
#endif

    dot_naive_table[dt]((const char *)a->data, (const char *)b->data,
                        (char *)out->data, m_dim, k_dim, n_dim, rsa, csa, rsb,
                        csb, rso, cso);
    return;
  }

  /* Batch loop (p_batch > 1) */
  for (size_t p_idx = 0; p_idx < p_batch; p_idx++) {
    const char *bp =
        (const char *)b->data + p_idx * k_dim * n_dim * b->elem_size;
    char *op = (char *)out->data + p_idx * n_dim * out->elem_size;

    if (out->dim == 3)
      op = (char *)out->data + p_idx * out->strides[1];
    if (b->dim == 3)
      bp = (const char *)b->data + p_idx * b->strides[0];

#ifdef HAVE_BLAS
    if (a->is_contiguous && b->is_contiguous && out->is_contiguous) {
      BlasBatchGemmKernel blas_kern = blas_batch_gemm_table[dt];
      if (blas_kern) {
        blas_kern((const char *)a->data, bp, op, m_dim, k_dim, n_dim, rsa, csa,
                  rsb, csb, rso, cso);
        continue;
      }
    }
#endif

    dot_naive_table[dt]((const char *)a->data, bp, op, m_dim, k_dim, n_dim, rsa,
                        csa, rsb, csb, rso, cso);
  }
}

/* ── Core dot dispatch ───────────────────────────────────────────── */

static inline void _reduce_dot_op(const NumcArray *a, const NumcArray *b,
                                  NumcArray *out,
                                  const NumcBinaryReductionKernel *table) {
  if (a->dim == 0 || b->dim == 0) {
    _dot_scalar_case(a, b, out);
    return;
  }
  if (a->dim == 1 && b->dim == 1) {
    _dot_1d_case(a, b, out, table);
    return;
  }
  _dot_nd_case(a, b, out);
}

/* ── Public API ──────────────────────────────────────────────────── */

int numc_dot(const NumcArray *a, const NumcArray *b, NumcArray *out) {
  int err = _check_dot(a, b, out);
  if (err)
    return err;
  _reduce_dot_op(a, b, out, dot_table);
  return 0;
}
