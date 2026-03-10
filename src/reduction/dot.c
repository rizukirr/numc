#include "arch_dispatch.h"
#include "dispatch.h"
#include "numc/dtype.h"
#include <numc/math.h>
#include <string.h>

#if NUMC_HAVE_AVX512
#include "intrinsics/dot_avx512.h"
#include "intrinsics/gemm_avx512.h"
#endif

#if NUMC_HAVE_AVX2
#include "intrinsics/dot_avx2.h"
#include "intrinsics/gemm_avx2.h"
#endif

#define NUMC_DTYPE_COUNT (NUMC_DTYPE_FLOAT64 + 1)

/* ── Dot product reduction kernels ─────────────────────────────────── */

#define STAMP_DOT(TE, CT) \
  DEFINE_BINARY_REDUCTION_KERNEL(dot, TE, CT, 0, acc + (val_a * val_b), +)
GENERATE_NUMC_TYPES(STAMP_DOT)
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
static void _dot_i8_avx2(const void *a, const void *b, size_t n, void *out) {
  dot_i8_avx2((const int8_t *)a, (const int8_t *)b, n, (int8_t *)out);
}
static void _dot_i16_avx2(const void *a, const void *b, size_t n, void *out) {
  dot_i16_avx2((const int16_t *)a, (const int16_t *)b, n, (int16_t *)out);
}
static void _dot_i32_avx2(const void *a, const void *b, size_t n, void *out) {
  dot_i32_avx2((const int32_t *)a, (const int32_t *)b, n, (int32_t *)out);
}
static void _dot_i64_avx2(const void *a, const void *b, size_t n, void *out) {
  dot_i64_avx2((const int64_t *)a, (const int64_t *)b, n, (int64_t *)out);
}
static void _dot_u8_avx2(const void *a, const void *b, size_t n, void *out) {
  dot_u8_avx2((const uint8_t *)a, (const uint8_t *)b, n, (uint8_t *)out);
}
static void _dot_u16_avx2(const void *a, const void *b, size_t n, void *out) {
  dot_u16_avx2((const uint16_t *)a, (const uint16_t *)b, n, (uint16_t *)out);
}
static void _dot_u32_avx2(const void *a, const void *b, size_t n, void *out) {
  dot_u32_avx2((const uint32_t *)a, (const uint32_t *)b, n, (uint32_t *)out);
}
static void _dot_u64_avx2(const void *a, const void *b, size_t n, void *out) {
  dot_u64_avx2((const uint64_t *)a, (const uint64_t *)b, n, (uint64_t *)out);
}
static void _dot_f32_avx2(const void *a, const void *b, size_t n, void *out) {
  dot_f32u_avx2((const float *)a, (const float *)b, n, (float *)out);
}
static void _dot_f64_avx2(const void *a, const void *b, size_t n, void *out) {
  dot_f64u_avx2((const double *)a, (const double *)b, n, (double *)out);
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
static void _gemm_i32_avx2_w(const void *a, const void *b, void *out, size_t M,
                             size_t K, size_t N, intptr_t rsa, intptr_t csa,
                             intptr_t rsb, intptr_t rso) {
  gemm_i32_avx2((const int32_t *)a, (const int32_t *)b, (int32_t *)out, M, K, N,
                rsa, csa, rsb, rso);
}
static void _gemm_u32_avx2_w(const void *a, const void *b, void *out, size_t M,
                             size_t K, size_t N, intptr_t rsa, intptr_t csa,
                             intptr_t rsb, intptr_t rso) {
  gemm_u32_avx2((const uint32_t *)a, (const uint32_t *)b, (uint32_t *)out, M, K,
                N, rsa, csa, rsb, rso);
}
static void _gemm_i16_avx2_w(const void *a, const void *b, void *out, size_t M,
                             size_t K, size_t N, intptr_t rsa, intptr_t csa,
                             intptr_t rsb, intptr_t rso) {
  gemm_i16_avx2((const int16_t *)a, (const int16_t *)b, (int16_t *)out, M, K, N,
                rsa, csa, rsb, rso);
}
static void _gemm_u16_avx2_w(const void *a, const void *b, void *out, size_t M,
                             size_t K, size_t N, intptr_t rsa, intptr_t csa,
                             intptr_t rsb, intptr_t rso) {
  gemm_u16_avx2((const uint16_t *)a, (const uint16_t *)b, (uint16_t *)out, M, K,
                N, rsa, csa, rsb, rso);
}
static void _gemm_i64_avx2_w(const void *a, const void *b, void *out, size_t M,
                             size_t K, size_t N, intptr_t rsa, intptr_t csa,
                             intptr_t rsb, intptr_t rso) {
  gemm_i64_avx2((const int64_t *)a, (const int64_t *)b, (int64_t *)out, M, K, N,
                rsa, csa, rsb, rso);
}
static void _gemm_u64_avx2_w(const void *a, const void *b, void *out, size_t M,
                             size_t K, size_t N, intptr_t rsa, intptr_t csa,
                             intptr_t rsb, intptr_t rso) {
  gemm_u64_avx2((const uint64_t *)a, (const uint64_t *)b, (uint64_t *)out, M, K,
                N, rsa, csa, rsb, rso);
}
static void _gemm_i8_avx2_w(const void *a, const void *b, void *out, size_t M,
                            size_t K, size_t N, intptr_t rsa, intptr_t csa,
                            intptr_t rsb, intptr_t rso) {
  gemm_i8_avx2((const int8_t *)a, (const int8_t *)b, (int8_t *)out, M, K, N,
               rsa, csa, rsb, rso);
}
static void _gemm_u8_avx2_w(const void *a, const void *b, void *out, size_t M,
                            size_t K, size_t N, intptr_t rsa, intptr_t csa,
                            intptr_t rsb, intptr_t rso) {
  gemm_u8_avx2((const uint8_t *)a, (const uint8_t *)b, (uint8_t *)out, M, K, N,
               rsa, csa, rsb, rso);
}
#endif

#if NUMC_HAVE_AVX512
static void _dot_i8_avx512(const void *a, const void *b, size_t n, void *out) {
  dot_i8_avx512((const int8_t *)a, (const int8_t *)b, n, (int8_t *)out);
}
static void _dot_i16_avx512(const void *a, const void *b, size_t n, void *out) {
  dot_i16_avx512((const int16_t *)a, (const int16_t *)b, n, (int16_t *)out);
}
static void _dot_i32_avx512(const void *a, const void *b, size_t n, void *out) {
  dot_i32_avx512((const int32_t *)a, (const int32_t *)b, n, (int32_t *)out);
}
static void _dot_i64_avx512(const void *a, const void *b, size_t n, void *out) {
  dot_i64_avx512((const int64_t *)a, (const int64_t *)b, n, (int64_t *)out);
}
static void _dot_u8_avx512(const void *a, const void *b, size_t n, void *out) {
  dot_u8_avx512((const uint8_t *)a, (const uint8_t *)b, n, (uint8_t *)out);
}
static void _dot_u16_avx512(const void *a, const void *b, size_t n, void *out) {
  dot_u16_avx512((const uint16_t *)a, (const uint16_t *)b, n, (uint16_t *)out);
}
static void _dot_u32_avx512(const void *a, const void *b, size_t n, void *out) {
  dot_u32_avx512((const uint32_t *)a, (const uint32_t *)b, n, (uint32_t *)out);
}
static void _dot_u64_avx512(const void *a, const void *b, size_t n, void *out) {
  dot_u64_avx512((const uint64_t *)a, (const uint64_t *)b, n, (uint64_t *)out);
}
static void _dot_f32_avx512(const void *a, const void *b, size_t n, void *out) {
  dot_f32u_avx512((const float *)a, (const float *)b, n, (float *)out);
}
static void _dot_f64_avx512(const void *a, const void *b, size_t n, void *out) {
  dot_f64u_avx512((const double *)a, (const double *)b, n, (double *)out);
}
#endif

static const DotSimdKernel dot_simd_table_1d[NUMC_DTYPE_COUNT] = {
#if NUMC_HAVE_AVX512
    [NUMC_DTYPE_INT8] = _dot_i8_avx512,
    [NUMC_DTYPE_INT16] = _dot_i16_avx512,
    [NUMC_DTYPE_INT32] = _dot_i32_avx512,
    [NUMC_DTYPE_INT64] = _dot_i64_avx512,
    [NUMC_DTYPE_UINT8] = _dot_u8_avx512,
    [NUMC_DTYPE_UINT16] = _dot_u16_avx512,
    [NUMC_DTYPE_UINT32] = _dot_u32_avx512,
    [NUMC_DTYPE_UINT64] = _dot_u64_avx512,
    [NUMC_DTYPE_FLOAT32] = _dot_f32_avx512,
    [NUMC_DTYPE_FLOAT64] = _dot_f64_avx512,
#elif NUMC_HAVE_AVX2
    [NUMC_DTYPE_INT8] = _dot_i8_avx2,     [NUMC_DTYPE_INT16] = _dot_i16_avx2,
    [NUMC_DTYPE_INT32] = _dot_i32_avx2,   [NUMC_DTYPE_INT64] = _dot_i64_avx2,
    [NUMC_DTYPE_UINT8] = _dot_u8_avx2,    [NUMC_DTYPE_UINT16] = _dot_u16_avx2,
    [NUMC_DTYPE_UINT32] = _dot_u32_avx2,  [NUMC_DTYPE_UINT64] = _dot_u64_avx2,
    [NUMC_DTYPE_FLOAT32] = _dot_f32_avx2, [NUMC_DTYPE_FLOAT64] = _dot_f64_avx2,
#endif
};

static const GemmSimdKernel gemm_simd_table[NUMC_DTYPE_COUNT] = {
#if NUMC_HAVE_AVX2
    [NUMC_DTYPE_INT8] = _gemm_i8_avx2_w,
    [NUMC_DTYPE_INT16] = _gemm_i16_avx2_w,
    [NUMC_DTYPE_INT32] = _gemm_i32_avx2_w,
    [NUMC_DTYPE_INT64] = _gemm_i64_avx2_w,
    [NUMC_DTYPE_UINT8] = _gemm_u8_avx2_w,
    [NUMC_DTYPE_UINT16] = _gemm_u16_avx2_w,
    [NUMC_DTYPE_UINT32] = _gemm_u32_avx2_w,
    [NUMC_DTYPE_UINT64] = _gemm_u64_avx2_w,
    [NUMC_DTYPE_FLOAT32] = _gemm_f32_avx2_w,
    [NUMC_DTYPE_FLOAT64] = _gemm_f64_avx2_w,
#endif
};

/* ── OMP-parallel SIMD dot (per-dtype, matches DEFINE_FLOAT_REDUCTION_KERNEL
 *    pattern: each thread calls SIMD kernel on its chunk, OMP reduces) ── */

#define DEFINE_DOT_SIMD_OMP(TE, CT)                                           \
  static inline void _dot_simd_omp_##TE(DotSimdKernel kern, const void *a,    \
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
        kern(pa + start, pb + start, end - start, &local);                    \
        global += local;                                                      \
      }                                                                       \
      *(CT *)out = global;                                                    \
    } else {                                                                  \
      kern(a, b, n, out);                                                     \
    }                                                                         \
  }

GENERATE_NUMC_TYPES(DEFINE_DOT_SIMD_OMP)
#undef DEFINE_DOT_SIMD_OMP

typedef void (*DotSimdOmpFn)(DotSimdKernel kern, const void *a, const void *b,
                             size_t n, void *out);

static const DotSimdOmpFn dot_simd_omp_table[NUMC_DTYPE_COUNT] = {
    [NUMC_DTYPE_INT8] = _dot_simd_omp_NUMC_DTYPE_INT8,
    [NUMC_DTYPE_INT16] = _dot_simd_omp_NUMC_DTYPE_INT16,
    [NUMC_DTYPE_INT32] = _dot_simd_omp_NUMC_DTYPE_INT32,
    [NUMC_DTYPE_INT64] = _dot_simd_omp_NUMC_DTYPE_INT64,
    [NUMC_DTYPE_UINT8] = _dot_simd_omp_NUMC_DTYPE_UINT8,
    [NUMC_DTYPE_UINT16] = _dot_simd_omp_NUMC_DTYPE_UINT16,
    [NUMC_DTYPE_UINT32] = _dot_simd_omp_NUMC_DTYPE_UINT32,
    [NUMC_DTYPE_UINT64] = _dot_simd_omp_NUMC_DTYPE_UINT64,
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
    DotSimdKernel kern = dot_simd_table_1d[dt];
    if (kern) {
      DotSimdOmpFn omp_fn = dot_simd_omp_table[dt];
      omp_fn(kern, a->data, b->data, a->size, out->data);
      return;
    }
  }

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
