#include "dispatch.h"
#include "kernel.h"
#include "numc/dtype.h"
#include "numc/math.h"
#include "internal.h"
#include "arch_dispatch.h"

#include <stdio.h>

#if NUMC_HAVE_AVX2
#include "intrinsics/gemm_avx2.h"
#include "intrinsics/gemmsup_avx2.h"
#endif

#if NUMC_HAVE_AVX512
#include "intrinsics/gemm_avx512.h"
#endif

#if NUMC_HAVE_NEON
#include "intrinsics/gemm_neon.h"
#endif

#if NUMC_HAVE_SVE
#include "intrinsics/gemm_sve.h"
#endif

#if NUMC_HAVE_RVV
#include "intrinsics/gemm_rvv.h"
#endif

#if defined(HAVE_BLAS) && defined(NUMC_PREFER_BLAS) && defined(HAVE_OPENBLAS)
#include <cblas.h>
#elif defined(HAVE_BLAS) && defined(NUMC_PREFER_BLAS)
#include <blis.h>
#endif

/* libomp defaults to KMP_BLOCKTIME=0, immediately sleeping OpenMP
 * threads after each parallel region. Waking them from OS idle
 * states costs 40-70 ms — catastrophic for sub-ms sgemm calls.
 * Setting KMP_BLOCKTIME=200 (the MKL default) keeps threads spinning
 * for 200 ms between calls. Must run before the first OpenMP
 * parallel region; overwrite=0 respects user-set values. */
#ifdef HAVE_OMP
__attribute__((constructor)) static void _numc_omp_init(void) {
  if (!getenv("KMP_BLOCKTIME") && !getenv("OMP_WAIT_POLICY")) {
    setenv("KMP_BLOCKTIME", "200", 0);
  }
}
#endif

void _numc_runtime_init(void) {
#ifdef HAVE_OMP
/* Pre-warm OpenMP thread pool by running a dummy parallel loop.
 * This avoids the ~50ms 'cold start' penalty on the first math call. */
#pragma omp parallel
  { (void)0; }
#endif
}

#if defined(HAVE_BLAS) && defined(NUMC_PREFER_BLAS)
#ifdef HAVE_OPENBLAS
/*
 * OpenBLAS threading: delegate to OpenBLAS's internal scheduler.
 * Only throttle to 1 thread for tiny problems where fork overhead
 * dominates. Users tune via OPENBLAS_NUM_THREADS or OMP_NUM_THREADS.
 */
static void _openblas_set_threading(size_t total_ops) {
  if (total_ops < 65536)
    openblas_set_num_threads(1);
}

static void _matmul_openblas_f32(const struct NumcArray *a,
                                 const struct NumcArray *b,
                                 struct NumcArray *out) {
  int m = (int)a->shape[0], k = (int)a->shape[1], n = (int)b->shape[1];
  int lda = (int)(a->strides[0] / (intptr_t)sizeof(float));
  int ldb = (int)(b->strides[0] / (intptr_t)sizeof(float));
  int ldc = (int)(out->strides[0] / (intptr_t)sizeof(float));

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f,
              (const float *)a->data, lda, (const float *)b->data, ldb, 0.0f,
              (float *)out->data, ldc);
}

static void _matmul_openblas_f64(const struct NumcArray *a,
                                 const struct NumcArray *b,
                                 struct NumcArray *out) {
  int m = (int)a->shape[0], k = (int)a->shape[1], n = (int)b->shape[1];
  int lda = (int)(a->strides[0] / (intptr_t)sizeof(double));
  int ldb = (int)(b->strides[0] / (intptr_t)sizeof(double));
  int ldc = (int)(out->strides[0] / (intptr_t)sizeof(double));

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0,
              (const double *)a->data, lda, (const double *)b->data, ldb, 0.0,
              (double *)out->data, ldc);
}

#else /* !HAVE_OPENBLAS → BLIS path */

/*
 * BLIS threading: delegate to BLIS's internal scheduler.
 * Only throttle to 1 thread for tiny problems where fork overhead
 * dominates. Users tune via BLIS_NUM_THREADS or OMP_NUM_THREADS.
 */
static void _blis_set_threading(size_t total_ops) {
#ifdef HAVE_OMP
  int nthreads = omp_get_max_threads();
  if (nthreads < 1)
    nthreads = 1;

#ifdef NUMC_BLIS_OPTIMIZED
  bli_thread_set_num_threads(total_ops < 65536 ? 1 : nthreads);
#else
  if (total_ops < 65536) {
    bli_thread_set_ways(1, 1, 1, 1, 1);
  } else {
    bli_thread_set_ways(1, 1, nthreads, 1, 1);
  }
#endif
#else
  (void)total_ops;
#endif
}

void _matmul_blis_f32(const struct NumcArray *a, const struct NumcArray *b,
                      struct NumcArray *out) {
  float alpha = 1.0f, beta = 0.0f;
  dim_t m = (dim_t)a->shape[0], k = (dim_t)a->shape[1], n = (dim_t)b->shape[1];

  /* BLIS Stride Support: allows zero-copy multiplication of views/slices.
   * Strides must be aligned to element size; guaranteed by arena allocator. */
  assert(a->strides[0] % sizeof(float) == 0 && "stride not aligned to float");
  assert(a->strides[1] % sizeof(float) == 0 && "stride not aligned to float");
  assert(b->strides[0] % sizeof(float) == 0 && "stride not aligned to float");
  assert(b->strides[1] % sizeof(float) == 0 && "stride not aligned to float");
  inc_t rs_a = (inc_t)(a->strides[0] / sizeof(float));
  inc_t cs_a = (inc_t)(a->strides[1] / sizeof(float));
  inc_t rs_b = (inc_t)(b->strides[0] / sizeof(float));
  inc_t cs_b = (inc_t)(b->strides[1] / sizeof(float));
  inc_t rs_c = (inc_t)(out->strides[0] / sizeof(float));
  inc_t cs_c = (inc_t)(out->strides[1] / sizeof(float));

  bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, m, n, k, &alpha,
            (float *)a->data, rs_a, cs_a, (float *)b->data, rs_b, cs_b, &beta,
            (float *)out->data, rs_c, cs_c);
}

void _matmul_blis_f64(const struct NumcArray *a, const struct NumcArray *b,
                      struct NumcArray *out) {
  double alpha = 1.0, beta = 0.0;
  dim_t m = (dim_t)a->shape[0], k = (dim_t)a->shape[1], n = (dim_t)b->shape[1];

  assert(a->strides[0] % sizeof(double) == 0 && "stride not aligned to double");
  assert(a->strides[1] % sizeof(double) == 0 && "stride not aligned to double");
  assert(b->strides[0] % sizeof(double) == 0 && "stride not aligned to double");
  assert(b->strides[1] % sizeof(double) == 0 && "stride not aligned to double");
  inc_t rs_a = (inc_t)(a->strides[0] / sizeof(double));
  inc_t cs_a = (inc_t)(a->strides[1] / sizeof(double));
  inc_t rs_b = (inc_t)(b->strides[0] / sizeof(double));
  inc_t cs_b = (inc_t)(b->strides[1] / sizeof(double));
  inc_t rs_c = (inc_t)(out->strides[0] / sizeof(double));
  inc_t cs_c = (inc_t)(out->strides[1] / sizeof(double));

  bli_dgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, m, n, k, &alpha,
            (double *)a->data, rs_a, cs_a, (double *)b->data, rs_b, cs_b, &beta,
            (double *)out->data, rs_c, cs_c);
}
#endif /* !HAVE_OPENBLAS */
#endif /* HAVE_BLAS && NUMC_PREFER_BLAS */

/* ── SIMD gemm wrappers ──────────────────────────────────────────── */

#define NUMC_DTYPE_COUNT (NUMC_DTYPE_FLOAT64 + 1)

typedef void (*GemmSimdKernel)(const void *a, const void *b, void *out,
                               size_t M, size_t K, size_t N, intptr_t rsa,
                               intptr_t csa, intptr_t rsb, intptr_t rso);

#define GEMM_WRAP(name, CT, fn)                                               \
  static void name(const void *a, const void *b, void *out, size_t M,         \
                   size_t K, size_t N, intptr_t rsa, intptr_t csa,            \
                   intptr_t rsb, intptr_t rso) {                              \
    fn((const CT *)a, (const CT *)b, (CT *)out, M, K, N, rsa, csa, rsb, rso); \
  }

#if NUMC_HAVE_AVX2
GEMM_WRAP(_gemm_i8_avx2, int8_t, gemm_i8_avx2)
GEMM_WRAP(_gemm_i16_avx2, int16_t, gemm_i16_avx2)
GEMM_WRAP(_gemm_i32_avx2, int32_t, gemm_i32_avx2)
GEMM_WRAP(_gemm_i64_avx2, int64_t, gemm_i64_avx2)
GEMM_WRAP(_gemm_u8_avx2, uint8_t, gemm_u8_avx2)
GEMM_WRAP(_gemm_u16_avx2, uint16_t, gemm_u16_avx2)
GEMM_WRAP(_gemm_u32_avx2, uint32_t, gemm_u32_avx2)
GEMM_WRAP(_gemm_u64_avx2, uint64_t, gemm_u64_avx2)
GEMM_WRAP(_gemm_f32_avx2, float, gemm_f32_avx2)
GEMM_WRAP(_gemm_f64_avx2, double, gemm_f64_avx2)
#endif

#if NUMC_HAVE_AVX512
GEMM_WRAP(_gemm_f32_avx512, float, gemm_f32_avx512)
GEMM_WRAP(_gemm_f64_avx512, double, gemm_f64_avx512)
GEMM_WRAP(_gemm_i32_avx512, int32_t, gemm_i32_avx512)
GEMM_WRAP(_gemm_u32_avx512, uint32_t, gemm_u32_avx512)
GEMM_WRAP(_gemm_i16_avx512, int16_t, gemm_i16_avx512)
GEMM_WRAP(_gemm_u16_avx512, uint16_t, gemm_u16_avx512)
GEMM_WRAP(_gemm_i64_avx512, int64_t, gemm_i64_avx512)
GEMM_WRAP(_gemm_u64_avx512, uint64_t, gemm_u64_avx512)
GEMM_WRAP(_gemm_i8_avx512, int8_t, gemm_i8_avx512)
GEMM_WRAP(_gemm_u8_avx512, uint8_t, gemm_u8_avx512)
#endif

#if NUMC_HAVE_NEON
GEMM_WRAP(_gemm_i8_neon, int8_t, gemm_i8_neon)
GEMM_WRAP(_gemm_i16_neon, int16_t, gemm_i16_neon)
GEMM_WRAP(_gemm_i32_neon, int32_t, gemm_i32_neon)
GEMM_WRAP(_gemm_i64_neon, int64_t, gemm_i64_neon)
GEMM_WRAP(_gemm_u8_neon, uint8_t, gemm_u8_neon)
GEMM_WRAP(_gemm_u16_neon, uint16_t, gemm_u16_neon)
GEMM_WRAP(_gemm_u32_neon, uint32_t, gemm_u32_neon)
GEMM_WRAP(_gemm_u64_neon, uint64_t, gemm_u64_neon)
GEMM_WRAP(_gemm_f32_neon, float, gemm_f32_neon)
GEMM_WRAP(_gemm_f64_neon, double, gemm_f64_neon)
#endif

#if NUMC_HAVE_SVE
GEMM_WRAP(_gemm_i8_sve, int8_t, gemm_i8_sve)
GEMM_WRAP(_gemm_i16_sve, int16_t, gemm_i16_sve)
GEMM_WRAP(_gemm_i32_sve, int32_t, gemm_i32_sve)
GEMM_WRAP(_gemm_i64_sve, int64_t, gemm_i64_sve)
GEMM_WRAP(_gemm_u8_sve, uint8_t, gemm_u8_sve)
GEMM_WRAP(_gemm_u16_sve, uint16_t, gemm_u16_sve)
GEMM_WRAP(_gemm_u32_sve, uint32_t, gemm_u32_sve)
GEMM_WRAP(_gemm_u64_sve, uint64_t, gemm_u64_sve)
GEMM_WRAP(_gemm_f32_sve, float, gemm_f32_sve)
GEMM_WRAP(_gemm_f64_sve, double, gemm_f64_sve)
#endif

#if NUMC_HAVE_RVV
GEMM_WRAP(_gemm_i8_rvv, int8_t, gemm_i8_rvv)
GEMM_WRAP(_gemm_i16_rvv, int16_t, gemm_i16_rvv)
GEMM_WRAP(_gemm_i32_rvv, int32_t, gemm_i32_rvv)
GEMM_WRAP(_gemm_i64_rvv, int64_t, gemm_i64_rvv)
GEMM_WRAP(_gemm_u8_rvv, uint8_t, gemm_u8_rvv)
GEMM_WRAP(_gemm_u16_rvv, uint16_t, gemm_u16_rvv)
GEMM_WRAP(_gemm_u32_rvv, uint32_t, gemm_u32_rvv)
GEMM_WRAP(_gemm_u64_rvv, uint64_t, gemm_u64_rvv)
GEMM_WRAP(_gemm_f32_rvv, float, gemm_f32_rvv)
GEMM_WRAP(_gemm_f64_rvv, double, gemm_f64_rvv)
#endif

#undef GEMM_WRAP

static const GemmSimdKernel gemm_simd_table[NUMC_DTYPE_COUNT] = {
#if NUMC_HAVE_AVX2
    [NUMC_DTYPE_INT8] = _gemm_i8_avx2,
    [NUMC_DTYPE_INT16] = _gemm_i16_avx2,
    [NUMC_DTYPE_INT32] = _gemm_i32_avx2,
    [NUMC_DTYPE_INT64] = _gemm_i64_avx2,
    [NUMC_DTYPE_UINT8] = _gemm_u8_avx2,
    [NUMC_DTYPE_UINT16] = _gemm_u16_avx2,
    [NUMC_DTYPE_UINT32] = _gemm_u32_avx2,
    [NUMC_DTYPE_UINT64] = _gemm_u64_avx2,
    [NUMC_DTYPE_FLOAT32] = _gemm_f32_avx2,
    [NUMC_DTYPE_FLOAT64] = _gemm_f64_avx2,
#endif
/* AVX-512 overrides all types when available (designated init allows it) */
#if NUMC_HAVE_AVX512
    [NUMC_DTYPE_INT8] = _gemm_i8_avx512,
    [NUMC_DTYPE_INT16] = _gemm_i16_avx512,
    [NUMC_DTYPE_INT32] = _gemm_i32_avx512,
    [NUMC_DTYPE_INT64] = _gemm_i64_avx512,
    [NUMC_DTYPE_UINT8] = _gemm_u8_avx512,
    [NUMC_DTYPE_UINT16] = _gemm_u16_avx512,
    [NUMC_DTYPE_UINT32] = _gemm_u32_avx512,
    [NUMC_DTYPE_UINT64] = _gemm_u64_avx512,
    [NUMC_DTYPE_FLOAT32] = _gemm_f32_avx512,
    [NUMC_DTYPE_FLOAT64] = _gemm_f64_avx512,
#endif
#if NUMC_HAVE_NEON
    [NUMC_DTYPE_INT8] = _gemm_i8_neon,
    [NUMC_DTYPE_INT16] = _gemm_i16_neon,
    [NUMC_DTYPE_INT32] = _gemm_i32_neon,
    [NUMC_DTYPE_INT64] = _gemm_i64_neon,
    [NUMC_DTYPE_UINT8] = _gemm_u8_neon,
    [NUMC_DTYPE_UINT16] = _gemm_u16_neon,
    [NUMC_DTYPE_UINT32] = _gemm_u32_neon,
    [NUMC_DTYPE_UINT64] = _gemm_u64_neon,
    [NUMC_DTYPE_FLOAT32] = _gemm_f32_neon,
    [NUMC_DTYPE_FLOAT64] = _gemm_f64_neon,
#endif
#if NUMC_HAVE_SVE
    [NUMC_DTYPE_INT8] = _gemm_i8_sve,
    [NUMC_DTYPE_INT16] = _gemm_i16_sve,
    [NUMC_DTYPE_INT32] = _gemm_i32_sve,
    [NUMC_DTYPE_INT64] = _gemm_i64_sve,
    [NUMC_DTYPE_UINT8] = _gemm_u8_sve,
    [NUMC_DTYPE_UINT16] = _gemm_u16_sve,
    [NUMC_DTYPE_UINT32] = _gemm_u32_sve,
    [NUMC_DTYPE_UINT64] = _gemm_u64_sve,
    [NUMC_DTYPE_FLOAT32] = _gemm_f32_sve,
    [NUMC_DTYPE_FLOAT64] = _gemm_f64_sve,
#endif
#if NUMC_HAVE_RVV
    [NUMC_DTYPE_INT8] = _gemm_i8_rvv,
    [NUMC_DTYPE_INT16] = _gemm_i16_rvv,
    [NUMC_DTYPE_INT32] = _gemm_i32_rvv,
    [NUMC_DTYPE_INT64] = _gemm_i64_rvv,
    [NUMC_DTYPE_UINT8] = _gemm_u8_rvv,
    [NUMC_DTYPE_UINT16] = _gemm_u16_rvv,
    [NUMC_DTYPE_UINT32] = _gemm_u32_rvv,
    [NUMC_DTYPE_UINT64] = _gemm_u64_rvv,
    [NUMC_DTYPE_FLOAT32] = _gemm_f32_rvv,
    [NUMC_DTYPE_FLOAT64] = _gemm_f64_rvv,
#endif
};

/* ── Dispatch table for naive C23 kernels ────────────────────────── */

#define E(TE) [TE] = _matmul_naive_##TE
static const MatmulKernel matmul_table[] = {
    E(NUMC_DTYPE_INT8),    E(NUMC_DTYPE_INT16),  E(NUMC_DTYPE_INT32),
    E(NUMC_DTYPE_INT64),   E(NUMC_DTYPE_UINT8),  E(NUMC_DTYPE_UINT16),
    E(NUMC_DTYPE_UINT32),  E(NUMC_DTYPE_UINT64), E(NUMC_DTYPE_FLOAT32),
    E(NUMC_DTYPE_FLOAT64),
};
#undef E

int numc_matmul_naive(const NumcArray *a, const NumcArray *b, NumcArray *out) {
  int err = _check_matmul(a, b, out);
  if (err)
    return err;

  MatmulKernel kern = matmul_table[a->dtype];
  kern((const char *)a->data, (const char *)b->data, (char *)out->data,
       a->shape[0], b->shape[0], out->shape[1]);
  return 0;
}

/* ── Public Unified API ──────────────────────────────────────────── */

int numc_matmul(const NumcArray *a, const NumcArray *b, NumcArray *out) {
  int err = _check_matmul(a, b, out);
  if (err)
    return err;

  /* Small-matrix path: unpacked SIMD GEMM (avoids packing overhead) */
#if NUMC_HAVE_AVX2
  {
    size_t M = a->shape[0], K = a->shape[1], N = b->shape[1];
    if ((uint64_t)M * K * N < GEMMSUP_FLOPS_THRESHOLD) {
      size_t elem = numc_dtype_size(a->dtype);
      intptr_t rsa = a->strides[0] / (intptr_t)elem;
      intptr_t csa = a->strides[1] / (intptr_t)elem;
      intptr_t rsb = b->strides[0] / (intptr_t)elem;
      intptr_t rso = out->strides[0] / (intptr_t)elem;
      if (a->dtype == NUMC_DTYPE_FLOAT32) {
        gemmsup_f32_avx2((const float *)a->data, (const float *)b->data,
                         (float *)out->data, M, K, N, rsa, csa, rsb, rso);
        return 0;
      }
      if (a->dtype == NUMC_DTYPE_FLOAT64) {
        gemmsup_f64_avx2((const double *)a->data, (const double *)b->data,
                         (double *)out->data, M, K, N, rsa, csa, rsb, rso);
        return 0;
      }
    }
  }
#endif

  /* Primary path: packed SIMD GEMM (all types, all strides) */
  {
    GemmSimdKernel simd_kern = gemm_simd_table[a->dtype];
    if (simd_kern) {
      size_t M = a->shape[0], K = a->shape[1], N = b->shape[1];
      size_t elem = numc_dtype_size(a->dtype);
      intptr_t rsa = a->strides[0] / (intptr_t)elem;
      intptr_t csa = a->strides[1] / (intptr_t)elem;
      intptr_t rsb = b->strides[0] / (intptr_t)elem;
      intptr_t rso = out->strides[0] / (intptr_t)elem;
      simd_kern(a->data, b->data, out->data, M, K, N, rsa, csa, rsb, rso);
      return 0;
    }
  }

#if defined(HAVE_BLAS) && defined(NUMC_PREFER_BLAS)
  /* Optional BLAS path — enabled with -DNUMC_PREFER_BLAS */
  {
    size_t total_ops;
    if (__builtin_mul_overflow(a->shape[0], a->shape[1], &total_ops) ||
        __builtin_mul_overflow(total_ops, b->shape[1], &total_ops))
      total_ops = SIZE_MAX;

#ifdef HAVE_OPENBLAS
    {
      size_t elem = numc_dtype_size(a->dtype);
      if (total_ops >= 65536 && a->strides[1] == (intptr_t)elem &&
          b->strides[1] == (intptr_t)elem) {
        if (a->dtype == NUMC_DTYPE_FLOAT32) {
          _openblas_set_threading(total_ops);
          _matmul_openblas_f32(a, b, out);
          return 0;
        }
        if (a->dtype == NUMC_DTYPE_FLOAT64) {
          _openblas_set_threading(total_ops);
          _matmul_openblas_f64(a, b, out);
          return 0;
        }
      }
    }
#elif defined(NUMC_BLIS_OPTIMIZED)
    if (total_ops >= 65536 && a->dtype == NUMC_DTYPE_FLOAT32) {
      _blis_set_threading(total_ops);
      _matmul_blis_f32(a, b, out);
      return 0;
    }
    if (total_ops >= 65536 && a->dtype == NUMC_DTYPE_FLOAT64) {
      _blis_set_threading(total_ops);
      _matmul_blis_f64(a, b, out);
      return 0;
    }
#else
    if (total_ops >= 65536 && a->dtype == NUMC_DTYPE_FLOAT64) {
      _blis_set_threading(total_ops);
      _matmul_blis_f64(a, b, out);
      return 0;
    }
#endif
  }
#endif /* HAVE_BLAS && NUMC_PREFER_BLAS */

  /* Fallback to naive kernels (C23 + OpenMP) */
  MatmulKernel kern = matmul_table[a->dtype];
  kern((const char *)a->data, (const char *)b->data, (char *)out->data,
       a->shape[0], b->shape[0], out->shape[1]);
  return 0;
}
