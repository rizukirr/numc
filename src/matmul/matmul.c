#include "dispatch.h"
#include "kernel.h"
#include "numc/dtype.h"
#include "numc/math.h"
#include "internal.h"
#include "arch_dispatch.h"

#include <stdio.h>
#include <string.h>

#if NUMC_HAVE_AVX2
#include "intrinsics/gemm_avx2.h"
#include "intrinsics/gemmsup_avx2.h"
#endif

#if NUMC_HAVE_NEON && !NUMC_HAVE_SVE
#include "intrinsics/gemmsup_neon.h"
#endif

#if NUMC_HAVE_SVE
#include "intrinsics/gemmsup_sve.h"
#endif

#if NUMC_HAVE_RVV
#include "intrinsics/gemmsup_rvv.h"
#endif

#if NUMC_HAVE_AVX512
#include "intrinsics/gemm_avx512.h"
#include "intrinsics/gemmsup_avx512.h"
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
  memset(out->data, 0, out->capacity);
  kern((const char *)a->data, (const char *)b->data, (char *)out->data,
       a->shape[0], b->shape[0], out->shape[1]);
  return 0;
}

/* ── Public Unified API ──────────────────────────────────────────── */

int numc_matmul(const NumcArray *a, const NumcArray *b, NumcArray *out) {
  int err = _check_matmul(a, b, out);
  if (err)
    return err;

  /* Small-matrix path: unpacked SIMD GEMM (avoids packing overhead).
   * AVX2 tuning note:
   * - f32 gemmsup is beneficial mostly for very small sizes; around 128^3,
   *   packed GEMM tends to win due to better A/B locality.
   * - f64 gemmsup remains good up to roughly 128^3 on typical AVX2 CPUs.
   */
#if NUMC_HAVE_AVX512 || NUMC_HAVE_AVX2 || NUMC_HAVE_NEON || NUMC_HAVE_SVE || \
    NUMC_HAVE_RVV
  {
    size_t m = a->shape[0], k = a->shape[1], n = b->shape[1];
    uint64_t flops = (uint64_t)m * k * n;
#if NUMC_HAVE_AVX512
    uint64_t gemmsup_threshold = GEMMSUP512_FLOPS_THRESHOLD;
#else
    uint64_t gemmsup_threshold = GEMMSUP_FLOPS_THRESHOLD;
#endif
#if NUMC_HAVE_AVX2 && !NUMC_HAVE_AVX512
    if (a->dtype == NUMC_DTYPE_FLOAT32) {
      gemmsup_threshold = (96ULL * 96ULL * 96ULL);
    } else if (a->dtype == NUMC_DTYPE_FLOAT64) {
      gemmsup_threshold = (48ULL * 48ULL * 48ULL);
    }
    if (a->dtype == NUMC_DTYPE_INT32 || a->dtype == NUMC_DTYPE_UINT32) {
      gemmsup_threshold = (96ULL * 96ULL * 96ULL);
    }
    if (a->dtype == NUMC_DTYPE_INT16 || a->dtype == NUMC_DTYPE_UINT16) {
      gemmsup_threshold = (128ULL * 128ULL * 128ULL);
    }
    if (a->dtype == NUMC_DTYPE_INT64 || a->dtype == NUMC_DTYPE_UINT64) {
      gemmsup_threshold = (48ULL * 48ULL * 48ULL);
    }
#endif
    if (flops <= gemmsup_threshold) {
      size_t elem = numc_dtype_size(a->dtype);
      intptr_t rsa = (intptr_t)(a->strides[0] / elem);
      intptr_t csa = (intptr_t)(a->strides[1] / elem);
      intptr_t rsb = (intptr_t)(b->strides[0] / elem);
      intptr_t rso = (intptr_t)(out->strides[0] / elem);
      if (a->dtype == NUMC_DTYPE_FLOAT32) {
#if NUMC_HAVE_AVX512
        gemmsup_f32_avx512((const float *)a->data, (const float *)b->data,
                           (float *)out->data, m, k, n, rsa, csa, rsb, rso);
#elif NUMC_HAVE_AVX2
        gemmsup_f32_avx2((const float *)a->data, (const float *)b->data,
                         (float *)out->data, m, k, n, rsa, csa, rsb, rso);
#elif NUMC_HAVE_SVE
        gemmsup_f32_sve((const float *)a->data, (const float *)b->data,
                        (float *)out->data, m, k, n, rsa, csa, rsb, rso);
#elif NUMC_HAVE_NEON
        gemmsup_f32_neon((const float *)a->data, (const float *)b->data,
                         (float *)out->data, m, k, n, rsa, csa, rsb, rso);
#elif NUMC_HAVE_RVV
        gemmsup_f32_rvv((const float *)a->data, (const float *)b->data,
                        (float *)out->data, m, k, n, rsa, csa, rsb, rso);
#endif
        return 0;
      }
      if (a->dtype == NUMC_DTYPE_FLOAT64) {
#if NUMC_HAVE_AVX512
        gemmsup_f64_avx512((const double *)a->data, (const double *)b->data,
                           (double *)out->data, m, k, n, rsa, csa, rsb, rso);
#elif NUMC_HAVE_AVX2
        gemmsup_f64_avx2((const double *)a->data, (const double *)b->data,
                         (double *)out->data, m, k, n, rsa, csa, rsb, rso);
#elif NUMC_HAVE_SVE
        gemmsup_f64_sve((const double *)a->data, (const double *)b->data,
                        (double *)out->data, m, k, n, rsa, csa, rsb, rso);
#elif NUMC_HAVE_NEON
        gemmsup_f64_neon((const double *)a->data, (const double *)b->data,
                         (double *)out->data, m, k, n, rsa, csa, rsb, rso);
#elif NUMC_HAVE_RVV
        gemmsup_f64_rvv((const double *)a->data, (const double *)b->data,
                        (double *)out->data, m, k, n, rsa, csa, rsb, rso);
#endif
        return 0;
      }
      if (a->dtype == NUMC_DTYPE_INT32 || a->dtype == NUMC_DTYPE_UINT32) {
#if NUMC_HAVE_AVX2
        gemmsup_i32_avx2((const int32_t *)a->data, (const int32_t *)b->data,
                         (int32_t *)out->data, m, k, n, rsa, csa, rsb, rso);
#endif
        return 0;
      }
      if (a->dtype == NUMC_DTYPE_INT16 || a->dtype == NUMC_DTYPE_UINT16) {
#if NUMC_HAVE_AVX2
        gemmsup_i16_avx2((const int16_t *)a->data, (const int16_t *)b->data,
                         (int16_t *)out->data, m, k, n, rsa, csa, rsb, rso);
#endif
        return 0;
      }
      if (a->dtype == NUMC_DTYPE_INT64 || a->dtype == NUMC_DTYPE_UINT64) {
#if NUMC_HAVE_AVX2
        gemmsup_i64_avx2((const int64_t *)a->data, (const int64_t *)b->data,
                         (int64_t *)out->data, m, k, n, rsa, csa, rsb, rso);
#endif
        return 0;
      }
    }
  }
#endif

  /* Primary path: packed SIMD GEMM (all types, all strides) */
  {
    GemmSimdKernel simd_kern = gemm_simd_table[a->dtype];
    if (simd_kern) {
      size_t m = a->shape[0], k = a->shape[1], n = b->shape[1];
      size_t elem = numc_dtype_size(a->dtype);
      intptr_t rsa = (intptr_t)(a->strides[0] / elem);
      intptr_t csa = (intptr_t)(a->strides[1] / elem);
      intptr_t rsb = (intptr_t)(b->strides[0] / elem);
      intptr_t rso = (intptr_t)(out->strides[0] / elem);
      simd_kern(a->data, b->data, out->data, m, k, n, rsa, csa, rsb, rso);
      return 0;
    }
  }

  /* Fallback to naive kernels (C23 + OpenMP) */
  MatmulKernel kern = matmul_table[a->dtype];
  memset(out->data, 0, out->capacity);
  kern((const char *)a->data, (const char *)b->data, (char *)out->data,
       a->shape[0], b->shape[0], out->shape[1]);
  return 0;
}
