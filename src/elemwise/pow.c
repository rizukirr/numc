#include "dispatch.h"
#include "helpers.h"
#include "numc/dtype.h"
#include <numc/math.h>

#include "arch_dispatch.h"
#if NUMC_HAVE_AVX2
#include "intrinsics/math_avx2.h"
#endif

/* ── Stamp out pow loop kernels ──────────────────────────────────────── */

/* 8/16-bit signed: branchless fixed-iteration (auto-vectorizes) */
DEFINE_BINARY_KERNEL(pow, NUMC_DTYPE_INT8, NUMC_INT8, _powi_i8(in1, in2))
DEFINE_BINARY_KERNEL(pow, NUMC_DTYPE_INT16, NUMC_INT16, _powi_i16(in1, in2))

/* 8/16-bit unsigned: branchless fixed-iteration (auto-vectorizes) */
DEFINE_BINARY_KERNEL(pow, NUMC_DTYPE_UINT8, NUMC_UINT8, _powi_u8(in1, in2))
DEFINE_BINARY_KERNEL(pow, NUMC_DTYPE_UINT16, NUMC_UINT16, _powi_u16(in1, in2))

/* 32/64-bit: variable-iteration early-exit (scalar, fast for small exp) */
DEFINE_BINARY_KERNEL_NOSIMD(pow, NUMC_DTYPE_INT32, NUMC_INT32,
                            (NUMC_INT32)_powi_signed((NUMC_INT64)in1,
                                                     (NUMC_INT64)in2))
DEFINE_BINARY_KERNEL_NOSIMD(pow, NUMC_DTYPE_UINT32, NUMC_UINT32,
                            (NUMC_UINT32)_powi_unsigned((NUMC_UINT64)in1,
                                                        (NUMC_UINT64)in2))
DEFINE_BINARY_KERNEL_NOSIMD(pow, NUMC_DTYPE_INT64, NUMC_INT64,
                            (NUMC_INT64)_powi_signed((NUMC_INT64)in1,
                                                     (NUMC_INT64)in2))
DEFINE_BINARY_KERNEL_NOSIMD(pow, NUMC_DTYPE_UINT64, NUMC_UINT64,
                            (NUMC_UINT64)_powi_unsigned((NUMC_UINT64)in1,
                                                        (NUMC_UINT64)in2))

/* float32/float64: SIMD exp(in2 * log(in1)) on AVX2, scalar fallback */
#if NUMC_HAVE_AVX2
static void _kern_pow_NUMC_DTYPE_FLOAT32(const char *a, const char *b,
                                         char *out, size_t n, intptr_t sa,
                                         intptr_t sb, intptr_t so) {
  const intptr_t es = (intptr_t)sizeof(float);
  if (sa == es && sb == es && so == es) {
    const float *pa = (const float *)a;
    const float *pb = (const float *)b;
    float *po = (float *)out;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
      __m256 va = _mm256_loadu_ps(pa + i);
      __m256 vb = _mm256_loadu_ps(pb + i);
      _mm256_storeu_ps(po + i,
                       _mm256_exp_ps(_mm256_mul_ps(vb, _mm256_log_ps(va))));
    }
    for (; i < n; i++)
      po[i] = _exp_f32(pb[i] * _log_f32(pa[i]));
  } else {
    for (size_t i = 0; i < n; i++) {
      float in1 = *(const float *)(a + i * sa);
      float in2 = *(const float *)(b + i * sb);
      *(float *)(out + i * so) = _exp_f32(in2 * _log_f32(in1));
    }
  }
}

static void _kern_pow_NUMC_DTYPE_FLOAT64(const char *a, const char *b,
                                         char *out, size_t n, intptr_t sa,
                                         intptr_t sb, intptr_t so) {
  const intptr_t es = (intptr_t)sizeof(double);
  if (sa == es && sb == es && so == es) {
    const double *pa = (const double *)a;
    const double *pb = (const double *)b;
    double *po = (double *)out;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
      __m256d va = _mm256_loadu_pd(pa + i);
      __m256d vb = _mm256_loadu_pd(pb + i);
      _mm256_storeu_pd(po + i,
                       _mm256_exp_pd(_mm256_mul_pd(vb, _mm256_log_pd(va))));
    }
    for (; i < n; i++)
      po[i] = _exp_f64(pb[i] * _log_f64(pa[i]));
  } else {
    for (size_t i = 0; i < n; i++) {
      double in1 = *(const double *)(a + i * sa);
      double in2 = *(const double *)(b + i * sb);
      *(double *)(out + i * so) = _exp_f64(in2 * _log_f64(in1));
    }
  }
}
#else
DEFINE_BINARY_KERNEL_NOSIMD(pow, NUMC_DTYPE_FLOAT32, NUMC_FLOAT32,
                            _exp_f32(in2 *_log_f32(in1)))
DEFINE_BINARY_KERNEL_NOSIMD(pow, NUMC_DTYPE_FLOAT64, NUMC_FLOAT64,
                            _exp_f64(in2 *_log_f64(in1)))
#endif

/* ── Dispatch table ──────────────────────────────────────────────── */

static const NumcBinaryKernel pow_table[] = {
    E(pow, NUMC_DTYPE_INT8),    E(pow, NUMC_DTYPE_INT16),
    E(pow, NUMC_DTYPE_INT32),   E(pow, NUMC_DTYPE_INT64),
    E(pow, NUMC_DTYPE_UINT8),   E(pow, NUMC_DTYPE_UINT16),
    E(pow, NUMC_DTYPE_UINT32),  E(pow, NUMC_DTYPE_UINT64),
    E(pow, NUMC_DTYPE_FLOAT32), E(pow, NUMC_DTYPE_FLOAT64),
};

/* ── Public API ──────────────────────────────────────────────────── */

/* pow: non-const signature differs, stays explicit */
int numc_pow(NumcArray *a, NumcArray *b, NumcArray *out) {
  int err = _check_binary(a, b, out);
  if (err)
    return err;
  _binary_op(a, b, out, pow_table);
  return 0;
}
