#include "dispatch.h"
#include "helpers.h"
#include <math.h>
#include <numc/math.h>

#include "arch_dispatch.h"
#if NUMC_HAVE_AVX2
#include "intrinsics/math_avx2.h"
#endif

/* ── Stamp unary neg loop typed kernels ────────────────────*/

/* neg: all 10 types, native - */
#define STAMP_NEG(TE, CT) DEFINE_UNARY_KERNEL(neg, TE, CT, -in1)
GENERATE_NUMC_TYPES(STAMP_NEG)
#undef STAMP_NEG

/* ── Stamp unary abs loop typed kernels ────────────────────*/

/* Signed types: conditional negate */
#define STAMP_ABS(TE, CT) \
  DEFINE_UNARY_KERNEL(abs, TE, CT, (CT)(in1 < 0 ? -in1 : in1))
GENERATE_SIGNED_INT8_INT16_INT32_NUMC_TYPES(STAMP_ABS)
GENERATE_SIGNED_64BIT_NUMC_TYPES(STAMP_ABS)
#undef STAMP_ABS
/* Float types: branchless conditional */
DEFINE_UNARY_KERNEL(abs, NUMC_DTYPE_FLOAT32, NUMC_FLOAT32,
                    (NUMC_FLOAT32)(in1 < 0.0f ? -in1 : in1))
DEFINE_UNARY_KERNEL(abs, NUMC_DTYPE_FLOAT64, NUMC_FLOAT64,
                    (NUMC_FLOAT64)(in1 < 0.0 ? -in1 : in1))

/* ── Stamp out log loop kernels (stride-aware, wrapping scalar bit-manip) ── */

/* < 32-bit integers: cast through float */
#define STAMP_LOG_SMALL(TE, CT) \
  DEFINE_UNARY_KERNEL_NOSIMD(log, TE, CT, (CT)_log_f32((float)in1))
GENERATE_INT8_INT16_NUMC_TYPES(STAMP_LOG_SMALL)
#undef STAMP_LOG_SMALL

/* 32-bit integers: cast through double */
#define STAMP_LOG_I32(TE, CT) \
  DEFINE_UNARY_KERNEL_NOSIMD(log, TE, CT, (CT)_log_f64((double)in1))
GENERATE_INT32_NUMC_TYPES(STAMP_LOG_I32)
#undef STAMP_LOG_I32

/* 64-bit integers: cast through double */
#define STAMP_LOG_I64(TE, CT) \
  DEFINE_UNARY_KERNEL_NOSIMD(log, TE, CT, (CT)_log_f64((double)in1))
GENERATE_SIGNED_64BIT_NUMC_TYPES(STAMP_LOG_I64)
#undef STAMP_LOG_I64
DEFINE_UNARY_KERNEL_NOSIMD(log, NUMC_DTYPE_UINT64, NUMC_UINT64,
                           (NUMC_UINT64)_log_f64((double)in1))

/* float types: SIMD fast path on AVX2, scalar fallback otherwise */
#if NUMC_HAVE_AVX2
static void _kern_log_NUMC_DTYPE_FLOAT32(const char *a, char *out, size_t n,
                                         intptr_t sa, intptr_t so) {
  const intptr_t es = (intptr_t)sizeof(float);
  if (sa == es && so == es) {
    const float *pa = (const float *)a;
    float *po = (float *)out;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
      __m256 va = _mm256_loadu_ps(pa + i);
      _mm256_storeu_ps(po + i, _mm256_log_ps(va));
    }
    for (; i < n; i++)
      po[i] = _log_f32(pa[i]);
  } else {
    for (size_t i = 0; i < n; i++) {
      float in1 = *(const float *)(a + i * sa);
      *(float *)(out + i * so) = _log_f32(in1);
    }
  }
}

static void _kern_log_NUMC_DTYPE_FLOAT64(const char *a, char *out, size_t n,
                                         intptr_t sa, intptr_t so) {
  const intptr_t es = (intptr_t)sizeof(double);
  if (sa == es && so == es) {
    const double *pa = (const double *)a;
    double *po = (double *)out;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
      __m256d va = _mm256_loadu_pd(pa + i);
      _mm256_storeu_pd(po + i, _mm256_log_pd(va));
    }
    for (; i < n; i++)
      po[i] = _log_f64(pa[i]);
  } else {
    for (size_t i = 0; i < n; i++) {
      double in1 = *(const double *)(a + i * sa);
      *(double *)(out + i * so) = _log_f64(in1);
    }
  }
}
#else
DEFINE_UNARY_KERNEL_NOSIMD(log, NUMC_DTYPE_FLOAT32, NUMC_FLOAT32, _log_f32(in1))
DEFINE_UNARY_KERNEL_NOSIMD(log, NUMC_DTYPE_FLOAT64, NUMC_FLOAT64, _log_f64(in1))
#endif

/* ── Stamp out exp loop kernels ─────────────────────────────────────── */

/* int8/int16/uint8/uint16: cast through float32 */
#define STAMP_EXP_SMALL(TE, CT) \
  DEFINE_UNARY_KERNEL_NOSIMD(exp, TE, CT, (CT)_exp_f32((float)in1))
GENERATE_INT8_INT16_NUMC_TYPES(STAMP_EXP_SMALL)
#undef STAMP_EXP_SMALL

/* int32/uint32: cast through float64 */
#define STAMP_EXP_I32(TE, CT) \
  DEFINE_UNARY_KERNEL_NOSIMD(exp, TE, CT, (CT)_exp_f64((double)in1))
GENERATE_INT32_NUMC_TYPES(STAMP_EXP_I32)
#undef STAMP_EXP_I32

/* int64: cast through float64 */
#define STAMP_EXP_I64(TE, CT) \
  DEFINE_UNARY_KERNEL_NOSIMD(exp, TE, CT, (CT)_exp_f64((double)in1))
GENERATE_SIGNED_64BIT_NUMC_TYPES(STAMP_EXP_I64)
#undef STAMP_EXP_I64

/* uint64: explicit — no X-macro covers just uint64 */
DEFINE_UNARY_KERNEL_NOSIMD(exp, NUMC_DTYPE_UINT64, NUMC_UINT64,
                           (NUMC_UINT64)_exp_f64((double)in1))

/* float32/float64: SIMD fast path on AVX2, scalar fallback */
#if NUMC_HAVE_AVX2
static void _kern_exp_NUMC_DTYPE_FLOAT32(const char *a, char *out, size_t n,
                                         intptr_t sa, intptr_t so) {
  const intptr_t es = (intptr_t)sizeof(float);
  if (sa == es && so == es) {
    const float *pa = (const float *)a;
    float *po = (float *)out;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
      __m256 va = _mm256_loadu_ps(pa + i);
      _mm256_storeu_ps(po + i, _mm256_exp_ps(va));
    }
    for (; i < n; i++)
      po[i] = _exp_f32(pa[i]);
  } else {
    for (size_t i = 0; i < n; i++) {
      float in1 = *(const float *)(a + i * sa);
      *(float *)(out + i * so) = _exp_f32(in1);
    }
  }
}

static void _kern_exp_NUMC_DTYPE_FLOAT64(const char *a, char *out, size_t n,
                                         intptr_t sa, intptr_t so) {
  const intptr_t es = (intptr_t)sizeof(double);
  if (sa == es && so == es) {
    const double *pa = (const double *)a;
    double *po = (double *)out;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
      __m256d va = _mm256_loadu_pd(pa + i);
      _mm256_storeu_pd(po + i, _mm256_exp_pd(va));
    }
    for (; i < n; i++)
      po[i] = _exp_f64(pa[i]);
  } else {
    for (size_t i = 0; i < n; i++) {
      double in1 = *(const double *)(a + i * sa);
      *(double *)(out + i * so) = _exp_f64(in1);
    }
  }
}
#else
DEFINE_UNARY_KERNEL_NOSIMD(exp, NUMC_DTYPE_FLOAT32, NUMC_FLOAT32, _exp_f32(in1))
DEFINE_UNARY_KERNEL_NOSIMD(exp, NUMC_DTYPE_FLOAT64, NUMC_FLOAT64, _exp_f64(in1))
#endif

/* ── Stamp unary sqrt loop typed kernels ─────────────────────────────────
 * float32: sqrtf -> hardware vsqrtps (auto-vectorized, -O3 -march=native)
 * float64: sqrt  -> hardware vsqrtpd (auto-vectorized)
 * signed integers:   clamp negative to 0 before cast (sqrt of negative is UB)
 * unsigned integers: always non-negative, cast directly
 * <32-bit: cast through float32; 32-bit+: cast through float64
 */

/* signed small: clamp negative to 0, cast through float32 */
DEFINE_UNARY_KERNEL(sqrt, NUMC_DTYPE_INT8, NUMC_INT8,
                    (NUMC_INT8)sqrtf((float)(in1 < 0 ? 0 : in1)))
DEFINE_UNARY_KERNEL(sqrt, NUMC_DTYPE_INT16, NUMC_INT16,
                    (NUMC_INT16)sqrtf((float)(in1 < 0 ? 0 : in1)))

/* unsigned small: cast through float32 */
DEFINE_UNARY_KERNEL(sqrt, NUMC_DTYPE_UINT8, NUMC_UINT8,
                    (NUMC_UINT8)sqrtf((float)in1))
DEFINE_UNARY_KERNEL(sqrt, NUMC_DTYPE_UINT16, NUMC_UINT16,
                    (NUMC_UINT16)sqrtf((float)in1))

/* int32: clamp, cast through float64 */
DEFINE_UNARY_KERNEL(sqrt, NUMC_DTYPE_INT32, NUMC_INT32,
                    (NUMC_INT32)sqrt((double)(in1 < 0 ? 0 : in1)))

/* uint32: cast through float64 */
DEFINE_UNARY_KERNEL(sqrt, NUMC_DTYPE_UINT32, NUMC_UINT32,
                    (NUMC_UINT32)sqrt((double)in1))

/* int64: clamp, cast through float64 */
DEFINE_UNARY_KERNEL(sqrt, NUMC_DTYPE_INT64, NUMC_INT64,
                    (NUMC_INT64)sqrt((double)(in1 < 0 ? 0 : in1)))

/* uint64: cast through float64 */
DEFINE_UNARY_KERNEL(sqrt, NUMC_DTYPE_UINT64, NUMC_UINT64,
                    (NUMC_UINT64)sqrt((double)in1))

/* float32: sqrtf -> hardware vsqrtps (auto-vectorized) */
DEFINE_UNARY_KERNEL(sqrt, NUMC_DTYPE_FLOAT32, NUMC_FLOAT32, sqrtf(in1))

/* float64: sqrt -> hardware vsqrtpd (auto-vectorized) */
DEFINE_UNARY_KERNEL(sqrt, NUMC_DTYPE_FLOAT64, NUMC_FLOAT64, sqrt(in1))

/* ── Dispatch tables (dtype -> kernel) ─────────────────────────────── */

static const NumcUnaryKernel neg_table[] = {
    E(neg, NUMC_DTYPE_INT8),    E(neg, NUMC_DTYPE_INT16),
    E(neg, NUMC_DTYPE_INT32),   E(neg, NUMC_DTYPE_INT64),
    E(neg, NUMC_DTYPE_UINT8),   E(neg, NUMC_DTYPE_UINT16),
    E(neg, NUMC_DTYPE_UINT32),  E(neg, NUMC_DTYPE_UINT64),
    E(neg, NUMC_DTYPE_FLOAT32), E(neg, NUMC_DTYPE_FLOAT64),
};

static const NumcUnaryKernel abs_table[] = {
    E(abs, NUMC_DTYPE_INT8),    E(abs, NUMC_DTYPE_INT16),
    E(abs, NUMC_DTYPE_INT32),   E(abs, NUMC_DTYPE_INT64),
    E(abs, NUMC_DTYPE_FLOAT32), E(abs, NUMC_DTYPE_FLOAT64),
    /* unsigned types: NULL — never reached; numc_abs/_inplace guards them */
};

static const NumcUnaryKernel log_table[] = {
    E(log, NUMC_DTYPE_INT8),    E(log, NUMC_DTYPE_INT16),
    E(log, NUMC_DTYPE_INT32),   E(log, NUMC_DTYPE_INT64),
    E(log, NUMC_DTYPE_UINT8),   E(log, NUMC_DTYPE_UINT16),
    E(log, NUMC_DTYPE_UINT32),  E(log, NUMC_DTYPE_UINT64),
    E(log, NUMC_DTYPE_FLOAT32), E(log, NUMC_DTYPE_FLOAT64),
};

static const NumcUnaryKernel exp_table[] = {
    E(exp, NUMC_DTYPE_INT8),    E(exp, NUMC_DTYPE_INT16),
    E(exp, NUMC_DTYPE_INT32),   E(exp, NUMC_DTYPE_INT64),
    E(exp, NUMC_DTYPE_UINT8),   E(exp, NUMC_DTYPE_UINT16),
    E(exp, NUMC_DTYPE_UINT32),  E(exp, NUMC_DTYPE_UINT64),
    E(exp, NUMC_DTYPE_FLOAT32), E(exp, NUMC_DTYPE_FLOAT64),
};

static const NumcUnaryKernel sqrt_table[] = {
    E(sqrt, NUMC_DTYPE_INT8),    E(sqrt, NUMC_DTYPE_INT16),
    E(sqrt, NUMC_DTYPE_INT32),   E(sqrt, NUMC_DTYPE_INT64),
    E(sqrt, NUMC_DTYPE_UINT8),   E(sqrt, NUMC_DTYPE_UINT16),
    E(sqrt, NUMC_DTYPE_UINT32),  E(sqrt, NUMC_DTYPE_UINT64),
    E(sqrt, NUMC_DTYPE_FLOAT32), E(sqrt, NUMC_DTYPE_FLOAT64),
};

/* ═══════════════════════════════════════════════════════════════════════
 * Public API — Unary ops
 * ═══════════════════════════════════════════════════════════════════════ */

#define DEFINE_ELEMWISE_UNARY(NAME, TABLE)        \
  int numc_##NAME(NumcArray *a, NumcArray *out) { \
    int err = _check_unary(a, out);               \
    if (err)                                      \
      return err;                                 \
    return _unary_op(a, out, TABLE);              \
  }                                               \
  int numc_##NAME##_inplace(NumcArray *a) {       \
    return _unary_op_inplace(a, TABLE);           \
  }

DEFINE_ELEMWISE_UNARY(neg, neg_table)

/* abs: unsigned types are always non-negative — just copy (or no-op). */
static inline bool _dtype_is_unsigned(NumcDType dt) {
  return dt == NUMC_DTYPE_UINT8 || dt == NUMC_DTYPE_UINT16 ||
         dt == NUMC_DTYPE_UINT32 || dt == NUMC_DTYPE_UINT64;
}

int numc_abs(NumcArray *a, NumcArray *out) {
  int err = _check_unary(a, out);
  if (err)
    return err;
  if (_dtype_is_unsigned(a->dtype)) {
    memcpy(out->data, a->data, a->capacity);
    return 0;
  }
  return _unary_op(a, out, abs_table);
}

int numc_abs_inplace(NumcArray *a) {
  if (!a)
    return NUMC_ERR_NULL;
  if (_dtype_is_unsigned(a->dtype))
    return 0; /* already non-negative, nothing to do */
  return _unary_op_inplace(a, abs_table);
}
DEFINE_ELEMWISE_UNARY(log, log_table)
DEFINE_ELEMWISE_UNARY(exp, exp_table)
DEFINE_ELEMWISE_UNARY(sqrt, sqrt_table)
