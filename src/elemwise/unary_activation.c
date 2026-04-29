#include "unary_dispatch.h"

#include <numc/math.h>
#include "math_helpers.h"

#include "arch_dispatch.h"
#if NUMC_HAVE_AVX512
#include "intrinsics/math_avx512.h"
#elif NUMC_HAVE_AVX2
#include "intrinsics/math_avx2.h"
#elif NUMC_HAVE_SVE
#include "intrinsics/math_sve.h"
#elif NUMC_HAVE_NEON
#include "intrinsics/math_neon.h"
#endif
#if NUMC_HAVE_RVV
#include "intrinsics/math_rvv.h"
#endif


static inline float _sigmoid_f32(float x) {
  float z = 0.0f;
  if (x > 0.0f) {
    z = _exp_f32(-x);
    return 1.0f / (1.0f + z);
  }
  z = _exp_f32(x);
  return z / (1.0f + z);
}

static inline double _sigmoid_f64(double x) {
  double z = 0.0;
  if (x > 0.0) {
    z = _exp_f64(-x);
    return 1.0 / (1.0 + z);
  }
  z = _exp_f64(x);
  return z / (1.0 + z);
}

/* -- Stamp out sigmoid loop kernels -------------------------------------- */

/* int8/int16/uint8/uint16: cast through float32 */
#define STAMP_SIGMOID_SMALL(TE, CT) \
  DEFINE_UNARY_KERNEL(sigmoid, TE, CT, (CT)_sigmoid_f32((float)in1))
GENERATE_INT8_INT16_NUMC_TYPES(STAMP_SIGMOID_SMALL)
#undef STAMP_SIGMOID_SMALL

/* int32/uint32: cast through float64 */
#define STAMP_SIGMOID_I32(TE, CT) \
  DEFINE_UNARY_KERNEL(sigmoid, TE, CT, (CT)_sigmoid_f64((double)in1))
GENERATE_INT32_NUMC_TYPES(STAMP_SIGMOID_I32)
#undef STAMP_SIGMOID_I32

/* int64/uint64: cast through float64 */
#define STAMP_SIGMOID_I64(TE, CT) \
  DEFINE_UNARY_KERNEL(sigmoid, TE, CT, (CT)_sigmoid_f64((double)in1))
GENERATE_INT64_NUMC_TYPES(STAMP_SIGMOID_I64)
#undef STAMP_SIGMOID_I64

DEFINE_UNARY_KERNEL(sigmoid, NUMC_DTYPE_FLOAT32, NUMC_FLOAT32,
                    _sigmoid_f32(in1))
DEFINE_UNARY_KERNEL(sigmoid, NUMC_DTYPE_FLOAT64, NUMC_FLOAT64,
                    _sigmoid_f64(in1))

static const NumcUnaryKernel sigmoid_table[] = {
    E(sigmoid, NUMC_DTYPE_INT8),    E(sigmoid, NUMC_DTYPE_INT16),
    E(sigmoid, NUMC_DTYPE_INT32),   E(sigmoid, NUMC_DTYPE_INT64),
    E(sigmoid, NUMC_DTYPE_UINT8),   E(sigmoid, NUMC_DTYPE_UINT16),
    E(sigmoid, NUMC_DTYPE_UINT32),  E(sigmoid, NUMC_DTYPE_UINT64),
    E(sigmoid, NUMC_DTYPE_FLOAT32), E(sigmoid, NUMC_DTYPE_FLOAT64),
};

/* -- Stamp out tanh loop kernels --------------------------------------- */

/* int8/int16/uint8/uint16: cast through float32 */
#define STAMP_TANH_SMALL(TE, CT) \
  DEFINE_UNARY_KERNEL(tanh, TE, CT, (CT)tanhf((float)in1))
GENERATE_INT8_INT16_NUMC_TYPES(STAMP_TANH_SMALL)
#undef STAMP_TANH_SMALL

/* int32/uint32: cast through float64 */
#define STAMP_TANH_I32(TE, CT) \
  DEFINE_UNARY_KERNEL(tanh, TE, CT, (CT)tanh((double)in1))
GENERATE_INT32_NUMC_TYPES(STAMP_TANH_I32)
#undef STAMP_TANH_I32

/* int64/uint64: cast through float64 */
#define STAMP_TANH_I64(TE, CT) \
  DEFINE_UNARY_KERNEL(tanh, TE, CT, (CT)tanh((double)in1))
GENERATE_INT64_NUMC_TYPES(STAMP_TANH_I64)
#undef STAMP_TANH_I64

/* float32/float64 */
#if NUMC_HAVE_AVX512
static void _kern_tanh_NUMC_DTYPE_FLOAT32(const char *a, char *out, size_t n,
                                          intptr_t sa, intptr_t so) {
  const intptr_t es = (intptr_t)sizeof(float);
  if (sa == es && so == es) {
    const float *pa = (const float *)a;
    float *po = (float *)out;
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
      __m512 va = _mm512_loadu_ps(pa + i);
      _mm512_storeu_ps(po + i, _mm512_tanh_ps(va));
    }
    for (; i < n; i++)
      po[i] = tanhf(pa[i]);
  } else {
    for (size_t i = 0; i < n; i++) {
      float in1 = *(const float *)(a + i * sa);
      *(float *)(out + i * so) = tanhf(in1);
    }
  }
}

static void _kern_tanh_NUMC_DTYPE_FLOAT64(const char *a, char *out, size_t n,
                                          intptr_t sa, intptr_t so) {
  const intptr_t es = (intptr_t)sizeof(double);
  if (sa == es && so == es) {
    const double *pa = (const double *)a;
    double *po = (double *)out;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
      __m512d va = _mm512_loadu_pd(pa + i);
      _mm512_storeu_pd(po + i, _mm512_tanh_pd(va));
    }
    for (; i < n; i++)
      po[i] = tanh(pa[i]);
  } else {
    for (size_t i = 0; i < n; i++) {
      double in1 = *(const double *)(a + i * sa);
      *(double *)(out + i * so) = tanh(in1);
    }
  }
}
#elif NUMC_HAVE_AVX2
static void _kern_tanh_NUMC_DTYPE_FLOAT32(const char *a, char *out, size_t n,
                                          intptr_t sa, intptr_t so) {
  const intptr_t es = (intptr_t)sizeof(float);
  if (sa == es && so == es) {
    const float *pa = (const float *)a;
    float *po = (float *)out;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
      __m256 va = _mm256_loadu_ps(pa + i);
      _mm256_storeu_ps(po + i, _mm256_tanh_ps(va));
    }
    for (; i < n; i++)
      po[i] = tanhf(pa[i]);
  } else {
    for (size_t i = 0; i < n; i++) {
      float in1 = *(const float *)(a + i * sa);
      *(float *)(out + i * so) = tanhf(in1);
    }
  }
}

static void _kern_tanh_NUMC_DTYPE_FLOAT64(const char *a, char *out, size_t n,
                                          intptr_t sa, intptr_t so) {
  const intptr_t es = (intptr_t)sizeof(double);
  if (sa == es && so == es) {
    const double *pa = (const double *)a;
    double *po = (double *)out;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
      __m256d va = _mm256_loadu_pd(pa + i);
      _mm256_storeu_pd(po + i, _mm256_tanh_pd(va));
    }
    for (; i < n; i++)
      po[i] = tanh(pa[i]);
  } else {
    for (size_t i = 0; i < n; i++) {
      double in1 = *(const double *)(a + i * sa);
      *(double *)(out + i * so) = tanh(in1);
    }
  }
}
#else
DEFINE_UNARY_KERNEL(tanh, NUMC_DTYPE_FLOAT32, NUMC_FLOAT32, tanhf(in1))
DEFINE_UNARY_KERNEL(tanh, NUMC_DTYPE_FLOAT64, NUMC_FLOAT64, tanh(in1))
#endif

static const NumcUnaryKernel tanh_table[] = {
    E(tanh, NUMC_DTYPE_INT8),    E(tanh, NUMC_DTYPE_INT16),
    E(tanh, NUMC_DTYPE_INT32),   E(tanh, NUMC_DTYPE_INT64),
    E(tanh, NUMC_DTYPE_UINT8),   E(tanh, NUMC_DTYPE_UINT16),
    E(tanh, NUMC_DTYPE_UINT32),  E(tanh, NUMC_DTYPE_UINT64),
    E(tanh, NUMC_DTYPE_FLOAT32), E(tanh, NUMC_DTYPE_FLOAT64),
};

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

#if NUMC_HAVE_AVX512 || NUMC_HAVE_AVX2 || NUMC_HAVE_SVE || NUMC_HAVE_NEON || \
    NUMC_HAVE_RVV
static const FastUnKern tanh_fast_table[] = {
    [NUMC_DTYPE_FLOAT32] = FUN(tanh, f32),
    [NUMC_DTYPE_FLOAT64] = FUN(tanh, f64),
};

static const FastUnKern sigmoid_fast_table[] = {
    [NUMC_DTYPE_FLOAT32] = FUN(sigmoid, f32),
    [NUMC_DTYPE_FLOAT64] = FUN(sigmoid, f64),
};

DEFINE_UNARY_SIMD(tanh, tanh_fast_table, tanh_table)
DEFINE_UNARY_SIMD(sigmoid, sigmoid_fast_table, sigmoid_table)
#else
DEFINE_ELEMWISE_UNARY(tanh, tanh_table)
DEFINE_ELEMWISE_UNARY(sigmoid, sigmoid_table)
#endif
