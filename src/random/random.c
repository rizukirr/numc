#include "kernel.h"
#include "internal.h"
#include <numc/array.h>
#include <numc/math.h>
#include <math.h>
#include <stdint.h>

#include "arch_dispatch.h"
#if NUMC_HAVE_AVX2
#include "intrinsics/math_avx2.h"
#endif

/* ── Stamp rand kernels for all 10 dtypes ───────────────────────────
 *
 * CONVERT_EXPR(raw) maps a raw uint64_t from the PRNG to C_TYPE.
 * It is called once per lane per loop iteration — a pure expression
 * on a local variable, enabling AVX2 vectorization.
 *
 * Float types:   uniform [0, 1) via IEEE 754 bit trick (_u64_to_f32/64).
 * Integer types: raw bits masked to type width.
 */

#define CONV_INT8(raw)   ((NUMC_INT8)((raw) & 0xFFu))
#define CONV_INT16(raw)  ((NUMC_INT16)((raw) & 0xFFFFu))
#define CONV_INT32(raw)  ((NUMC_INT32)((raw) & 0xFFFFFFFFu))
#define CONV_INT64(raw)  ((NUMC_INT64)(raw))
#define CONV_UINT8(raw)  ((NUMC_UINT8)((raw) & 0xFFu))
#define CONV_UINT16(raw) ((NUMC_UINT16)((raw) & 0xFFFFu))
#define CONV_UINT32(raw) ((NUMC_UINT32)((raw) & 0xFFFFFFFFu))
#define CONV_UINT64(raw) ((NUMC_UINT64)(raw))
#define CONV_F32(raw)    (_u64_to_f32(raw))
#define CONV_F64(raw)    (_u64_to_f64(raw))

DEFINE_RAND_KERNEL(NUMC_DTYPE_INT8, NUMC_INT8, CONV_INT8)
DEFINE_RAND_KERNEL(NUMC_DTYPE_INT16, NUMC_INT16, CONV_INT16)
DEFINE_RAND_KERNEL(NUMC_DTYPE_INT32, NUMC_INT32, CONV_INT32)
DEFINE_RAND_KERNEL(NUMC_DTYPE_INT64, NUMC_INT64, CONV_INT64)
DEFINE_RAND_KERNEL(NUMC_DTYPE_UINT8, NUMC_UINT8, CONV_UINT8)
DEFINE_RAND_KERNEL(NUMC_DTYPE_UINT16, NUMC_UINT16, CONV_UINT16)
DEFINE_RAND_KERNEL(NUMC_DTYPE_UINT32, NUMC_UINT32, CONV_UINT32)
DEFINE_RAND_KERNEL(NUMC_DTYPE_UINT64, NUMC_UINT64, CONV_UINT64)
DEFINE_RAND_KERNEL(NUMC_DTYPE_FLOAT32, NUMC_FLOAT32, CONV_F32)
DEFINE_RAND_KERNEL(NUMC_DTYPE_FLOAT64, NUMC_FLOAT64, CONV_F64)

#undef CONV_INT8
#undef CONV_INT16
#undef CONV_INT32
#undef CONV_INT64
#undef CONV_UINT8
#undef CONV_UINT16
#undef CONV_UINT32
#undef CONV_UINT64
#undef CONV_F32
#undef CONV_F64

/* ── Stamp randn kernels for all 10 dtypes ──────────────────────────
 *
 * Float types:    N(0,1) via Box-Muller transform (true normal samples).
 * Integer types:  N(0,1) double sample, truncated and cast.
 *                 Useful for near-zero initialisation of quantised
 *                 weights — most values land in [-3, 3] before cast.
 */

DEFINE_RANDN_KERNEL(NUMC_DTYPE_INT8, NUMC_INT8, (NUMC_INT8)_prng_normal_f64())
DEFINE_RANDN_KERNEL(NUMC_DTYPE_INT16, NUMC_INT16,
                    (NUMC_INT16)_prng_normal_f64())
DEFINE_RANDN_KERNEL(NUMC_DTYPE_INT32, NUMC_INT32,
                    (NUMC_INT32)_prng_normal_f64())
DEFINE_RANDN_KERNEL(NUMC_DTYPE_INT64, NUMC_INT64,
                    (NUMC_INT64)_prng_normal_f64())
DEFINE_RANDN_KERNEL(NUMC_DTYPE_UINT8, NUMC_UINT8,
                    (NUMC_UINT8)_prng_normal_f64())
DEFINE_RANDN_KERNEL(NUMC_DTYPE_UINT16, NUMC_UINT16,
                    (NUMC_UINT16)_prng_normal_f64())
DEFINE_RANDN_KERNEL(NUMC_DTYPE_UINT32, NUMC_UINT32,
                    (NUMC_UINT32)_prng_normal_f64())
DEFINE_RANDN_KERNEL(NUMC_DTYPE_UINT64, NUMC_UINT64,
                    (NUMC_UINT64)_prng_normal_f64())
/* float32 randn: SIMD Box-Muller on AVX2, scalar fallback otherwise.
 * Generates 8 normals per iteration from 8 uniforms (4 Box-Muller pairs).
 * SIMD log + sqrt for magnitude, SIMD sincos for angle. */
#if NUMC_HAVE_AVX2
static void _kern_randn_NUMC_DTYPE_FLOAT32(char *out, size_t n) {
  _prng_ensure_seeded();
  float *restrict po = (float *)out;
  const __m256 two_pi = _mm256_set1_ps(6.2831853071795864f);
  const __m256 neg2 = _mm256_set1_ps(-2.0f);
  const __m128 tiny128 = _mm_set1_ps(1e-20f);
  const __m128 two_pi128 = _mm_set1_ps(6.2831853071795864f);
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    /* Generate 8 uniform f32: u[0..3] = u1, u[4..7] = u2 */
    float u[8];
    for (int k = 0; k < 8; k++)
      u[k] = _prng_f32();
    __m128 u1 = _mm_max_ps(_mm_loadu_ps(u), tiny128);
    __m128 u2 = _mm_loadu_ps(u + 4);
    /* mag = sqrt(-2 * log(u1)) — 4 magnitudes */
    __m256 u1_256 = _mm256_insertf128_ps(_mm256_castps128_ps256(u1), u1, 1);
    __m128 mag = _mm256_castps256_ps128(
        _mm256_sqrt_ps(_mm256_mul_ps(neg2, _mm256_log_ps(u1_256))));
    /* angle = 2π * u2, then sincos */
    __m128 angle = _mm_mul_ps(u2, two_pi128);
    __m256 angle256 =
        _mm256_insertf128_ps(_mm256_castps128_ps256(angle), angle, 1);
    __m256 sin_v, cos_v;
    _mm256_sincos_ps(angle256, &sin_v, &cos_v);
    /* out[i..i+3] = mag * cos, out[i+4..i+7] = mag * sin */
    _mm_storeu_ps(po + i, _mm_mul_ps(mag, _mm256_castps256_ps128(cos_v)));
    _mm_storeu_ps(po + i + 4, _mm_mul_ps(mag, _mm256_castps256_ps128(sin_v)));
  }
  for (; i < n; i++)
    po[i] = _prng_normal_f32();
}
#else
DEFINE_RANDN_KERNEL(NUMC_DTYPE_FLOAT32, NUMC_FLOAT32, _prng_normal_f32())
#endif
DEFINE_RANDN_KERNEL(NUMC_DTYPE_FLOAT64, NUMC_FLOAT64, _prng_normal_f64())

/* ── Dispatch tables (dtype -> kernel) ─────────────────────────────*/

static const NumcRandKernel rand_table[] = {
    ER(rand, NUMC_DTYPE_INT8),    ER(rand, NUMC_DTYPE_INT16),
    ER(rand, NUMC_DTYPE_INT32),   ER(rand, NUMC_DTYPE_INT64),
    ER(rand, NUMC_DTYPE_UINT8),   ER(rand, NUMC_DTYPE_UINT16),
    ER(rand, NUMC_DTYPE_UINT32),  ER(rand, NUMC_DTYPE_UINT64),
    ER(rand, NUMC_DTYPE_FLOAT32), ER(rand, NUMC_DTYPE_FLOAT64),
};

static const NumcRandKernel randn_table[] = {
    ER(randn, NUMC_DTYPE_INT8),    ER(randn, NUMC_DTYPE_INT16),
    ER(randn, NUMC_DTYPE_INT32),   ER(randn, NUMC_DTYPE_INT64),
    ER(randn, NUMC_DTYPE_UINT8),   ER(randn, NUMC_DTYPE_UINT16),
    ER(randn, NUMC_DTYPE_UINT32),  ER(randn, NUMC_DTYPE_UINT64),
    ER(randn, NUMC_DTYPE_FLOAT32), ER(randn, NUMC_DTYPE_FLOAT64),
};

/* ── Shared creation helper ─────────────────────────────────────────*/

static NumcArray *_rand_impl(NumcCtx *ctx, const size_t *shape, size_t dim,
                             NumcDType dtype, const NumcRandKernel *table) {
  if (!ctx || !shape || dim == 0) {
    NUMC_SET_ERROR(NUMC_ERR_NULL,
                   "rand: NULL or invalid args (ctx=%p shape=%p dim=%zu)", ctx,
                   shape, dim);
    return NULL;
  }

  NumcArray *arr = numc_array_create(ctx, shape, dim, dtype);
  if (!arr)
    return NULL;

  table[dtype]((char *)arr->data, arr->size);
  return arr;
}

/* ── Public API ─────────────────────────────────────────────────────*/

void numc_manual_seed(uint64_t seed) {
  prng_seed(seed);
}

NumcArray *numc_array_rand(NumcCtx *ctx, const size_t *shape, size_t dim,
                           NumcDType dtype) {
  return _rand_impl(ctx, shape, dim, dtype, rand_table);
}

NumcArray *numc_array_randn(NumcCtx *ctx, const size_t *shape, size_t dim,
                            NumcDType dtype) {
  return _rand_impl(ctx, shape, dim, dtype, randn_table);
}

/* He (Kaiming): scale randn by sqrt(2 / fan_in) */
NumcArray *numc_array_random_he(NumcCtx *ctx, const size_t *shape, size_t dim,
                                NumcDType dtype, size_t fan_in) {
  if (fan_in == 0) {
    NUMC_SET_ERROR(NUMC_ERR_SHAPE, "random_he: fan_in must be > 0");
    return NULL;
  }

  NumcArray *arr = numc_array_randn(ctx, shape, dim, dtype);
  if (!arr)
    return NULL;

  double scale = sqrt(2.0 / (double)fan_in);
  if (numc_mul_scalar_inplace(arr, scale) < 0)
    return NULL;
  return arr;
}

/* Xavier (Glorot): uniform [-limit, limit),
 * limit = sqrt(6 / (fan_in + fan_out)) */
NumcArray *numc_array_random_xavier(NumcCtx *ctx, const size_t *shape,
                                    size_t dim, NumcDType dtype, size_t fan_in,
                                    size_t fan_out) {
  if (fan_in == 0 || fan_out == 0) {
    NUMC_SET_ERROR(NUMC_ERR_SHAPE,
                   "random_xavier: fan_in and fan_out must be > 0");
    return NULL;
  }

  NumcArray *arr = numc_array_rand(ctx, shape, dim, dtype);
  if (!arr)
    return NULL;

  /* rand gives [0, 1) — shift and scale to [-limit, limit) */
  double limit = sqrt(6.0 / (double)(fan_in + fan_out));
  if (numc_mul_scalar_inplace(arr, 2.0 * limit) < 0) /* [0, 2*limit)    */
    return NULL;
  if (numc_sub_scalar_inplace(arr, limit) < 0) /* [-limit, limit) */
    return NULL;
  return arr;
}
