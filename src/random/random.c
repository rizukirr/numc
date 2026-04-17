#include "kernel.h"
#include "internal.h"
#include <numc/array.h>
#include <numc/math.h>
#include <math.h>
#include <stdint.h>

#include "arch_dispatch.h"
#if NUMC_HAVE_AVX2
#include "intrinsics/math_avx2.h"
#elif NUMC_HAVE_SVE
#include "intrinsics/math_sve.h"
#elif NUMC_HAVE_NEON
#include "intrinsics/math_neon.h"
#endif
#if NUMC_HAVE_RVV
#include "intrinsics/math_rvv.h"
#endif

/* -- Stamp rand kernels for all 10 dtypes ---------------------------
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
#if !NUMC_HAVE_AVX2
DEFINE_RAND_KERNEL(NUMC_DTYPE_FLOAT64, NUMC_FLOAT64, CONV_F64)
#endif

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

/* -- AVX2-optimized rand f64: batch uint64→f64 conversion ----------
 *
 * The generic macro uses scalar _u64_to_f64 per element.  With AVX2 we
 * convert 4 uint64 at a time: shift right 11, OR in the exponent bias
 * (0x3FF << 52), reinterpret as f64, subtract 1.0 → uniform [0, 1).
 */
#if NUMC_HAVE_AVX2
static inline void _rand_f64_convert4_avx2(const uint64_t *raw, double *dst) {
  __m256i r = _mm256_loadu_si256((const __m256i *)raw);
  __m256i bits =
      _mm256_or_si256(_mm256_srli_epi64(r, 11),
                      _mm256_set1_epi64x((int64_t)0x3FF0000000000000LL));
  __m256d v = _mm256_sub_pd(_mm256_castsi256_pd(bits), _mm256_set1_pd(1.0));
  _mm256_storeu_pd(dst, v);
}

static void _kern_rand_f64_avx2(char *out, size_t n) {
  _prng_ensure_seeded();
  double *restrict po = (double *)out;
  size_t total_bytes = n * sizeof(double);
  if (total_bytes <= NUMC_OMP_BYTE_THRESHOLD) {
    /* -- Small path: single-threaded 4-wide SIMD ----------------- */
    uint64_t s0[4], s1[4], s2[4], s3[4];
    memcpy(s0, prng_s[0], sizeof s0);
    memcpy(s1, prng_s[1], sizeof s1);
    memcpy(s2, prng_s[2], sizeof s2);
    memcpy(s3, prng_s[3], sizeof s3);
    size_t i = 0;
    uint64_t raw[4];
    for (; i + NUMC_PRNG_LANES <= n; i += NUMC_PRNG_LANES) {
      XOSHIRO_STEP_4(s0, s1, s2, s3, raw);
      _rand_f64_convert4_avx2(raw, po + i);
    }
    memcpy(prng_s[0], s0, sizeof s0);
    memcpy(prng_s[1], s1, sizeof s1);
    memcpy(prng_s[2], s2, sizeof s2);
    memcpy(prng_s[3], s3, sizeof s3);
    for (; i < n; i++)
      po[i] = _u64_to_f64(_prng_next_scalar());
  } else {
    /* -- Large path: per-thread sub-states, OMP when available --- */
    int _nthreads = (int)(total_bytes / NUMC_OMP_BYTES_PER_THREAD);
    NUMC_OMP_CAP_THREADS(_nthreads);
    if (_nthreads < 1)
      _nthreads = 1;
    uint64_t base[4] = {
        prng_s[0][0],
        prng_s[1][0],
        prng_s[2][0],
        prng_s[3][0],
    };
    size_t chunk = (n + (size_t)_nthreads - 1) / (size_t)_nthreads;
    NUMC_PRAGMA(omp parallel num_threads(_nthreads)) {
      int tid = _prng_get_tid();
      size_t start = (size_t)tid * chunk;
      size_t end = start + chunk < n ? start + chunk : n;
      uint64_t ts[4] = {base[0], base[1], base[2], base[3]};
      prng_skip(ts, start);
      uint64_t s0[4], s1[4], s2[4], s3[4];
      for (int l = 0; l < 4; l++) {
        uint64_t lt[4] = {ts[0], ts[1], ts[2], ts[3]};
        prng_skip(lt, (size_t)l);
        s0[l] = lt[0];
        s1[l] = lt[1];
        s2[l] = lt[2];
        s3[l] = lt[3];
      }
      size_t ri = start;
      uint64_t raw[4];
      for (; ri + NUMC_PRNG_LANES <= end; ri += NUMC_PRNG_LANES) {
        XOSHIRO_STEP_4(s0, s1, s2, s3, raw);
        _rand_f64_convert4_avx2(raw, po + ri);
      }
      uint64_t rs[4] = {s0[0], s1[0], s2[0], s3[0]};
      for (; ri < end; ri++) {
        const uint64_t res = numc_rotl64(rs[1] * 5, 7) * 9;
        const uint64_t t = rs[1] << 17;
        rs[2] ^= rs[0];
        rs[3] ^= rs[1];
        rs[1] ^= rs[2];
        rs[0] ^= rs[3];
        rs[2] ^= t;
        rs[3] = numc_rotl64(rs[3], 45);
        po[ri] = _u64_to_f64(res);
      }
    } /* end omp parallel */
    prng_skip(base, n);
    prng_s[0][0] = base[0];
    prng_s[1][0] = base[1];
    prng_s[2][0] = base[2];
    prng_s[3][0] = base[3];
    /* advance lanes 1-3 by n steps to keep all lanes in sync */
    for (int lane = 1; lane < NUMC_PRNG_LANES; lane++) {
      uint64_t ls[4] = {
          prng_s[0][lane],
          prng_s[1][lane],
          prng_s[2][lane],
          prng_s[3][lane],
      };
      prng_skip(ls, n);
      prng_s[0][lane] = ls[0];
      prng_s[1][lane] = ls[1];
      prng_s[2][lane] = ls[2];
      prng_s[3][lane] = ls[3];
    }
  }
}
#endif /* NUMC_HAVE_AVX2 */

/* -- Stamp randn kernels for all 10 dtypes --------------------------
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
#elif NUMC_HAVE_NEON
static void _kern_randn_NUMC_DTYPE_FLOAT32(char *out, size_t n) {
  _prng_ensure_seeded();
  float *restrict po = (float *)out;
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    /* Generate 4 normals from 2 Box-Muller pairs */
    float u[4];
    for (int k = 0; k < 4; k++)
      u[k] = _prng_f32();
    /* u1 = u[0..1], u2 = u[2..3] */
    float u1_0 = u[0] < 1e-20f ? 1e-20f : u[0];
    float u1_1 = u[1] < 1e-20f ? 1e-20f : u[1];
    float32x2_t u1 = vmax_f32(vld1_f32(u), vdup_n_f32(1e-20f));
    float32x2_t u2 = vld1_f32(u + 2);
    /* mag = sqrt(-2 * log(u1)) */
    float32x4_t u1_4 = vcombine_f32(u1, u1);
    float32x4_t neg2 = vdupq_n_f32(-2.0f);
    float32x4_t log_u1 = _neon_log_f32(u1_4);
    float32x4_t mag_sq = vmulq_f32(neg2, log_u1);
    float32x2_t mag = vget_low_f32(vsqrtq_f32(mag_sq));
    /* angle = 2*pi*u2 */
    float32x2_t two_pi = vdup_n_f32(6.2831853071795864f);
    float32x2_t angle = vmul_f32(u2, two_pi);
    /* sincos via NEON */
    float32x4_t angle4 = vcombine_f32(angle, angle);
    float32x4_t sin_v, cos_v;
    _neon_sincos_f32(angle4, &sin_v, &cos_v);
    /* out = mag * cos (first 2), mag * sin (next 2) */
    float32x2_t cos2 = vget_low_f32(cos_v);
    float32x2_t sin2 = vget_low_f32(sin_v);
    vst1_f32(po + i, vmul_f32(mag, cos2));
    vst1_f32(po + i + 2, vmul_f32(mag, sin2));
    (void)u1_0;
    (void)u1_1;
  }
  for (; i < n; i++)
    po[i] = _prng_normal_f32();
}
#elif NUMC_HAVE_SVE
static void _kern_randn_NUMC_DTYPE_FLOAT32(char *out, size_t n) {
  _prng_ensure_seeded();
  float *restrict po = (float *)out;
  size_t vl = svcntw();
  size_t i = 0;
  /* SVE: generate vl normals per iteration */
  for (; i + vl <= n; i += vl) {
    float ubuf[64]; /* max VL for SVE2048 = 64 f32 */
    size_t cnt = vl < 64 ? vl : 64;
    svbool_t ptrue = svptrue_b32();
    /* Generate u1 */
    for (size_t k = 0; k < cnt; k++)
      ubuf[k] = _prng_f32();
    svfloat32_t u1 = svld1_f32(ptrue, ubuf);
    u1 = svmax_f32_x(ptrue, u1, svdup_f32(1e-20f));
    /* Generate u2 (independent randoms) */
    for (size_t k = 0; k < cnt; k++)
      ubuf[k] = _prng_f32();
    svfloat32_t u2 = svld1_f32(ptrue, ubuf);
    /* mag = sqrt(-2 * log(u1)), angle = 2*pi*u2 */
    svfloat32_t mag = svsqrt_f32_x(
        ptrue, svmul_f32_x(ptrue, svdup_f32(-2.0f), _sve_log_f32(u1)));
    svfloat32_t angle = svmul_f32_x(ptrue, svdup_f32(6.2831853071795864f), u2);
    svfloat32_t sin_v, cos_v;
    _sve_sincos_f32(angle, &sin_v, &cos_v);
    /* Store mag*cos */
    svst1_f32(ptrue, po + i, svmul_f32_x(ptrue, mag, cos_v));
  }
  for (; i < n; i++)
    po[i] = _prng_normal_f32();
}
#else
DEFINE_RANDN_KERNEL(NUMC_DTYPE_FLOAT32, NUMC_FLOAT32, _prng_normal_f32())
#endif
DEFINE_RANDN_KERNEL(NUMC_DTYPE_FLOAT64, NUMC_FLOAT64, _prng_normal_f64())

/* -- Dispatch tables (dtype -> kernel) -----------------------------*/

static const NumcRandKernel rand_table[] = {
    ER(rand, NUMC_DTYPE_INT8),
    ER(rand, NUMC_DTYPE_INT16),
    ER(rand, NUMC_DTYPE_INT32),
    ER(rand, NUMC_DTYPE_INT64),
    ER(rand, NUMC_DTYPE_UINT8),
    ER(rand, NUMC_DTYPE_UINT16),
    ER(rand, NUMC_DTYPE_UINT32),
    ER(rand, NUMC_DTYPE_UINT64),
    ER(rand, NUMC_DTYPE_FLOAT32),
#if NUMC_HAVE_AVX2
    [NUMC_DTYPE_FLOAT64] = _kern_rand_f64_avx2,
#else
    ER(rand, NUMC_DTYPE_FLOAT64),
#endif
};

static const NumcRandKernel randn_table[] = {
    ER(randn, NUMC_DTYPE_INT8),    ER(randn, NUMC_DTYPE_INT16),
    ER(randn, NUMC_DTYPE_INT32),   ER(randn, NUMC_DTYPE_INT64),
    ER(randn, NUMC_DTYPE_UINT8),   ER(randn, NUMC_DTYPE_UINT16),
    ER(randn, NUMC_DTYPE_UINT32),  ER(randn, NUMC_DTYPE_UINT64),
    ER(randn, NUMC_DTYPE_FLOAT32), ER(randn, NUMC_DTYPE_FLOAT64),
};

/* -- Shared creation helper -----------------------------------------*/

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

/* -- Public API -----------------------------------------------------*/

// NOLINTNEXTLINE(misc-use-internal-linkage)
void numc_manual_seed(uint64_t seed) {
  prng_seed(seed);
}

// NOLINTNEXTLINE(misc-use-internal-linkage)
NumcArray *numc_array_rand(NumcCtx *ctx, const size_t *shape, size_t dim,
                           NumcDType dtype) {
  return _rand_impl(ctx, shape, dim, dtype, rand_table);
}

// NOLINTNEXTLINE(misc-use-internal-linkage)
NumcArray *numc_array_randn(NumcCtx *ctx, const size_t *shape, size_t dim,
                            NumcDType dtype) {
  return _rand_impl(ctx, shape, dim, dtype, randn_table);
}

/* He (Kaiming): scale randn by sqrt(2 / fan_in) */
// NOLINTNEXTLINE(misc-use-internal-linkage)
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
// NOLINTNEXTLINE(misc-use-internal-linkage)
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
