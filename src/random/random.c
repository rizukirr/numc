#include "kernel.h"
#include "internal.h"
#include <numc/array.h>
#include <numc/error.h>
#include <numc/math.h>
#include <numc/random.h>

#include <math.h>
#include <stdint.h>

/* ── Stamp rand kernels for all 10 dtypes ───────────────────────────
 *
 * CONVERT_EXPR(raw) maps a raw uint64_t from the PRNG to C_TYPE.
 * It is called once per lane per loop iteration — a pure expression
 * on a local variable, enabling AVX2 vectorization.
 *
 * Float types:   uniform [0, 1) via IEEE 754 bit trick (_u64_to_f32/64).
 * Integer types: raw bits masked to type width.
 */

#define _CONV_INT8(raw)   ((NUMC_INT8)  ((raw) & 0xFFu))
#define _CONV_INT16(raw)  ((NUMC_INT16) ((raw) & 0xFFFFu))
#define _CONV_INT32(raw)  ((NUMC_INT32) ((raw) & 0xFFFFFFFFu))
#define _CONV_INT64(raw)  ((NUMC_INT64) (raw))
#define _CONV_UINT8(raw)  ((NUMC_UINT8) ((raw) & 0xFFu))
#define _CONV_UINT16(raw) ((NUMC_UINT16)((raw) & 0xFFFFu))
#define _CONV_UINT32(raw) ((NUMC_UINT32)((raw) & 0xFFFFFFFFu))
#define _CONV_UINT64(raw) ((NUMC_UINT64)(raw))
#define _CONV_F32(raw)    (_u64_to_f32(raw))
#define _CONV_F64(raw)    (_u64_to_f64(raw))

DEFINE_RAND_KERNEL(NUMC_DTYPE_INT8,    NUMC_INT8,    _CONV_INT8)
DEFINE_RAND_KERNEL(NUMC_DTYPE_INT16,   NUMC_INT16,   _CONV_INT16)
DEFINE_RAND_KERNEL(NUMC_DTYPE_INT32,   NUMC_INT32,   _CONV_INT32)
DEFINE_RAND_KERNEL(NUMC_DTYPE_INT64,   NUMC_INT64,   _CONV_INT64)
DEFINE_RAND_KERNEL(NUMC_DTYPE_UINT8,   NUMC_UINT8,   _CONV_UINT8)
DEFINE_RAND_KERNEL(NUMC_DTYPE_UINT16,  NUMC_UINT16,  _CONV_UINT16)
DEFINE_RAND_KERNEL(NUMC_DTYPE_UINT32,  NUMC_UINT32,  _CONV_UINT32)
DEFINE_RAND_KERNEL(NUMC_DTYPE_UINT64,  NUMC_UINT64,  _CONV_UINT64)
DEFINE_RAND_KERNEL(NUMC_DTYPE_FLOAT32, NUMC_FLOAT32, _CONV_F32)
DEFINE_RAND_KERNEL(NUMC_DTYPE_FLOAT64, NUMC_FLOAT64, _CONV_F64)

#undef _CONV_INT8
#undef _CONV_INT16
#undef _CONV_INT32
#undef _CONV_INT64
#undef _CONV_UINT8
#undef _CONV_UINT16
#undef _CONV_UINT32
#undef _CONV_UINT64
#undef _CONV_F32
#undef _CONV_F64

/* ── Stamp randn kernels for all 10 dtypes ──────────────────────────
 *
 * Float types:    N(0,1) via Box-Muller transform (true normal samples).
 * Integer types:  N(0,1) double sample, truncated and cast.
 *                 Useful for near-zero initialisation of quantised
 *                 weights — most values land in [-3, 3] before cast.
 */

DEFINE_RANDN_KERNEL(NUMC_DTYPE_INT8,    NUMC_INT8,
                    (NUMC_INT8)_prng_normal_f64())
DEFINE_RANDN_KERNEL(NUMC_DTYPE_INT16,   NUMC_INT16,
                    (NUMC_INT16)_prng_normal_f64())
DEFINE_RANDN_KERNEL(NUMC_DTYPE_INT32,   NUMC_INT32,
                    (NUMC_INT32)_prng_normal_f64())
DEFINE_RANDN_KERNEL(NUMC_DTYPE_INT64,   NUMC_INT64,
                    (NUMC_INT64)_prng_normal_f64())
DEFINE_RANDN_KERNEL(NUMC_DTYPE_UINT8,   NUMC_UINT8,
                    (NUMC_UINT8)_prng_normal_f64())
DEFINE_RANDN_KERNEL(NUMC_DTYPE_UINT16,  NUMC_UINT16,
                    (NUMC_UINT16)_prng_normal_f64())
DEFINE_RANDN_KERNEL(NUMC_DTYPE_UINT32,  NUMC_UINT32,
                    (NUMC_UINT32)_prng_normal_f64())
DEFINE_RANDN_KERNEL(NUMC_DTYPE_UINT64,  NUMC_UINT64,
                    (NUMC_UINT64)_prng_normal_f64())
DEFINE_RANDN_KERNEL(NUMC_DTYPE_FLOAT32, NUMC_FLOAT32, _prng_normal_f32())
DEFINE_RANDN_KERNEL(NUMC_DTYPE_FLOAT64, NUMC_FLOAT64, _prng_normal_f64())

/* ── Dispatch tables (dtype -> kernel) ─────────────────────────────*/

static const NumcRandKernel _rand_table[] = {
    ER(rand, NUMC_DTYPE_INT8),    ER(rand, NUMC_DTYPE_INT16),
    ER(rand, NUMC_DTYPE_INT32),   ER(rand, NUMC_DTYPE_INT64),
    ER(rand, NUMC_DTYPE_UINT8),   ER(rand, NUMC_DTYPE_UINT16),
    ER(rand, NUMC_DTYPE_UINT32),  ER(rand, NUMC_DTYPE_UINT64),
    ER(rand, NUMC_DTYPE_FLOAT32), ER(rand, NUMC_DTYPE_FLOAT64),
};

static const NumcRandKernel _randn_table[] = {
    ER(randn, NUMC_DTYPE_INT8),    ER(randn, NUMC_DTYPE_INT16),
    ER(randn, NUMC_DTYPE_INT32),   ER(randn, NUMC_DTYPE_INT64),
    ER(randn, NUMC_DTYPE_UINT8),   ER(randn, NUMC_DTYPE_UINT16),
    ER(randn, NUMC_DTYPE_UINT32),  ER(randn, NUMC_DTYPE_UINT64),
    ER(randn, NUMC_DTYPE_FLOAT32), ER(randn, NUMC_DTYPE_FLOAT64),
};

/* ── Shared creation helper ─────────────────────────────────────────*/

static NumcArray *_rand_impl(NumcCtx *ctx, const size_t *shape, size_t dim,
                             NumcDType dtype,
                             const NumcRandKernel *table) {
  if (!ctx || !shape || dim == 0) {
    NUMC_SET_ERROR(NUMC_ERR_NULL,
                   "rand: NULL or invalid args (ctx=%p shape=%p dim=%zu)",
                   ctx, shape, dim);
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
  _prng_seed(seed);
}

NumcArray *numc_array_rand(NumcCtx *ctx, const size_t *shape,
                           size_t dim, NumcDType dtype) {
  return _rand_impl(ctx, shape, dim, dtype, _rand_table);
}

NumcArray *numc_array_randn(NumcCtx *ctx, const size_t *shape,
                            size_t dim, NumcDType dtype) {
  return _rand_impl(ctx, shape, dim, dtype, _randn_table);
}

/* He (Kaiming): scale randn by sqrt(2 / fan_in) */
NumcArray *numc_array_random_he(NumcCtx *ctx, const size_t *shape,
                                size_t dim, NumcDType dtype,
                                size_t fan_in) {
  if (fan_in == 0) {
    NUMC_SET_ERROR(NUMC_ERR_SHAPE, "random_he: fan_in must be > 0");
    return NULL;
  }

  NumcArray *arr = numc_array_randn(ctx, shape, dim, dtype);
  if (!arr)
    return NULL;

  double scale = sqrt(2.0 / (double)fan_in);
  numc_mul_scalar_inplace(arr, scale);
  return arr;
}

/* Xavier (Glorot): uniform [-limit, limit),
 * limit = sqrt(6 / (fan_in + fan_out)) */
NumcArray *numc_array_random_xavier(NumcCtx *ctx, const size_t *shape,
                                    size_t dim, NumcDType dtype,
                                    size_t fan_in, size_t fan_out) {
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
  numc_mul_scalar_inplace(arr, 2.0 * limit);  /* [0, 2*limit)    */
  numc_sub_scalar_inplace(arr, limit);         /* [-limit, limit) */
  return arr;
}
