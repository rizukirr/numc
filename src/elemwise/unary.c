#include "dispatch.h"
#include "helpers.h"
#include <math.h>
#include <numc/math.h>

/* ── Stamp unary neg loop typed kernels ────────────────────*/

/* neg: all 10 types, native - */
#define STAMP_NEG(TE, CT) DEFINE_UNARY_KERNEL(neg, TE, CT, -in1)
GENERATE_NUMC_TYPES(STAMP_NEG)
#undef STAMP_NEG

/* ── Stamp unary abs loop typed kernels ────────────────────*/

#define STAMP_ABS(TE, CT)                                                      \
  DEFINE_UNARY_KERNEL(abs, TE, CT, (CT)(in1 < 0 ? -in1 : in1))
GENERATE_SIGNED_INT8_INT16_INT32_NUMC_TYPES(STAMP_ABS)
GENERATE_SIGNED_64BIT_NUMC_TYPES(STAMP_ABS)
#undef STAMP_ABS
DEFINE_UNARY_KERNEL(abs, NUMC_DTYPE_FLOAT32, NUMC_FLOAT32,
                    (NUMC_FLOAT32)(in1 < 0.0f ? -in1 : in1))
DEFINE_UNARY_KERNEL(abs, NUMC_DTYPE_FLOAT64, NUMC_FLOAT64,
                    (NUMC_FLOAT64)(in1 < 0.0 ? -in1 : in1))

/* ── Stamp out log loop kernels (stride-aware, wrapping scalar bit-manip) ── */

/* < 32-bit integers: cast through float */
#define STAMP_LOG_SMALL(TE, CT)                                                \
  DEFINE_UNARY_KERNEL(log, TE, CT, (CT)_log_f32((float)in1))
GENERATE_INT8_INT16_NUMC_TYPES(STAMP_LOG_SMALL)
#undef STAMP_LOG_SMALL

/* 32-bit integers: cast through double */
#define STAMP_LOG_I32(TE, CT)                                                  \
  DEFINE_UNARY_KERNEL(log, TE, CT, (CT)_log_f64((double)in1))
GENERATE_INT32_NUMC_TYPES(STAMP_LOG_I32)
#undef STAMP_LOG_I32

/* 64-bit integers: cast through double */
#define STAMP_LOG_I64(TE, CT)                                                  \
  DEFINE_UNARY_KERNEL(log, TE, CT, (CT)_log_f64((double)in1))
GENERATE_SIGNED_64BIT_NUMC_TYPES(STAMP_LOG_I64)
#undef STAMP_LOG_I64
DEFINE_UNARY_KERNEL(log, NUMC_DTYPE_UINT64, NUMC_UINT64,
                    (NUMC_UINT64)_log_f64((double)in1))

/* float types: call their own bit-manipulation helpers directly */
DEFINE_UNARY_KERNEL(log, NUMC_DTYPE_FLOAT32, NUMC_FLOAT32, _log_f32(in1))
DEFINE_UNARY_KERNEL(log, NUMC_DTYPE_FLOAT64, NUMC_FLOAT64, _log_f64(in1))

/* ── Stamp out exp loop kernels ─────────────────────────────────────── */

/* int8/int16/uint8/uint16: cast through float32 */
#define STAMP_EXP_SMALL(TE, CT)                                                \
  DEFINE_UNARY_KERNEL(exp, TE, CT, (CT)_exp_f32((float)in1))
GENERATE_INT8_INT16_NUMC_TYPES(STAMP_EXP_SMALL)
#undef STAMP_EXP_SMALL

/* int32/uint32: cast through float64 */
#define STAMP_EXP_I32(TE, CT)                                                  \
  DEFINE_UNARY_KERNEL(exp, TE, CT, (CT)_exp_f64((double)in1))
GENERATE_INT32_NUMC_TYPES(STAMP_EXP_I32)
#undef STAMP_EXP_I32

/* int64: cast through float64 */
#define STAMP_EXP_I64(TE, CT)                                                  \
  DEFINE_UNARY_KERNEL(exp, TE, CT, (CT)_exp_f64((double)in1))
GENERATE_SIGNED_64BIT_NUMC_TYPES(STAMP_EXP_I64)
#undef STAMP_EXP_I64

/* uint64: explicit — no X-macro covers just uint64 */
DEFINE_UNARY_KERNEL(exp, NUMC_DTYPE_UINT64, NUMC_UINT64,
                    (NUMC_UINT64)_exp_f64((double)in1))

/* float32/float64: call helpers directly */
DEFINE_UNARY_KERNEL(exp, NUMC_DTYPE_FLOAT32, NUMC_FLOAT32, _exp_f32(in1))
DEFINE_UNARY_KERNEL(exp, NUMC_DTYPE_FLOAT64, NUMC_FLOAT64, _exp_f64(in1))

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

static const NumcUnaryKernel _neg_table[] = {
    E(neg, NUMC_DTYPE_INT8),    E(neg, NUMC_DTYPE_INT16),
    E(neg, NUMC_DTYPE_INT32),   E(neg, NUMC_DTYPE_INT64),
    E(neg, NUMC_DTYPE_UINT8),   E(neg, NUMC_DTYPE_UINT16),
    E(neg, NUMC_DTYPE_UINT32),  E(neg, NUMC_DTYPE_UINT64),
    E(neg, NUMC_DTYPE_FLOAT32), E(neg, NUMC_DTYPE_FLOAT64),
};

static const NumcUnaryKernel _abs_table[] = {
    E(abs, NUMC_DTYPE_INT8),    E(abs, NUMC_DTYPE_INT16),
    E(abs, NUMC_DTYPE_INT32),   E(abs, NUMC_DTYPE_INT64),
    E(abs, NUMC_DTYPE_FLOAT32), E(abs, NUMC_DTYPE_FLOAT64),
};

static const NumcUnaryKernel _log_table[] = {
    E(log, NUMC_DTYPE_INT8),    E(log, NUMC_DTYPE_INT16),
    E(log, NUMC_DTYPE_INT32),   E(log, NUMC_DTYPE_INT64),
    E(log, NUMC_DTYPE_UINT8),   E(log, NUMC_DTYPE_UINT16),
    E(log, NUMC_DTYPE_UINT32),  E(log, NUMC_DTYPE_UINT64),
    E(log, NUMC_DTYPE_FLOAT32), E(log, NUMC_DTYPE_FLOAT64),
};

static const NumcUnaryKernel _exp_table[] = {
    E(exp, NUMC_DTYPE_INT8),    E(exp, NUMC_DTYPE_INT16),
    E(exp, NUMC_DTYPE_INT32),   E(exp, NUMC_DTYPE_INT64),
    E(exp, NUMC_DTYPE_UINT8),   E(exp, NUMC_DTYPE_UINT16),
    E(exp, NUMC_DTYPE_UINT32),  E(exp, NUMC_DTYPE_UINT64),
    E(exp, NUMC_DTYPE_FLOAT32), E(exp, NUMC_DTYPE_FLOAT64),
};

static const NumcUnaryKernel _sqrt_table[] = {
    E(sqrt, NUMC_DTYPE_INT8),    E(sqrt, NUMC_DTYPE_INT16),
    E(sqrt, NUMC_DTYPE_INT32),   E(sqrt, NUMC_DTYPE_INT64),
    E(sqrt, NUMC_DTYPE_UINT8),   E(sqrt, NUMC_DTYPE_UINT16),
    E(sqrt, NUMC_DTYPE_UINT32),  E(sqrt, NUMC_DTYPE_UINT64),
    E(sqrt, NUMC_DTYPE_FLOAT32), E(sqrt, NUMC_DTYPE_FLOAT64),
};

/* ═══════════════════════════════════════════════════════════════════════
 * Public API — Unary ops
 * ═══════════════════════════════════════════════════════════════════════ */

#define DEFINE_ELEMWISE_UNARY(NAME, TABLE)                                     \
  int numc_##NAME(NumcArray *a, NumcArray *out) {                              \
    int err = _check_unary(a, out);                                            \
    if (err)                                                                   \
      return err;                                                              \
    return _unary_op(a, out, TABLE);                                           \
  }                                                                            \
  int numc_##NAME##_inplace(NumcArray *a) {                                    \
    return _unary_op_inplace(a, TABLE);                                        \
  }

DEFINE_ELEMWISE_UNARY(neg,  _neg_table)
DEFINE_ELEMWISE_UNARY(abs,  _abs_table)
DEFINE_ELEMWISE_UNARY(log,  _log_table)
DEFINE_ELEMWISE_UNARY(exp,  _exp_table)
DEFINE_ELEMWISE_UNARY(sqrt, _sqrt_table)
