#include "dispatch.h"
#include "helpers.h"
#include "numc/dtype.h"
#include <numc/math.h>

/* ── Stamp binary elem-wise arithmetic typed kernels ────────────────────*/

/* add: all 10 types, native + */
#define STAMP_ADD(TE, CT) DEFINE_BINARY_KERNEL(add, TE, CT, in1 + in2)
GENERATE_NUMC_TYPES(STAMP_ADD)
#undef STAMP_ADD

/* sub: all 10 types, native - */
#define STAMP_SUB(TE, CT) DEFINE_BINARY_KERNEL(sub, TE, CT, in1 - in2)
GENERATE_NUMC_TYPES(STAMP_SUB)
#undef STAMP_SUB

/* mul: all 10 types, native * */
#define STAMP_MUL(TE, CT) DEFINE_BINARY_KERNEL(mul, TE, CT, in1 *in2)
GENERATE_NUMC_TYPES(STAMP_MUL)
#undef STAMP_MUL

/* div: int8/int16 -> cast through float, int32 -> through double, rest native
 */
#define STAMP_DIV_SMALL(TE, CT)                                                \
  DEFINE_BINARY_KERNEL(div, TE, CT, (CT)((float)in1 / (float)in2))
GENERATE_INT8_INT16_NUMC_TYPES(STAMP_DIV_SMALL)
#undef STAMP_DIV_SMALL

#define STAMP_DIV_I32(TE, CT)                                                  \
  DEFINE_BINARY_KERNEL(div, TE, CT, (CT)((double)in1 / (double)in2))
GENERATE_INT32(STAMP_DIV_I32)
#undef STAMP_DIV_I32

#define STAMP_DIV_NATIVE(TE, CT) DEFINE_BINARY_KERNEL(div, TE, CT, in1 / in2)
GENERATE_64BIT_NUMC_TYPES(STAMP_DIV_NATIVE)
DEFINE_BINARY_KERNEL(div, NUMC_DTYPE_FLOAT32, NUMC_FLOAT32, in1 / in2)
#undef STAMP_DIV_NATIVE

/* ── Stamp out pow loop kernels ──────────────────────────────────────── */

/* 8/16-bit signed: branchless fixed-iteration (auto-vectorizes) */
DEFINE_BINARY_KERNEL(pow, NUMC_DTYPE_INT8, NUMC_INT8, _powi_i8(in1, in2))
DEFINE_BINARY_KERNEL(pow, NUMC_DTYPE_INT16, NUMC_INT16, _powi_i16(in1, in2))

/* 8/16-bit unsigned: branchless fixed-iteration (auto-vectorizes) */
DEFINE_BINARY_KERNEL(pow, NUMC_DTYPE_UINT8, NUMC_UINT8, _powi_u8(in1, in2))
DEFINE_BINARY_KERNEL(pow, NUMC_DTYPE_UINT16, NUMC_UINT16, _powi_u16(in1, in2))

/* 32/64-bit: variable-iteration early-exit (scalar, fast for small exp) */
DEFINE_BINARY_KERNEL(pow, NUMC_DTYPE_INT32, NUMC_INT32,
                     (NUMC_INT32)_powi_signed((NUMC_INT64)in1, (NUMC_INT64)in2))
DEFINE_BINARY_KERNEL(pow, NUMC_DTYPE_UINT32, NUMC_UINT32,
                     (NUMC_UINT32)_powi_unsigned((NUMC_UINT64)in1, (NUMC_UINT64)in2))
DEFINE_BINARY_KERNEL(pow, NUMC_DTYPE_INT64, NUMC_INT64,
                     (NUMC_INT64)_powi_signed((NUMC_INT64)in1, (NUMC_INT64)in2))
DEFINE_BINARY_KERNEL(pow, NUMC_DTYPE_UINT64, NUMC_UINT64,
                     (NUMC_UINT64)_powi_unsigned((NUMC_UINT64)in1, (NUMC_UINT64)in2))

/* float32: fused exp(in2 * log(in1)), single-precision */
DEFINE_BINARY_KERNEL(pow, NUMC_DTYPE_FLOAT32, NUMC_FLOAT32,
                     _exp_f32(in2 *_log_f32(in1)))

/* float64: fused exp(in2 * log(in1)), double-precision */
DEFINE_BINARY_KERNEL(pow, NUMC_DTYPE_FLOAT64, NUMC_FLOAT64,
                     _exp_f64(in2 *_log_f64(in1)))

/* ── Stamp out maximum and minimum ──────────────────────────────────────*/

#define STAMP_MAX(TE, CT)                                                      \
  DEFINE_BINARY_KERNEL(maximum, TE, CT, in1 > in2 ? in1 : in2)
GENERATE_NUMC_TYPES(STAMP_MAX)
#undef STAMP_MAX

#define STAMP_MIN(TE, CT)                                                      \
  DEFINE_BINARY_KERNEL(minimum, TE, CT, in1 < in2 ? in1 : in2)
GENERATE_NUMC_TYPES(STAMP_MIN)
#undef STAMP_MIN

/* ── Dispatch tables (dtype -> kernel) ─────────────────────────────── */

static const NumcBinaryKernel _add_table[] = {
    E(add, NUMC_DTYPE_INT8),    E(add, NUMC_DTYPE_INT16),
    E(add, NUMC_DTYPE_INT32),   E(add, NUMC_DTYPE_INT64),
    E(add, NUMC_DTYPE_UINT8),   E(add, NUMC_DTYPE_UINT16),
    E(add, NUMC_DTYPE_UINT32),  E(add, NUMC_DTYPE_UINT64),
    E(add, NUMC_DTYPE_FLOAT32), E(add, NUMC_DTYPE_FLOAT64),
};

static const NumcBinaryKernel _sub_table[] = {
    E(sub, NUMC_DTYPE_INT8),    E(sub, NUMC_DTYPE_INT16),
    E(sub, NUMC_DTYPE_INT32),   E(sub, NUMC_DTYPE_INT64),
    E(sub, NUMC_DTYPE_UINT8),   E(sub, NUMC_DTYPE_UINT16),
    E(sub, NUMC_DTYPE_UINT32),  E(sub, NUMC_DTYPE_UINT64),
    E(sub, NUMC_DTYPE_FLOAT32), E(sub, NUMC_DTYPE_FLOAT64),
};

static const NumcBinaryKernel _mul_table[] = {
    E(mul, NUMC_DTYPE_INT8),    E(mul, NUMC_DTYPE_INT16),
    E(mul, NUMC_DTYPE_INT32),   E(mul, NUMC_DTYPE_INT64),
    E(mul, NUMC_DTYPE_UINT8),   E(mul, NUMC_DTYPE_UINT16),
    E(mul, NUMC_DTYPE_UINT32),  E(mul, NUMC_DTYPE_UINT64),
    E(mul, NUMC_DTYPE_FLOAT32), E(mul, NUMC_DTYPE_FLOAT64),
};

static const NumcBinaryKernel _div_table[] = {
    E(div, NUMC_DTYPE_INT8),    E(div, NUMC_DTYPE_INT16),
    E(div, NUMC_DTYPE_INT32),   E(div, NUMC_DTYPE_INT64),
    E(div, NUMC_DTYPE_UINT8),   E(div, NUMC_DTYPE_UINT16),
    E(div, NUMC_DTYPE_UINT32),  E(div, NUMC_DTYPE_UINT64),
    E(div, NUMC_DTYPE_FLOAT32), E(div, NUMC_DTYPE_FLOAT64),
};

static const NumcBinaryKernel _pow_table[] = {
    E(pow, NUMC_DTYPE_INT8),    E(pow, NUMC_DTYPE_INT16),
    E(pow, NUMC_DTYPE_INT32),   E(pow, NUMC_DTYPE_INT64),
    E(pow, NUMC_DTYPE_UINT8),   E(pow, NUMC_DTYPE_UINT16),
    E(pow, NUMC_DTYPE_UINT32),  E(pow, NUMC_DTYPE_UINT64),
    E(pow, NUMC_DTYPE_FLOAT32), E(pow, NUMC_DTYPE_FLOAT64),
};

static const NumcBinaryKernel _maximum_table[] = {
    E(maximum, NUMC_DTYPE_INT8),    E(maximum, NUMC_DTYPE_INT16),
    E(maximum, NUMC_DTYPE_INT32),   E(maximum, NUMC_DTYPE_INT64),
    E(maximum, NUMC_DTYPE_UINT8),   E(maximum, NUMC_DTYPE_UINT16),
    E(maximum, NUMC_DTYPE_UINT32),  E(maximum, NUMC_DTYPE_UINT64),
    E(maximum, NUMC_DTYPE_FLOAT32), E(maximum, NUMC_DTYPE_FLOAT64),
};

static const NumcBinaryKernel _minimum_table[] = {
    E(minimum, NUMC_DTYPE_INT8),    E(minimum, NUMC_DTYPE_INT16),
    E(minimum, NUMC_DTYPE_INT32),   E(minimum, NUMC_DTYPE_INT64),
    E(minimum, NUMC_DTYPE_UINT8),   E(minimum, NUMC_DTYPE_UINT16),
    E(minimum, NUMC_DTYPE_UINT32),  E(minimum, NUMC_DTYPE_UINT64),
    E(minimum, NUMC_DTYPE_FLOAT32), E(minimum, NUMC_DTYPE_FLOAT64),
};

/* ═══════════════════════════════════════════════════════════════════════
 * Public API — Binary + Scalar ops
 * ═══════════════════════════════════════════════════════════════════════ */

#define DEFINE_ELEMWISE_BINARY(NAME, TABLE)                                    \
  int numc_##NAME(const NumcArray *a, const NumcArray *b, NumcArray *out) {    \
    int err = _check_binary(a, b, out);                                        \
    if (err)                                                                   \
      return err;                                                              \
    _binary_op(a, b, out, TABLE);                                              \
    return 0;                                                                  \
  }

#define DEFINE_ELEMWISE_SCALAR(NAME, TABLE)                                    \
  int numc_##NAME##_scalar(const NumcArray *a, double scalar,                  \
                           NumcArray *out) {                                   \
    int err = _check_unary(a, out);                                            \
    if (err)                                                                   \
      return err;                                                              \
    char buf[8];                                                               \
    _double_to_dtype(scalar, a->dtype, buf);                                   \
    _scalar_op(a, buf, out, TABLE);                                            \
    return 0;                                                                  \
  }                                                                            \
  int numc_##NAME##_scalar_inplace(NumcArray *a, double scalar) {              \
    return _scalar_op_inplace(a, scalar, TABLE);                               \
  }

DEFINE_ELEMWISE_BINARY(add, _add_table)
DEFINE_ELEMWISE_BINARY(sub, _sub_table)
DEFINE_ELEMWISE_BINARY(mul, _mul_table)
DEFINE_ELEMWISE_BINARY(div, _div_table)

DEFINE_ELEMWISE_BINARY(maximum, _maximum_table)
DEFINE_ELEMWISE_BINARY(minimum, _minimum_table)

DEFINE_ELEMWISE_SCALAR(add, _add_table)
DEFINE_ELEMWISE_SCALAR(sub, _sub_table)
DEFINE_ELEMWISE_SCALAR(mul, _mul_table)
DEFINE_ELEMWISE_SCALAR(div, _div_table)

/* pow: non-const signature differs, stays explicit */
int numc_pow(NumcArray *a, NumcArray *b, NumcArray *out) {
  int err = _check_binary(a, b, out);
  if (err)
    return err;
  _binary_op(a, b, out, _pow_table);
  return 0;
}

int numc_pow_inplace(NumcArray *a, NumcArray *b) {
  int err = _check_binary(a, b, a);
  if (err)
    return err;
  _binary_op(a, b, a, _pow_table);
  return 0;
}

int numc_maximum_inplace(NumcArray *a, const NumcArray *b) {
  int err = _check_binary(a, b, a);
  if (err)
    return err;
  _binary_op(a, b, a, _maximum_table);
  return 0;
}

int numc_minimum_inplace(NumcArray *a, const NumcArray *b) {
  int err = _check_binary(a, b, a);
  if (err)
    return err;
  _binary_op(a, b, a, _minimum_table);
  return 0;
}
