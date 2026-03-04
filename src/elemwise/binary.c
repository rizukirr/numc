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

/* div: specialized kernel with reciprocal optimization for scalars */
#define STAMP_DIV_S(TE, CT) \
  DEFINE_INT_DIV_KERNEL(TE, CT, true)
GENERATE_SIGNED_INT_NUMC_TYPES(STAMP_DIV_S)
#undef STAMP_DIV_S

#define STAMP_DIV_U(TE, CT) \
  DEFINE_INT_DIV_KERNEL(TE, CT, false)
GENERATE_UNSIGNED_INT_NUMC_TYPES(STAMP_DIV_U)
#undef STAMP_DIV_U

#define STAMP_DIV_F(TE, CT) \
  DEFINE_FLOAT_DIV_KERNEL(TE, CT)
GENERATE_FLOAT_NUMC_TYPES(STAMP_DIV_F)
#undef STAMP_DIV_F

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
                     (NUMC_UINT32)_powi_unsigned((NUMC_UINT64)in1,
                                                 (NUMC_UINT64)in2))
DEFINE_BINARY_KERNEL(pow, NUMC_DTYPE_INT64, NUMC_INT64,
                     (NUMC_INT64)_powi_signed((NUMC_INT64)in1, (NUMC_INT64)in2))
DEFINE_BINARY_KERNEL(pow, NUMC_DTYPE_UINT64, NUMC_UINT64,
                     (NUMC_UINT64)_powi_unsigned((NUMC_UINT64)in1,
                                                 (NUMC_UINT64)in2))

/* float32: fused exp(in2 * log(in1)), single-precision */
DEFINE_BINARY_KERNEL(pow, NUMC_DTYPE_FLOAT32, NUMC_FLOAT32,
                     _exp_f32(in2 *_log_f32(in1)))

/* float64: fused exp(in2 * log(in1)), double-precision */
DEFINE_BINARY_KERNEL(pow, NUMC_DTYPE_FLOAT64, NUMC_FLOAT64,
                     _exp_f64(in2 *_log_f64(in1)))

/* ── Stamp out maximum and minimum ──────────────────────────────────────*/

#define STAMP_MAX(TE, CT) \
  DEFINE_BINARY_KERNEL(maximum, TE, CT, in1 > in2 ? in1 : in2)
GENERATE_NUMC_TYPES(STAMP_MAX)
#undef STAMP_MAX

#define STAMP_MIN(TE, CT) \
  DEFINE_BINARY_KERNEL(minimum, TE, CT, in1 < in2 ? in1 : in2)
GENERATE_NUMC_TYPES(STAMP_MIN)
#undef STAMP_MIN

/* ── Stamp out comparison / selection ─────────────────────────────── */

#define STAMP_EQ(TE, CT) DEFINE_BINARY_KERNEL(eq, TE, CT, in1 == in2)
GENERATE_NUMC_TYPES(STAMP_EQ)
#undef STAMP_EQ

#define STAMP_GT(TE, CT) DEFINE_BINARY_KERNEL(gt, TE, CT, in1 > in2)
GENERATE_NUMC_TYPES(STAMP_GT)
#undef STAMP_GT

#define STAMP_LT(TE, CT) DEFINE_BINARY_KERNEL(lt, TE, CT, in1 < in2)
GENERATE_NUMC_TYPES(STAMP_LT)
#undef STAMP_LT

#define STAMP_GE(TE, CT) DEFINE_BINARY_KERNEL(ge, TE, CT, in1 >= in2)
GENERATE_NUMC_TYPES(STAMP_GE)
#undef STAMP_GE

#define STAMP_LE(TE, CT) DEFINE_BINARY_KERNEL(le, TE, CT, in1 <= in2)
GENERATE_NUMC_TYPES(STAMP_LE)
#undef STAMP_LE

/* ── Stamp out ternary where ───────────────────────────────────────── */

#define STAMP_WHERE(TE, CT) \
  DEFINE_TERNARY_KERNEL(where, TE, CT, (in_cond != 0 ? in_a : in_b))
GENERATE_NUMC_TYPES(STAMP_WHERE)
#undef STAMP_WHERE

/* ── Dispatch tables (dtype -> kernel) ─────────────────────────────── */

static const NumcBinaryKernel add_table[] = {
    E(add, NUMC_DTYPE_INT8),    E(add, NUMC_DTYPE_INT16),
    E(add, NUMC_DTYPE_INT32),   E(add, NUMC_DTYPE_INT64),
    E(add, NUMC_DTYPE_UINT8),   E(add, NUMC_DTYPE_UINT16),
    E(add, NUMC_DTYPE_UINT32),  E(add, NUMC_DTYPE_UINT64),
    E(add, NUMC_DTYPE_FLOAT32), E(add, NUMC_DTYPE_FLOAT64),
};

static const NumcBinaryKernel sub_table[] = {
    E(sub, NUMC_DTYPE_INT8),    E(sub, NUMC_DTYPE_INT16),
    E(sub, NUMC_DTYPE_INT32),   E(sub, NUMC_DTYPE_INT64),
    E(sub, NUMC_DTYPE_UINT8),   E(sub, NUMC_DTYPE_UINT16),
    E(sub, NUMC_DTYPE_UINT32),  E(sub, NUMC_DTYPE_UINT64),
    E(sub, NUMC_DTYPE_FLOAT32), E(sub, NUMC_DTYPE_FLOAT64),
};

static const NumcBinaryKernel mul_table[] = {
    E(mul, NUMC_DTYPE_INT8),    E(mul, NUMC_DTYPE_INT16),
    E(mul, NUMC_DTYPE_INT32),   E(mul, NUMC_DTYPE_INT64),
    E(mul, NUMC_DTYPE_UINT8),   E(mul, NUMC_DTYPE_UINT16),
    E(mul, NUMC_DTYPE_UINT32),  E(mul, NUMC_DTYPE_UINT64),
    E(mul, NUMC_DTYPE_FLOAT32), E(mul, NUMC_DTYPE_FLOAT64),
};

static const NumcBinaryKernel div_table[] = {
    E(div, NUMC_DTYPE_INT8),    E(div, NUMC_DTYPE_INT16),
    E(div, NUMC_DTYPE_INT32),   E(div, NUMC_DTYPE_INT64),
    E(div, NUMC_DTYPE_UINT8),   E(div, NUMC_DTYPE_UINT16),
    E(div, NUMC_DTYPE_UINT32),  E(div, NUMC_DTYPE_UINT64),
    E(div, NUMC_DTYPE_FLOAT32), E(div, NUMC_DTYPE_FLOAT64),
};

static const NumcBinaryKernel pow_table[] = {
    E(pow, NUMC_DTYPE_INT8),    E(pow, NUMC_DTYPE_INT16),
    E(pow, NUMC_DTYPE_INT32),   E(pow, NUMC_DTYPE_INT64),
    E(pow, NUMC_DTYPE_UINT8),   E(pow, NUMC_DTYPE_UINT16),
    E(pow, NUMC_DTYPE_UINT32),  E(pow, NUMC_DTYPE_UINT64),
    E(pow, NUMC_DTYPE_FLOAT32), E(pow, NUMC_DTYPE_FLOAT64),
};

static const NumcBinaryKernel maximum_table[] = {
    E(maximum, NUMC_DTYPE_INT8),    E(maximum, NUMC_DTYPE_INT16),
    E(maximum, NUMC_DTYPE_INT32),   E(maximum, NUMC_DTYPE_INT64),
    E(maximum, NUMC_DTYPE_UINT8),   E(maximum, NUMC_DTYPE_UINT16),
    E(maximum, NUMC_DTYPE_UINT32),  E(maximum, NUMC_DTYPE_UINT64),
    E(maximum, NUMC_DTYPE_FLOAT32), E(maximum, NUMC_DTYPE_FLOAT64),
};

static const NumcBinaryKernel minimum_table[] = {
    E(minimum, NUMC_DTYPE_INT8),    E(minimum, NUMC_DTYPE_INT16),
    E(minimum, NUMC_DTYPE_INT32),   E(minimum, NUMC_DTYPE_INT64),
    E(minimum, NUMC_DTYPE_UINT8),   E(minimum, NUMC_DTYPE_UINT16),
    E(minimum, NUMC_DTYPE_UINT32),  E(minimum, NUMC_DTYPE_UINT64),
    E(minimum, NUMC_DTYPE_FLOAT32), E(minimum, NUMC_DTYPE_FLOAT64),
};

static const NumcBinaryKernel eq_table[] = {
    E(eq, NUMC_DTYPE_INT8),    E(eq, NUMC_DTYPE_INT16),
    E(eq, NUMC_DTYPE_INT32),   E(eq, NUMC_DTYPE_INT64),
    E(eq, NUMC_DTYPE_UINT8),   E(eq, NUMC_DTYPE_UINT16),
    E(eq, NUMC_DTYPE_UINT32),  E(eq, NUMC_DTYPE_UINT64),
    E(eq, NUMC_DTYPE_FLOAT32), E(eq, NUMC_DTYPE_FLOAT64),
};

static const NumcBinaryKernel gt_table[] = {
    E(gt, NUMC_DTYPE_INT8),    E(gt, NUMC_DTYPE_INT16),
    E(gt, NUMC_DTYPE_INT32),   E(gt, NUMC_DTYPE_INT64),
    E(gt, NUMC_DTYPE_UINT8),   E(gt, NUMC_DTYPE_UINT16),
    E(gt, NUMC_DTYPE_UINT32),  E(gt, NUMC_DTYPE_UINT64),
    E(gt, NUMC_DTYPE_FLOAT32), E(gt, NUMC_DTYPE_FLOAT64),
};

static const NumcBinaryKernel lt_table[] = {
    E(lt, NUMC_DTYPE_INT8),    E(lt, NUMC_DTYPE_INT16),
    E(lt, NUMC_DTYPE_INT32),   E(lt, NUMC_DTYPE_INT64),
    E(lt, NUMC_DTYPE_UINT8),   E(lt, NUMC_DTYPE_UINT16),
    E(lt, NUMC_DTYPE_UINT32),  E(lt, NUMC_DTYPE_UINT64),
    E(lt, NUMC_DTYPE_FLOAT32), E(lt, NUMC_DTYPE_FLOAT64),
};

static const NumcBinaryKernel ge_table[] = {
    E(ge, NUMC_DTYPE_INT8),    E(ge, NUMC_DTYPE_INT16),
    E(ge, NUMC_DTYPE_INT32),   E(ge, NUMC_DTYPE_INT64),
    E(ge, NUMC_DTYPE_UINT8),   E(ge, NUMC_DTYPE_UINT16),
    E(ge, NUMC_DTYPE_UINT32),  E(ge, NUMC_DTYPE_UINT64),
    E(ge, NUMC_DTYPE_FLOAT32), E(ge, NUMC_DTYPE_FLOAT64),
};

static const NumcBinaryKernel le_table[] = {
    E(le, NUMC_DTYPE_INT8),    E(le, NUMC_DTYPE_INT16),
    E(le, NUMC_DTYPE_INT32),   E(le, NUMC_DTYPE_INT64),
    E(le, NUMC_DTYPE_UINT8),   E(le, NUMC_DTYPE_UINT16),
    E(le, NUMC_DTYPE_UINT32),  E(le, NUMC_DTYPE_UINT64),
    E(le, NUMC_DTYPE_FLOAT32), E(le, NUMC_DTYPE_FLOAT64),
};

static const NumcTernaryKernel where_table[] = {
    E(where, NUMC_DTYPE_INT8),    E(where, NUMC_DTYPE_INT16),
    E(where, NUMC_DTYPE_INT32),   E(where, NUMC_DTYPE_INT64),
    E(where, NUMC_DTYPE_UINT8),   E(where, NUMC_DTYPE_UINT16),
    E(where, NUMC_DTYPE_UINT32),  E(where, NUMC_DTYPE_UINT64),
    E(where, NUMC_DTYPE_FLOAT32), E(where, NUMC_DTYPE_FLOAT64),
};

/* ═══════════════════════════════════════════════════════════════════════
 * Public API — Binary + Scalar ops
 * ═══════════════════════════════════════════════════════════════════════ */

#define DEFINE_ELEMWISE_BINARY(NAME, TABLE)                                 \
  int numc_##NAME(const NumcArray *a, const NumcArray *b, NumcArray *out) { \
    int err = _check_binary(a, b, out);                                     \
    if (err)                                                                \
      return err;                                                           \
    _binary_op(a, b, out, TABLE);                                           \
    return 0;                                                               \
  }

#define DEFINE_ELEMWISE_SCALAR(NAME, TABLE)                       \
  int numc_##NAME##_scalar(const NumcArray *a, double scalar,     \
                           NumcArray *out) {                      \
    int err = _check_unary(a, out);                               \
    if (err)                                                      \
      return err;                                                 \
    char buf[8];                                                  \
    _double_to_dtype(scalar, a->dtype, buf);                      \
    _scalar_op(a, buf, out, TABLE);                               \
    return 0;                                                     \
  }                                                               \
  int numc_##NAME##_scalar_inplace(NumcArray *a, double scalar) { \
    return _scalar_op_inplace(a, scalar, TABLE);                  \
  }

DEFINE_ELEMWISE_BINARY(add, add_table)
DEFINE_ELEMWISE_BINARY(sub, sub_table)
DEFINE_ELEMWISE_BINARY(mul, mul_table)
DEFINE_ELEMWISE_BINARY(div, div_table)

DEFINE_ELEMWISE_BINARY(maximum, maximum_table)
DEFINE_ELEMWISE_BINARY(minimum, minimum_table)

DEFINE_ELEMWISE_BINARY(eq, eq_table)
DEFINE_ELEMWISE_BINARY(gt, gt_table)
DEFINE_ELEMWISE_BINARY(lt, lt_table)
DEFINE_ELEMWISE_BINARY(ge, ge_table)
DEFINE_ELEMWISE_BINARY(le, le_table)

DEFINE_ELEMWISE_SCALAR(add, add_table)
DEFINE_ELEMWISE_SCALAR(sub, sub_table)
DEFINE_ELEMWISE_SCALAR(mul, mul_table)
DEFINE_ELEMWISE_SCALAR(div, div_table)

DEFINE_ELEMWISE_SCALAR(eq, eq_table)
DEFINE_ELEMWISE_SCALAR(gt, gt_table)
DEFINE_ELEMWISE_SCALAR(lt, lt_table)
DEFINE_ELEMWISE_SCALAR(ge, ge_table)
DEFINE_ELEMWISE_SCALAR(le, le_table)

/* pow: non-const signature differs, stays explicit */
int numc_pow(NumcArray *a, NumcArray *b, NumcArray *out) {
  int err = _check_binary(a, b, out);
  if (err)
    return err;
  _binary_op(a, b, out, pow_table);
  return 0;
}

/* where: ternary selection */
int numc_where(const NumcArray *cond, const NumcArray *a, const NumcArray *b,
               NumcArray *out) {
  int err = _check_ternary(cond, a, b, out);
  if (err)
    return err;
  _ternary_op(cond, a, b, out, where_table);
  return 0;
}
