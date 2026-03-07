#include "dispatch.h"
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
#define STAMP_DIV_S(TE, CT) DEFINE_INT_DIV_KERNEL(TE, CT, true)
GENERATE_SIGNED_INT_NUMC_TYPES(STAMP_DIV_S)
#undef STAMP_DIV_S

#define STAMP_DIV_U(TE, CT) DEFINE_INT_DIV_KERNEL(TE, CT, false)
GENERATE_UNSIGNED_INT_NUMC_TYPES(STAMP_DIV_U)
#undef STAMP_DIV_U

#define STAMP_DIV_F(TE, CT) DEFINE_FLOAT_DIV_KERNEL(TE, CT)
GENERATE_FLOAT_NUMC_TYPES(STAMP_DIV_F)
#undef STAMP_DIV_F

/* ── Dispatch tables ─────────────────────────────────────────────── */

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

/* ── Public API ──────────────────────────────────────────────────── */

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

DEFINE_ELEMWISE_SCALAR(add, add_table)
DEFINE_ELEMWISE_SCALAR(sub, sub_table)
DEFINE_ELEMWISE_SCALAR(mul, mul_table)
DEFINE_ELEMWISE_SCALAR(div, div_table)
