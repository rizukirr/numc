#include "dispatch.h"
#include "numc/dtype.h"
#include <math.h>
#include <numc/math.h>

/* -- Stamp out ternary where ----------------------------------------- */

#define STAMP_WHERE(TE, CT) \
  DEFINE_TERNARY_KERNEL(where, TE, CT, (in_cond != 0 ? in_a : in_b))
GENERATE_NUMC_TYPES(STAMP_WHERE)
#undef STAMP_WHERE

/* -- Stamp out quaternary fma --------------------------------------- */

#define STAMP_FMA(TE, CT) \
  DEFINE_QUATERNARY_KERNEL(fma, TE, CT, in_a *in_b + in_c)
GENERATE_INT_NUMC_TYPES(STAMP_FMA)
#undef STAMP_FMA

/* float32/float64: use fma/fmaf for hardware-accelerated fused multiply-add */
DEFINE_QUATERNARY_KERNEL(fma, NUMC_DTYPE_FLOAT32, NUMC_FLOAT32,
                         fmaf(in_a, in_b, in_c))
DEFINE_QUATERNARY_KERNEL(fma, NUMC_DTYPE_FLOAT64, NUMC_FLOAT64,
                         fma(in_a, in_b, in_c))

/* -- Dispatch tables ----------------------------------------------- */

static const NumcTernaryKernel where_table[] = {
    E(where, NUMC_DTYPE_INT8),    E(where, NUMC_DTYPE_INT16),
    E(where, NUMC_DTYPE_INT32),   E(where, NUMC_DTYPE_INT64),
    E(where, NUMC_DTYPE_UINT8),   E(where, NUMC_DTYPE_UINT16),
    E(where, NUMC_DTYPE_UINT32),  E(where, NUMC_DTYPE_UINT64),
    E(where, NUMC_DTYPE_FLOAT32), E(where, NUMC_DTYPE_FLOAT64),
};

static const NumcQuaternaryKernel fma_table[] = {
    E(fma, NUMC_DTYPE_INT8),    E(fma, NUMC_DTYPE_INT16),
    E(fma, NUMC_DTYPE_INT32),   E(fma, NUMC_DTYPE_INT64),
    E(fma, NUMC_DTYPE_UINT8),   E(fma, NUMC_DTYPE_UINT16),
    E(fma, NUMC_DTYPE_UINT32),  E(fma, NUMC_DTYPE_UINT64),
    E(fma, NUMC_DTYPE_FLOAT32), E(fma, NUMC_DTYPE_FLOAT64),
};

/* -- Public API ---------------------------------------------------- */

/* where: ternary selection */
int numc_where(const NumcArray *cond, const NumcArray *a, const NumcArray *b,
               NumcArray *out) {
  int err = _check_ternary(cond, a, b, out);
  if (err)
    return err;
  _ternary_op(cond, a, b, out, where_table);
  return 0;
}

/* fma: fused multiply-add */
int numc_fma(const NumcArray *a, const NumcArray *b, const NumcArray *c,
             NumcArray *out) {
  int err = _check_quaternary(a, b, c, out);
  if (err)
    return err;
  _quaternary_op(a, b, c, out, fma_table);
  return 0;
}
