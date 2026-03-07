#include "dispatch.h"
#include "helpers.h"
#include "numc/dtype.h"
#include <numc/math.h>

/* ── Stamp out pow loop kernels ──────────────────────────────────────── */

/* 8/16-bit signed: branchless fixed-iteration (auto-vectorizes) */
DEFINE_BINARY_KERNEL(pow, NUMC_DTYPE_INT8, NUMC_INT8, _powi_i8(in1, in2))
DEFINE_BINARY_KERNEL(pow, NUMC_DTYPE_INT16, NUMC_INT16, _powi_i16(in1, in2))

/* 8/16-bit unsigned: branchless fixed-iteration (auto-vectorizes) */
DEFINE_BINARY_KERNEL(pow, NUMC_DTYPE_UINT8, NUMC_UINT8, _powi_u8(in1, in2))
DEFINE_BINARY_KERNEL(pow, NUMC_DTYPE_UINT16, NUMC_UINT16, _powi_u16(in1, in2))

/* 32/64-bit: variable-iteration early-exit (scalar, fast for small exp) */
DEFINE_BINARY_KERNEL_NOSIMD(pow, NUMC_DTYPE_INT32, NUMC_INT32,
                            (NUMC_INT32)_powi_signed((NUMC_INT64)in1,
                                                     (NUMC_INT64)in2))
DEFINE_BINARY_KERNEL_NOSIMD(pow, NUMC_DTYPE_UINT32, NUMC_UINT32,
                            (NUMC_UINT32)_powi_unsigned((NUMC_UINT64)in1,
                                                        (NUMC_UINT64)in2))
DEFINE_BINARY_KERNEL_NOSIMD(pow, NUMC_DTYPE_INT64, NUMC_INT64,
                            (NUMC_INT64)_powi_signed((NUMC_INT64)in1,
                                                     (NUMC_INT64)in2))
DEFINE_BINARY_KERNEL_NOSIMD(pow, NUMC_DTYPE_UINT64, NUMC_UINT64,
                            (NUMC_UINT64)_powi_unsigned((NUMC_UINT64)in1,
                                                        (NUMC_UINT64)in2))

/* float32: fused exp(in2 * log(in1)), single-precision */
DEFINE_BINARY_KERNEL_NOSIMD(pow, NUMC_DTYPE_FLOAT32, NUMC_FLOAT32,
                            _exp_f32(in2 *_log_f32(in1)))

/* float64: fused exp(in2 * log(in1)), double-precision */
DEFINE_BINARY_KERNEL_NOSIMD(pow, NUMC_DTYPE_FLOAT64, NUMC_FLOAT64,
                            _exp_f64(in2 *_log_f64(in1)))

/* ── Dispatch table ──────────────────────────────────────────────── */

static const NumcBinaryKernel pow_table[] = {
    E(pow, NUMC_DTYPE_INT8),    E(pow, NUMC_DTYPE_INT16),
    E(pow, NUMC_DTYPE_INT32),   E(pow, NUMC_DTYPE_INT64),
    E(pow, NUMC_DTYPE_UINT8),   E(pow, NUMC_DTYPE_UINT16),
    E(pow, NUMC_DTYPE_UINT32),  E(pow, NUMC_DTYPE_UINT64),
    E(pow, NUMC_DTYPE_FLOAT32), E(pow, NUMC_DTYPE_FLOAT64),
};

/* ── Public API ──────────────────────────────────────────────────── */

/* pow: non-const signature differs, stays explicit */
int numc_pow(NumcArray *a, NumcArray *b, NumcArray *out) {
  int err = _check_binary(a, b, out);
  if (err)
    return err;
  _binary_op(a, b, out, pow_table);
  return 0;
}
