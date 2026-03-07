#include "dispatch.h"
#include "helpers.h"
#include "numc/dtype.h"
#include <numc/math.h>

/* ── Argmax reduction kernels ────────────────────────────────────────
 *
 * Per-type INIT = type minimum so any element is > INIT.
 * Output is always int64_t (index of maximum element). */

DEFINE_ARGREDUCTION_KERNEL(argmax, NUMC_DTYPE_INT8, NUMC_INT8, INT8_MIN, >)
DEFINE_ARGREDUCTION_KERNEL(argmax, NUMC_DTYPE_INT16, NUMC_INT16, INT16_MIN, >)
DEFINE_ARGREDUCTION_KERNEL(argmax, NUMC_DTYPE_INT32, NUMC_INT32, INT32_MIN, >)
DEFINE_ARGREDUCTION_KERNEL(argmax, NUMC_DTYPE_INT64, NUMC_INT64, INT64_MIN, >)
DEFINE_ARGREDUCTION_KERNEL(argmax, NUMC_DTYPE_UINT8, NUMC_UINT8, 0, >)
DEFINE_ARGREDUCTION_KERNEL(argmax, NUMC_DTYPE_UINT16, NUMC_UINT16, 0, >)
DEFINE_ARGREDUCTION_KERNEL(argmax, NUMC_DTYPE_UINT32, NUMC_UINT32, 0, >)
DEFINE_ARGREDUCTION_KERNEL(argmax, NUMC_DTYPE_UINT64, NUMC_UINT64, 0, >)
DEFINE_FLOAT_ARGREDUCTION_KERNEL(argmax, NUMC_DTYPE_FLOAT32, NUMC_FLOAT32,
                                 -INFINITY, _vec_max_f32, >)
DEFINE_FLOAT_ARGREDUCTION_KERNEL(argmax, NUMC_DTYPE_FLOAT64, NUMC_FLOAT64,
                                 -INFINITY, _vec_max_f64, >)

/* ── Dispatch table ──────────────────────────────────────────────── */

#define R(OP, TE) [TE] = _kern_##OP##_##TE

static const NumcReductionKernel argmax_table[] = {
    R(argmax, NUMC_DTYPE_INT8),    R(argmax, NUMC_DTYPE_INT16),
    R(argmax, NUMC_DTYPE_INT32),   R(argmax, NUMC_DTYPE_INT64),
    R(argmax, NUMC_DTYPE_UINT8),   R(argmax, NUMC_DTYPE_UINT16),
    R(argmax, NUMC_DTYPE_UINT32),  R(argmax, NUMC_DTYPE_UINT64),
    R(argmax, NUMC_DTYPE_FLOAT32), R(argmax, NUMC_DTYPE_FLOAT64),
};

#undef R

/* ── Fused argmax row-reduce kernels ──────────────────────────────
 *
 * Tracks best value (in VLA scratch buffer) + index (in output).
 * Inner loop auto-vectorizes the comparison; the conditional index
 * update is scalar but amortized across rows. */

typedef void (*NumcArgRowReduceKernel)(const char *restrict base,
                                       intptr_t row_stride, size_t nrows,
                                       char *restrict dst, size_t ncols);

#define STAMP_ARGMAX_FUSED(TE, CT)                                   \
  static void _argmax_fused_##TE(const char *restrict base,          \
                                 intptr_t row_stride, size_t nrows,  \
                                 char *restrict dst, size_t ncols) { \
    int64_t *restrict idx = (int64_t *)dst;                          \
    CT best[ncols]; /* VLA scratch for best values */                \
    const CT *restrict first = (const CT *)base;                     \
    for (size_t i = 0; i < ncols; i++) {                             \
      best[i] = first[i];                                            \
      idx[i] = 0;                                                    \
    }                                                                \
    for (size_t r = 1; r < nrows; r++) {                             \
      const CT *restrict row = (const CT *)(base + r * row_stride);  \
      for (size_t i = 0; i < ncols; i++) {                           \
        if (row[i] > best[i]) {                                      \
          best[i] = row[i];                                          \
          idx[i] = (int64_t)r;                                       \
        }                                                            \
      }                                                              \
    }                                                                \
  }
GENERATE_NUMC_TYPES(STAMP_ARGMAX_FUSED)
#undef STAMP_ARGMAX_FUSED

#define F(OP, TE) [TE] = _##OP##_fused_##TE
static const NumcArgRowReduceKernel argmax_fused_table[] = {
    F(argmax, NUMC_DTYPE_INT8),    F(argmax, NUMC_DTYPE_INT16),
    F(argmax, NUMC_DTYPE_INT32),   F(argmax, NUMC_DTYPE_INT64),
    F(argmax, NUMC_DTYPE_UINT8),   F(argmax, NUMC_DTYPE_UINT16),
    F(argmax, NUMC_DTYPE_UINT32),  F(argmax, NUMC_DTYPE_UINT64),
    F(argmax, NUMC_DTYPE_FLOAT32), F(argmax, NUMC_DTYPE_FLOAT64),
};
#undef F

/* ── Public API ──────────────────────────────────────────────────── */

int numc_argmax(const NumcArray *a, NumcArray *out) {
  int err = _check_argreduce_full(a, out);
  if (err)
    return err;
  _reduce_full_op(a, out, argmax_table);
  return 0;
}

int numc_argmax_axis(const NumcArray *a, int axis, int keepdim,
                     NumcArray *out) {
  int err = _check_argreduce_axis(a, axis, keepdim, out);
  if (err)
    return err;

  size_t ax = (size_t)axis;

  /* Fast path: fused row-reduce with value+index tracking */
  if (out->is_contiguous && _iter_contiguous(a, ax)) {
    size_t reduce_len = a->shape[ax];
    intptr_t reduce_stride = (intptr_t)a->strides[ax];
    size_t slice_elems = out->size;

    argmax_fused_table[a->dtype]((const char *)a->data, reduce_stride,
                                 reduce_len, (char *)out->data, slice_elems);
    return 0;
  }

  /* Generic path: per-element reduction via ND iterator */
  _reduce_axis_op(a, ax, keepdim, out, argmax_table);
  return 0;
}
