#include "dispatch.h"
#include "fused.h"
#include "helpers.h"
#include "numc/dtype.h"
#include <numc/math.h>
#include <string.h>

/* ── Sum reduction kernels (integer types) ────────────────────────── */

#define STAMP_SUM(TE, CT) DEFINE_REDUCTION_KERNEL(sum, TE, CT, 0, acc + val, +)
GENERATE_INT8_INT16_NUMC_TYPES(STAMP_SUM)
GENERATE_INT32_NUMC_TYPES(STAMP_SUM)
DEFINE_REDUCTION_KERNEL(sum, NUMC_DTYPE_INT64, NUMC_INT64, 0, acc + val, +)
DEFINE_REDUCTION_KERNEL(sum, NUMC_DTYPE_UINT64, NUMC_UINT64, 0, acc + val, +)
#undef STAMP_SUM

/* ── Sum reduction kernels (float types — pairwise summation) ─────── */

DEFINE_FLOAT_REDUCTION_KERNEL(sum, NUMC_DTYPE_FLOAT32, NUMC_FLOAT32, 0,
                              _pairwise_sum_f32, +, global += local, acc + val)
DEFINE_FLOAT_REDUCTION_KERNEL(sum, NUMC_DTYPE_FLOAT64, NUMC_FLOAT64, 0,
                              _pairwise_sum_f64, +, global += local, acc + val)

/* ── Dispatch table ──────────────────────────────────────────────── */

#define R(OP, TE) [TE] = _kern_##OP##_##TE

static const NumcReductionKernel sum_table[] = {
    R(sum, NUMC_DTYPE_INT8),    R(sum, NUMC_DTYPE_INT16),
    R(sum, NUMC_DTYPE_INT32),   R(sum, NUMC_DTYPE_INT64),
    R(sum, NUMC_DTYPE_UINT8),   R(sum, NUMC_DTYPE_UINT16),
    R(sum, NUMC_DTYPE_UINT32),  R(sum, NUMC_DTYPE_UINT64),
    R(sum, NUMC_DTYPE_FLOAT32), R(sum, NUMC_DTYPE_FLOAT64),
};

#undef R

/* ── Fused row-reduce kernels for axis fast path ─────────────────
 *
 * Process all rows in a single call, eliminating per-row function
 * pointer overhead. Compiler sees the full nested loop, enabling
 * software pipelining and load/compute overlap.
 *
 * Each d[i] is independent across columns, so the inner loop
 * auto-vectorizes (vpaddd/vaddps for sum). */

#define STAMP_SUM_FUSED(TE, CT)                                               \
  static void _sum_fused_##TE(const char *restrict base, intptr_t row_stride, \
                              size_t nrows, char *restrict dst,               \
                              size_t ncols) {                                 \
    CT *restrict d = (CT *)dst;                                               \
    for (size_t r = 0; r < nrows; r++) {                                      \
      const CT *restrict s = (const CT *)(base + r * row_stride);             \
      for (size_t i = 0; i < ncols; i++)                                      \
        d[i] += s[i];                                                         \
    }                                                                         \
  }
GENERATE_NUMC_TYPES(STAMP_SUM_FUSED)
#undef STAMP_SUM_FUSED

#define F(OP, TE) [TE] = _##OP##_fused_##TE
const NumcRowReduceKernel sum_fused_table[] = {
    F(sum, NUMC_DTYPE_INT8),    F(sum, NUMC_DTYPE_INT16),
    F(sum, NUMC_DTYPE_INT32),   F(sum, NUMC_DTYPE_INT64),
    F(sum, NUMC_DTYPE_UINT8),   F(sum, NUMC_DTYPE_UINT16),
    F(sum, NUMC_DTYPE_UINT32),  F(sum, NUMC_DTYPE_UINT64),
    F(sum, NUMC_DTYPE_FLOAT32), F(sum, NUMC_DTYPE_FLOAT64),
};
#undef F

/* ── Public API ──────────────────────────────────────────────────── */

int numc_sum(const NumcArray *a, NumcArray *out) {
  int err = _check_reduce_full(a, out);
  if (err)
    return err;
  _reduce_full_op(a, out, sum_table);
  return 0;
}

int numc_sum_axis(const NumcArray *a, int axis, int keepdim, NumcArray *out) {
  int err = _check_reduce_axis(a, axis, keepdim, out);
  if (err)
    return err;

  size_t ax = (size_t)axis;

  /* Fast path: fused row-reduce.
   * Single call processes all rows, eliminating per-row call overhead. */
  if (out->is_contiguous && _iter_contiguous(a, ax)) {
    size_t reduce_len = a->shape[ax];
    intptr_t reduce_stride = (intptr_t)a->strides[ax];
    size_t slice_elems = out->size;

    memset(out->data, 0, slice_elems * a->elem_size);
    sum_fused_table[a->dtype]((const char *)a->data, reduce_stride, reduce_len,
                              (char *)out->data, slice_elems);
    return 0;
  }

  /* Generic path: per-element reduction via ND iterator */
  _reduce_axis_op(a, ax, keepdim, out, sum_table);
  return 0;
}
