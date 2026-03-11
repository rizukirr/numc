#include "dispatch.h"
#include "helpers.h"
#include "numc/dtype.h"
#include <numc/math.h>
#include <string.h>

/* ── Max reduction kernels ───────────────────────────────────────────
 *
 * Per-type INIT = type minimum so any element is >= INIT.
 * EXPR = val > acc ? val : acc. OMP reduction(max:acc). */

#define MAX_EXPR val > acc ? val : acc

DEFINE_FLOAT_REDUCTION_KERNEL(max, NUMC_DTYPE_INT8, NUMC_INT8, INT8_MIN,
                              _vec_max_i8, max,
                              if (local > global) global = local,
                              val > acc ? val : acc)
DEFINE_FLOAT_REDUCTION_KERNEL(max, NUMC_DTYPE_INT16, NUMC_INT16, INT16_MIN,
                              _vec_max_i16, max,
                              if (local > global) global = local,
                              val > acc ? val : acc)
DEFINE_FLOAT_REDUCTION_KERNEL(max, NUMC_DTYPE_INT32, NUMC_INT32, INT32_MIN,
                              _vec_max_i32, max,
                              if (local > global) global = local,
                              val > acc ? val : acc)
DEFINE_FLOAT_REDUCTION_KERNEL(max, NUMC_DTYPE_INT64, NUMC_INT64, INT64_MIN,
                              _vec_max_i64, max,
                              if (local > global) global = local,
                              val > acc ? val : acc)
DEFINE_FLOAT_REDUCTION_KERNEL(max, NUMC_DTYPE_UINT8, NUMC_UINT8, 0, _vec_max_u8,
                              max, if (local > global) global = local,
                              val > acc ? val : acc)
DEFINE_FLOAT_REDUCTION_KERNEL(max, NUMC_DTYPE_UINT16, NUMC_UINT16, 0,
                              _vec_max_u16, max,
                              if (local > global) global = local,
                              val > acc ? val : acc)
DEFINE_FLOAT_REDUCTION_KERNEL(max, NUMC_DTYPE_UINT32, NUMC_UINT32, 0,
                              _vec_max_u32, max,
                              if (local > global) global = local,
                              val > acc ? val : acc)
DEFINE_FLOAT_REDUCTION_KERNEL(max, NUMC_DTYPE_UINT64, NUMC_UINT64, 0,
                              _vec_max_u64, max,
                              if (local > global) global = local,
                              val > acc ? val : acc)
DEFINE_FLOAT_REDUCTION_KERNEL(max, NUMC_DTYPE_FLOAT32, NUMC_FLOAT32, -INFINITY,
                              _vec_max_f32, max,
                              if (local > global) global = local,
                              val > acc ? val : acc)
DEFINE_FLOAT_REDUCTION_KERNEL(max, NUMC_DTYPE_FLOAT64, NUMC_FLOAT64, -INFINITY,
                              _vec_max_f64, max,
                              if (local > global) global = local,
                              val > acc ? val : acc)

#undef MAX_EXPR

/* ── Dispatch table ──────────────────────────────────────────────── */

#define R(OP, TE) [TE] = _kern_##OP##_##TE

static const NumcReductionKernel max_table[] = {
    R(max, NUMC_DTYPE_INT8),    R(max, NUMC_DTYPE_INT16),
    R(max, NUMC_DTYPE_INT32),   R(max, NUMC_DTYPE_INT64),
    R(max, NUMC_DTYPE_UINT8),   R(max, NUMC_DTYPE_UINT16),
    R(max, NUMC_DTYPE_UINT32),  R(max, NUMC_DTYPE_UINT64),
    R(max, NUMC_DTYPE_FLOAT32), R(max, NUMC_DTYPE_FLOAT64),
};

#undef R

/* ── Fused row-reduce kernels ───────────────────────────────────── */

typedef void (*NumcRowReduceKernel)(const char *restrict base,
                                    intptr_t row_stride, size_t nrows,
                                    char *restrict dst, size_t ncols);

#define STAMP_MAX_FUSED(TE, CT)                                               \
  static void _max_fused_##TE(const char *restrict base, intptr_t row_stride, \
                              size_t nrows, char *restrict dst,               \
                              size_t ncols) {                                 \
    CT *restrict d = (CT *)dst;                                               \
    for (size_t r = 0; r < nrows; r++) {                                      \
      const CT *restrict s = (const CT *)(base + r * row_stride);             \
      for (size_t i = 0; i < ncols; i++)                                      \
        d[i] = s[i] > d[i] ? s[i] : d[i];                                     \
    }                                                                         \
  }
GENERATE_NUMC_TYPES(STAMP_MAX_FUSED)
#undef STAMP_MAX_FUSED

#define F(OP, TE) [TE] = _##OP##_fused_##TE
static const NumcRowReduceKernel max_fused_table[] = {
    F(max, NUMC_DTYPE_INT8),    F(max, NUMC_DTYPE_INT16),
    F(max, NUMC_DTYPE_INT32),   F(max, NUMC_DTYPE_INT64),
    F(max, NUMC_DTYPE_UINT8),   F(max, NUMC_DTYPE_UINT16),
    F(max, NUMC_DTYPE_UINT32),  F(max, NUMC_DTYPE_UINT64),
    F(max, NUMC_DTYPE_FLOAT32), F(max, NUMC_DTYPE_FLOAT64),
};
#undef F

/* ── Public API ──────────────────────────────────────────────────── */

int numc_max(const NumcArray *a, NumcArray *out) {
  int err = _check_reduce_full(a, out);
  if (err)
    return err;
  _reduce_full_op(a, out, max_table);
  return 0;
}

int numc_max_axis(const NumcArray *a, int axis, int keepdim, NumcArray *out) {
  int err = _check_reduce_axis(a, axis, keepdim, out);
  if (err)
    return err;

  size_t ax = (size_t)axis;

  /* Fast path: copy first slice, then fused row-reduce remaining.
   * Single call processes all remaining rows. */
  if (out->is_contiguous && _iter_contiguous(a, ax)) {
    size_t reduce_len = a->shape[ax];
    intptr_t reduce_stride = (intptr_t)a->strides[ax];
    size_t slice_elems = out->size;

    const char *base = (const char *)a->data;
    memcpy(out->data, base, slice_elems * a->elem_size);
    max_fused_table[a->dtype](base + reduce_stride, reduce_stride,
                              reduce_len - 1, (char *)out->data, slice_elems);
    return 0;
  }

  /* Generic path: per-element reduction via ND iterator */
  _reduce_axis_op(a, ax, keepdim, out, max_table);
  return 0;
}
