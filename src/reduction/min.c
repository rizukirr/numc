#include "dispatch.h"
#include "helpers.h"
#include "numc/dtype.h"
#include <numc/math.h>
#include <string.h>

/* ── Min reduction kernels ───────────────────────────────────────────
 *
 * Per-type INIT = type maximum so any element is <= INIT.
 * EXPR = val < acc ? val : acc. OMP reduction(min:acc). */

#define MIN_EXPR val < acc ? val : acc

DEFINE_FLOAT_REDUCTION_KERNEL(min, NUMC_DTYPE_INT8, NUMC_INT8, INT8_MAX,
                              _vec_min_i8, min,
                              if (local < global) global = local,
                              val < acc ? val : acc)
DEFINE_FLOAT_REDUCTION_KERNEL(min, NUMC_DTYPE_INT16, NUMC_INT16, INT16_MAX,
                              _vec_min_i16, min,
                              if (local < global) global = local,
                              val < acc ? val : acc)
DEFINE_FLOAT_REDUCTION_KERNEL(min, NUMC_DTYPE_INT32, NUMC_INT32, INT32_MAX,
                              _vec_min_i32, min,
                              if (local < global) global = local,
                              val < acc ? val : acc)
DEFINE_FLOAT_REDUCTION_KERNEL(min, NUMC_DTYPE_INT64, NUMC_INT64, INT64_MAX,
                              _vec_min_i64, min,
                              if (local < global) global = local,
                              val < acc ? val : acc)
DEFINE_FLOAT_REDUCTION_KERNEL(min, NUMC_DTYPE_UINT8, NUMC_UINT8, UINT8_MAX,
                              _vec_min_u8, min,
                              if (local < global) global = local,
                              val < acc ? val : acc)
DEFINE_FLOAT_REDUCTION_KERNEL(min, NUMC_DTYPE_UINT16, NUMC_UINT16, UINT16_MAX,
                              _vec_min_u16, min,
                              if (local < global) global = local,
                              val < acc ? val : acc)
DEFINE_FLOAT_REDUCTION_KERNEL(min, NUMC_DTYPE_UINT32, NUMC_UINT32, UINT32_MAX,
                              _vec_min_u32, min,
                              if (local < global) global = local,
                              val < acc ? val : acc)
DEFINE_FLOAT_REDUCTION_KERNEL(min, NUMC_DTYPE_UINT64, NUMC_UINT64, UINT64_MAX,
                              _vec_min_u64, min,
                              if (local < global) global = local,
                              val < acc ? val : acc)
DEFINE_FLOAT_REDUCTION_KERNEL(min, NUMC_DTYPE_FLOAT32, NUMC_FLOAT32, INFINITY,
                              _vec_min_f32, min,
                              if (local < global) global = local,
                              val < acc ? val : acc)
DEFINE_FLOAT_REDUCTION_KERNEL(min, NUMC_DTYPE_FLOAT64, NUMC_FLOAT64, INFINITY,
                              _vec_min_f64, min,
                              if (local < global) global = local,
                              val < acc ? val : acc)

#undef MIN_EXPR

/* ── Dispatch table ──────────────────────────────────────────────── */

#define R(OP, TE) [TE] = _kern_##OP##_##TE

static const NumcReductionKernel min_table[] = {
    R(min, NUMC_DTYPE_INT8),    R(min, NUMC_DTYPE_INT16),
    R(min, NUMC_DTYPE_INT32),   R(min, NUMC_DTYPE_INT64),
    R(min, NUMC_DTYPE_UINT8),   R(min, NUMC_DTYPE_UINT16),
    R(min, NUMC_DTYPE_UINT32),  R(min, NUMC_DTYPE_UINT64),
    R(min, NUMC_DTYPE_FLOAT32), R(min, NUMC_DTYPE_FLOAT64),
};

#undef R

/* ── Fused row-reduce kernels ───────────────────────────────────── */

typedef void (*NumcRowReduceKernel)(const char *restrict base,
                                    intptr_t row_stride, size_t nrows,
                                    char *restrict dst, size_t ncols);

#define STAMP_MIN_FUSED(TE, CT)                                               \
  static void _min_fused_##TE(const char *restrict base, intptr_t row_stride, \
                              size_t nrows, char *restrict dst,               \
                              size_t ncols) {                                 \
    CT *restrict d = (CT *)dst;                                               \
    size_t r = 0;                                                             \
    for (; r + 4 <= nrows; r += 4) {                                          \
      const CT *restrict s0 = (const CT *)(base + r * row_stride);            \
      const CT *restrict s1 = (const CT *)(base + (r + 1) * row_stride);      \
      const CT *restrict s2 = (const CT *)(base + (r + 2) * row_stride);      \
      const CT *restrict s3 = (const CT *)(base + (r + 3) * row_stride);      \
      for (size_t i = 0; i < ncols; i++) {                                    \
        CT v = s0[i];                                                         \
        CT v1 = s1[i];                                                        \
        CT v2 = s2[i];                                                        \
        CT v3 = s3[i];                                                        \
        v = v1 < v ? v1 : v;                                                  \
        v = v2 < v ? v2 : v;                                                  \
        v = v3 < v ? v3 : v;                                                  \
        d[i] = v < d[i] ? v : d[i];                                           \
      }                                                                       \
    }                                                                         \
    for (; r < nrows; r++) {                                                  \
      const CT *restrict s = (const CT *)(base + r * row_stride);             \
      for (size_t i = 0; i < ncols; i++)                                      \
        d[i] = s[i] < d[i] ? s[i] : d[i];                                     \
    }                                                                         \
  }
GENERATE_NUMC_TYPES(STAMP_MIN_FUSED)
#undef STAMP_MIN_FUSED

#define F(OP, TE) [TE] = _##OP##_fused_##TE
static const NumcRowReduceKernel min_fused_table[] = {
    F(min, NUMC_DTYPE_INT8),    F(min, NUMC_DTYPE_INT16),
    F(min, NUMC_DTYPE_INT32),   F(min, NUMC_DTYPE_INT64),
    F(min, NUMC_DTYPE_UINT8),   F(min, NUMC_DTYPE_UINT16),
    F(min, NUMC_DTYPE_UINT32),  F(min, NUMC_DTYPE_UINT64),
    F(min, NUMC_DTYPE_FLOAT32), F(min, NUMC_DTYPE_FLOAT64),
};
#undef F

/* ── Public API ──────────────────────────────────────────────────── */

int numc_min(const NumcArray *a, NumcArray *out) {
  int err = _check_reduce_full(a, out);
  if (err)
    return err;
  _reduce_full_op(a, out, min_table);
  return 0;
}

int numc_min_axis(const NumcArray *a, int axis, int keepdim, NumcArray *out) {
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
    min_fused_table[a->dtype](base + reduce_stride, reduce_stride,
                              reduce_len - 1, (char *)out->data, slice_elems);
    return 0;
  }

  /* Generic path: per-element reduction via ND iterator */
  _reduce_axis_op(a, ax, keepdim, out, min_table);
  return 0;
}
