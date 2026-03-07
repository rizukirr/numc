#include "dispatch.h"
#include "fused.h"
#include "helpers.h"
#include "numc/dtype.h"
#include <numc/math.h>
#include <string.h>

/* ── Private sum kernels (mean calls sum then divides) ────────────── */

#define STAMP_SUM(TE, CT) DEFINE_REDUCTION_KERNEL(sum, TE, CT, 0, acc + val, +)
GENERATE_INT8_INT16_NUMC_TYPES(STAMP_SUM)
GENERATE_INT32_NUMC_TYPES(STAMP_SUM)
DEFINE_REDUCTION_KERNEL(sum, NUMC_DTYPE_INT64, NUMC_INT64, 0, acc + val, +)
DEFINE_REDUCTION_KERNEL(sum, NUMC_DTYPE_UINT64, NUMC_UINT64, 0, acc + val, +)
#undef STAMP_SUM

DEFINE_FLOAT_REDUCTION_KERNEL(sum, NUMC_DTYPE_FLOAT32, NUMC_FLOAT32, 0,
                              _pairwise_sum_f32, +, global += local, acc + val)
DEFINE_FLOAT_REDUCTION_KERNEL(sum, NUMC_DTYPE_FLOAT64, NUMC_FLOAT64, 0,
                              _pairwise_sum_f64, +, global += local, acc + val)

/* ── Mean reduction kernels ──────────────────────────────────────────
 *
 * mean = sum / count. Reuse existing sum kernels (including pairwise
 * summation for floats), then divide by n. Integer types get truncating
 * division — same overflow semantics as sum. */

/* int8/int16: promote through float for division */
#define STAMP_MEAN_SMALL(TE, CT)                                  \
  static void _kern_mean_##TE(const char *a, char *out, size_t n, \
                              intptr_t sa) {                      \
    _kern_sum_##TE(a, out, n, sa);                                \
    if (n > 0)                                                    \
      *(CT *)out = (CT)((float)*(CT *)out / (float)n);            \
  }
GENERATE_INT8_INT16_NUMC_TYPES(STAMP_MEAN_SMALL)
#undef STAMP_MEAN_SMALL

/* int32/uint32/int64/uint64: promote through double for division */
#define STAMP_MEAN_DBL(TE, CT)                                    \
  static void _kern_mean_##TE(const char *a, char *out, size_t n, \
                              intptr_t sa) {                      \
    _kern_sum_##TE(a, out, n, sa);                                \
    if (n > 0)                                                    \
      *(CT *)out = (CT)((double)*(CT *)out / (double)n);          \
  }
GENERATE_INT32_NUMC_TYPES(STAMP_MEAN_DBL)
STAMP_MEAN_DBL(NUMC_DTYPE_INT64, NUMC_INT64)
STAMP_MEAN_DBL(NUMC_DTYPE_UINT64, NUMC_UINT64)
#undef STAMP_MEAN_DBL

/* float32/float64: native division */
#define STAMP_MEAN_FLOAT(TE, CT)                                  \
  static void _kern_mean_##TE(const char *a, char *out, size_t n, \
                              intptr_t sa) {                      \
    _kern_sum_##TE(a, out, n, sa);                                \
    if (n > 0)                                                    \
      *(CT *)out /= (CT)n;                                        \
  }
STAMP_MEAN_FLOAT(NUMC_DTYPE_FLOAT32, NUMC_FLOAT32)
STAMP_MEAN_FLOAT(NUMC_DTYPE_FLOAT64, NUMC_FLOAT64)
#undef STAMP_MEAN_FLOAT

/* ── Dispatch table ──────────────────────────────────────────────── */

#define R(OP, TE) [TE] = _kern_##OP##_##TE

static const NumcReductionKernel mean_table[] = {
    R(mean, NUMC_DTYPE_INT8),    R(mean, NUMC_DTYPE_INT16),
    R(mean, NUMC_DTYPE_INT32),   R(mean, NUMC_DTYPE_INT64),
    R(mean, NUMC_DTYPE_UINT8),   R(mean, NUMC_DTYPE_UINT16),
    R(mean, NUMC_DTYPE_UINT32),  R(mean, NUMC_DTYPE_UINT64),
    R(mean, NUMC_DTYPE_FLOAT32), R(mean, NUMC_DTYPE_FLOAT64),
};

#undef R

/* ── Divide-by-count kernels for axis mean fast path ─────────────
 *
 * d[i] /= count — element-wise divide of output buffer.
 * Each d[i] is independent, so this auto-vectorizes. */

/* int8/int16: promote through float (vdivps — 8 lanes) */
#define STAMP_DIV_COUNT_SMALL(TE, CT)                               \
  static void _div_count_##TE(char *data, size_t n, size_t count) { \
    CT *d = (CT *)data;                                             \
    float fc = (float)count;                                        \
    for (size_t i = 0; i < n; i++)                                  \
      d[i] = (CT)((float)d[i] / fc);                                \
  }
GENERATE_INT8_INT16_NUMC_TYPES(STAMP_DIV_COUNT_SMALL)
#undef STAMP_DIV_COUNT_SMALL

/* int32/uint32/int64/uint64: promote through double (vdivpd — 4 lanes).
 * int64/uint64 lose precision above 2^53, acceptable for mean. */
#define STAMP_DIV_COUNT_DBL(TE, CT)                                 \
  static void _div_count_##TE(char *data, size_t n, size_t count) { \
    CT *d = (CT *)data;                                             \
    double dc = (double)count;                                      \
    for (size_t i = 0; i < n; i++)                                  \
      d[i] = (CT)((double)d[i] / dc);                               \
  }
GENERATE_INT32_NUMC_TYPES(STAMP_DIV_COUNT_DBL)
STAMP_DIV_COUNT_DBL(NUMC_DTYPE_INT64, NUMC_INT64)
STAMP_DIV_COUNT_DBL(NUMC_DTYPE_UINT64, NUMC_UINT64)
#undef STAMP_DIV_COUNT_DBL

/* float32/float64: native division */
static void _div_count_NUMC_DTYPE_FLOAT32(char *data, size_t n, size_t count) {
  NUMC_FLOAT32 *d = (NUMC_FLOAT32 *)data;
  NUMC_FLOAT32 cc = (NUMC_FLOAT32)count;
  for (size_t i = 0; i < n; i++)
    d[i] /= cc;
}

static void _div_count_NUMC_DTYPE_FLOAT64(char *data, size_t n, size_t count) {
  NUMC_FLOAT64 *d = (NUMC_FLOAT64 *)data;
  NUMC_FLOAT64 dc = (NUMC_FLOAT64)count;
  for (size_t i = 0; i < n; i++)
    d[i] /= dc;
}

#define D(TE) [TE] = _div_count_##TE
static const NumcDivCountKernel div_count_table[] = {
    D(NUMC_DTYPE_INT8),    D(NUMC_DTYPE_INT16),  D(NUMC_DTYPE_INT32),
    D(NUMC_DTYPE_INT64),   D(NUMC_DTYPE_UINT8),  D(NUMC_DTYPE_UINT16),
    D(NUMC_DTYPE_UINT32),  D(NUMC_DTYPE_UINT64), D(NUMC_DTYPE_FLOAT32),
    D(NUMC_DTYPE_FLOAT64),
};
#undef D

/* ── Public API ──────────────────────────────────────────────────── */

int numc_mean(const NumcArray *a, NumcArray *out) {
  int err = _check_reduce_full(a, out);
  if (err)
    return err;
  _reduce_full_op(a, out, mean_table);
  return 0;
}

int numc_mean_axis(const NumcArray *a, int axis, int keepdim, NumcArray *out) {
  int err = _check_reduce_axis(a, axis, keepdim, out);
  if (err)
    return err;

  size_t ax = (size_t)axis;

  /* Fast path: fused row-reduce + divide by count.
   * Single call processes all rows, eliminating per-row call overhead. */
  if (out->is_contiguous && _iter_contiguous(a, ax)) {
    size_t reduce_len = a->shape[ax];
    intptr_t reduce_stride = (intptr_t)a->strides[ax];
    size_t slice_elems = out->size;

    memset(out->data, 0, slice_elems * a->elem_size);
    sum_fused_table[a->dtype]((const char *)a->data, reduce_stride, reduce_len,
                              (char *)out->data, slice_elems);
    div_count_table[a->dtype]((char *)out->data, slice_elems, reduce_len);
    return 0;
  }

  /* Generic path: per-element reduction via ND iterator */
  _reduce_axis_op(a, ax, keepdim, out, mean_table);
  return 0;
}
