#include "dispatch.h"
#include <numc/math.h>
#include <string.h>

/* ── Stamp sum reduction kernels (integer types only) ────────────── */

#define STAMP_SUM(TE, CT) DEFINE_REDUCTION_KERNEL(sum, TE, CT, 0, acc + val, +)
GENERATE_INT8_INT16_NUMC_TYPES(STAMP_SUM)
GENERATE_INT32(STAMP_SUM)
DEFINE_REDUCTION_KERNEL(sum, NUMC_DTYPE_INT64, int64_t, 0, acc + val, +)
DEFINE_REDUCTION_KERNEL(sum, NUMC_DTYPE_UINT64, uint64_t, 0, acc + val, +)
#undef STAMP_SUM

/* ── Pairwise summation for float types ──────────────────────────────
 *
 * IEEE-754 non-associativity prevents the compiler from vectorizing
 * a serial `acc += val` loop — it emits scalar vaddss/vaddsd with a
 * single accumulator (latency-bound, ~4 cycles per add).
 *
 * Pairwise summation uses 8 independent accumulators in the leaf,
 * which the SLP vectorizer packs into vaddps/vaddpd (throughput-bound,
 * 8 or 4 floats per vector add). Recursive splitting keeps the
 * accumulators independent across blocks.
 *
 * Block size 128: matches NumPy's pairwise_sum implementation.
 * 8 accumulators: fills one ymm (float32) or two ymm (float64).
 */

#define PAIRWISE_BLOCKSIZE 128

static float _pairwise_sum_f32(const float *restrict a, size_t n) {
  if (n <= PAIRWISE_BLOCKSIZE) {
    float r0 = 0, r1 = 0, r2 = 0, r3 = 0;
    float r4 = 0, r5 = 0, r6 = 0, r7 = 0;
    size_t i = 0, n8 = n & ~(size_t)7;
    for (; i < n8; i += 8) {
      r0 += a[i];
      r1 += a[i + 1];
      r2 += a[i + 2];
      r3 += a[i + 3];
      r4 += a[i + 4];
      r5 += a[i + 5];
      r6 += a[i + 6];
      r7 += a[i + 7];
    }
    float sum = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
    for (; i < n; i++)
      sum += a[i];
    return sum;
  }
  size_t half = n / 2;
  return _pairwise_sum_f32(a, half) + _pairwise_sum_f32(a + half, n - half);
}

static double _pairwise_sum_f64(const double *restrict a, size_t n) {
  if (n <= PAIRWISE_BLOCKSIZE) {
    double r0 = 0, r1 = 0, r2 = 0, r3 = 0;
    double r4 = 0, r5 = 0, r6 = 0, r7 = 0;
    size_t i = 0, n8 = n & ~(size_t)7;
    for (; i < n8; i += 8) {
      r0 += a[i];
      r1 += a[i + 1];
      r2 += a[i + 2];
      r3 += a[i + 3];
      r4 += a[i + 4];
      r5 += a[i + 5];
      r6 += a[i + 6];
      r7 += a[i + 7];
    }
    double sum = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
    for (; i < n; i++)
      sum += a[i];
    return sum;
  }
  size_t half = n / 2;
  return _pairwise_sum_f64(a, half) + _pairwise_sum_f64(a + half, n - half);
}

static void _kern_sum_NUMC_DTYPE_FLOAT32(const char *a, char *out, size_t n,
                                         intptr_t sa) {
  if (n == 0) {
    *(float *)out = 0;
    return;
  }
  if (sa == (intptr_t)sizeof(float)) {
    *(float *)out = _pairwise_sum_f32((const float *)a, n);
  } else {
    float acc = 0;
    for (size_t i = 0; i < n; i++)
      acc += *(const float *)(a + i * sa);
    *(float *)out = acc;
  }
}

static void _kern_sum_NUMC_DTYPE_FLOAT64(const char *a, char *out, size_t n,
                                         intptr_t sa) {
  if (n == 0) {
    *(double *)out = 0;
    return;
  }
  if (sa == (intptr_t)sizeof(double)) {
    *(double *)out = _pairwise_sum_f64((const double *)a, n);
  } else {
    double acc = 0;
    for (size_t i = 0; i < n; i++)
      acc += *(const double *)(a + i * sa);
    *(double *)out = acc;
  }
}

/* ── Accumulate kernels for axis fast path ────────────────────────
 *
 * d[i] += s[i] — element-wise add into output buffer.
 * Unlike reduction (serial acc += val), each d[i] is independent,
 * so this auto-vectorizes to vaddps/vpaddd regardless of IEEE-754.
 *
 * Used by the axis fast path: zero output, then for each slice along
 * the reduction axis, accumulate it into output via contiguous vector adds.
 * This flips the loop order from per-column strided reduction (cache-hostile)
 * to per-row contiguous adds (cache-friendly).
 */

typedef void (*NumcAccumKernel)(const char *restrict src, char *restrict dst,
                                size_t n);

#define STAMP_ACCUM(TE, CT)                                                    \
  static void _accum_##TE(const char *restrict src, char *restrict dst,        \
                          size_t n) {                                          \
    const CT *restrict s = (const CT *)src;                                    \
    CT *restrict d = (CT *)dst;                                                \
    for (size_t i = 0; i < n; i++)                                             \
      d[i] += s[i];                                                            \
  }
GENERATE_NUMC_TYPES(STAMP_ACCUM)
#undef STAMP_ACCUM

#define A(TE) [TE] = _accum_##TE
static const NumcAccumKernel _sum_accum_table[] = {
    A(NUMC_DTYPE_INT8),    A(NUMC_DTYPE_INT16),  A(NUMC_DTYPE_INT32),
    A(NUMC_DTYPE_INT64),   A(NUMC_DTYPE_UINT8),  A(NUMC_DTYPE_UINT16),
    A(NUMC_DTYPE_UINT32),  A(NUMC_DTYPE_UINT64), A(NUMC_DTYPE_FLOAT32),
    A(NUMC_DTYPE_FLOAT64),
};
#undef A

/* Check if the non-reduced dims of 'a' form a contiguous block.
 * Walk from last dim backwards (skipping reduction axis), verify
 * strides match C-order contiguous layout. */
static inline bool _iter_contiguous(const struct NumcArray *a, size_t axis) {
  size_t expected = a->elem_size;
  for (size_t i = a->dim; i-- > 0;) {
    if (i == axis)
      continue;
    if (a->strides[i] != expected)
      return false;
    expected *= a->shape[i];
  }
  return true;
}

/* ── Dispatch table ──────────────────────────────────────────────── */

#define R(OP, TE) [TE] = _kern_##OP##_##TE

static const NumcReductionKernel _sum_table[] = {
    R(sum, NUMC_DTYPE_INT8),    R(sum, NUMC_DTYPE_INT16),
    R(sum, NUMC_DTYPE_INT32),   R(sum, NUMC_DTYPE_INT64),
    R(sum, NUMC_DTYPE_UINT8),   R(sum, NUMC_DTYPE_UINT16),
    R(sum, NUMC_DTYPE_UINT32),  R(sum, NUMC_DTYPE_UINT64),
    R(sum, NUMC_DTYPE_FLOAT32), R(sum, NUMC_DTYPE_FLOAT64),
};

/* ── Public API ──────────────────────────────────────────────────── */

int numc_sum(const NumcArray *a, NumcArray *out) {
  int err = _check_reduce_full(a, out);
  if (err)
    return err;
  _reduce_full_op(a, out, _sum_table);
  return 0;
}

int numc_sum_axis(const NumcArray *a, int axis, int keepdim, NumcArray *out) {
  int err = _check_reduce_axis(a, axis, keepdim, out);
  if (err)
    return err;

  size_t ax = (size_t)axis;

  /* Fast path: accumulate contiguous slices.
   * When the non-reduced dims are contiguous in both input and output,
   * we can flip the loop order: instead of N per-element strided reductions,
   * do reduce_len contiguous vector adds (d[i] += s[i]).
   * This is cache-friendly and auto-vectorizes to vaddps/vpaddd. */
  if (out->is_contiguous && _iter_contiguous(a, ax)) {
    size_t reduce_len = a->shape[ax];
    intptr_t reduce_stride = (intptr_t)a->strides[ax];
    size_t slice_elems = out->size;

    memset(out->data, 0, slice_elems * a->elem_size);

    NumcAccumKernel accum = _sum_accum_table[a->dtype];
    const char *base = (const char *)a->data;
    for (size_t r = 0; r < reduce_len; r++)
      accum(base + r * reduce_stride, (char *)out->data, slice_elems);

    return 0;
  }

  /* Generic path: per-element reduction via ND iterator */
  _reduce_axis_op(a, ax, keepdim, out, _sum_table);
  return 0;
}
