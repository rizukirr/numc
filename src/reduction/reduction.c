#include "dispatch.h"
#include "helpers.h"
#include <math.h>
#include <numc/math.h>
#include <string.h>

/* ── Sum reduction kernels (integer types) ────────────────────────── */

#define STAMP_SUM(TE, CT) DEFINE_REDUCTION_KERNEL(sum, TE, CT, 0, acc + val, +)
GENERATE_INT8_INT16_NUMC_TYPES(STAMP_SUM)
GENERATE_INT32(STAMP_SUM)
DEFINE_REDUCTION_KERNEL(sum, NUMC_DTYPE_INT64, int64_t, 0, acc + val, +)
DEFINE_REDUCTION_KERNEL(sum, NUMC_DTYPE_UINT64, uint64_t, 0, acc + val, +)
#undef STAMP_SUM

/* ── Sum reduction kernels (float types — pairwise summation) ─────── */

static void _kern_sum_NUMC_DTYPE_FLOAT32(const char *a, char *out, size_t n,
                                         intptr_t sa) {
  if (n == 0) {
    *(float *)out = 0;
    return;
  }
  if (sa == (intptr_t)sizeof(float)) {
    const float *pa = (const float *)a;
    size_t total_bytes = n * sizeof(float);
    if (total_bytes > NUMC_OMP_BYTE_THRESHOLD) {
      int nt = (int)(total_bytes / NUMC_OMP_BYTES_PER_THREAD);
      if (nt < 1) nt = 1;
      float global = 0;
      #pragma omp parallel for reduction(+:global) schedule(static) num_threads(nt)
      for (int t = 0; t < nt; t++) {
        size_t start = (size_t)t * (n / nt);
        size_t end = (t == nt - 1) ? n : start + n / nt;
        float local = _pairwise_sum_f32(pa + start, end - start);
        global += local;
      }
      *(float *)out = global;
    } else {
      *(float *)out = _pairwise_sum_f32(pa, n);
    }
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
    const double *pa = (const double *)a;
    size_t total_bytes = n * sizeof(double);
    if (total_bytes > NUMC_OMP_BYTE_THRESHOLD) {
      int nt = (int)(total_bytes / NUMC_OMP_BYTES_PER_THREAD);
      if (nt < 1) nt = 1;
      double global = 0;
      #pragma omp parallel for reduction(+:global) schedule(static) num_threads(nt)
      for (int t = 0; t < nt; t++) {
        size_t start = (size_t)t * (n / nt);
        size_t end = (t == nt - 1) ? n : start + n / nt;
        double local = _pairwise_sum_f64(pa + start, end - start);
        global += local;
      }
      *(double *)out = global;
    } else {
      *(double *)out = _pairwise_sum_f64(pa, n);
    }
  } else {
    double acc = 0;
    for (size_t i = 0; i < n; i++)
      acc += *(const double *)(a + i * sa);
    *(double *)out = acc;
  }
}

/* ── Mean reduction kernels ──────────────────────────────────────────
 *
 * mean = sum / count. Reuse existing sum kernels (including pairwise
 * summation for floats), then divide by n. Integer types get truncating
 * division — same overflow semantics as sum. */

/* int8/int16: promote through float for division */
#define STAMP_MEAN_SMALL(TE, CT)                                               \
  static void _kern_mean_##TE(const char *a, char *out, size_t n,              \
                              intptr_t sa) {                                   \
    _kern_sum_##TE(a, out, n, sa);                                             \
    if (n > 0)                                                                 \
      *(CT *)out = (CT)((float)*(CT *)out / (float)n);                         \
  }
GENERATE_INT8_INT16_NUMC_TYPES(STAMP_MEAN_SMALL)
#undef STAMP_MEAN_SMALL

/* int32/uint32/int64/uint64: promote through double for division */
#define STAMP_MEAN_DBL(TE, CT)                                                 \
  static void _kern_mean_##TE(const char *a, char *out, size_t n,              \
                              intptr_t sa) {                                   \
    _kern_sum_##TE(a, out, n, sa);                                             \
    if (n > 0)                                                                 \
      *(CT *)out = (CT)((double)*(CT *)out / (double)n);                       \
  }
GENERATE_INT32(STAMP_MEAN_DBL)
STAMP_MEAN_DBL(NUMC_DTYPE_INT64, int64_t)
STAMP_MEAN_DBL(NUMC_DTYPE_UINT64, uint64_t)
#undef STAMP_MEAN_DBL

/* float32/float64: native division */
#define STAMP_MEAN_FLOAT(TE, CT)                                               \
  static void _kern_mean_##TE(const char *a, char *out, size_t n,              \
                              intptr_t sa) {                                   \
    _kern_sum_##TE(a, out, n, sa);                                             \
    if (n > 0)                                                                 \
      *(CT *)out /= (CT)n;                                                     \
  }
STAMP_MEAN_FLOAT(NUMC_DTYPE_FLOAT32, float)
STAMP_MEAN_FLOAT(NUMC_DTYPE_FLOAT64, double)
#undef STAMP_MEAN_FLOAT

/* ── Max reduction kernels ───────────────────────────────────────────
 *
 * Per-type INIT = type minimum so any element is >= INIT.
 * EXPR = val > acc ? val : acc. OMP reduction(max:acc). */

#define MAX_EXPR val > acc ? val : acc

DEFINE_REDUCTION_KERNEL(max, NUMC_DTYPE_INT8, int8_t, INT8_MIN, MAX_EXPR, max)
DEFINE_REDUCTION_KERNEL(max, NUMC_DTYPE_INT16, int16_t, INT16_MIN, MAX_EXPR,
                        max)
DEFINE_REDUCTION_KERNEL(max, NUMC_DTYPE_INT32, int32_t, INT32_MIN, MAX_EXPR,
                        max)
DEFINE_REDUCTION_KERNEL(max, NUMC_DTYPE_INT64, int64_t, INT64_MIN, MAX_EXPR,
                        max)
DEFINE_REDUCTION_KERNEL(max, NUMC_DTYPE_UINT8, uint8_t, 0, MAX_EXPR, max)
DEFINE_REDUCTION_KERNEL(max, NUMC_DTYPE_UINT16, uint16_t, 0, MAX_EXPR, max)
DEFINE_REDUCTION_KERNEL(max, NUMC_DTYPE_UINT32, uint32_t, 0, MAX_EXPR, max)
DEFINE_REDUCTION_KERNEL(max, NUMC_DTYPE_UINT64, uint64_t, 0, MAX_EXPR, max)
/* float32/float64: multi-accumulator for contiguous (SLP-vectorizes to
 * vmaxps/vmaxpd), serial fallback for strided. */
static void _kern_max_NUMC_DTYPE_FLOAT32(const char *a, char *out, size_t n,
                                         intptr_t sa) {
  if (n == 0) {
    *(float *)out = -INFINITY;
    return;
  }
  if (sa == (intptr_t)sizeof(float)) {
    const float *pa = (const float *)a;
    size_t total_bytes = n * sizeof(float);
    if (total_bytes > NUMC_OMP_BYTE_THRESHOLD) {
      int nt = (int)(total_bytes / NUMC_OMP_BYTES_PER_THREAD);
      if (nt < 1) nt = 1;
      float global = -INFINITY;
      #pragma omp parallel for reduction(max:global) schedule(static) num_threads(nt)
      for (int t = 0; t < nt; t++) {
        size_t start = (size_t)t * (n / nt);
        size_t end = (t == nt - 1) ? n : start + n / nt;
        float local = _vec_max_f32(pa + start, end - start);
        if (local > global) global = local;
      }
      *(float *)out = global;
    } else {
      *(float *)out = _vec_max_f32(pa, n);
    }
  } else {
    float acc = -INFINITY;
    for (size_t i = 0; i < n; i++) {
      float val = *(const float *)(a + i * sa);
      acc = val > acc ? val : acc;
    }
    *(float *)out = acc;
  }
}
static void _kern_max_NUMC_DTYPE_FLOAT64(const char *a, char *out, size_t n,
                                         intptr_t sa) {
  if (n == 0) {
    *(double *)out = -INFINITY;
    return;
  }
  if (sa == (intptr_t)sizeof(double)) {
    const double *pa = (const double *)a;
    size_t total_bytes = n * sizeof(double);
    if (total_bytes > NUMC_OMP_BYTE_THRESHOLD) {
      int nt = (int)(total_bytes / NUMC_OMP_BYTES_PER_THREAD);
      if (nt < 1) nt = 1;
      double global = -INFINITY;
      #pragma omp parallel for reduction(max:global) schedule(static) num_threads(nt)
      for (int t = 0; t < nt; t++) {
        size_t start = (size_t)t * (n / nt);
        size_t end = (t == nt - 1) ? n : start + n / nt;
        double local = _vec_max_f64(pa + start, end - start);
        if (local > global) global = local;
      }
      *(double *)out = global;
    } else {
      *(double *)out = _vec_max_f64(pa, n);
    }
  } else {
    double acc = -INFINITY;
    for (size_t i = 0; i < n; i++) {
      double val = *(const double *)(a + i * sa);
      acc = val > acc ? val : acc;
    }
    *(double *)out = acc;
  }
}

#undef MAX_EXPR

/* ── Min reduction kernels ───────────────────────────────────────────
 *
 * Per-type INIT = type maximum so any element is <= INIT.
 * EXPR = val < acc ? val : acc. OMP reduction(min:acc). */

#define MIN_EXPR val < acc ? val : acc

DEFINE_REDUCTION_KERNEL(min, NUMC_DTYPE_INT8, int8_t, INT8_MAX, MIN_EXPR, min)
DEFINE_REDUCTION_KERNEL(min, NUMC_DTYPE_INT16, int16_t, INT16_MAX, MIN_EXPR,
                        min)
DEFINE_REDUCTION_KERNEL(min, NUMC_DTYPE_INT32, int32_t, INT32_MAX, MIN_EXPR,
                        min)
DEFINE_REDUCTION_KERNEL(min, NUMC_DTYPE_INT64, int64_t, INT64_MAX, MIN_EXPR,
                        min)
DEFINE_REDUCTION_KERNEL(min, NUMC_DTYPE_UINT8, uint8_t, UINT8_MAX, MIN_EXPR,
                        min)
DEFINE_REDUCTION_KERNEL(min, NUMC_DTYPE_UINT16, uint16_t, UINT16_MAX, MIN_EXPR,
                        min)
DEFINE_REDUCTION_KERNEL(min, NUMC_DTYPE_UINT32, uint32_t, UINT32_MAX, MIN_EXPR,
                        min)
DEFINE_REDUCTION_KERNEL(min, NUMC_DTYPE_UINT64, uint64_t, UINT64_MAX, MIN_EXPR,
                        min)
/* float32/float64: multi-accumulator for contiguous (SLP-vectorizes to
 * vminps/vminpd), serial fallback for strided. */
static void _kern_min_NUMC_DTYPE_FLOAT32(const char *a, char *out, size_t n,
                                         intptr_t sa) {
  if (n == 0) {
    *(float *)out = INFINITY;
    return;
  }
  if (sa == (intptr_t)sizeof(float)) {
    const float *pa = (const float *)a;
    size_t total_bytes = n * sizeof(float);
    if (total_bytes > NUMC_OMP_BYTE_THRESHOLD) {
      int nt = (int)(total_bytes / NUMC_OMP_BYTES_PER_THREAD);
      if (nt < 1) nt = 1;
      float global = INFINITY;
      #pragma omp parallel for reduction(min:global) schedule(static) num_threads(nt)
      for (int t = 0; t < nt; t++) {
        size_t start = (size_t)t * (n / nt);
        size_t end = (t == nt - 1) ? n : start + n / nt;
        float local = _vec_min_f32(pa + start, end - start);
        if (local < global) global = local;
      }
      *(float *)out = global;
    } else {
      *(float *)out = _vec_min_f32(pa, n);
    }
  } else {
    float acc = INFINITY;
    for (size_t i = 0; i < n; i++) {
      float val = *(const float *)(a + i * sa);
      acc = val < acc ? val : acc;
    }
    *(float *)out = acc;
  }
}
static void _kern_min_NUMC_DTYPE_FLOAT64(const char *a, char *out, size_t n,
                                         intptr_t sa) {
  if (n == 0) {
    *(double *)out = INFINITY;
    return;
  }
  if (sa == (intptr_t)sizeof(double)) {
    const double *pa = (const double *)a;
    size_t total_bytes = n * sizeof(double);
    if (total_bytes > NUMC_OMP_BYTE_THRESHOLD) {
      int nt = (int)(total_bytes / NUMC_OMP_BYTES_PER_THREAD);
      if (nt < 1) nt = 1;
      double global = INFINITY;
      #pragma omp parallel for reduction(min:global) schedule(static) num_threads(nt)
      for (int t = 0; t < nt; t++) {
        size_t start = (size_t)t * (n / nt);
        size_t end = (t == nt - 1) ? n : start + n / nt;
        double local = _vec_min_f64(pa + start, end - start);
        if (local < global) global = local;
      }
      *(double *)out = global;
    } else {
      *(double *)out = _vec_min_f64(pa, n);
    }
  } else {
    double acc = INFINITY;
    for (size_t i = 0; i < n; i++) {
      double val = *(const double *)(a + i * sa);
      acc = val < acc ? val : acc;
    }
    *(double *)out = acc;
  }
}

#undef MIN_EXPR

/* ── Dispatch tables ─────────────────────────────────────────────── */

#define R(OP, TE) [TE] = _kern_##OP##_##TE

static const NumcReductionKernel _sum_table[] = {
    R(sum, NUMC_DTYPE_INT8),    R(sum, NUMC_DTYPE_INT16),
    R(sum, NUMC_DTYPE_INT32),   R(sum, NUMC_DTYPE_INT64),
    R(sum, NUMC_DTYPE_UINT8),   R(sum, NUMC_DTYPE_UINT16),
    R(sum, NUMC_DTYPE_UINT32),  R(sum, NUMC_DTYPE_UINT64),
    R(sum, NUMC_DTYPE_FLOAT32), R(sum, NUMC_DTYPE_FLOAT64),
};

static const NumcReductionKernel _mean_table[] = {
    R(mean, NUMC_DTYPE_INT8),    R(mean, NUMC_DTYPE_INT16),
    R(mean, NUMC_DTYPE_INT32),   R(mean, NUMC_DTYPE_INT64),
    R(mean, NUMC_DTYPE_UINT8),   R(mean, NUMC_DTYPE_UINT16),
    R(mean, NUMC_DTYPE_UINT32),  R(mean, NUMC_DTYPE_UINT64),
    R(mean, NUMC_DTYPE_FLOAT32), R(mean, NUMC_DTYPE_FLOAT64),
};

static const NumcReductionKernel _max_table[] = {
    R(max, NUMC_DTYPE_INT8),    R(max, NUMC_DTYPE_INT16),
    R(max, NUMC_DTYPE_INT32),   R(max, NUMC_DTYPE_INT64),
    R(max, NUMC_DTYPE_UINT8),   R(max, NUMC_DTYPE_UINT16),
    R(max, NUMC_DTYPE_UINT32),  R(max, NUMC_DTYPE_UINT64),
    R(max, NUMC_DTYPE_FLOAT32), R(max, NUMC_DTYPE_FLOAT64),
};

static const NumcReductionKernel _min_table[] = {
    R(min, NUMC_DTYPE_INT8),    R(min, NUMC_DTYPE_INT16),
    R(min, NUMC_DTYPE_INT32),   R(min, NUMC_DTYPE_INT64),
    R(min, NUMC_DTYPE_UINT8),   R(min, NUMC_DTYPE_UINT16),
    R(min, NUMC_DTYPE_UINT32),  R(min, NUMC_DTYPE_UINT64),
    R(min, NUMC_DTYPE_FLOAT32), R(min, NUMC_DTYPE_FLOAT64),
};

#undef R

/* ── Fused row-reduce kernels for axis fast path ─────────────────
 *
 * Process all rows in a single call, eliminating per-row function
 * pointer overhead. Compiler sees the full nested loop, enabling
 * software pipelining and load/compute overlap.
 *
 * Each d[i] is independent across columns, so the inner loop
 * auto-vectorizes (vpaddd/vaddps for sum, vpmaxsb/vmaxps for max, etc).
 * Uses unconditional ternary for max/min (not conditional if-store)
 * because AVX2 has no byte-level masked store (vpmaskmovb). */

typedef void (*NumcRowReduceKernel)(const char *restrict base,
                                    intptr_t row_stride, size_t nrows,
                                    char *restrict dst, size_t ncols);

/* Sum: d[i] += s[r][i] */
#define STAMP_SUM_FUSED(TE, CT)                                                \
  static void _sum_fused_##TE(const char *restrict base, intptr_t row_stride,  \
                              size_t nrows, char *restrict dst,                \
                              size_t ncols) {                                  \
    CT *restrict d = (CT *)dst;                                                \
    for (size_t r = 0; r < nrows; r++) {                                       \
      const CT *restrict s = (const CT *)(base + r * row_stride);              \
      for (size_t i = 0; i < ncols; i++)                                       \
        d[i] += s[i];                                                          \
    }                                                                          \
  }
GENERATE_NUMC_TYPES(STAMP_SUM_FUSED)
#undef STAMP_SUM_FUSED

/* Max: d[i] = max(d[i], s[r][i]) */
#define STAMP_MAX_FUSED(TE, CT)                                                \
  static void _max_fused_##TE(const char *restrict base, intptr_t row_stride,  \
                              size_t nrows, char *restrict dst,                \
                              size_t ncols) {                                  \
    CT *restrict d = (CT *)dst;                                                \
    for (size_t r = 0; r < nrows; r++) {                                       \
      const CT *restrict s = (const CT *)(base + r * row_stride);              \
      for (size_t i = 0; i < ncols; i++)                                       \
        d[i] = s[i] > d[i] ? s[i] : d[i];                                      \
    }                                                                          \
  }
GENERATE_NUMC_TYPES(STAMP_MAX_FUSED)
#undef STAMP_MAX_FUSED

/* Min: d[i] = min(d[i], s[r][i]) */
#define STAMP_MIN_FUSED(TE, CT)                                                \
  static void _min_fused_##TE(const char *restrict base, intptr_t row_stride,  \
                              size_t nrows, char *restrict dst,                \
                              size_t ncols) {                                  \
    CT *restrict d = (CT *)dst;                                                \
    for (size_t r = 0; r < nrows; r++) {                                       \
      const CT *restrict s = (const CT *)(base + r * row_stride);              \
      for (size_t i = 0; i < ncols; i++)                                       \
        d[i] = s[i] < d[i] ? s[i] : d[i];                                      \
    }                                                                          \
  }
GENERATE_NUMC_TYPES(STAMP_MIN_FUSED)
#undef STAMP_MIN_FUSED

#define F(OP, TE) [TE] = _##OP##_fused_##TE
static const NumcRowReduceKernel _sum_fused_table[] = {
    F(sum, NUMC_DTYPE_INT8),    F(sum, NUMC_DTYPE_INT16),
    F(sum, NUMC_DTYPE_INT32),   F(sum, NUMC_DTYPE_INT64),
    F(sum, NUMC_DTYPE_UINT8),   F(sum, NUMC_DTYPE_UINT16),
    F(sum, NUMC_DTYPE_UINT32),  F(sum, NUMC_DTYPE_UINT64),
    F(sum, NUMC_DTYPE_FLOAT32), F(sum, NUMC_DTYPE_FLOAT64),
};
static const NumcRowReduceKernel _max_fused_table[] = {
    F(max, NUMC_DTYPE_INT8),    F(max, NUMC_DTYPE_INT16),
    F(max, NUMC_DTYPE_INT32),   F(max, NUMC_DTYPE_INT64),
    F(max, NUMC_DTYPE_UINT8),   F(max, NUMC_DTYPE_UINT16),
    F(max, NUMC_DTYPE_UINT32),  F(max, NUMC_DTYPE_UINT64),
    F(max, NUMC_DTYPE_FLOAT32), F(max, NUMC_DTYPE_FLOAT64),
};
static const NumcRowReduceKernel _min_fused_table[] = {
    F(min, NUMC_DTYPE_INT8),    F(min, NUMC_DTYPE_INT16),
    F(min, NUMC_DTYPE_INT32),   F(min, NUMC_DTYPE_INT64),
    F(min, NUMC_DTYPE_UINT8),   F(min, NUMC_DTYPE_UINT16),
    F(min, NUMC_DTYPE_UINT32),  F(min, NUMC_DTYPE_UINT64),
    F(min, NUMC_DTYPE_FLOAT32), F(min, NUMC_DTYPE_FLOAT64),
};
#undef F

/* ── Divide-by-count kernels for axis mean fast path ─────────────
 *
 * d[i] /= count — element-wise divide of output buffer.
 * Each d[i] is independent, so this auto-vectorizes. */

typedef void (*NumcDivCountKernel)(char *data, size_t n, size_t count);

/* int8/int16: promote through float (vdivps — 8 lanes) */
#define STAMP_DIV_COUNT_SMALL(TE, CT)                                          \
  static void _div_count_##TE(char *data, size_t n, size_t count) {            \
    CT *d = (CT *)data;                                                        \
    float fc = (float)count;                                                   \
    for (size_t i = 0; i < n; i++)                                             \
      d[i] = (CT)((float)d[i] / fc);                                           \
  }
GENERATE_INT8_INT16_NUMC_TYPES(STAMP_DIV_COUNT_SMALL)
#undef STAMP_DIV_COUNT_SMALL

/* int32/uint32/int64/uint64: promote through double (vdivpd — 4 lanes).
 * int64/uint64 lose precision above 2^53, acceptable for mean. */
#define STAMP_DIV_COUNT_DBL(TE, CT)                                            \
  static void _div_count_##TE(char *data, size_t n, size_t count) {            \
    CT *d = (CT *)data;                                                        \
    double dc = (double)count;                                                 \
    for (size_t i = 0; i < n; i++)                                             \
      d[i] = (CT)((double)d[i] / dc);                                          \
  }
GENERATE_INT32(STAMP_DIV_COUNT_DBL)
STAMP_DIV_COUNT_DBL(NUMC_DTYPE_INT64, int64_t)
STAMP_DIV_COUNT_DBL(NUMC_DTYPE_UINT64, uint64_t)
#undef STAMP_DIV_COUNT_DBL

/* float32/float64: native division */
static void _div_count_NUMC_DTYPE_FLOAT32(char *data, size_t n, size_t count) {
  float *d = (float *)data;
  float cc = (float)count;
  for (size_t i = 0; i < n; i++)
    d[i] /= cc;
}
static void _div_count_NUMC_DTYPE_FLOAT64(char *data, size_t n, size_t count) {
  double *d = (double *)data;
  double dc = (double)count;
  for (size_t i = 0; i < n; i++)
    d[i] /= dc;
}

#define D(TE) [TE] = _div_count_##TE
static const NumcDivCountKernel _div_count_table[] = {
    D(NUMC_DTYPE_INT8),    D(NUMC_DTYPE_INT16),  D(NUMC_DTYPE_INT32),
    D(NUMC_DTYPE_INT64),   D(NUMC_DTYPE_UINT8),  D(NUMC_DTYPE_UINT16),
    D(NUMC_DTYPE_UINT32),  D(NUMC_DTYPE_UINT64), D(NUMC_DTYPE_FLOAT32),
    D(NUMC_DTYPE_FLOAT64),
};
#undef D

/* ── Public API ──────────────────────────────────────────────────── */

int numc_sum(const NumcArray *a, NumcArray *out) {
  int err = _check_reduce_full(a, out);
  if (err)
    return err;
  _reduce_full_op(a, out, _sum_table);
  return 0;
}

int numc_mean(const NumcArray *a, NumcArray *out) {
  int err = _check_reduce_full(a, out);
  if (err)
    return err;
  _reduce_full_op(a, out, _mean_table);
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
    _sum_fused_table[a->dtype]((const char *)a->data, reduce_stride, reduce_len,
                               (char *)out->data, slice_elems);
    _div_count_table[a->dtype]((char *)out->data, slice_elems, reduce_len);
    return 0;
  }

  /* Generic path: per-element reduction via ND iterator */
  _reduce_axis_op(a, ax, keepdim, out, _mean_table);
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
    _sum_fused_table[a->dtype]((const char *)a->data, reduce_stride, reduce_len,
                               (char *)out->data, slice_elems);
    return 0;
  }

  /* Generic path: per-element reduction via ND iterator */
  _reduce_axis_op(a, ax, keepdim, out, _sum_table);
  return 0;
}

int numc_max(const NumcArray *a, NumcArray *out) {
  int err = _check_reduce_full(a, out);
  if (err)
    return err;
  _reduce_full_op(a, out, _max_table);
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
    _max_fused_table[a->dtype](base + reduce_stride, reduce_stride,
                               reduce_len - 1, (char *)out->data, slice_elems);
    return 0;
  }

  /* Generic path: per-element reduction via ND iterator */
  _reduce_axis_op(a, ax, keepdim, out, _max_table);
  return 0;
}

int numc_min(const NumcArray *a, NumcArray *out) {
  int err = _check_reduce_full(a, out);
  if (err)
    return err;
  _reduce_full_op(a, out, _min_table);
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
    _min_fused_table[a->dtype](base + reduce_stride, reduce_stride,
                               reduce_len - 1, (char *)out->data, slice_elems);
    return 0;
  }

  /* Generic path: per-element reduction via ND iterator */
  _reduce_axis_op(a, ax, keepdim, out, _min_table);
  return 0;
}
