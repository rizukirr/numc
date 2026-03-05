#include "dispatch.h"
#include "helpers.h"
#include <math.h>
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

/* ── Max reduction kernels ───────────────────────────────────────────
 *
 * Per-type INIT = type minimum so any element is >= INIT.
 * EXPR = val > acc ? val : acc. OMP reduction(max:acc). */

#define MAX_EXPR val > acc ? val : acc

DEFINE_REDUCTION_KERNEL(max, NUMC_DTYPE_INT8, NUMC_INT8, INT8_MIN, MAX_EXPR,
                        max)
DEFINE_REDUCTION_KERNEL(max, NUMC_DTYPE_INT16, NUMC_INT16, INT16_MIN, MAX_EXPR,
                        max)
DEFINE_REDUCTION_KERNEL(max, NUMC_DTYPE_INT32, NUMC_INT32, INT32_MIN, MAX_EXPR,
                        max)
DEFINE_REDUCTION_KERNEL(max, NUMC_DTYPE_INT64, NUMC_INT64, INT64_MIN, MAX_EXPR,
                        max)
DEFINE_REDUCTION_KERNEL(max, NUMC_DTYPE_UINT8, NUMC_UINT8, 0, MAX_EXPR, max)
DEFINE_REDUCTION_KERNEL(max, NUMC_DTYPE_UINT16, NUMC_UINT16, 0, MAX_EXPR, max)
DEFINE_REDUCTION_KERNEL(max, NUMC_DTYPE_UINT32, NUMC_UINT32, 0, MAX_EXPR, max)
DEFINE_REDUCTION_KERNEL(max, NUMC_DTYPE_UINT64, NUMC_UINT64, 0, MAX_EXPR, max)
/* float32/float64: multi-accumulator for contiguous (SLP-vectorizes to
 * vmaxps/vmaxpd), serial fallback for strided. */
DEFINE_FLOAT_REDUCTION_KERNEL(max, NUMC_DTYPE_FLOAT32, NUMC_FLOAT32, -INFINITY,
                              _vec_max_f32, max,
                              if (local > global) global = local,
                              val > acc ? val : acc)
DEFINE_FLOAT_REDUCTION_KERNEL(max, NUMC_DTYPE_FLOAT64, NUMC_FLOAT64, -INFINITY,
                              _vec_max_f64, max,
                              if (local > global) global = local,
                              val > acc ? val : acc)

#undef MAX_EXPR

/* ── Min reduction kernels ───────────────────────────────────────────
 *
 * Per-type INIT = type maximum so any element is <= INIT.
 * EXPR = val < acc ? val : acc. OMP reduction(min:acc). */

#define MIN_EXPR val < acc ? val : acc

DEFINE_REDUCTION_KERNEL(min, NUMC_DTYPE_INT8, NUMC_INT8, INT8_MAX, MIN_EXPR,
                        min)
DEFINE_REDUCTION_KERNEL(min, NUMC_DTYPE_INT16, NUMC_INT16, INT16_MAX, MIN_EXPR,
                        min)
DEFINE_REDUCTION_KERNEL(min, NUMC_DTYPE_INT32, NUMC_INT32, INT32_MAX, MIN_EXPR,
                        min)
DEFINE_REDUCTION_KERNEL(min, NUMC_DTYPE_INT64, NUMC_INT64, INT64_MAX, MIN_EXPR,
                        min)
DEFINE_REDUCTION_KERNEL(min, NUMC_DTYPE_UINT8, NUMC_UINT8, UINT8_MAX, MIN_EXPR,
                        min)
DEFINE_REDUCTION_KERNEL(min, NUMC_DTYPE_UINT16, NUMC_UINT16, UINT16_MAX,
                        MIN_EXPR, min)
DEFINE_REDUCTION_KERNEL(min, NUMC_DTYPE_UINT32, NUMC_UINT32, UINT32_MAX,
                        MIN_EXPR, min)
DEFINE_REDUCTION_KERNEL(min, NUMC_DTYPE_UINT64, NUMC_UINT64, UINT64_MAX,
                        MIN_EXPR, min)
/* float32/float64: multi-accumulator for contiguous (SLP-vectorizes to
 * vminps/vminpd), serial fallback for strided. */
DEFINE_FLOAT_REDUCTION_KERNEL(min, NUMC_DTYPE_FLOAT32, NUMC_FLOAT32, INFINITY,
                              _vec_min_f32, min,
                              if (local < global) global = local,
                              val < acc ? val : acc)
DEFINE_FLOAT_REDUCTION_KERNEL(min, NUMC_DTYPE_FLOAT64, NUMC_FLOAT64, INFINITY,
                              _vec_min_f64, min,
                              if (local < global) global = local,
                              val < acc ? val : acc)

#undef MIN_EXPR

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

/* ── Argmin reduction kernels ────────────────────────────────────────
 *
 * Per-type INIT = type maximum so any element is < INIT.
 * Output is always int64_t (index of minimum element). */

DEFINE_ARGREDUCTION_KERNEL(argmin, NUMC_DTYPE_INT8, NUMC_INT8, INT8_MAX, <)
DEFINE_ARGREDUCTION_KERNEL(argmin, NUMC_DTYPE_INT16, NUMC_INT16, INT16_MAX, <)
DEFINE_ARGREDUCTION_KERNEL(argmin, NUMC_DTYPE_INT32, NUMC_INT32, INT32_MAX, <)
DEFINE_ARGREDUCTION_KERNEL(argmin, NUMC_DTYPE_INT64, NUMC_INT64, INT64_MAX, <)
DEFINE_ARGREDUCTION_KERNEL(argmin, NUMC_DTYPE_UINT8, NUMC_UINT8, UINT8_MAX, <)
DEFINE_ARGREDUCTION_KERNEL(argmin, NUMC_DTYPE_UINT16, NUMC_UINT16, UINT16_MAX,
                           <)
DEFINE_ARGREDUCTION_KERNEL(argmin, NUMC_DTYPE_UINT32, NUMC_UINT32, UINT32_MAX,
                           <)
DEFINE_ARGREDUCTION_KERNEL(argmin, NUMC_DTYPE_UINT64, NUMC_UINT64, UINT64_MAX,
                           <)
DEFINE_FLOAT_ARGREDUCTION_KERNEL(argmin, NUMC_DTYPE_FLOAT32, NUMC_FLOAT32,
                                 INFINITY, _vec_min_f32, <)
DEFINE_FLOAT_ARGREDUCTION_KERNEL(argmin, NUMC_DTYPE_FLOAT64, NUMC_FLOAT64,
                                 INFINITY, _vec_min_f64, <)

/* ── Dot product reduction kernels ─────────────────────────────────── */

#define STAMP_DOT(TE, CT) \
  DEFINE_BINARY_REDUCTION_KERNEL(dot, TE, CT, 0, acc + (val_a * val_b), +)
GENERATE_INT8_INT16_NUMC_TYPES(STAMP_DOT)
GENERATE_INT32_NUMC_TYPES(STAMP_DOT)
STAMP_DOT(NUMC_DTYPE_INT64, NUMC_INT64)
STAMP_DOT(NUMC_DTYPE_UINT64, NUMC_UINT64)
STAMP_DOT(NUMC_DTYPE_FLOAT32, NUMC_FLOAT32)
STAMP_DOT(NUMC_DTYPE_FLOAT64, NUMC_FLOAT64)
#undef STAMP_DOT

/* ── Dispatch tables ─────────────────────────────────────────────── */

#define R(OP, TE) [TE] = _kern_##OP##_##TE

static const NumcReductionKernel sum_table[] = {
    R(sum, NUMC_DTYPE_INT8),    R(sum, NUMC_DTYPE_INT16),
    R(sum, NUMC_DTYPE_INT32),   R(sum, NUMC_DTYPE_INT64),
    R(sum, NUMC_DTYPE_UINT8),   R(sum, NUMC_DTYPE_UINT16),
    R(sum, NUMC_DTYPE_UINT32),  R(sum, NUMC_DTYPE_UINT64),
    R(sum, NUMC_DTYPE_FLOAT32), R(sum, NUMC_DTYPE_FLOAT64),
};

static const NumcReductionKernel mean_table[] = {
    R(mean, NUMC_DTYPE_INT8),    R(mean, NUMC_DTYPE_INT16),
    R(mean, NUMC_DTYPE_INT32),   R(mean, NUMC_DTYPE_INT64),
    R(mean, NUMC_DTYPE_UINT8),   R(mean, NUMC_DTYPE_UINT16),
    R(mean, NUMC_DTYPE_UINT32),  R(mean, NUMC_DTYPE_UINT64),
    R(mean, NUMC_DTYPE_FLOAT32), R(mean, NUMC_DTYPE_FLOAT64),
};

static const NumcReductionKernel max_table[] = {
    R(max, NUMC_DTYPE_INT8),    R(max, NUMC_DTYPE_INT16),
    R(max, NUMC_DTYPE_INT32),   R(max, NUMC_DTYPE_INT64),
    R(max, NUMC_DTYPE_UINT8),   R(max, NUMC_DTYPE_UINT16),
    R(max, NUMC_DTYPE_UINT32),  R(max, NUMC_DTYPE_UINT64),
    R(max, NUMC_DTYPE_FLOAT32), R(max, NUMC_DTYPE_FLOAT64),
};

static const NumcReductionKernel min_table[] = {
    R(min, NUMC_DTYPE_INT8),    R(min, NUMC_DTYPE_INT16),
    R(min, NUMC_DTYPE_INT32),   R(min, NUMC_DTYPE_INT64),
    R(min, NUMC_DTYPE_UINT8),   R(min, NUMC_DTYPE_UINT16),
    R(min, NUMC_DTYPE_UINT32),  R(min, NUMC_DTYPE_UINT64),
    R(min, NUMC_DTYPE_FLOAT32), R(min, NUMC_DTYPE_FLOAT64),
};

static const NumcReductionKernel argmax_table[] = {
    R(argmax, NUMC_DTYPE_INT8),    R(argmax, NUMC_DTYPE_INT16),
    R(argmax, NUMC_DTYPE_INT32),   R(argmax, NUMC_DTYPE_INT64),
    R(argmax, NUMC_DTYPE_UINT8),   R(argmax, NUMC_DTYPE_UINT16),
    R(argmax, NUMC_DTYPE_UINT32),  R(argmax, NUMC_DTYPE_UINT64),
    R(argmax, NUMC_DTYPE_FLOAT32), R(argmax, NUMC_DTYPE_FLOAT64),
};

static const NumcReductionKernel argmin_table[] = {
    R(argmin, NUMC_DTYPE_INT8),    R(argmin, NUMC_DTYPE_INT16),
    R(argmin, NUMC_DTYPE_INT32),   R(argmin, NUMC_DTYPE_INT64),
    R(argmin, NUMC_DTYPE_UINT8),   R(argmin, NUMC_DTYPE_UINT16),
    R(argmin, NUMC_DTYPE_UINT32),  R(argmin, NUMC_DTYPE_UINT64),
    R(argmin, NUMC_DTYPE_FLOAT32), R(argmin, NUMC_DTYPE_FLOAT64),
};

static const NumcBinaryReductionKernel dot_table[] = {
    R(dot, NUMC_DTYPE_INT8),    R(dot, NUMC_DTYPE_INT16),
    R(dot, NUMC_DTYPE_INT32),   R(dot, NUMC_DTYPE_INT64),
    R(dot, NUMC_DTYPE_UINT8),   R(dot, NUMC_DTYPE_UINT16),
    R(dot, NUMC_DTYPE_UINT32),  R(dot, NUMC_DTYPE_UINT64),
    R(dot, NUMC_DTYPE_FLOAT32), R(dot, NUMC_DTYPE_FLOAT64),
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

/* Max: d[i] = max(d[i], s[r][i]) */
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

/* Min: d[i] = min(d[i], s[r][i]) */
#define STAMP_MIN_FUSED(TE, CT)                                               \
  static void _min_fused_##TE(const char *restrict base, intptr_t row_stride, \
                              size_t nrows, char *restrict dst,               \
                              size_t ncols) {                                 \
    CT *restrict d = (CT *)dst;                                               \
    for (size_t r = 0; r < nrows; r++) {                                      \
      const CT *restrict s = (const CT *)(base + r * row_stride);             \
      for (size_t i = 0; i < ncols; i++)                                      \
        d[i] = s[i] < d[i] ? s[i] : d[i];                                     \
    }                                                                         \
  }
GENERATE_NUMC_TYPES(STAMP_MIN_FUSED)
#undef STAMP_MIN_FUSED

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

#define STAMP_ARGMIN_FUSED(TE, CT)                                   \
  static void _argmin_fused_##TE(const char *restrict base,          \
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
        if (row[i] < best[i]) {                                      \
          best[i] = row[i];                                          \
          idx[i] = (int64_t)r;                                       \
        }                                                            \
      }                                                              \
    }                                                                \
  }
GENERATE_NUMC_TYPES(STAMP_ARGMIN_FUSED)
#undef STAMP_ARGMIN_FUSED

#define F(OP, TE) [TE] = _##OP##_fused_##TE
static const NumcArgRowReduceKernel argmax_fused_table[] = {
    F(argmax, NUMC_DTYPE_INT8),    F(argmax, NUMC_DTYPE_INT16),
    F(argmax, NUMC_DTYPE_INT32),   F(argmax, NUMC_DTYPE_INT64),
    F(argmax, NUMC_DTYPE_UINT8),   F(argmax, NUMC_DTYPE_UINT16),
    F(argmax, NUMC_DTYPE_UINT32),  F(argmax, NUMC_DTYPE_UINT64),
    F(argmax, NUMC_DTYPE_FLOAT32), F(argmax, NUMC_DTYPE_FLOAT64),
};
static const NumcArgRowReduceKernel argmin_fused_table[] = {
    F(argmin, NUMC_DTYPE_INT8),    F(argmin, NUMC_DTYPE_INT16),
    F(argmin, NUMC_DTYPE_INT32),   F(argmin, NUMC_DTYPE_INT64),
    F(argmin, NUMC_DTYPE_UINT8),   F(argmin, NUMC_DTYPE_UINT16),
    F(argmin, NUMC_DTYPE_UINT32),  F(argmin, NUMC_DTYPE_UINT64),
    F(argmin, NUMC_DTYPE_FLOAT32), F(argmin, NUMC_DTYPE_FLOAT64),
};
#undef F

#define F(OP, TE) [TE] = _##OP##_fused_##TE
static const NumcRowReduceKernel sum_fused_table[] = {
    F(sum, NUMC_DTYPE_INT8),    F(sum, NUMC_DTYPE_INT16),
    F(sum, NUMC_DTYPE_INT32),   F(sum, NUMC_DTYPE_INT64),
    F(sum, NUMC_DTYPE_UINT8),   F(sum, NUMC_DTYPE_UINT16),
    F(sum, NUMC_DTYPE_UINT32),  F(sum, NUMC_DTYPE_UINT64),
    F(sum, NUMC_DTYPE_FLOAT32), F(sum, NUMC_DTYPE_FLOAT64),
};
static const NumcRowReduceKernel max_fused_table[] = {
    F(max, NUMC_DTYPE_INT8),    F(max, NUMC_DTYPE_INT16),
    F(max, NUMC_DTYPE_INT32),   F(max, NUMC_DTYPE_INT64),
    F(max, NUMC_DTYPE_UINT8),   F(max, NUMC_DTYPE_UINT16),
    F(max, NUMC_DTYPE_UINT32),  F(max, NUMC_DTYPE_UINT64),
    F(max, NUMC_DTYPE_FLOAT32), F(max, NUMC_DTYPE_FLOAT64),
};
static const NumcRowReduceKernel min_fused_table[] = {
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
/**
 * @brief Divide float32 array elements by a count (for mean calculation).
 *
 * @param data  Pointer to the data buffer.
 * @param n     Number of elements.
 * @param count Value to divide by.
 */
static void _div_count_NUMC_DTYPE_FLOAT32(char *data, size_t n, size_t count) {
  NUMC_FLOAT32 *d = (NUMC_FLOAT32 *)data;
  NUMC_FLOAT32 cc = (NUMC_FLOAT32)count;
  for (size_t i = 0; i < n; i++)
    d[i] /= cc;
}

/**
 * @brief Divide float64 array elements by a count (for mean calculation).
 *
 * @param data  Pointer to the data buffer.
 * @param n     Number of elements.
 * @param count Value to divide by.
 */
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

int numc_sum(const NumcArray *a, NumcArray *out) {
  int err = _check_reduce_full(a, out);
  if (err)
    return err;
  _reduce_full_op(a, out, sum_table);
  return 0;
}

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

int numc_argmin(const NumcArray *a, NumcArray *out) {
  int err = _check_argreduce_full(a, out);
  if (err)
    return err;
  _reduce_full_op(a, out, argmin_table);
  return 0;
}

int numc_argmin_axis(const NumcArray *a, int axis, int keepdim,
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

    argmin_fused_table[a->dtype]((const char *)a->data, reduce_stride,
                                 reduce_len, (char *)out->data, slice_elems);
    return 0;
  }

  /* Generic path: per-element reduction via ND iterator */
  _reduce_axis_op(a, ax, keepdim, out, argmin_table);
  return 0;
}

#ifdef HAVE_BLAS
#include <blis.h>
#include <pthread.h>
#endif

#ifdef HAVE_BLAS
static void _dot_blis_f32(const NumcArray *a, const NumcArray *b,
                          NumcArray *out) {
  /* Only for 1D-1D vector dot product */
  float rho = 0.0f;
  inc_t inca = (inc_t)(a->strides[0] / sizeof(float));
  inc_t incb = (inc_t)(b->strides[0] / sizeof(float));
  bli_sdotv(BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE, (dim_t)a->size,
            (float *)a->data, inca, (float *)b->data, incb, &rho);
  *(float *)out->data = rho;
}

static void _dot_blis_f64(const NumcArray *a, const NumcArray *b,
                          NumcArray *out) {
  /* Only for 1D-1D vector dot product */
  double rho = 0.0;
  inc_t inca = (inc_t)(a->strides[0] / sizeof(double));
  inc_t incb = (inc_t)(b->strides[0] / sizeof(double));
  bli_ddotv(BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE, (dim_t)a->size,
            (double *)a->data, inca, (double *)b->data, incb, &rho);
  *(double *)out->data = rho;
}

static void _dot_blis_gemm_f32(const NumcArray *a, const NumcArray *b,
                               NumcArray *out, size_t M, size_t K, size_t N) {
  float alpha = 1.0f, beta = 0.0f;

  /* Strides for reshaped matrices (M, K) @ (K, N) -> (M, N) */
  inc_t rs_a =
      (inc_t)(a->dim > 1 ? a->strides[a->dim - 2] / sizeof(float) : (inc_t)K);
  inc_t cs_a = (inc_t)(a->strides[a->dim - 1] / sizeof(float));

  inc_t rs_b =
      (inc_t)(b->dim > 1 ? b->strides[b->dim - 2] / sizeof(float) : (inc_t)N);
  inc_t cs_b = (inc_t)(b->strides[b->dim - 1] / sizeof(float));

  inc_t rs_c = (inc_t)(out->dim > 1 ? out->strides[out->dim - 2] / sizeof(float)
                                    : (inc_t)N);
  inc_t cs_c = (inc_t)(out->strides[out->dim - 1] / sizeof(float));

  bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, (dim_t)M, (dim_t)N, (dim_t)K,
            &alpha, (float *)a->data, rs_a, cs_a, (float *)b->data, rs_b, cs_b,
            &beta, (float *)out->data, rs_c, cs_c);
}

static void _dot_blis_gemm_f64(const NumcArray *a, const NumcArray *b,
                               NumcArray *out, size_t M, size_t K, size_t N) {
  double alpha = 1.0, beta = 0.0;

  inc_t rs_a =
      (inc_t)(a->dim > 1 ? a->strides[a->dim - 2] / sizeof(double) : (inc_t)K);
  inc_t cs_a = (inc_t)(a->strides[a->dim - 1] / sizeof(double));

  inc_t rs_b =
      (inc_t)(b->dim > 1 ? b->strides[b->dim - 2] / sizeof(double) : (inc_t)N);
  inc_t cs_b = (inc_t)(b->strides[b->dim - 1] / sizeof(double));

  inc_t rs_c =
      (inc_t)(out->dim > 1 ? out->strides[out->dim - 2] / sizeof(double)
                           : (inc_t)N);
  inc_t cs_c = (inc_t)(out->strides[out->dim - 1] / sizeof(double));

  bli_dgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, (dim_t)M, (dim_t)N, (dim_t)K,
            &alpha, (double *)a->data, rs_a, cs_a, (double *)b->data, rs_b,
            cs_b, &beta, (double *)out->data, rs_c, cs_c);
}
#endif

/* ── Dot product helper kernels ────────────────────────────────── */

#define STAMP_DOT_NAIVE(TE, CT, ACC_CT)                                        \
  static void _dot_naive_##TE(const char *pa, const char *pb, char *po,        \
                              size_t M, size_t K, size_t N, intptr_t rsa,      \
                              intptr_t csa, intptr_t rsb, intptr_t csb,        \
                              intptr_t rso, intptr_t cso) {                    \
    const CT *a = (const CT *)pa;                                              \
    const CT *b = (const CT *)pb;                                              \
    CT *out = (CT *)po;                                                        \
    NUMC_OMP_FOR(                                                              \
        M * N, sizeof(CT), for (size_t i = 0; i < M; i++) {                    \
          for (size_t j = 0; j < N; j++) {                                     \
            ACC_CT acc = 0;                                                    \
            for (size_t k = 0; k < K; k++) {                                   \
              acc +=                                                           \
                  (ACC_CT)a[i * rsa + k * csa] * (ACC_CT)b[k * rsb + j * csb]; \
            }                                                                  \
            out[i * rso + j * cso] = (CT)acc;                                  \
          }                                                                    \
        });                                                                    \
  }

STAMP_DOT_NAIVE(NUMC_DTYPE_INT8, int8_t, int32_t)
STAMP_DOT_NAIVE(NUMC_DTYPE_INT16, int16_t, int64_t)
STAMP_DOT_NAIVE(NUMC_DTYPE_INT32, int32_t, int64_t)
STAMP_DOT_NAIVE(NUMC_DTYPE_INT64, int64_t, int64_t)
STAMP_DOT_NAIVE(NUMC_DTYPE_UINT8, uint8_t, uint32_t)
STAMP_DOT_NAIVE(NUMC_DTYPE_UINT16, uint16_t, uint64_t)
STAMP_DOT_NAIVE(NUMC_DTYPE_UINT32, uint32_t, uint64_t)
STAMP_DOT_NAIVE(NUMC_DTYPE_UINT64, uint64_t, uint64_t)
STAMP_DOT_NAIVE(NUMC_DTYPE_FLOAT32, float, float)
STAMP_DOT_NAIVE(NUMC_DTYPE_FLOAT64, double, double)

typedef void (*DotNaiveKernel)(const char *pa, const char *pb, char *po,
                               size_t M, size_t K, size_t N, intptr_t rsa,
                               intptr_t csa, intptr_t rsb, intptr_t csb,
                               intptr_t rso, intptr_t cso);

static const DotNaiveKernel dot_naive_table[] = {
    [NUMC_DTYPE_INT8] = _dot_naive_NUMC_DTYPE_INT8,
    [NUMC_DTYPE_INT16] = _dot_naive_NUMC_DTYPE_INT16,
    [NUMC_DTYPE_INT32] = _dot_naive_NUMC_DTYPE_INT32,
    [NUMC_DTYPE_INT64] = _dot_naive_NUMC_DTYPE_INT64,
    [NUMC_DTYPE_UINT8] = _dot_naive_NUMC_DTYPE_UINT8,
    [NUMC_DTYPE_UINT16] = _dot_naive_NUMC_DTYPE_UINT16,
    [NUMC_DTYPE_UINT32] = _dot_naive_NUMC_DTYPE_UINT32,
    [NUMC_DTYPE_UINT64] = _dot_naive_NUMC_DTYPE_UINT64,
    [NUMC_DTYPE_FLOAT32] = _dot_naive_NUMC_DTYPE_FLOAT32,
    [NUMC_DTYPE_FLOAT64] = _dot_naive_NUMC_DTYPE_FLOAT64,
};

static double _to_double(const void *ptr, NumcDType dt) {
  switch (dt) {
  case NUMC_DTYPE_INT8:
    return (double)*(const int8_t *)ptr;
  case NUMC_DTYPE_INT16:
    return (double)*(const int16_t *)ptr;
  case NUMC_DTYPE_INT32:
    return (double)*(const int32_t *)ptr;
  case NUMC_DTYPE_INT64:
    return (double)*(const int64_t *)ptr;
  case NUMC_DTYPE_UINT8:
    return (double)*(const uint8_t *)ptr;
  case NUMC_DTYPE_UINT16:
    return (double)*(const uint16_t *)ptr;
  case NUMC_DTYPE_UINT32:
    return (double)*(const uint32_t *)ptr;
  case NUMC_DTYPE_UINT64:
    return (double)*(const uint64_t *)ptr;
  case NUMC_DTYPE_FLOAT32:
    return (double)*(const float *)ptr;
  case NUMC_DTYPE_FLOAT64:
    return *(const double *)ptr;
  }
  return 0.0;
}

static inline void _reduce_dot_op(const struct NumcArray *a,
                                  const struct NumcArray *b,
                                  struct NumcArray *out,
                                  const NumcBinaryReductionKernel *table) {
  /* Case 3: Either is 0-D (scalar) */
  if (a->dim == 0 || b->dim == 0) {
    const struct NumcArray *scalar_arr = (a->dim == 0) ? a : b;
    const struct NumcArray *other_arr = (a->dim == 0) ? b : a;
    double val = _to_double(scalar_arr->data, scalar_arr->dtype);
    numc_mul_scalar(other_arr, val, out);
    return;
  }

  /* Case 1: Both are 1-D (vector dot product) */
  if (a->dim == 1 && b->dim == 1) {
#ifdef HAVE_BLAS
    if (a->dtype == NUMC_DTYPE_FLOAT32) {
      _dot_blis_f32(a, b, out);
      return;
    }
    if (a->dtype == NUMC_DTYPE_FLOAT64) {
      _dot_blis_f64(a, b, out);
      return;
    }
#endif
    NumcBinaryReductionKernel kern = table[a->dtype];
    kern((const char *)a->data, (const char *)b->data, (char *)out->data,
         a->size, (intptr_t)a->strides[0], (intptr_t)b->strides[0]);
    return;
  }

  /* Unified ND support: Collapse to (M, K) @ (P, K, N)
   * Results in out(M, P, N) where P is the batch of b.
   */
  size_t k_dim = a->shape[a->dim - 1];
  size_t m_dim = a->size / k_dim;
  size_t n_dim = (b->dim == 1) ? 1 : b->shape[b->dim - 1];
  size_t p_batch = b->size / (k_dim * n_dim);

  /* a is (M, K). Row stride is stride[0], Col stride is stride[last] */
  intptr_t rsa =
      (intptr_t)(a->dim > 0 ? a->strides[0] : 0) / (intptr_t)a->elem_size;
  intptr_t csa = (intptr_t)a->strides[a->dim - 1] / (intptr_t)a->elem_size;

  /* b matrix is (K, N) block. Row stride is b->strides[last-1], Col stride is
   * b->strides[last] */
  intptr_t rsb = (intptr_t)(b->dim > 1 ? b->strides[b->dim - 2]
                                       : (b->dim > 0 ? b->strides[0] : 0)) /
                 (intptr_t)b->elem_size;
  intptr_t csb = (intptr_t)(b->dim > 0 ? b->strides[b->dim - 1] : 0) /
                 (intptr_t)b->elem_size;

  /* out matrix is (M, N) block for a fixed p. Row stride is out->strides[0],
   * Col stride is out->strides[last] */
  intptr_t rso =
      (intptr_t)(out->dim > 0 ? out->strides[0] : 0) / (intptr_t)out->elem_size;
  intptr_t cso = (intptr_t)(out->dim > 0 ? out->strides[out->dim - 1] : 0) /
                 (intptr_t)out->elem_size;

  /* If a or b are not essentially 2D (matrix) after collapsing M/P,
   * we must use a more careful batching loop. */

  if (p_batch == 1) {
#ifdef HAVE_BLAS
    if (a->dtype == NUMC_DTYPE_FLOAT32 && a->is_contiguous &&
        b->is_contiguous) {
      _dot_blis_gemm_f32(a, b, out, m_dim, k_dim, n_dim);
      return;
    }
    if (a->dtype == NUMC_DTYPE_FLOAT64 && a->is_contiguous &&
        b->is_contiguous) {
      _dot_blis_gemm_f64(a, b, out, m_dim, k_dim, n_dim);
      return;
    }
#endif
    dot_naive_table[a->dtype]((const char *)a->data, (const char *)b->data,
                              (char *)out->data, m_dim, k_dim, n_dim, rsa, csa,
                              rsb, csb, rso, cso);
    return;
  }

  /* For P > 1, loop over batches of b and slices of out.
   * out is (M, P, N). For each batch p in P:
   * out[m, p, n] corresponds to a matrix (M, N) with:
   * - row stride: out->strides[0] (rsa)
   * - col stride: out->strides[last] (cso)
   * - base pointer offset: p * out->strides[1] (if 3D)
   */

  for (size_t p_idx = 0; p_idx < p_batch; p_idx++) {
    /* Calculate base pointers for this batch p */
    const char *bp =
        (const char *)b->data + p_idx * k_dim * n_dim * b->elem_size;
    char *op = (char *)out->data + p_idx * n_dim * out->elem_size;

    /* Note: This assumes b and out are contiguous in their batch/N dimensions.
     * If they are not, we'd need more complex pointer arithmetic.
     * For (M, P, N) out, the pointer to out[0, p, 0] is data + p*stride[1].
     */
    if (out->dim == 3) {
      op = (char *)out->data + p_idx * out->strides[1];
    }
    if (b->dim == 3) {
      bp = (const char *)b->data + p_idx * b->strides[0];
    }

#ifdef HAVE_BLAS
    if (a->dtype == NUMC_DTYPE_FLOAT32 && a->is_contiguous &&
        b->is_contiguous && out->is_contiguous) {
      /* We can use BLIS sgemm for each slice */
      float alpha = 1.0f, beta = 0.0f;
      bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, (dim_t)m_dim,
                (dim_t)n_dim, (dim_t)k_dim, &alpha, (float *)a->data,
                (inc_t)rsa, (inc_t)csa, (float *)bp, (inc_t)rsb, (inc_t)csb,
                &beta, (float *)op, (inc_t)rso, (inc_t)cso);
      continue;
    }
#endif
    dot_naive_table[a->dtype]((const char *)a->data, (const char *)bp,
                              (char *)op, m_dim, k_dim, n_dim, rsa, csa, rsb,
                              csb, rso, cso);
  }
}

int numc_dot(const NumcArray *a, const NumcArray *b, NumcArray *out) {
  int err = _check_dot(a, b, out);
  if (err)
    return err;
  _reduce_dot_op(a, b, out, dot_table);
  return 0;
}
