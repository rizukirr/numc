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
GENERATE_INT32_NUMC_TYPES(STAMP_MEAN_DBL)
STAMP_MEAN_DBL(NUMC_DTYPE_INT64, NUMC_INT64)
STAMP_MEAN_DBL(NUMC_DTYPE_UINT64, NUMC_UINT64)
#undef STAMP_MEAN_DBL

/* float32/float64: native division */
#define STAMP_MEAN_FLOAT(TE, CT)                                               \
  static void _kern_mean_##TE(const char *a, char *out, size_t n,              \
                              intptr_t sa) {                                   \
    _kern_sum_##TE(a, out, n, sa);                                             \
    if (n > 0)                                                                 \
      *(CT *)out /= (CT)n;                                                     \
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

static const NumcReductionKernel _argmax_table[] = {
    R(argmax, NUMC_DTYPE_INT8),    R(argmax, NUMC_DTYPE_INT16),
    R(argmax, NUMC_DTYPE_INT32),   R(argmax, NUMC_DTYPE_INT64),
    R(argmax, NUMC_DTYPE_UINT8),   R(argmax, NUMC_DTYPE_UINT16),
    R(argmax, NUMC_DTYPE_UINT32),  R(argmax, NUMC_DTYPE_UINT64),
    R(argmax, NUMC_DTYPE_FLOAT32), R(argmax, NUMC_DTYPE_FLOAT64),
};

static const NumcReductionKernel _argmin_table[] = {
    R(argmin, NUMC_DTYPE_INT8),    R(argmin, NUMC_DTYPE_INT16),
    R(argmin, NUMC_DTYPE_INT32),   R(argmin, NUMC_DTYPE_INT64),
    R(argmin, NUMC_DTYPE_UINT8),   R(argmin, NUMC_DTYPE_UINT16),
    R(argmin, NUMC_DTYPE_UINT32),  R(argmin, NUMC_DTYPE_UINT64),
    R(argmin, NUMC_DTYPE_FLOAT32), R(argmin, NUMC_DTYPE_FLOAT64),
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

/* ── Fused argmax row-reduce kernels ──────────────────────────────
 *
 * Tracks best value (in VLA scratch buffer) + index (in output).
 * Inner loop auto-vectorizes the comparison; the conditional index
 * update is scalar but amortized across rows. */

typedef void (*NumcArgRowReduceKernel)(const char *restrict base,
                                       intptr_t row_stride, size_t nrows,
                                       char *restrict dst, size_t ncols);

#define STAMP_ARGMAX_FUSED(TE, CT)                                             \
  static void _argmax_fused_##TE(const char *restrict base,                    \
                                  intptr_t row_stride, size_t nrows,           \
                                  char *restrict dst, size_t ncols) {          \
    int64_t *restrict idx = (int64_t *)dst;                                    \
    CT best[ncols]; /* VLA scratch for best values */                          \
    const CT *restrict first = (const CT *)base;                               \
    for (size_t i = 0; i < ncols; i++) {                                       \
      best[i] = first[i];                                                      \
      idx[i] = 0;                                                              \
    }                                                                          \
    for (size_t r = 1; r < nrows; r++) {                                       \
      const CT *restrict row = (const CT *)(base + r * row_stride);            \
      for (size_t i = 0; i < ncols; i++) {                                     \
        if (row[i] > best[i]) {                                                \
          best[i] = row[i];                                                    \
          idx[i] = (int64_t)r;                                                 \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }
GENERATE_NUMC_TYPES(STAMP_ARGMAX_FUSED)
#undef STAMP_ARGMAX_FUSED

#define STAMP_ARGMIN_FUSED(TE, CT)                                             \
  static void _argmin_fused_##TE(const char *restrict base,                    \
                                  intptr_t row_stride, size_t nrows,           \
                                  char *restrict dst, size_t ncols) {          \
    int64_t *restrict idx = (int64_t *)dst;                                    \
    CT best[ncols]; /* VLA scratch for best values */                          \
    const CT *restrict first = (const CT *)base;                               \
    for (size_t i = 0; i < ncols; i++) {                                       \
      best[i] = first[i];                                                      \
      idx[i] = 0;                                                              \
    }                                                                          \
    for (size_t r = 1; r < nrows; r++) {                                       \
      const CT *restrict row = (const CT *)(base + r * row_stride);            \
      for (size_t i = 0; i < ncols; i++) {                                     \
        if (row[i] < best[i]) {                                                \
          best[i] = row[i];                                                    \
          idx[i] = (int64_t)r;                                                 \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }
GENERATE_NUMC_TYPES(STAMP_ARGMIN_FUSED)
#undef STAMP_ARGMIN_FUSED

#define F(OP, TE) [TE] = _##OP##_fused_##TE
static const NumcArgRowReduceKernel _argmax_fused_table[] = {
    F(argmax, NUMC_DTYPE_INT8),    F(argmax, NUMC_DTYPE_INT16),
    F(argmax, NUMC_DTYPE_INT32),   F(argmax, NUMC_DTYPE_INT64),
    F(argmax, NUMC_DTYPE_UINT8),   F(argmax, NUMC_DTYPE_UINT16),
    F(argmax, NUMC_DTYPE_UINT32),  F(argmax, NUMC_DTYPE_UINT64),
    F(argmax, NUMC_DTYPE_FLOAT32), F(argmax, NUMC_DTYPE_FLOAT64),
};
static const NumcArgRowReduceKernel _argmin_fused_table[] = {
    F(argmin, NUMC_DTYPE_INT8),    F(argmin, NUMC_DTYPE_INT16),
    F(argmin, NUMC_DTYPE_INT32),   F(argmin, NUMC_DTYPE_INT64),
    F(argmin, NUMC_DTYPE_UINT8),   F(argmin, NUMC_DTYPE_UINT16),
    F(argmin, NUMC_DTYPE_UINT32),  F(argmin, NUMC_DTYPE_UINT64),
    F(argmin, NUMC_DTYPE_FLOAT32), F(argmin, NUMC_DTYPE_FLOAT64),
};
#undef F

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

int numc_argmax(const NumcArray *a, NumcArray *out) {
  int err = _check_argreduce_full(a, out);
  if (err)
    return err;
  _reduce_full_op(a, out, _argmax_table);
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

    _argmax_fused_table[a->dtype]((const char *)a->data, reduce_stride,
                                   reduce_len, (char *)out->data, slice_elems);
    return 0;
  }

  /* Generic path: per-element reduction via ND iterator */
  _reduce_axis_op(a, ax, keepdim, out, _argmax_table);
  return 0;
}

int numc_argmin(const NumcArray *a, NumcArray *out) {
  int err = _check_argreduce_full(a, out);
  if (err)
    return err;
  _reduce_full_op(a, out, _argmin_table);
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

    _argmin_fused_table[a->dtype]((const char *)a->data, reduce_stride,
                                   reduce_len, (char *)out->data, slice_elems);
    return 0;
  }

  /* Generic path: per-element reduction via ND iterator */
  _reduce_axis_op(a, ax, keepdim, out, _argmin_table);
  return 0;
}
