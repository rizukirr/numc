#include "dispatch.h"
#include "numc/dtype.h"
#include <numc/math.h>

#include "arch_dispatch.h"
#if NUMC_HAVE_AVX512
#include "intrinsics/compare_avx2.h"
#include "intrinsics/compare_avx512.h"
#include "intrinsics/compare_scalar_avx512.h"
#include "intrinsics/elemwise_avx2.h"
#include "intrinsics/elemwise_avx512.h"
#elif NUMC_HAVE_AVX2
#include "intrinsics/compare_avx2.h"
#include "intrinsics/compare_scalar_avx2.h"
#include "intrinsics/elemwise_avx2.h"
#elif NUMC_HAVE_SVE
#include "intrinsics/compare_sve.h"
#include "intrinsics/compare_scalar_sve.h"
#include "intrinsics/elemwise_sve.h"
#elif NUMC_HAVE_NEON
#include "intrinsics/compare_neon.h"
#include "intrinsics/compare_scalar_neon.h"
#include "intrinsics/elemwise_neon.h"
#endif
#if NUMC_HAVE_RVV
#include "intrinsics/compare_rvv.h"
#include "intrinsics/compare_scalar_rvv.h"
#include "intrinsics/elemwise_rvv.h"
#endif

/* ── Stamp out maximum and minimum ──────────────────────────────────────*/

#define STAMP_MAX(TE, CT) \
  DEFINE_BINARY_KERNEL(maximum, TE, CT, in1 > in2 ? in1 : in2)
GENERATE_NUMC_TYPES(STAMP_MAX)
#undef STAMP_MAX

#define STAMP_MIN(TE, CT) \
  DEFINE_BINARY_KERNEL(minimum, TE, CT, in1 < in2 ? in1 : in2)
GENERATE_NUMC_TYPES(STAMP_MIN)
#undef STAMP_MIN

/* ── Comparison kernels (uint8 output) ─────────────────────────────── */

typedef void (*NumcCmpKernel)(const char *a, const char *b, char *out, size_t n,
                              intptr_t sa, intptr_t sb, intptr_t so);

#define DEFINE_CMP_KERNEL(NAME, TE, CT, EXPR)                                 \
  static void _cmpkern_##NAME##_##TE(const char *a, const char *b, char *out, \
                                     size_t n, intptr_t sa, intptr_t sb,      \
                                     intptr_t so) {                           \
    for (size_t i = 0; i < n; i++) {                                          \
      CT in1 = *(const CT *)a;                                                \
      CT in2 = *(const CT *)b;                                                \
      *(uint8_t *)out = (uint8_t)(EXPR);                                      \
      a += sa;                                                                \
      b += sb;                                                                \
      out += so;                                                              \
    }                                                                         \
  }

#define CE(NAME, TE) [TE] = _cmpkern_##NAME##_##TE

/* Strided ND iteration for comparison ops */
static inline void _elemwise_cmp_nd(NumcCmpKernel kern, const char *a,
                                    const size_t *sa, const char *b,
                                    const size_t *sb, char *out,
                                    const size_t *so, const size_t *shape,
                                    size_t ndim) {
  if (ndim == 1) {
    kern(a, b, out, shape[0], (intptr_t)sa[0], (intptr_t)sb[0],
         (intptr_t)so[0]);
    return;
  }
  for (size_t i = 0; i < shape[0]; i++) {
    _elemwise_cmp_nd(kern, a + i * sa[0], sa + 1, b + i * sb[0], sb + 1,
                     out + i * so[0], so + 1, shape + 1, ndim - 1);
  }
}

static inline void _cmp_binary_op(const NumcArray *a, const NumcArray *b,
                                  NumcArray *out, const NumcCmpKernel *table) {
  NumcCmpKernel kern = table[a->dtype];
  size_t bcast_ndim = a->dim > b->dim ? a->dim : b->dim;
  if (bcast_ndim == 0)
    return;

  size_t bcast_shape[NUMC_MAX_DIMENSIONS];
  size_t sa[NUMC_MAX_DIMENSIONS], sb[NUMC_MAX_DIMENSIONS],
      so[NUMC_MAX_DIMENSIONS];

  size_t a_off = bcast_ndim - a->dim;
  size_t b_off = bcast_ndim - b->dim;

  for (size_t i = 0; i < bcast_ndim; i++) {
    size_t da = (i < a_off) ? 1 : a->shape[i - a_off];
    size_t db = (i < b_off) ? 1 : b->shape[i - b_off];
    bcast_shape[i] = da > db ? da : db;
    sa[i] = (da == 1 && bcast_shape[i] > 1) ? 0
            : (i < a_off)                   ? 0
                                            : a->strides[i - a_off];
    sb[i] = (db == 1 && bcast_shape[i] > 1) ? 0
            : (i < b_off)                   ? 0
                                            : b->strides[i - b_off];
    so[i] = out->strides[i];
  }

  _elemwise_cmp_nd(kern, (const char *)a->data, sa, (const char *)b->data, sb,
                   (char *)out->data, so, bcast_shape, bcast_ndim);
}

/* ── Stamp out comparison (same-type, scalar_inplace in SIMD path) ────── */

#if NUMC_HAVE_AVX512 || NUMC_HAVE_AVX2 || NUMC_HAVE_SVE || NUMC_HAVE_NEON || \
    NUMC_HAVE_RVV
#define STAMP_EQ(TE, CT) DEFINE_BINARY_KERNEL(eq, TE, CT, in1 == in2)
GENERATE_NUMC_TYPES(STAMP_EQ)
#undef STAMP_EQ

#define STAMP_GT(TE, CT) DEFINE_BINARY_KERNEL(gt, TE, CT, in1 > in2)
GENERATE_NUMC_TYPES(STAMP_GT)
#undef STAMP_GT

#define STAMP_LT(TE, CT) DEFINE_BINARY_KERNEL(lt, TE, CT, in1 < in2)
GENERATE_NUMC_TYPES(STAMP_LT)
#undef STAMP_LT

#define STAMP_GE(TE, CT) DEFINE_BINARY_KERNEL(ge, TE, CT, in1 >= in2)
GENERATE_NUMC_TYPES(STAMP_GE)
#undef STAMP_GE

#define STAMP_LE(TE, CT) DEFINE_BINARY_KERNEL(le, TE, CT, in1 <= in2)
GENERATE_NUMC_TYPES(STAMP_LE)
#undef STAMP_LE
#endif

/* ── Comparison kernels producing uint8 output ───────────────────── */

#define STAMP_CMP_EQ(TE, CT) DEFINE_CMP_KERNEL(eq, TE, CT, in1 == in2)
GENERATE_NUMC_TYPES(STAMP_CMP_EQ)
#undef STAMP_CMP_EQ

#define STAMP_CMP_GT(TE, CT) DEFINE_CMP_KERNEL(gt, TE, CT, in1 > in2)
GENERATE_NUMC_TYPES(STAMP_CMP_GT)
#undef STAMP_CMP_GT

#define STAMP_CMP_LT(TE, CT) DEFINE_CMP_KERNEL(lt, TE, CT, in1 < in2)
GENERATE_NUMC_TYPES(STAMP_CMP_LT)
#undef STAMP_CMP_LT

#define STAMP_CMP_GE(TE, CT) DEFINE_CMP_KERNEL(ge, TE, CT, in1 >= in2)
GENERATE_NUMC_TYPES(STAMP_CMP_GE)
#undef STAMP_CMP_GE

#define STAMP_CMP_LE(TE, CT) DEFINE_CMP_KERNEL(le, TE, CT, in1 <= in2)
GENERATE_NUMC_TYPES(STAMP_CMP_LE)
#undef STAMP_CMP_LE

static const NumcCmpKernel eq_cmp_table[] = {
    CE(eq, NUMC_DTYPE_INT8),    CE(eq, NUMC_DTYPE_INT16),
    CE(eq, NUMC_DTYPE_INT32),   CE(eq, NUMC_DTYPE_INT64),
    CE(eq, NUMC_DTYPE_UINT8),   CE(eq, NUMC_DTYPE_UINT16),
    CE(eq, NUMC_DTYPE_UINT32),  CE(eq, NUMC_DTYPE_UINT64),
    CE(eq, NUMC_DTYPE_FLOAT32), CE(eq, NUMC_DTYPE_FLOAT64),
};
static const NumcCmpKernel gt_cmp_table[] = {
    CE(gt, NUMC_DTYPE_INT8),    CE(gt, NUMC_DTYPE_INT16),
    CE(gt, NUMC_DTYPE_INT32),   CE(gt, NUMC_DTYPE_INT64),
    CE(gt, NUMC_DTYPE_UINT8),   CE(gt, NUMC_DTYPE_UINT16),
    CE(gt, NUMC_DTYPE_UINT32),  CE(gt, NUMC_DTYPE_UINT64),
    CE(gt, NUMC_DTYPE_FLOAT32), CE(gt, NUMC_DTYPE_FLOAT64),
};
static const NumcCmpKernel lt_cmp_table[] = {
    CE(lt, NUMC_DTYPE_INT8),    CE(lt, NUMC_DTYPE_INT16),
    CE(lt, NUMC_DTYPE_INT32),   CE(lt, NUMC_DTYPE_INT64),
    CE(lt, NUMC_DTYPE_UINT8),   CE(lt, NUMC_DTYPE_UINT16),
    CE(lt, NUMC_DTYPE_UINT32),  CE(lt, NUMC_DTYPE_UINT64),
    CE(lt, NUMC_DTYPE_FLOAT32), CE(lt, NUMC_DTYPE_FLOAT64),
};
static const NumcCmpKernel ge_cmp_table[] = {
    CE(ge, NUMC_DTYPE_INT8),    CE(ge, NUMC_DTYPE_INT16),
    CE(ge, NUMC_DTYPE_INT32),   CE(ge, NUMC_DTYPE_INT64),
    CE(ge, NUMC_DTYPE_UINT8),   CE(ge, NUMC_DTYPE_UINT16),
    CE(ge, NUMC_DTYPE_UINT32),  CE(ge, NUMC_DTYPE_UINT64),
    CE(ge, NUMC_DTYPE_FLOAT32), CE(ge, NUMC_DTYPE_FLOAT64),
};
static const NumcCmpKernel le_cmp_table[] = {
    CE(le, NUMC_DTYPE_INT8),    CE(le, NUMC_DTYPE_INT16),
    CE(le, NUMC_DTYPE_INT32),   CE(le, NUMC_DTYPE_INT64),
    CE(le, NUMC_DTYPE_UINT8),   CE(le, NUMC_DTYPE_UINT16),
    CE(le, NUMC_DTYPE_UINT32),  CE(le, NUMC_DTYPE_UINT64),
    CE(le, NUMC_DTYPE_FLOAT32), CE(le, NUMC_DTYPE_FLOAT64),
};

/* ── Dispatch tables ─────────────────────────────────────────────── */

static const NumcBinaryKernel maximum_table[] = {
    E(maximum, NUMC_DTYPE_INT8),    E(maximum, NUMC_DTYPE_INT16),
    E(maximum, NUMC_DTYPE_INT32),   E(maximum, NUMC_DTYPE_INT64),
    E(maximum, NUMC_DTYPE_UINT8),   E(maximum, NUMC_DTYPE_UINT16),
    E(maximum, NUMC_DTYPE_UINT32),  E(maximum, NUMC_DTYPE_UINT64),
    E(maximum, NUMC_DTYPE_FLOAT32), E(maximum, NUMC_DTYPE_FLOAT64),
};

static const NumcBinaryKernel minimum_table[] = {
    E(minimum, NUMC_DTYPE_INT8),    E(minimum, NUMC_DTYPE_INT16),
    E(minimum, NUMC_DTYPE_INT32),   E(minimum, NUMC_DTYPE_INT64),
    E(minimum, NUMC_DTYPE_UINT8),   E(minimum, NUMC_DTYPE_UINT16),
    E(minimum, NUMC_DTYPE_UINT32),  E(minimum, NUMC_DTYPE_UINT64),
    E(minimum, NUMC_DTYPE_FLOAT32), E(minimum, NUMC_DTYPE_FLOAT64),
};

/* ── Public API ──────────────────────────────────────────────────── */

#define DEFINE_ELEMWISE_BINARY(NAME, TABLE)                                 \
  int numc_##NAME(const NumcArray *a, const NumcArray *b, NumcArray *out) { \
    int err = _check_binary(a, b, out);                                     \
    if (err)                                                                \
      return err;                                                           \
    _binary_op(a, b, out, TABLE);                                           \
    return 0;                                                               \
  }

#define DEFINE_ELEMWISE_SCALAR(NAME, TABLE)                       \
  int numc_##NAME##_scalar(const NumcArray *a, double scalar,     \
                           NumcArray *out) {                      \
    int err = _check_unary(a, out);                               \
    if (err)                                                      \
      return err;                                                 \
    char buf[8];                                                  \
    _double_to_dtype(scalar, a->dtype, buf);                      \
    _scalar_op(a, buf, out, TABLE);                               \
    return 0;                                                     \
  }                                                               \
  int numc_##NAME##_scalar_inplace(NumcArray *a, double scalar) { \
    return _scalar_op_inplace(a, scalar, TABLE);                  \
  }

/* ── SIMD fast-path for maximum/minimum ──────────────────────────── */

#if NUMC_HAVE_AVX512 || NUMC_HAVE_AVX2 || NUMC_HAVE_SVE || NUMC_HAVE_NEON || \
    NUMC_HAVE_RVV

typedef void (*FastBinKern)(const void *restrict, const void *restrict,
                            void *restrict, size_t);

#if NUMC_HAVE_AVX512
#define FBIN(OP, SFX) (FastBinKern) _fast_##OP##_##SFX##_avx512
#elif NUMC_HAVE_AVX2
#define FBIN(OP, SFX) (FastBinKern) _fast_##OP##_##SFX##_avx2
#elif NUMC_HAVE_SVE
#define FBIN(OP, SFX) (FastBinKern) _fast_##OP##_##SFX##_sve
#elif NUMC_HAVE_NEON
#define FBIN(OP, SFX) (FastBinKern) _fast_##OP##_##SFX##_neon
#elif NUMC_HAVE_RVV
#define FBIN(OP, SFX) (FastBinKern) _fast_##OP##_##SFX##_rvv
#endif

#define FBIN_TABLE(OP)                           \
  static const FastBinKern OP##_fast_table[] = { \
      [NUMC_DTYPE_INT8] = FBIN(OP, i8),          \
      [NUMC_DTYPE_INT16] = FBIN(OP, i16),        \
      [NUMC_DTYPE_INT32] = FBIN(OP, i32),        \
      [NUMC_DTYPE_INT64] = FBIN(OP, i64),        \
      [NUMC_DTYPE_UINT8] = FBIN(OP, u8),         \
      [NUMC_DTYPE_UINT16] = FBIN(OP, u16),       \
      [NUMC_DTYPE_UINT32] = FBIN(OP, u32),       \
      [NUMC_DTYPE_UINT64] = FBIN(OP, u64),       \
      [NUMC_DTYPE_FLOAT32] = FBIN(OP, f32),      \
      [NUMC_DTYPE_FLOAT64] = FBIN(OP, f64),      \
  }

FBIN_TABLE(maximum);
FBIN_TABLE(minimum);
FBIN_TABLE(eq);
FBIN_TABLE(gt);
FBIN_TABLE(lt);
FBIN_TABLE(ge);
FBIN_TABLE(le);
#undef FBIN_TABLE
#undef FBIN

#define DEFINE_BINARY_SIMD(NAME, FAST_TABLE, FALLBACK_TABLE)                   \
  int numc_##NAME(const NumcArray *a, const NumcArray *b, NumcArray *out) {    \
    int err = _check_binary(a, b, out);                                        \
    if (err)                                                                   \
      return err;                                                              \
    if (a->is_contiguous && b->is_contiguous && out->is_contiguous &&          \
        a->dim == b->dim) {                                                    \
      bool same_shape = true;                                                  \
      for (size_t d = 0; d < a->dim; d++)                                      \
        if (a->shape[d] != b->shape[d]) {                                      \
          same_shape = false;                                                  \
          break;                                                               \
        }                                                                      \
      if (same_shape) {                                                        \
        FastBinKern kern = FAST_TABLE[a->dtype];                               \
        size_t n = a->size, es = a->elem_size;                                 \
        int nt = (int)((n * es) / NUMC_OMP_BYTES_PER_THREAD);                  \
        NUMC_OMP_CAP_THREADS(nt);                                              \
        if (nt >= 2) {                                                         \
          size_t chunk = (n + (size_t)nt - 1) / (size_t)nt;                    \
          NUMC_PRAGMA(                                                      \
              omp parallel for schedule(static) num_threads(nt))               \
          for (int t = 0; t < nt; t++) {                                       \
            size_t s = (size_t)t * chunk;                                      \
            size_t e = s + chunk;                                              \
            if (e > n)                                                         \
              e = n;                                                           \
            if (s < n)                                                         \
              kern((const char *)a->data + s * es,                             \
                   (const char *)b->data + s * es, (char *)out->data + s * es, \
                   e - s);                                                     \
          }                                                                    \
        } else {                                                               \
          kern(a->data, b->data, out->data, n);                                \
        }                                                                      \
        return 0;                                                              \
      }                                                                        \
    }                                                                          \
    _binary_op(a, b, out, FALLBACK_TABLE);                                     \
    return 0;                                                                  \
  }

DEFINE_BINARY_SIMD(maximum, maximum_fast_table, maximum_table)
DEFINE_BINARY_SIMD(minimum, minimum_fast_table, minimum_table)
#undef DEFINE_BINARY_SIMD

/* Comparison SIMD dispatch: output is uint8 (1 byte per element) */
#define DEFINE_CMP_BINARY_SIMD(NAME, FAST_TABLE, FALLBACK_TABLE)             \
  int numc_##NAME(const NumcArray *a, const NumcArray *b, NumcArray *out) {  \
    int err = _check_cmp_binary(a, b, out);                                  \
    if (err)                                                                 \
      return err;                                                            \
    if (a->is_contiguous && b->is_contiguous && out->is_contiguous &&        \
        a->dim == b->dim) {                                                  \
      bool same_shape = true;                                                \
      for (size_t d = 0; d < a->dim; d++)                                    \
        if (a->shape[d] != b->shape[d]) {                                    \
          same_shape = false;                                                \
          break;                                                             \
        }                                                                    \
      if (same_shape) {                                                      \
        FastBinKern kern = FAST_TABLE[a->dtype];                             \
        size_t n = a->size;                                                  \
        size_t in_es = a->elem_size;                                         \
        size_t io_total = n * (2 * in_es + 1);                               \
        int nt = (int)(io_total / NUMC_OMP_BYTES_PER_THREAD);                \
        NUMC_OMP_CAP_THREADS(nt);                                            \
        if (nt >= 2) {                                                       \
          size_t chunk = (n + (size_t)nt - 1) / (size_t)nt;                  \
          NUMC_PRAGMA(                                                      \
              omp parallel for schedule(static) num_threads(nt))             \
          for (int t = 0; t < nt; t++) {                                     \
            size_t s = (size_t)t * chunk;                                    \
            size_t e = s + chunk;                                            \
            if (e > n)                                                       \
              e = n;                                                         \
            if (s < n)                                                       \
              kern((const char *)a->data + s * in_es,                        \
                   (const char *)b->data + s * in_es, (char *)out->data + s, \
                   e - s);                                                   \
          }                                                                  \
        } else {                                                             \
          kern(a->data, b->data, out->data, n);                              \
        }                                                                    \
        return 0;                                                            \
      }                                                                      \
    }                                                                        \
    _cmp_binary_op(a, b, out, FALLBACK_TABLE);                               \
    return 0;                                                                \
  }

DEFINE_CMP_BINARY_SIMD(eq, eq_fast_table, eq_cmp_table)
DEFINE_CMP_BINARY_SIMD(gt, gt_fast_table, gt_cmp_table)
DEFINE_CMP_BINARY_SIMD(lt, lt_fast_table, lt_cmp_table)
DEFINE_CMP_BINARY_SIMD(ge, ge_fast_table, ge_cmp_table)
DEFINE_CMP_BINARY_SIMD(le, le_fast_table, le_cmp_table)
#undef DEFINE_CMP_BINARY_SIMD

#else
DEFINE_ELEMWISE_BINARY(maximum, maximum_table)
DEFINE_ELEMWISE_BINARY(minimum, minimum_table)

/* Non-SIMD comparison: uint8 output via generic strided kernels */
#define DEFINE_CMP_BINARY_GENERIC(NAME, CMP_TABLE)                          \
  int numc_##NAME(const NumcArray *a, const NumcArray *b, NumcArray *out) { \
    int err = _check_cmp_binary(a, b, out);                                 \
    if (err)                                                                \
      return err;                                                           \
    _cmp_binary_op(a, b, out, CMP_TABLE);                                   \
    return 0;                                                               \
  }
DEFINE_CMP_BINARY_GENERIC(eq, eq_cmp_table)
DEFINE_CMP_BINARY_GENERIC(gt, gt_cmp_table)
DEFINE_CMP_BINARY_GENERIC(lt, lt_cmp_table)
DEFINE_CMP_BINARY_GENERIC(ge, ge_cmp_table)
DEFINE_CMP_BINARY_GENERIC(le, le_cmp_table)
#undef DEFINE_CMP_BINARY_GENERIC
#endif

/* ── SIMD scalar comparison dispatch ─────────────────────────────── */

#if NUMC_HAVE_AVX512 || NUMC_HAVE_AVX2 || NUMC_HAVE_SVE || NUMC_HAVE_NEON || \
    NUMC_HAVE_RVV

typedef void (*CmpScKern)(const void *restrict, const void *restrict,
                          void *restrict, size_t);

#if NUMC_HAVE_AVX512
#define CMPSC(OP, SFX) (CmpScKern) _cmpsc_##OP##_##SFX##_avx512
#elif NUMC_HAVE_AVX2
#define CMPSC(OP, SFX) (CmpScKern) _cmpsc_##OP##_##SFX##_avx2
#elif NUMC_HAVE_SVE
#define CMPSC(OP, SFX) (CmpScKern) _cmpsc_##OP##_##SFX##_sve
#elif NUMC_HAVE_NEON
#define CMPSC(OP, SFX) (CmpScKern) _cmpsc_##OP##_##SFX##_neon
#elif NUMC_HAVE_RVV
#define CMPSC(OP, SFX) (CmpScKern) _cmpsc_##OP##_##SFX##_rvv
#endif

#define CMPSC_TABLE(OP)                      \
  static const CmpScKern OP##_sc_table[] = { \
      [NUMC_DTYPE_INT8] = CMPSC(OP, i8),     \
      [NUMC_DTYPE_INT16] = CMPSC(OP, i16),   \
      [NUMC_DTYPE_INT32] = CMPSC(OP, i32),   \
      [NUMC_DTYPE_INT64] = CMPSC(OP, i64),   \
      [NUMC_DTYPE_UINT8] = CMPSC(OP, u8),    \
      [NUMC_DTYPE_UINT16] = CMPSC(OP, u16),  \
      [NUMC_DTYPE_UINT32] = CMPSC(OP, u32),  \
      [NUMC_DTYPE_UINT64] = CMPSC(OP, u64),  \
      [NUMC_DTYPE_FLOAT32] = CMPSC(OP, f32), \
      [NUMC_DTYPE_FLOAT64] = CMPSC(OP, f64), \
  }

CMPSC_TABLE(eq);
CMPSC_TABLE(gt);
CMPSC_TABLE(lt);
CMPSC_TABLE(ge);
CMPSC_TABLE(le);
#undef CMPSC_TABLE
#undef CMPSC

#define DEFINE_CMP_SCALAR_WITH_SIMD(NAME, TABLE)                   \
  int numc_##NAME##_scalar(const NumcArray *a, double scalar,      \
                           NumcArray *out) {                       \
    int err = _check_cmp_unary(a, out);                            \
    if (err)                                                       \
      return err;                                                  \
    char buf[8];                                                   \
    _double_to_dtype(scalar, a->dtype, buf);                       \
    if (a->is_contiguous && out->is_contiguous) {                  \
      CmpScKern kern = NAME##_sc_table[a->dtype];                  \
      size_t n = a->size;                                          \
      size_t in_es = a->elem_size;                                 \
      size_t io_total = n * (2 * in_es + 1);                       \
      int nt = (int)(io_total / NUMC_OMP_BYTES_PER_THREAD);        \
      NUMC_OMP_CAP_THREADS(nt);                                    \
      if (nt >= 2) {                                               \
        size_t chunk = (n + (size_t)nt - 1) / (size_t)nt;          \
        NUMC_PRAGMA(                                                   \
            omp parallel for schedule(static) num_threads(nt))     \
        for (int t = 0; t < nt; t++) {                             \
          size_t s = (size_t)t * chunk;                            \
          size_t e = s + chunk;                                    \
          if (e > n)                                               \
            e = n;                                                 \
          if (s < n)                                               \
            kern((const char *)a->data + s * in_es, buf,           \
                 (char *)out->data + s, e - s);                    \
        }                                                          \
      } else {                                                     \
        kern(a->data, buf, out->data, n);                          \
      }                                                            \
      return 0;                                                    \
    }                                                              \
    /* TODO: non-contiguous scalar comparison with uint8 output */ \
    return NUMC_ERR_SHAPE;                                         \
  }                                                                \
  int numc_##NAME##_scalar_inplace(NumcArray *a, double scalar) {  \
    return _scalar_op_inplace(a, scalar, TABLE);                   \
  }

static const NumcBinaryKernel eq_table[] = {
    E(eq, NUMC_DTYPE_INT8),    E(eq, NUMC_DTYPE_INT16),
    E(eq, NUMC_DTYPE_INT32),   E(eq, NUMC_DTYPE_INT64),
    E(eq, NUMC_DTYPE_UINT8),   E(eq, NUMC_DTYPE_UINT16),
    E(eq, NUMC_DTYPE_UINT32),  E(eq, NUMC_DTYPE_UINT64),
    E(eq, NUMC_DTYPE_FLOAT32), E(eq, NUMC_DTYPE_FLOAT64),
};
static const NumcBinaryKernel gt_table[] = {
    E(gt, NUMC_DTYPE_INT8),    E(gt, NUMC_DTYPE_INT16),
    E(gt, NUMC_DTYPE_INT32),   E(gt, NUMC_DTYPE_INT64),
    E(gt, NUMC_DTYPE_UINT8),   E(gt, NUMC_DTYPE_UINT16),
    E(gt, NUMC_DTYPE_UINT32),  E(gt, NUMC_DTYPE_UINT64),
    E(gt, NUMC_DTYPE_FLOAT32), E(gt, NUMC_DTYPE_FLOAT64),
};
static const NumcBinaryKernel lt_table[] = {
    E(lt, NUMC_DTYPE_INT8),    E(lt, NUMC_DTYPE_INT16),
    E(lt, NUMC_DTYPE_INT32),   E(lt, NUMC_DTYPE_INT64),
    E(lt, NUMC_DTYPE_UINT8),   E(lt, NUMC_DTYPE_UINT16),
    E(lt, NUMC_DTYPE_UINT32),  E(lt, NUMC_DTYPE_UINT64),
    E(lt, NUMC_DTYPE_FLOAT32), E(lt, NUMC_DTYPE_FLOAT64),
};
static const NumcBinaryKernel ge_table[] = {
    E(ge, NUMC_DTYPE_INT8),    E(ge, NUMC_DTYPE_INT16),
    E(ge, NUMC_DTYPE_INT32),   E(ge, NUMC_DTYPE_INT64),
    E(ge, NUMC_DTYPE_UINT8),   E(ge, NUMC_DTYPE_UINT16),
    E(ge, NUMC_DTYPE_UINT32),  E(ge, NUMC_DTYPE_UINT64),
    E(ge, NUMC_DTYPE_FLOAT32), E(ge, NUMC_DTYPE_FLOAT64),
};
static const NumcBinaryKernel le_table[] = {
    E(le, NUMC_DTYPE_INT8),    E(le, NUMC_DTYPE_INT16),
    E(le, NUMC_DTYPE_INT32),   E(le, NUMC_DTYPE_INT64),
    E(le, NUMC_DTYPE_UINT8),   E(le, NUMC_DTYPE_UINT16),
    E(le, NUMC_DTYPE_UINT32),  E(le, NUMC_DTYPE_UINT64),
    E(le, NUMC_DTYPE_FLOAT32), E(le, NUMC_DTYPE_FLOAT64),
};

// NOLINTNEXTLINE(misc-use-internal-linkage)
DEFINE_CMP_SCALAR_WITH_SIMD(eq, eq_table)
// NOLINTNEXTLINE(misc-use-internal-linkage)
DEFINE_CMP_SCALAR_WITH_SIMD(gt, gt_table)
// NOLINTNEXTLINE(misc-use-internal-linkage)
DEFINE_CMP_SCALAR_WITH_SIMD(lt, lt_table)
// NOLINTNEXTLINE(misc-use-internal-linkage)
DEFINE_CMP_SCALAR_WITH_SIMD(ge, ge_table)
// NOLINTNEXTLINE(misc-use-internal-linkage)
DEFINE_CMP_SCALAR_WITH_SIMD(le, le_table)
#undef DEFINE_CMP_SCALAR_WITH_SIMD

#else /* No SIMD available */

/* Generic scalar comparison: output uint8 via cmp kernel */
#define DEFINE_CMP_SCALAR_GENERIC(NAME, TABLE)                     \
  int numc_##NAME##_scalar(const NumcArray *a, double scalar,      \
                           NumcArray *out) {                       \
    int err = _check_cmp_unary(a, out);                            \
    if (err)                                                       \
      return err;                                                  \
    char buf[8];                                                   \
    _double_to_dtype(scalar, a->dtype, buf);                       \
    NumcCmpKernel kern = TABLE[a->dtype];                          \
    if (a->is_contiguous && out->is_contiguous) {                  \
      kern((const char *)a->data, buf, (char *)out->data, a->size, \
           (intptr_t)a->elem_size, 0, 1);                          \
    } else {                                                       \
      /* 1D strided fallback */                                    \
      kern((const char *)a->data, buf, (char *)out->data, a->size, \
           (intptr_t)a->strides[a->dim - 1], 0,                    \
           (intptr_t)out->strides[out->dim - 1]);                  \
    }                                                              \
    return 0;                                                      \
  }                                                                \
  int numc_##NAME##_scalar_inplace(NumcArray *a, double scalar) {  \
    return _scalar_op_inplace(a, scalar, TABLE);                   \
  }

// NOLINTNEXTLINE(misc-use-internal-linkage)
DEFINE_CMP_SCALAR_GENERIC(eq, eq_cmp_table)
// NOLINTNEXTLINE(misc-use-internal-linkage)
DEFINE_CMP_SCALAR_GENERIC(gt, gt_cmp_table)
// NOLINTNEXTLINE(misc-use-internal-linkage)
DEFINE_CMP_SCALAR_GENERIC(lt, lt_cmp_table)
// NOLINTNEXTLINE(misc-use-internal-linkage)
DEFINE_CMP_SCALAR_GENERIC(ge, ge_cmp_table)
// NOLINTNEXTLINE(misc-use-internal-linkage)
DEFINE_CMP_SCALAR_GENERIC(le, le_cmp_table)
#undef DEFINE_CMP_SCALAR_GENERIC

#endif
