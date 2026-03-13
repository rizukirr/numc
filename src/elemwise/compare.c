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

/* ── Stamp out comparison ──────────────────────────────────────────── */

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

#if NUMC_HAVE_AVX512 || NUMC_HAVE_AVX2 || NUMC_HAVE_SVE || \
    NUMC_HAVE_NEON || NUMC_HAVE_RVV

typedef void (*FastBinKern)(const void *restrict, const void *restrict,
                            void *restrict, size_t);

#if NUMC_HAVE_AVX512
#define FBIN(OP, SFX) (FastBinKern)_fast_##OP##_##SFX##_avx512
#elif NUMC_HAVE_AVX2
#define FBIN(OP, SFX) (FastBinKern)_fast_##OP##_##SFX##_avx2
#elif NUMC_HAVE_SVE
#define FBIN(OP, SFX) (FastBinKern)_fast_##OP##_##SFX##_sve
#elif NUMC_HAVE_NEON
#define FBIN(OP, SFX) (FastBinKern)_fast_##OP##_##SFX##_neon
#elif NUMC_HAVE_RVV
#define FBIN(OP, SFX) (FastBinKern)_fast_##OP##_##SFX##_rvv
#endif

#define FBIN_TABLE(OP)                                                 \
  static const FastBinKern OP##_fast_table[] = {                       \
      [NUMC_DTYPE_INT8] = FBIN(OP, i8),                               \
      [NUMC_DTYPE_INT16] = FBIN(OP, i16),                             \
      [NUMC_DTYPE_INT32] = FBIN(OP, i32),                             \
      [NUMC_DTYPE_INT64] = FBIN(OP, i64),                             \
      [NUMC_DTYPE_UINT8] = FBIN(OP, u8),                              \
      [NUMC_DTYPE_UINT16] = FBIN(OP, u16),                            \
      [NUMC_DTYPE_UINT32] = FBIN(OP, u32),                            \
      [NUMC_DTYPE_UINT64] = FBIN(OP, u64),                            \
      [NUMC_DTYPE_FLOAT32] = FBIN(OP, f32),                           \
      [NUMC_DTYPE_FLOAT64] = FBIN(OP, f64),                           \
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

#define DEFINE_BINARY_SIMD(NAME, FAST_TABLE, FALLBACK_TABLE)                \
  int numc_##NAME(const NumcArray *a, const NumcArray *b, NumcArray *out) { \
    int err = _check_binary(a, b, out);                                     \
    if (err)                                                                \
      return err;                                                           \
    if (a->is_contiguous && b->is_contiguous && out->is_contiguous &&       \
        a->dim == b->dim) {                                                 \
      bool same_shape = true;                                               \
      for (size_t d = 0; d < a->dim; d++)                                   \
        if (a->shape[d] != b->shape[d]) {                                   \
          same_shape = false;                                               \
          break;                                                            \
        }                                                                   \
      if (same_shape) {                                                     \
        FastBinKern kern = FAST_TABLE[a->dtype];                            \
        size_t n = a->size, es = a->elem_size, total = n * es;             \
        int nt = (int)(total / NUMC_OMP_BYTES_PER_THREAD);                 \
        if (nt >= 2) {                                                      \
          size_t chunk = (n + (size_t)nt - 1) / (size_t)nt;                \
          NUMC_PRAGMA(                                                      \
              omp parallel for schedule(static) num_threads(nt))            \
          for (int t = 0; t < nt; t++) {                                   \
            size_t s = (size_t)t * chunk;                                  \
            size_t e = s + chunk;                                           \
            if (e > n)                                                      \
              e = n;                                                        \
            if (s < n)                                                      \
              kern((const char *)a->data + s * es,                         \
                   (const char *)b->data + s * es,                         \
                   (char *)out->data + s * es, e - s);                     \
          }                                                                 \
        } else {                                                            \
          kern(a->data, b->data, out->data, n);                            \
        }                                                                   \
        return 0;                                                           \
      }                                                                     \
    }                                                                       \
    _binary_op(a, b, out, FALLBACK_TABLE);                                  \
    return 0;                                                               \
  }

DEFINE_BINARY_SIMD(maximum, maximum_fast_table, maximum_table)
DEFINE_BINARY_SIMD(minimum, minimum_fast_table, minimum_table)
DEFINE_BINARY_SIMD(eq, eq_fast_table, eq_table)
DEFINE_BINARY_SIMD(gt, gt_fast_table, gt_table)
DEFINE_BINARY_SIMD(lt, lt_fast_table, lt_table)
DEFINE_BINARY_SIMD(ge, ge_fast_table, ge_table)
DEFINE_BINARY_SIMD(le, le_fast_table, le_table)
#undef DEFINE_BINARY_SIMD

#else
DEFINE_ELEMWISE_BINARY(maximum, maximum_table)
DEFINE_ELEMWISE_BINARY(minimum, minimum_table)
DEFINE_ELEMWISE_BINARY(eq, eq_table)
DEFINE_ELEMWISE_BINARY(gt, gt_table)
DEFINE_ELEMWISE_BINARY(lt, lt_table)
DEFINE_ELEMWISE_BINARY(ge, ge_table)
DEFINE_ELEMWISE_BINARY(le, le_table)
#endif

/* ── SIMD scalar comparison dispatch ─────────────────────────────── */

#if NUMC_HAVE_AVX512 || NUMC_HAVE_AVX2 || NUMC_HAVE_SVE || \
    NUMC_HAVE_NEON || NUMC_HAVE_RVV

typedef void (*CmpScKern)(const void *restrict, const void *restrict,
                          void *restrict, size_t);

#if NUMC_HAVE_AVX512
#define CMPSC(OP, SFX) (CmpScKern)_cmpsc_##OP##_##SFX##_avx512
#elif NUMC_HAVE_AVX2
#define CMPSC(OP, SFX) (CmpScKern)_cmpsc_##OP##_##SFX##_avx2
#elif NUMC_HAVE_SVE
#define CMPSC(OP, SFX) (CmpScKern)_cmpsc_##OP##_##SFX##_sve
#elif NUMC_HAVE_NEON
#define CMPSC(OP, SFX) (CmpScKern)_cmpsc_##OP##_##SFX##_neon
#elif NUMC_HAVE_RVV
#define CMPSC(OP, SFX) (CmpScKern)_cmpsc_##OP##_##SFX##_rvv
#endif

#define CMPSC_TABLE(OP)                                                \
  static const CmpScKern OP##_sc_table[] = {                           \
      [NUMC_DTYPE_INT8] = CMPSC(OP, i8),                              \
      [NUMC_DTYPE_INT16] = CMPSC(OP, i16),                            \
      [NUMC_DTYPE_INT32] = CMPSC(OP, i32),                            \
      [NUMC_DTYPE_INT64] = CMPSC(OP, i64),                            \
      [NUMC_DTYPE_UINT8] = CMPSC(OP, u8),                             \
      [NUMC_DTYPE_UINT16] = CMPSC(OP, u16),                           \
      [NUMC_DTYPE_UINT32] = CMPSC(OP, u32),                           \
      [NUMC_DTYPE_UINT64] = CMPSC(OP, u64),                           \
      [NUMC_DTYPE_FLOAT32] = CMPSC(OP, f32),                          \
      [NUMC_DTYPE_FLOAT64] = CMPSC(OP, f64),                          \
  }

CMPSC_TABLE(eq);
CMPSC_TABLE(gt);
CMPSC_TABLE(lt);
CMPSC_TABLE(ge);
CMPSC_TABLE(le);
#undef CMPSC_TABLE
#undef CMPSC

#define DEFINE_CMP_SCALAR_WITH_SIMD(NAME, TABLE)                       \
  int numc_##NAME##_scalar(const NumcArray *a, double scalar,          \
                           NumcArray *out) {                           \
    int err = _check_unary(a, out);                                    \
    if (err)                                                           \
      return err;                                                      \
    char buf[8];                                                       \
    _double_to_dtype(scalar, a->dtype, buf);                           \
    if (a->is_contiguous && out->is_contiguous) {                      \
      CmpScKern kern = NAME##_sc_table[a->dtype];                     \
      size_t n = a->size;                                              \
      size_t es = a->elem_size;                                        \
      size_t total = n * es;                                           \
      int nt = (int)(total / NUMC_OMP_BYTES_PER_THREAD);              \
      if (nt >= 2) {                                                   \
        size_t chunk = (n + (size_t)nt - 1) / (size_t)nt;            \
        NUMC_PRAGMA(                                                   \
            omp parallel for schedule(static) num_threads(nt))         \
        for (int t = 0; t < nt; t++) {                                \
          size_t s = (size_t)t * chunk;                               \
          size_t e = s + chunk;                                        \
          if (e > n)                                                   \
            e = n;                                                     \
          if (s < n)                                                   \
            kern((const char *)a->data + s * es, buf,                 \
                 (char *)out->data + s * es, e - s);                  \
        }                                                              \
      } else {                                                         \
        kern(a->data, buf, out->data, n);                             \
      }                                                                \
      return 0;                                                        \
    }                                                                  \
    _scalar_op(a, buf, out, TABLE);                                    \
    return 0;                                                          \
  }                                                                    \
  int numc_##NAME##_scalar_inplace(NumcArray *a, double scalar) {      \
    return _scalar_op_inplace(a, scalar, TABLE);                       \
  }

DEFINE_CMP_SCALAR_WITH_SIMD(eq, eq_table)
DEFINE_CMP_SCALAR_WITH_SIMD(gt, gt_table)
DEFINE_CMP_SCALAR_WITH_SIMD(lt, lt_table)
DEFINE_CMP_SCALAR_WITH_SIMD(ge, ge_table)
DEFINE_CMP_SCALAR_WITH_SIMD(le, le_table)
#undef DEFINE_CMP_SCALAR_WITH_SIMD

#else /* No SIMD available */

DEFINE_ELEMWISE_SCALAR(eq, eq_table)
DEFINE_ELEMWISE_SCALAR(gt, gt_table)
DEFINE_ELEMWISE_SCALAR(lt, lt_table)
DEFINE_ELEMWISE_SCALAR(ge, ge_table)
DEFINE_ELEMWISE_SCALAR(le, le_table)

#endif
