#include "dispatch.h"
#include "numc/dtype.h"
#include <numc/math.h>

#include "arch_dispatch.h"
#if NUMC_HAVE_AVX2
#include "intrinsics/compare_avx2.h"
#elif NUMC_HAVE_SVE
#include "intrinsics/compare_sve.h"
#elif NUMC_HAVE_NEON
#include "intrinsics/compare_neon.h"
#endif
#if NUMC_HAVE_RVV
#include "intrinsics/compare_rvv.h"
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

DEFINE_ELEMWISE_BINARY(maximum, maximum_table)
DEFINE_ELEMWISE_BINARY(minimum, minimum_table)

#if NUMC_HAVE_AVX2 || NUMC_HAVE_SVE || NUMC_HAVE_NEON || NUMC_HAVE_RVV
#define DEFINE_CMP_WITH_SIMD(NAME, TABLE, SIMD_FN)                          \
  int numc_##NAME(const NumcArray *a, const NumcArray *b, NumcArray *out) { \
    int err = _check_binary(a, b, out);                                     \
    if (err)                                                                \
      return err;                                                           \
    if (a->dtype == NUMC_DTYPE_UINT8 && a->is_contiguous &&                 \
        b->is_contiguous && out->is_contiguous && a->dim == b->dim) {       \
      bool same_shape = true;                                               \
      for (size_t d = 0; d < a->dim; d++)                                   \
        if (a->shape[d] != b->shape[d]) {                                   \
          same_shape = false;                                               \
          break;                                                            \
        }                                                                   \
      if (same_shape) {                                                     \
        SIMD_FN((const uint8_t *)a->data, (const uint8_t *)b->data,         \
                (uint8_t *)out->data, a->size);                             \
        return 0;                                                           \
      }                                                                     \
    }                                                                       \
    _binary_op(a, b, out, TABLE);                                           \
    return 0;                                                               \
  }
#if NUMC_HAVE_AVX2
DEFINE_CMP_WITH_SIMD(eq, eq_table, _cmp_eq_u8_avx2)
DEFINE_CMP_WITH_SIMD(gt, gt_table, _cmp_gt_u8_avx2)
DEFINE_CMP_WITH_SIMD(lt, lt_table, _cmp_lt_u8_avx2)
DEFINE_CMP_WITH_SIMD(ge, ge_table, _cmp_ge_u8_avx2)
DEFINE_CMP_WITH_SIMD(le, le_table, _cmp_le_u8_avx2)
#elif NUMC_HAVE_SVE
DEFINE_CMP_WITH_SIMD(eq, eq_table, _cmp_eq_u8_sve)
DEFINE_CMP_WITH_SIMD(gt, gt_table, _cmp_gt_u8_sve)
DEFINE_CMP_WITH_SIMD(lt, lt_table, _cmp_lt_u8_sve)
DEFINE_CMP_WITH_SIMD(ge, ge_table, _cmp_ge_u8_sve)
DEFINE_CMP_WITH_SIMD(le, le_table, _cmp_le_u8_sve)
#elif NUMC_HAVE_NEON
DEFINE_CMP_WITH_SIMD(eq, eq_table, _cmp_eq_u8_neon)
DEFINE_CMP_WITH_SIMD(gt, gt_table, _cmp_gt_u8_neon)
DEFINE_CMP_WITH_SIMD(lt, lt_table, _cmp_lt_u8_neon)
DEFINE_CMP_WITH_SIMD(ge, ge_table, _cmp_ge_u8_neon)
DEFINE_CMP_WITH_SIMD(le, le_table, _cmp_le_u8_neon)
#elif NUMC_HAVE_RVV
DEFINE_CMP_WITH_SIMD(eq, eq_table, _cmp_eq_u8_rvv)
DEFINE_CMP_WITH_SIMD(gt, gt_table, _cmp_gt_u8_rvv)
DEFINE_CMP_WITH_SIMD(lt, lt_table, _cmp_lt_u8_rvv)
DEFINE_CMP_WITH_SIMD(ge, ge_table, _cmp_ge_u8_rvv)
DEFINE_CMP_WITH_SIMD(le, le_table, _cmp_le_u8_rvv)
#endif
#undef DEFINE_CMP_WITH_SIMD
#else
DEFINE_ELEMWISE_BINARY(eq, eq_table)
DEFINE_ELEMWISE_BINARY(gt, gt_table)
DEFINE_ELEMWISE_BINARY(lt, lt_table)
DEFINE_ELEMWISE_BINARY(ge, ge_table)
DEFINE_ELEMWISE_BINARY(le, le_table)
#endif

DEFINE_ELEMWISE_SCALAR(eq, eq_table)
DEFINE_ELEMWISE_SCALAR(gt, gt_table)
DEFINE_ELEMWISE_SCALAR(lt, lt_table)
DEFINE_ELEMWISE_SCALAR(ge, ge_table)
DEFINE_ELEMWISE_SCALAR(le, le_table)
