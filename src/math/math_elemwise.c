#include "array/_array_core.h"
#include "array/array_dtype.h"
#include "math/math.h"
#include "math_helper.h"
#include "numc_error.h"

#define _OP_FNAME(op, type) _array_##op##_##type
#define OP_FNAME(op, type) _OP_FNAME(op, type)

#define GENERATE_OP_DIV_INT8_INT16(TYPE_ENUM, NUMC_TYPE)                       \
  static inline void OP_FNAME(OP_NAME, TYPE_ENUM)(                             \
      void *a, void *b, void *out, size_t n, size_t elem_size) {               \
    NUMC_TYPE *restrict pa = __builtin_assume_aligned(a, NUMC_SIMD_ALIGN);     \
    NUMC_TYPE *restrict pb = __builtin_assume_aligned(b, NUMC_SIMD_ALIGN);     \
    NUMC_TYPE *restrict pout = __builtin_assume_aligned(out, NUMC_SIMD_ALIGN); \
    NUMC_OMP_FOR(                                                              \
        n, elem_size, for (size_t i = 0; i < n; i++) {                         \
          pout[i] = (NUMC_TYPE)((float)pa[i] / (float)pb[i]);                  \
        });                                                                    \
  }

#define GENERATE_OP_DIV_INT32(TYPE_ENUM, NUMC_TYPE)                            \
  static inline void OP_FNAME(OP_NAME, TYPE_ENUM)(                             \
      void *a, void *b, void *out, size_t n, size_t elem_size) {               \
    NUMC_TYPE *restrict pa = __builtin_assume_aligned(a, NUMC_SIMD_ALIGN);     \
    NUMC_TYPE *restrict pb = __builtin_assume_aligned(b, NUMC_SIMD_ALIGN);     \
    NUMC_TYPE *restrict pout = __builtin_assume_aligned(out, NUMC_SIMD_ALIGN); \
    NUMC_OMP_FOR(                                                              \
        n, elem_size, for (size_t i = 0; i < n; i++) {                         \
          pout[i] = (NUMC_TYPE)((double)pa[i] / (double)pb[i]);                \
        });                                                                    \
  }

#define GENERATE_OP(TYPE_ENUM, NUMC_TYPE)                                      \
  static inline void OP_FNAME(OP_NAME, TYPE_ENUM)(                             \
      void *a, void *b, void *out, size_t n, size_t elem_size) {               \
    NUMC_TYPE *restrict pa = __builtin_assume_aligned(a, NUMC_SIMD_ALIGN);     \
    NUMC_TYPE *restrict pb = __builtin_assume_aligned(b, NUMC_SIMD_ALIGN);     \
    NUMC_TYPE *restrict pout = __builtin_assume_aligned(out, NUMC_SIMD_ALIGN); \
    NUMC_OMP_FOR(                                                              \
        n, elem_size,                                                          \
        for (size_t i = 0; i < n; i++) { pout[i] = pa[i] OP_SYMBOL pb[i]; });  \
  }

#define GENERATE_OP_ENTRY(TYPE_ENUM, NUMC_TYPE)                                \
  [TYPE_ENUM] = OP_FNAME(OP_NAME, TYPE_ENUM),

typedef void (*ArrayOpFunc)(void *, void *, void *, size_t, size_t);

/* --- add --- */
#define OP_NAME add
#define OP_SYMBOL +
GENERATE_NUMC_TYPES(GENERATE_OP)
static const ArrayOpFunc _array_add_table[] = {
    GENERATE_NUMC_TYPES(GENERATE_OP_ENTRY)};
#undef OP_NAME
#undef OP_SYMBOL

/* --- sub --- */
#define OP_NAME sub
#define OP_SYMBOL -
GENERATE_NUMC_TYPES(GENERATE_OP)
static const ArrayOpFunc _array_sub_table[] = {
    GENERATE_NUMC_TYPES(GENERATE_OP_ENTRY)};
#undef OP_NAME
#undef OP_SYMBOL

/* --- mul --- */
#define OP_NAME mul
#define OP_SYMBOL *
GENERATE_NUMC_TYPES(GENERATE_OP)
static const ArrayOpFunc _array_mul_table[] = {
    GENERATE_NUMC_TYPES(GENERATE_OP_ENTRY)};
#undef OP_NAME
#undef OP_SYMBOL

/* --- div (int8/int16 → float, int32 → double, rest native) --- */
#define OP_NAME div
#define OP_SYMBOL /
GENERATE_INT8_INT16_NUMC_TYPES(GENERATE_OP_DIV_INT8_INT16)
GENERATE_INT32(GENERATE_OP_DIV_INT32)
GENERATE_64BIT_NUMC_TYPES(GENERATE_OP)
GENERATE_OP(NUMC_DTYPE_FLOAT32, NUMC_FLOAT32)
static const ArrayOpFunc _array_div_table[] = {
    GENERATE_NUMC_TYPES(GENERATE_OP_ENTRY)};
#undef OP_NAME
#undef OP_SYMBOL

#undef GENERATE_OP
#undef GENERATE_OP_ENTRY

static inline int _array_elemwise_check(const NumcArray *a,
                                        const NumcArray *b) {
  if (!a || !b)
    return NUMC_ERR_NULL;

  if (a->dim != b->dim) {
    numc_set_error(
        NUMC_ERR_DIM,
        "numc: array_add: arrays must have the same dimensions, you "
        "may want to use array_transpose_copy or array_slice for array b");
    return NUMC_ERR_DIM;
  }

  if (!a->is_contiguous) {
    numc_set_error(NUMC_ERR_CONTIGUOUS,
                   "numc: array_add: array a must be contiguous, you may want "
                   "to use array_as_contiguous for array a");
    return NUMC_ERR_CONTIGUOUS;
  }

  if (!b->is_contiguous) {
    numc_set_error(NUMC_ERR_CONTIGUOUS,
                   "numc: array_add: array b must be contiguous, you may want "
                   "to use array_as_contiguous for array b");
    return NUMC_ERR_CONTIGUOUS;
  }

  return 0;
}

int array_add(const NumcArray *a, const NumcArray *b, NumcArray *out) {
  int err = _array_elemwise_check(a, b);
  if (err < 0)
    return err;

  _array_add_table[a->dtype](a->data, b->data, out->data, a->size,
                             a->elem_size);
  return 0;
}

int array_sub(const NumcArray *a, const NumcArray *b, NumcArray *out) {
  int err = _array_elemwise_check(a, b);
  if (err < 0)
    return err;

  _array_sub_table[a->dtype](a->data, b->data, out->data, a->size,
                             a->elem_size);
  return 0;
}

int array_mul(const NumcArray *a, const NumcArray *b, NumcArray *out) {
  int err = _array_elemwise_check(a, b);
  if (err < 0)
    return err;

  _array_mul_table[a->dtype](a->data, b->data, out->data, a->size,
                             a->elem_size);
  return 0;
}

int array_div(const NumcArray *a, const NumcArray *b, NumcArray *out) {
  int err = _array_elemwise_check(a, b);
  if (err < 0)
    return err;

  _array_div_table[a->dtype](a->data, b->data, out->data, a->size,
                             a->elem_size);
  return 0;
}
