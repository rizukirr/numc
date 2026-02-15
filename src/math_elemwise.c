#include "internal.h"
#include "numc/dtype.h"
#include <numc/math.h>
#include <string.h>

/*
 * Stride-aware binary kernels
 *
 * Inspired by NumPy's BINARY_LOOP_FAST (fast_loop_macros.h).
 * Each typed kernel has three runtime paths:
 *
 *   PATH 1 — Contiguous:  sa == sb == so == sizeof(T)
 *            Tight indexed loop, auto-vectorizes with -O3 -march=native.
 *
 *   PATH 2 — Scalar broadcast:  sb == 0, a and out contiguous
 *            Reads b once, applies to all elements. Used for scalar ops.
 *
 *   PATH 3 — Generic strided:  arbitrary sa, sb, so
 *            Handles views, slices, transposes via pointer arithmetic.
 * */

typedef void (*NumcBinaryKernel)(const char *a, const char *b, char *out,
                                 size_t n, intptr_t sa, intptr_t sb,
                                 intptr_t so);

#define DEFINE_BINARY_KERNEL(OP_NAME, TYPE_ENUM, C_TYPE, EXPR)                 \
  static void _kern_##OP_NAME##_##TYPE_ENUM(const char *a, const char *b,      \
                                            char *out, size_t n, intptr_t sa,  \
                                            intptr_t sb, intptr_t so) {        \
    const intptr_t es = (intptr_t)sizeof(C_TYPE);                              \
    if (sa == es && sb == es && so == es) {                                    \
      /* PATH 1: all contiguous */                                             \
      const C_TYPE *restrict pa = (const C_TYPE *)a;                           \
      const C_TYPE *restrict pb = (const C_TYPE *)b;                           \
      C_TYPE *restrict po = (C_TYPE *)out;                                     \
      NUMC_OMP_FOR(                                                            \
          n, sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {                  \
            C_TYPE in1 = pa[i];                                                \
            C_TYPE in2 = pb[i];                                                \
            po[i] = (EXPR);                                                    \
          });                                                                  \
    } else if (sb == 0 && sa == es && so == es) {                              \
      /* PATH 2: scalar broadcast */                                           \
      const C_TYPE in2 = *(const C_TYPE *)b;                                   \
      if (a == out) {                                                          \
        /* PATH 2a: inplace — single pointer, no aliasing check */             \
        C_TYPE *restrict p = (C_TYPE *)out;                                    \
        NUMC_OMP_FOR(                                                          \
            n, sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {                \
              C_TYPE in1 = p[i];                                               \
              p[i] = (EXPR);                                                   \
            });                                                                \
      } else {                                                                 \
        /* PATH 2b: separate src/dst */                                        \
        const C_TYPE *restrict pa = (const C_TYPE *)a;                         \
        C_TYPE *restrict po = (C_TYPE *)out;                                   \
        NUMC_OMP_FOR(                                                          \
            n, sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {                \
              C_TYPE in1 = pa[i];                                              \
              po[i] = (EXPR);                                                  \
            });                                                                \
      }                                                                        \
    } else {                                                                   \
      /* PATH 3: generic strided */                                            \
      for (size_t i = 0; i < n; i++) {                                         \
        C_TYPE in1 = *(const C_TYPE *)(a + i * sa);                            \
        C_TYPE in2 = *(const C_TYPE *)(b + i * sb);                            \
        *(C_TYPE *)(out + i * so) = (EXPR);                                    \
      }                                                                        \
    }                                                                          \
  }

/* ── Stamp out typed kernels ────────────────────────────────────────── */

/* add: all 10 types, native + */
#define STAMP_ADD(TE, CT) DEFINE_BINARY_KERNEL(add, TE, CT, in1 + in2)
GENERATE_NUMC_TYPES(STAMP_ADD)
#undef STAMP_ADD

/* sub: all 10 types, native - */
#define STAMP_SUB(TE, CT) DEFINE_BINARY_KERNEL(sub, TE, CT, in1 - in2)
GENERATE_NUMC_TYPES(STAMP_SUB)
#undef STAMP_SUB

/* mul: all 10 types, native * */
#define STAMP_MUL(TE, CT) DEFINE_BINARY_KERNEL(mul, TE, CT, in1 *in2)
GENERATE_NUMC_TYPES(STAMP_MUL)
#undef STAMP_MUL

/* div: int8/int16 → cast through float, int32 → through double, rest native */
#define STAMP_DIV_SMALL(TE, CT)                                                \
  DEFINE_BINARY_KERNEL(div, TE, CT, (CT)((float)in1 / (float)in2))
GENERATE_INT8_INT16_NUMC_TYPES(STAMP_DIV_SMALL)
#undef STAMP_DIV_SMALL

#define STAMP_DIV_I32(TE, CT)                                                  \
  DEFINE_BINARY_KERNEL(div, TE, CT, (CT)((double)in1 / (double)in2))
GENERATE_INT32(STAMP_DIV_I32)
#undef STAMP_DIV_I32

#define STAMP_DIV_NATIVE(TE, CT) DEFINE_BINARY_KERNEL(div, TE, CT, in1 / in2)
GENERATE_64BIT_NUMC_TYPES(STAMP_DIV_NATIVE)
DEFINE_BINARY_KERNEL(div, NUMC_DTYPE_FLOAT32, float, in1 / in2)
#undef STAMP_DIV_NATIVE

/* ── Dispatch tables (dtype → kernel) ─────────────────────────────── */

#define E(OP, TE) [TE] = _kern_##OP##_##TE

static const NumcBinaryKernel _add_table[] = {
    E(add, NUMC_DTYPE_INT8),    E(add, NUMC_DTYPE_INT16),
    E(add, NUMC_DTYPE_INT32),   E(add, NUMC_DTYPE_INT64),
    E(add, NUMC_DTYPE_UINT8),   E(add, NUMC_DTYPE_UINT16),
    E(add, NUMC_DTYPE_UINT32),  E(add, NUMC_DTYPE_UINT64),
    E(add, NUMC_DTYPE_FLOAT32), E(add, NUMC_DTYPE_FLOAT64),
};

static const NumcBinaryKernel _sub_table[] = {
    E(sub, NUMC_DTYPE_INT8),    E(sub, NUMC_DTYPE_INT16),
    E(sub, NUMC_DTYPE_INT32),   E(sub, NUMC_DTYPE_INT64),
    E(sub, NUMC_DTYPE_UINT8),   E(sub, NUMC_DTYPE_UINT16),
    E(sub, NUMC_DTYPE_UINT32),  E(sub, NUMC_DTYPE_UINT64),
    E(sub, NUMC_DTYPE_FLOAT32), E(sub, NUMC_DTYPE_FLOAT64),
};

static const NumcBinaryKernel _mul_table[] = {
    E(mul, NUMC_DTYPE_INT8),    E(mul, NUMC_DTYPE_INT16),
    E(mul, NUMC_DTYPE_INT32),   E(mul, NUMC_DTYPE_INT64),
    E(mul, NUMC_DTYPE_UINT8),   E(mul, NUMC_DTYPE_UINT16),
    E(mul, NUMC_DTYPE_UINT32),  E(mul, NUMC_DTYPE_UINT64),
    E(mul, NUMC_DTYPE_FLOAT32), E(mul, NUMC_DTYPE_FLOAT64),
};

static const NumcBinaryKernel _div_table[] = {
    E(div, NUMC_DTYPE_INT8),    E(div, NUMC_DTYPE_INT16),
    E(div, NUMC_DTYPE_INT32),   E(div, NUMC_DTYPE_INT64),
    E(div, NUMC_DTYPE_UINT8),   E(div, NUMC_DTYPE_UINT16),
    E(div, NUMC_DTYPE_UINT32),  E(div, NUMC_DTYPE_UINT64),
    E(div, NUMC_DTYPE_FLOAT32), E(div, NUMC_DTYPE_FLOAT64),
};

#undef E

/*
 * ND iteration — recursive, calls kernel on innermost dimension.
 * Outer dimensions loop to compute base pointers.
 * Max recursion depth = NUMC_MAX_DIMENSIONS (8).
 *
 * For contiguous arrays this is never called — the flat fast path
 * in _binary_op handles it directly.
 * */

static void _elemwise_nd(NumcBinaryKernel kern, const char *a, const size_t *sa,
                         const char *b, const size_t *sb, char *out,
                         const size_t *so, const size_t *shape, size_t ndim) {
  if (ndim == 1) {
    kern(a, b, out, shape[0], (intptr_t)sa[0], (intptr_t)sb[0],
         (intptr_t)so[0]);
    return;
  }

  for (size_t i = 0; i < shape[0]; i++) {
    _elemwise_nd(kern, a + i * sa[0], sa + 1, b + i * sb[0], sb + 1,
                 out + i * so[0], so + 1, shape + 1, ndim - 1);
  }
}

/* ── Validation ───────────────────────────────────────────────────── */

static int _check_binary(const struct NumcArray *a, const struct NumcArray *b,
                         const struct NumcArray *out) {
  if (!a || !b || !out)
    return NUMC_ERR_NULL;
  if (a->dtype != b->dtype || a->dtype != out->dtype)
    return NUMC_ERR_TYPE;
  if (a->dim != b->dim || a->dim != out->dim)
    return NUMC_ERR_SHAPE;
  for (size_t d = 0; d < a->dim; d++)
    if (a->shape[d] != b->shape[d] || a->shape[d] != out->shape[d])
      return NUMC_ERR_SHAPE;
  return 0;
}

/* ── Binary op dispatch ───────────────────────────────────────────── */

static void _binary_op(const struct NumcArray *a, const struct NumcArray *b,
                       struct NumcArray *out, const NumcBinaryKernel *table) {
  NumcBinaryKernel kern = table[a->dtype];
  intptr_t es = (intptr_t)a->elem_size;

  if (a->is_contiguous && b->is_contiguous && out->is_contiguous) {
    /* All contiguous: single flat kernel call — fastest path */
    kern((const char *)a->data, (const char *)b->data, (char *)out->data,
         a->size, es, es, es);
  } else {
    /* ND iteration: recurse over outer dims, kernel on inner dim */
    _elemwise_nd(kern, (const char *)a->data, a->strides, (const char *)b->data,
                 b->strides, (char *)out->data, out->strides, a->shape, a->dim);
  }
}

/* ── Scalar Conversion ──────────────────────────────────────────────── */
static void _double_to_dtype(double value, NumcDType dtype,
                             char buf[static 8]) {
  memset(buf, 0, 8);

  switch (dtype) {
  case NUMC_DTYPE_INT8:
    *(int8_t *)buf = (int8_t)value;
    break;
  case NUMC_DTYPE_INT16:
    *(int16_t *)buf = (int16_t)value;
    break;
  case NUMC_DTYPE_INT32:
    *(int32_t *)buf = (int32_t)value;
    break;
  case NUMC_DTYPE_INT64:
    *(int64_t *)buf = (int64_t)value;
    break;
  case NUMC_DTYPE_UINT8:
    *(uint8_t *)buf = (uint8_t)value;
    break;
  case NUMC_DTYPE_UINT16:
    *(uint16_t *)buf = (uint16_t)value;
    break;
  case NUMC_DTYPE_UINT32:
    *(uint32_t *)buf = (uint32_t)value;
    break;
  case NUMC_DTYPE_UINT64:
    *(uint64_t *)buf = (uint64_t)value;
    break;
  case NUMC_DTYPE_FLOAT32:
    *(float *)buf = (float)value;
    break;
  case NUMC_DTYPE_FLOAT64:
    *(double *)buf = (double)value;
    break;
  default:
    break;
  }
}

/* ── Scalar op dispatch ───────────────────────────────────────────── */

static void _scalar_op(const struct NumcArray *a, const char *scalar_buf,
                       struct NumcArray *out, const NumcBinaryKernel *table) {
  NumcBinaryKernel kern = table[a->dtype];

  if (a->is_contiguous && out->is_contiguous) {
    /* Flat fast path: sa = es, sb = 0, so = es → hits kernel PATH 2 */
    intptr_t es = (intptr_t)a->elem_size;
    kern((const char *)a->data, scalar_buf, (char *)out->data, a->size, es, 0,
         es);
  } else {
    /* ND iteration with zero strides for the scalar.
     * We build a fake strides array of all-zeros so _elemwise_nd
     * passes sb=0 at every recursion level. */
    size_t zero_strides[NUMC_MAX_DIMENSIONS] = {0};
    _elemwise_nd(kern, (const char *)a->data, a->strides, scalar_buf,
                 zero_strides, (char *)out->data, out->strides, a->shape,
                 a->dim);
  }
}

static int _scalar_op_inplace(NumcArray *a, double scalar,
                              const NumcBinaryKernel *table) {
  if (!a)
    return NUMC_ERR_NULL;

  char buf[8];
  _double_to_dtype(scalar, a->dtype, buf);
  NumcBinaryKernel kern = table[a->dtype];

  if (a->is_contiguous) {
    intptr_t es = (intptr_t)a->elem_size;
    kern((const char *)a->data, buf, (char *)a->data, a->size, es, 0, es);
  } else {
    size_t zero_strides[NUMC_MAX_DIMENSIONS] = {0};
    _elemwise_nd(kern, (const char *)a->data, a->strides, buf, zero_strides,
                 (char *)a->data, a->strides, a->shape, a->dim);
  }
  return 0;
}

/* ── Scalar Validation ──────────────────────────────────────────────── */

static int _check_scalar(const struct NumcArray *a,
                         const struct NumcArray *out) {
  if (!a || !out)
    return NUMC_ERR_NULL;
  if (a->dtype != out->dtype)
    return NUMC_ERR_TYPE;
  if (a->dim != out->dim)
    return NUMC_ERR_SHAPE;
  for (size_t d = 0; d < a->dim; d++)
    if (a->shape[d] != out->shape[d])
      return NUMC_ERR_SHAPE;
  return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Public API
 * ═══════════════════════════════════════════════════════════════════════ */

/* ── Element-wise binary ops
 * ──────────────────────────────────────────────────── */

int numc_add(const NumcArray *a, const NumcArray *b, NumcArray *out) {
  int err = _check_binary(a, b, out);
  if (err)
    return err;
  _binary_op(a, b, out, _add_table);
  return 0;
}

int numc_sub(const NumcArray *a, const NumcArray *b, NumcArray *out) {
  int err = _check_binary(a, b, out);
  if (err)
    return err;
  _binary_op(a, b, out, _sub_table);
  return 0;
}

int numc_mul(const NumcArray *a, const NumcArray *b, NumcArray *out) {
  int err = _check_binary(a, b, out);
  if (err)
    return err;
  _binary_op(a, b, out, _mul_table);
  return 0;
}

int numc_div(const NumcArray *a, const NumcArray *b, NumcArray *out) {
  int err = _check_binary(a, b, out);
  if (err)
    return err;
  _binary_op(a, b, out, _div_table);
  return 0;
}

/* ── Element-wise scalar ops
 * ──────────────────────────────────────────────────── */

int numc_add_scalar(const NumcArray *a, double scalar, NumcArray *out) {
  int err = _check_scalar(a, out);
  if (err)
    return err;
  char buf[8];
  _double_to_dtype(scalar, a->dtype, buf);
  _scalar_op(a, buf, out, _add_table);
  return 0;
}

int numc_sub_scalar(const NumcArray *a, double scalar, NumcArray *out) {
  int err = _check_scalar(a, out);
  if (err)
    return err;
  char buf[8];
  _double_to_dtype(scalar, a->dtype, buf);
  _scalar_op(a, buf, out, _sub_table);
  return 0;
}

int numc_mul_scalar(const NumcArray *a, double scalar, NumcArray *out) {
  int err = _check_scalar(a, out);
  if (err)
    return err;
  char buf[8];
  _double_to_dtype(scalar, a->dtype, buf);
  _scalar_op(a, buf, out, _mul_table);
  return 0;
}

int numc_div_scalar(const NumcArray *a, double scalar, NumcArray *out) {
  int err = _check_scalar(a, out);
  if (err)
    return err;
  char buf[8];
  _double_to_dtype(scalar, a->dtype, buf);
  _scalar_op(a, buf, out, _div_table);
  return 0;
}

int numc_add_scalar_inplace(NumcArray *a, double scalar) {
  return _scalar_op_inplace(a, scalar, _add_table);
}
int numc_sub_scalar_inplace(NumcArray *a, double scalar) {
  return _scalar_op_inplace(a, scalar, _sub_table);
}
int numc_mul_scalar_inplace(NumcArray *a, double scalar) {
  return _scalar_op_inplace(a, scalar, _mul_table);
}
int numc_div_scalar_inplace(NumcArray *a, double scalar) {
  return _scalar_op_inplace(a, scalar, _div_table);
}
