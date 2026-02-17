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

/* Stride-aware unary kernels
 * Inspired by NumPy's UNARY_LOOP_FAST (fast_loop_macros.h).
 * Each typed kernel has three runtime paths:
 *
 *   PATH 1 — Contiguous:  sa == es, so == es
 *            Tight indexed loop, auto-vectorizes with -O3 -march=native.
 *
 *   PATH 2 — In-place:    a == out
 *            Reads a once, applies to all elements. Used for scalar ops.
 *
 *   PATH 3 — Generic strided:  arbitrary sa, so
 *            Handles views, slices, transposes via pointer arithmetic.
 *
 * */

typedef void (*NumcUnaryKernel)(const char *a, char *out, size_t n, intptr_t sa,
                                intptr_t so);

#define DEFINE_UNARY_KERNEL(OP_NAME, TYPE_ENUM, C_TYPE, EXPR)                  \
  static void _kern_##OP_NAME##_##TYPE_ENUM(                                   \
      const char *a, char *out, size_t n, intptr_t sa, intptr_t so) {          \
    const intptr_t es = (intptr_t)sizeof(C_TYPE);                              \
    if (sa == es && so == es && a != out) {                                    \
      /* PATH 1: contiguous, distinct buffers — restrict is valid */           \
      const C_TYPE *restrict pa = (const C_TYPE *)a;                           \
      C_TYPE *restrict po = (C_TYPE *)out;                                     \
      NUMC_OMP_FOR(                                                            \
          n, sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {                  \
            C_TYPE in1 = pa[i];                                                \
            po[i] = (EXPR);                                                    \
          });                                                                  \
    } else if (sa == es && so == es) {                                         \
      /* PATH 2: contiguous inplace (a == out) — no restrict to avoid UB */    \
      C_TYPE *p = (C_TYPE *)a;                                                 \
      NUMC_OMP_FOR(                                                            \
          n, sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {                  \
            C_TYPE in1 = p[i];                                                 \
            p[i] = (EXPR);                                                     \
          });                                                                  \
    } else {                                                                   \
      /* PATH 2: generic strided */                                            \
      for (size_t i = 0; i < n; i++) {                                         \
        C_TYPE in1 = *(const C_TYPE *)(a + i * sa);                            \
        *(C_TYPE *)(out + i * so) = (EXPR);                                    \
      }                                                                        \
    }                                                                          \
  }

/* ── Accurate scalar log helpers (fdlibm, argument reduction + Horner) ── */
/*
 * Algorithm (same as fdlibm / musl libm):
 *   1. Argument reduction: x = 2^k * m, m in [sqrt(2)/2, sqrt(2))
 *   2. f = m - 1  (f in [-0.293, 0.414])
 *   3. s = f/(2+f)  (s in [-0.172, 0.172] — keeps polynomial convergent)
 *   4. log(m) = f - hfsq + s*(hfsq + R)  where R is Horner polynomial in s²
 *   5. log(x) = k*ln2 + log(m)
 *
 * float32: 4 Horner coefficients, max error < 0.5 ULP
 * float64: 7 Horner coefficients, max error < 1 ULP
 */

static float _log_NUMC_DTYPE_FLOAT32(float x) {
  static const float
      ln2 = 6.9314718056e-01f,
      Lg1 = 6.6666668653e-01f, /* 0x3F2AAAAB */
      Lg2 = 4.0000004172e-01f, /* 0x3ECCCCCD */
      Lg3 = 2.8571429849e-01f, /* 0x3E924925 */
      Lg4 = 2.2222198009e-01f; /* 0x3E638E29 */

  if (x <= 0.0f)
    return 0.0f;

  /* Argument reduction: x = 2^k * m, m in [sqrt(2)/2, sqrt(2)) */
  uint32_t ix;
  memcpy(&ix, &x, sizeof ix);
  int k = (int)((ix >> 23) & 0xffu) - 127;
  ix    = (ix & 0x007fffffu) | 0x3f800000u; /* set biased exponent to 0 */
  float m;
  memcpy(&m, &ix, sizeof m);
  if (m > 1.41421356f) { m *= 0.5f; k++; }
  float f = m - 1.0f;

  float s    = f / (2.0f + f);
  float z    = s * s;
  float w    = z * z;
  float t1   = w * (Lg2 + w * Lg4);
  float t2   = z * (Lg1 + w * Lg3);
  float R    = t1 + t2;
  float hfsq = 0.5f * f * f;
  return (float)k * ln2 + f - hfsq + s * (hfsq + R);
}

static double _log_NUMC_DTYPE_FLOAT64(double x) {
  static const double
      ln2 = 6.9314718055994530942e-01,
      Lg1 = 6.6666666666666735130e-01, /* 0x3FE5555555555593 */
      Lg2 = 3.9999999999940941908e-01, /* 0x3FD999999997FA04 */
      Lg3 = 2.8571428743662391490e-01, /* 0x3FD2492494229359 */
      Lg4 = 2.2221984321497839600e-01, /* 0x3FCC71C51D8E78AF */
      Lg5 = 1.8183572161618050120e-01, /* 0x3FC7466496CB03DE */
      Lg6 = 1.5313837699209373320e-01, /* 0x3FC39A09D078C69F */
      Lg7 = 1.4798198605116585910e-01; /* 0x3FC2F112DF3E5244 */

  if (x <= 0.0)
    return 0.0;

  /* Argument reduction: x = 2^k * m, m in [sqrt(2)/2, sqrt(2)) */
  uint64_t ix;
  memcpy(&ix, &x, sizeof ix);
  int k = (int)((ix >> 52) & 0x7ffULL) - 1023;
  ix    = (ix & 0x000fffffffffffffULL) | 0x3ff0000000000000ULL;
  double m;
  memcpy(&m, &ix, sizeof m);
  if (m > 1.4142135623730951) { m *= 0.5; k++; }
  double f = m - 1.0;

  double s    = f / (2.0 + f);
  double z    = s * s;
  double w    = z * z;
  double t1   = w * (Lg2 + w * (Lg4 + w * Lg6));
  double t2   = z * (Lg1 + w * (Lg3 + w * (Lg5 + w * Lg7)));
  double R    = t1 + t2;
  double hfsq = 0.5 * f * f;
  return (double)k * ln2 + f - hfsq + s * (hfsq + R);
}

/* ── Stamp out log loop kernels (stride-aware, wrapping scalar bit-manip) ── */

/* < 32-bit integers: cast through float */
#define STAMP_LOG_SMALL(TE, CT)                                                \
  DEFINE_UNARY_KERNEL(log, TE, CT, (CT)_log_NUMC_DTYPE_FLOAT32((float)in1))
GENERATE_INT8_INT16_NUMC_TYPES(STAMP_LOG_SMALL)
#undef STAMP_LOG_SMALL

/* 32-bit integers: cast through double */
#define STAMP_LOG_I32(TE, CT)                                                  \
  DEFINE_UNARY_KERNEL(log, TE, CT, (CT)_log_NUMC_DTYPE_FLOAT64((double)in1))
GENERATE_INT32(STAMP_LOG_I32)
#undef STAMP_LOG_I32

/* 64-bit integers: cast through double */
#define STAMP_LOG_I64(TE, CT)                                                  \
  DEFINE_UNARY_KERNEL(log, TE, CT, (CT)_log_NUMC_DTYPE_FLOAT64((double)in1))
GENERATE_SIGNED_64BIT_NUMC_TYPES(STAMP_LOG_I64)
#undef STAMP_LOG_I64
DEFINE_UNARY_KERNEL(log, NUMC_DTYPE_UINT64, uint64_t,
                    (uint64_t)_log_NUMC_DTYPE_FLOAT64((double)in1))

/* float types: call their own bit-manipulation helpers directly */
DEFINE_UNARY_KERNEL(log, NUMC_DTYPE_FLOAT32, float, _log_NUMC_DTYPE_FLOAT32(in1))
DEFINE_UNARY_KERNEL(log, NUMC_DTYPE_FLOAT64, double, _log_NUMC_DTYPE_FLOAT64(in1))

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

/* neg: all 10 types, native - */
#define STAMP_NEG(TE, CT) DEFINE_UNARY_KERNEL(neg, TE, CT, -in1)
GENERATE_NUMC_TYPES(STAMP_NEG)
#undef STAMP_NEG

#define STAMP_FABS(TE, CT)                                                     \
  DEFINE_UNARY_KERNEL(fabs, TE, CT, (CT)(in1 < 0.0 ? -in1 : in1))
GENERATE_FLOAT64_NUMC_TYPES(STAMP_FABS)
#undef STAMP_FABS

#define STAMP_FABSF(TE, CT)                                                    \
  DEFINE_UNARY_KERNEL(fabsf, TE, CT, (CT)(in1 < 0.0f ? -in1 : in1))
GENERATE_FLOAT32_NUMC_TYPES(STAMP_FABSF)
#undef STAMP_FABSF

/* conditional expression: cleaner pattern for vectorizer → vpabsb/vpabsw/vpabsd
 */
#define STAMP_ABS(TE, CT)                                                      \
  DEFINE_UNARY_KERNEL(abs, TE, CT, (CT)(in1 < 0 ? -in1 : in1))
GENERATE_SIGNED_INT8_INT16_INT32_NUMC_TYPES(STAMP_ABS)
#undef STAMP_ABS

/* int64: AVX2 has no VPABSQ — conditional emits compare+negate+blend sequence
 */
#define STANP_LLABS(TE, CT)                                                    \
  DEFINE_UNARY_KERNEL(llabs, TE, CT, (CT)(in1 < 0 ? -in1 : in1))
GENERATE_SIGNED_64BIT_NUMC_TYPES(STANP_LLABS)
#undef STANP_LLABS

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

static const NumcUnaryKernel _neg_table[] = {
    E(neg, NUMC_DTYPE_INT8),    E(neg, NUMC_DTYPE_INT16),
    E(neg, NUMC_DTYPE_INT32),   E(neg, NUMC_DTYPE_INT64),
    E(neg, NUMC_DTYPE_UINT8),   E(neg, NUMC_DTYPE_UINT16),
    E(neg, NUMC_DTYPE_UINT32),  E(neg, NUMC_DTYPE_UINT64),
    E(neg, NUMC_DTYPE_FLOAT32), E(neg, NUMC_DTYPE_FLOAT64),
};

static const NumcUnaryKernel _abs_table[] = {
    E(abs, NUMC_DTYPE_INT8),      E(abs, NUMC_DTYPE_INT16),
    E(abs, NUMC_DTYPE_INT32),     E(fabs, NUMC_DTYPE_FLOAT64),
    E(fabsf, NUMC_DTYPE_FLOAT32), E(llabs, NUMC_DTYPE_INT64),
};

static const NumcUnaryKernel _log_table[] = {
    E(log, NUMC_DTYPE_INT8),    E(log, NUMC_DTYPE_INT16),
    E(log, NUMC_DTYPE_INT32),   E(log, NUMC_DTYPE_INT64),
    E(log, NUMC_DTYPE_UINT8),   E(log, NUMC_DTYPE_UINT16),
    E(log, NUMC_DTYPE_UINT32),  E(log, NUMC_DTYPE_UINT64),
    E(log, NUMC_DTYPE_FLOAT32), E(log, NUMC_DTYPE_FLOAT64),
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

static void _elemwise_binary_nd(NumcBinaryKernel kern, const char *a,
                                const size_t *sa, const char *b,
                                const size_t *sb, char *out, const size_t *so,
                                const size_t *shape, size_t ndim) {
  if (ndim == 1) {
    kern(a, b, out, shape[0], (intptr_t)sa[0], (intptr_t)sb[0],
         (intptr_t)so[0]);
    return;
  }

  for (size_t i = 0; i < shape[0]; i++) {
    _elemwise_binary_nd(kern, a + i * sa[0], sa + 1, b + i * sb[0], sb + 1,
                        out + i * so[0], so + 1, shape + 1, ndim - 1);
  }
}

static void _elemwise_unary_nd(NumcUnaryKernel kern, const char *a,
                               const size_t *sa, char *out, const size_t *so,
                               const size_t *shape, size_t ndim) {
  if (ndim == 1) {
    kern(a, out, shape[0], (intptr_t)sa[0], (intptr_t)so[0]);
    return;
  }

  for (size_t i = 0; i < shape[0]; i++) {
    _elemwise_unary_nd(kern, a + i * sa[0], sa + 1, out + i * so[0], so + 1,
                       shape + 1, ndim - 1);
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
    _elemwise_binary_nd(kern, (const char *)a->data, a->strides,
                        (const char *)b->data, b->strides, (char *)out->data,
                        out->strides, a->shape, a->dim);
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
     * We build a fake strides array of all-zeros so _elemwise_binary_nd
     * passes sb=0 at every recursion level. */
    size_t zero_strides[NUMC_MAX_DIMENSIONS] = {0};
    _elemwise_binary_nd(kern, (const char *)a->data, a->strides, scalar_buf,
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
    _elemwise_binary_nd(kern, (const char *)a->data, a->strides, buf,
                        zero_strides, (char *)a->data, a->strides, a->shape,
                        a->dim);
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

/* ── Unary ops dispatch ───────────────────────────────────────────── */

static int _unary_op(const struct NumcArray *a, struct NumcArray *out,
                     const NumcUnaryKernel *table) {
  NumcUnaryKernel kern = table[a->dtype];

  if (a->is_contiguous && out->is_contiguous) {
    intptr_t es = (intptr_t)a->elem_size;
    kern((const char *)a->data, (char *)out->data, a->size, es, es);
  } else {
    _elemwise_unary_nd(kern, (const char *)a->data, a->strides,
                       (char *)out->data, out->strides, a->shape, a->dim);
  }

  return 0;
}

static int _unary_op_inplace(NumcArray *a, const NumcUnaryKernel *table) {
  if (!a)
    return NUMC_ERR_NULL;

  NumcUnaryKernel kern = table[a->dtype];

  if (a->is_contiguous) {
    intptr_t es = (intptr_t)a->elem_size;
    kern((const char *)a->data, (char *)a->data, a->size, es, es);
  } else {
    _elemwise_unary_nd(kern, (const char *)a->data, a->strides, (char *)a->data,
                       a->strides, a->shape, a->dim);
  }
  return 0;
}

/* ── Unary Validation ───────────────────────────────────────────────────── */

static int _check_unary(const struct NumcArray *a,
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

/* ── Element-wise unary ops
 * ──────────────────────────────────────────────────── */

int numc_neg(NumcArray *a, NumcArray *out) {
  int err = _check_unary(a, out);
  if (err)
    return err;
  return _unary_op(a, out, _neg_table);
}

int numc_neg_inplace(NumcArray *a) { return _unary_op_inplace(a, _neg_table); }

int numc_abs(NumcArray *a, NumcArray *out) {
  int err = _check_unary(a, out);
  if (err)
    return err;
  return _unary_op(a, out, _abs_table);
}
int numc_abs_inplace(NumcArray *a) { return _unary_op_inplace(a, _abs_table); }

int numc_log(NumcArray *a, NumcArray *out) {
  int err = _check_unary(a, out);
  if (err)
    return err;
  return _unary_op(a, out, _log_table);
}
int numc_log_inplace(NumcArray *a) { return _unary_op_inplace(a, _log_table); }
