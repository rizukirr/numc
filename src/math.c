/**
 * @file math.c
 * @brief Mathematical operations on arrays.
 */

#include "alloc.h"
#include "array.h"
#include <stdio.h>
#include <stdlib.h>

// =============================================================================
// Type-specific binary operation kernels (generated via X-Macro)
// Compiler auto-vectorizes these simple loops with -O3
// =============================================================================

/**
 * @brief Macro to iterate over 32-bit and smaller types only
 *
 * For optimal performance, 64-bit types (LONG, ULONG, DOUBLE) are handled
 * separately without alignment hints.
 */
#define FOREACH_NUMC_TYPE_32BIT(X)                                             \
  X(BYTE, NUMC_BYTE)                                                           \
  X(UBYTE, NUMC_UBYTE)                                                         \
  X(SHORT, NUMC_SHORT)                                                         \
  X(USHORT, NUMC_USHORT)                                                       \
  X(INT, NUMC_INT)                                                             \
  X(UINT, NUMC_UINT)                                                           \
  X(FLOAT, NUMC_FLOAT)

#define FOREACH_NUMC_TYPE_64BIT(X)                                             \
  X(LONG, NUMC_LONG)                                                           \
  X(ULONG, NUMC_ULONG)                                                         \
  X(DOUBLE, NUMC_DOUBLE)

/**
 * @brief Generate type-specific binary operation function
 *
 * Each function is a simple loop that the compiler can auto-vectorize.
 * The function signature uses void* for generic handling, but internally
 * we cast to the specific type for type safety and SIMD optimization.
 *
 * The __restrict__ qualifiers tell the compiler that pointers don't alias,
 * enabling better auto-vectorization for floating-point types.
 *
 * @param op_name     Operation name (add, sub, mul, div)
 * @param numc_type_name  NUMC_TYPE name (BYTE, UBYTE, SHORT, etc.)
 * @param c_type      Corresponding C type (int8_t, uint8_t, int16_t, etc.)
 * @param op_symbol   Operator symbol (+, -, *, /)
 */
#define GENERATE_BINARY_OP_FUNC(op_name, numc_type_name, c_type, op_symbol)    \
  static inline void op_name##_##numc_type_name(                               \
      const void *restrict a, const void *restrict b, void *restrict out,      \
      size_t n) {                                                              \
    const c_type *restrict pa = __builtin_assume_aligned(a, NUMC_ALIGN);       \
    const c_type *restrict pb = __builtin_assume_aligned(b, NUMC_ALIGN);       \
    c_type *restrict pout = __builtin_assume_aligned(out, NUMC_ALIGN);         \
    for (size_t i = 0; i < n; i++) {                                           \
      pout[i] = pa[i] op_symbol pb[i];                                         \
    }                                                                          \
  }

/**
 * @brief Generate binary operation function for 64-bit types WITHOUT alignment
 * hints
 *
 * For 64-bit types (LONG, ULONG, DOUBLE), we use a simpler approach without
 * __builtin_assume_aligned to avoid potential performance issues with
 * cache line alignment and memory bandwidth.
 *
 * @param op_name     Operation name (add, sub, mul, div)
 * @param numc_type_name  NUMC_TYPE name (LONG, ULONG, DOUBLE)
 * @param c_type      Corresponding C type (int64_t, uint64_t, double)
 * @param op_symbol   Operator symbol (+, -, *, /)
 */
#define GENERATE_BINARY_OP_FUNC_64BIT(op_name, numc_type_name, c_type,         \
                                      op_symbol)                               \
  static inline void op_name##_##numc_type_name(                               \
      const void *restrict a, const void *restrict b, void *restrict out,      \
      size_t n) {                                                              \
    const c_type *restrict pa = (const c_type *)a;                             \
    const c_type *restrict pb = (const c_type *)b;                             \
    c_type *restrict pout = (c_type *)out;                                     \
    for (size_t i = 0; i < n; i++) {                                           \
      pout[i] = pa[i] op_symbol pb[i];                                         \
    }                                                                          \
  }

// Generate add functions: add_BYTE, add_UBYTE, add_SHORT, etc. (32-bit and
// smaller)
#define GENERATE_ADD_FUNC(numc_type_name, c_type)                              \
  GENERATE_BINARY_OP_FUNC(add, numc_type_name, c_type, +)
FOREACH_NUMC_TYPE_32BIT(GENERATE_ADD_FUNC)
#undef GENERATE_ADD_FUNC

// Generate 64-bit add functions without alignment hints
#define GENERATE_ADD_FUNC_64BIT(numc_type_name, c_type)                        \
  GENERATE_BINARY_OP_FUNC_64BIT(add, numc_type_name, c_type, +)
FOREACH_NUMC_TYPE_64BIT(GENERATE_ADD_FUNC_64BIT)
#undef GENERATE_ADD_FUNC_64BIT

// Generate sub functions: sub_BYTE, sub_UBYTE, sub_SHORT, etc. (32-bit and
// smaller)
#define GENERATE_SUB_FUNC(numc_type_name, c_type)                              \
  GENERATE_BINARY_OP_FUNC(sub, numc_type_name, c_type, -)
FOREACH_NUMC_TYPE_32BIT(GENERATE_SUB_FUNC)
#undef GENERATE_SUB_FUNC

// Generate 64-bit sub functions without alignment hints
#define GENERATE_SUB_FUNC_64BIT(numc_type_name, c_type)                        \
  GENERATE_BINARY_OP_FUNC_64BIT(sub, numc_type_name, c_type, -)
FOREACH_NUMC_TYPE_64BIT(GENERATE_SUB_FUNC_64BIT)
#undef GENERATE_SUB_FUNC_64BIT

// Generate mul functions: mul_BYTE, mul_UBYTE, mul_SHORT, etc. (32-bit and
// smaller)
#define GENERATE_MUL_FUNC(numc_type_name, c_type)                              \
  GENERATE_BINARY_OP_FUNC(mul, numc_type_name, c_type, *)
FOREACH_NUMC_TYPE_32BIT(GENERATE_MUL_FUNC)
#undef GENERATE_MUL_FUNC

// Generate 64-bit mul functions without alignment hints
#define GENERATE_MUL_FUNC_64BIT(numc_type_name, c_type)                        \
  GENERATE_BINARY_OP_FUNC_64BIT(mul, numc_type_name, c_type, *)
GENERATE_MUL_FUNC_64BIT(LONG, NUMC_LONG)
GENERATE_MUL_FUNC_64BIT(ULONG, NUMC_ULONG)
GENERATE_MUL_FUNC_64BIT(DOUBLE, NUMC_DOUBLE)
#undef GENERATE_MUL_FUNC_64BIT

// Generate div functions: div_BYTE, div_UBYTE, div_SHORT, etc. (32-bit and
// smaller)
#define GENERATE_DIV_FUNC(numc_type_name, c_type)                              \
  GENERATE_BINARY_OP_FUNC(div, numc_type_name, c_type, /)
FOREACH_NUMC_TYPE_32BIT(GENERATE_DIV_FUNC)
#undef GENERATE_DIV_FUNC

// Generate 64-bit div functions without alignment hints
#define GENERATE_DIV_FUNC_64BIT(numc_type_name, c_type)                        \
  GENERATE_BINARY_OP_FUNC_64BIT(div, numc_type_name, c_type, /)
FOREACH_NUMC_TYPE_64BIT(GENERATE_DIV_FUNC_64BIT)
#undef GENERATE_DIV_FUNC_64BIT

// Undefine the generic macro to prevent misuse
#undef GENERATE_BINARY_OP_FUNC

// Function pointer type for binary operations
typedef void (*binary_op_func)(const void *, const void *, void *, size_t);

// Function pointer tables indexed by NUMC_TYPE enum
#define ADD_FUNC_ENTRY(numc_type_name, c_type)                                 \
  [NUMC_TYPE_##numc_type_name] = add_##numc_type_name,
static const binary_op_func add_funcs[] = {FOREACH_NUMC_TYPE(ADD_FUNC_ENTRY)};
#undef ADD_FUNC_ENTRY

#define SUB_FUNC_ENTRY(numc_type_name, c_type)                                 \
  [NUMC_TYPE_##numc_type_name] = sub_##numc_type_name,
static const binary_op_func sub_funcs[] = {FOREACH_NUMC_TYPE(SUB_FUNC_ENTRY)};
#undef SUB_FUNC_ENTRY

#define MUL_FUNC_ENTRY(numc_type_name, c_type)                                 \
  [NUMC_TYPE_##numc_type_name] = mul_##numc_type_name,
static const binary_op_func mul_funcs[] = {FOREACH_NUMC_TYPE(MUL_FUNC_ENTRY)};
#undef MUL_FUNC_ENTRY

#define DIV_FUNC_ENTRY(numc_type_name, c_type)                                 \
  [NUMC_TYPE_##numc_type_name] = div_##numc_type_name,
static const binary_op_func div_funcs[] = {FOREACH_NUMC_TYPE(DIV_FUNC_ENTRY)};
#undef DIV_FUNC_ENTRY

// =============================================================================
//                          Helper Functions
// =============================================================================

/**
 * @brief Generic binary operation with pre-allocated output.
 *
 * @param a         Pointer to the first array.
 * @param b         Pointer to the second array.
 * @param out       Pointer to the output array.
 * @param op_funcs  Function pointer table for the operation.
 * @param op_name   Operation name (for error messages).
 * @return 0 on success, -1 on failure.
 */
static int array_binary_op_out(const Array *a, const Array *b, Array *out,
                               const binary_op_func *op_funcs) {
  if (!a || !b || !out)
    return -1;

  if (a->numc_type != b->numc_type || a->numc_type != out->numc_type ||
      a->ndim != b->ndim || a->ndim != out->ndim)
    return -1;

  for (size_t i = 0; i < a->ndim; i++) {
    if (a->shape[i] != b->shape[i] || a->shape[i] != out->shape[i])
      return -1;
  }

  op_funcs[a->numc_type](a->data, b->data, out->data, a->size);
  return 0;
}

// =============================================================================
//                          Public Functions
// =============================================================================

int array_add(const Array *a, const Array *b, Array *out) {
  return array_binary_op_out(a, b, out, add_funcs);
}

int array_subtract(const Array *a, const Array *b, Array *out) {
  return array_binary_op_out(a, b, out, sub_funcs);
}

int array_multiply(const Array *a, const Array *b, Array *out) {
  return array_binary_op_out(a, b, out, mul_funcs);
}

int array_divide(const Array *a, const Array *b, Array *out) {
  return array_binary_op_out(a, b, out, div_funcs);
}

// =============================================================================
// Type-specific reduction kernels (generated via X-Macro)
// =============================================================================

// 32-bit types: Use alignment hints for better performance
#define GENERATE_SUM_FUNC(numc_type_name, c_type)                              \
  static inline void sum_##numc_type_name(const void *restrict a, void *out,   \
                                          size_t n) {                          \
    const c_type *restrict pa = __builtin_assume_aligned(a, NUMC_ALIGN);       \
    c_type acc = 0;                                                            \
    for (size_t i = 0; i < n; i++) {                                           \
      acc += pa[i];                                                            \
    }                                                                          \
    *(c_type *)out = acc;                                                      \
  }
FOREACH_NUMC_TYPE_32BIT(GENERATE_SUM_FUNC)
#undef GENERATE_SUM_FUNC

// 64-bit types: Omit alignment hints to avoid cache line conflicts
#define GENERATE_SUM_FUNC_64BIT(numc_type_name, c_type)                        \
  static inline void sum_##numc_type_name(const void *restrict a, void *out,   \
                                          size_t n) {                          \
    const c_type *restrict pa = (const c_type *)a;                             \
    c_type acc = 0;                                                            \
    for (size_t i = 0; i < n; i++) {                                           \
      acc += pa[i];                                                            \
    }                                                                          \
    *(c_type *)out = acc;                                                      \
  }
GENERATE_SUM_FUNC_64BIT(LONG, NUMC_LONG)
GENERATE_SUM_FUNC_64BIT(ULONG, NUMC_ULONG)
GENERATE_SUM_FUNC_64BIT(DOUBLE, NUMC_DOUBLE)
#undef GENERATE_SUM_FUNC_64BIT

#define GENERATE_MIN_FUNC(numc_type_name, c_type)                              \
  static inline void min_##numc_type_name(const void *restrict a, void *out,   \
                                          size_t n) {                          \
    const c_type *restrict pa = __builtin_assume_aligned(a, NUMC_ALIGN);       \
    c_type m = pa[0];                                                          \
    for (size_t i = 1; i < n; i++) {                                           \
      if (pa[i] < m)                                                           \
        m = pa[i];                                                             \
    }                                                                          \
    *(c_type *)out = m;                                                        \
  }
FOREACH_NUMC_TYPE_32BIT(GENERATE_MIN_FUNC)
#undef GENERATE_MIN_FUNC

#define GENERATE_MIN_FUNC_64BIT(numc_type_name, c_type)                        \
  static inline void min_##numc_type_name(const void *restrict a, void *out,   \
                                          size_t n) {                          \
    const c_type *restrict pa = (const c_type *)a;                             \
    c_type m = pa[0];                                                          \
    for (size_t i = 1; i < n; i++) {                                           \
      if (pa[i] < m)                                                           \
        m = pa[i];                                                             \
    }                                                                          \
    *(c_type *)out = m;                                                        \
  }
GENERATE_MIN_FUNC_64BIT(LONG, NUMC_LONG)
GENERATE_MIN_FUNC_64BIT(ULONG, NUMC_ULONG)
GENERATE_MIN_FUNC_64BIT(DOUBLE, NUMC_DOUBLE)
#undef GENERATE_MIN_FUNC_64BIT

#define GENERATE_MAX_FUNC(numc_type_name, c_type)                              \
  static inline void max_##numc_type_name(const void *restrict a, void *out,   \
                                          size_t n) {                          \
    const c_type *restrict pa = __builtin_assume_aligned(a, NUMC_ALIGN);       \
    c_type m = pa[0];                                                          \
    for (size_t i = 1; i < n; i++) {                                           \
      if (pa[i] > m)                                                           \
        m = pa[i];                                                             \
    }                                                                          \
    *(c_type *)out = m;                                                        \
  }
FOREACH_NUMC_TYPE_32BIT(GENERATE_MAX_FUNC)
#undef GENERATE_MAX_FUNC

#define GENERATE_MAX_FUNC_64BIT(numc_type_name, c_type)                        \
  static inline void max_##numc_type_name(const void *restrict a, void *out,   \
                                          size_t n) {                          \
    const c_type *restrict pa = (const c_type *)a;                             \
    c_type m = pa[0];                                                          \
    for (size_t i = 1; i < n; i++) {                                           \
      if (pa[i] > m)                                                           \
        m = pa[i];                                                             \
    }                                                                          \
    *(c_type *)out = m;                                                        \
  }
GENERATE_MAX_FUNC_64BIT(LONG, NUMC_LONG)
GENERATE_MAX_FUNC_64BIT(ULONG, NUMC_ULONG)
GENERATE_MAX_FUNC_64BIT(DOUBLE, NUMC_DOUBLE)
#undef GENERATE_MAX_FUNC_64BIT

#define GENERATE_DOT_FUNC(numc_type_name, c_type)                              \
  static inline void dot_##numc_type_name(                                     \
      const void *restrict a, const void *restrict b, void *out, size_t n) {   \
    const c_type *restrict pa = __builtin_assume_aligned(a, NUMC_ALIGN);       \
    const c_type *restrict pb = __builtin_assume_aligned(b, NUMC_ALIGN);       \
    c_type acc = 0;                                                            \
    for (size_t i = 0; i < n; i++) {                                           \
      acc += pa[i] * pb[i];                                                    \
    }                                                                          \
    *(c_type *)out = acc;                                                      \
  }
FOREACH_NUMC_TYPE_32BIT(GENERATE_DOT_FUNC)
#undef GENERATE_DOT_FUNC

#define GENERATE_DOT_FUNC_64BIT(numc_type_name, c_type)                        \
  static inline void dot_##numc_type_name(                                     \
      const void *restrict a, const void *restrict b, void *out, size_t n) {   \
    const c_type *restrict pa = (const c_type *)a;                             \
    const c_type *restrict pb = (const c_type *)b;                             \
    c_type acc = 0;                                                            \
    for (size_t i = 0; i < n; i++) {                                           \
      acc += pa[i] * pb[i];                                                    \
    }                                                                          \
    *(c_type *)out = acc;                                                      \
  }
GENERATE_DOT_FUNC_64BIT(LONG, NUMC_LONG)
GENERATE_DOT_FUNC_64BIT(ULONG, NUMC_ULONG)
GENERATE_DOT_FUNC_64BIT(DOUBLE, NUMC_DOUBLE)
#undef GENERATE_DOT_FUNC_64BIT

// Reduction function pointer types
typedef void (*reduce_func)(const void *, void *, size_t);
typedef void (*dot_func)(const void *, const void *, void *, size_t);

// Reduction lookup tables
#define SUM_ENTRY(numc_type_name, c_type)                                      \
  [NUMC_TYPE_##numc_type_name] = sum_##numc_type_name,
static const reduce_func sum_funcs[] = {FOREACH_NUMC_TYPE(SUM_ENTRY)};
#undef SUM_ENTRY

#define MIN_ENTRY(numc_type_name, c_type)                                      \
  [NUMC_TYPE_##numc_type_name] = min_##numc_type_name,
static const reduce_func min_funcs[] = {FOREACH_NUMC_TYPE(MIN_ENTRY)};
#undef MIN_ENTRY

#define MAX_ENTRY(numc_type_name, c_type)                                      \
  [NUMC_TYPE_##numc_type_name] = max_##numc_type_name,
static const reduce_func max_funcs[] = {FOREACH_NUMC_TYPE(MAX_ENTRY)};
#undef MAX_ENTRY

#define DOT_ENTRY(numc_type_name, c_type)                                      \
  [NUMC_TYPE_##numc_type_name] = dot_##numc_type_name,
static const dot_func dot_funcs[] = {FOREACH_NUMC_TYPE(DOT_ENTRY)};
#undef DOT_ENTRY

// =============================================================================
// Type-specific scalar operation kernels (generated via X-Macro)
// =============================================================================

// 32-bit types: Use alignment hints for better performance
#define GENERATE_SCALAR_OP_FUNC(op_name, numc_type_name, c_type, op_symbol)    \
  static inline void op_name##s_##numc_type_name(                              \
      const void *restrict a, const void *scalar, void *restrict out,          \
      size_t n) {                                                              \
    const c_type *restrict pa = __builtin_assume_aligned(a, NUMC_ALIGN);       \
    const c_type s = *(const c_type *)scalar;                                  \
    c_type *restrict pout = __builtin_assume_aligned(out, NUMC_ALIGN);         \
    for (size_t i = 0; i < n; i++) {                                           \
      pout[i] = pa[i] op_symbol s;                                             \
    }                                                                          \
  }

#define GENERATE_ADDS_FUNC(numc_type_name, c_type)                             \
  GENERATE_SCALAR_OP_FUNC(add, numc_type_name, c_type, +)
FOREACH_NUMC_TYPE_32BIT(GENERATE_ADDS_FUNC)
#undef GENERATE_ADDS_FUNC

#define GENERATE_SUBS_FUNC(numc_type_name, c_type)                             \
  GENERATE_SCALAR_OP_FUNC(sub, numc_type_name, c_type, -)
FOREACH_NUMC_TYPE_32BIT(GENERATE_SUBS_FUNC)
#undef GENERATE_SUBS_FUNC

#define GENERATE_MULS_FUNC(numc_type_name, c_type)                             \
  GENERATE_SCALAR_OP_FUNC(mul, numc_type_name, c_type, *)
FOREACH_NUMC_TYPE_32BIT(GENERATE_MULS_FUNC)
#undef GENERATE_MULS_FUNC

#define GENERATE_DIVS_FUNC(numc_type_name, c_type)                             \
  GENERATE_SCALAR_OP_FUNC(div, numc_type_name, c_type, /)
FOREACH_NUMC_TYPE_32BIT(GENERATE_DIVS_FUNC)
#undef GENERATE_DIVS_FUNC

#undef GENERATE_SCALAR_OP_FUNC

// 64-bit types: Omit alignment hints to avoid cache line conflicts
#define GENERATE_SCALAR_OP_FUNC_64BIT(op_name, numc_type_name, c_type, op_symbol) \
  static inline void op_name##s_##numc_type_name(                                 \
      const void *restrict a, const void *scalar, void *restrict out,             \
      size_t n) {                                                                 \
    const c_type *restrict pa = (const c_type *)a;                                \
    const c_type s = *(const c_type *)scalar;                                     \
    c_type *restrict pout = (c_type *)out;                                        \
    for (size_t i = 0; i < n; i++) {                                              \
      pout[i] = pa[i] op_symbol s;                                                \
    }                                                                             \
  }

#define GENERATE_ADDS_FUNC_64BIT(numc_type_name, c_type)                       \
  GENERATE_SCALAR_OP_FUNC_64BIT(add, numc_type_name, c_type, +)
GENERATE_ADDS_FUNC_64BIT(LONG, NUMC_LONG)
GENERATE_ADDS_FUNC_64BIT(ULONG, NUMC_ULONG)
GENERATE_ADDS_FUNC_64BIT(DOUBLE, NUMC_DOUBLE)
#undef GENERATE_ADDS_FUNC_64BIT

#define GENERATE_SUBS_FUNC_64BIT(numc_type_name, c_type)                       \
  GENERATE_SCALAR_OP_FUNC_64BIT(sub, numc_type_name, c_type, -)
GENERATE_SUBS_FUNC_64BIT(LONG, NUMC_LONG)
GENERATE_SUBS_FUNC_64BIT(ULONG, NUMC_ULONG)
GENERATE_SUBS_FUNC_64BIT(DOUBLE, NUMC_DOUBLE)
#undef GENERATE_SUBS_FUNC_64BIT

#define GENERATE_MULS_FUNC_64BIT(numc_type_name, c_type)                       \
  GENERATE_SCALAR_OP_FUNC_64BIT(mul, numc_type_name, c_type, *)
GENERATE_MULS_FUNC_64BIT(LONG, NUMC_LONG)
GENERATE_MULS_FUNC_64BIT(ULONG, NUMC_ULONG)
GENERATE_MULS_FUNC_64BIT(DOUBLE, NUMC_DOUBLE)
#undef GENERATE_MULS_FUNC_64BIT

#define GENERATE_DIVS_FUNC_64BIT(numc_type_name, c_type)                       \
  GENERATE_SCALAR_OP_FUNC_64BIT(div, numc_type_name, c_type, /)
GENERATE_DIVS_FUNC_64BIT(LONG, NUMC_LONG)
GENERATE_DIVS_FUNC_64BIT(ULONG, NUMC_ULONG)
GENERATE_DIVS_FUNC_64BIT(DOUBLE, NUMC_DOUBLE)
#undef GENERATE_DIVS_FUNC_64BIT

#undef GENERATE_SCALAR_OP_FUNC_64BIT

// Scalar operation function pointer type
typedef void (*scalar_op_func)(const void *, const void *, void *, size_t);

#define ADDS_ENTRY(numc_type_name, c_type)                                     \
  [NUMC_TYPE_##numc_type_name] = adds_##numc_type_name,
static const scalar_op_func adds_funcs[] = {FOREACH_NUMC_TYPE(ADDS_ENTRY)};
#undef ADDS_ENTRY

#define SUBS_ENTRY(numc_type_name, c_type)                                     \
  [NUMC_TYPE_##numc_type_name] = subs_##numc_type_name,
static const scalar_op_func subs_funcs[] = {FOREACH_NUMC_TYPE(SUBS_ENTRY)};
#undef SUBS_ENTRY

#define MULS_ENTRY(numc_type_name, c_type)                                     \
  [NUMC_TYPE_##numc_type_name] = muls_##numc_type_name,
static const scalar_op_func muls_funcs[] = {FOREACH_NUMC_TYPE(MULS_ENTRY)};
#undef MULS_ENTRY

#define DIVS_ENTRY(numc_type_name, c_type)                                     \
  [NUMC_TYPE_##numc_type_name] = divs_##numc_type_name,
static const scalar_op_func divs_funcs[] = {FOREACH_NUMC_TYPE(DIVS_ENTRY)};
#undef DIVS_ENTRY

// =============================================================================
//                   Reduction & Scalar Public Functions
// =============================================================================

int array_sum(const Array *a, void *out) {
  if (!a || !out || !a->is_contiguous || a->size == 0)
    return -1;
  sum_funcs[a->numc_type](a->data, out, a->size);
  return 0;
}

int array_min(const Array *a, void *out) {
  if (!a || !out || !a->is_contiguous || a->size == 0)
    return -1;
  min_funcs[a->numc_type](a->data, out, a->size);
  return 0;
}

int array_max(const Array *a, void *out) {
  if (!a || !out || !a->is_contiguous || a->size == 0)
    return -1;
  max_funcs[a->numc_type](a->data, out, a->size);
  return 0;
}

int array_dot(const Array *a, const Array *b, void *out) {
  if (!a || !b || !out)
    return -1;
  if (!a->is_contiguous || !b->is_contiguous)
    return -1;
  if (a->numc_type != b->numc_type || a->size != b->size || a->size == 0)
    return -1;
  dot_funcs[a->numc_type](a->data, b->data, out, a->size);
  return 0;
}

static int array_scalar_op(const Array *a, const void *scalar, Array *out,
                           const scalar_op_func *op_funcs) {
  if (!a || !scalar || !out)
    return -1;
  if (a->numc_type != out->numc_type || a->ndim != out->ndim)
    return -1;
  if (!a->is_contiguous || !out->is_contiguous)
    return -1;
  for (size_t i = 0; i < a->ndim; i++) {
    if (a->shape[i] != out->shape[i])
      return -1;
  }
  op_funcs[a->numc_type](a->data, scalar, out->data, a->size);
  return 0;
}

int array_add_scalar(const Array *a, const void *scalar, Array *out) {
  return array_scalar_op(a, scalar, out, adds_funcs);
}

int array_subtract_scalar(const Array *a, const void *scalar, Array *out) {
  return array_scalar_op(a, scalar, out, subs_funcs);
}

int array_multiply_scalar(const Array *a, const void *scalar, Array *out) {
  return array_scalar_op(a, scalar, out, muls_funcs);
}

int array_divide_scalar(const Array *a, const void *scalar, Array *out) {
  return array_scalar_op(a, scalar, out, divs_funcs);
}
