/**
 * @file math.c
 * @brief Mathematical operations on arrays.
 */

#include "array.h"

#include <stdio.h>
#include <stdlib.h>

// =============================================================================
// Type-specific binary operation kernels (generated via X-Macro)
// Compiler auto-vectorizes these simple loops with -O3
// =============================================================================

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
 * @param dtype_name  DType name (BYTE, UBYTE, SHORT, etc.)
 * @param c_type      Corresponding C type (int8_t, uint8_t, int16_t, etc.)
 * @param op_symbol   Operator symbol (+, -, *, /)
 */
#define GENERATE_BINARY_OP_FUNC(op_name, dtype_name, c_type, op_symbol)        \
  static inline void op_name##_##dtype_name(const void *restrict a,            \
                                            const void *restrict b,             \
                                            void *restrict out, size_t n) {    \
    const c_type *restrict pa = (const c_type *)a;                             \
    const c_type *restrict pb = (const c_type *)b;                             \
    c_type *restrict pout = (c_type *)out;                                     \
    for (size_t i = 0; i < n; i++) {                                           \
      pout[i] = pa[i] op_symbol pb[i];                                         \
    }                                                                          \
  }

// Generate add functions: add_BYTE, add_UBYTE, add_SHORT, etc.
#define GENERATE_ADD_FUNC(dtype_name, c_type)                                  \
  GENERATE_BINARY_OP_FUNC(add, dtype_name, c_type, +)
FOREACH_DTYPE(GENERATE_ADD_FUNC)
#undef GENERATE_ADD_FUNC

// Generate sub functions: sub_BYTE, sub_UBYTE, sub_SHORT, etc.
#define GENERATE_SUB_FUNC(dtype_name, c_type)                                  \
  GENERATE_BINARY_OP_FUNC(sub, dtype_name, c_type, -)
FOREACH_DTYPE(GENERATE_SUB_FUNC)
#undef GENERATE_SUB_FUNC

// Generate mul functions: mul_BYTE, mul_UBYTE, mul_SHORT, etc.
#define GENERATE_MUL_FUNC(dtype_name, c_type)                                  \
  GENERATE_BINARY_OP_FUNC(mul, dtype_name, c_type, *)
FOREACH_DTYPE(GENERATE_MUL_FUNC)
#undef GENERATE_MUL_FUNC

// Generate div functions: div_BYTE, div_UBYTE, div_SHORT, etc.
#define GENERATE_DIV_FUNC(dtype_name, c_type)                                  \
  GENERATE_BINARY_OP_FUNC(div, dtype_name, c_type, /)
FOREACH_DTYPE(GENERATE_DIV_FUNC)
#undef GENERATE_DIV_FUNC

// Undefine the generic macro to prevent misuse
#undef GENERATE_BINARY_OP_FUNC

// Function pointer type for binary operations
typedef void (*binary_op_func)(const void *, const void *, void *, size_t);

// Function pointer tables indexed by DType enum
#define ADD_FUNC_ENTRY(dtype_name, c_type)                                     \
  [DTYPE_##dtype_name] = add_##dtype_name,
static const binary_op_func add_funcs[] = {FOREACH_DTYPE(ADD_FUNC_ENTRY)};
#undef ADD_FUNC_ENTRY

#define SUB_FUNC_ENTRY(dtype_name, c_type)                                     \
  [DTYPE_##dtype_name] = sub_##dtype_name,
static const binary_op_func sub_funcs[] = {FOREACH_DTYPE(SUB_FUNC_ENTRY)};
#undef SUB_FUNC_ENTRY

#define MUL_FUNC_ENTRY(dtype_name, c_type)                                     \
  [DTYPE_##dtype_name] = mul_##dtype_name,
static const binary_op_func mul_funcs[] = {FOREACH_DTYPE(MUL_FUNC_ENTRY)};
#undef MUL_FUNC_ENTRY

#define DIV_FUNC_ENTRY(dtype_name, c_type)                                     \
  [DTYPE_##dtype_name] = div_##dtype_name,
static const binary_op_func div_funcs[] = {FOREACH_DTYPE(DIV_FUNC_ENTRY)};
#undef DIV_FUNC_ENTRY

// =============================================================================
//                          Helper Functions
// =============================================================================

/**
 * @brief Generic binary operation implementation.
 *
 * Validates inputs and dispatches to the appropriate type-specific function.
 *
 * @param a         Pointer to the first array.
 * @param b         Pointer to the second array.
 * @param op_funcs  Function pointer table for the operation.
 * @param op_name   Operation name (for error messages).
 * @return Pointer to a new array containing the result, or NULL on failure.
 */
static Array *array_binary_op(const Array *a, const Array *b,
                              const binary_op_func *op_funcs,
                              const char *op_name) {
  if (!a || !b)
    return NULL;

  if (!array_is_contiguous(a)) {
    fprintf(stderr,
            "[ERROR] array_%s: array a is not contiguous, "
            "use array_to_contiguous() first\n",
            op_name);
    abort();
  }

  if (!array_is_contiguous(b)) {
    fprintf(stderr,
            "[ERROR] array_%s: array b is not contiguous, "
            "use array_to_contiguous() first\n",
            op_name);
    abort();
  }

  if (a->dtype != b->dtype)
    return NULL;

  if (a->ndim != b->ndim)
    return NULL;

  for (size_t i = 0; i < a->ndim; i++) {
    if (a->shape[i] != b->shape[i])
      return NULL;
  }

  Array *result = array_create(a->ndim, a->shape, a->dtype, NULL);
  if (!result)
    return NULL;

  op_funcs[a->dtype](a->data, b->data, result->data, a->size);
  return result;
}

// =============================================================================
//                          Public Functions
// =============================================================================

Array *array_add(const Array *a, const Array *b) {
  return array_binary_op(a, b, add_funcs, "add");
}

Array *array_sub(const Array *a, const Array *b) {
  return array_binary_op(a, b, sub_funcs, "sub");
}

Array *array_mul(const Array *a, const Array *b) {
  return array_binary_op(a, b, mul_funcs, "mul");
}

Array *array_div(const Array *a, const Array *b) {
  return array_binary_op(a, b, div_funcs, "div");
}
