/**
 * @file math.c
 * @brief Mathematical operations on arrays.
 *
 * == Architecture Overview ==
 *
 * This file implements type-generic math using a 3-tier X-Macro pipeline:
 *
 *   Tier 1: TEMPLATE MACROS — define the kernel loop body
 *   Tier 2: ADAPTER MACROS  — bind a template to a specific operator (+,-,*,/)
 *   Tier 3: LOOKUP TABLES   — map NUMC_TYPE enum → function pointer at runtime
 *
 * The flow for a single operation (e.g., array_add) is:
 *
 *   1. GENERATE_BINARY_OP_FUNC(add, INT, NUMC_INT, +)
 *      → generates: static void add_INT(void *a, void *b, void *out, size_t n)
 *
 *   2. Repeat for all 10 types → add_BYTE, add_UBYTE, ..., add_DOUBLE
 *
 *   3. Build lookup table:
 *      add_funcs[NUMC_TYPE_INT] = add_INT
 *      add_funcs[NUMC_TYPE_FLOAT] = add_FLOAT
 *      ...
 *
 *   4. Public API dispatches at runtime:
 *      array_add(a, b, out) → add_funcs[a->numc_type](a->data, b->data, ...)
 *
 * == Performance Tiers ==
 *
 * Types are split into groups with different optimization strategies:
 *
 *   FOREACH_NUMC_TYPE_32BIT:  BYTE, UBYTE, SHORT, USHORT, INT, UINT, FLOAT
 *     → Uses __builtin_assume_aligned(ptr, NUMC_ALIGN) for SIMD hints
 *
 *   FOREACH_NUMC_TYPE_64BIT:  LONG, ULONG, DOUBLE
 *     → Plain casts only (alignment hints cause cache line conflicts for
 * 64-bit)
 *
 * Some operations need further specialization:
 *
 *   Division (BYTE/UBYTE/SHORT/USHORT):
 *     → Promote to float before dividing (x86 has no SIMD integer division)
 *     → Safe because all 8/16-bit values fit exactly in float's 23-bit mantissa
 *
 *   Reductions (FLOAT/DOUBLE sum, min, max, dot):
 *     → Uses `omp simd reduction` to permit SIMD vectorization of FP reductions
 *     → Without `simd`, compiler can't auto-vectorize FP reductions (non-associative
 *       addition, NaN-aware comparisons) and emits scalar instructions
 *
 *   Reductions (all integer types):
 *     → Simple single-accumulator loops (compiler auto-vectorizes these with
 *       paddd/pminsd/pmaxsd SIMD instructions since integer ops are
 * associative)
 */

#include "alloc.h"
#include "array.h"
#include "error.h"

#include "omp.h"

// #############################################################################
// #                                                                           #
// #  TYPE ITERATOR MACROS                                          #
// #                                                                           #
// #  Subsets of FOREACH_NUMC_TYPE for different optimization strategies. # # #
// #############################################################################

/**
 * @brief Iterate over types that are 32-bit or smaller (7 types).
 *
 * These types benefit from __builtin_assume_aligned(ptr, NUMC_ALIGN) which
 * tells the compiler the data is SIMD-aligned, enabling better vectorization.
 *
 * Includes: BYTE(1B), UBYTE(1B), SHORT(2B), USHORT(2B), INT(4B), UINT(4B),
 *           FLOAT(4B)
 *
 * @param X  Macro accepting (numc_name, c_type).
 */
#define FOREACH_NUMC_TYPE_32BIT(X)                                             \
  X(BYTE, NUMC_BYTE)                                                           \
  X(UBYTE, NUMC_UBYTE)                                                         \
  X(SHORT, NUMC_SHORT)                                                         \
  X(USHORT, NUMC_USHORT)                                                       \
  X(INT, NUMC_INT)                                                             \
  X(UINT, NUMC_UINT)                                                           \
  X(FLOAT, NUMC_FLOAT)

/**
 * @brief Iterate over 64-bit types (3 types).
 *
 * These types do NOT use __builtin_assume_aligned because benchmarks showed
 * alignment hints cause 6-41% regressions for 64-bit types due to cache line
 * conflicts (64-byte cache line / 8-byte element = exactly 8 elements per
 * line).
 *
 * Includes: LONG(8B), ULONG(8B), DOUBLE(8B)
 *
 * @param X  Macro accepting (numc_name, c_type).
 */
#define FOREACH_NUMC_TYPE_64BIT(X)                                             \
  X(LONG, NUMC_LONG)                                                           \
  X(ULONG, NUMC_ULONG)                                                         \
  X(DOUBLE, NUMC_DOUBLE)

// #############################################################################
// #                                                                           #
// #  KERNEL TEMPLATE MACROS                                        #
// #                                                                           #
// #  These macros define the shape of a kernel function. They are NOT called #
// #  directly — adapter macros  bind them to a specific operator. #
// #                                                                           #
// #############################################################################

/**
 * @brief [Template] Generate a binary operation kernel for 32-bit types.
 *
 * Produces a function: op_name##_##numc_type_name(a, b, out, n)
 * that loops over n elements performing: out[i] = a[i] OP b[i]
 *
 * Uses __builtin_assume_aligned for SIMD auto-vectorization hints.
 * The compiler turns these simple loops into SSE/AVX instructions with -O3.
 *
 * @param op_name       Operation prefix (add, sub, mul, div)
 * @param numc_type_name Type suffix (BYTE, INT, FLOAT, etc.)
 * @param c_type        C type (NUMC_BYTE → int8_t, NUMC_INT → int32_t, etc.)
 * @param op_symbol     C operator (+, -, *, /)
 *
 * Example: GENERATE_BINARY_OP_FUNC(add, INT, NUMC_INT, +)
 *   → static void add_INT(void *a, void *b, void *out, size_t n) {
 *       int32_t *pa = a; int32_t *pb = b; int32_t *pout = out;
 *       for (i=0; i<n; i++) pout[i] = pa[i] + pb[i];
 *     }
 */
#define GENERATE_BINARY_OP_FUNC(op_name, numc_type_name, c_type, op_symbol)    \
  static inline void op_name##_##numc_type_name(                               \
      const void *restrict a, const void *restrict b, void *out, size_t n) {   \
    const c_type *restrict pa = __builtin_assume_aligned(a, NUMC_ALIGN);       \
    const c_type *restrict pb = __builtin_assume_aligned(b, NUMC_ALIGN);       \
    c_type *restrict pout = __builtin_assume_aligned(out, NUMC_ALIGN);         \
    NUMC_OMP_FOR                                                               \
    for (size_t i = 0; i < n; i++) {                                           \
      pout[i] = pa[i] op_symbol pb[i];                                         \
    }                                                                          \
  }

/**
 * @brief [Template] Generate a binary operation kernel for 64-bit types.
 *
 * Same as GENERATE_BINARY_OP_FUNC but WITHOUT __builtin_assume_aligned.
 * Uses plain pointer casts to avoid cache line conflict performance regressions
 * that occur with alignment hints on 8-byte types.
 *
 * @param op_name        Operation prefix (add, sub, mul, div)
 * @param numc_type_name Type suffix (LONG, ULONG, DOUBLE)
 * @param c_type         C type (NUMC_LONG → int64_t, etc.)
 * @param op_symbol      C operator (+, -, *, /)
 */
#define GENERATE_BINARY_OP_FUNC_64BIT(op_name, numc_type_name, c_type,         \
                                      op_symbol)                               \
  static inline void op_name##_##numc_type_name(                               \
      const void *restrict a, const void *restrict b, void *out, size_t n) {   \
    const c_type *restrict pa = (const c_type *)a;                             \
    const c_type *restrict pb = (const c_type *)b;                             \
    c_type *restrict pout = (c_type *)out;                                     \
    NUMC_OMP_FOR                                                               \
    for (size_t i = 0; i < n; i++) {                                           \
      pout[i] = pa[i] op_symbol pb[i];                                         \
    }                                                                          \
  }

// #############################################################################
// #                                                                           #
// #  BINARY OPERATION GENERATION                                   #
// #                                                                           #
// #  Adapter macros bind a template to a specific operator,       #
// #  then iterate over types to produce all 10 kernel functions.              #
// #                                                                           #
// #  Each block follows the same pattern:                                     #
// #    1. Define adapter: GENERATE_ADD_FUNC(name, type) → template(add, +)    #
// #    2. Iterate: FOREACH_NUMC_TYPE_32BIT(GENERATE_ADD_FUNC)                 #
// #    3. Undef adapter (single-use, prevents accidental reuse)               #
// #    4. Repeat for 64-bit types                                             #
// #                                                                           #
// #  Result: 10 functions per operation (add_BYTE ... add_DOUBLE)             #
// #                                                                           #
// #############################################################################

// --------------- ADD: out[i] = a[i] + b[i] ---------------
// Generates: add_BYTE, add_UBYTE, add_SHORT, add_USHORT, add_INT, add_UINT,
//            add_FLOAT (32-bit, with alignment hints)
#define GENERATE_ADD_FUNC(numc_type_name, c_type)                              \
  GENERATE_BINARY_OP_FUNC(add, numc_type_name, c_type, +)
FOREACH_NUMC_TYPE_32BIT(GENERATE_ADD_FUNC)
#undef GENERATE_ADD_FUNC

// Generates: add_LONG, add_ULONG, add_DOUBLE (64-bit, no alignment hints)
#define GENERATE_ADD_FUNC_64BIT(numc_type_name, c_type)                        \
  GENERATE_BINARY_OP_FUNC_64BIT(add, numc_type_name, c_type, +)
FOREACH_NUMC_TYPE_64BIT(GENERATE_ADD_FUNC_64BIT)
#undef GENERATE_ADD_FUNC_64BIT

// --------------- SUB: out[i] = a[i] - b[i] ---------------
// Generates: sub_BYTE ... sub_FLOAT
#define GENERATE_SUB_FUNC(numc_type_name, c_type)                              \
  GENERATE_BINARY_OP_FUNC(sub, numc_type_name, c_type, -)
FOREACH_NUMC_TYPE_32BIT(GENERATE_SUB_FUNC)
#undef GENERATE_SUB_FUNC

// Generates: sub_LONG, sub_ULONG, sub_DOUBLE
#define GENERATE_SUB_FUNC_64BIT(numc_type_name, c_type)                        \
  GENERATE_BINARY_OP_FUNC_64BIT(sub, numc_type_name, c_type, -)
FOREACH_NUMC_TYPE_64BIT(GENERATE_SUB_FUNC_64BIT)
#undef GENERATE_SUB_FUNC_64BIT

// --------------- MUL: out[i] = a[i] * b[i] ---------------
// Generates: mul_BYTE ... mul_FLOAT
#define GENERATE_MUL_FUNC(numc_type_name, c_type)                              \
  GENERATE_BINARY_OP_FUNC(mul, numc_type_name, c_type, *)
FOREACH_NUMC_TYPE_32BIT(GENERATE_MUL_FUNC)
#undef GENERATE_MUL_FUNC

// Generates: mul_LONG, mul_ULONG, mul_DOUBLE
#define GENERATE_MUL_FUNC_64BIT(numc_type_name, c_type)                        \
  GENERATE_BINARY_OP_FUNC_64BIT(mul, numc_type_name, c_type, *)
FOREACH_NUMC_TYPE_64BIT(GENERATE_MUL_FUNC_64BIT)
#undef GENERATE_MUL_FUNC_64BIT

// --------------- DIV: out[i] = a[i] / b[i] ---------------
//
// Division has 3 strategies depending on type width:
//
//   8/16-bit (BYTE, UBYTE, SHORT, USHORT):
//     Promote to float → SIMD divps → truncate back.
//     x86 has NO SIMD integer division instruction, so scalar idiv is ~50x
//     slower than add. Float promotion enables vectorization. This is safe
//     because all 8/16-bit integers are exactly representable in float
//     (23-bit mantissa > 16 bits).
//
//   32-bit (INT, UINT, FLOAT):
//     Use the standard template. INT/UINT use scalar idiv (no SIMD available).
//     FLOAT uses SIMD divps natively.
//
//   64-bit (LONG, ULONG, DOUBLE):
//     Use the 64-bit template. LONG/ULONG use scalar idiv.
//     DOUBLE uses SIMD divpd natively.

/**
 * @brief [Template] Narrow integer division via float promotion.
 *
 * Produces: div_##name(a, b, out, n) where out[i] = (c_type)(float(a[i]) /
 * float(b[i]))
 *
 * Enables SIMD divps for types that would otherwise use scalar idiv (~10-27x
 * speedup).
 *
 * @param numc_type_name Type suffix (BYTE, UBYTE, SHORT, USHORT)
 * @param c_type         C type (int8_t, uint8_t, int16_t, uint16_t)
 */
#define GENERATE_DIV_FUNC_NARROW(numc_type_name, c_type)                       \
  static inline void div_##numc_type_name(                                     \
      const void *restrict a, const void *restrict b, void *out, size_t n) {   \
    const c_type *restrict pa = __builtin_assume_aligned(a, NUMC_ALIGN);       \
    const c_type *restrict pb = __builtin_assume_aligned(b, NUMC_ALIGN);       \
    c_type *restrict pout = __builtin_assume_aligned(out, NUMC_ALIGN);         \
    NUMC_OMP_FOR                                                               \
    for (size_t i = 0; i < n; i++) {                                           \
      pout[i] = (c_type)((float)pa[i] / (float)pb[i]);                         \
    }                                                                          \
  }
// Generates: div_BYTE, div_UBYTE, div_SHORT, div_USHORT (float promotion)
GENERATE_DIV_FUNC_NARROW(BYTE, NUMC_BYTE)
GENERATE_DIV_FUNC_NARROW(UBYTE, NUMC_UBYTE)
GENERATE_DIV_FUNC_NARROW(SHORT, NUMC_SHORT)
GENERATE_DIV_FUNC_NARROW(USHORT, NUMC_USHORT)
#undef GENERATE_DIV_FUNC_NARROW

// Generates: div_FLOAT (SIMD divps — native hardware FP division)
GENERATE_BINARY_OP_FUNC(div, FLOAT, NUMC_FLOAT, /)

/**
 * @brief INT32/UINT32 binary division via double promotion.
 *
 * x86 has no SIMD integer division. Scalar `idiv` processes 1 element per
 * ~10 cycles. By promoting to double, we use SIMD `vdivpd` (4 doubles per
 * ~14 cycles on AVX2), giving ~2.8x theoretical speedup.
 *
 * This is safe because all int32/uint32 values are exactly representable in
 * double (52-bit mantissa > 32 bits). The truncation back to integer matches
 * C's integer division semantics (truncation toward zero).
 */
static inline void div_INT(const void *restrict a, const void *restrict b,
                           void *out, size_t n) {
  const NUMC_INT *restrict pa = __builtin_assume_aligned(a, NUMC_ALIGN);
  const NUMC_INT *restrict pb = __builtin_assume_aligned(b, NUMC_ALIGN);
  NUMC_INT *restrict pout = __builtin_assume_aligned(out, NUMC_ALIGN);
  NUMC_OMP_FOR
  for (size_t i = 0; i < n; i++) {
    pout[i] = (NUMC_INT)((double)pa[i] / (double)pb[i]);
  }
}

static inline void div_UINT(const void *restrict a, const void *restrict b,
                            void *out, size_t n) {
  const NUMC_UINT *restrict pa = __builtin_assume_aligned(a, NUMC_ALIGN);
  const NUMC_UINT *restrict pb = __builtin_assume_aligned(b, NUMC_ALIGN);
  NUMC_UINT *restrict pout = __builtin_assume_aligned(out, NUMC_ALIGN);
  NUMC_OMP_FOR
  for (size_t i = 0; i < n; i++) {
    pout[i] = (NUMC_UINT)((double)pa[i] / (double)pb[i]);
  }
}

// Generates: div_LONG, div_ULONG (scalar idiv), div_DOUBLE (SIMD divpd)
#define GENERATE_DIV_FUNC_64BIT(numc_type_name, c_type)                        \
  GENERATE_BINARY_OP_FUNC_64BIT(div, numc_type_name, c_type, /)
FOREACH_NUMC_TYPE_64BIT(GENERATE_DIV_FUNC_64BIT)
#undef GENERATE_DIV_FUNC_64BIT

// Templates no longer needed — prevent accidental use in later code
#undef GENERATE_BINARY_OP_FUNC
#undef GENERATE_BINARY_OP_FUNC_64BIT

// #############################################################################
// #                                                                           #
// #  BINARY OPERATION DISPATCH TABLES                              #
// #                                                                           #
// #  Each table maps NUMC_TYPE enum → kernel function pointer.                #
// #  Used by the public API: add_funcs[a->numc_type](a->data, b->data, ...)   #
// #                                                                           #
// #  The ENTRY macros use designated initializers: # #    [NUMC_TYPE_##name] =
// op_##name                                         # #  so
// table[NUMC_TYPE_INT] = add_INT, table[NUMC_TYPE_FLOAT] = add_FLOAT    # # #
// #############################################################################

/** @brief Function pointer type for binary ops: f(a, b, out, n). */
typedef void (*binary_op_func)(const void *, const void *, void *, size_t);

// --- add_funcs[10]: NUMC_TYPE → add_BYTE, add_UBYTE, ..., add_DOUBLE ---
#define ADD_FUNC_ENTRY(numc_type_name, c_type)                                 \
  [NUMC_TYPE_##numc_type_name] = add_##numc_type_name,
static const binary_op_func add_funcs[] = {FOREACH_NUMC_TYPE(ADD_FUNC_ENTRY)};
#undef ADD_FUNC_ENTRY

// --- sub_funcs[10]: NUMC_TYPE → sub_BYTE, ..., sub_DOUBLE ---
#define SUB_FUNC_ENTRY(numc_type_name, c_type)                                 \
  [NUMC_TYPE_##numc_type_name] = sub_##numc_type_name,
static const binary_op_func sub_funcs[] = {FOREACH_NUMC_TYPE(SUB_FUNC_ENTRY)};
#undef SUB_FUNC_ENTRY

// --- mul_funcs[10]: NUMC_TYPE → mul_BYTE, ..., mul_DOUBLE ---
#define MUL_FUNC_ENTRY(numc_type_name, c_type)                                 \
  [NUMC_TYPE_##numc_type_name] = mul_##numc_type_name,
static const binary_op_func mul_funcs[] = {FOREACH_NUMC_TYPE(MUL_FUNC_ENTRY)};
#undef MUL_FUNC_ENTRY

// --- div_funcs[10]: NUMC_TYPE → div_BYTE, ..., div_DOUBLE ---
#define DIV_FUNC_ENTRY(numc_type_name, c_type)                                 \
  [NUMC_TYPE_##numc_type_name] = div_##numc_type_name,
static const binary_op_func div_funcs[] = {FOREACH_NUMC_TYPE(DIV_FUNC_ENTRY)};
#undef DIV_FUNC_ENTRY

// #############################################################################
// #                                                                           #
// #  BINARY OPERATION PUBLIC API                                   #
// #                                                                           #
// #  All public functions share the same validation logic via # #
// array_binary_op_out(), then dispatch through the lookup table.            #
// #                                                                           #
// #############################################################################

/**
 * @brief Shared validation + dispatch for all binary operations.
 *
 * Checks: non-null, matching types/shapes, contiguous arrays.
 * Then dispatches: op_funcs[a->numc_type](a->data, b->data, out->data, n)
 *
 * @param a         First input array (contiguous).
 * @param b         Second input array (contiguous, same shape/type as a).
 * @param out       Pre-allocated output array (same shape/type as a).
 * @param op_funcs  Lookup table (add_funcs, sub_funcs, mul_funcs, or
 * div_funcs).
 * @return 0 on success, -1 on failure.
 */
static int array_binary_op_out(const Array *a, const Array *b, Array *out,
                               const binary_op_func *op_funcs) {
  if (!a || !b || !out) {
    numc_set_error(NUMC_ERR_NULL, "numc: array_binary_op: NULL argument");
    return NUMC_ERR_NULL;
  }

  if (a->numc_type != b->numc_type || a->numc_type != out->numc_type) {
    numc_set_error(NUMC_ERR_TYPE, "numc: array_binary_op: type mismatch");
    return NUMC_ERR_TYPE;
  }

  if (a->ndim != b->ndim || a->ndim != out->ndim) {
    numc_set_error(NUMC_ERR_SHAPE, "numc: array_binary_op: ndim mismatch");
    return NUMC_ERR_SHAPE;
  }

  if (!a->is_contiguous || !b->is_contiguous) {
    numc_set_error(NUMC_ERR_CONTIGUOUS,
                   "numc: array_binary_op: arrays must be contiguous, call "
                   "numc: array_ascontiguousarray() first");
    return NUMC_ERR_CONTIGUOUS;
  }

  for (size_t i = 0; i < a->ndim; i++) {
    if (a->shape[i] != b->shape[i] || a->shape[i] != out->shape[i]) {
      numc_set_error(NUMC_ERR_SHAPE, "numc: array_binary_op: shape mismatch");
      return NUMC_ERR_SHAPE;
    }
  }

  op_funcs[a->numc_type](a->data, b->data, out->data, a->size);
  return 0;
}

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

// #############################################################################
// #                                                                           #
// #  REDUCTION KERNELS (sum, min, max, dot)                        #
// #                                                                           #
// #  Reductions collapse an array to a single value.                          #
// #                                                                           #
// #  Two tiers (matching binary ops):                                         #
// #                                                                           #
// #    32-bit types (BYTE..UINT, FLOAT):                                      #
// #      Uses __builtin_assume_aligned for SIMD hints.                        #
// #      Compiler auto-vectorizes to paddd/addps/pminsd/maxps etc.            #
// #                                                                           #
// #    64-bit types (LONG, ULONG, DOUBLE):                                    #
// #      Plain casts only (no alignment hints).               #
// #                                                                           #
// #############################################################################

// ========================== SUM: acc += a[i] ==========================

/**
 * @brief [Template] Sum with single accumulator (32-bit types).
 *
 * Uses __builtin_assume_aligned for SIMD hints. Compiler auto-vectorizes
 * integer types to paddb/paddw/paddd. FLOAT uses addps with OpenMP reduction.
 */
#define GENERATE_SUM_FUNC(numc_type_name, c_type)                              \
  static inline void sum_##numc_type_name(const void *restrict a, void *out,   \
                                          size_t n) {                          \
    const c_type *restrict pa = __builtin_assume_aligned(a, NUMC_ALIGN);       \
    c_type acc = 0;                                                            \
    NUMC_OMP_REDUCE_SUM                                                        \
    for (size_t i = 0; i < n; i++) {                                           \
      acc += pa[i];                                                            \
    }                                                                          \
    *(c_type *)out = acc;                                                      \
  }
// Generates: sum_BYTE, sum_UBYTE, sum_SHORT, sum_USHORT, sum_INT, sum_UINT,
//            sum_FLOAT
FOREACH_NUMC_TYPE_32BIT(GENERATE_SUM_FUNC)
#undef GENERATE_SUM_FUNC

/**
 * @brief [Template] Sum with single accumulator (64-bit types, no alignment
 * hints).
 */
#define GENERATE_SUM_FUNC_64BIT(numc_type_name, c_type)                        \
  static inline void sum_##numc_type_name(const void *restrict a, void *out,   \
                                          size_t n) {                          \
    const c_type *restrict pa = (const c_type *)a;                             \
    c_type acc = 0;                                                            \
    NUMC_OMP_REDUCE_SUM                                                        \
    for (size_t i = 0; i < n; i++) {                                           \
      acc += pa[i];                                                            \
    }                                                                          \
    *(c_type *)out = acc;                                                      \
  }
// Generates: sum_LONG, sum_ULONG, sum_DOUBLE
FOREACH_NUMC_TYPE_64BIT(GENERATE_SUM_FUNC_64BIT)
#undef GENERATE_SUM_FUNC_64BIT

// ========================== MIN: m = min(m, a[i]) ==========================

/**
 * @brief [Template] Min with single accumulator (32-bit types).
 *
 * Compiler auto-vectorizes integer types to pminsb/pminsw/pminsd.
 * FLOAT uses minps with OpenMP reduction.
 */
#define GENERATE_MIN_FUNC(numc_type_name, c_type)                              \
  static inline void min_##numc_type_name(const void *restrict a, void *out,   \
                                          size_t n) {                          \
    const c_type *restrict pa = __builtin_assume_aligned(a, NUMC_ALIGN);       \
    c_type m = pa[0];                                                          \
    NUMC_OMP_REDUCE_MIN                                                        \
    for (size_t i = 1; i < n; i++) {                                           \
      if (pa[i] < m)                                                           \
        m = pa[i];                                                             \
    }                                                                          \
    *(c_type *)out = m;                                                        \
  }
// Generates: min_BYTE, min_UBYTE, min_SHORT, min_USHORT, min_INT, min_UINT,
//            min_FLOAT
FOREACH_NUMC_TYPE_32BIT(GENERATE_MIN_FUNC)
#undef GENERATE_MIN_FUNC

/**
 * @brief [Template] Min with single accumulator (64-bit types, no alignment
 * hints).
 */
#define GENERATE_MIN_FUNC_64BIT(numc_type_name, c_type)                        \
  static inline void min_##numc_type_name(const void *restrict a, void *out,   \
                                          size_t n) {                          \
    const c_type *restrict pa = (const c_type *)a;                             \
    c_type m = pa[0];                                                          \
    NUMC_OMP_REDUCE_MIN                                                        \
    for (size_t i = 1; i < n; i++) {                                           \
      if (pa[i] < m)                                                           \
        m = pa[i];                                                             \
    }                                                                          \
    *(c_type *)out = m;                                                        \
  }
// Generates: min_LONG, min_ULONG, min_DOUBLE
FOREACH_NUMC_TYPE_64BIT(GENERATE_MIN_FUNC_64BIT)
#undef GENERATE_MIN_FUNC_64BIT

// ========================== MAX: m = max(m, a[i]) ==========================

/**
 * @brief [Template] Max with single accumulator (32-bit types).
 *
 * Compiler auto-vectorizes integer types to pmaxsb/pmaxsw/pmaxsd.
 * FLOAT uses maxps with OpenMP reduction.
 */
#define GENERATE_MAX_FUNC(numc_type_name, c_type)                              \
  static inline void max_##numc_type_name(const void *restrict a, void *out,   \
                                          size_t n) {                          \
    const c_type *restrict pa = __builtin_assume_aligned(a, NUMC_ALIGN);       \
    c_type m = pa[0];                                                          \
    NUMC_OMP_REDUCE_MAX                                                        \
    for (size_t i = 1; i < n; i++) {                                           \
      if (pa[i] > m)                                                           \
        m = pa[i];                                                             \
    }                                                                          \
    *(c_type *)out = m;                                                        \
  }
// Generates: max_BYTE, max_UBYTE, max_SHORT, max_USHORT, max_INT, max_UINT,
//            max_FLOAT
FOREACH_NUMC_TYPE_32BIT(GENERATE_MAX_FUNC)
#undef GENERATE_MAX_FUNC

/**
 * @brief [Template] Max with single accumulator (64-bit types, no alignment
 * hints).
 */
#define GENERATE_MAX_FUNC_64BIT(numc_type_name, c_type)                        \
  static inline void max_##numc_type_name(const void *restrict a, void *out,   \
                                          size_t n) {                          \
    const c_type *restrict pa = (const c_type *)a;                             \
    c_type m = pa[0];                                                          \
    NUMC_OMP_REDUCE_MAX                                                        \
    for (size_t i = 1; i < n; i++) {                                           \
      if (pa[i] > m)                                                           \
        m = pa[i];                                                             \
    }                                                                          \
    *(c_type *)out = m;                                                        \
  }
// Generates: max_LONG, max_ULONG, max_DOUBLE
FOREACH_NUMC_TYPE_64BIT(GENERATE_MAX_FUNC_64BIT)
#undef GENERATE_MAX_FUNC_64BIT

// ========================== DOT: acc += a[i] * b[i] ==========================

/**
 * @brief [Template] Dot product (32-bit types, no OpenMP).
 *
 * OpenMP outlining forces accumulators to memory, preventing vectorization.
 * Instead, we rely on the compiler's vectorizer with interleave hints to
 * create multiple independent vector accumulators (breaking the vaddps
 * dependency chain).
 */
#define GENERATE_DOT_FUNC(numc_type_name, c_type)                              \
  static inline void dot_##numc_type_name(                                     \
      const void *restrict a, const void *restrict b, void *out, size_t n) {   \
    const c_type *restrict pa = __builtin_assume_aligned(a, NUMC_ALIGN);       \
    const c_type *restrict pb = __builtin_assume_aligned(b, NUMC_ALIGN);       \
    c_type acc = 0;                                                            \
    NUMC_PRAGMA(clang loop vectorize_width(8) interleave_count(4))             \
    for (size_t i = 0; i < n; i++) {                                           \
      acc += pa[i] * pb[i];                                                    \
    }                                                                          \
    *(c_type *)out = acc;                                                      \
  }
// Generates: dot_BYTE, dot_UBYTE, dot_SHORT, dot_USHORT, dot_INT, dot_UINT,
//            dot_FLOAT
FOREACH_NUMC_TYPE_32BIT(GENERATE_DOT_FUNC)
#undef GENERATE_DOT_FUNC

/**
 * @brief [Template] Dot product (64-bit types, no OpenMP, no alignment hints).
 */
#define GENERATE_DOT_FUNC_64BIT(numc_type_name, c_type)                        \
  static inline void dot_##numc_type_name(                                     \
      const void *restrict a, const void *restrict b, void *out, size_t n) {   \
    const c_type *restrict pa = (const c_type *)a;                             \
    const c_type *restrict pb = (const c_type *)b;                             \
    c_type acc = 0;                                                            \
    NUMC_PRAGMA(clang loop vectorize_width(8) interleave_count(4))             \
    for (size_t i = 0; i < n; i++) {                                           \
      acc += pa[i] * pb[i];                                                    \
    }                                                                          \
    *(c_type *)out = acc;                                                      \
  }
// Generates: dot_LONG, dot_ULONG, dot_DOUBLE
FOREACH_NUMC_TYPE_64BIT(GENERATE_DOT_FUNC_64BIT)
#undef GENERATE_DOT_FUNC_64BIT

// #############################################################################
// #                                                                           #
// #  REDUCTION DISPATCH TABLES                                     #
// #                                                                           #
// #############################################################################

/** @brief Function pointer type for reductions: f(a, out, n). */
typedef void (*reduce_func)(const void *, void *, size_t);

/** @brief Function pointer type for dot product: f(a, b, out, n). */
typedef void (*dot_func)(const void *, const void *, void *, size_t);

// --- sum_funcs[10]: NUMC_TYPE → sum_BYTE, ..., sum_DOUBLE ---
#define SUM_ENTRY(numc_type_name, c_type)                                      \
  [NUMC_TYPE_##numc_type_name] = sum_##numc_type_name,
static const reduce_func sum_funcs[] = {FOREACH_NUMC_TYPE(SUM_ENTRY)};
#undef SUM_ENTRY

// --- min_funcs[10]: NUMC_TYPE → min_BYTE, ..., min_DOUBLE ---
#define MIN_ENTRY(numc_type_name, c_type)                                      \
  [NUMC_TYPE_##numc_type_name] = min_##numc_type_name,
static const reduce_func min_funcs[] = {FOREACH_NUMC_TYPE(MIN_ENTRY)};
#undef MIN_ENTRY

// --- max_funcs[10]: NUMC_TYPE → max_BYTE, ..., max_DOUBLE ---
#define MAX_ENTRY(numc_type_name, c_type)                                      \
  [NUMC_TYPE_##numc_type_name] = max_##numc_type_name,
static const reduce_func max_funcs[] = {FOREACH_NUMC_TYPE(MAX_ENTRY)};
#undef MAX_ENTRY

// --- dot_funcs[10]: NUMC_TYPE → dot_BYTE, ..., dot_DOUBLE ---
#define DOT_ENTRY(numc_type_name, c_type)                                      \
  [NUMC_TYPE_##numc_type_name] = dot_##numc_type_name,
static const dot_func dot_funcs[] = {FOREACH_NUMC_TYPE(DOT_ENTRY)};
#undef DOT_ENTRY

// #############################################################################
// #                                                                           #
// #  SCALAR OPERATION KERNELS                                      #
// #                                                                           #
// #  Scalar ops: out[i] = a[i] OP scalar                                      #
// #  Same 3-tier pattern as binary ops (template → adapter → dispatch).       #
// #                                                                           #
// #  Note: scalar op functions have an "numc: s" suffix in the name: # #
// adds_INT, subs_INT, muls_INT, divs_INT                                 # # to
// distinguish from binary op functions (add_INT, sub_INT, etc.)         # # #
// #############################################################################

/**
 * @brief [Template] Scalar operation kernel for 32-bit types.
 *
 * Produces: op_name##s_##numc_type_name(a, scalar, out, n)
 * where out[i] = a[i] OP *(scalar)
 *
 * The scalar value is loaded once before the loop, so the compiler can hoist
 * it into a register and broadcast for SIMD.
 *
 * @param op_name        Operation prefix (add, sub, mul, div)
 * @param numc_type_name Type suffix (BYTE, INT, FLOAT, etc.)
 * @param c_type         C type
 * @param op_symbol      C operator (+, -, *, /)
 *
 * Example: GENERATE_SCALAR_OP_FUNC(add, INT, NUMC_INT, +)
 *   → static void adds_INT(void *a, void *scalar, void *out, size_t n)
 *     { int32_t s = *(int32_t*)scalar; for(...) out[i] = a[i] + s; }
 */
#define GENERATE_SCALAR_OP_FUNC(op_name, numc_type_name, c_type, op_symbol)    \
  static inline void op_name##s_##numc_type_name(const void *restrict a,       \
                                                 const void *restrict scalar,  \
                                                 void *out, size_t n) {        \
    const c_type *restrict pa = __builtin_assume_aligned(a, NUMC_ALIGN);       \
    const c_type s = *(const c_type *)scalar;                                  \
    c_type *restrict pout = __builtin_assume_aligned(out, NUMC_ALIGN);         \
    NUMC_OMP_FOR                                                               \
    for (size_t i = 0; i < n; i++) {                                           \
      pout[i] = pa[i] op_symbol s;                                             \
    }                                                                          \
  }

// --------------- SCALAR ADD: out[i] = a[i] + scalar ---------------
// Generates: adds_BYTE, adds_UBYTE, ..., adds_FLOAT
#define GENERATE_ADDS_FUNC(numc_type_name, c_type)                             \
  GENERATE_SCALAR_OP_FUNC(add, numc_type_name, c_type, +)
FOREACH_NUMC_TYPE_32BIT(GENERATE_ADDS_FUNC)
#undef GENERATE_ADDS_FUNC

// --------------- SCALAR SUB: out[i] = a[i] - scalar ---------------
// Generates: subs_BYTE, subs_UBYTE, ..., subs_FLOAT
#define GENERATE_SUBS_FUNC(numc_type_name, c_type)                             \
  GENERATE_SCALAR_OP_FUNC(sub, numc_type_name, c_type, -)
FOREACH_NUMC_TYPE_32BIT(GENERATE_SUBS_FUNC)
#undef GENERATE_SUBS_FUNC

// --------------- SCALAR MUL: out[i] = a[i] * scalar ---------------
// Generates: muls_BYTE, muls_UBYTE, ..., muls_FLOAT
#define GENERATE_MULS_FUNC(numc_type_name, c_type)                             \
  GENERATE_SCALAR_OP_FUNC(mul, numc_type_name, c_type, *)
FOREACH_NUMC_TYPE_32BIT(GENERATE_MULS_FUNC)
#undef GENERATE_MULS_FUNC

// --------------- SCALAR DIV: out[i] = a[i] / scalar ---------------
// Same 3-strategy split as binary division .

/**
 * @brief [Template] Narrow integer scalar division via float promotion.
 *
 * Produces: divs_##name(a, scalar, out, n) where out[i] = (c_type)(float(a[i])
 * / float(scalar))
 *
 * The scalar is converted to float once before the loop. The compiler may
 * further optimize to multiply by reciprocal (1.0f / s) since s is
 * loop-invariant.
 */
#define GENERATE_DIVS_FUNC_NARROW(numc_type_name, c_type)                      \
  static inline void divs_##numc_type_name(const void *restrict a,             \
                                           const void *restrict scalar,        \
                                           void *out, size_t n) {              \
    const c_type *restrict pa = __builtin_assume_aligned(a, NUMC_ALIGN);       \
    const float s = (float)*(const c_type *)scalar;                            \
    c_type *restrict pout = __builtin_assume_aligned(out, NUMC_ALIGN);         \
    NUMC_OMP_FOR                                                               \
    for (size_t i = 0; i < n; i++) {                                           \
      pout[i] = (c_type)((float)pa[i] / s);                                    \
    }                                                                          \
  }
// Generates: divs_BYTE, divs_UBYTE, divs_SHORT, divs_USHORT (float promotion)
GENERATE_DIVS_FUNC_NARROW(BYTE, NUMC_BYTE)
GENERATE_DIVS_FUNC_NARROW(UBYTE, NUMC_UBYTE)
GENERATE_DIVS_FUNC_NARROW(SHORT, NUMC_SHORT)
GENERATE_DIVS_FUNC_NARROW(USHORT, NUMC_USHORT)
#undef GENERATE_DIVS_FUNC_NARROW

// Generates: divs_FLOAT (SIMD divps — native hardware FP division)
GENERATE_SCALAR_OP_FUNC(div, FLOAT, NUMC_FLOAT, /)

/**
 * @brief INT32/UINT32 scalar division via double-promotion reciprocal.
 *
 * Since the divisor is a constant, we precompute 1.0/scalar as a double
 * reciprocal, then multiply each element by it. This converts N divisions
 * into N multiplications (SIMD vmulpd) + 1 division (precomputed).
 *
 * Multiply throughput on modern x86: ~2 cycles per SIMD op
 * vs divide throughput: ~10 cycles per scalar idiv
 *
 * Safe because all int32/uint32 values fit exactly in double's 52-bit mantissa.
 * Floor-toward-zero truncation matches C integer division semantics.
 */
static inline void divs_INT(const void *restrict a, const void *restrict scalar,
                            void *out, size_t n) {
  const NUMC_INT *restrict pa = __builtin_assume_aligned(a, NUMC_ALIGN);
  const double recip = 1.0 / (double)*(const NUMC_INT *)scalar;
  NUMC_INT *restrict pout = __builtin_assume_aligned(out, NUMC_ALIGN);
  NUMC_OMP_FOR
  for (size_t i = 0; i < n; i++) {
    pout[i] = (NUMC_INT)((double)pa[i] * recip);
  }
}

static inline void divs_UINT(const void *restrict a,
                             const void *restrict scalar, void *out, size_t n) {
  const NUMC_UINT *restrict pa = __builtin_assume_aligned(a, NUMC_ALIGN);
  const double recip = 1.0 / (double)*(const NUMC_UINT *)scalar;
  NUMC_UINT *restrict pout = __builtin_assume_aligned(out, NUMC_ALIGN);
  NUMC_OMP_FOR
  for (size_t i = 0; i < n; i++) {
    pout[i] = (NUMC_UINT)((double)pa[i] * recip);
  }
}

#undef GENERATE_SCALAR_OP_FUNC

/**
 * @brief [Template] Scalar operation kernel for 64-bit types.
 *
 * Same as GENERATE_SCALAR_OP_FUNC but without alignment hints.
 */
#define GENERATE_SCALAR_OP_FUNC_64BIT(op_name, numc_type_name, c_type,         \
                                      op_symbol)                               \
  static inline void op_name##s_##numc_type_name(                              \
      const void *restrict a, const void *scalar, void *out, size_t n) {       \
    const c_type *restrict pa = (const c_type *)a;                             \
    const c_type s = *(const c_type *)scalar;                                  \
    c_type *restrict pout = (c_type *)out;                                     \
    NUMC_OMP_FOR                                                               \
    for (size_t i = 0; i < n; i++) {                                           \
      pout[i] = pa[i] op_symbol s;                                             \
    }                                                                          \
  }

// Generates: adds_LONG, adds_ULONG, adds_DOUBLE
#define GENERATE_ADDS_FUNC_64BIT(numc_type_name, c_type)                       \
  GENERATE_SCALAR_OP_FUNC_64BIT(add, numc_type_name, c_type, +)
FOREACH_NUMC_TYPE_64BIT(GENERATE_ADDS_FUNC_64BIT)
#undef GENERATE_ADDS_FUNC_64BIT

// Generates: subs_LONG, subs_ULONG, subs_DOUBLE
#define GENERATE_SUBS_FUNC_64BIT(numc_type_name, c_type)                       \
  GENERATE_SCALAR_OP_FUNC_64BIT(sub, numc_type_name, c_type, -)
FOREACH_NUMC_TYPE_64BIT(GENERATE_SUBS_FUNC_64BIT)
#undef GENERATE_SUBS_FUNC_64BIT

// Generates: muls_LONG, muls_ULONG, muls_DOUBLE
#define GENERATE_MULS_FUNC_64BIT(numc_type_name, c_type)                       \
  GENERATE_SCALAR_OP_FUNC_64BIT(mul, numc_type_name, c_type, *)
FOREACH_NUMC_TYPE_64BIT(GENERATE_MULS_FUNC_64BIT)
#undef GENERATE_MULS_FUNC_64BIT

// Generates: divs_LONG, divs_ULONG (scalar idiv), divs_DOUBLE (SIMD divpd)
#define GENERATE_DIVS_FUNC_64BIT(numc_type_name, c_type)                       \
  GENERATE_SCALAR_OP_FUNC_64BIT(div, numc_type_name, c_type, /)
FOREACH_NUMC_TYPE_64BIT(GENERATE_DIVS_FUNC_64BIT)
#undef GENERATE_DIVS_FUNC_64BIT

#undef GENERATE_SCALAR_OP_FUNC_64BIT

// #############################################################################
// #                                                                           #
// #  SCALAR OPERATION DISPATCH TABLES                              #
// #                                                                           #
// #############################################################################

/** @brief Function pointer type for scalar ops: f(a, scalar, out, n). */
typedef void (*scalar_op_func)(const void *, const void *, void *, size_t);

// --- adds_funcs[10]: NUMC_TYPE → adds_BYTE, ..., adds_DOUBLE ---
#define ADDS_ENTRY(numc_type_name, c_type)                                     \
  [NUMC_TYPE_##numc_type_name] = adds_##numc_type_name,
static const scalar_op_func adds_funcs[] = {FOREACH_NUMC_TYPE(ADDS_ENTRY)};
#undef ADDS_ENTRY

// --- subs_funcs[10]: NUMC_TYPE → subs_BYTE, ..., subs_DOUBLE ---
#define SUBS_ENTRY(numc_type_name, c_type)                                     \
  [NUMC_TYPE_##numc_type_name] = subs_##numc_type_name,
static const scalar_op_func subs_funcs[] = {FOREACH_NUMC_TYPE(SUBS_ENTRY)};
#undef SUBS_ENTRY

// --- muls_funcs[10]: NUMC_TYPE → muls_BYTE, ..., muls_DOUBLE ---
#define MULS_ENTRY(numc_type_name, c_type)                                     \
  [NUMC_TYPE_##numc_type_name] = muls_##numc_type_name,
static const scalar_op_func muls_funcs[] = {FOREACH_NUMC_TYPE(MULS_ENTRY)};
#undef MULS_ENTRY

// --- divs_funcs[10]: NUMC_TYPE → divs_BYTE, ..., divs_DOUBLE ---
#define DIVS_ENTRY(numc_type_name, c_type)                                     \
  [NUMC_TYPE_##numc_type_name] = divs_##numc_type_name,
static const scalar_op_func divs_funcs[] = {FOREACH_NUMC_TYPE(DIVS_ENTRY)};
#undef DIVS_ENTRY

// #############################################################################
// #                                                                           #
// #  REDUCTION & SCALAR PUBLIC API                                #
// #                                                                           #
// #############################################################################

int array_sum(const Array *a, void *out) {
  if (!a || !out) {
    numc_set_error(NUMC_ERR_NULL, "numc: array_sum: NULL argument");
    return NUMC_ERR_NULL;
  }
  if (!a->is_contiguous) {
    numc_set_error(NUMC_ERR_CONTIGUOUS,
                   "numc: array_sum: array must be contiguous");
    return NUMC_ERR_CONTIGUOUS;
  }
  if (a->size == 0) {
    numc_set_error(NUMC_ERR_INVALID, "numc: array_sum: empty array");
    return NUMC_ERR_INVALID;
  }
  sum_funcs[a->numc_type](a->data, out, a->size);
  return NUMC_OK;
}

int array_min(const Array *a, void *out) {
  if (!a || !out) {
    numc_set_error(NUMC_ERR_NULL, "numc: array_min: NULL argument");
    return NUMC_ERR_NULL;
  }
  if (!a->is_contiguous) {
    numc_set_error(NUMC_ERR_CONTIGUOUS,
                   "numc: array_min: array must be contiguous");
    return NUMC_ERR_CONTIGUOUS;
  }
  if (a->size == 0) {
    numc_set_error(NUMC_ERR_INVALID, "numc: array_min: empty array");
    return NUMC_ERR_INVALID;
  }
  min_funcs[a->numc_type](a->data, out, a->size);
  return NUMC_OK;
}

int array_max(const Array *a, void *out) {
  if (!a || !out) {
    numc_set_error(NUMC_ERR_NULL, "numc: array_max: NULL argument");
    return NUMC_ERR_NULL;
  }
  if (!a->is_contiguous) {
    numc_set_error(NUMC_ERR_CONTIGUOUS,
                   "numc: array_max: array must be contiguous");
    return NUMC_ERR_CONTIGUOUS;
  }
  if (a->size == 0) {
    numc_set_error(NUMC_ERR_INVALID, "numc: array_max: empty array");
    return NUMC_ERR_INVALID;
  }
  max_funcs[a->numc_type](a->data, out, a->size);
  return NUMC_OK;
}

int array_dot(const Array *a, const Array *b, void *out) {
  if (!a || !b || !out) {
    numc_set_error(NUMC_ERR_NULL, "numc: array_dot: NULL argument");
    return NUMC_ERR_NULL;
  }
  if (!a->is_contiguous || !b->is_contiguous) {
    numc_set_error(NUMC_ERR_CONTIGUOUS,
                   "numc: array_dot: arrays must be contiguous");
    return NUMC_ERR_CONTIGUOUS;
  }
  if (a->numc_type != b->numc_type) {
    numc_set_error(NUMC_ERR_TYPE, "numc: array_dot: type mismatch");
    return NUMC_ERR_TYPE;
  }
  if (a->size != b->size || a->size == 0) {
    numc_set_error(NUMC_ERR_SHAPE, "numc: array_dot: size mismatch or empty");
    return NUMC_ERR_SHAPE;
  }
  dot_funcs[a->numc_type](a->data, b->data, out, a->size);
  return NUMC_OK;
}

/**
 * @brief Shared validation + dispatch for all scalar operations.
 *
 * @param a        Input array (contiguous).
 * @param scalar   Pointer to a single element of matching type.
 * @param out      Pre-allocated output array (same shape/type).
 * @param op_funcs Lookup table (adds_funcs, subs_funcs, etc.).
 * @return 0 on success, -1 on failure.
 */
static int array_scalar_op(const Array *a, const void *scalar, Array *out,
                           const scalar_op_func *op_funcs) {
  if (!a || !scalar || !out) {
    numc_set_error(NUMC_ERR_NULL, "numc: array_scalar_op: NULL argument");
    return NUMC_ERR_NULL;
  }
  if (a->numc_type != out->numc_type) {
    numc_set_error(NUMC_ERR_TYPE, "numc: array_scalar_op: type mismatch");
    return NUMC_ERR_TYPE;
  }
  if (a->ndim != out->ndim) {
    numc_set_error(NUMC_ERR_SHAPE, "numc: array_scalar_op: ndim mismatch");
    return NUMC_ERR_SHAPE;
  }
  if (!a->is_contiguous || !out->is_contiguous) {
    numc_set_error(NUMC_ERR_CONTIGUOUS,
                   "numc: array_scalar_op: arrays must be contiguous");
    return NUMC_ERR_CONTIGUOUS;
  }
  for (size_t i = 0; i < a->ndim; i++) {
    if (a->shape[i] != out->shape[i]) {
      numc_set_error(NUMC_ERR_SHAPE, "numc: array_scalar_op: shape mismatch");
      return NUMC_ERR_SHAPE;
    }
  }
  op_funcs[a->numc_type](a->data, scalar, out->data, a->size);
  return NUMC_OK;
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
