/**
 * @file types.h
 * @brief Data type system for numc arrays.
 *
 * This header defines the type system used throughout numc:
 *
 * 1. **Type aliases** (NUMC_BYTE, NUMC_INT, etc.) — portable C type names
 * 2. **NUMC_TYPE enum** — runtime type identifier stored in each Array
 * 3. **FOREACH_NUMC_TYPE(X)** — X-Macro for generating type-generic code
 * 4. **numc_type_size()** — runtime sizeof() lookup by enum value
 *
 * ## X-Macro Pattern
 *
 * The FOREACH_NUMC_TYPE(X) macro calls X(name, c_type) for each type.
 * This is the foundation for all type-generic code generation in numc.
 *
 * Example — generating a function for every type:
 *
 *   #define MAKE_FUNC(name, c_type)  void process_##name(c_type *data) { ... }
 *   FOREACH_NUMC_TYPE(MAKE_FUNC)
 *   #undef MAKE_FUNC
 *
 * Expands to: process_BYTE(int8_t*), process_UBYTE(uint8_t*), ...,
 *             process_DOUBLE(double*)
 *
 * Example — building a lookup table:
 *
 *   #define ENTRY(name, c_type)  [NUMC_TYPE_##name] = process_##name,
 *   func_ptr table[] = { FOREACH_NUMC_TYPE(ENTRY) };
 *   #undef ENTRY
 *
 * This produces: table[NUMC_TYPE_BYTE] = process_BYTE, etc.
 * At runtime, table[array->numc_type](...) dispatches to the right function.
 *
 * See math.c for the full pattern: template macros → adapter macros →
 * lookup tables → public API.
 */

#ifndef TYPES_H
#define TYPES_H

#include <stddef.h>
#include <stdint.h>

// =============================================================================
//  Type Aliases — maps numc names to C standard types
// =============================================================================

#define NUMC_BYTE int8_t
#define NUMC_UBYTE uint8_t
#define NUMC_SHORT int16_t
#define NUMC_USHORT uint16_t
#define NUMC_INT int32_t
#define NUMC_UINT uint32_t
#define NUMC_LONG int64_t
#define NUMC_ULONG uint64_t
#define NUMC_FLOAT float
#define NUMC_DOUBLE double

// =============================================================================
//  X-Macro — iterates X(numc_name, c_type) over all 10 supported types
// =============================================================================

/**
 * @brief Master X-Macro: invokes X(numc_name, c_type) for all 10 types.
 *
 * @param X  A macro that accepts two arguments: (numc_name, c_type).
 *           - numc_name: type suffix used in function names (BYTE, INT, etc.)
 *           - c_type:    actual C type (NUMC_BYTE → int8_t, NUMC_INT → int32_t)
 *
 * Used throughout numc to generate type-specific functions and lookup tables
 * without writing 10 copies of each function by hand.
 */
#define FOREACH_NUMC_TYPE(X)                                                   \
  X(BYTE, NUMC_BYTE)                                                           \
  X(UBYTE, NUMC_UBYTE)                                                         \
  X(SHORT, NUMC_SHORT)                                                         \
  X(USHORT, NUMC_USHORT)                                                       \
  X(INT, NUMC_INT)                                                             \
  X(UINT, NUMC_UINT)                                                           \
  X(LONG, NUMC_LONG)                                                           \
  X(ULONG, NUMC_ULONG)                                                         \
  X(FLOAT, NUMC_FLOAT)                                                         \
  X(DOUBLE, NUMC_DOUBLE)

// =============================================================================
//  Enum — runtime type identifier (stored in Array.numc_type)
// =============================================================================

/**
 * @brief Runtime type identifier for array elements.
 *
 * Enum values are used as indices into lookup tables (e.g., add_funcs[],
 * sum_funcs[]). The order must match FOREACH_NUMC_TYPE iteration order.
 */
typedef enum {
  NUMC_TYPE_BYTE,   // int8_t    (1 byte)
  NUMC_TYPE_UBYTE,  // uint8_t   (1 byte)
  NUMC_TYPE_SHORT,  // int16_t   (2 bytes)
  NUMC_TYPE_USHORT, // uint16_t  (2 bytes)
  NUMC_TYPE_INT,    // int32_t   (4 bytes)
  NUMC_TYPE_UINT,   // uint32_t  (4 bytes)
  NUMC_TYPE_LONG,   // int64_t   (8 bytes)
  NUMC_TYPE_ULONG,  // uint64_t  (8 bytes)
  NUMC_TYPE_FLOAT,  // float     (4 bytes)
  NUMC_TYPE_DOUBLE  // double    (8 bytes)
} NUMC_TYPE;

// =============================================================================
//  Type Size Lookup — sizeof() by enum value at runtime
// =============================================================================

/** @brief Lookup table: numc_type_sizes[NUMC_TYPE_X] = sizeof(X). */
static const size_t numc_type_sizes[] = {
    [NUMC_TYPE_BYTE] = sizeof(NUMC_BYTE),
    [NUMC_TYPE_UBYTE] = sizeof(NUMC_UBYTE),
    [NUMC_TYPE_SHORT] = sizeof(NUMC_SHORT),
    [NUMC_TYPE_USHORT] = sizeof(NUMC_USHORT),
    [NUMC_TYPE_INT] = sizeof(NUMC_INT),
    [NUMC_TYPE_UINT] = sizeof(NUMC_UINT),
    [NUMC_TYPE_LONG] = sizeof(NUMC_LONG),
    [NUMC_TYPE_ULONG] = sizeof(NUMC_ULONG),
    [NUMC_TYPE_FLOAT] = sizeof(NUMC_FLOAT),
    [NUMC_TYPE_DOUBLE] = sizeof(NUMC_DOUBLE),
};

/**
 * @brief Get the size in bytes for a given data type.
 *
 * @param numc_type The data type enum value.
 * @return Size in bytes (1, 2, 4, or 8).
 */
static inline size_t numc_type_size(NUMC_TYPE numc_type) {
  return numc_type_sizes[numc_type];
}

#endif
