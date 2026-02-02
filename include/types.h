/**
 * @file types.h
 * @brief Data type enumeration for numc arrays.
 *
 * Provides type identifiers to distinguish between numeric types
 * of the same size (e.g., float vs int32_t).
 */

#ifndef TYPES_H
#define TYPES_H

#include <stddef.h>
#include <stdint.h>

// Floating-point types
#define NUMC_FLOAT float
#define NUMC_DOUBLE double

// 32-bit integers
#define NUMC_INT int32_t
#define NUMC_UINT uint32_t

// 64-bit integers
#define NUMC_LONG int64_t
#define NUMC_ULONG uint64_t

// 16-bit integers
#define NUMC_SHORT int16_t
#define NUMC_USHORT uint16_t

// 8-bit integers
#define NUMC_BYTE int8_t
#define NUMC_UBYTE uint8_t

/**
 * @brief X-Macro: Define all data types in one place
 *
 * This macro is used to generate type-specific functions automatically,
 * eliminating code duplication and switch statements.
 */
#define FOREACH_DTYPE(X)                                                       \
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

/**
 * @brief Enumeration of supported data types.
 */
typedef enum {
  DTYPE_BYTE,   // int8_t
  DTYPE_UBYTE,  // uint8_t
  DTYPE_SHORT,  // int16_t
  DTYPE_USHORT, // uint16_t
  DTYPE_INT,    // int32_t
  DTYPE_UINT,   // uint32_t
  DTYPE_LONG,   // int64_t
  DTYPE_ULONG,  // uint64_t
  DTYPE_FLOAT,  // float
  DTYPE_DOUBLE  // double
} DType;

/**
 * @brief Get the size in bytes for a given data type.
 *
 * @param dtype The data type.
 * @return Size in bytes, or 0 for invalid type.
 */
size_t dtype_size(DType dtype);

/**
 * @brief Check if a data type is floating-point.
 *
 * @param dtype The data type.
 * @return Non-zero if floating-point, 0 otherwise.
 */
static inline int dtype_is_float(DType dtype) {
  return (dtype == DTYPE_FLOAT || dtype == DTYPE_DOUBLE);
}

/**
 * @brief Check if a data type is signed integer.
 *
 * @param dtype The data type.
 * @return Non-zero if signed integer, 0 otherwise.
 */
static inline int dtype_is_signed(DType dtype) {
  return (dtype == DTYPE_BYTE || dtype == DTYPE_SHORT || dtype == DTYPE_INT ||
          dtype == DTYPE_LONG);
}

#endif
