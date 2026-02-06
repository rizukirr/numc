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

/**
 * @brief Enumeration of supported data types.
 */
typedef enum {
  NUMC_TYPE_BYTE,   // int8_t
  NUMC_TYPE_UBYTE,  // uint8_t
  NUMC_TYPE_SHORT,  // int16_t
  NUMC_TYPE_USHORT, // uint16_t
  NUMC_TYPE_INT,    // int32_t
  NUMC_TYPE_UINT,   // uint32_t
  NUMC_TYPE_LONG,   // int64_t
  NUMC_TYPE_ULONG,  // uint64_t
  NUMC_TYPE_FLOAT,  // float
  NUMC_TYPE_DOUBLE  // double
} NUMC_TYPE;

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
 * @param numc_type The data type.
 * @return Size in bytes, or 0 for invalid type.
 */

static inline size_t numc_type_size(NUMC_TYPE numc_type) {
  return numc_type_sizes[numc_type];
}

#endif
