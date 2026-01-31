/**
 * @file dtype.h
 * @brief Data type enumeration for numc arrays.
 *
 * Provides type identifiers to distinguish between numeric types
 * of the same size (e.g., float vs int32_t).
 */

#ifndef DTYPE_H
#define DTYPE_H

#include <stddef.h>
#include <stdint.h>

// Floating-point types (best SIMD support)
#define NUMC_FLOAT float
#define NUMC_DOUBLE double

// 32-bit integers (excellent SIMD support)
#define NUMC_INT int32_t
#define NUMC_UINT uint32_t

// 64-bit integers (good SIMD support)
#define NUMC_LONG int64_t
#define NUMC_ULONG uint64_t

// 16-bit integers (very good SIMD support)
#define NUMC_SHORT int16_t
#define NUMC_USHORT uint16_t

// 8-bit integers (excellent SIMD support - 16 elements per SSE register)
#define NUMC_BYTE int8_t
#define NUMC_UBYTE uint8_t

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
static inline size_t dtype_size(DType dtype) {
  switch (dtype) {
  case DTYPE_BYTE:
  case DTYPE_UBYTE:
    return sizeof(NUMC_BYTE);
  case DTYPE_SHORT:
  case DTYPE_USHORT:
    return sizeof(NUMC_SHORT);
  case DTYPE_INT:
  case DTYPE_UINT:
    return sizeof(NUMC_INT);
  case DTYPE_LONG:
  case DTYPE_ULONG:
    return sizeof(NUMC_LONG);
  case DTYPE_FLOAT:
    return sizeof(NUMC_FLOAT);
  case DTYPE_DOUBLE:
    return sizeof(NUMC_DOUBLE);
  default:
    return 0;
  }
}

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
