/**
 * @file array.h
 * @brief Multi-dimensional array library for C.
 *
 * Provides a generic N-dimensional array structure with support for
 * creating, accessing, reshaping, slicing, and copying arrays of any
 * element type.
 */

#ifndef NUMC_ARRAY_H
#define NUMC_ARRAY_H

#include "create.h"
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>

// =============================================================================
//                          Element Access
// =============================================================================

/**
 * @brief Compute the byte offset for the given indices.
 *
 * @param array   Pointer to the array.
 * @param indices Array of indices for each dimension.
 * @return Byte offset from the start of data.
 */
size_t array_offset(const Array *array, const size_t *indices);

/**
 * @brief Bounds check for array indices.
 *
 * @param array   Pointer to the array.
 * @param indices Array of indices for each dimension.
 * @return 0 on success, -1 on failure.
 */
int array_bounds_check(const Array *array, const size_t *indices);

/**
 * @brief Get a pointer to an element at the specified indices.
 *
 * @param array   Pointer to the array.
 * @param indices Array of indices for each dimension.
 * @return Pointer to the element.
 */
void *array_get(const Array *array, const size_t *indices);

// =============================================================================
//                      Type-safe Array Access Helpers
// =============================================================================

/**
 * @brief Type-safe helpers for array element access.
 *
 * These inline functions check the array numc_type and return the appropriate
 * pointer type. They are only available when the array numc_type is known at
 * compile time.
 */

/**
 * @brief Get pointer to float32 element at indices (with type assertion).
 * @param array   Pointer to the array (must have numc_type == NUMC_TYPE_FLOAT).
 * @param indices Array of indices for each dimension.
 * @return Pointer to the element as NUMC_FLOAT*.
 */
static inline NUMC_FLOAT *array_get_float32(const Array *array,
                                            const size_t *indices) {
  assert(array->numc_type == NUMC_TYPE_FLOAT);
  return (NUMC_FLOAT *)array_get(array, indices);
}

/**
 * @brief Get pointer to float64 element at indices (with type assertion).
 * @param array   Pointer to the array (must have numc_type == NUMC_TYPE_DOUBLE).
 * @param indices Array of indices for each dimension.
 * @return Pointer to the element as NUMC_DOUBLE*.
 */
static inline NUMC_DOUBLE *array_get_float64(const Array *array,
                                             const size_t *indices) {
  assert(array->numc_type == NUMC_TYPE_DOUBLE);
  return (NUMC_DOUBLE *)array_get(array, indices);
}

/**
 * @brief Get pointer to int32 element at indices (with type assertion).
 * @param array   Pointer to the array (must have numc_type == NUMC_TYPE_INT).
 * @param indices Array of indices for each dimension.
 * @return Pointer to the element as NUMC_INT*.
 */
static inline NUMC_INT *array_get_int32(const Array *array,
                                        const size_t *indices) {
  assert(array->numc_type == NUMC_TYPE_INT);
  return (NUMC_INT *)array_get(array, indices);
}

/**
 * @brief Get pointer to uint32 element at indices (with type assertion).
 * @param array   Pointer to the array (must have numc_type == NUMC_TYPE_UINT).
 * @param indices Array of indices for each dimension.
 * @return Pointer to the element as NUMC_UINT*.
 */
static inline NUMC_UINT *array_get_uint32(const Array *array,
                                          const size_t *indices) {
  assert(array->numc_type == NUMC_TYPE_UINT);
  return (NUMC_UINT *)array_get(array, indices);
}

/**
 * @brief Get pointer to int64 element at indices (with type assertion).
 * @param array   Pointer to the array (must have numc_type == NUMC_TYPE_LONG).
 * @param indices Array of indices for each dimension.
 * @return Pointer to the element as NUMC_LONG*.
 */
static inline NUMC_LONG *array_get_int64(const Array *array,
                                         const size_t *indices) {
  assert(array->numc_type == NUMC_TYPE_LONG);
  return (NUMC_LONG *)array_get(array, indices);
}

/**
 * @brief Get pointer to uint64 element at indices (with type assertion).
 * @param array   Pointer to the array (must have numc_type == NUMC_TYPE_ULONG).
 * @param indices Array of indices for each dimension.
 * @return Pointer to the element as NUMC_ULONG*.
 */
static inline NUMC_ULONG *array_get_uint64(const Array *array,
                                           const size_t *indices) {
  assert(array->numc_type == NUMC_TYPE_ULONG);
  return (NUMC_ULONG *)array_get(array, indices);
}

/**
 * @brief Get pointer to int16 element at indices (with type assertion).
 * @param array   Pointer to the array (must have numc_type == NUMC_TYPE_SHORT).
 * @param indices Array of indices for each dimension.
 * @return Pointer to the element as NUMC_SHORT*.
 */
static inline NUMC_SHORT *array_get_int16(const Array *array,
                                          const size_t *indices) {
  assert(array->numc_type == NUMC_TYPE_SHORT);
  return (NUMC_SHORT *)array_get(array, indices);
}

/**
 * @brief Get pointer to uint16 element at indices (with type assertion).
 * @param array   Pointer to the array (must have numc_type == NUMC_TYPE_USHORT).
 * @param indices Array of indices for each dimension.
 * @return Pointer to the element as NUMC_USHORT*.
 */
static inline NUMC_USHORT *array_get_uint16(const Array *array,
                                            const size_t *indices) {
  assert(array->numc_type == NUMC_TYPE_USHORT);
  return (NUMC_USHORT *)array_get(array, indices);
}

/**
 * @brief Get pointer to int8 element at indices (with type assertion).
 * @param array   Pointer to the array (must have numc_type == NUMC_TYPE_BYTE).
 * @param indices Array of indices for each dimension.
 * @return Pointer to the element as NUMC_BYTE*.
 */
static inline NUMC_BYTE *array_get_int8(const Array *array,
                                        const size_t *indices) {
  assert(array->numc_type == NUMC_TYPE_BYTE);
  return (NUMC_BYTE *)array_get(array, indices);
}

/**
 * @brief Get pointer to uint8 element at indices (with type assertion).
 * @param array   Pointer to the array (must have numc_type == NUMC_TYPE_UBYTE).
 * @param indices Array of indices for each dimension.
 * @return Pointer to the element as NUMC_UBYTE*.
 */
static inline NUMC_UBYTE *array_get_uint8(const Array *array,
                                          const size_t *indices) {
  assert(array->numc_type == NUMC_TYPE_UBYTE);
  return (NUMC_UBYTE *)array_get(array, indices);
}

/**
 * @brief Increment multi-dimensional indices (row-major order).
 *
 * Advances the indices array to the next element in row-major (C-style) order.
 * When the last dimension reaches its limit, it wraps to 0 and carries to the
 * previous dimension, similar to incrementing a multi-digit number.
 *
 * @param indices Array of current indices (modified in place).
 * @param shape   Array of dimension sizes.
 * @param ndim    Number of dimensions.
 */
void increment_indices(size_t *indices, const size_t *shape, size_t ndim);

// =============================================================================
//                          Array Properties
// =============================================================================

/**
 * @brief Check if an array is contiguous in memory.
 *
 * @param array Pointer to the array.
 * @return Non-zero if contiguous, 0 otherwise.
 */
bool array_is_contiguous(const Array *array);

// =============================================================================
//                          Array Views & Slicing
// =============================================================================

/**
 * @brief Create a slice (view) of an array.
 *
 * The slice shares data with the base array and does not own it.
 *
 * @param base  Pointer to the source array.
 * @param start Array of start indices for each dimension.
 * @param stop  Array of stop indices for each dimension (exclusive).
 * @param step  Array of step sizes for each dimension.
 * @return Pointer to the new slice, or NULL on failure.
 */
Array *array_slice(Array *base, const size_t *start, const size_t *stop,
                   const size_t *step);

// =============================================================================
//                          Array Copying
// =============================================================================

/**
 * @brief Create a copy of an array.
 *
 * If the source array is contiguous, uses fast memcpy.
 * If the source array is non-contiguous (e.g., sliced or transposed),
 * creates a new contiguous array with all elements copied.
 *
 * @param src Pointer to the source array (may be contiguous or non-contiguous).
 * @return Pointer to a new contiguous array, or NULL on failure.
 */
Array *array_copy(const Array *src);

/**
 * @brief Convert a non-contiguous array to a contiguous array.
 *
 * If the source array is already contiguous, does nothing.
 * If the source array is non-contiguous (e.g., sliced or transposed),
 * transform to contiguous array with all elements copied in memory order.
 *
 * @param arr Pointer to the array to convert in-place.
 * @return 0 on success, -1 on failure.
 */
int array_ascontiguousarray(Array *arr);

// =============================================================================
//                          Mathematical Operations
// =============================================================================

/**
 * @brief Element-wise addition with pre-allocated output (no allocation).
 *
 * All arrays must be contiguous, have the same numc_type, shape, and ndim.
 * This function avoids memory allocation by writing to a pre-allocated array.
 *
 * @param a   Pointer to the first input array.
 * @param b   Pointer to the second input array.
 * @param out Pointer to the output array (must be pre-allocated).
 * @return 0 on success, -1 on failure.
 */
int array_add(const Array *a, const Array *b, Array *out);

/**
 * @brief Element-wise subtraction with pre-allocated output (no allocation).
 *
 * All arrays must be contiguous, have the same numc_type, shape, and ndim.
 * This function avoids memory allocation by writing to a pre-allocated array.
 *
 * @param a   Pointer to the first input array.
 * @param b   Pointer to the second input array.
 * @param out Pointer to the output array (must be pre-allocated).
 * @return 0 on success, -1 on failure.
 */
int array_subtract(const Array *a, const Array *b, Array *out);

/**
 * @brief Element-wise multiplication with pre-allocated output (no allocation).
 *
 * All arrays must be contiguous, have the same numc_type, shape, and ndim.
 * This function avoids memory allocation by writing to a pre-allocated array.
 *
 * @param a   Pointer to the first input array.
 * @param b   Pointer to the second input array.
 * @param out Pointer to the output array (must be pre-allocated).
 * @return 0 on success, -1 on failure.
 */
int array_multiply(const Array *a, const Array *b, Array *out);

/**
 * @brief Element-wise division with pre-allocated output (no allocation).
 *
 * All arrays must be contiguous, have the same numc_type, shape, and ndim.
 * This function avoids memory allocation by writing to a pre-allocated array.
 *
 * @warning Division by zero is undefined behavior and may cause crashes.
 *
 * @param a   Pointer to the first input array.
 * @param b   Pointer to the second input array.
 * @param out Pointer to the output array (must be pre-allocated).
 * @return 0 on success, -1 on failure.
 */
int array_divide(const Array *a, const Array *b, Array *out);

// =============================================================================
//                          Reduction Operations
// =============================================================================

/**
 * @brief Compute the sum of all elements. Requires contiguous input.
 * @param a   Pointer to the input array.
 * @param out Pointer to store the result (must match array numc_type).
 * @return 0 on success, -1 on failure.
 */
int array_sum(const Array *a, void *out);

/**
 * @brief Find the minimum element. Requires contiguous input.
 * @param a   Pointer to the input array.
 * @param out Pointer to store the result (must match array numc_type).
 * @return 0 on success, -1 on failure.
 */
int array_min(const Array *a, void *out);

/**
 * @brief Find the maximum element. Requires contiguous input.
 * @param a   Pointer to the input array.
 * @param out Pointer to store the result (must match array numc_type).
 * @return 0 on success, -1 on failure.
 */
int array_max(const Array *a, void *out);

/**
 * @brief Compute the dot product of two arrays. Requires contiguous inputs
 *        with same numc_type and size.
 * @param a   Pointer to the first input array.
 * @param b   Pointer to the second input array.
 * @param out Pointer to store the result (must match array numc_type).
 * @return 0 on success, -1 on failure.
 */
int array_dot(const Array *a, const Array *b, void *out);

// =============================================================================
//                          Scalar-Array Operations
// =============================================================================

/**
 * @brief Element-wise add scalar with pre-allocated output.
 * @param a      Pointer to the input array (contiguous).
 * @param scalar Pointer to a single element of matching numc_type.
 * @param out    Pointer to the output array (contiguous, same shape/numc_type).
 * @return 0 on success, -1 on failure.
 */
int array_add_scalar(const Array *a, const void *scalar, Array *out);

/**
 * @brief Element-wise subtract scalar with pre-allocated output.
 * @param a      Pointer to the input array (contiguous).
 * @param scalar Pointer to a single element of matching numc_type.
 * @param out    Pointer to the output array (contiguous, same shape/numc_type).
 * @return 0 on success, -1 on failure.
 */
int array_subtract_scalar(const Array *a, const void *scalar, Array *out);

/**
 * @brief Element-wise multiply by scalar with pre-allocated output.
 * @param a      Pointer to the input array (contiguous).
 * @param scalar Pointer to a single element of matching numc_type.
 * @param out    Pointer to the output array (contiguous, same shape/numc_type).
 * @return 0 on success, -1 on failure.
 */
int array_multiply_scalar(const Array *a, const void *scalar, Array *out);

/**
 * @brief Element-wise divide by scalar with pre-allocated output.
 * @param a      Pointer to the input array (contiguous).
 * @param scalar Pointer to a single element of matching numc_type.
 * @param out    Pointer to the output array (contiguous, same shape/numc_type).
 * @return 0 on success, -1 on failure.
 */
int array_divide_scalar(const Array *a, const void *scalar, Array *out);

#endif
