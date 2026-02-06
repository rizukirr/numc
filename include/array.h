/**
 * @file array.h
 * @brief Multi-dimensional array library for C.
 *
 * Provides a generic N-dimensional array structure with support for
 * creating, accessing, reshaping, slicing, and copying arrays of any
 * element type.
 */

#ifndef ARRAY_H
#define ARRAY_H

#include "types.h"
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>

#define MAX_STACK_NDIM 8

/**
 * @brief Multi-dimensional array structure.
 *
 * @param data      Pointer to the raw data buffer.
 * @param shape     Array of dimension sizes (length = ndim).
 * @param strides   Array of byte strides for each dimension (length = ndim).
 * @param ndim      Number of dimensions.
 * @param numc_type     Data type of array elements.
 * @param elem_size Size of each element in bytes.
 * @param size      Total number of elements.
 * @param capacity  Allocated capacity in number of elements (for dynamic
 * growth).
 * @param owns_data Non-zero if this array owns its data buffer.
 */
typedef struct {
  void *data;
  size_t *shape;
  size_t *strides;
  size_t ndim;
  NUMC_TYPE numc_type;
  size_t elem_size;
  size_t size;
  size_t capacity;
  bool is_contiguous;
  bool owns_data;
  size_t _shape_buff[MAX_STACK_NDIM];
  size_t _strides_buff[MAX_STACK_NDIM];
} Array;

typedef struct {
  const size_t ndim;
  const size_t *shape;
  const NUMC_TYPE numc_type;
  const void *data;
  const bool owns_data;
} ArrayCreate;

// =============================================================================
//                          Array Creation & Destruction
// =============================================================================

/**
 * @brief Create a new array with the specified shape and data type.
 *
 * @param ndim  Number of dimensions.
 * @param shape Array of dimension sizes.
 * @param numc_type Data type of array elements.
 * @param data  Pointer to contiguous source data to copy.
 * @return Pointer to the new array, or NULL on failure.
 */
Array *array_create(const ArrayCreate *src);

/**
 * @brief Create an array filled with zeros.
 *
 * @param ndim  Number of dimensions.
 * @param shape Shape of the array.
 * @param numc_type Data type of array elements.
 * @return Pointer to a new array, or NULL on failure.
 */
Array *array_zeros(size_t ndim, const size_t *shape, NUMC_TYPE numc_type);

/**
 * @brief Create an array filled with ones.
 *
 * @param ndim      Number of dimensions.
 * @param shape     Shape of the array.
 * @param numc_type     Data type of array elements.
 * @return Pointer to a new array, or NULL on failure.
 */
Array *array_ones(size_t ndim, const size_t *shape, NUMC_TYPE numc_type);

/**
 * @brief Create an array filled with a single value.
 *
 * @param spec  Array specification (ndim, shape, numc_type).
 * @param elem  Pointer to the element to fill with.
 * @return Pointer to a new array, or NULL on failure.
 */
Array *array_full(ArrayCreate *spec, const void *elem);

/**
 * @brief Free an array and its associated memory.
 *
 * Only frees the data buffer if the array owns it.
 *
 * @param array Pointer to the array to free.
 */
void array_free(Array *array);

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

static inline NUMC_FLOAT *array_get_float32(const Array *array,
                                             const size_t *indices) {
  assert(array->numc_type == NUMC_TYPE_FLOAT);
  return (NUMC_FLOAT *)array_get(array, indices);
}

static inline NUMC_DOUBLE *array_get_float64(const Array *array,
                                              const size_t *indices) {
  assert(array->numc_type == NUMC_TYPE_DOUBLE);
  return (NUMC_DOUBLE *)array_get(array, indices);
}

static inline NUMC_INT *array_get_int32(const Array *array,
                                        const size_t *indices) {
  assert(array->numc_type == NUMC_TYPE_INT);
  return (NUMC_INT *)array_get(array, indices);
}

static inline NUMC_UINT *array_get_uint32(const Array *array,
                                          const size_t *indices) {
  assert(array->numc_type == NUMC_TYPE_UINT);
  return (NUMC_UINT *)array_get(array, indices);
}

static inline NUMC_LONG *array_get_int64(const Array *array,
                                         const size_t *indices) {
  assert(array->numc_type == NUMC_TYPE_LONG);
  return (NUMC_LONG *)array_get(array, indices);
}

static inline NUMC_ULONG *array_get_uint64(const Array *array,
                                           const size_t *indices) {
  assert(array->numc_type == NUMC_TYPE_ULONG);
  return (NUMC_ULONG *)array_get(array, indices);
}

static inline NUMC_SHORT *array_get_int16(const Array *array,
                                          const size_t *indices) {
  assert(array->numc_type == NUMC_TYPE_SHORT);
  return (NUMC_SHORT *)array_get(array, indices);
}

static inline NUMC_USHORT *array_get_uint16(const Array *array,
                                            const size_t *indices) {
  assert(array->numc_type == NUMC_TYPE_USHORT);
  return (NUMC_USHORT *)array_get(array, indices);
}

static inline NUMC_BYTE *array_get_int8(const Array *array,
                                        const size_t *indices) {
  assert(array->numc_type == NUMC_TYPE_BYTE);
  return (NUMC_BYTE *)array_get(array, indices);
}

static inline NUMC_UBYTE *array_get_uint8(const Array *array,
                                          const size_t *indices) {
  assert(array->numc_type == NUMC_TYPE_UBYTE);
  return (NUMC_UBYTE *)array_get(array, indices);
}

// =============================================================================
//                          Array Properties
// =============================================================================

/**
 * @brief Check if an array is contiguous in memory.
 *
 * @param array Pointer to the array.
 * @return Non-zero if contiguous, 0 otherwise.
 */
int array_is_contiguous(const Array *array);

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
 * If the source array is already contiguous, creates a contiguous copy.
 * If the source array is non-contiguous (e.g., sliced or transposed),
 * creates a new contiguous array with all elements copied in memory order.
 *
 * This function is equivalent to array_copy() and exists for API clarity.
 *
 * @param src Pointer to the source array (may be contiguous or non-contiguous).
 * @return Pointer to a new contiguous array, or NULL on failure.
 */
Array *array_ascontiguousarray(const Array *src);

// =============================================================================
//                          Array Manipulation
// =============================================================================

/**
 * @brief Reshape an array to a new shape.
 *
 * The total number of elements must remain unchanged.
 * Only works on contiguous arrays.
 *
 * @param array Pointer to the array.
 * @param ndim  New number of dimensions.
 * @param shape New dimension sizes.
 * @return 0 on success, ERROR or ERROR_DIM on failure.
 */
int array_reshape(Array *array, size_t ndim, const size_t *shape);

/**
 * @brief Transpose an array in place.
 *
 * Changes the order of the array dimensions.
 *
 * @param array Pointer to the array.
 * @param axes  Array of axis indices specifying the new order.
 *
 * @return 0 on success, ERROR or ERROR_DIM on failure.
 */
int array_transpose(Array *array, size_t *axes);

/**
 * @brief Concatenate two arrays along a specified axis.
 *
 * Creates a new array containing the elements of both arrays
 * joined along the given axis. Both arrays must have the same
 * shape except in the concatenation axis.
 *
 * Uses fast memcpy when both arrays are contiguous and axis is 0.
 * Otherwise, uses strided copy for non-contiguous arrays or other axes.
 *
 * @param a    Pointer to the first array.
 * @param b    Pointer to the second array.
 * @param axis Axis along which to concatenate.
 * @return Pointer to a new concatenated array, or NULL on failure.
 */
Array *array_concatenate(const Array *a, const Array *b, size_t axis);

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
