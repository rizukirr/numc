/**
 * @file array.h
 * @brief Multi-dimensional array library for C.
 *
 * Provides the Array struct and all operations: creation, destruction,
 * element access, slicing, copying, type conversion, reshape, transpose,
 * concatenation, flatten, and printing.
 */

#ifndef NUMC_ARRAY_H
#define NUMC_ARRAY_H

#include "dtype.h"
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>

#define MAX_STACK_NDIM 8

// =============================================================================
//                          Array Struct & Creation Types
// =============================================================================

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
 * @brief Create an empty array with the specified shape and data type.
 *
 * @param ndim  Number of dimensions.
 * @param shape Array of dimension sizes.
 * @param numc_type Data type of array elements.
 * @return Pointer to the new array, or NULL on failure.
 */
Array *array_empty(const ArrayCreate *src);

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
 * @brief Create 1D array with a range of evenly spaced values.
 *
 * @param start Start value.
 * @param stop  Stop value.
 * @param step  Step size.
 *
 * @return Pointer to a new array, or NULL on failure.
 */
Array *array_arange(const int start, const int stop, const int step,
                    const NUMC_TYPE type);

/**
 * @brief Create 1D array of evenly or linearly spaced number over a
 * specified interval.
 *
 * If `num` is evenly divisible by `step`, then the last element is
 * `stop-1`, otherwise it is `stop`.
 *
 * @param start Start value.
 * @param stop  Stop value.
 * @param num   Number of elements.
 * @param type  Data type of array elements.
 *
 * @return Pointer to a new array, or NULL on failure.
 */
Array *array_linspace(const int start, const int stop, const size_t num,
                      const NUMC_TYPE type);

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
 * @param array   Pointer to the array (must have numc_type ==
 * NUMC_TYPE_DOUBLE).
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
 * @param array   Pointer to the array (must have numc_type ==
 * NUMC_TYPE_USHORT).
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
//                          Type Conversion
// =============================================================================

/**
 * @brief Convert array to a different data type in-place.
 *
 * Converts all elements from the current type to the specified target type.
 * The conversion is performed in-place, reallocating memory if needed when
 * the new type requires more space than currently allocated.
 *
 * @param array Pointer to the array to convert (must own its data).
 * @param type  Target data type.
 * @return 0 on success, error code on failure.
 *
 * @warning Cannot convert views (arrays with owns_data=false). The array must
 *          be contiguous or will be converted to contiguous first.
 *
 * @note Narrowing conversions (e.g., float -> int, int64 -> int8) may result
 *       in data loss due to truncation or overflow.
 */
int array_astype(Array *array, NUMC_TYPE type);

// =============================================================================
//                          Shape Operations
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

/**
 * @brief Flatten an array in place.
 *
 * Changes the array shape to be 1D by inserting a new axis at the
 * beginning. The new axis is inserted at the end of the array shape.
 * The total number of elements must remain unchanged.
 * Only works on contiguous arrays.
 *
 * @param array Pointer to the array.
 * @return 0 on success, ERROR or ERROR_CONTIGUOUS on failure.
 */
int array_flatten(Array *array);

// =============================================================================
//                          Array Printing
// =============================================================================

/**
 * @brief Print an array to stdout.
 *
 * Prints the contents of any Array in a nested bracket format, e.g.:
 *   [[1, 2, 3], [4, 5, 6]]
 *
 * Supports all 10 NUMC_TYPE types and handles non-contiguous arrays
 * (slices, transposes) correctly via stride-based indexing.
 *
 * @param array Pointer to the array to print.
 */
void array_print(const Array *array);

#endif
