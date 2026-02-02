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

#include "dtype.h"
#include <assert.h>
#include <stddef.h>

#define MAX_STACK_NDIM 8

/**
 * @brief Multi-dimensional array structure.
 *
 * @param data      Pointer to the raw data buffer.
 * @param shape     Array of dimension sizes (length = ndim).
 * @param strides   Array of byte strides for each dimension (length = ndim).
 * @param ndim      Number of dimensions.
 * @param dtype     Data type of array elements.
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
  DType dtype;
  size_t elem_size;
  size_t size;
  size_t capacity;
  int owns_data;
} Array;

// -----------------------------------------------------------------------------
//                              Array Creation
// -----------------------------------------------------------------------------

/**
 * @brief Create a new array with the specified shape and data type.
 *
 * @param ndim  Number of dimensions.
 * @param shape Array of dimension sizes.
 * @param dtype Data type of array elements.
 * @param data  Pointer to contiguous source data to copy.
 * @return Pointer to the new array, or NULL on failure.
 */
Array *array_create(size_t ndim, const size_t *shape, DType dtype,
                    const void *data);

/**
 * @brief Create an array filled with zeros.
 *
 * @param ndim  Number of dimensions.
 * @param shape Shape of the array.
 * @param dtype Data type of array elements.
 * @return Pointer to a new array, or NULL on failure.
 */
Array *array_zeros(size_t ndim, const size_t *shape, DType dtype);

/**
 * @brief Create an array filled with ones.
 *
 * @param ndim      Number of dimensions.
 * @param shape     Shape of the array.
 * @param dtype     Data type of array elements.
 * @return Pointer to a new array, or NULL on failure.
 */
Array *array_ones(size_t ndim, const size_t *shape, DType dtype);

/**
 * @brief Create an array filled with a single value.
 *
 * @param ndim  Number of dimensions.
 * @param shape Shape of the array.
 * @param dtype Data type of array elements.
 * @param elem  Pointer to the element to fill with.
 * @return Pointer to a new array, or NULL on failure.
 */
Array *array_fill(size_t ndim, const size_t *shape, DType dtype,
                  const void *elem);

/**
 * @brief Free an array and its associated memory.
 *
 * Only frees the data buffer if the array owns it.
 *
 * @param array Pointer to the array to free.
 */
void array_free(Array *array);

// -----------------------------------------------------------------------------
//                              Array Access
// -----------------------------------------------------------------------------

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

/**
 * @brief Type-safe helpers for array element access.
 *
 * These macros check the array dtype and return the appropriate pointer type.
 * They are only available when the array dtype is known at compile time.
 */

static inline NUMC_FLOAT *array_getf(const Array *array,
                                     const size_t *indices) {
  assert(array->dtype == DTYPE_FLOAT);
  return (NUMC_FLOAT *)array_get(array, indices);
}

static inline NUMC_INT *array_geti(const Array *array, const size_t *indices) {
  assert(array->dtype == DTYPE_INT);
  return (NUMC_INT *)array_get(array, indices);
}

static inline NUMC_UINT *array_getui(const Array *array,
                                     const size_t *indices) {
  assert(array->dtype == DTYPE_UINT);
  return (NUMC_UINT *)array_get(array, indices);
}

static inline NUMC_LONG *array_getl(const Array *array, const size_t *indices) {
  assert(array->dtype == DTYPE_LONG);
  return (NUMC_LONG *)array_get(array, indices);
}

static inline NUMC_ULONG *array_getul(const Array *array,
                                      const size_t *indices) {
  assert(array->dtype == DTYPE_ULONG);
  return (NUMC_ULONG *)array_get(array, indices);
}

static inline NUMC_SHORT *array_gets(const Array *array,
                                     const size_t *indices) {
  assert(array->dtype == DTYPE_SHORT);
  return (NUMC_SHORT *)array_get(array, indices);
}

static inline NUMC_USHORT *array_getus(const Array *array,
                                       const size_t *indices) {
  assert(array->dtype == DTYPE_USHORT);
  return (NUMC_USHORT *)array_get(array, indices);
}

static inline NUMC_BYTE *array_getb(const Array *array, const size_t *indices) {
  assert(array->dtype == DTYPE_BYTE);
  return (NUMC_BYTE *)array_get(array, indices);
}

static inline NUMC_UBYTE *array_getub(const Array *array,
                                      const size_t *indices) {
  assert(array->dtype == DTYPE_UBYTE);
  return (NUMC_UBYTE *)array_get(array, indices);
}

// -----------------------------------------------------------------------------
//                              Array Properties
// -----------------------------------------------------------------------------

/**
 * @brief Check if an array is contiguous in memory.
 *
 * @param array Pointer to the array.
 * @return Non-zero if contiguous, 0 otherwise.
 */
int array_is_contiguous(const Array *array);

// -----------------------------------------------------------------------------
//                              Array Manipulation
// -----------------------------------------------------------------------------

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
Array *array_to_contiguous(const Array *src);

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
Array *array_concat(const Array *a, const Array *b, size_t axis);

/**
 * @brief Transpose an array in place.
 *
 * Changes the order of the array dimensions.
 *
 * @param array Pointer to the array.
 *
 * @return 0 on success, ERROR or ERROR_DIM on failure.
 */
int array_transpose(Array *array, size_t *axes);

// -----------------------------------------------------------------------------
//                          Mathematical Operations
// -----------------------------------------------------------------------------

/**
 * @brief Element-wise addition.
 *
 * @param a Pointer to the first array.
 * @param b Pointer to the second array.
 *
 * @return Pointer to a new array, or NULL on failure.
 */
Array *array_add(const Array *a, const Array *b);

#endif
