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
#include <stddef.h>

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

/**
 * @brief Create a new array with the specified shape and data type.
 *
 * @param ndim  Number of dimensions.
 * @param shape Array of dimension sizes.
 * @param dtype Data type of array elements.
 * @return Pointer to the new array, or NULL on failure.
 */
Array *array_create(size_t ndim, const size_t *shape, DType dtype);

/**
 * @brief Create a new array from existing contiguous data.
 *
 * Copies data from a contiguous buffer (like a C array) into a new Array.
 * The data is assumed to be in row-major (C) order.
 *
 * @param ndim  Number of dimensions.
 * @param shape Array of dimension sizes.
 * @param dtype Data type of array elements.
 * @param data  Pointer to contiguous source data to copy.
 * @return Pointer to the new array, or NULL on failure.
 */
Array *array_batch(size_t ndim, const size_t *shape, DType dtype,
                   const void *data);

/**
 * @brief Free an array and its associated memory.
 *
 * Only frees the data buffer if the array owns it.
 *
 * @param array Pointer to the array to free.
 */
void array_free(Array *array);

/**
 * @brief Get a pointer to an element using variadic indices.
 *
 * @warning This function has overhead from variadic argument processing.
 *          Use array_get_ptr() in performance-critical loops.
 *
 * @param array Pointer to the array.
 * @param ...   Indices for each dimension (as size_t values).
 * @return Pointer to the element, or NULL on error.
 */
void *array_get(Array *array, ...);

/**
 * @brief Compute the byte offset for the given indices.
 *
 * @param array   Pointer to the array.
 * @param indices Array of indices for each dimension.
 * @return Byte offset from the start of data.
 */
size_t array_offset(const Array *array, const size_t *indices);

/**
 * @brief Get a pointer to an element at the specified indices.
 *
 * @param array   Pointer to the array.
 * @param indices Array of indices for each dimension.
 * @return Pointer to the element.
 */
void *array_at(const Array *array, const size_t *indices);

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
 * @brief Get the total number of elements in an array.
 *
 * @param array Pointer to the array.
 * @return Total number of elements.
 */
size_t array_size(const Array *array);

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
 * @brief Check if an array is contiguous in memory.
 *
 * @param array Pointer to the array.
 * @return Non-zero if contiguous, 0 otherwise.
 */
int array_is_contiguous(const Array *array);

/**
 * @brief Create a contiguous copy of an array.
 *
 * @param src Pointer to the source array.
 * @return Pointer to a new contiguous array, or NULL on failure.
 */
Array *array_copy(const Array *src);

/**
 * @brief Append a single element to a 1D array.
 *
 * Reallocates the array to accommodate the new element.
 * Only works on 1D contiguous arrays that own their data.
 *
 * @param array Pointer to the array.
 * @param elem  Pointer to the element to append.
 * @return 0 on success, ERROR on failure.
 */
int array_append(Array *array, const void *elem);

/**
 * @brief Concatenate two arrays along a specified axis.
 *
 * Creates a new array containing the elements of both arrays
 * joined along the given axis. Both arrays must have the same
 * shape except in the concatenation axis.
 *
 * @param a    Pointer to the first array.
 * @param b    Pointer to the second array.
 * @param axis Axis along which to concatenate.
 * @return Pointer to a new concatenated array, or NULL on failure.
 */
Array *array_concat(const Array *a, const Array *b, size_t axis);

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
 * @param ndim  Number of dimensions.
 * @param shape Shape of the array.
 * @param dtype Data type of array elements.
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
Array *array_fill(size_t ndim, const size_t *shape, DType dtype, const void *elem);

#endif
