#ifndef NUMC_SHAPE_H
#define NUMC_SHAPE_H

#include "array.h"

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

#endif
