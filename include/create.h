#ifndef NUMC_CREATE_H
#define NUMC_CREATE_H

#include "types.h"
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

#endif
