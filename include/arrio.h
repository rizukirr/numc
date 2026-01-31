/**
 * @file arrio.h
 * @brief Input/Output operations for the Array library.
 *
 * Provides functions for printing and displaying array contents.
 */

#ifndef ARRIO_H
#define ARRIO_H

#include <stddef.h>

/**
 * Prints the contents of an array.
 *
 * @param data The array data.
 * @param shape The array shape.
 * @param ndim The number of dimensions.
 * @param dim The current dimension.
 * @param offset The current offset.
 */
void array_print(const float *data, const size_t *shape, size_t ndim,
                 size_t dim, size_t *offset);

#endif
