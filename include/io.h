/**
 * @file io.h
 * @brief Input/Output operations for the Array library.
 *
 * Provides functions for printing and displaying array contents.
 */

#ifndef NUMC_IO_H
#define NUMC_IO_H

#include "array.h"

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
