#ifndef NUMC_MATH_H
#define NUMC_MATH_H

#include "array.h"

int array_add_float(const Array *a, const Array *b, Array *out);

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
