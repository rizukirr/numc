#ifndef NUMC_MATH_H
#define NUMC_MATH_H

/* Forward declaration — no struct access needed */
typedef struct NumcArray NumcArray;

/* Element-wise binary: out = a op b
 * Works on contiguous AND non-contiguous (views, slices, transposes).
 * All return 0 on success, negative error code on failure. */

/**
 * @brief Element-wise addition: out = a + b.
 *
 * @param a   First input array.
 * @param b   Second input array.
 * @param out Output array.
 * @return 0 on success, negative error code on failure.
 */
int numc_add(const NumcArray *a, const NumcArray *b, NumcArray *out);

/**
 * @brief Element-wise subtraction: out = a - b.
 *
 * @param a   First input array.
 * @param b   Second input array.
 * @param out Output array.
 * @return 0 on success, negative error code on failure.
 */
int numc_sub(const NumcArray *a, const NumcArray *b, NumcArray *out);

/**
 * @brief Element-wise multiplication: out = a * b.
 *
 * @param a   First input array.
 * @param b   Second input array.
 * @param out Output array.
 * @return 0 on success, negative error code on failure.
 */
int numc_mul(const NumcArray *a, const NumcArray *b, NumcArray *out);

/**
 * @brief Element-wise division: out = a / b.
 *
 * @param a   First input array.
 * @param b   Second input array.
 * @param out Output array.
 * @return 0 on success, negative error code on failure.
 */
int numc_div(const NumcArray *a, const NumcArray *b, NumcArray *out);

/* Element-wise scalar: out = a op scalar
 * Scalar is converted to a's dtype before the operation.
 * Works on contiguous AND non-contiguous arrays. */

/**
 * @brief Element-wise scalar addition: out = a + scalar.
 *
 * @param a      Input array.
 * @param scalar Scalar value.
 * @param out    Output array.
 * @return 0 on success, negative error code on failure.
 */
int numc_add_scalar(const NumcArray *a, double scalar, NumcArray *out);

/**
 * @brief Element-wise scalar subtraction: out = a - scalar.
 *
 * @param a      Input array.
 * @param scalar Scalar value.
 * @param out    Output array.
 * @return 0 on success, negative error code on failure.
 */
int numc_sub_scalar(const NumcArray *a, double scalar, NumcArray *out);

/**
 * @brief Element-wise scalar multiplication: out = a * scalar.
 *
 * @param a      Input array.
 * @param scalar Scalar value.
 * @param out    Output array.
 * @return 0 on success, negative error code on failure.
 */
int numc_mul_scalar(const NumcArray *a, double scalar, NumcArray *out);

/**
 * @brief Element-wise scalar division: out = a / scalar.
 *
 * @param a      Input array.
 * @param scalar Scalar value.
 * @param out    Output array.
 * @return 0 on success, negative error code on failure.
 */
int numc_div_scalar(const NumcArray *a, double scalar, NumcArray *out);

/* Element-wise scalar: in-place operations */

/**
 * @brief Element-wise scalar addition in-place: a += scalar.
 *
 * @param a      Array to be modified.
 * @param scalar Scalar value.
 * @return 0 on success, negative error code on failure.
 */
int numc_add_scalar_inplace(NumcArray *a, double scalar);

/**
 * @brief Element-wise scalar subtraction in-place: a -= scalar.
 *
 * @param a      Array to be modified.
 * @param scalar Scalar value.
 * @return 0 on success, negative error code on failure.
 */
int numc_sub_scalar_inplace(NumcArray *a, double scalar);

/**
 * @brief Element-wise scalar multiplication in-place: a *= scalar.
 *
 * @param a      Array to be modified.
 * @param scalar Scalar value.
 * @return 0 on success, negative error code on failure.
 */
int numc_mul_scalar_inplace(NumcArray *a, double scalar);

/**
 * @brief Element-wise scalar division in-place: a /= scalar.
 *
 * @param a      Array to be modified.
 * @param scalar Scalar value.
 * @return 0 on success, negative error code on failure.
 */
int numc_div_scalar_inplace(NumcArray *a, double scalar);

/**
 * @brief Element-wise power: out = a ^ b.
 *
 * @param a   Base array.
 * @param b   Exponent array.
 * @param out Output array.
 * @return 0 on success, negative error code on failure.
 */
int numc_pow(NumcArray *a, NumcArray *b, NumcArray *out);

/**
 * @brief Element-wise power in-place: a = a ^ b.
 *
 * @param a Base array (modified).
 * @param b Exponent array.
 * @return 0 on success, negative error code on failure.
 */
int numc_pow_inplace(NumcArray *a, NumcArray *b);

/* Element-wise unary: out = op a
 * Works on contiguous AND non-contiguous arrays. */

/**
 * @brief Element-wise negation: out = -a.
 *
 * @param a   Input array.
 * @param out Output array.
 * @return 0 on success, negative error code on failure.
 */
int numc_neg(NumcArray *a, NumcArray *out);

/**
 * @brief Element-wise negation in-place: a = -a.
 *
 * @param a Array to be modified.
 * @return 0 on success, negative error code on failure.
 */
int numc_neg_inplace(NumcArray *a);

/**
 * @brief Element-wise absolute value: out = |a|.
 *
 * @param a   Input array.
 * @param out Output array.
 * @return 0 on success, negative error code on failure.
 */
int numc_abs(NumcArray *a, NumcArray *out);

/**
 * @brief Element-wise absolute value in-place: a = |a|.
 *
 * @param a Array to be modified.
 * @return 0 on success, negative error code on failure.
 */
int numc_abs_inplace(NumcArray *a);

/**
 * @brief Element-wise natural logarithm: out = ln(a).
 *
 * @param a   Input array.
 * @param out Output array.
 * @return 0 on success, negative error code on failure.
 */
int numc_log(NumcArray *a, NumcArray *out);

/**
 * @brief Element-wise natural logarithm in-place: a = ln(a).
 *
 * @param a Array to be modified.
 * @return 0 on success, negative error code on failure.
 */
int numc_log_inplace(NumcArray *a);

/**
 * @brief Element-wise exponential: out = exp(a).
 *
 * @param a   Input array.
 * @param out Output array.
 * @return 0 on success, negative error code on failure.
 */
int numc_exp(NumcArray *a, NumcArray *out);

/**
 * @brief Element-wise exponential in-place: a = exp(a).
 *
 * @param a Array to be modified.
 * @return 0 on success, negative error code on failure.
 */
int numc_exp_inplace(NumcArray *a);

/**
 * @brief Element-wise square root: out = sqrt(a).
 *
 * @param a   Input array.
 * @param out Output array.
 * @return 0 on success, negative error code on failure.
 */
int numc_sqrt(NumcArray *a, NumcArray *out);

/**
 * @brief Element-wise square root in-place: a = sqrt(a).
 *
 * @param a Array to be modified.
 * @return 0 on success, negative error code on failure.
 */
int numc_sqrt_inplace(NumcArray *a);

/**
 * @brief Clip array values to a range: out = clip(a, min, max).
 *
 * @param a   Input array.
 * @param out Output array.
 * @param min Minimum value.
 * @param max Maximum value.
 * @return 0 on success, negative error code on failure.
 */
int numc_clip(NumcArray *a, NumcArray *out, double min, double max);

/**
 * @brief Clip array values to a range in-place: a = clip(a, min, max).
 *
 * @param a   Array to be modified.
 * @param min Minimum value.
 * @param max Maximum value.
 * @return 0 on success, negative error code on failure.
 */
int numc_clip_inplace(NumcArray *a, double min, double max);

/**
 * @brief Element-wise maximum: out = max(a, b).
 *
 * @param a   First input array.
 * @param b   Second input array.
 * @param out Output array.
 * @return 0 on success, negative error code on failure.
 */
int numc_maximum(const NumcArray *a, const NumcArray *b, NumcArray *out);

/**
 * @brief Element-wise maximum in-place: a = max(a, b).
 *
 * @param a Base array (modified).
 * @param b Second input array.
 * @return 0 on success, negative error code on failure.
 */
int numc_maximum_inplace(NumcArray *a, const NumcArray *b);

/**
 * @brief Element-wise minimum: out = min(a, b).
 *
 * @param a   First input array.
 * @param b   Second input array.
 * @param out Output array.
 * @return 0 on success, negative error code on failure.
 */
int numc_minimum(const NumcArray *a, const NumcArray *b, NumcArray *out);

/**
 * @brief Element-wise minimum in-place: a = min(a, b).
 *
 * @param a Base array (modified).
 * @param b Second input array.
 * @return 0 on success, negative error code on failure.
 */
int numc_minimum_inplace(NumcArray *a, const NumcArray *b);

/**
 * @brief Sum of all array elements.
 *
 * @param a   Input array.
 * @param out Output array (0-d or 1-element array).
 * @return 0 on success, negative error code on failure.
 */
int numc_sum(const NumcArray *a, NumcArray *out);

/**
 * @brief Sum of array elements along a given axis.
 *
 * @param a       Input array.
 * @param axis    Axis along which to sum.
 * @param keepdim If true, the reduced axis is kept with size 1.
 * @param out     Output array.
 * @return 0 on success, negative error code on failure.
 */
int numc_sum_axis(const NumcArray *a, int axis, int keepdim, NumcArray *out);

/**
 * @brief Mean of all array elements.
 *
 * @param a   Input array.
 * @param out Output array.
 * @return 0 on success, negative error code on failure.
 */
int numc_mean(const NumcArray *a, NumcArray *out);

/**
 * @brief Mean of array elements along a given axis.
 *
 * @param a       Input array.
 * @param axis    Axis along which to mean.
 * @param keepdim If true, the reduced axis is kept with size 1.
 * @param out     Output array.
 * @return 0 on success, negative error code on failure.
 */
int numc_mean_axis(const NumcArray *a, int axis, int keepdim, NumcArray *out);

/**
 * @brief Maximum of all array elements.
 *
 * @param a   Input array.
 * @param out Output array.
 * @return 0 on success, negative error code on failure.
 */
int numc_max(const NumcArray *a, NumcArray *out);

/**
 * @brief Maximum of array elements along a given axis.
 *
 * @param a       Input array.
 * @param axis    Axis along which to find maximum.
 * @param keepdim If true, the reduced axis is kept with size 1.
 * @param out     Output array.
 * @return 0 on success, negative error code on failure.
 */
int numc_max_axis(const NumcArray *a, int axis, int keepdim, NumcArray *out);

/**
 * @brief Minimum of all array elements.
 *
 * @param a   Input array.
 * @param out Output array.
 * @return 0 on success, negative error code on failure.
 */
int numc_min(const NumcArray *a, NumcArray *out);

/**
 * @brief Minimum of array elements along a given axis.
 *
 * @param a       Input array.
 * @param axis    Axis along which to find minimum.
 * @param keepdim If true, the reduced axis is kept with size 1.
 * @param out     Output array.
 * @return 0 on success, negative error code on failure.
 */
int numc_min_axis(const NumcArray *a, int axis, int keepdim, NumcArray *out);

/**
 * @brief Index of the maximum element in the flattened array.
 *
 * @param a   Input array.
 * @param out Output array (must be of integer type).
 * @return 0 on success, negative error code on failure.
 */
int numc_argmax(const NumcArray *a, NumcArray *out);

/**
 * @brief Indices of the maximum elements along a given axis.
 *
 * @param a       Input array.
 * @param axis    Axis along which to find argmax.
 * @param keepdim If true, the reduced axis is kept with size 1.
 * @param out     Output array (must be of integer type).
 * @return 0 on success, negative error code on failure.
 */
int numc_argmax_axis(const NumcArray *a, int axis, int keepdim, NumcArray *out);

/**
 * @brief Index of the minimum element in the flattened array.
 *
 * @param a   Input array.
 * @param out Output array (must be of integer type).
 * @return 0 on success, negative error code on failure.
 */
int numc_argmin(const NumcArray *a, NumcArray *out);

/**
 * @brief Indices of the minimum elements along a given axis.
 *
 * @param a       Input array.
 * @param axis    Axis along which to find argmin.
 * @param keepdim If true, the reduced axis is kept with size 1.
 * @param out     Output array (must be of integer type).
 * @return 0 on success, negative error code on failure.
 */
int numc_argmin_axis(const NumcArray *a, int axis, int keepdim, NumcArray *out);

/**
 * @brief Naive matrix multiplication: out = a @ b.
 *
 * @param a   First input matrix.
 * @param b   Second input matrix.
 * @param out Output matrix.
 * @return 0 on success, negative error code on failure.
 */
int numc_matmul_naive(const NumcArray *a, const NumcArray *b, NumcArray *out);

#endif
