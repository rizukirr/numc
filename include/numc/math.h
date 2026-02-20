#ifndef NUMC_MATH_H
#define NUMC_MATH_H

/* Forward declaration — no struct access needed */
typedef struct NumcArray NumcArray;

/* Element-wise binary: out = a op b
 * Works on contiguous AND non-contiguous (views, slices, transposes).
 * All return 0 on success, negative error code on failure. */
int numc_add(const NumcArray *a, const NumcArray *b, NumcArray *out);
int numc_sub(const NumcArray *a, const NumcArray *b, NumcArray *out);
int numc_mul(const NumcArray *a, const NumcArray *b, NumcArray *out);
int numc_div(const NumcArray *a, const NumcArray *b, NumcArray *out);

/* Element-wise scalar: out = a op scalar
 * Scalar is converted to a's dtype before the operation.
 * Works on contiguous AND non-contiguous arrays. */
int numc_add_scalar(const NumcArray *a, double scalar, NumcArray *out);
int numc_sub_scalar(const NumcArray *a, double scalar, NumcArray *out);
int numc_mul_scalar(const NumcArray *a, double scalar, NumcArray *out);
int numc_div_scalar(const NumcArray *a, double scalar, NumcArray *out);

/* Element-wise scalar: in-place operations */
int numc_add_scalar_inplace(NumcArray *a, double scalar);
int numc_sub_scalar_inplace(NumcArray *a, double scalar);
int numc_mul_scalar_inplace(NumcArray *a, double scalar);
int numc_div_scalar_inplace(NumcArray *a, double scalar);

int numc_pow(NumcArray *a, NumcArray *b, NumcArray *out);
int numc_pow_inplace(NumcArray *a, NumcArray *b);

/* Element-wise unary: out = op a
 * Works on contiguous AND non-contiguous arrays. */
int numc_neg(NumcArray *a, NumcArray *out);
int numc_neg_inplace(NumcArray *a);

int numc_abs(NumcArray *a, NumcArray *out);
int numc_abs_inplace(NumcArray *a);

int numc_log(NumcArray *a, NumcArray *out);
int numc_log_inplace(NumcArray *a);

int numc_exp(NumcArray *a, NumcArray *out);
int numc_exp_inplace(NumcArray *a);

int numc_sqrt(NumcArray *a, NumcArray *out);
int numc_sqrt_inplace(NumcArray *a);

int numc_clip(NumcArray *a, NumcArray *out, double min, double max);
int numc_clip_inplace(NumcArray *a, double min, double max);

int numc_maximum(const NumcArray *a, const NumcArray *b, NumcArray *out);
int numc_maximum_inplace(NumcArray *a, const NumcArray *b);

int numc_minimum(const NumcArray *a, const NumcArray *b, NumcArray *out);
int numc_minimum_inplace(NumcArray *a, const NumcArray *b);

int numc_sum(const NumcArray *a, NumcArray *out);
int numc_sum_axis(const NumcArray *a, int axis, int keepdim, NumcArray *out);

#endif
