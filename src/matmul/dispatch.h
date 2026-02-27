#ifndef NUMC_MATH_MATMUL_DISPACTH_H
#define NUMC_MATH_MATMUL_DISPACTH_H

#include "internal.h"
#include "numc/error.h"
#include <stdint.h>

/**
 * @brief Validate shapes and types for matrix multiplication.
 *
 * @param a   First input matrix.
 * @param b   Second input matrix.
 * @param out Output matrix.
 * @return 0 on success, negative error code on failure.
 */
static inline int _check_matmul(const struct NumcArray *a,
                                const struct NumcArray *b,
                                struct NumcArray *out) {
  if (!a || !b || !out) {
    NUMC_SET_ERROR(NUMC_ERR_NULL, "matmul: NULL pointer (a=%p b=%p out=%p)", a,
                   b, out);
    return NUMC_ERR_NULL;
  }

  if (a->dtype != b->dtype || a->dtype != out->dtype) {
    NUMC_SET_ERROR(NUMC_ERR_TYPE, "matmul: dtype mismatch (a=%d b=%d out=%d)",
                   a->dtype, b->dtype, out->dtype);
    return NUMC_ERR_TYPE;
  }

  if (a->dim != 2 || b->dim != 2 || out->dim != 2) {
    NUMC_SET_ERROR(NUMC_ERR_SHAPE,
                   "matmul: ndim mismatch (a.dim=%zu b.dim=%zu out.dim=%zu)",
                   a->dim, b->dim, out->dim);
    return NUMC_ERR_SHAPE;
  }

  if (a->shape[1] != b->shape[0]) {
    NUMC_SET_ERROR(NUMC_ERR_SHAPE,
                   "matmul: shape mismatch (a.shape[1]=%zu b.shape[0]=%zu) "
                   "expected a is (M, K) and b is (K, N)",
                   a->shape[1], b->shape[0]);
    return NUMC_ERR_SHAPE;
  }

  if (out->shape[0] != a->shape[0] || out->shape[1] != b->shape[1]) {
    NUMC_SET_ERROR(NUMC_ERR_SHAPE,
                   "matmul: shape mismatch (out.shape[0]=%zu out.shape[1]=%zu) "
                   "expected a is(M, K) and b is (K, N) and out is (M, N)",
                   out->shape[0], out->shape[1]);
    return NUMC_ERR_SHAPE;
  }

  return 0;
}

#endif
