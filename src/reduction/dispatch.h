#ifndef NUMC_REDUCTION_DISPATCH_H
#define NUMC_REDUCTION_DISPATCH_H

#include "kernel.h"
#include <numc/array.h>
#include <numc/error.h>
#include <string.h>

/* Check if the non-reduced dims of 'a' form a contiguous block.
 * Walk from last dim backwards (skipping reduction axis), verify
 * strides match C-order contiguous layout. */
/**
 * @brief Check if the non-reduced dimensions of an array form a contiguous block.
 *
 * @param a    Input array.
 * @param axis The reduction axis.
 * @return true if contiguous, false otherwise.
 */
static inline bool _iter_contiguous(const struct NumcArray *a, size_t axis) {
  size_t expected = a->elem_size;
  for (size_t i = a->dim; i-- > 0;) {
    if (i == axis)
      continue;
    if (a->strides[i] != expected)
      return false;
    expected *= a->shape[i];
  }
  return true;
}

/**
 * @brief Recursive N-dimensional iteration for axis reduction.
 *
 * @param kern          Reduction kernel function.
 * @param a             Pointer to input data.
 * @param sa            Strides for input iteration space.
 * @param out           Pointer to output data.
 * @param so            Strides for output iteration space.
 * @param iter_shape    Shape of the iteration space (excluding reduction axis).
 * @param iter_ndim     Number of dimensions in the iteration space.
 * @param reduce_stride Stride along the reduction axis.
 * @param reduce_len    Length of the reduction axis.
 */
static inline void _reduce_axis_nd(NumcReductionKernel kern, const char *a,
                                   const size_t *sa, char *out,
                                   const size_t *so, const size_t *iter_shape,
                                   size_t iter_ndim, intptr_t reduce_stride,
                                   size_t reduce_len) {
  if (iter_ndim == 0) {
    kern(a, out, reduce_len, reduce_stride);
    return;
  }
  for (size_t i = 0; i < iter_shape[0]; i++) {
    _reduce_axis_nd(kern, a + i * sa[0], sa + 1, out + i * so[0], so + 1,
                    iter_shape + 1, iter_ndim - 1, reduce_stride, reduce_len);
  }
}

/**
 * @brief Dispatch a full reduction operation (all elements reduced to a scalar).
 *
 * @param a     Input array.
 * @param out   Output array (scalar).
 * @param table Reduction kernel function table.
 */
static inline void _reduce_full_op(const struct NumcArray *a,
                                   struct NumcArray *out,
                                   const NumcReductionKernel *table) {
  NumcReductionKernel kern = table[a->dtype];

  if (a->is_contiguous) {
    kern((const char *)a->data, (char *)out->data, a->size,
         (intptr_t)a->elem_size);
  } else {
    /* Non-contiguous: create a contiguous copy directly.
     * Uses dimension collapsing for efficient chunk copying.
     * Single allocation (vs. numc_array_copy + numc_array_contiguous
     * which allocates twice and misses elements for strided views). */
    size_t cap = a->size * a->elem_size;
    char *buf =
        (char *)arena_alloc(a->ctx->arena, cap, NUMC_SIMD_ALIGN);
    if (!buf)
      return;

    /* Dimension collapse: merge adjacent contiguous dims */
    size_t c_shape[NUMC_MAX_DIMENSIONS];
    size_t c_strides[NUMC_MAX_DIMENSIONS];
    c_shape[0] = a->shape[0];
    c_strides[0] = a->strides[0];
    size_t cdim = 1;
    for (size_t i = 1; i < a->dim; i++) {
      if (c_strides[cdim - 1] == a->strides[i] * a->shape[i]) {
        c_shape[cdim - 1] *= a->shape[i];
        c_strides[cdim - 1] = a->strides[i];
      } else {
        c_shape[cdim] = a->shape[i];
        c_strides[cdim] = a->strides[i];
        cdim++;
      }
    }

    char *dst = buf;
    size_t coord[NUMC_MAX_DIMENSIONS] = {0};

    if (c_strides[cdim - 1] == a->elem_size) {
      /* Inner dim is contiguous → memcpy whole chunks */
      size_t chunk = c_shape[cdim - 1] * a->elem_size;
      size_t outer = a->size / c_shape[cdim - 1];
      for (size_t i = 0; i < outer; i++) {
        const char *src = (const char *)a->data;
        for (size_t d = 0; d < cdim - 1; d++)
          src += coord[d] * c_strides[d];
        memcpy(dst, src, chunk);
        dst += chunk;
        for (int d = (int)cdim - 2; d >= 0; d--) {
          if (++coord[d] < c_shape[d])
            break;
          coord[d] = 0;
        }
      }
    } else {
      /* No contiguous inner dim → element-wise copy */
      for (size_t i = 0; i < a->size; i++) {
        const char *src = (const char *)a->data;
        for (size_t d = 0; d < cdim; d++)
          src += coord[d] * c_strides[d];
        memcpy(dst, src, a->elem_size);
        dst += a->elem_size;
        for (int d = (int)cdim - 1; d >= 0; d--) {
          if (++coord[d] < c_shape[d])
            break;
          coord[d] = 0;
        }
      }
    }

    kern((const char *)buf, (char *)out->data, a->size,
         (intptr_t)a->elem_size);
  }
}

/**
 * @brief Dispatch a reduction operation along a specific axis.
 *
 * @param a       Input array.
 * @param axis    The axis to reduce.
 * @param keepdim If true, the reduced axis is kept with size 1.
 * @param out     Output array.
 * @param table   Reduction kernel function table.
 */
static inline void _reduce_axis_op(const struct NumcArray *a, size_t axis,
                                   int keepdim, struct NumcArray *out,
                                   const NumcReductionKernel *table) {
  NumcReductionKernel kern = table[a->dtype];
  size_t reduce_len = a->shape[axis];
  intptr_t reduce_stride = (intptr_t)a->strides[axis];

  /* Build iteration arrays: shape/strides with reduction axis removed.
   * For input (a), skip the reduction axis.
   * For output, skip the reduction axis when keepdim (it has size 1 there),
   * otherwise out already has ndim-1 dims so index sequentially. */
  size_t iter_ndim = a->dim - 1;
  size_t iter_shape[NUMC_MAX_DIMENSIONS];
  size_t iter_sa[NUMC_MAX_DIMENSIONS];
  size_t iter_so[NUMC_MAX_DIMENSIONS];

  for (size_t i = 0, j = 0, k = 0; i < a->dim; i++) {
    if (i == axis)
      continue;
    iter_shape[j] = a->shape[i];
    iter_sa[j] = a->strides[i];
    /* When keepdim, out has same ndim as a — skip axis in out too.
     * When !keepdim, out has ndim-1 — use sequential index k. */
    if (keepdim) {
      /* k tracks out dim, skip when k == axis */
      while (k == axis)
        k++;
      iter_so[j] = out->strides[k];
      k++;
    } else {
      iter_so[j] = out->strides[j];
    }
    j++;
  }

  if (iter_ndim == 0) {
    /* Reducing a 1D array along axis 0 → scalar */
    kern((const char *)a->data, (char *)out->data, reduce_len, reduce_stride);
  } else {
    _reduce_axis_nd(kern, (const char *)a->data, iter_sa, (char *)out->data,
                    iter_so, iter_shape, iter_ndim, reduce_stride, reduce_len);
  }
}

/**
 * @brief Validate full reduction operation.
 *
 * @param a   Input array.
 * @param out Output array.
 * @return 0 on success, negative error code on failure.
 */
static inline int _check_reduce_full(const struct NumcArray *a,
                                     const struct NumcArray *out) {
  if (!a || !out) {
    NUMC_SET_ERROR(NUMC_ERR_NULL, "reduce full: NULL pointer (a=%p out=%p)", a, out);
    return NUMC_ERR_NULL;
  }
  if (a->dtype != out->dtype) {
    NUMC_SET_ERROR(NUMC_ERR_TYPE, "reduce full: dtype mismatch (a=%d out=%d)", a->dtype, out->dtype);
    return NUMC_ERR_TYPE;
  }
  if (out->size != 1) {
    NUMC_SET_ERROR(NUMC_ERR_SHAPE, "reduce full: output not scalar (out.size=%zu)", out->size);
    return NUMC_ERR_SHAPE;
  }
  return 0;
}

/**
 * @brief Validate axis reduction operation.
 *
 * @param a       Input array.
 * @param axis    The reduction axis.
 * @param keepdim If true, the reduced axis is kept with size 1.
 * @param out     Output array.
 * @return 0 on success, negative error code on failure.
 */
static inline int _check_reduce_axis(const struct NumcArray *a, int axis,
                                     int keepdim,
                                     const struct NumcArray *out) {
  if (!a || !out) {
    NUMC_SET_ERROR(NUMC_ERR_NULL, "reduce axis: NULL pointer (a=%p out=%p)", a, out);
    return NUMC_ERR_NULL;
  }
  if (a->dtype != out->dtype) {
    NUMC_SET_ERROR(NUMC_ERR_TYPE, "reduce axis: dtype mismatch (a=%d out=%d)", a->dtype, out->dtype);
    return NUMC_ERR_TYPE;
  }
  if (axis < 0 || (size_t)axis >= a->dim) {
    NUMC_SET_ERROR(NUMC_ERR_SHAPE, "reduce axis: invalid axis %d (ndim=%zu)", axis, a->dim);
    return NUMC_ERR_SHAPE;
  }

  if (keepdim) {
    /* Output has same ndim, with shape[axis] == 1 */
    if (out->dim != a->dim)
      {
        NUMC_SET_ERROR(NUMC_ERR_SHAPE, "reduce axis: keepdim expects out.dim == a.dim (out.dim=%zu a.dim=%zu)", out->dim, a->dim);
        return NUMC_ERR_SHAPE;
      }
    for (size_t i = 0; i < a->dim; i++) {
      if (i == (size_t)axis) {
        if (out->shape[i] != 1) {
          NUMC_SET_ERROR(NUMC_ERR_SHAPE, "reduce axis: expected out.shape[%zu] == 1 (got %zu)", i, out->shape[i]);
          return NUMC_ERR_SHAPE;
        }
      } else {
        if (out->shape[i] != a->shape[i]) {
          NUMC_SET_ERROR(NUMC_ERR_SHAPE, "reduce axis: shape mismatch at dim %zu (a=%zu out=%zu)", i, a->shape[i], out->shape[i]);
          return NUMC_ERR_SHAPE;
        }
      }
    }
  } else {
    /* Output has ndim-1 dims matching a's non-reduced dims */
    size_t expected_ndim = a->dim - 1;
    if (expected_ndim == 0) {
      if (out->size != 1) {
        NUMC_SET_ERROR(NUMC_ERR_SHAPE, "reduce axis: expected scalar output for 1D reduction (out.size=%zu)", out->size);
        return NUMC_ERR_SHAPE;
      }
    } else {
      if (out->dim != expected_ndim)
        {
          NUMC_SET_ERROR(NUMC_ERR_SHAPE, "reduce axis: expected out.dim=%zu but got %zu", expected_ndim, out->dim);
          return NUMC_ERR_SHAPE;
        }
      for (size_t i = 0, j = 0; i < a->dim; i++) {
        if (i == (size_t)axis)
          continue;
        if (out->shape[j] != a->shape[i]) {
          NUMC_SET_ERROR(NUMC_ERR_SHAPE, "reduce axis: shape mismatch at output dim %zu (a=%zu out=%zu)", j, a->shape[i], out->shape[j]);
          return NUMC_ERR_SHAPE;
        }
        j++;
      }
    }
  }
  return 0;
}

/**
 * @brief Validate full arg-reduction operation (argmax/argmin).
 *
 * @param a   Input array.
 * @param out Output array (must be INT64 scalar).
 * @return 0 on success, negative error code on failure.
 */
static inline int _check_argreduce_full(const struct NumcArray *a,
                                        const struct NumcArray *out) {
  if (!a || !out) {
    NUMC_SET_ERROR(NUMC_ERR_NULL, "argreduce full: NULL pointer (a=%p out=%p)",
                   a, out);
    return NUMC_ERR_NULL;
  }
  if (out->dtype != NUMC_DTYPE_INT64) {
    NUMC_SET_ERROR(NUMC_ERR_TYPE,
                   "argreduce full: output dtype must be INT64 (got %d)",
                   out->dtype);
    return NUMC_ERR_TYPE;
  }
  if (out->size != 1) {
    NUMC_SET_ERROR(NUMC_ERR_SHAPE,
                   "argreduce full: output not scalar (out.size=%zu)",
                   out->size);
    return NUMC_ERR_SHAPE;
  }
  return 0;
}

/**
 * @brief Validate axis arg-reduction operation.
 *
 * @param a       Input array.
 * @param axis    The reduction axis.
 * @param keepdim If true, the reduced axis is kept with size 1.
 * @param out     Output array (must be INT64).
 * @return 0 on success, negative error code on failure.
 */
static inline int _check_argreduce_axis(const struct NumcArray *a, int axis,
                                        int keepdim,
                                        const struct NumcArray *out) {
  if (!a || !out) {
    NUMC_SET_ERROR(NUMC_ERR_NULL,
                   "argreduce axis: NULL pointer (a=%p out=%p)", a, out);
    return NUMC_ERR_NULL;
  }
  if (out->dtype != NUMC_DTYPE_INT64) {
    NUMC_SET_ERROR(NUMC_ERR_TYPE,
                   "argreduce axis: output dtype must be INT64 (got %d)",
                   out->dtype);
    return NUMC_ERR_TYPE;
  }
  if (axis < 0 || (size_t)axis >= a->dim) {
    NUMC_SET_ERROR(NUMC_ERR_SHAPE,
                   "argreduce axis: invalid axis %d (ndim=%zu)", axis, a->dim);
    return NUMC_ERR_SHAPE;
  }

  if (keepdim) {
    if (out->dim != a->dim) {
      NUMC_SET_ERROR(
          NUMC_ERR_SHAPE,
          "argreduce axis: keepdim expects out.dim == a.dim (out.dim=%zu "
          "a.dim=%zu)",
          out->dim, a->dim);
      return NUMC_ERR_SHAPE;
    }
    for (size_t i = 0; i < a->dim; i++) {
      if (i == (size_t)axis) {
        if (out->shape[i] != 1) {
          NUMC_SET_ERROR(
              NUMC_ERR_SHAPE,
              "argreduce axis: expected out.shape[%zu] == 1 (got %zu)", i,
              out->shape[i]);
          return NUMC_ERR_SHAPE;
        }
      } else {
        if (out->shape[i] != a->shape[i]) {
          NUMC_SET_ERROR(NUMC_ERR_SHAPE,
                         "argreduce axis: shape mismatch at dim %zu (a=%zu "
                         "out=%zu)",
                         i, a->shape[i], out->shape[i]);
          return NUMC_ERR_SHAPE;
        }
      }
    }
  } else {
    size_t expected_ndim = a->dim - 1;
    if (expected_ndim == 0) {
      if (out->size != 1) {
        NUMC_SET_ERROR(NUMC_ERR_SHAPE,
                       "argreduce axis: expected scalar output for 1D "
                       "reduction (out.size=%zu)",
                       out->size);
        return NUMC_ERR_SHAPE;
      }
    } else {
      if (out->dim != expected_ndim) {
        NUMC_SET_ERROR(NUMC_ERR_SHAPE,
                       "argreduce axis: expected out.dim=%zu but got %zu",
                       expected_ndim, out->dim);
        return NUMC_ERR_SHAPE;
      }
      for (size_t i = 0, j = 0; i < a->dim; i++) {
        if (i == (size_t)axis)
          continue;
        if (out->shape[j] != a->shape[i]) {
          NUMC_SET_ERROR(NUMC_ERR_SHAPE,
                         "argreduce axis: shape mismatch at output dim %zu "
                         "(a=%zu out=%zu)",
                         j, a->shape[i], out->shape[j]);
          return NUMC_ERR_SHAPE;
        }
        j++;
      }
    }
  }
  return 0;
}

#endif
