#ifndef NUMC_REDUCTION_DISPATCH_H
#define NUMC_REDUCTION_DISPATCH_H

#include "internal.h"
#include "kernel.h"
#include <numc/array.h>
#include <numc/error.h>

/* ── ND iteration for axis reduction ──────────────────────────────
 *
 * Recursive — at each leaf position, calls the reduction kernel
 * along the reduction axis.
 *
 * iter_shape / sa / so have the reduction axis REMOVED — they
 * describe the iteration space (all dims except the one being reduced).
 * reduce_stride / reduce_len describe the reduction axis itself.
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

/* ── Full reduction dispatch ──────────────────────────────────────── */

static inline void _reduce_full_op(const struct NumcArray *a,
                                   struct NumcArray *out,
                                   const NumcReductionKernel *table) {
  NumcReductionKernel kern = table[a->dtype];

  if (a->is_contiguous) {
    kern((const char *)a->data, (char *)out->data, a->size,
         (intptr_t)a->elem_size);
  } else {
    /* Non-contiguous: copy to contiguous first (arena memory is cheap,
     * non-contiguous full reduction is rare). */
    NumcArray *tmp = numc_array_copy(a);
    numc_array_contiguous(tmp);
    kern((const char *)tmp->data, (char *)out->data, tmp->size,
         (intptr_t)tmp->elem_size);
  }
}

/* ── Axis reduction dispatch ──────────────────────────────────────── */

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

/* ── Validation ───────────────────────────────────────────────────── */

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

#endif
