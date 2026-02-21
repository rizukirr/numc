#ifndef NUMC_MATH_DISPATCH_H
#define NUMC_MATH_DISPATCH_H

#include "kernel.h"
#include <numc/error.h>
#include <string.h>

/* ── ND iteration ──────────────────────────────────────────────────
 *
 * Recursive — calls kernel on innermost dimension.
 * Outer dimensions loop to compute base pointers.
 * Max recursion depth = NUMC_MAX_DIMENSIONS (8).
 *
 * For contiguous arrays this is never called — the flat fast path
 * in _binary_op / _unary_op handles it directly.
 */

static inline void _elemwise_binary_nd(NumcBinaryKernel kern, const char *a,
                                       const size_t *sa, const char *b,
                                       const size_t *sb, char *out,
                                       const size_t *so, const size_t *shape,
                                       size_t ndim) {
  if (ndim == 1) {
    kern(a, b, out, shape[0], (intptr_t)sa[0], (intptr_t)sb[0],
         (intptr_t)so[0]);
    return;
  }

  for (size_t i = 0; i < shape[0]; i++) {
    _elemwise_binary_nd(kern, a + i * sa[0], sa + 1, b + i * sb[0], sb + 1,
                        out + i * so[0], so + 1, shape + 1, ndim - 1);
  }
}

static inline void _elemwise_unary_nd(NumcUnaryKernel kern, const char *a,
                                      const size_t *sa, char *out,
                                      const size_t *so, const size_t *shape,
                                      size_t ndim) {
  if (ndim == 1) {
    kern(a, out, shape[0], (intptr_t)sa[0], (intptr_t)so[0]);
    return;
  }

  for (size_t i = 0; i < shape[0]; i++) {
    _elemwise_unary_nd(kern, a + i * sa[0], sa + 1, out + i * so[0], so + 1,
                       shape + 1, ndim - 1);
  }
}

static inline void _elemwise_clip_nd(NumcClipKernel kern, const char *a,
                                     const size_t *sa, char *out,
                                     const size_t *so, const size_t *shape,
                                     size_t ndim, double min, double max) {
  if (ndim == 1) {
    kern(a, out, shape[0], (intptr_t)sa[0], (intptr_t)so[0], min, max);
    return;
  }

  for (size_t i = 0; i < shape[0]; i++) {
    _elemwise_clip_nd(kern, a + i * sa[0], sa + 1, out + i * so[0], so + 1,
                      shape + 1, ndim - 1, min, max);
  }
}

/* ── Axis sorting for optimal iteration order ────────────────────────
 *
 * Sort dimensions by descending stride sum so the smallest-stride axis
 * ends up innermost (where the kernel runs).  This maximises spatial
 * locality: when two inputs are transposed and one output is contiguous,
 * the sum metric naturally places the axis with the most contiguous
 * operands innermost — matching NumPy's nditer axis-sorting heuristic.
 *
 * Insertion sort (stable, max NUMC_MAX_DIMENSIONS = 8 elements).
 */

static inline void _sort_axes_binary(size_t ndim, const size_t *shape,
                                     const size_t *sa, const size_t *sb,
                                     const size_t *so, size_t *ps, size_t *pa,
                                     size_t *pb, size_t *po) {
  size_t perm[ndim];
  for (size_t i = 0; i < ndim; i++)
    perm[i] = i;

  for (size_t i = 1; i < ndim; i++) {
    size_t key = perm[i];
    size_t kv = sa[key] + sb[key] + so[key];
    size_t j = i;
    while (j > 0) {
      size_t p = perm[j - 1];
      if (sa[p] + sb[p] + so[p] >= kv)
        break;
      perm[j] = perm[j - 1];
      j--;
    }
    perm[j] = key;
  }

  for (size_t i = 0; i < ndim; i++) {
    ps[i] = shape[perm[i]];
    pa[i] = sa[perm[i]];
    pb[i] = sb[perm[i]];
    po[i] = so[perm[i]];
  }
}

static inline void _sort_axes_unary(size_t ndim, const size_t *shape,
                                    const size_t *sa, const size_t *so,
                                    size_t *ps, size_t *pa, size_t *po) {
  size_t perm[ndim];
  for (size_t i = 0; i < ndim; i++)
    perm[i] = i;

  for (size_t i = 1; i < ndim; i++) {
    size_t key = perm[i];
    size_t kv = sa[key] + so[key];
    size_t j = i;
    while (j > 0) {
      size_t p = perm[j - 1];
      if (sa[p] + so[p] >= kv)
        break;
      perm[j] = perm[j - 1];
      j--;
    }
    perm[j] = key;
  }

  for (size_t i = 0; i < ndim; i++) {
    ps[i] = shape[perm[i]];
    pa[i] = sa[perm[i]];
    po[i] = so[perm[i]];
  }
}

/* ── Validation ───────────────────────────────────────────────────── */

static inline int _check_binary(const struct NumcArray *a,
                                const struct NumcArray *b,
                                const struct NumcArray *out) {
  if (!a || !b || !out) {
    NUMC_SET_ERROR(NUMC_ERR_NULL, "binary op: NULL pointer (a=%p b=%p out=%p)",
                   a, b, out);
    return NUMC_ERR_NULL;
  }
  if (a->dtype != b->dtype || a->dtype != out->dtype) {
    NUMC_SET_ERROR(NUMC_ERR_TYPE,
                   "binary op: dtype mismatch (a=%d b=%d out=%d)", a->dtype,
                   b->dtype, out->dtype);
    return NUMC_ERR_TYPE;
  }

  size_t bcast_ndim = a->dim > b->dim ? a->dim : b->dim;
  if (out->dim != bcast_ndim) {
    NUMC_SET_ERROR(
        NUMC_ERR_SHAPE,
        "binary op: output ndim mismatch (expected %zu, got %zu)",
        bcast_ndim, out->dim);
    return NUMC_ERR_SHAPE;
  }

  size_t a_off = bcast_ndim - a->dim;
  size_t b_off = bcast_ndim - b->dim;

  for (size_t i = 0; i < bcast_ndim; i++) {
    size_t da = (i < a_off) ? 1 : a->shape[i - a_off];
    size_t db = (i < b_off) ? 1 : b->shape[i - b_off];

    if (da != db && da != 1 && db != 1) {
      NUMC_SET_ERROR(
          NUMC_ERR_SHAPE,
          "binary op: incompatible broadcast shapes at dim %zu (a=%zu b=%zu)",
          i, da, db);
      return NUMC_ERR_SHAPE;
    }

    size_t expected = da > db ? da : db;
    if (out->shape[i] != expected) {
      NUMC_SET_ERROR(NUMC_ERR_SHAPE,
                     "binary op: output shape mismatch at dim %zu "
                     "(expected %zu, got %zu)",
                     i, expected, out->shape[i]);
      return NUMC_ERR_SHAPE;
    }
  }
  return 0;
}

static inline int _check_unary(const struct NumcArray *a,
                               const struct NumcArray *out) {
  if (!a || !out) {
    NUMC_SET_ERROR(NUMC_ERR_NULL, "unary op: NULL pointer (a=%p out=%p)", a, out);
    return NUMC_ERR_NULL;
  }
  if (a->dtype != out->dtype) {
    NUMC_SET_ERROR(NUMC_ERR_TYPE, "unary op: dtype mismatch (a=%d out=%d)", a->dtype, out->dtype);
    return NUMC_ERR_TYPE;
  }
  if (a->dim != out->dim) {
    NUMC_SET_ERROR(NUMC_ERR_SHAPE, "unary op: ndim mismatch (a.dim=%zu out.dim=%zu)", a->dim, out->dim);
    return NUMC_ERR_SHAPE;
  }
  for (size_t d = 0; d < a->dim; d++) {
    if (a->shape[d] != out->shape[d]) {
      NUMC_SET_ERROR(NUMC_ERR_SHAPE, "unary op: shape mismatch at dim %zu (a=%zu out=%zu)", d, a->shape[d], out->shape[d]);
      return NUMC_ERR_SHAPE;
    }
  }
  return 0;
}

/* ── Binary op dispatch ───────────────────────────────────────────── */

static inline void _binary_op(const struct NumcArray *a,
                              const struct NumcArray *b, struct NumcArray *out,
                              const NumcBinaryKernel *table) {
  NumcBinaryKernel kern = table[a->dtype];
  intptr_t es = (intptr_t)a->elem_size;

  /* Check if broadcasting is needed */
  bool needs_broadcast = (a->dim != b->dim);
  if (!needs_broadcast) {
    for (size_t i = 0; i < a->dim; i++) {
      if (a->shape[i] != b->shape[i]) {
        needs_broadcast = true;
        break;
      }
    }
  }

  if (!needs_broadcast) {
    /* Same shapes — original fast paths */
    if (a->is_contiguous && b->is_contiguous && out->is_contiguous) {
      kern((const char *)a->data, (const char *)b->data, (char *)out->data,
           a->size, es, es, es);
    } else {
      size_t ps[NUMC_MAX_DIMENSIONS], pa[NUMC_MAX_DIMENSIONS],
          pb[NUMC_MAX_DIMENSIONS], po[NUMC_MAX_DIMENSIONS];
      _sort_axes_binary(a->dim, a->shape, a->strides, b->strides, out->strides,
                        ps, pa, pb, po);
      _elemwise_binary_nd(kern, (const char *)a->data, pa,
                          (const char *)b->data, pb, (char *)out->data, po, ps,
                          a->dim);
    }
  } else {
    /* Broadcast path: compute virtual strides (stride=0 where dim==1) */
    size_t bcast_ndim = a->dim > b->dim ? a->dim : b->dim;
    size_t a_off = bcast_ndim - a->dim;
    size_t b_off = bcast_ndim - b->dim;
    size_t va[NUMC_MAX_DIMENSIONS], vb[NUMC_MAX_DIMENSIONS];

    for (size_t i = 0; i < bcast_ndim; i++) {
      size_t da = (i < a_off) ? 1 : a->shape[i - a_off];
      size_t db = (i < b_off) ? 1 : b->shape[i - b_off];
      va[i] = (i >= a_off && da > 1) ? a->strides[i - a_off] : 0;
      vb[i] = (i >= b_off && db > 1) ? b->strides[i - b_off] : 0;
    }

    size_t ps[NUMC_MAX_DIMENSIONS], pa[NUMC_MAX_DIMENSIONS],
        pb[NUMC_MAX_DIMENSIONS], po[NUMC_MAX_DIMENSIONS];
    _sort_axes_binary(bcast_ndim, out->shape, va, vb, out->strides, ps, pa, pb,
                      po);
    _elemwise_binary_nd(kern, (const char *)a->data, pa,
                        (const char *)b->data, pb, (char *)out->data, po, ps,
                        bcast_ndim);
  }
}

/* ── Scalar Conversion ──────────────────────────────────────────────── */

static inline void _double_to_dtype(double value, NumcDType dtype,
                                    char buf[static 8]) {
  memset(buf, 0, 8);

  switch (dtype) {
  case NUMC_DTYPE_INT8:
    *(int8_t *)buf = (int8_t)value;
    break;
  case NUMC_DTYPE_INT16:
    *(int16_t *)buf = (int16_t)value;
    break;
  case NUMC_DTYPE_INT32:
    *(int32_t *)buf = (int32_t)value;
    break;
  case NUMC_DTYPE_INT64:
    *(int64_t *)buf = (int64_t)value;
    break;
  case NUMC_DTYPE_UINT8:
    *(uint8_t *)buf = (uint8_t)value;
    break;
  case NUMC_DTYPE_UINT16:
    *(uint16_t *)buf = (uint16_t)value;
    break;
  case NUMC_DTYPE_UINT32:
    *(uint32_t *)buf = (uint32_t)value;
    break;
  case NUMC_DTYPE_UINT64:
    *(uint64_t *)buf = (uint64_t)value;
    break;
  case NUMC_DTYPE_FLOAT32:
    *(float *)buf = (float)value;
    break;
  case NUMC_DTYPE_FLOAT64:
    *(double *)buf = (double)value;
    break;
  default:
    break;
  }
}

/* ── Scalar op dispatch ───────────────────────────────────────────── */

static inline void _scalar_op(const struct NumcArray *a, const char *scalar_buf,
                              struct NumcArray *out,
                              const NumcBinaryKernel *table) {
  NumcBinaryKernel kern = table[a->dtype];

  if (a->is_contiguous && out->is_contiguous) {
    /* Flat fast path: sa = es, sb = 0, so = es -> hits kernel PATH 2 */
    intptr_t es = (intptr_t)a->elem_size;
    kern((const char *)a->data, scalar_buf, (char *)out->data, a->size, es, 0,
         es);
  } else {
    size_t zero_strides[NUMC_MAX_DIMENSIONS] = {0};
    size_t ps[NUMC_MAX_DIMENSIONS], pa[NUMC_MAX_DIMENSIONS],
        pb[NUMC_MAX_DIMENSIONS], po[NUMC_MAX_DIMENSIONS];
    _sort_axes_binary(a->dim, a->shape, a->strides, zero_strides, out->strides,
                      ps, pa, pb, po);
    _elemwise_binary_nd(kern, (const char *)a->data, pa, scalar_buf, pb,
                        (char *)out->data, po, ps, a->dim);
  }
}

static inline int _scalar_op_inplace(struct NumcArray *a, double scalar,
                                     const NumcBinaryKernel *table) {
  if (!a)
    return NUMC_ERR_NULL;

  char buf[8];
  _double_to_dtype(scalar, a->dtype, buf);
  NumcBinaryKernel kern = table[a->dtype];

  if (a->is_contiguous) {
    intptr_t es = (intptr_t)a->elem_size;
    kern((const char *)a->data, buf, (char *)a->data, a->size, es, 0, es);
  } else {
    size_t zero_strides[NUMC_MAX_DIMENSIONS] = {0};
    size_t ps[NUMC_MAX_DIMENSIONS], pa[NUMC_MAX_DIMENSIONS],
        pb[NUMC_MAX_DIMENSIONS], po[NUMC_MAX_DIMENSIONS];
    _sort_axes_binary(a->dim, a->shape, a->strides, zero_strides, a->strides,
                      ps, pa, pb, po);
    _elemwise_binary_nd(kern, (const char *)a->data, pa, buf, pb,
                        (char *)a->data, po, ps, a->dim);
  }
  return 0;
}

/* ── Unary op dispatch ────────────────────────────────────────────── */

static inline int _unary_op(const struct NumcArray *a, struct NumcArray *out,
                            const NumcUnaryKernel *table) {
  NumcUnaryKernel kern = table[a->dtype];

  if (a->is_contiguous && out->is_contiguous) {
    intptr_t es = (intptr_t)a->elem_size;
    kern((const char *)a->data, (char *)out->data, a->size, es, es);
  } else {
    size_t ps[NUMC_MAX_DIMENSIONS], pa[NUMC_MAX_DIMENSIONS],
        po[NUMC_MAX_DIMENSIONS];
    _sort_axes_unary(a->dim, a->shape, a->strides, out->strides, ps, pa, po);
    _elemwise_unary_nd(kern, (const char *)a->data, pa, (char *)out->data, po,
                       ps, a->dim);
  }

  return 0;
}

static inline int _unary_op_inplace(struct NumcArray *a,
                                    const NumcUnaryKernel *table) {
  if (!a)
    return NUMC_ERR_NULL;

  NumcUnaryKernel kern = table[a->dtype];

  if (a->is_contiguous) {
    intptr_t es = (intptr_t)a->elem_size;
    kern((const char *)a->data, (char *)a->data, a->size, es, es);
  } else {
    size_t ps[NUMC_MAX_DIMENSIONS], pa[NUMC_MAX_DIMENSIONS],
        po[NUMC_MAX_DIMENSIONS];
    _sort_axes_unary(a->dim, a->shape, a->strides, a->strides, ps, pa, po);
    _elemwise_unary_nd(kern, (const char *)a->data, pa, (char *)a->data, po, ps,
                       a->dim);
  }
  return 0;
}

#endif
