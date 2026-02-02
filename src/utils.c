/**
 * @file utils.c
 * @brief Array manipulation: reshape, transpose, concatenation.
 */

#include "array.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

// =============================================================================
//                    Strided Copy Helpers (Static Functions)
// =============================================================================

/**
 * @brief Unchecked element access for internal use (no bounds checking).
 */
static inline void *array_get_unchecked(const Array *array,
                                        const size_t *indices) {
  size_t offset = 0;
  for (size_t i = 0; i < array->ndim; i++) {
    offset += indices[i] * array->strides[i];
  }
  return (char *)array->data + offset;
}

/**
 * @brief Increment multi-dimensional indices (row-major order).
 */
static inline void increment_indices(size_t *indices, const size_t *shape,
                                     size_t ndim) {
  for (ssize_t i = ndim - 1; i >= 0; i--) {
    indices[i]++;
    if (indices[i] < shape[i])
      break;
    indices[i] = 0;
  }
}

// Generate strided-to-strided copy functions with axis offset
#define GEN_STRIDED_TO_STRIDED_COPY_WITH_OFFSET_FUNC(dtype, ctype)             \
  static inline void array_strided_to_strided_copy_with_offset_##dtype(        \
      const Array *src, size_t *src_indices, Array *dst, size_t axis,          \
      size_t axis_offset, size_t count) {                                      \
    const size_t ndim = src->ndim;                                             \
    const size_t *restrict src_shape = src->shape;                             \
    const size_t *restrict src_strides = src->strides;                         \
    const size_t *restrict dst_strides = dst->strides;                         \
    const char *restrict src_data = (const char *)src->data;                   \
    char *restrict dst_data = (char *)dst->data;                               \
                                                                               \
    for (size_t i = 0; i < count; i++) {                                       \
      /* Calculate source offset */                                            \
      size_t src_offset = 0;                                                   \
      for (size_t d = 0; d < ndim; d++) {                                      \
        src_offset += src_indices[d] * src_strides[d];                         \
      }                                                                        \
                                                                               \
      /* Calculate destination offset (copy indices + adjust axis) */          \
      size_t dst_offset = 0;                                                   \
      for (size_t d = 0; d < ndim; d++) {                                      \
        size_t idx = src_indices[d];                                           \
        if (d == axis)                                                         \
          idx += axis_offset;                                                  \
        dst_offset += idx * dst_strides[d];                                    \
      }                                                                        \
                                                                               \
      /* Copy element */                                                       \
      *((ctype *)(dst_data + dst_offset)) =                                    \
          *((const ctype *)(src_data + src_offset));                           \
                                                                               \
      /* Increment source indices */                                           \
      increment_indices(src_indices, src_shape, ndim);                         \
    }                                                                          \
  }

FOREACH_DTYPE(GEN_STRIDED_TO_STRIDED_COPY_WITH_OFFSET_FUNC)
#undef GEN_STRIDED_TO_STRIDED_COPY_WITH_OFFSET_FUNC

typedef void (*strided_to_strided_copy_with_offset_func)(const Array *,
                                                         size_t *, Array *,
                                                         size_t, size_t,
                                                         size_t);

#define STRIDED_TO_STRIDED_COPY_WITH_OFFSET_ENTRY(dtype, ctype)                \
  [DTYPE_##dtype] = array_strided_to_strided_copy_with_offset_##dtype,

static const strided_to_strided_copy_with_offset_func
    strided_to_strided_copy_with_offset_funcs[] = {
        FOREACH_DTYPE(STRIDED_TO_STRIDED_COPY_WITH_OFFSET_ENTRY)};
#undef STRIDED_TO_STRIDED_COPY_WITH_OFFSET_ENTRY

static inline void array_strided_copy_with_offset(const Array *src,
                                                  size_t *src_indices,
                                                  Array *dst, size_t axis,
                                                  size_t axis_offset,
                                                  size_t count) {
  strided_to_strided_copy_with_offset_funcs[src->dtype](
      src, src_indices, dst, axis, axis_offset, count);
}

// Generate strided-to-strided copy functions (same indices)
#define GEN_STRIDED_TO_STRIDED_COPY_FUNC(dtype, ctype)                         \
  static inline void array_strided_to_strided_copy_##dtype(                    \
      const Array *src, size_t *src_indices, Array *dst, size_t count) {       \
    const size_t ndim = src->ndim;                                             \
    const size_t *restrict shape = src->shape;                                 \
    const size_t *restrict src_strides = src->strides;                         \
    const size_t *restrict dst_strides = dst->strides;                         \
    const char *restrict src_data = (const char *)src->data;                   \
    char *restrict dst_data = (char *)dst->data;                               \
                                                                               \
    for (size_t i = 0; i < count; i++) {                                       \
      /* Calculate offsets for both src and dst using same indices */          \
      size_t src_offset = 0, dst_offset = 0;                                   \
      for (size_t d = 0; d < ndim; d++) {                                      \
        size_t idx = src_indices[d];                                           \
        src_offset += idx * src_strides[d];                                    \
        dst_offset += idx * dst_strides[d];                                    \
      }                                                                        \
                                                                               \
      /* Copy element */                                                       \
      *((ctype *)(dst_data + dst_offset)) =                                    \
          *((const ctype *)(src_data + src_offset));                           \
                                                                               \
      /* Increment indices */                                                  \
      increment_indices(src_indices, shape, ndim);                             \
    }                                                                          \
  }

FOREACH_DTYPE(GEN_STRIDED_TO_STRIDED_COPY_FUNC)
#undef GEN_STRIDED_TO_STRIDED_COPY_FUNC

typedef void (*strided_to_strided_copy_func)(const Array *, size_t *, Array *,
                                             size_t);

#define STRIDED_TO_STRIDED_COPY_ENTRY(dtype, ctype)                            \
  [DTYPE_##dtype] = array_strided_to_strided_copy_##dtype,

static const strided_to_strided_copy_func strided_to_strided_copy_funcs[] = {
    FOREACH_DTYPE(STRIDED_TO_STRIDED_COPY_ENTRY)};
#undef STRIDED_TO_STRIDED_COPY_ENTRY

static inline void array_strided_copy(const Array *src, size_t *src_indices,
                                      Array *dst, size_t count) {
  strided_to_strided_copy_funcs[src->dtype](src, src_indices, dst, count);
}

// =============================================================================
//                          Reshape Function
// =============================================================================

int array_reshape(Array *array, size_t ndim, const size_t *shape) {
  if (!array || !shape || ndim == 0)
    return -1;

  if (!array_is_contiguous(array)) {
    fprintf(stderr, "array_reshape: array is not contiguous. "
                    "Use array_to_contiguous() first.\n");
    abort();
  }

  size_t new_size = 1;
  for (size_t i = 0; i < ndim; i++) {
    if (shape[i] == 0)
      return -1;

    // Check for overflow before multiplication
    if (new_size > SIZE_MAX / shape[i])
      return -1;

    new_size *= shape[i];
  }

  if (new_size != array->size)
    return -1;

  size_t *new_shape = malloc(2 * ndim * sizeof(size_t));
  if (!new_shape)
    return -1;
  size_t *new_strides = new_shape + ndim;

  for (size_t i = 0; i < ndim; i++)
    new_shape[i] = shape[i];

  new_strides[ndim - 1] = array->elem_size;
  for (ssize_t j = ndim - 2; j >= 0; j--)
    new_strides[j] = new_strides[j + 1] * new_shape[j + 1];

  free(array->shape); // frees both old shape and strides

  array->shape = new_shape;
  array->strides = new_strides;
  array->ndim = ndim;

  return 0;
}

// =============================================================================
//                          Transpose Function
// =============================================================================

int array_transpose(Array *array, size_t *axes) {
  if (!array)
    return -1;

  size_t *new_shape = malloc(array->ndim * sizeof(size_t));
  size_t *new_strides = malloc(array->ndim * sizeof(size_t));

  if (!new_shape || !new_strides) {
    free(new_shape);
    free(new_strides);
    return -1;
  }

  for (size_t i = 0; i < array->ndim; i++) {
    new_shape[i] = array->shape[axes[i]];
    new_strides[i] = array->strides[axes[i]];
  }

  memcpy(array->shape, new_shape, array->ndim * sizeof(size_t));
  memcpy(array->strides, new_strides, array->ndim * sizeof(size_t));

  free(new_shape);
  free(new_strides);

  return 0;
}

// =============================================================================
//                          Concatenation Function
// =============================================================================

Array *array_concat(const Array *a, const Array *b, size_t axis) {
  if (!a || !b)
    return NULL;

  if (a->ndim != b->ndim)
    return NULL;

  if (axis >= a->ndim)
    return NULL;

  if (a->elem_size != b->elem_size)
    return NULL;

  for (size_t i = 0; i < a->ndim; i++) {
    if (i != axis && a->shape[i] != b->shape[i])
      return NULL;
  }

  size_t *new_shape = malloc(a->ndim * sizeof(size_t));
  if (!new_shape)
    return NULL;

  for (size_t i = 0; i < a->ndim; i++)
    new_shape[i] = a->shape[i];
  new_shape[axis] = a->shape[axis] + b->shape[axis];

  Array *result = array_create(a->ndim, new_shape, a->dtype, NULL);
  free(new_shape);
  if (!result)
    return NULL;

  // Fast path: both contiguous and concatenating along axis 0
  if (array_is_contiguous(a) && array_is_contiguous(b) && axis == 0) {
    memcpy(result->data, a->data, a->size * a->elem_size);
    memcpy((char *)result->data + (a->size * a->elem_size), b->data,
           b->size * b->elem_size);
    return result;
  }

  // Slow path: use strided copy for non-contiguous or non-axis-0
  size_t indices_buf[MAX_STACK_NDIM] = {0};
  size_t *indices = (a->ndim <= MAX_STACK_NDIM)
                        ? indices_buf
                        : calloc(a->ndim, sizeof(size_t));

  if (a->ndim > MAX_STACK_NDIM && !indices)
    goto fail;

  array_strided_copy(a, indices, result, a->size);

  for (size_t i = 0; i < a->ndim; i++)
    indices[i] = 0;

  array_strided_copy_with_offset(b, indices, result, axis, a->shape[axis],
                                 b->size);

  if (indices && a->ndim > MAX_STACK_NDIM)
    free(indices);

  return result;

fail:
  if (indices && a->ndim > MAX_STACK_NDIM)
    free(indices);
  array_free(result);
  return NULL;
}
