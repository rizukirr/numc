/**
 * @file utils.c
 * @brief Array manipulation: reshape, transpose, concatenation.
 */

#include "array.h"

#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

// =============================================================================
//                    Strided Copy Helpers
// =============================================================================

/**
 * @brief Strided element copy with destination axis offset.
 *
 * Copies elements one-by-one from src to dst, applying an axis offset to
 * destination indices. Used by array_concat for the second array.
 * memcpy with elem_size handles all types — the compiler optimizes
 * small fixed-size copies (1/2/4/8 bytes) into single load/store.
 */
static inline void array_strided_copy_with_offset(const Array *src,
                                                  size_t *src_indices,
                                                  Array *dst, size_t axis,
                                                  size_t axis_offset,
                                                  size_t count) {
  const size_t ndim = src->ndim;
  const size_t esize = src->elem_size;
  const size_t *restrict src_shape = src->shape;
  const size_t *restrict src_strides = src->strides;
  const size_t *restrict dst_strides = dst->strides;
  const char *restrict src_data = (const char *)src->data;
  char *restrict dst_data = (char *)dst->data;

  // Inner-dimension contiguous fast path (only when axis != last dim)
  if (ndim >= 1 && axis != ndim - 1 && src_strides[ndim - 1] == esize &&
      dst_strides[ndim - 1] == esize) {
    size_t remaining = count;
    while (remaining > 0) {
      size_t inner_idx = src_indices[ndim - 1];
      size_t row_remaining = src_shape[ndim - 1] - inner_idx;
      size_t chunk = (remaining < row_remaining) ? remaining : row_remaining;

      size_t src_offset = 0;
      for (size_t d = 0; d < ndim; d++)
        src_offset += src_indices[d] * src_strides[d];

      size_t dst_offset = 0;
      for (size_t d = 0; d < ndim; d++) {
        size_t idx = src_indices[d];
        if (d == axis)
          idx += axis_offset;
        dst_offset += idx * dst_strides[d];
      }

      memcpy(dst_data + dst_offset, src_data + src_offset, chunk * esize);
      remaining -= chunk;

      src_indices[ndim - 1] += chunk;
      if (src_indices[ndim - 1] >= src_shape[ndim - 1]) {
        src_indices[ndim - 1] = 0;
        for (ssize_t d = ndim - 2; d >= 0; d--) {
          src_indices[d]++;
          if (src_indices[d] < src_shape[d])
            break;
          src_indices[d] = 0;
        }
      }
    }
    return;
  }

  for (size_t i = 0; i < count; i++) {
    size_t src_offset = 0;
    for (size_t d = 0; d < ndim; d++)
      src_offset += src_indices[d] * src_strides[d];

    size_t dst_offset = 0;
    for (size_t d = 0; d < ndim; d++) {
      size_t idx = src_indices[d];
      if (d == axis)
        idx += axis_offset;
      dst_offset += idx * dst_strides[d];
    }

    memcpy(dst_data + dst_offset, src_data + src_offset, esize);
    increment_indices(src_indices, src_shape, ndim);
  }
}

/**
 * @brief Strided element copy (same indices for src and dst).
 *
 * Copies elements one-by-one from src to dst using the same logical indices.
 * Used by array_concat for the first array.
 */
static inline void array_strided_copy(const Array *src, size_t *src_indices,
                                      Array *dst, size_t count) {
  const size_t ndim = src->ndim;
  const size_t esize = src->elem_size;
  const size_t *restrict shape = src->shape;
  const size_t *restrict src_strides = src->strides;
  const size_t *restrict dst_strides = dst->strides;
  const char *restrict src_data = (const char *)src->data;
  char *restrict dst_data = (char *)dst->data;

  // Inner-dimension contiguous fast path: copy rows with memcpy
  if (ndim >= 1 && src_strides[ndim - 1] == esize &&
      dst_strides[ndim - 1] == esize) {
    size_t remaining = count;
    while (remaining > 0) {
      size_t inner_idx = src_indices[ndim - 1];
      size_t row_remaining = shape[ndim - 1] - inner_idx;
      size_t chunk = (remaining < row_remaining) ? remaining : row_remaining;

      size_t src_offset = 0, dst_offset = 0;
      for (size_t d = 0; d < ndim; d++) {
        src_offset += src_indices[d] * src_strides[d];
        dst_offset += src_indices[d] * dst_strides[d];
      }

      memcpy(dst_data + dst_offset, src_data + src_offset, chunk * esize);
      remaining -= chunk;

      // Advance indices past the chunk
      src_indices[ndim - 1] += chunk;
      if (src_indices[ndim - 1] >= shape[ndim - 1]) {
        src_indices[ndim - 1] = 0;
        for (ssize_t d = ndim - 2; d >= 0; d--) {
          src_indices[d]++;
          if (src_indices[d] < shape[d])
            break;
          src_indices[d] = 0;
        }
      }
    }
    return;
  }

  for (size_t i = 0; i < count; i++) {
    size_t src_offset = 0, dst_offset = 0;
    for (size_t d = 0; d < ndim; d++) {
      size_t idx = src_indices[d];
      src_offset += idx * src_strides[d];
      dst_offset += idx * dst_strides[d];
    }

    memcpy(dst_data + dst_offset, src_data + src_offset, esize);
    increment_indices(src_indices, shape, ndim);
  }
}

// =============================================================================
//                          Reshape Function
// =============================================================================

int array_reshape(Array *array, size_t ndim, const size_t *shape) {
  if (!array || !shape || ndim == 0)
    return -1;

  if (!array->is_contiguous)
    return -1;

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

  bool old_on_heap = array->ndim > MAX_STACK_NDIM;
  bool new_use_stack = ndim <= MAX_STACK_NDIM;

  if (new_use_stack) {
    // New shape fits in embedded buffers
    size_t tmp_shape[MAX_STACK_NDIM];
    size_t tmp_strides[MAX_STACK_NDIM];

    for (size_t i = 0; i < ndim; i++)
      tmp_shape[i] = shape[i];

    tmp_strides[ndim - 1] = array->elem_size;
    for (ssize_t j = ndim - 2; j >= 0; j--)
      tmp_strides[j] = tmp_strides[j + 1] * tmp_shape[j + 1];

    if (old_on_heap)
      free(array->shape);

    array->shape = array->_shape_buff;
    array->strides = array->_strides_buff;
    memcpy(array->shape, tmp_shape, ndim * sizeof(size_t));
    memcpy(array->strides, tmp_strides, ndim * sizeof(size_t));
  } else {
    // New shape needs heap allocation
    size_t *new_shape = malloc(2 * ndim * sizeof(size_t));
    if (!new_shape)
      return -1;
    size_t *new_strides = new_shape + ndim;

    for (size_t i = 0; i < ndim; i++)
      new_shape[i] = shape[i];

    new_strides[ndim - 1] = array->elem_size;
    for (ssize_t j = ndim - 2; j >= 0; j--)
      new_strides[j] = new_strides[j + 1] * new_shape[j + 1];

    if (old_on_heap)
      free(array->shape);

    array->shape = new_shape;
    array->strides = new_strides;
  }

  array->ndim = ndim;

  return 0;
}

// =============================================================================
//                          Transpose Function
// =============================================================================

int array_transpose(Array *array, size_t *axes) {
  if (!array || !axes)
    return -1;

  // Validate axes: each value must be < ndim and no duplicates
  size_t seen_buf[MAX_STACK_NDIM] = {0};
  size_t *seen = (array->ndim <= MAX_STACK_NDIM)
                     ? seen_buf
                     : calloc(array->ndim, sizeof(size_t));
  if (array->ndim > MAX_STACK_NDIM && !seen)
    return -1;

  for (size_t i = 0; i < array->ndim; i++) {
    if (axes[i] >= array->ndim || seen[axes[i]]) {
      if (array->ndim > MAX_STACK_NDIM)
        free(seen);
      return -1;
    }
    seen[axes[i]] = 1;
  }
  if (array->ndim > MAX_STACK_NDIM)
    free(seen);

  size_t new_shape_buf[MAX_STACK_NDIM] = {0};
  size_t new_strides_buf[MAX_STACK_NDIM] = {0};
  bool use_stack = array->ndim <= MAX_STACK_NDIM;
  size_t *new_shape =
      use_stack ? new_shape_buf : malloc(array->ndim * sizeof(size_t));
  size_t *new_strides =
      use_stack ? new_strides_buf : malloc(array->ndim * sizeof(size_t));

  if ((!new_shape || !new_strides) && !use_stack) {
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

  if (!use_stack) {
    free(new_shape);
    free(new_strides);
  }

  array->is_contiguous = array_is_contiguous(array);

  return 0;
}

// =============================================================================
//                          Concatenation Function
// =============================================================================

Array *array_concatenate(const Array *a, const Array *b, size_t axis) {
  if (!a || !b)
    return NULL;

  if (a->ndim != b->ndim)
    return NULL;

  if (axis >= a->ndim)
    return NULL;

  if (a->elem_size != b->elem_size || a->numc_type != b->numc_type)
    return NULL;

  for (size_t i = 0; i < a->ndim; i++) {
    if (i != axis && a->shape[i] != b->shape[i])
      return NULL;
  }

  size_t new_shape_buf[MAX_STACK_NDIM] = {0};
  bool use_stack = a->ndim <= MAX_STACK_NDIM;
  size_t *new_shape =
      use_stack ? new_shape_buf : malloc(a->ndim * sizeof(size_t));
  if (!new_shape && !use_stack)
    return NULL;

  for (size_t i = 0; i < a->ndim; i++)
    new_shape[i] = a->shape[i];
  new_shape[axis] = a->shape[axis] + b->shape[axis];

  ArrayCreate result_create = {
      .ndim = a->ndim,
      .shape = new_shape,
      .numc_type = a->numc_type,
      .data = NULL,
      .owns_data = true,
  };
  Array *result = array_create(&result_create);
  if (!use_stack)
    free(new_shape);

  if (!result)
    return NULL;

  // Fast path: both contiguous — generalized memcpy for any axis
  if (a->is_contiguous && b->is_contiguous) {
    size_t outer = 1;
    for (size_t i = 0; i < axis; i++)
      outer *= a->shape[i];

    size_t a_inner = 1;
    for (size_t i = axis; i < a->ndim; i++)
      a_inner *= a->shape[i];

    size_t b_inner = 1;
    for (size_t i = axis; i < b->ndim; i++)
      b_inner *= b->shape[i];

    const size_t elem = a->elem_size;
    const size_t a_bytes = a_inner * elem;
    const size_t b_bytes = b_inner * elem;
    const size_t dst_stride = a_bytes + b_bytes;

    const char *a_ptr = (const char *)a->data;
    const char *b_ptr = (const char *)b->data;
    char *dst = (char *)result->data;

    for (size_t i = 0; i < outer; i++) {
      memcpy(dst, a_ptr, a_bytes);
      memcpy(dst + a_bytes, b_ptr, b_bytes);
      a_ptr += a_bytes;
      b_ptr += b_bytes;
      dst += dst_stride;
    }

    return result;
  }

  // Slow path: use strided copy for non-contiguous or non-axis-0
  size_t indices_buf[MAX_STACK_NDIM] = {0};
  size_t *indices = (a->ndim <= MAX_STACK_NDIM)
                        ? indices_buf
                        : malloc(a->ndim * sizeof(size_t));

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
