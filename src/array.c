/**
 * @file array.c
 * @brief Core array operations: creation, access, properties, copy, and views.
 */

#include "array.h"
#include "alloc.h"
#include "error.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

/**
 * @brief General strided-to-contiguous copy using byte offsets.
 *
 * Copies elements one-by-one from a non-contiguous source to a contiguous
 * destination. Uses elem_size and memcpy so it works for all types — the
 * compiler optimizes small fixed-size memcpy (1/2/4/8 bytes) into single
 * load/store instructions.
 */
static inline void strided_to_contiguous_copy_general(size_t *src_indices,
                                                      const Array *src,
                                                      size_t count,
                                                      char *dest) {

  const size_t *restrict strides = src->strides;
  const char *restrict src_data = (const char *)src->data;
  const size_t esize = src->elem_size;

  for (size_t i = 0; i < count; i++) {
    size_t offset = 0;
    for (size_t d = 0; d < src->ndim; d++) {
      offset += src_indices[d] * strides[d];
    }
    memcpy(dest + i * esize, src_data + offset, esize);
    increment_indices(src_indices, src->shape, src->ndim);
  }
}

/**
 * @brief Fast strided-to-contiguous copy with optimizations.
 *
 * Optimized version of strided_to_contiguous_copy_general with fast paths for:
 * - 1D contiguous arrays: single memcpy
 * - 2D inner-contiguous arrays (e.g., sliced rows): row-by-row memcpy
 * - General case: falls back to element-by-element copy
 *
 * @param src         Source array (may be non-contiguous).
 * @param src_indices Starting indices in source (modified in place).
 * @param dest_data   Destination data buffer (contiguous).
 * @param dest_offset Offset in destination buffer (in elements).
 * @param count       Number of elements to copy.
 */
static inline void
strided_to_contiguous_copy(const Array *src, size_t *src_indices,
                           void *dest_data, size_t dest_offset, size_t count) {

  const char *restrict src_data = (const char *)src->data;
  const size_t esize = src->elem_size;
  char *dest = (char *)dest_data + dest_offset * esize;

  if (src->ndim == 1 && src->strides[0] == esize) {
    memcpy(dest, src_data + src_indices[0] * src->strides[0], count * esize);
    src_indices[0] += count;
    return;
  }

  // Inner-dimension contiguous: copy row-by-row with memcpy.
  // This handles sliced 2D arrays where rows are contiguous but may have gaps.
  bool is_2d_inner_contig = src->ndim == 2 && src->strides[1] == esize;

  if (is_2d_inner_contig) {
    size_t remaining = count;
    while (remaining > 0) {
      size_t row = src_indices[0];
      size_t col = src_indices[1];
      size_t row_remaining = src->shape[1] - col;
      size_t chunk = (remaining < row_remaining) ? remaining : row_remaining;
      memcpy(dest, src_data + row * src->strides[0] + col * src->strides[1],
             chunk * esize);
      dest += chunk * esize;
      remaining -= chunk;
      col += chunk;
      if (col >= src->shape[1]) {
        col = 0;
        row++;
      }
      src_indices[0] = row;
      src_indices[1] = col;
    }
    return;
  }

  strided_to_contiguous_copy_general(src_indices, src, count, dest);
}

// =============================================================================
//                          Public Functions
// =============================================================================

size_t array_offset(const Array *array, const size_t *indices) {
  if (!array || !indices) {
    numc_set_error(NUMC_ERR_NULL, "numc: array_offset: NULL argument");
    return NUMC_OK;
  }
  size_t offset = 0;
  for (size_t i = 0; i < array->ndim; i++) {
    offset += indices[i] * array->strides[i];
  }

  return offset;
}

int array_bounds_check(const Array *array, const size_t *indices) {
  if (!array || !indices) {
    numc_set_error(NUMC_ERR_NULL, "numc: array_bounds_check: NULL argument");
    return NUMC_ERR_NULL;
  }

  for (size_t i = 0; i < array->ndim; i++) {
    if (indices[i] >= array->shape[i]) {
      numc_set_error(NUMC_ERR_BOUNDS,
                     "array_bounds_check: index out of bounds");
      return NUMC_ERR_BOUNDS;
    }
  }

  return NUMC_OK;
}

void *array_get(const Array *array, const size_t *indices) {
  if (!array || !indices) {
    numc_set_error(NUMC_ERR_NULL, "numc: array_get: NULL argument passed");
    return NULL;
  }

  size_t offset = array_offset(array, indices);
  return (char *)array->data + offset;
}

bool array_is_contiguous(const Array *array) {
  if (!array || array->ndim == 0) {
    numc_set_error(NUMC_ERR_NULL, "numc: array_is_contiguous: NULL argument");
    return false;
  }

  size_t expected = array->elem_size;
  for (size_t i = array->ndim; i-- > 0;) {
    if (array->strides[i] != expected)
      return false;
    expected *= array->shape[i];
  }

  return true;
}

Array *array_slice(Array *base, const size_t *start, const size_t *stop,
                   const size_t *step) {
  if (!base || !start || !stop || !step) {
    numc_set_error(NUMC_ERR_NULL, "numc: array_slice: NULL argument");
    return NULL;
  }

  Array *view = malloc(sizeof(Array));
  if (!view) {
    numc_set_error(NUMC_ERR_ALLOC, "numc: array_slice: allocation failed");
    return NULL;
  }

  view->ndim = base->ndim;
  bool use_stack = base->ndim <= MAX_STACK_NDIM;
  if (use_stack) {
    view->shape = view->_shape_buff;
    view->strides = view->_strides_buff;
  } else {
    view->shape = malloc(2 * base->ndim * sizeof(size_t));
    if (!view->shape) {
      free(view);
      numc_set_error(NUMC_ERR_ALLOC,
                     "numc: array_slice: allocation failed for shape buffer");
      return NULL;
    }
    view->strides = view->shape + base->ndim;
  }
  view->numc_type = base->numc_type;
  view->elem_size = base->elem_size;
  view->owns_data = false;

  size_t offset = 0;
  view->size = 1;
  for (size_t i = 0; i < base->ndim; i++) {
    if (start[i] >= base->shape[i] || stop[i] > base->shape[i] ||
        start[i] >= stop[i] || step[i] == 0) {
      numc_set_error(NUMC_ERR_INVALID, "array_slice: invalid slice parameters");
      if (!use_stack)
        free(view->shape);
      free(view);
      return NULL;
    }

    size_t len = (stop[i] - start[i] + step[i] - 1) / step[i];
    view->shape[i] = len;
    view->size *= len;
    view->strides[i] = base->strides[i] * step[i];
    offset += start[i] * base->strides[i];
  }

  view->is_contiguous = array_is_contiguous(view);
  view->capacity = view->size;
  view->data = (char *)base->data + offset;
  return view;
}

void increment_indices(size_t *indices, const size_t *shape, size_t ndim) {
  for (ssize_t i = ndim - 1; i >= 0; i--) {
    indices[i]++;
    if (indices[i] < shape[i])
      break;
    indices[i] = 0;
  }
}

Array *array_copy(const Array *src) {
  if (!src) {
    numc_set_error(NUMC_ERR_NULL, "numc: array_copy: NULL argument");
    return NULL;
  }

  if (src->is_contiguous) {
    ArrayCreate d = {
        .ndim = src->ndim,
        .shape = src->shape,
        .numc_type = src->numc_type,
        .data = src->data,
        .owns_data = true,
    };
    return array_create(&d);
  }

  ArrayCreate d = {
      .ndim = src->ndim,
      .shape = src->shape,
      .numc_type = src->numc_type,
      .data = NULL,
      .owns_data = true,
  };
  Array *dst = array_create(&d);
  if (!dst)
    return NULL;

  size_t indices_buf[MAX_STACK_NDIM] = {0};
  size_t *indices = (src->ndim <= MAX_STACK_NDIM)
                        ? indices_buf
                        : calloc(src->ndim, sizeof(size_t));
  if (src->ndim > MAX_STACK_NDIM && !indices) {
    array_free(dst);
    numc_set_error(NUMC_ERR_ALLOC,
                   "numc: array_copy: allocation failed for indices buffer");
    return NULL;
  }

  strided_to_contiguous_copy(src, indices, dst->data, 0, src->size);

  if (src->ndim > MAX_STACK_NDIM)
    free(indices);

  return dst;
}

int array_ascontiguousarray(Array *arr) {
  if (!arr) {
    numc_set_error(NUMC_ERR_NULL, "array_ascontiguousarray: NULL argument");
    return NUMC_ERR_NULL;
  }

  if (arr->is_contiguous)
    return NUMC_OK;

  size_t total_bytes = arr->size * arr->elem_size;
  void *new_data = numc_malloc(NUMC_ALIGN, total_bytes);
  if (!new_data) {
    numc_set_error(NUMC_ERR_ALLOC,
                   "array_ascontiguousarray: allocation failed");
    return NUMC_ERR_ALLOC;
  }

  size_t indices_buf[MAX_STACK_NDIM] = {0};
  size_t *indices = (arr->ndim <= MAX_STACK_NDIM)
                        ? indices_buf
                        : calloc(arr->ndim, sizeof(size_t));
  if (arr->ndim > MAX_STACK_NDIM && !indices) {
    numc_free(new_data);
    numc_set_error(NUMC_ERR_ALLOC,
                   "array_ascontiguousarray: indices allocation failed");
    return NUMC_ERR_ALLOC;
  }

  strided_to_contiguous_copy(arr, indices, new_data, 0, arr->size);

  if (arr->ndim > MAX_STACK_NDIM)
    free(indices);

  if (arr->owns_data)
    numc_free(arr->data);

  arr->data = new_data;
  arr->owns_data = true;

  arr->strides[arr->ndim - 1] = arr->elem_size;
  for (ssize_t i = arr->ndim - 2; i >= 0; i--)
    arr->strides[i] = arr->strides[i + 1] * arr->shape[i + 1];

  arr->is_contiguous = true;

  return NUMC_OK;
}
