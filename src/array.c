#include "array.h"
#include "dtype.h"
#include "memory.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

// Unchecked element access for internal use (no bounds checking)
static inline void *array_get_unchecked(const Array *array,
                                        const size_t *indices) {
  size_t offset = 0;
  for (size_t i = 0; i < array->ndim; i++) {
    offset += indices[i] * array->strides[i];
  }
  return (char *)array->data + offset;
}

Array *array_create(size_t ndim, const size_t *shape, DType dtype) {
  size_t elem_size = dtype_size(dtype);
  if (ndim == 0 || elem_size == 0 || !shape)
    return NULL;

  Array *array = malloc(sizeof(Array));
  if (!array)
    return NULL;

  array->ndim = ndim;
  array->dtype = dtype;
  array->elem_size = elem_size;

  // Combined allocation for shape and strides
  array->shape = malloc(2 * ndim * sizeof(size_t));
  if (!array->shape) {
    free(array);
    return NULL;
  }
  array->strides = array->shape + ndim;

  array->owns_data = 1;

  array->size = 1;
  for (size_t i = 0; i < ndim; i++) {
    array->shape[i] = shape[i];
    array->size *= shape[i];
  }

  array->strides[ndim - 1] = elem_size;
  for (ssize_t i = ndim - 2; i >= 0; i--)
    array->strides[i] = array->strides[i + 1] * shape[i + 1];

  // Initialize capacity equal to size (no over-allocation initially)
  array->capacity = array->size;
  array->data = aligned_calloc(NUMC_ALIGN, array->size * elem_size);
  if (!array->data) {
    free(array->shape);
    free(array);
    return NULL;
  }

  return array;
}

Array *array_batch(size_t ndim, const size_t *shape, DType dtype,
                   const void *data) {
  if (!data)
    return NULL;

  size_t elem_size = dtype_size(dtype);

  // Create array structure with same shape
  Array *array = array_create(ndim, shape, dtype);
  if (!array)
    return NULL;

  // Copy data (both array and data are contiguous in row-major order)
  memcpy(array->data, data, array->size * elem_size);

  return array;
}

void array_free(Array *array) {
  if (!array)
    return;

  if (array->owns_data)
    aligned_free(array->data);

  free(array->shape);
  free(array);
}

void *array_get(Array *array, ...) {
  if (!array) {
    fprintf(stderr, "array_get: NULL array\n");
    abort();
  }

  va_list ap;
  va_start(ap, array);

  size_t offset = 0;
  for (size_t i = 0; i < array->ndim; i++) {
    offset += array->strides[i] * va_arg(ap, size_t);
  }

  va_end(ap);

  return (char *)array->data + offset;
}

size_t array_offset(const Array *array, const size_t *indices) {
  if (!array || !indices)
    return 0;

  size_t offset = 0;
  for (size_t i = 0; i < array->ndim; i++) {
    offset += indices[i] * array->strides[i];
  }

  return offset;
}

void *array_at(const Array *array, const size_t *indices) {
  if (!array || !indices) {
    fprintf(stderr, "array_at: NULL array or indices\n");
    abort();
  }

  for (size_t i = 0; i < array->ndim; i++) {
    if (indices[i] >= array->shape[i]) {
      fprintf(stderr,
              "array_at: index out of bounds at dimension %zu "
              "(index=%zu, shape=%zu)\n",
              i, indices[i], array->shape[i]);
      abort();
    }
  }

  size_t offset = array_offset(array, indices);
  return (char *)array->data + offset;
}

int array_reshape(Array *array, size_t ndim, const size_t *shape) {
  if (!array || !shape || ndim == 0)
    return -1;

  if (!array_is_contiguous(array)) {
    fprintf(stderr, "array_reshape: array is not contiguous\n");
    abort();
  }

  size_t new_size = 1;
  for (size_t i = 0; i < ndim; i++) {
    if (shape[i] == 0)
      return -1;

    new_size *= shape[i];
  }

  if (new_size != array->size)
    return -1;

  // Allocate combined shape+strides
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

size_t array_size(const Array *array) { return array ? array->size : 0; }

Array *array_slice(Array *base, const size_t *start, const size_t *stop,
                   const size_t *step) {
  if (!base || !start || !stop || !step) {
    fprintf(stderr, "array_slice: NULL argument\n");
    abort();
  }

  Array *view = malloc(sizeof(Array));
  if (!view)
    return NULL;

  view->ndim = base->ndim;
  view->dtype = base->dtype;
  view->elem_size = base->elem_size;

  // Combined allocation for shape and strides
  view->shape = malloc(2 * view->ndim * sizeof(size_t));
  if (!view->shape) {
    free(view);
    return NULL;
  }
  view->strides = view->shape + view->ndim;

  // Fix: view does NOT own data, base keeps ownership
  view->owns_data = 0;
  view->capacity = 0; // Views don't own data, so no capacity

  size_t offset = 0;
  view->size = 1;
  for (size_t i = 0; i < view->ndim; i++) {
    if (start[i] >= base->shape[i] || stop[i] > base->shape[i] ||
        start[i] >= stop[i] || step[i] == 0) {
      fprintf(stderr,
              "array_slice: invalid slice at dimension %zu "
              "(start=%zu, stop=%zu, step=%zu, shape=%zu)\n",
              i, start[i], stop[i], step[i], base->shape[i]);
      free(view->shape);
      free(view);
      abort();
    }

    size_t len = (stop[i] - start[i] + step[i] - 1) / step[i];
    view->shape[i] = len;
    view->size *= len;
    view->strides[i] = base->strides[i] * step[i];
    offset += start[i] * base->strides[i];
  }

  view->data = (char *)base->data + offset;
  return view;
}

int array_is_contiguous(const Array *array) {
  if (!array || array->ndim == 0)
    return 0;

  size_t expected = array->elem_size;
  for (size_t i = array->ndim; i-- > 0;) {
    if (array->strides[i] != expected)
      return 0;
    expected *= array->shape[i];
  }

  return 1;
}

Array *array_copy(const Array *src) {
  if (!src)
    return NULL;

  Array *dst = array_create(src->ndim, src->shape, src->dtype);
  if (!dst)
    return NULL;

  // Fast path: if source is contiguous, use memcpy
  if (array_is_contiguous(src)) {
    memcpy(dst->data, src->data, src->size * src->elem_size);
    return dst;
  }

  // Slow path: element-by-element copy using unchecked access
  size_t total = src->size;
  size_t *indices = calloc(src->ndim, sizeof(size_t));
  if (!indices) {
    array_free(dst);
    return NULL;
  }

  for (size_t i = 0; i < total; i++) {
    void *s = array_get_unchecked(src, indices);
    void *d = (char *)dst->data + i * dst->elem_size;
    memcpy(d, s, dst->elem_size);

    for (ssize_t k = src->ndim - 1; k >= 0; k--) {
      if (++indices[k] < src->shape[k])
        break;
      indices[k] = 0;
    }
  }

  free(indices);
  return dst;
}

int array_append(Array *array, const void *elem) {
  if (!array || !elem)
    return -1;

  if (array->ndim != 1)
    return -2;

  if (!array->owns_data)
    return -1;

  if (array->size >= array->capacity) {
    // Geometric growth: double capacity (or start with 8 if capacity is 0)
    size_t new_capacity = array->capacity == 0 ? 8 : array->capacity * 2;
    void *new_data = aligned_realloc(array->data, NUMC_ALIGN,
                                     array->capacity * array->elem_size,
                                     new_capacity * array->elem_size);
    if (!new_data)
      return -1;

    array->data = new_data;
    array->capacity = new_capacity;
  }

  // Append the element
  memcpy((char *)array->data + array->size * array->elem_size, elem,
         array->elem_size);

  array->size++;
  array->shape[0] = array->size;

  return 0;
}

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

  Array *result = array_create(a->ndim, new_shape, a->dtype);
  free(new_shape);
  if (!result)
    return NULL;

  size_t *indices = calloc(a->ndim, sizeof(size_t));
  size_t *dst_indices = malloc(a->ndim * sizeof(size_t));
  if (!indices || !dst_indices)
    goto fail;

  // Copy array a using unchecked access
  size_t total_a = a->size;
  for (size_t i = 0; i < total_a; i++) {
    void *src = array_get_unchecked(a, indices);
    void *dst = array_get_unchecked(result, indices);
    memcpy(dst, src, a->elem_size);

    for (ssize_t k = a->ndim - 1; k >= 0; k--) {
      if (++indices[k] < a->shape[k])
        break;
      indices[k] = 0;
    }
  }

  memset(indices, 0, a->ndim * sizeof(size_t));

  // Copy array b using unchecked access
  size_t total_b = b->size;
  for (size_t i = 0; i < total_b; i++) {
    for (size_t j = 0; j < a->ndim; j++)
      dst_indices[j] = indices[j];
    dst_indices[axis] += a->shape[axis];

    void *src = array_get_unchecked(b, indices);
    void *dst = array_get_unchecked(result, dst_indices);
    memcpy(dst, src, b->elem_size);

    for (ssize_t k = b->ndim - 1; k >= 0; k--) {
      if (++indices[k] < b->shape[k])
        break;
      indices[k] = 0;
    }
  }

  free(indices);
  free(dst_indices);
  return result;

fail:
  if (indices)
    free(indices);
  if (dst_indices)
    free(dst_indices);
  array_free(result);
  return NULL;
}

Array *array_zeros(size_t ndim, const size_t *shape, DType dtype) {
  Array *arr = array_create(ndim, shape, dtype);
  if (!arr)
    return NULL;

  // memset with 0 is safe for all types (all bits zero = 0)
  memset(arr->data, 0, arr->size * arr->elem_size);
  return arr;
}

Array *array_fill(size_t ndim, const size_t *shape, DType dtype,
                  const void *elem) {
  Array *arr = array_create(ndim, shape, dtype);
  if (!arr)
    return NULL;

  // Now we can use dtype to determine the correct fill value!
  switch (dtype) {
  case DTYPE_BYTE: {
    NUMC_BYTE *data = (NUMC_BYTE *)arr->data;
    NUMC_BYTE value = *((const NUMC_BYTE *)elem);
    for (size_t i = 0; i < arr->size; i++)
      data[i] = value;
    break;
  }
  case DTYPE_UBYTE: {
    NUMC_UBYTE *data = (NUMC_UBYTE *)arr->data;
    NUMC_UBYTE value = *((const NUMC_UBYTE *)elem);
    for (size_t i = 0; i < arr->size; i++)
      data[i] = value;
    break;
  }
  case DTYPE_SHORT: {
    NUMC_SHORT *data = (NUMC_SHORT *)arr->data;
    NUMC_SHORT value = *((const NUMC_SHORT *)elem);
    for (size_t i = 0; i < arr->size; i++)
      data[i] = value;
    break;
  }
  case DTYPE_USHORT: {
    NUMC_USHORT *data = (NUMC_USHORT *)arr->data;
    NUMC_USHORT value = *((const NUMC_USHORT *)elem);
    for (size_t i = 0; i < arr->size; i++)
      data[i] = value;
    break;
  }
  case DTYPE_INT: {
    NUMC_INT *data = (NUMC_INT *)arr->data;
    NUMC_INT value = *((const NUMC_INT *)elem);
    for (size_t i = 0; i < arr->size; i++)
      data[i] = value;
    break;
  }
  case DTYPE_UINT: {
    NUMC_UINT *data = (NUMC_UINT *)arr->data;
    NUMC_UINT value = *((const NUMC_UINT *)elem);
    for (size_t i = 0; i < arr->size; i++)
      data[i] = value;
    break;
  }
  case DTYPE_LONG: {
    NUMC_LONG *data = (NUMC_LONG *)arr->data;
    NUMC_LONG value = *((const NUMC_LONG *)elem);
    for (size_t i = 0; i < arr->size; i++)
      data[i] = value;
    break;
  }
  case DTYPE_ULONG: {
    NUMC_ULONG *data = (NUMC_ULONG *)arr->data;
    NUMC_ULONG value = *((const NUMC_ULONG *)elem);
    for (size_t i = 0; i < arr->size; i++)
      data[i] = value;
    break;
  }
  case DTYPE_FLOAT: {
    NUMC_FLOAT *data = (NUMC_FLOAT *)arr->data;
    NUMC_FLOAT value = *((const NUMC_FLOAT *)elem);
    for (size_t i = 0; i < arr->size; i++)
      data[i] = value;
    break;
  }
  case DTYPE_DOUBLE: {
    NUMC_DOUBLE *data = (NUMC_DOUBLE *)arr->data;
    NUMC_DOUBLE value = *((const NUMC_DOUBLE *)elem);
    for (size_t i = 0; i < arr->size; i++)
      data[i] = value;
    break;
  }
  }

  return arr;
}

Array *array_ones(size_t ndim, const size_t *shape, DType dtype) {
  Array *arr = array_create(ndim, shape, dtype);
  if (!arr)
    return NULL;

  // Now we can use dtype to determine the correct fill value!
  switch (dtype) {
  case DTYPE_BYTE: {
    NUMC_BYTE *data = (NUMC_BYTE *)arr->data;
    for (size_t i = 0; i < arr->size; i++)
      data[i] = 1;
    break;
  }
  case DTYPE_UBYTE: {
    NUMC_UBYTE *data = (NUMC_UBYTE *)arr->data;
    for (size_t i = 0; i < arr->size; i++)
      data[i] = 1;
    break;
  }
  case DTYPE_SHORT: {
    NUMC_SHORT *data = (NUMC_SHORT *)arr->data;
    for (size_t i = 0; i < arr->size; i++)
      data[i] = 1;
    break;
  }
  case DTYPE_USHORT: {
    NUMC_USHORT *data = (NUMC_USHORT *)arr->data;
    for (size_t i = 0; i < arr->size; i++)
      data[i] = 1;
    break;
  }
  case DTYPE_INT: {
    NUMC_INT *data = (NUMC_INT *)arr->data;
    for (size_t i = 0; i < arr->size; i++)
      data[i] = 1;
    break;
  }
  case DTYPE_UINT: {
    NUMC_UINT *data = (NUMC_UINT *)arr->data;
    for (size_t i = 0; i < arr->size; i++)
      data[i] = 1;
    break;
  }
  case DTYPE_LONG: {
    NUMC_LONG *data = (NUMC_LONG *)arr->data;
    for (size_t i = 0; i < arr->size; i++)
      data[i] = 1;
    break;
  }
  case DTYPE_ULONG: {
    NUMC_ULONG *data = (NUMC_ULONG *)arr->data;
    for (size_t i = 0; i < arr->size; i++)
      data[i] = 1;
    break;
  }
  case DTYPE_FLOAT: {
    NUMC_FLOAT *data = (NUMC_FLOAT *)arr->data;
    for (size_t i = 0; i < arr->size; i++)
      data[i] = 1.0f; // ✓ Now we can use 1.0f for float!
    break;
  }
  case DTYPE_DOUBLE: {
    NUMC_DOUBLE *data = (NUMC_DOUBLE *)arr->data;
    for (size_t i = 0; i < arr->size; i++)
      data[i] = 1.0; // ✓ And 1.0 for double!
    break;
  }
  }

  return arr;
}
