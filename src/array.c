/**
 * @file array.c
 * @brief Core array operations: creation, access, properties, copy, and views.
 */

#include "array.h"
#include "alloc.h"
#include "types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

// =============================================================================
//                          Type-Specific Fill Functions
// =============================================================================

// Optimized fill for BYTE types using memset
static inline void array_fill_BYTE(Array *arr, const void *elem) {
  const int8_t value = *((const int8_t *)elem);
  memset(arr->data, (unsigned char)value, arr->size);
}

static inline void array_fill_UBYTE(Array *arr, const void *elem) {
  const uint8_t value = *((const uint8_t *)elem);
  memset(arr->data, value, arr->size);
}

// Optimized fill for LONG types with vectorization hints
static inline void array_fill_LONG(Array *arr, const void *elem) {
  const int64_t value = *((const int64_t *)elem);
  int64_t *data = (int64_t *)arr->data;
  const size_t size = arr->size;
  
#pragma clang loop vectorize(enable) interleave(enable)
  for (size_t i = 0; i < size; i++) {
    data[i] = value;
  }
}

static inline void array_fill_ULONG(Array *arr, const void *elem) {
  const uint64_t value = *((const uint64_t *)elem);
  uint64_t *data = (uint64_t *)arr->data;
  const size_t size = arr->size;
  
#pragma clang loop vectorize(enable) interleave(enable)
  for (size_t i = 0; i < size; i++) {
    data[i] = value;
  }
}

// Generate fill functions for remaining dtypes
#define GEN_FILL_FUNC(dtype, ctype)                                            \
  static inline void array_fill_##dtype(Array *arr, const void *elem) {        \
    const ctype value = *((const ctype *)elem);                                \
    ctype *data = (ctype *)arr->data;                                          \
    for (size_t i = 0; i < arr->size; i++) {                                   \
      data[i] = value;                                                         \
    }                                                                          \
  }

// Only generate for types not manually defined above
GEN_FILL_FUNC(SHORT, NUMC_SHORT)
GEN_FILL_FUNC(USHORT, NUMC_USHORT)
GEN_FILL_FUNC(INT, NUMC_INT)
GEN_FILL_FUNC(UINT, NUMC_UINT)
GEN_FILL_FUNC(FLOAT, NUMC_FLOAT)
GEN_FILL_FUNC(DOUBLE, NUMC_DOUBLE)
#undef GEN_FILL_FUNC

// Generate constants for "1" value for each type
#define GEN_ONE_CONSTANT(dtype, ctype)                                         \
  static const ctype one_##dtype = (ctype)1;
FOREACH_DTYPE(GEN_ONE_CONSTANT)
#undef GEN_ONE_CONSTANT

// Dispatch tables
typedef void (*fill_func)(Array *, const void *);

#define FILL_ENTRY(dtype, ctype) [DTYPE_##dtype] = array_fill_##dtype,
static const fill_func fill_funcs[] = {FOREACH_DTYPE(FILL_ENTRY)};
#undef FILL_ENTRY

#define ONE_PTR_ENTRY(dtype, ctype) [DTYPE_##dtype] = &one_##dtype,
static const void *const one_ptrs[] = {FOREACH_DTYPE(ONE_PTR_ENTRY)};
#undef ONE_PTR_ENTRY

// =============================================================================
//                          Internal Helper Functions
// =============================================================================

/**
 * @brief Fill array with a specific value (type-aware).
 */
static inline void array_fill_with(Array *arr, const void *elem) {
  fill_funcs[arr->dtype](arr, elem);
}

// =============================================================================
//                          Array Creation Functions
// =============================================================================

Array *array_create(size_t ndim, const size_t *shape, DType dtype,
                    const void *data) {
  size_t elem_size = dtype_size(dtype);
  if (ndim == 0 || elem_size == 0 || !shape)
    return NULL;

  Array *array = malloc(sizeof(Array));
  if (!array)
    return NULL;

  array->ndim = ndim;
  array->dtype = dtype;
  array->elem_size = elem_size;

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
    // Check for overflow before multiplication
    if (shape[i] > 0 && array->size > SIZE_MAX / shape[i]) {
      free(array->shape);
      free(array);
      return NULL;
    }
    array->size *= shape[i];
  }

  array->strides[ndim - 1] = elem_size;
  for (ssize_t i = ndim - 2; i >= 0; i--)
    array->strides[i] = array->strides[i + 1] * shape[i + 1];

  array->capacity = array->size;
  array->data = numc_calloc(NUMC_ALIGN, array->size * elem_size);
  if (!array->data) {
    free(array->shape);
    free(array);
    return NULL;
  }

  if (data != NULL)
    memcpy(array->data, data, array->size * elem_size);

  return array;
}

Array *array_zeros(size_t ndim, const size_t *shape, DType dtype) {
  Array *arr = array_create(ndim, shape, dtype, NULL);
  if (!arr)
    return NULL;

  memset(arr->data, 0, arr->size * arr->elem_size);
  return arr;
}

Array *array_fill(size_t ndim, const size_t *shape, DType dtype,
                  const void *elem) {
  Array *arr = array_create(ndim, shape, dtype, NULL);
  if (!arr)
    return NULL;

  array_fill_with(arr, elem);

  return arr;
}

Array *array_ones(size_t ndim, const size_t *shape, DType dtype) {
  Array *arr = array_create(ndim, shape, dtype, NULL);
  if (!arr)
    return NULL;

  fill_funcs[arr->dtype](arr, one_ptrs[arr->dtype]);
  return arr;
}

void array_free(Array *array) {
  if (!array)
    return;

  if (array->owns_data)
    numc_free(array->data);

  free(array->shape);
  free(array);
}

// =============================================================================
//                          Element Access Functions
// =============================================================================

size_t array_offset(const Array *array, const size_t *indices) {
  if (!array || !indices)
    return 0;

  size_t offset = 0;
  for (size_t i = 0; i < array->ndim; i++) {
    offset += indices[i] * array->strides[i];
  }

  return offset;
}

int array_bounds_check(const Array *array, const size_t *indices) {
  if (!array || !indices)
    return -1;

  for (size_t i = 0; i < array->ndim; i++) {
    if (indices[i] >= array->shape[i])
      return -1;
  }

  return 0;
}

void *array_get(const Array *array, const size_t *indices) {
  if (!array || !indices)
    return NULL;

  size_t offset = array_offset(array, indices);
  return (char *)array->data + offset;
}

// =============================================================================
//                          Array Properties
// =============================================================================

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

// =============================================================================
//                          Array Slicing (Views)
// =============================================================================

Array *array_slice(Array *base, const size_t *start, const size_t *stop,
                   const size_t *step) {
  if (!base || !start || !stop || !step)
    return NULL;

  Array *view = malloc(sizeof(Array));
  if (!view)
    return NULL;

  view->ndim = base->ndim;
  view->dtype = base->dtype;
  view->elem_size = base->elem_size;

  view->shape = malloc(2 * view->ndim * sizeof(size_t));
  if (!view->shape) {
    free(view);
    return NULL;
  }
  view->strides = view->shape + view->ndim;

  // view does NOT own data, base keeps ownership
  view->owns_data = 0;
  view->capacity = 0;

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

// Generate strided-to-contiguous copy functions for each dtype
#define GEN_STRIDED_TO_CONTIGUOUS_COPY_FUNC(dtype, ctype)                      \
  static inline void array_strided_to_contiguous_copy_##dtype(                 \
      const Array *src, size_t *src_indices, void *dest_data,                  \
      size_t dest_offset, size_t count) {                                      \
    ctype *restrict dest = (ctype *)dest_data + dest_offset;                   \
    const size_t ndim = src->ndim;                                             \
    const size_t *restrict shape = src->shape;                                 \
    const size_t *restrict strides = src->strides;                             \
    const char *restrict src_data = (const char *)src->data;                   \
                                                                               \
    /* Fast path: Contiguous array - use memcpy */                             \
    if (ndim == 1 && strides[0] == sizeof(ctype)) {                            \
      memcpy(dest, src_data + src_indices[0] * strides[0],                     \
             count * sizeof(ctype));                                           \
      src_indices[0] += count;                                                 \
      return;                                                                  \
    }                                                                          \
                                                                               \
    /* Fast path: 2D contiguous (row-major) */                                 \
    if (ndim == 2 && strides[1] == sizeof(ctype) &&                            \
        strides[0] == shape[1] * sizeof(ctype)) {                              \
      size_t remaining = count;                                                \
      while (remaining > 0) {                                                  \
        size_t row = src_indices[0];                                           \
        size_t col = src_indices[1];                                           \
        size_t row_remaining = shape[1] - col;                                 \
        size_t chunk =                                                         \
            (remaining < row_remaining) ? remaining : row_remaining;           \
        memcpy(dest, src_data + row * strides[0] + col * strides[1],           \
               chunk * sizeof(ctype));                                         \
        dest += chunk;                                                         \
        remaining -= chunk;                                                    \
        col += chunk;                                                          \
        if (col >= shape[1]) {                                                 \
          col = 0;                                                             \
          row++;                                                               \
        }                                                                      \
        src_indices[0] = row;                                                  \
        src_indices[1] = col;                                                  \
      }                                                                        \
      return;                                                                  \
    }                                                                          \
                                                                               \
    /* General case: strided access */                                         \
    for (size_t i = 0; i < count; i++) {                                       \
      size_t offset = 0;                                                       \
      for (size_t d = 0; d < ndim; d++) {                                      \
        offset += src_indices[d] * strides[d];                                 \
      }                                                                        \
      dest[i] = *((const ctype *)(src_data + offset));                         \
                                                                               \
      increment_indices(src_indices, shape, ndim);                             \
    }                                                                          \
  }

FOREACH_DTYPE(GEN_STRIDED_TO_CONTIGUOUS_COPY_FUNC)
#undef GEN_STRIDED_TO_CONTIGUOUS_COPY_FUNC

typedef void (*strided_to_contiguous_copy_func)(const Array *, size_t *, void *,
                                                size_t, size_t);

#define STRIDED_TO_CONTIGUOUS_COPY_ENTRY(dtype, ctype)                         \
  [DTYPE_##dtype] = array_strided_to_contiguous_copy_##dtype,

static const strided_to_contiguous_copy_func
    strided_to_contiguous_copy_funcs[] = {
        FOREACH_DTYPE(STRIDED_TO_CONTIGUOUS_COPY_ENTRY)};
#undef STRIDED_TO_CONTIGUOUS_COPY_ENTRY

static inline void array_strided_copy_to_contiguous(const Array *src,
                                                    size_t *src_indices,
                                                    void *dest_data,
                                                    size_t dest_offset,
                                                    size_t count) {
  strided_to_contiguous_copy_funcs[src->dtype](src, src_indices, dest_data,
                                               dest_offset, count);
}

// =============================================================================
//                          Array Copying
// =============================================================================

Array *array_copy(const Array *src) {
  if (!src)
    return NULL;

  // If contiguous, use fast memcpy path
  if (!array_is_contiguous(src)) {
    fprintf(stderr, "[ERROR] array_copy: array is not contiguous, "
                    "use array_to_contiguous() first\n");
    abort();
  }

  return array_create(src->ndim, src->shape, src->dtype, src->data);
}

Array *array_to_contiguous(const Array *src) {
  if (!src)
    return NULL;

  // If already contiguous, just create a copy
  if (array_is_contiguous(src)) {
    return array_create(src->ndim, src->shape, src->dtype, src->data);
  }

  // Non-contiguous: use strided copy
  size_t indices_buf[MAX_STACK_NDIM] = {0};
  size_t *indices = (src->ndim <= MAX_STACK_NDIM)
                        ? indices_buf
                        : calloc(src->ndim, sizeof(size_t));
  if (src->ndim > MAX_STACK_NDIM && !indices)
    return NULL;

  Array *dst = array_create(src->ndim, src->shape, src->dtype, NULL);
  if (!dst) {
    if (src->ndim > MAX_STACK_NDIM)
      free(indices);
    return NULL;
  }

  array_strided_copy_to_contiguous(src, indices, dst->data, 0, src->size);

  if (src->ndim > MAX_STACK_NDIM)
    free(indices);

  return dst;
}
