/**
 * @file array.c
 * @brief Core array operations: creation, access, properties, copy, and views.
 */

#include "array.h"
#include "alloc.h"
#include "types.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

// =============================================================================
//                          Type-Specific Fill Functions
// =============================================================================

// Optimized fill for BYTE types using memset
static inline void array_fill_BYTE(Array *arr, const void *elem) {
  const NUMC_BYTE value = *((const NUMC_BYTE *)elem);
  memset(arr->data, (unsigned char)value, arr->size);
}

static inline void array_fill_UBYTE(Array *arr, const void *elem) {
  const NUMC_UBYTE value = *((const NUMC_UBYTE *)elem);
  memset(arr->data, value, arr->size);
}

// Generate fill functions for remaining numc_types
#define GEN_FILL_FUNC(numc_type, ctype)                                            \
  static inline void array_fill_##numc_type(Array *restrict arr,                   \
                                        const void *elem) {                    \
    const ctype value = *((const ctype *)elem);                                \
    ctype *restrict data = __builtin_assume_aligned(arr->data, NUMC_ALIGN);    \
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
GEN_FILL_FUNC(LONG, NUMC_LONG)
GEN_FILL_FUNC(ULONG, NUMC_ULONG)
#undef GEN_FILL_FUNC

// Generate constants for "1" value for each type
#define GEN_ONE_CONSTANT(numc_type, ctype)                                         \
  static const ctype one_##numc_type = (ctype)1;
FOREACH_NUMC_TYPE(GEN_ONE_CONSTANT)
#undef GEN_ONE_CONSTANT

// Dispatch tables
typedef void (*fill_func)(Array *, const void *);

#define FILL_ENTRY(numc_type, ctype) [NUMC_TYPE_##numc_type] = array_fill_##numc_type,
static const fill_func fill_funcs[] = {FOREACH_NUMC_TYPE(FILL_ENTRY)};
#undef FILL_ENTRY

#define ONE_PTR_ENTRY(numc_type, ctype) [NUMC_TYPE_##numc_type] = &one_##numc_type,
static const void *const one_ptrs[] = {FOREACH_NUMC_TYPE(ONE_PTR_ENTRY)};
#undef ONE_PTR_ENTRY

// =============================================================================
//                          Internal Helper Functions
// =============================================================================

// =============================================================================
//                          Array Creation Functions
// =============================================================================

Array *array_create(const ArrayCreate *src) {
  if (!src || src->ndim == 0 || !src->shape)
    return NULL;

  size_t elem_size = numc_type_size(src->numc_type);

  Array *array = malloc(sizeof(Array));
  if (!array)
    return NULL;

  array->ndim = src->ndim;
  array->numc_type = src->numc_type;
  array->elem_size = elem_size;
  array->is_contiguous = true;

  bool use_stack = src->ndim <= MAX_STACK_NDIM;
  if (use_stack) {
    array->shape = array->_shape_buff;
    array->strides = array->_strides_buff;
  } else {
    array->shape = malloc(2 * src->ndim * sizeof(size_t));
    array->strides = array->shape + src->ndim;
    if (!array->shape) {
      free(array);
      return NULL;
    }
  }

  array->owns_data = src->owns_data;

  array->size = 1;
  for (size_t i = 0; i < src->ndim; i++) {
    array->shape[i] = src->shape[i];
    // Check for overflow before multiplication
    if (src->shape[i] > 0 && array->size > SIZE_MAX / src->shape[i]) {
      if (!use_stack)
        free(array->shape);
      free(array);
      return NULL;
    }
    array->size *= src->shape[i];
  }

  array->strides[src->ndim - 1] = elem_size;
  for (ssize_t i = src->ndim - 2; i >= 0; i--)
    array->strides[i] = array->strides[i + 1] * src->shape[i + 1];

  array->capacity = array->size;
  if (src->owns_data) {
    array->data = !src->data ? numc_calloc(NUMC_ALIGN, array->size * elem_size)
                             : numc_malloc(NUMC_ALIGN, array->size * elem_size);

    if (!array->data) {
      if (!use_stack)
        free(array->shape);
      free(array);
      return NULL;
    }

    if (src->data != NULL)
      memcpy(array->data, src->data, array->size * elem_size);

  } else {
    array->data = (void *)src->data;
  }

  return array;
}

Array *array_zeros(size_t ndim, const size_t *shape, NUMC_TYPE numc_type) {
  ArrayCreate src = {
      .ndim = ndim,
      .shape = shape,
      .numc_type = numc_type,
      .data = NULL,
      .owns_data = true,
  };

  return array_create(&src);
}

Array *array_full(ArrayCreate *spec, const void *elem) {
  Array *arr = array_create(spec);
  if (!arr)
    return NULL;

  fill_funcs[arr->numc_type](arr, elem);

  return arr;
}

Array *array_ones(size_t ndim, const size_t *shape, NUMC_TYPE numc_type) {
  ArrayCreate src = {
      .ndim = ndim,
      .shape = shape,
      .numc_type = numc_type,
      .data = NULL,
      .owns_data = true,
  };

  Array *arr = array_create(&src);
  if (!arr)
    return NULL;

  fill_funcs[arr->numc_type](arr, one_ptrs[arr->numc_type]);
  return arr;
}

void array_free(Array *array) {
  if (!array)
    return;

  if (array->owns_data)
    numc_free(array->data);

  if (array->ndim > MAX_STACK_NDIM)
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

  // Lightweight view allocation — skip array_create overhead
  Array *view = malloc(sizeof(Array));
  if (!view)
    return NULL;

  view->ndim = base->ndim;
  bool use_stack = base->ndim <= MAX_STACK_NDIM;
  if (use_stack) {
    view->shape = view->_shape_buff;
    view->strides = view->_strides_buff;
  } else {
    view->shape = malloc(2 * base->ndim * sizeof(size_t));
    if (!view->shape) {
      free(view);
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
      fprintf(stderr,
              "array_slice: invalid slice at dimension %zu "
              "(start=%zu, stop=%zu, step=%zu, shape=%zu)\n",
              i, start[i], stop[i], step[i], base->shape[i]);
      if (!use_stack)
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

  view->is_contiguous = array_is_contiguous(view);
  view->capacity = view->size;
  view->data = (char *)base->data + offset;
  return view;
}

// =============================================================================
//                    Strided Copy Helpers (Static Functions)
// =============================================================================

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

#define GET_DEST_DATA(numc_type, ctype)                                            \
  static inline void *get_dest_data_##numc_type(void *data, size_t dest_offset) {  \
    return (ctype *)data + dest_offset;                                        \
  }

#define MOVE_DEST_DATA(numc_type, ctype)                                           \
  static inline void *move_dest_data_##numc_type(void *data, size_t chunk) {       \
    return (ctype *)data + chunk;                                              \
  }

#define STRIDED_TO_CONTIGUOUS_COPY_GENERAL(numc_type, ctype)                       \
  static inline void strided_to_contiguous_copy_##numc_type(                       \
      size_t *src_indices, const Array *src, size_t count, void *dest) {       \
                                                                               \
    const size_t *restrict strides = src->strides;                             \
    const char *restrict src_data = (const char *)src->data;                   \
    ctype *restrict ddest = (ctype *)dest;                                     \
                                                                               \
    for (size_t i = 0; i < count; i++) {                                       \
      size_t offset = 0;                                                       \
      for (size_t d = 0; d < src->ndim; d++) {                                 \
        offset += src_indices[d] * strides[d];                                 \
      }                                                                        \
      ddest[i] = *((const ctype *)(src_data + offset));                        \
      increment_indices(src_indices, src->shape, src->ndim);                   \
    }                                                                          \
  }

FOREACH_NUMC_TYPE(GET_DEST_DATA)
FOREACH_NUMC_TYPE(MOVE_DEST_DATA)
FOREACH_NUMC_TYPE(STRIDED_TO_CONTIGUOUS_COPY_GENERAL)

#undef GET_DEST_DATA
#undef MOVE_DEST_DATA
#undef STRIDED_TO_CONTIGUOUS_COPY_GENERAL

typedef void *(*get_dest_data_func)(void *, size_t);
typedef void *(*move_dest_data_func)(void *, size_t);
typedef void (*strided_to_contiguous_copy_func)(size_t *, const Array *, size_t,
                                                void *);

#define GET_DEST_DATA_ENTRY(numc_type, ctype)                                      \
  [NUMC_TYPE_##numc_type] = (get_dest_data_func)get_dest_data_##numc_type,

static const get_dest_data_func get_dest_data_funcs[] = {
    FOREACH_NUMC_TYPE(GET_DEST_DATA_ENTRY)};

#define MOVE_DEST_DATA_ENTRY(numc_type, ctype)                                     \
  [NUMC_TYPE_##numc_type] = (move_dest_data_func)move_dest_data_##numc_type,

static const move_dest_data_func move_dest_data_funcs[] = {
    FOREACH_NUMC_TYPE(MOVE_DEST_DATA_ENTRY)};

#define STRIDED_TO_CONTIGUOUS_COPY_ENTRY(numc_type, ctype)                         \
  [NUMC_TYPE_##numc_type] =                                                            \
      (strided_to_contiguous_copy_func)strided_to_contiguous_copy_##numc_type,

static const strided_to_contiguous_copy_func
    strided_to_contiguous_copy_funcs[] = {
        FOREACH_NUMC_TYPE(STRIDED_TO_CONTIGUOUS_COPY_ENTRY)};

#undef GET_ARRAY_DATA_ENTRY
#undef GET_DATA_ENTRY
#undef GET_DEST_DATA_ENTRY
#undef MOVE_DEST_DATA_ENTRY

static inline void
strided_to_contiguous_copy(const Array *src, size_t *src_indices,
                           void *dest_data, size_t dest_offset, size_t count) {

  const char *restrict src_data = (const char *)src->data;
  const size_t esize = src->elem_size;
  void *dest = get_dest_data_funcs[src->numc_type](dest_data, dest_offset);
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
      dest = move_dest_data_funcs[src->numc_type](dest, chunk);
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

  strided_to_contiguous_copy_funcs[src->numc_type](src_indices, src, count, dest);
}

// =============================================================================
//                          Array Copying
// =============================================================================

Array *array_copy(const Array *src) {
  if (!src)
    return NULL;

  // If contiguous, use fast memcpy path
  if (!src->is_contiguous) {
    fprintf(stderr, "[ERROR] array_copy: array is not contiguous, "
                    "use array_ascontiguousarray() first\n");
    abort();
  }

  ArrayCreate d = {
      .ndim = src->ndim,
      .shape = src->shape,
      .numc_type = src->numc_type,
      .data = src->data,
      .owns_data = true,
  };

  Array *arr = array_create(&d);
  if (!arr)
    return NULL;

  return arr;
}

Array *array_ascontiguousarray(const Array *src) {
  if (!src)
    return NULL;

  ArrayCreate d = {
      .ndim = src->ndim,
      .shape = src->shape,
      .numc_type = src->numc_type,
      .data = src->data,
      .owns_data = true,
  };

  // If already contiguous, just create a copy
  if (src->is_contiguous) {
    Array *arr = array_create(&d);
    if (!arr)
      return NULL;
    return arr;
  }

  // Non-contiguous: use strided copy
  size_t indices_buf[MAX_STACK_NDIM] = {0};
  size_t *indices = (src->ndim <= MAX_STACK_NDIM)
                        ? indices_buf
                        : calloc(src->ndim, sizeof(size_t));
  if (src->ndim > MAX_STACK_NDIM && !indices)
    return NULL;

  // Don't pass src->data — it would be memcpy'd incorrectly for
  // non-contiguous arrays. strided_to_contiguous_copy handles the copy.
  ArrayCreate d_nodata = {
      .ndim = src->ndim,
      .shape = src->shape,
      .numc_type = src->numc_type,
      .data = NULL,
      .owns_data = true,
  };
  Array *dst = array_create(&d_nodata);
  if (!dst) {
    if (src->ndim > MAX_STACK_NDIM)
      free(indices);
    return NULL;
  }

  strided_to_contiguous_copy(src, indices, dst->data, 0, src->size);

  if (src->ndim > MAX_STACK_NDIM)
    free(indices);

  return dst;
}
