#include "array.h"
#include "dtype.h"
#include "memory.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

// =============================================================================
//                          X-Macro Type Definitions
// =============================================================================

/**
 * @brief X-Macro: Define all data types in one place
 *
 * This macro is used to generate type-specific functions automatically,
 * eliminating code duplication and switch statements.
 */
#define FOREACH_DTYPE(X)                                                       \
  X(BYTE, NUMC_BYTE)                                                           \
  X(UBYTE, NUMC_UBYTE)                                                         \
  X(SHORT, NUMC_SHORT)                                                         \
  X(USHORT, NUMC_USHORT)                                                       \
  X(INT, NUMC_INT)                                                             \
  X(UINT, NUMC_UINT)                                                           \
  X(LONG, NUMC_LONG)                                                           \
  X(ULONG, NUMC_ULONG)                                                         \
  X(FLOAT, NUMC_FLOAT)                                                         \
  X(DOUBLE, NUMC_DOUBLE)

// -----------------------------------------------------------------------------
//                              Static Functions
// -----------------------------------------------------------------------------

/**
 * @brief Unchecked element access for internal use (no bounds checking)
 *
 * @param array Pointer to the array.
 * @param indices Array of indices for each dimension.
 * @return Pointer to the element.
 */
static inline void *array_get_unchecked(const Array *array,
                                        const size_t *indices) {
  size_t offset = 0;
  for (size_t i = 0; i < array->ndim; i++) {
    offset += indices[i] * array->strides[i];
  }
  return (char *)array->data + offset;
}

// =============================================================================
// Type-specific addition kernels (generated via X-Macro)
// Compiler auto-vectorizes these simple loops with -O3
// =============================================================================

/**
 * @brief Generate type-specific addition function
 *
 * Each function is a simple loop that the compiler can auto-vectorize.
 * The function signature uses void* for generic handling, but internally
 * we cast to the specific type for type safety and SIMD optimization.
 */
#define GENERATE_ADD_FUNC(dtype_name, c_type)                                  \
  static inline void add_##dtype_name(const void *a, const void *b, void *out, \
                                      size_t n) {                              \
    const c_type *pa = (const c_type *)a;                                      \
    const c_type *pb = (const c_type *)b;                                      \
    c_type *pout = (c_type *)out;                                              \
    for (size_t i = 0; i < n; i++) {                                           \
      pout[i] = pa[i] + pb[i];                                                 \
    }                                                                          \
  }

// Generate all add functions: add_BYTE, add_UBYTE, add_SHORT, etc.
FOREACH_DTYPE(GENERATE_ADD_FUNC)
#undef GENERATE_ADD_FUNC

// Function pointer type for binary operations
typedef void (*binary_op_func)(const void *, const void *, void *, size_t);

// Function pointer table indexed by DType enum
#define ADD_FUNC_ENTRY(dtype_name, c_type)                                     \
  [DTYPE_##dtype_name] = add_##dtype_name,
static const binary_op_func add_funcs[] = {FOREACH_DTYPE(ADD_FUNC_ENTRY)};
#undef ADD_FUNC_ENTRY

/**
 * @brief Increment multi-dimensional indices (row-major order).
 *
 * @param indices Current indices to increment.
 * @param shape   Shape of the array.
 * @param ndim    Number of dimensions.
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

/**
 * @brief Copy elements from strided source to contiguous destination.
 *
 * Iterates through source array using strides, writes sequentially to dest.
 * This is a common pattern used in array_copy() and array_concat().
 *
 * @param src Source array (may be non-contiguous/strided)
 * @param src_indices Starting indices in source (will be modified)
 * @param dest_data Destination data pointer (contiguous)
 * @param dest_offset Starting offset in destination (in elements)
 * @param count Number of elements to copy
 */
static void array_strided_copy_to_contiguous(const Array *src,
                                             size_t *src_indices,
                                             void *dest_data,
                                             size_t dest_offset, size_t count) {
  switch (src->dtype) {
  case DTYPE_BYTE:
    for (size_t i = 0; i < count; i++) {
      NUMC_BYTE *src_elem = (NUMC_BYTE *)array_get_unchecked(src, src_indices);
      ((NUMC_BYTE *)dest_data)[dest_offset + i] = *src_elem;
      increment_indices(src_indices, src->shape, src->ndim);
    }
    break;
  case DTYPE_UBYTE:
    for (size_t i = 0; i < count; i++) {
      NUMC_UBYTE *src_elem =
          (NUMC_UBYTE *)array_get_unchecked(src, src_indices);
      ((NUMC_UBYTE *)dest_data)[dest_offset + i] = *src_elem;
      increment_indices(src_indices, src->shape, src->ndim);
    }
    break;
  case DTYPE_SHORT:
    for (size_t i = 0; i < count; i++) {
      NUMC_SHORT *src_elem =
          (NUMC_SHORT *)array_get_unchecked(src, src_indices);
      ((NUMC_SHORT *)dest_data)[dest_offset + i] = *src_elem;
      increment_indices(src_indices, src->shape, src->ndim);
    }
    break;
  case DTYPE_USHORT:
    for (size_t i = 0; i < count; i++) {
      NUMC_USHORT *src_elem =
          (NUMC_USHORT *)array_get_unchecked(src, src_indices);
      ((NUMC_USHORT *)dest_data)[dest_offset + i] = *src_elem;
      increment_indices(src_indices, src->shape, src->ndim);
    }
    break;
  case DTYPE_INT:
    for (size_t i = 0; i < count; i++) {
      NUMC_INT *src_elem = (NUMC_INT *)array_get_unchecked(src, src_indices);
      ((NUMC_INT *)dest_data)[dest_offset + i] = *src_elem;
      increment_indices(src_indices, src->shape, src->ndim);
    }
    break;
  case DTYPE_UINT:
    for (size_t i = 0; i < count; i++) {
      NUMC_UINT *src_elem = (NUMC_UINT *)array_get_unchecked(src, src_indices);
      ((NUMC_UINT *)dest_data)[dest_offset + i] = *src_elem;
      increment_indices(src_indices, src->shape, src->ndim);
    }
    break;
  case DTYPE_LONG:
    for (size_t i = 0; i < count; i++) {
      NUMC_LONG *src_elem = (NUMC_LONG *)array_get_unchecked(src, src_indices);
      ((NUMC_LONG *)dest_data)[dest_offset + i] = *src_elem;
      increment_indices(src_indices, src->shape, src->ndim);
    }
    break;
  case DTYPE_ULONG:
    for (size_t i = 0; i < count; i++) {
      NUMC_ULONG *src_elem =
          (NUMC_ULONG *)array_get_unchecked(src, src_indices);
      ((NUMC_ULONG *)dest_data)[dest_offset + i] = *src_elem;
      increment_indices(src_indices, src->shape, src->ndim);
    }
    break;
  case DTYPE_FLOAT:
    for (size_t i = 0; i < count; i++) {
      NUMC_FLOAT *src_elem =
          (NUMC_FLOAT *)array_get_unchecked(src, src_indices);
      ((NUMC_FLOAT *)dest_data)[dest_offset + i] = *src_elem;
      increment_indices(src_indices, src->shape, src->ndim);
    }
    break;
  case DTYPE_DOUBLE:
    for (size_t i = 0; i < count; i++) {
      NUMC_DOUBLE *src_elem =
          (NUMC_DOUBLE *)array_get_unchecked(src, src_indices);
      ((NUMC_DOUBLE *)dest_data)[dest_offset + i] = *src_elem;
      increment_indices(src_indices, src->shape, src->ndim);
    }
    break;
  }
}

/**
 * @brief Copy elements from strided source to strided destination with axis
 * offset.
 *
 * Used in array_concat to copy the second array with an offset along the
 * concatenation axis. Iterates through source using strides, writes to
 * destination with indices offset by axis_offset on the given axis.
 *
 * @param src Source array (may be non-contiguous)
 * @param src_indices Starting indices in source (will be modified)
 * @param dst Destination array (may be non-contiguous)
 * @param dst_indices_buf Buffer for destination indices (will be modified)
 * @param axis Axis along which to apply offset
 * @param axis_offset Offset to add to axis dimension
 * @param count Number of elements to copy
 */
static void array_strided_copy_with_offset(const Array *src,
                                           size_t *src_indices, Array *dst,
                                           size_t *dst_indices_buf, size_t axis,
                                           size_t axis_offset, size_t count) {
  switch (src->dtype) {
  case DTYPE_BYTE:
    for (size_t i = 0; i < count; i++) {
      for (size_t j = 0; j < src->ndim; j++)
        dst_indices_buf[j] = src_indices[j];
      dst_indices_buf[axis] += axis_offset;
      NUMC_BYTE *src_elem = (NUMC_BYTE *)array_get_unchecked(src, src_indices);
      NUMC_BYTE *dst_elem =
          (NUMC_BYTE *)array_get_unchecked(dst, dst_indices_buf);
      *dst_elem = *src_elem;
      increment_indices(src_indices, src->shape, src->ndim);
    }
    break;
  case DTYPE_UBYTE:
    for (size_t i = 0; i < count; i++) {
      for (size_t j = 0; j < src->ndim; j++)
        dst_indices_buf[j] = src_indices[j];
      dst_indices_buf[axis] += axis_offset;
      NUMC_UBYTE *src_elem =
          (NUMC_UBYTE *)array_get_unchecked(src, src_indices);
      NUMC_UBYTE *dst_elem =
          (NUMC_UBYTE *)array_get_unchecked(dst, dst_indices_buf);
      *dst_elem = *src_elem;
      increment_indices(src_indices, src->shape, src->ndim);
    }
    break;
  case DTYPE_SHORT:
    for (size_t i = 0; i < count; i++) {
      for (size_t j = 0; j < src->ndim; j++)
        dst_indices_buf[j] = src_indices[j];
      dst_indices_buf[axis] += axis_offset;
      NUMC_SHORT *src_elem =
          (NUMC_SHORT *)array_get_unchecked(src, src_indices);
      NUMC_SHORT *dst_elem =
          (NUMC_SHORT *)array_get_unchecked(dst, dst_indices_buf);
      *dst_elem = *src_elem;
      increment_indices(src_indices, src->shape, src->ndim);
    }
    break;
  case DTYPE_USHORT:
    for (size_t i = 0; i < count; i++) {
      for (size_t j = 0; j < src->ndim; j++)
        dst_indices_buf[j] = src_indices[j];
      dst_indices_buf[axis] += axis_offset;
      NUMC_USHORT *src_elem =
          (NUMC_USHORT *)array_get_unchecked(src, src_indices);
      NUMC_USHORT *dst_elem =
          (NUMC_USHORT *)array_get_unchecked(dst, dst_indices_buf);
      *dst_elem = *src_elem;
      increment_indices(src_indices, src->shape, src->ndim);
    }
    break;
  case DTYPE_INT:
    for (size_t i = 0; i < count; i++) {
      for (size_t j = 0; j < src->ndim; j++)
        dst_indices_buf[j] = src_indices[j];
      dst_indices_buf[axis] += axis_offset;
      NUMC_INT *src_elem = (NUMC_INT *)array_get_unchecked(src, src_indices);
      NUMC_INT *dst_elem =
          (NUMC_INT *)array_get_unchecked(dst, dst_indices_buf);
      *dst_elem = *src_elem;
      increment_indices(src_indices, src->shape, src->ndim);
    }
    break;
  case DTYPE_UINT:
    for (size_t i = 0; i < count; i++) {
      for (size_t j = 0; j < src->ndim; j++)
        dst_indices_buf[j] = src_indices[j];
      dst_indices_buf[axis] += axis_offset;
      NUMC_UINT *src_elem = (NUMC_UINT *)array_get_unchecked(src, src_indices);
      NUMC_UINT *dst_elem =
          (NUMC_UINT *)array_get_unchecked(dst, dst_indices_buf);
      *dst_elem = *src_elem;
      increment_indices(src_indices, src->shape, src->ndim);
    }
    break;
  case DTYPE_LONG:
    for (size_t i = 0; i < count; i++) {
      for (size_t j = 0; j < src->ndim; j++)
        dst_indices_buf[j] = src_indices[j];
      dst_indices_buf[axis] += axis_offset;
      NUMC_LONG *src_elem = (NUMC_LONG *)array_get_unchecked(src, src_indices);
      NUMC_LONG *dst_elem =
          (NUMC_LONG *)array_get_unchecked(dst, dst_indices_buf);
      *dst_elem = *src_elem;
      increment_indices(src_indices, src->shape, src->ndim);
    }
    break;
  case DTYPE_ULONG:
    for (size_t i = 0; i < count; i++) {
      for (size_t j = 0; j < src->ndim; j++)
        dst_indices_buf[j] = src_indices[j];
      dst_indices_buf[axis] += axis_offset;
      NUMC_ULONG *src_elem =
          (NUMC_ULONG *)array_get_unchecked(src, src_indices);
      NUMC_ULONG *dst_elem =
          (NUMC_ULONG *)array_get_unchecked(dst, dst_indices_buf);
      *dst_elem = *src_elem;
      increment_indices(src_indices, src->shape, src->ndim);
    }
    break;
  case DTYPE_FLOAT:
    for (size_t i = 0; i < count; i++) {
      for (size_t j = 0; j < src->ndim; j++)
        dst_indices_buf[j] = src_indices[j];
      dst_indices_buf[axis] += axis_offset;
      NUMC_FLOAT *src_elem =
          (NUMC_FLOAT *)array_get_unchecked(src, src_indices);
      NUMC_FLOAT *dst_elem =
          (NUMC_FLOAT *)array_get_unchecked(dst, dst_indices_buf);
      *dst_elem = *src_elem;
      increment_indices(src_indices, src->shape, src->ndim);
    }
    break;
  case DTYPE_DOUBLE:
    for (size_t i = 0; i < count; i++) {
      for (size_t j = 0; j < src->ndim; j++)
        dst_indices_buf[j] = src_indices[j];
      dst_indices_buf[axis] += axis_offset;
      NUMC_DOUBLE *src_elem =
          (NUMC_DOUBLE *)array_get_unchecked(src, src_indices);
      NUMC_DOUBLE *dst_elem =
          (NUMC_DOUBLE *)array_get_unchecked(dst, dst_indices_buf);
      *dst_elem = *src_elem;
      increment_indices(src_indices, src->shape, src->ndim);
    }
    break;
  }
}

/**
 * @brief Copy elements from strided source to strided destination.
 *
 * Used in array_concat to copy array 'a' with matching stride patterns.
 * Both source and destination use stride-based indexing.
 *
 * @param src Source array (may be non-contiguous)
 * @param src_indices Starting indices in source (will be modified)
 * @param dst Destination array (may be non-contiguous)
 * @param count Number of elements to copy
 */
static void array_strided_copy(const Array *src, size_t *src_indices,
                               Array *dst, size_t count) {
  switch (src->dtype) {
  case DTYPE_BYTE:
    for (size_t i = 0; i < count; i++) {
      NUMC_BYTE *src_elem = (NUMC_BYTE *)array_get_unchecked(src, src_indices);
      NUMC_BYTE *dst_elem = (NUMC_BYTE *)array_get_unchecked(dst, src_indices);
      *dst_elem = *src_elem;
      increment_indices(src_indices, src->shape, src->ndim);
    }
    break;
  case DTYPE_UBYTE:
    for (size_t i = 0; i < count; i++) {
      NUMC_UBYTE *src_elem =
          (NUMC_UBYTE *)array_get_unchecked(src, src_indices);
      NUMC_UBYTE *dst_elem =
          (NUMC_UBYTE *)array_get_unchecked(dst, src_indices);
      *dst_elem = *src_elem;
      increment_indices(src_indices, src->shape, src->ndim);
    }
    break;
  case DTYPE_SHORT:
    for (size_t i = 0; i < count; i++) {
      NUMC_SHORT *src_elem =
          (NUMC_SHORT *)array_get_unchecked(src, src_indices);
      NUMC_SHORT *dst_elem =
          (NUMC_SHORT *)array_get_unchecked(dst, src_indices);
      *dst_elem = *src_elem;
      increment_indices(src_indices, src->shape, src->ndim);
    }
    break;
  case DTYPE_USHORT:
    for (size_t i = 0; i < count; i++) {
      NUMC_USHORT *src_elem =
          (NUMC_USHORT *)array_get_unchecked(src, src_indices);
      NUMC_USHORT *dst_elem =
          (NUMC_USHORT *)array_get_unchecked(dst, src_indices);
      *dst_elem = *src_elem;
      increment_indices(src_indices, src->shape, src->ndim);
    }
    break;
  case DTYPE_INT:
    for (size_t i = 0; i < count; i++) {
      NUMC_INT *src_elem = (NUMC_INT *)array_get_unchecked(src, src_indices);
      NUMC_INT *dst_elem = (NUMC_INT *)array_get_unchecked(dst, src_indices);
      *dst_elem = *src_elem;
      increment_indices(src_indices, src->shape, src->ndim);
    }
    break;
  case DTYPE_UINT:
    for (size_t i = 0; i < count; i++) {
      NUMC_UINT *src_elem = (NUMC_UINT *)array_get_unchecked(src, src_indices);
      NUMC_UINT *dst_elem = (NUMC_UINT *)array_get_unchecked(dst, src_indices);
      *dst_elem = *src_elem;
      increment_indices(src_indices, src->shape, src->ndim);
    }
    break;
  case DTYPE_LONG:
    for (size_t i = 0; i < count; i++) {
      NUMC_LONG *src_elem = (NUMC_LONG *)array_get_unchecked(src, src_indices);
      NUMC_LONG *dst_elem = (NUMC_LONG *)array_get_unchecked(dst, src_indices);
      *dst_elem = *src_elem;
      increment_indices(src_indices, src->shape, src->ndim);
    }
    break;
  case DTYPE_ULONG:
    for (size_t i = 0; i < count; i++) {
      NUMC_ULONG *src_elem =
          (NUMC_ULONG *)array_get_unchecked(src, src_indices);
      NUMC_ULONG *dst_elem =
          (NUMC_ULONG *)array_get_unchecked(dst, src_indices);
      *dst_elem = *src_elem;
      increment_indices(src_indices, src->shape, src->ndim);
    }
    break;
  case DTYPE_FLOAT:
    for (size_t i = 0; i < count; i++) {
      NUMC_FLOAT *src_elem =
          (NUMC_FLOAT *)array_get_unchecked(src, src_indices);
      NUMC_FLOAT *dst_elem =
          (NUMC_FLOAT *)array_get_unchecked(dst, src_indices);
      *dst_elem = *src_elem;
      increment_indices(src_indices, src->shape, src->ndim);
    }
    break;
  case DTYPE_DOUBLE:
    for (size_t i = 0; i < count; i++) {
      NUMC_DOUBLE *src_elem =
          (NUMC_DOUBLE *)array_get_unchecked(src, src_indices);
      NUMC_DOUBLE *dst_elem =
          (NUMC_DOUBLE *)array_get_unchecked(dst, src_indices);
      *dst_elem = *src_elem;
      increment_indices(src_indices, src->shape, src->ndim);
    }
    break;
  }
}

static inline void array_fill_with(const Array *arr, const void *elem) {
  switch (arr->dtype) {
  case DTYPE_BYTE:
    // memset works for byte types with any value
    memset(arr->data, *((const NUMC_BYTE *)elem), arr->size);
    break;
  case DTYPE_UBYTE:
    memset(arr->data, *((const NUMC_UBYTE *)elem), arr->size);
    break;
  case DTYPE_SHORT: {
    // Multi-byte types need loops (memset only works per-byte)
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
}

// -----------------------------------------------------------------------------
//                              Public Functions
// -----------------------------------------------------------------------------

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

void array_free(Array *array) {
  if (!array)
    return;

  if (array->owns_data)
    numc_free(array->data);

  free(array->shape);
  free(array);
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
  size_t dst_indices_buf[MAX_STACK_NDIM] = {0};
  size_t *indices = (a->ndim <= MAX_STACK_NDIM)
                        ? indices_buf
                        : calloc(a->ndim, sizeof(size_t));
  size_t *dst_indices = (a->ndim <= MAX_STACK_NDIM)
                            ? dst_indices_buf
                            : calloc(a->ndim, sizeof(size_t));

  if (a->ndim > MAX_STACK_NDIM && (!indices || !dst_indices))
    goto fail;

  array_strided_copy(a, indices, result, a->size);

  for (size_t i = 0; i < a->ndim; i++)
    indices[i] = 0;

  array_strided_copy_with_offset(b, indices, result, dst_indices, axis,
                                 a->shape[axis], b->size);

  if (indices && a->ndim > MAX_STACK_NDIM)
    free(indices);
  if (dst_indices && a->ndim > MAX_STACK_NDIM)
    free(dst_indices);

  return result;

fail:
  if (indices && a->ndim > MAX_STACK_NDIM)
    free(indices);
  if (dst_indices && a->ndim > MAX_STACK_NDIM)
    free(dst_indices);
  array_free(result);
  return NULL;
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

  switch (arr->dtype) {
  case DTYPE_BYTE:
    memset(arr->data, 1, arr->size);
    break;
  case DTYPE_UBYTE:
    memset(arr->data, 1, arr->size);
    break;
  case DTYPE_SHORT: {
    // Multi-byte types need loops (memset only works per-byte)
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
      data[i] = 1.0f;
    break;
  }
  case DTYPE_DOUBLE: {
    NUMC_DOUBLE *data = (NUMC_DOUBLE *)arr->data;
    for (size_t i = 0; i < arr->size; i++)
      data[i] = 1.0;
    break;
  }
  }

  return arr;
}

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

Array *array_add(const Array *a, const Array *b) {
  if (!a || !b)
    return NULL;

  if (!array_is_contiguous(a)) {
    fprintf(stderr, "[ERROR] array_add: array a is not contiguous, "
                    "use array_to_contiguous() first\n");
    abort();
  }

  if (!array_is_contiguous(b)) {
    fprintf(stderr, "[ERROR] array_add: array b is not contiguous, "
                    "use array_to_contiguous() first\n");
    abort();
  }

  if (a->dtype != b->dtype)
    return NULL;

  if (a->ndim != b->ndim)
    return NULL;

  for (size_t i = 0; i < a->ndim; i++) {
    if (a->shape[i] != b->shape[i])
      return NULL;
  }

  Array *result = array_create(a->ndim, a->shape, a->dtype, NULL);
  if (!result)
    return NULL;

  add_funcs[a->dtype](a->data, b->data, result->data, a->size);
  return result;
}
