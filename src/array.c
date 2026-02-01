#include "array.h"
#include "dtype.h"
#include "memory.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

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

// -----------------------------------------------------------------------------
// Type-specific addition kernels for contiguous arrays.
// These simple loops allow compiler auto-vectorization with -O3.
// -----------------------------------------------------------------------------

static inline void add_byte(const NUMC_BYTE *a, const NUMC_BYTE *b,
                            NUMC_BYTE *out, size_t n) {
  for (size_t i = 0; i < n; i++) {
    out[i] = a[i] + b[i];
  }
}

static inline void add_ubyte(const NUMC_UBYTE *a, const NUMC_UBYTE *b,
                             NUMC_UBYTE *out, size_t n) {
  for (size_t i = 0; i < n; i++) {
    out[i] = a[i] + b[i];
  }
}

static inline void add_short(const NUMC_SHORT *a, const NUMC_SHORT *b,
                             NUMC_SHORT *out, size_t n) {
  for (size_t i = 0; i < n; i++) {
    out[i] = a[i] + b[i];
  }
}

static inline void add_ushort(const NUMC_USHORT *a, const NUMC_USHORT *b,
                              NUMC_USHORT *out, size_t n) {
  for (size_t i = 0; i < n; i++) {
    out[i] = a[i] + b[i];
  }
}

static inline void add_int(const NUMC_INT *a, const NUMC_INT *b, NUMC_INT *out,
                           size_t n) {
  for (size_t i = 0; i < n; i++) {
    out[i] = a[i] + b[i];
  }
}

static inline void add_uint(const NUMC_UINT *a, const NUMC_UINT *b,
                            NUMC_UINT *out, size_t n) {
  for (size_t i = 0; i < n; i++) {
    out[i] = a[i] + b[i];
  }
}

static inline void add_long(const NUMC_LONG *a, const NUMC_LONG *b,
                            NUMC_LONG *out, size_t n) {
  for (size_t i = 0; i < n; i++) {
    out[i] = a[i] + b[i];
  }
}

static inline void add_ulong(const NUMC_ULONG *a, const NUMC_ULONG *b,
                             NUMC_ULONG *out, size_t n) {
  for (size_t i = 0; i < n; i++) {
    out[i] = a[i] + b[i];
  }
}

static inline void add_float(const NUMC_FLOAT *a, const NUMC_FLOAT *b,
                             NUMC_FLOAT *out, size_t n) {
  for (size_t i = 0; i < n; i++) {
    out[i] = a[i] + b[i];
  }
}

static inline void add_double(const NUMC_DOUBLE *a, const NUMC_DOUBLE *b,
                              NUMC_DOUBLE *out, size_t n) {
  for (size_t i = 0; i < n; i++) {
    out[i] = a[i] + b[i];
  }
}

/**
 * @brief Add element-wise for contiguous arrays (fast path).
 *
 * Switch outside loop enables compiler auto-vectorization.
 *
 * @param a Pointer to the first array.
 * @param b Pointer to the second array.
 * @param dest Pointer to the destination array.
 */
static inline void array_add_contiguous(const Array *a, const Array *b,
                                        Array *dest) {
  size_t n = a->size;

  switch (a->dtype) {
  case DTYPE_BYTE:
    add_byte(a->data, b->data, dest->data, n);
    break;
  case DTYPE_UBYTE:
    add_ubyte(a->data, b->data, dest->data, n);
    break;
  case DTYPE_SHORT:
    add_short(a->data, b->data, dest->data, n);
    break;
  case DTYPE_USHORT:
    add_ushort(a->data, b->data, dest->data, n);
    break;
  case DTYPE_INT:
    add_int(a->data, b->data, dest->data, n);
    break;
  case DTYPE_UINT:
    add_uint(a->data, b->data, dest->data, n);
    break;
  case DTYPE_LONG:
    add_long(a->data, b->data, dest->data, n);
    break;
  case DTYPE_ULONG:
    add_ulong(a->data, b->data, dest->data, n);
    break;
  case DTYPE_FLOAT:
    add_float(a->data, b->data, dest->data, n);
    break;
  case DTYPE_DOUBLE:
    add_double(a->data, b->data, dest->data, n);
    break;
  }
}

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

/**
 * @brief Add element-wise for non-contiguous arrays (slow path).
 *
 * Uses stride-based access for sliced/transposed arrays.
 *
 * @param a Pointer to the first array.
 * @param b Pointer to the second array.
 * @param dest Pointer to the destination array.
 */
static inline void array_add_uncontiguous(const Array *a, const Array *b,
                                          Array *dest) {
  size_t indices_buf[MAX_STACK_NDIM] = {0};
  size_t *indices = (a->ndim <= MAX_STACK_NDIM)
                        ? indices_buf
                        : calloc(a->ndim, sizeof(size_t));
  if (a->ndim > MAX_STACK_NDIM && !indices)
    return;

  switch (a->dtype) {
  case DTYPE_BYTE:
    for (size_t i = 0; i < a->size; i++) {
      NUMC_BYTE *a_elem = (NUMC_BYTE *)array_get_unchecked(a, indices);
      NUMC_BYTE *b_elem = (NUMC_BYTE *)array_get_unchecked(b, indices);
      ((NUMC_BYTE *)dest->data)[i] = *a_elem + *b_elem;
      increment_indices(indices, a->shape, a->ndim);
    }
    break;
  case DTYPE_UBYTE:
    for (size_t i = 0; i < a->size; i++) {
      NUMC_UBYTE *a_elem = (NUMC_UBYTE *)array_get_unchecked(a, indices);
      NUMC_UBYTE *b_elem = (NUMC_UBYTE *)array_get_unchecked(b, indices);
      ((NUMC_UBYTE *)dest->data)[i] = *a_elem + *b_elem;
      increment_indices(indices, a->shape, a->ndim);
    }
    break;
  case DTYPE_SHORT:
    for (size_t i = 0; i < a->size; i++) {
      NUMC_SHORT *a_elem = (NUMC_SHORT *)array_get_unchecked(a, indices);
      NUMC_SHORT *b_elem = (NUMC_SHORT *)array_get_unchecked(b, indices);
      ((NUMC_SHORT *)dest->data)[i] = *a_elem + *b_elem;
      increment_indices(indices, a->shape, a->ndim);
    }
    break;
  case DTYPE_USHORT:
    for (size_t i = 0; i < a->size; i++) {
      NUMC_USHORT *a_elem = (NUMC_USHORT *)array_get_unchecked(a, indices);
      NUMC_USHORT *b_elem = (NUMC_USHORT *)array_get_unchecked(b, indices);
      ((NUMC_USHORT *)dest->data)[i] = *a_elem + *b_elem;
      increment_indices(indices, a->shape, a->ndim);
    }
    break;
  case DTYPE_INT:
    for (size_t i = 0; i < a->size; i++) {
      NUMC_INT *a_elem = (NUMC_INT *)array_get_unchecked(a, indices);
      NUMC_INT *b_elem = (NUMC_INT *)array_get_unchecked(b, indices);
      ((NUMC_INT *)dest->data)[i] = *a_elem + *b_elem;
      increment_indices(indices, a->shape, a->ndim);
    }
    break;
  case DTYPE_UINT:
    for (size_t i = 0; i < a->size; i++) {
      NUMC_UINT *a_elem = (NUMC_UINT *)array_get_unchecked(a, indices);
      NUMC_UINT *b_elem = (NUMC_UINT *)array_get_unchecked(b, indices);
      ((NUMC_UINT *)dest->data)[i] = *a_elem + *b_elem;
      increment_indices(indices, a->shape, a->ndim);
    }
    break;
  case DTYPE_LONG:
    for (size_t i = 0; i < a->size; i++) {
      NUMC_LONG *a_elem = (NUMC_LONG *)array_get_unchecked(a, indices);
      NUMC_LONG *b_elem = (NUMC_LONG *)array_get_unchecked(b, indices);
      ((NUMC_LONG *)dest->data)[i] = *a_elem + *b_elem;
      increment_indices(indices, a->shape, a->ndim);
    }
    break;
  case DTYPE_ULONG:
    for (size_t i = 0; i < a->size; i++) {
      NUMC_ULONG *a_elem = (NUMC_ULONG *)array_get_unchecked(a, indices);
      NUMC_ULONG *b_elem = (NUMC_ULONG *)array_get_unchecked(b, indices);
      ((NUMC_ULONG *)dest->data)[i] = *a_elem + *b_elem;
      increment_indices(indices, a->shape, a->ndim);
    }
    break;
  case DTYPE_FLOAT:
    for (size_t i = 0; i < a->size; i++) {
      NUMC_FLOAT *a_elem = (NUMC_FLOAT *)array_get_unchecked(a, indices);
      NUMC_FLOAT *b_elem = (NUMC_FLOAT *)array_get_unchecked(b, indices);
      ((NUMC_FLOAT *)dest->data)[i] = *a_elem + *b_elem;
      increment_indices(indices, a->shape, a->ndim);
    }
    break;
  case DTYPE_DOUBLE:
    for (size_t i = 0; i < a->size; i++) {
      NUMC_DOUBLE *a_elem = (NUMC_DOUBLE *)array_get_unchecked(a, indices);
      NUMC_DOUBLE *b_elem = (NUMC_DOUBLE *)array_get_unchecked(b, indices);
      ((NUMC_DOUBLE *)dest->data)[i] = *a_elem + *b_elem;
      increment_indices(indices, a->shape, a->ndim);
    }
    break;
  }

  if (a->ndim > MAX_STACK_NDIM)
    free(indices);
}

static void array_fill_with(const Array *arr, const void *elem) {
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

  Array *dst = NULL;

  // Fast path: if source is contiguous, use memcpy
  if (array_is_contiguous(src)) {
    dst = array_create(src->ndim, src->shape, src->dtype, src->data);
    return dst;
  }

  // Slow path: element-by-element copy using unchecked access
  size_t indices_buf[MAX_STACK_NDIM] = {0};
  size_t *indices = (src->ndim <= MAX_STACK_NDIM)
                        ? indices_buf
                        : calloc(src->ndim, sizeof(size_t));
  if (src->ndim > MAX_STACK_NDIM && !indices)
    return NULL;

  dst = array_create(src->ndim, src->shape, src->dtype, NULL);
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

  // Fast path: bulk both arrays contiguous and concatenating along axis 0
  if (array_is_contiguous(a) && array_is_contiguous(b) && axis == 0) {
    memcpy(result->data, a->data, a->size * a->elem_size);
    memcpy((char *)result->data + (a->size * a->elem_size), b->data,
           b->size * b->elem_size);

    return result;
  }

  // Slow path: non-contiguous arrays or concatenating along non-zero axis
  // Need element-by-element copy with stride calculations
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
    // memset works for byte types with any value
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

  if (a->dtype != b->dtype)
    return NULL;

  if (a->ndim != b->ndim)
    return NULL;

  // Validate shapes match
  for (size_t i = 0; i < a->ndim; i++) {
    if (a->shape[i] != b->shape[i])
      return NULL;
  }

  Array *result = array_create(a->ndim, a->shape, a->dtype, NULL);
  if (!result)
    return NULL;

  if (array_is_contiguous(a) && array_is_contiguous(b)) {
    array_add_contiguous(a, b, result);
  } else {
    array_add_uncontiguous(a, b, result);
  }

  return result;
}
