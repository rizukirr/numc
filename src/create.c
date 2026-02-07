#include "create.h"
#include "alloc.h"
#include "error.h"
#include "omp.h"
#include "types.h"

#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

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
// clang-format off
#define GEN_FILL_FUNC(numc_type, ctype)                                     \
  static inline void array_fill_##numc_type(Array *restrict arr,            \
                                            const void *elem) {             \
    const ctype value = *((const ctype *)elem);                             \
    ctype *restrict data = __builtin_assume_aligned(arr->data, NUMC_ALIGN); \
    const size_t n = arr->size;                                             \
    NUMC_OMP_FOR                                                               \
    for (size_t i = 0; i < n; i++) {                                        \
      data[i] = value;                                                      \
    }                                                                       \
  }
// clang-format on

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
#define GEN_ONE_CONSTANT(numc_type, ctype)                                     \
  static const ctype one_##numc_type = (ctype)1;
FOREACH_NUMC_TYPE(GEN_ONE_CONSTANT)
#undef GEN_ONE_CONSTANT

// Dispatch tables
typedef void (*fill_func)(Array *, const void *);

#define FILL_ENTRY(numc_type, ctype)                                           \
  [NUMC_TYPE_##numc_type] = array_fill_##numc_type,
static const fill_func fill_funcs[] = {FOREACH_NUMC_TYPE(FILL_ENTRY)};
#undef FILL_ENTRY

#define ONE_PTR_ENTRY(numc_type, ctype)                                        \
  [NUMC_TYPE_##numc_type] = &one_##numc_type,
static const void *const one_ptrs[] = {FOREACH_NUMC_TYPE(ONE_PTR_ENTRY)};
#undef ONE_PTR_ENTRY

#define GEN_ARRAY_ARRANGE(numc_type, ctype)                                    \
  static void array_arange_##numc_type(const int start, const int stop,        \
                                       const int step, void *restrict ret) {   \
    ctype *restrict arr = (ctype *)ret;                                        \
    const int n = (step > 0) ? ((stop - start + step - 1) / step)              \
                             : ((start - stop - step - 1) / (-step));          \
    NUMC_OMP_FOR                                                               \
    for (int i = 0; i < n; i++) {                                              \
      arr[i] = (ctype)(start + i * step);                                      \
    }                                                                          \
  }

FOREACH_NUMC_TYPE(GEN_ARRAY_ARRANGE)
#undef GEN_ARRAY_ARRANGE

typedef void (*array_arange_func)(const int start, const int stop,
                                  const int step, void *ret);

#define ARRAY_ARRANGE_ENTRY(numc_type, ctype)                                  \
  [NUMC_TYPE_##numc_type] = array_arange_##numc_type,
static const array_arange_func array_arange_funcs[] = {
    FOREACH_NUMC_TYPE(ARRAY_ARRANGE_ENTRY)};
#undef ARRAY_ARRANGE_ENTRY

#define GEN_ARRAY_LINSPACE(numc_type, ctype)                                   \
  static void array_linspace_##numc_type(                                      \
      const int start, const int stop, const size_t num, void *restrict ret) { \
    ctype *restrict arr = (ctype *)ret;                                        \
    if (num == 1) {                                                            \
      arr[0] = (ctype)start;                                                   \
      return;                                                                  \
    }                                                                          \
    const ctype step = (ctype)(stop - start) / (ctype)(num - 1);               \
    const size_t n = num;                                                      \
    NUMC_OMP_FOR                                                               \
    for (size_t i = 0; i < n; i++) {                                           \
      arr[i] = (ctype)start + (ctype)i * step;                                 \
    }                                                                          \
    arr[num - 1] = (ctype)stop;                                                \
  }

FOREACH_NUMC_TYPE(GEN_ARRAY_LINSPACE)
#undef GEN_ARRAY_LINSPACE

typedef void (*array_linspace_func)(const int start, const int stop,
                                    const size_t num, void *ret);

#define ARRAY_LINSPACE_ENTRY(numc_type, ctype)                                 \
  [NUMC_TYPE_##numc_type] = array_linspace_##numc_type,
static const array_linspace_func array_linspace_funcs[] = {
    FOREACH_NUMC_TYPE(ARRAY_LINSPACE_ENTRY)};
#undef ARRAY_LINSPACE_ENTRY

// =============================================================================
//                          Array Creation Functions
// =============================================================================

Array *array_empty(const ArrayCreate *src) {
  if (!src) {
    numc_set_error(NUMC_ERR_NULL,
                   "numc: array_empty: NULL argument, maybe you forgot to "
                   "initialize the array_create struct?");
    return NULL;
  }

  if (src->ndim == 0) {
    numc_set_error(NUMC_ERR_INVALID, "numc: array_empty: ndim must be > 0");
    return NULL;
  }

  if (!src->shape) {
    numc_set_error(NUMC_ERR_NULL, "numc: array_empty: NULL shape");
    return NULL;
  }

  size_t elem_size = numc_type_size(src->numc_type);

  Array *array = malloc(sizeof(Array));
  if (!array) {
    numc_set_error(NUMC_ERR_ALLOC,
                   "numc: array_empty: allocation failed for array struct");
    return NULL;
  }

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
      numc_set_error(NUMC_ERR_ALLOC,
                     "numc: array_empty: allocation failed for shape buffer");
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
      numc_set_error(NUMC_ERR_OVERFLOW,
                     "array creation: overflow in shape dimensions, shape "
                     "too large");
      return NULL;
    }
    array->size *= src->shape[i];
  }

  array->strides[src->ndim - 1] = elem_size;
  for (ssize_t i = src->ndim - 2; i >= 0; i--)
    array->strides[i] = array->strides[i + 1] * src->shape[i + 1];

  array->capacity = array->size;

  array->data = numc_malloc(NUMC_ALIGN, array->size * elem_size);

  if (!array->data) {
    if (!use_stack)
      free(array->shape);
    free(array);
    numc_set_error(NUMC_ERR_ALLOC,
                   "array creation: allocation failed for data buffer");
    return NULL;
  }

  return array;
}

Array *array_create(const ArrayCreate *src) {
  if (!src)
    return NULL;

  Array *array = array_empty(src);
  if (!array)
    return NULL;

  if (!src->owns_data) {
    numc_free(array->data);
    array->data = (void *)src->data;
    array->owns_data = false;
    return array;
  }

  if (src->data != NULL)
    memcpy(array->data, src->data, array->size * array->elem_size);
  else
    memset(array->data, 0, array->size * array->elem_size);

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
  Array *arr = array_empty(spec);
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

  Array *arr = array_empty(&src);
  if (!arr)
    return NULL;

  fill_funcs[arr->numc_type](arr, one_ptrs[arr->numc_type]);
  return arr;
}

Array *array_arange(const int start, const int stop, const int step,
                    const NUMC_TYPE type) {
  // Validate step
  if (step == 0) {
    numc_set_error(NUMC_ERR_INVALID, "numc: array_arange: step cannot be zero");
    return NULL;
  }

  // Validate range based on step direction
  if (step > 0 && start >= stop) {
    numc_set_error(NUMC_ERR_INVALID,
                   "numc: array_arange: start must be less than stop for "
                   "positive step");
    return NULL;
  }

  if (step < 0 && start <= stop) {
    numc_set_error(NUMC_ERR_INVALID,
                   "numc: array_arange: start must be greater than stop for "
                   "negative step");
    return NULL;
  }

  if (numc_type_is_unsigned(type) && (start < 0 || stop < 0)) {
    numc_set_error(
        NUMC_ERR_INVALID,
        "numc: array_arange: unsigned types must have positive step");
    return NULL;
  }

  // Calculate size using ceiling division for positive step
  size_t size;
  if (step > 0) {
    size = (size_t)((stop - start + step - 1) / step);
  } else {
    size = (size_t)((start - stop - step - 1) / (-step));
  }

  // Create array
  size_t shape[] = {size};
  ArrayCreate d = {
      .ndim = 1,
      .shape = shape,
      .numc_type = type,
      .data = NULL,
      .owns_data = true,
  };
  Array *arr = array_empty(&d);
  if (!arr)
    return NULL;

  // Fill array using type-specific function
  array_arange_funcs[arr->numc_type](start, stop, step, arr->data);
  return arr;
}

Array *array_linspace(const int start, const int stop, const size_t num,
                      const NUMC_TYPE type) {
  // Validate num
  if (num == 0) {
    numc_set_error(NUMC_ERR_INVALID, "numc: array_linspace: num must be > 0");
    return NULL;
  }

  if (numc_type_is_unsigned(type) && (start < 0 || stop < 0)) {
    numc_set_error(
        NUMC_ERR_INVALID,
        "numc: array_linspace: start and stop must be non-negative for unsigned types");
    return NULL;
  }

  // Create array
  size_t shape[] = {num};
  ArrayCreate d = {
      .ndim = 1,
      .shape = shape,
      .numc_type = type,
      .data = NULL,
      .owns_data = true,
  };
  Array *arr = array_empty(&d);
  if (!arr)
    return NULL;

  // Fill array using type-specific function
  array_linspace_funcs[arr->numc_type](start, stop, num, arr->data);
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
