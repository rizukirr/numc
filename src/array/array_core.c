#define ARENA_IMPLEMENTATION
#include "array/array_core.h"
#include "array/array_dtype.h"
#include "memory/arena.h"
#include "memory/numc_alloc.h"
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Static functions

static inline void _calculate_strides(size_t *strides, const size_t *shape,
                                      size_t dim, size_t elem_size) {
  strides[dim - 1] = elem_size;
  for (size_t i = dim - 1; i > 0; i--) {
    strides[i - 1] = strides[i] * shape[i];
  }
}

static inline void _array_fill_with(NumcArray *arr, const NumcDType dtype,
                                    const void *value) {
  char *ptr = arr->data;
  size_t cordinate[arr->dim];
  memset(cordinate, 0, sizeof(cordinate));

  while (true) {
    _assign_value[dtype](ptr, value);

    int d = arr->dim - 1;
    ptr += arr->strides[d];
    cordinate[d]++;

    if (cordinate[d] < arr->shape[d])
      continue;

    while (d >= 0) {
      cordinate[d] = 0;

      if (d == 0)
        return;

      d--;

      cordinate[d]++;

      ptr += arr->strides[d] - arr->shape[d + 1] * arr->strides[d + 1];

      if (cordinate[d] < arr->shape[d])
        break;
    }
  }
}

// End Of static functions

NumcCtx *array_create_ctx(void) {
  NumcCtx *ctx = numc_malloc(ARENA_ALIGNOF(NumcCtx), sizeof(NumcCtx));
  if (!ctx)
    return NULL;

  ctx->arena = arena_create(NUMC_MAX_MEMORY);
  if (!ctx->arena) {
    numc_free(ctx);
    return NULL;
  }

  return ctx;
}

NumcArray *array_create(NumcCtx *ctx, const size_t *shape, size_t dim,
                        NumcDType dtype) {
  if (!ctx || !shape || dim == 0) {
    return NULL;
  }

  size_t size = 1, capacity = 0, elem_size = numc_type_size[dtype];

  for (size_t i = 0; i < dim; i++) {
    size *= shape[i];
  }

  capacity = size * elem_size;

  NumcArray *arr =
      arena_alloc(ctx->arena, sizeof(NumcArray), ARENA_ALIGNOF(NumcArray));
  if (!arr)
    return NULL;

  arr->ctx = ctx;
  arr->data = arena_alloc(ctx->arena, capacity, NUMC_SIMD_ALIGN);
  if (!arr->data) {
    arena_free(ctx->arena);
    return NULL;
  }

  arr->use_stack = dim <= NUMC_MAX_DIMENSIONS;
  if (arr->use_stack) {
    arr->shape = arr->_shape_buff;
    arr->strides = arr->_strides_buff;
  } else {
    arr->shape = arena_alloc(ctx->arena, 2 * dim * sizeof(size_t),
                             ARENA_ALIGNOF(size_t));
    arr->strides = arr->shape + dim;

    if (!arr->shape) {
      arena_free(ctx->arena);
      return NULL;
    }
  }

  memcpy(arr->shape, shape, dim * sizeof(size_t));

  _calculate_strides(arr->strides, shape, dim, elem_size);
  arr->dim = dim;
  arr->is_contiguous = true;
  arr->elem_size = elem_size;
  arr->size = size;
  arr->capacity = capacity;
  arr->dtype = dtype;

  return arr;
}

bool is_contiguous(NumcArray *arr) {
  size_t expected = arr->elem_size;
  for (int i = arr->dim - 1; i >= 0; i--) {
    if (arr->strides[i] != expected)
      return false;
    expected *= arr->shape[i];
  }
  return true;
}

int array_as_contiguous(NumcArray *arr) {
  if (!arr)
    return -1;

  if (is_contiguous(arr))
    return 0;

  // Dimension collapse: merge adjacent contiguous dims
  size_t c_shape[NUMC_MAX_DIMENSIONS];
  size_t c_strides[NUMC_MAX_DIMENSIONS];
  c_shape[0] = arr->shape[0];
  c_strides[0] = arr->strides[0];
  size_t cdim = 1;

  for (size_t i = 1; i < arr->dim; i++) {
    if (c_strides[cdim - 1] == arr->strides[i] * arr->shape[i]) {
      c_shape[cdim - 1] *= arr->shape[i];
      c_strides[cdim - 1] = arr->strides[i];
    } else {
      c_shape[cdim] = arr->shape[i];
      c_strides[cdim] = arr->strides[i];
      cdim++;
    }
  }

  // Allocate new contiguous buffer from arena
  void *new_data =
      arena_alloc(arr->ctx->arena, arr->capacity, NUMC_SIMD_ALIGN);
  if (!new_data)
    return -1;

  char *dst = (char *)new_data;
  size_t coord[NUMC_MAX_DIMENSIONS] = {0};

  if (c_strides[cdim - 1] == arr->elem_size) {
    // Inner dim is contiguous → memcpy whole chunks
    size_t chunk = c_shape[cdim - 1] * arr->elem_size;
    size_t outer = arr->size / c_shape[cdim - 1];

    for (size_t i = 0; i < outer; i++) {
      char *src = (char *)arr->data;
      for (size_t d = 0; d < cdim - 1; d++)
        src += coord[d] * c_strides[d];

      memcpy(dst, src, chunk);
      dst += chunk;

      for (int d = (int)cdim - 2; d >= 0; d--) {
        if (++coord[d] < c_shape[d])
          break;
        coord[d] = 0;
      }
    }
  } else {
    // No contiguous inner dim → element-wise copy
    for (size_t i = 0; i < arr->size; i++) {
      char *src = (char *)arr->data;
      for (size_t d = 0; d < cdim; d++)
        src += coord[d] * c_strides[d];

      memcpy(dst, src, arr->elem_size);
      dst += arr->elem_size;

      for (int d = (int)cdim - 1; d >= 0; d--) {
        if (++coord[d] < c_shape[d])
          break;
        coord[d] = 0;
      }
    }
  }

  arr->data = new_data;
  _calculate_strides(arr->strides, arr->shape, arr->dim, arr->elem_size);
  arr->is_contiguous = true;
  return 0;
}

NumcArray *array_zeros(NumcCtx *ctx, const size_t *shape, size_t dim,
                       NumcDType dtype) {
  NumcArray *arr = array_create(ctx, shape, dim, dtype);
  if (!arr) {
    return NULL;
  }

  memset(arr->data, 0, arr->capacity);
  return arr;
}

NumcArray *array_fill_with(NumcCtx *ctx, const size_t *shape, size_t dim,
                           NumcDType dtype, const void *value) {
  NumcArray *arr = array_create(ctx, shape, dim, dtype);
  if (!arr)
    return NULL;

  _array_fill_with(arr, dtype, value);
  return arr;
}

void array_write_data(NumcArray *arr, const void *data) {
  if (!arr || !data)
    return;

  memset(arr->data, 0, arr->capacity);
  memcpy(arr->data, data, arr->capacity);
}

NumcArray *array_copy(const NumcArray *arr) {
  if (!arr)
    return NULL;

  NumcArray *copy = array_create(arr->ctx, arr->shape, arr->dim, arr->dtype);
  if (!copy)
    return NULL;

  memcpy(copy->data, arr->data, arr->capacity);
  copy->is_contiguous = arr->is_contiguous;
  return copy;
}

int array_reshape_inplace(NumcArray *arr, const size_t *new_shape,
                          size_t new_dim) {
  if (!arr || !new_shape || new_dim == 0)
    return -1;

  size_t size = 1;
  for (size_t i = 0; i < new_dim; i++) {
    size *= new_shape[i];
  }

  if (size != arr->size) {
    return -1;
  }

  arr->dim = new_dim;
  memcpy(arr->shape, new_shape, new_dim * sizeof(size_t));

  _calculate_strides(arr->strides, new_shape, new_dim, arr->elem_size);
  arr->is_contiguous = is_contiguous(arr);
  return 0;
}

NumcArray *array_reshape_copy(const NumcArray *arr, const size_t *new_shape,
                              size_t new_dim) {
  if (!arr || !new_shape || new_dim == 0)
    return NULL;

  NumcArray *copy = array_create(arr->ctx, arr->shape, arr->dim, arr->dtype);
  if (!copy)
    return NULL;

  memcpy(copy->data, arr->data, arr->capacity);

  if (array_reshape_inplace(copy, new_shape, new_dim) < 0) {
    return NULL;
  }

  return copy;
}

int array_transpose_inplace(NumcArray *arr, const size_t *axes) {
  size_t axes_buff[arr->dim];
  memcpy(axes_buff, axes, arr->dim * sizeof(size_t));

  uint8_t seen = 0;
  size_t shape_buff[arr->dim];
  size_t stride_buff[arr->dim];
  for (size_t i = 0; i < arr->dim; i++) {
    size_t ax = axes_buff[i];
    if (ax >= arr->dim)
      return -1;

    uint8_t bit = 1u << ax;
    if (seen & bit)
      return -1;

    seen |= bit;

    shape_buff[i] = arr->shape[ax];
    stride_buff[i] = arr->strides[ax];
  }

  memcpy(arr->shape, shape_buff, arr->dim * sizeof(size_t));
  memcpy(arr->strides, stride_buff, arr->dim * sizeof(size_t));
  arr->is_contiguous = is_contiguous(arr);
  return 0;
}

NumcArray *array_transpose_copy(const NumcArray *arr, const size_t *axes) {
  if (!arr || !axes)
    return NULL;

  NumcArray *copy = array_create(arr->ctx, arr->shape, arr->dim, arr->dtype);
  if (!copy)
    return NULL;

  memcpy(copy->data, arr->data, arr->capacity);

  if (array_transpose_inplace(copy, axes) < 0) {
    return NULL;
  }

  return copy;
}

NumcArray *_array_slice(const NumcArray *arr, NumcSlice *slice) {
  if (!arr || slice->axis >= arr->dim)
    return NULL;

  size_t dim_size = arr->shape[slice->axis];

  // Default: step=0 → 1, stop=0 → full extent
  if (slice->step == 0)
    slice->step = 1;
  if (slice->stop == 0 || slice->stop > dim_size)
    slice->stop = dim_size;
  if (slice->start >= dim_size)
    slice->start = dim_size - 1;
  if (slice->start >= slice->stop)
    return NULL;

  NumcArray *view =
      arena_alloc(arr->ctx->arena, sizeof(NumcArray), ARENA_ALIGNOF(NumcArray));
  if (!view)
    return NULL;

  view->ctx = arr->ctx;
  view->dim = arr->dim;
  view->elem_size = arr->elem_size;
  view->dtype = arr->dtype;
  view->is_contiguous = false;
  view->use_stack = arr->dim <= NUMC_MAX_DIMENSIONS;

  if (view->use_stack) {
    view->shape = view->_shape_buff;
    view->strides = view->_strides_buff;
  } else {
    view->shape = arena_alloc(arr->ctx->arena, 2 * arr->dim * sizeof(size_t),
                              ARENA_ALIGNOF(size_t));
    view->strides = view->shape + arr->dim;
    if (!view->shape)
      return NULL;
  }

  memcpy(view->shape, arr->shape, arr->dim * sizeof(size_t));
  memcpy(view->strides, arr->strides, arr->dim * sizeof(size_t));

  view->data = (char *)arr->data + slice->start * arr->strides[slice->axis];
  view->shape[slice->axis] =
      (slice->stop - slice->start + slice->step - 1) / slice->step;
  view->strides[slice->axis] = arr->strides[slice->axis] * slice->step;

  size_t size = 1;
  for (size_t d = 0; d < view->dim; d++)
    size *= view->shape[d];
  view->size = size;
  view->capacity = size * view->elem_size;

  return view;
}

size_t array_size(const NumcArray *arr) { return arr->size; }
size_t array_capacity(const NumcArray *arr) { return arr->capacity; }
size_t array_elem_size(const NumcArray *arr) { return arr->elem_size; }
size_t array_dim(const NumcArray *arr) { return arr->dim; }
void array_shape(const NumcArray *arr, size_t *shape) {
  memcpy(shape, arr->shape, arr->dim * sizeof(size_t));
}
void array_strides(const NumcArray *arr, size_t *strides) {
  memcpy(strides, arr->strides, arr->dim * sizeof(size_t));
}

NumcDType array_dtype(const NumcArray *arr) { return arr->dtype; }
void *array_data(const NumcArray *arr) { return arr->data; }

void array_free(NumcCtx *ctx) {
  if (!ctx)
    return;

  arena_free(ctx->arena);
  free(ctx);
}
