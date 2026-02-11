#define ARENA_IMPLEMENTATION
#include "array/array_core.h"
#include "memory/arena.h"
#include "memory/numc_alloc.h"
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

struct NumcCtx {
  Arena *arena;
};

struct NumcArray {
  NumcCtx *ctx;
  void *data;
  size_t *shape, *strides;
  size_t dim, elem_size, size, capacity;
  size_t _shape_buff[NUMC_MAX_DIMENSIONS], _strides_buff[NUMC_MAX_DIMENSIONS];
  bool _use_stack;
  NumcDType dtype;
};

// Static functions

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
  if (!ctx || !shape || dim == 0)
    return NULL;

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
  arr->data = arena_alloc(ctx->arena, capacity, numc_type_align[dtype]);
  if (!arr->data) {
    arena_free(ctx->arena);
    return NULL;
  }

  arr->_use_stack = dim <= NUMC_MAX_DIMENSIONS;
  if (arr->_use_stack) {
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

  arr->strides[dim - 1] = elem_size;
  for (size_t i = dim - 1; i > 0; i--) {
    arr->strides[i - 1] = arr->strides[i] * shape[i];
  }

  arr->dim = dim;
  arr->elem_size = elem_size;
  arr->size = size;
  arr->capacity = capacity;
  arr->dtype = dtype;

  return arr;
}

NumcArray *array_zeros(NumcCtx *ctx, const size_t *shape, size_t dim,
                       NumcDType dtype) {
  NumcArray *arr = array_create(ctx, shape, dim, dtype);
  if (!arr)
    return NULL;

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

  arr->strides[new_dim - 1] = arr->elem_size;
  for (size_t i = new_dim - 1; i > 0; i--) {
    arr->strides[i - 1] = arr->strides[i] * new_shape[i];
    if (arr->strides[i - 1] == 0) {
      return -1;
    }
  }

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
