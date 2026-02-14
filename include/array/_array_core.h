#ifndef NUMC__ARRAY_CORE_H
#define NUMC__ARRAY_CORE_H

#include "array/array_dtype.h"
#include "memory/arena.h"

#define NUMC_MAX_DIMENSIONS 8
#define NUMC_MAX_MEMORY 8388608 // 8MB

typedef struct {
  Arena *arena;
} NumcCtx;

typedef struct {
  NumcCtx *ctx;
  void *data;
  size_t *shape, *strides;
  size_t dim, elem_size, size, capacity;
  size_t _shape_buff[NUMC_MAX_DIMENSIONS], _strides_buff[NUMC_MAX_DIMENSIONS];
  bool use_stack, is_contiguous;
  NumcDType dtype;
} NumcArray;

#endif
