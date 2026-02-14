#ifndef NUMC_ARRAY_CORE_H
#define NUMC_ARRAY_CORE_H

#include "array_dtype.h"
#include <stddef.h>

#define NUMC_MAX_DIMENSIONS 8
#define NUMC_MAX_MEMORY 8388608 // 8MB

#define array_slice(arr, ...) _array_slice((arr), &(NumcSlice){__VA_ARGS__})

/* X-macro: generates _assign_value_<DTYPE>(data, value) for each type.
 * Copies one scalar from `value` to `data`. Both pointers must be
 * properly aligned for the target type. */

#define GENERATE_VALUE_ASSIGNER(TYPE_ENUM, NUMC_TYPE)                          \
  static inline void _assign_value_##TYPE_ENUM(void *data,                     \
                                               const void *value) {            \
    *(NUMC_TYPE *)data = *(const NUMC_TYPE *)value;                            \
  }

GENERATE_NUMC_TYPES(GENERATE_VALUE_ASSIGNER)
#undef GENERATE_VALUE_ASSIGNER

/* Dispatch table: _assign_value[dtype](dest, src) copies one element. */
typedef void (*AssignValueFunc)(void *, const void *);
#define GENERATE_VALUE_ASSIGNER_ENTRY(TYPE_ENUM, NUMC_TYPE)                    \
  [TYPE_ENUM] = _assign_value_##TYPE_ENUM,

static const AssignValueFunc _assign_value[] = {
    GENERATE_NUMC_TYPES(GENERATE_VALUE_ASSIGNER_ENTRY)};

#undef GENERATE_VALUE_ASSIGNER_ENTRY

typedef struct NumcCtx NumcCtx;

typedef struct NumcArray NumcArray;

typedef struct {
  size_t axis, start, stop, step;
} NumcSlice;

/* Create a context. All arrays allocated from it are freed together
 * via array_free(). Returns NULL on failure. */
NumcCtx *array_create_ctx(void);

/* Create an uninitialized array with the given shape and dtype.
 * Returns NULL on failure. */
NumcArray *array_create(NumcCtx *ctx, const size_t *shape, size_t dim,
                        NumcDType dtype);

/* Check if the array is contiguous. */
bool is_contiguous(NumcArray *arr);

/* Convert the array to contiguous layout. Returns 0 on success, -1 on error. */
int array_as_contiguous(NumcArray *arr);

/* Copy raw bytes into the array's data buffer. `data` must have at least
 * array_size() * array_elem_size() bytes or the remainder will be filled with
 * the zeros otherwise truncated. */
void array_write_data(NumcArray *arr, const void *data);

/* Create an array with all elements set to zero. */
NumcArray *array_zeros(NumcCtx *ctx, const size_t *shape, size_t dim,
                       NumcDType dtype);

/* Create an array with all elements set to *value.
 * `value` must point to a scalar matching `dtype`. */
NumcArray *array_fill_with(NumcCtx *ctx, const size_t *shape, size_t dim,
                           NumcDType dtype, const void *value);

/* Deep-copy an array (same ctx, new data). Returns NULL on failure. */
NumcArray *array_copy(const NumcArray *arr);

/* Reshape in-place. Total element count must stay the same.
 * Returns 0 on success, -1 on error. */
int array_reshape_inplace(NumcArray *arr, const size_t *new_shape,
                          size_t new_dim);

/* Copy an array and reshape the copy. Returns NULL on failure. */
NumcArray *array_reshape_copy(const NumcArray *arr, const size_t *new_shape,
                              size_t new_dim);

/* Swap dimensions in-place. Returns 0 on success, -1 on error. */
int array_transpose_inplace(NumcArray *arr, const size_t *axes);

/* Swap dimensions and return new array. Returns NULL on failure. */
NumcArray *array_transpose_copy(const NumcArray *arr, const size_t *axes);

/* Slice a single axis. Returns a view (no data copy). Returns NULL on failure.
 * `start` is inclusive, `stop` is exclusive, `step` must be >= 1. */
NumcArray *_array_slice(const NumcArray *arr, NumcSlice *slice);

/* Total number of elements (product of all dimensions). */
size_t array_size(const NumcArray *arr);

/* Allocated capacity in elements. */
size_t array_capacity(const NumcArray *arr);

/* Size of one element in bytes. */
size_t array_elem_size(const NumcArray *arr);

/* Number of dimensions (rank). */
size_t array_dim(const NumcArray *arr);

/* Copy shape into `shape`. Buffer must hold array_dim() elements. */
void array_shape(const NumcArray *arr, size_t *shape);

/* Copy byte-strides into `strides`. Buffer must hold array_dim() elements. */
void array_strides(const NumcArray *arr, size_t *strides);

/* Element type of the array. */
NumcDType array_dtype(const NumcArray *arr);

/* Pointer to the raw data buffer. */
void *array_data(const NumcArray *arr);

/* Free the context and all arrays created from it. NULL-safe. */
void array_free(NumcCtx *ctx);

#endif
