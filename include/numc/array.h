#ifndef NUMC_ARRAY_H
#define NUMC_ARRAY_H

#include "numc/dtype.h"
#include <stdbool.h>
#include <stddef.h>

/* Opaque types — defined in src/internal.h */
typedef struct NumcCtx NumcCtx;
typedef struct NumcArray NumcArray;

typedef struct {
  size_t axis, start, stop, step;
} NumcSlice;

#define numc_slice(arr, ...) numc_array_slice((arr), &(NumcSlice){__VA_ARGS__})

/* --- Context --- */

/* Create a context. All arrays allocated from it are freed together
 * via numc_ctx_free(). Returns NULL on failure. */
NumcCtx *numc_ctx_create(void);

/* Free the context and all arrays created from it. NULL-safe. */
void numc_ctx_free(NumcCtx *ctx);

/* --- Array creation --- */

/* Create an uninitialized array with the given shape and dtype.
 * Returns NULL on failure. */
NumcArray *numc_array_create(NumcCtx *ctx, const size_t *shape, size_t dim,
                             NumcDType dtype);

/* Create an array with all elements set to zero. */
NumcArray *numc_array_zeros(NumcCtx *ctx, const size_t *shape, size_t dim,
                            NumcDType dtype);

/* Create an array with all elements set to *value.
 * `value` must point to a scalar matching `dtype`. */
NumcArray *numc_array_fill(NumcCtx *ctx, const size_t *shape, size_t dim,
                           NumcDType dtype, const void *value);

/* Deep-copy an array (same ctx, new data). Returns NULL on failure. */
NumcArray *numc_array_copy(const NumcArray *arr);

/* Copy raw bytes into the array's data buffer. */
void numc_array_write(NumcArray *arr, const void *data);

/* --- Shape manipulation --- */

/* Reshape in-place. Total element count must stay the same.
 * Returns 0 on success, -1 on error. */
int numc_array_reshape(NumcArray *arr, const size_t *new_shape, size_t new_dim);

/* Copy an array and reshape the copy. Returns NULL on failure. */
NumcArray *numc_array_reshape_copy(const NumcArray *arr,
                                   const size_t *new_shape, size_t new_dim);

/* Swap dimensions in-place. Returns 0 on success, -1 on error. */
int numc_array_transpose(NumcArray *arr, const size_t *axes);

/* Swap dimensions and return new array. Returns NULL on failure. */
NumcArray *numc_array_transpose_copy(const NumcArray *arr, const size_t *axes);

/* Slice a single axis. Returns a view (no data copy). Returns NULL on failure.
 * `start` is inclusive, `stop` is exclusive, `step` must be >= 1. */
NumcArray *numc_array_slice(const NumcArray *arr, NumcSlice *slice);

/* Check if the array is contiguous. */
bool numc_array_is_contiguous(NumcArray *arr);

/* Convert the array to contiguous layout. Returns 0 on success, -1 on error. */
int numc_array_contiguous(NumcArray *arr);

/* --- Properties --- */

/* Total number of elements (product of all dimensions). */
size_t numc_array_size(const NumcArray *arr);

/* Allocated capacity in elements. */
size_t numc_array_capacity(const NumcArray *arr);

/* Size of one element in bytes. */
size_t numc_array_elem_size(const NumcArray *arr);

/* Number of dimensions (rank). */
size_t numc_array_ndim(const NumcArray *arr);

/* Copy shape into `shape`. Buffer must hold numc_array_ndim() elements. */
void numc_array_shape(const NumcArray *arr, size_t *shape);

/* Copy byte-strides into `strides`. Buffer must hold numc_array_ndim()
 * elements. */
void numc_array_strides(const NumcArray *arr, size_t *strides);

/* Element type of the array. */
NumcDType numc_array_dtype(const NumcArray *arr);

/* Pointer to the raw data buffer. */
void *numc_array_data(const NumcArray *arr);

/* --- Print --- */

/* Print array contents to stdout. */
void numc_array_print(const NumcArray *array);

#endif
