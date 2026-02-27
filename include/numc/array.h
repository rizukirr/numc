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

/**
 * @brief Create a new context.
 *
 * All arrays allocated from this context are freed together via numc_ctx_free().
 *
 * @return Pointer to a newly created context, or NULL on failure.
 */
NumcCtx *numc_ctx_create(void);

/**
 * @brief Free the context and all arrays created from it.
 *
 * NULL-safe.
 *
 * @param ctx Pointer to the context to be freed.
 */
void numc_ctx_free(NumcCtx *ctx);

/* --- Array creation --- */

/**
 * @brief Create an uninitialized array with the given shape and dtype.
 *
 * @param ctx   Pointer to the context.
 * @param shape Array of dimensions.
 * @param dim   Number of dimensions.
 * @param dtype Data type of elements.
 * @return Pointer to the newly created array, or NULL on failure.
 */
NumcArray *numc_array_create(NumcCtx *ctx, const size_t *shape, size_t dim,
                             NumcDType dtype);

/**
 * @brief Create an array with all elements set to zero.
 *
 * @param ctx   Pointer to the context.
 * @param shape Array of dimensions.
 * @param dim   Number of dimensions.
 * @param dtype Data type of elements.
 * @return Pointer to the newly created array, or NULL on failure.
 */
NumcArray *numc_array_zeros(NumcCtx *ctx, const size_t *shape, size_t dim,
                            NumcDType dtype);

/**
 * @brief Create an array with all elements set to a specific value.
 *
 * @param ctx   Pointer to the context.
 * @param shape Array of dimensions.
 * @param dim   Number of dimensions.
 * @param dtype Data type of elements.
 * @param value Pointer to a scalar value matching `dtype`.
 * @return Pointer to the newly created array, or NULL on failure.
 */
NumcArray *numc_array_fill(NumcCtx *ctx, const size_t *shape, size_t dim,
                           NumcDType dtype, const void *value);

/**
 * @brief Deep-copy an array within the same context.
 *
 * @param arr Pointer to the array to be copied.
 * @return Pointer to the newly created copy, or NULL on failure.
 */
NumcArray *numc_array_copy(const NumcArray *arr);

/**
 * @brief Copy raw bytes into the array's data buffer.
 *
 * @param arr  Pointer to the array.
 * @param data Pointer to the source data buffer.
 */
void numc_array_write(NumcArray *arr, const void *data);

/* --- Shape manipulation --- */

/**
 * @brief Reshape an array in-place.
 *
 * Total element count must stay the same.
 *
 * @param arr       Pointer to the array.
 * @param new_shape Array of new dimensions.
 * @param new_dim   Number of new dimensions.
 * @return 0 on success, -1 on error.
 */
int numc_array_reshape(NumcArray *arr, const size_t *new_shape, size_t new_dim);

/**
 * @brief Create a copy of an array with a new shape.
 *
 * @param arr       Pointer to the array.
 * @param new_shape Array of new dimensions.
 * @param new_dim   Number of new dimensions.
 * @return Pointer to the newly created reshaped array, or NULL on failure.
 */
NumcArray *numc_array_reshape_copy(const NumcArray *arr,
                                   const size_t *new_shape, size_t new_dim);

/**
 * @brief Swap dimensions of an array in-place.
 *
 * @param arr  Pointer to the array.
 * @param axes Array of dimension indices in the new order.
 * @return 0 on success, -1 on error.
 */
int numc_array_transpose(NumcArray *arr, const size_t *axes);

/**
 * @brief Create a transposed copy of an array.
 *
 * @param arr  Pointer to the array.
 * @param axes Array of dimension indices in the new order.
 * @return Pointer to the newly created transposed array, or NULL on failure.
 */
NumcArray *numc_array_transpose_copy(const NumcArray *arr, const size_t *axes);

/**
 * @brief Slice a single axis of an array.
 *
 * Returns a view (no data copy). `start` is inclusive, `stop` is exclusive,
 * `step` must be >= 1.
 *
 * @param arr   Pointer to the array.
 * @param slice Pointer to the slice specification.
 * @return Pointer to the newly created array view, or NULL on failure.
 */
NumcArray *numc_array_slice(const NumcArray *arr, NumcSlice *slice);

/**
 * @brief Check if the array is contiguous in memory.
 *
 * @param arr Pointer to the array.
 * @return true if contiguous, false otherwise.
 */
bool numc_array_is_contiguous(NumcArray *arr);

/**
 * @brief Convert the array to a contiguous layout in-place.
 *
 * @param arr Pointer to the array.
 * @return 0 on success, -1 on error.
 */
int numc_array_contiguous(NumcArray *arr);

/* --- Properties --- */

/**
 * @brief Get the total number of elements in the array.
 *
 * @param arr Pointer to the array.
 * @return Product of all dimensions.
 */
size_t numc_array_size(const NumcArray *arr);

/**
 * @brief Get the allocated capacity of the array in elements.
 *
 * @param arr Pointer to the array.
 * @return Capacity in elements.
 */
size_t numc_array_capacity(const NumcArray *arr);

/**
 * @brief Get the size of one element in bytes.
 *
 * @param arr Pointer to the array.
 * @return Element size in bytes.
 */
size_t numc_array_elem_size(const NumcArray *arr);

/**
 * @brief Get the number of dimensions (rank) of the array.
 *
 * @param arr Pointer to the array.
 * @return Number of dimensions.
 */
size_t numc_array_ndim(const NumcArray *arr);

/**
 * @brief Copy the shape of the array into a buffer.
 *
 * Buffer must hold at least numc_array_ndim() elements.
 *
 * @param arr   Pointer to the array.
 * @param shape Pointer to the destination buffer.
 */
void numc_array_shape(const NumcArray *arr, size_t *shape);

/**
 * @brief Copy the byte-strides of the array into a buffer.
 *
 * Buffer must hold at least numc_array_ndim() elements.
 *
 * @param arr     Pointer to the array.
 * @param strides Pointer to the destination buffer.
 */
void numc_array_strides(const NumcArray *arr, size_t *strides);

/**
 * @brief Get the data type of the array elements.
 *
 * @param arr Pointer to the array.
 * @return Data type.
 */
NumcDType numc_array_dtype(const NumcArray *arr);

/**
 * @brief Get a pointer to the raw data buffer.
 *
 * @param arr Pointer to the array.
 * @return Pointer to raw data.
 */
void *numc_array_data(const NumcArray *arr);

/* --- Print --- */

/**
 * @brief Print the array contents to stdout.
 *
 * @param array Pointer to the array.
 */
void numc_array_print(const NumcArray *array);

#endif
