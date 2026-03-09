/**
 * @file fused.h
 * @brief Fused row-reduce kernel types for axis reduction fast paths.
 *
 * Defines function pointer types for row-reduce, argmin/argmax row-reduce,
 * and post-reduction division kernels used by sum/mean/argmin/argmax
 * when reducing along a single axis of a contiguous array.
 */
#ifndef NUMC_REDUCTION_FUSED_H
#define NUMC_REDUCTION_FUSED_H

#include <stddef.h>
#include <stdint.h>

typedef void (*NumcRowReduceKernel)(const char *restrict base,
                                    intptr_t row_stride, size_t nrows,
                                    char *restrict dst, size_t ncols);

typedef void (*NumcArgRowReduceKernel)(const char *restrict base,
                                       intptr_t row_stride, size_t nrows,
                                       char *restrict dst, size_t ncols);

typedef void (*NumcDivCountKernel)(char *data, size_t n, size_t count);

extern const NumcRowReduceKernel sum_fused_table[];

#endif
