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
