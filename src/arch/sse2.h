#ifndef NUMC_ARCH_SSE2_H
#define NUMC_ARCH_SSE2_H

#include <stddef.h>

// BINARY OPERATION

void adds_float_avx2(const float *a, const float *b, float *out, size_t n);
void muls_float_avx2(const float *a, const float *b, float *out, size_t n);
void subs_float_avx2(const float *a, const float *b, float *out, size_t n);

#endif
