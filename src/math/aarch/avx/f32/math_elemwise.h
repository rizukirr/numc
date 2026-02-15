#ifndef NUMC_MATH_AARCH_AVX2_F32_H
#define NUMC_MATH_AARCH_AVX2_F32_H

#include <stddef.h>

void array_add_f32_avx(float *a, float *b, float *out, size_t n);
void array_mul_f32_avx(float *a, float *b, float *out, size_t n);
void array_sub_f32_avx(float *a, float *b, float *out, size_t n);
void array_div_f32_avx(float *a, float *b, float *out, size_t n);

#endif
