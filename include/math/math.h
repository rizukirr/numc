#ifndef NUMC_MATH_H
#define NUMC_MATH_H

#include "array/_array_core.h"

// Element-wise math
int array_add(const NumcArray *a, const NumcArray *b, NumcArray *out);
int array_sub(const NumcArray *a, const NumcArray *b, NumcArray *out);
int array_mul(const NumcArray *a, const NumcArray *b, NumcArray *out);
int array_div(const NumcArray *a, const NumcArray *b, NumcArray *out);

#endif
