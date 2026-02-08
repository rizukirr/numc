#ifndef NUMC_ARCH_AVX2_H
#define NUMC_ARCH_AVX2_H

#include <numc/dtype.h>
#include <stddef.h>

// BINARY OPERATION

// clang-format off
void adds_float_avx2(const NUMC_FLOAT *a, const NUMC_FLOAT *b, NUMC_FLOAT *out, size_t n);
void muls_float_avx2(const NUMC_FLOAT *a, const NUMC_FLOAT *b, NUMC_FLOAT *out, size_t n);
void subs_float_avx2(const NUMC_FLOAT *a, const NUMC_FLOAT *b, NUMC_FLOAT *out, size_t n);
void div_float_avx2(const NUMC_FLOAT *a, const NUMC_FLOAT *b, NUMC_FLOAT *out, size_t n);

void adds_double_avx2(const NUMC_DOUBLE *a, const NUMC_DOUBLE *b, NUMC_DOUBLE *out, size_t n);
void muls_double_avx2(const NUMC_DOUBLE *a, const NUMC_DOUBLE *b, NUMC_DOUBLE *out, size_t n);
void subs_double_avx2(const NUMC_DOUBLE *a, const NUMC_DOUBLE *b, NUMC_DOUBLE *out, size_t n);
void div_double_avx2(const NUMC_DOUBLE *a, const NUMC_DOUBLE *b, NUMC_DOUBLE *out, size_t n);

void adds_int_avx2(const NUMC_INT *a, const NUMC_INT *b, NUMC_INT *out, size_t n);
void muls_int_avx2(const NUMC_INT *a, const NUMC_INT *b, NUMC_INT *out, size_t n);
void subs_int_avx2(const NUMC_INT *a, const NUMC_INT *b, NUMC_INT *out, size_t n);

void adds_long_avx2(const NUMC_LONG *a, const NUMC_LONG *b, NUMC_LONG *out, size_t n);
void muls_long_avx2(const NUMC_LONG *a, const NUMC_LONG *b, NUMC_LONG *out, size_t n);
void subs_long_avx2(const NUMC_LONG *a, const NUMC_LONG *b, NUMC_LONG *out, size_t n);

void adds_short_avx2(const NUMC_SHORT *a, const NUMC_SHORT *b, NUMC_SHORT *out, size_t n);
void muls_short_avx2(const NUMC_SHORT *a, const NUMC_SHORT *b, NUMC_SHORT *out, size_t n);
void subs_short_avx2(const NUMC_SHORT *a, const NUMC_SHORT *b, NUMC_SHORT *out, size_t n);


void adds_byte_avx2(const NUMC_BYTE *a, const NUMC_BYTE *b, NUMC_BYTE *out, size_t n);
void muls_byte_avx2(const NUMC_BYTE *a, const NUMC_BYTE *b, NUMC_BYTE *out, size_t n);
void subs_byte_avx2(const NUMC_BYTE *a, const NUMC_BYTE *b, NUMC_BYTE *out, size_t n);

void adds_uint_avx2(const NUMC_UINT *a, const NUMC_UINT *b, NUMC_UINT *out, size_t n);
void muls_uint_avx2(const NUMC_UINT *a, const NUMC_UINT *b, NUMC_UINT *out, size_t n);
void subs_uint_avx2(const NUMC_UINT *a, const NUMC_UINT *b, NUMC_UINT *out, size_t n);

void adds_ulong_avx2(const NUMC_ULONG *a, const NUMC_ULONG *b, NUMC_ULONG *out, size_t n);
void muls_ulong_avx2(const NUMC_ULONG *a, const NUMC_ULONG *b, NUMC_ULONG *out, size_t n);
void subs_ulong_avx2(const NUMC_ULONG *a, const NUMC_ULONG *b, NUMC_ULONG *out, size_t n);

void adds_ushort_avx2(const NUMC_USHORT *a, const NUMC_USHORT *b, NUMC_USHORT *out, size_t n);
void muls_ushort_avx2(const NUMC_USHORT *a, const NUMC_USHORT *b, NUMC_USHORT *out, size_t n);
void subs_ushort_avx2(const NUMC_USHORT *a, const NUMC_USHORT *b, NUMC_USHORT *out, size_t n);
// clang-format on

#endif
