/**
 * @file avx2.h
 * @brief AVX2-accelerated element-wise binary operations.
 *
 * All functions require 32-byte aligned, contiguous input/output buffers.
 * Uses aligned loads/stores (e.g. _mm256_load_ps, _mm256_store_si256) --
 * passing unaligned pointers is undefined behavior.
 *
 * OpenMP parallelization kicks in when n > NUMC_MAX_ELEMENT_LOOP.
 * Remaining elements beyond the SIMD-width boundary are handled by a
 * scalar tail loop.
 *
 * Division strategy varies by type width:
 *   - 8/16-bit signed:   promote to float via sign-extend (cvtepi*), divps
 *   - 8/16-bit unsigned: promote to float via zero-extend (cvtepu*), divps
 *   - 32-bit int:        promote to double (cvtepi32_pd), divpd
 *   - 32-bit uint:       promote to double via hi16/lo16 split, divpd
 *   - 32-bit float:      native divps
 *   - 64-bit double:     native divpd
 *   - 64-bit int/uint:   no SIMD div (no hardware support)
 *
 * @note Compile this translation unit with -mavx2.
 */

#ifndef NUMC_ARCH_AVX2_H
#define NUMC_ARCH_AVX2_H

#include <numc/dtype.h>
#include <stddef.h>

// clang-format off

// =============================================================================
//  Float (32-bit) — 8 elements per 256-bit register
// =============================================================================

/** @brief out[i] = a[i] + b[i] using vaddps. */
void adds_float_avx2(const NUMC_FLOAT *a, const NUMC_FLOAT *b, NUMC_FLOAT *out, size_t n);
/** @brief out[i] = a[i] * b[i] using vmulps. */
void muls_float_avx2(const NUMC_FLOAT *a, const NUMC_FLOAT *b, NUMC_FLOAT *out, size_t n);
/** @brief out[i] = a[i] - b[i] using vsubps. */
void subs_float_avx2(const NUMC_FLOAT *a, const NUMC_FLOAT *b, NUMC_FLOAT *out, size_t n);
/** @brief out[i] = a[i] / b[i] using vdivps. */
void div_float_avx2(const NUMC_FLOAT *a, const NUMC_FLOAT *b, NUMC_FLOAT *out, size_t n);

// =============================================================================
//  Double (64-bit) — 4 elements per 256-bit register
// =============================================================================

/** @brief out[i] = a[i] + b[i] using vaddpd. */
void adds_double_avx2(const NUMC_DOUBLE *a, const NUMC_DOUBLE *b, NUMC_DOUBLE *out, size_t n);
/** @brief out[i] = a[i] * b[i] using vmulpd. */
void muls_double_avx2(const NUMC_DOUBLE *a, const NUMC_DOUBLE *b, NUMC_DOUBLE *out, size_t n);
/** @brief out[i] = a[i] - b[i] using vsubpd. */
void subs_double_avx2(const NUMC_DOUBLE *a, const NUMC_DOUBLE *b, NUMC_DOUBLE *out, size_t n);
/** @brief out[i] = a[i] / b[i] using vdivpd. */
void div_double_avx2(const NUMC_DOUBLE *a, const NUMC_DOUBLE *b, NUMC_DOUBLE *out, size_t n);

// =============================================================================
//  Int32 (signed) — 8 elements per 256-bit register
// =============================================================================

/** @brief out[i] = a[i] + b[i] using vpaddd. */
void adds_int_avx2(const NUMC_INT *a, const NUMC_INT *b, NUMC_INT *out, size_t n);
/** @brief out[i] = a[i] * b[i] using vpmulld. */
void muls_int_avx2(const NUMC_INT *a, const NUMC_INT *b, NUMC_INT *out, size_t n);
/** @brief out[i] = a[i] - b[i] using vpsubd. */
void subs_int_avx2(const NUMC_INT *a, const NUMC_INT *b, NUMC_INT *out, size_t n);
/** @brief out[i] = a[i] / b[i] via int32→double promotion, vdivpd. */
void div_int_avx2(const NUMC_INT *a, const NUMC_INT *b, NUMC_INT *out, size_t n);

// =============================================================================
//  Int64 (signed) — 4 elements per 256-bit register
// =============================================================================

/** @brief out[i] = a[i] + b[i] using vpaddq. */
void adds_long_avx2(const NUMC_LONG *a, const NUMC_LONG *b, NUMC_LONG *out, size_t n);
/** @brief out[i] = a[i] * b[i] via partial-product emulation (no native mul). */
void muls_long_avx2(const NUMC_LONG *a, const NUMC_LONG *b, NUMC_LONG *out, size_t n);
/** @brief out[i] = a[i] - b[i] using vpsubq. */
void subs_long_avx2(const NUMC_LONG *a, const NUMC_LONG *b, NUMC_LONG *out, size_t n);

// =============================================================================
//  Int16 (signed) — 16 elements per 256-bit register
// =============================================================================

/** @brief out[i] = a[i] + b[i] using vpaddw. */
void adds_short_avx2(const NUMC_SHORT *a, const NUMC_SHORT *b, NUMC_SHORT *out, size_t n);
/** @brief out[i] = a[i] * b[i] using vpmullw. */
void muls_short_avx2(const NUMC_SHORT *a, const NUMC_SHORT *b, NUMC_SHORT *out, size_t n);
/** @brief out[i] = a[i] - b[i] using vpsubw. */
void subs_short_avx2(const NUMC_SHORT *a, const NUMC_SHORT *b, NUMC_SHORT *out, size_t n);
/** @brief out[i] = a[i] / b[i] via int16→float promotion (cvtepi16), vdivps. */
void div_short_avx2(const NUMC_SHORT *a, const NUMC_SHORT *b, NUMC_SHORT *out, size_t n);

// =============================================================================
//  Int8 (signed) — 32 elements per 256-bit register
// =============================================================================

/** @brief out[i] = a[i] + b[i] using vpaddb. */
void adds_byte_avx2(const NUMC_BYTE *a, const NUMC_BYTE *b, NUMC_BYTE *out, size_t n);
/** @brief out[i] = a[i] * b[i] via unpack→epi16 mul→signed pack. */
void muls_byte_avx2(const NUMC_BYTE *a, const NUMC_BYTE *b, NUMC_BYTE *out, size_t n);
/** @brief out[i] = a[i] - b[i] using vpsubb. */
void subs_byte_avx2(const NUMC_BYTE *a, const NUMC_BYTE *b, NUMC_BYTE *out, size_t n);
/** @brief out[i] = a[i] / b[i] via int8→float promotion (cvtepi8), vdivps. */
void div_byte_avx2(const NUMC_BYTE *a, const NUMC_BYTE *b, NUMC_BYTE *out, size_t n);

// =============================================================================
//  Uint32 — 8 elements per 256-bit register
// =============================================================================

/** @brief out[i] = a[i] + b[i] using vpaddd (same as signed). */
void adds_uint_avx2(const NUMC_UINT *a, const NUMC_UINT *b, NUMC_UINT *out, size_t n);
/** @brief out[i] = a[i] * b[i] using vpmulld (low 32 bits, same as signed). */
void muls_uint_avx2(const NUMC_UINT *a, const NUMC_UINT *b, NUMC_UINT *out, size_t n);
/** @brief out[i] = a[i] - b[i] using vpsubd (same as signed). */
void subs_uint_avx2(const NUMC_UINT *a, const NUMC_UINT *b, NUMC_UINT *out, size_t n);
/** @brief out[i] = a[i] / b[i] via uint32→double (hi16/lo16 split), vdivpd. */
void div_uint_avx2(const NUMC_UINT *a, const NUMC_UINT *b, NUMC_UINT *out, size_t n);

// =============================================================================
//  Uint64 — 4 elements per 256-bit register
// =============================================================================

/** @brief out[i] = a[i] + b[i] using vpaddq (same as signed). */
void adds_ulong_avx2(const NUMC_ULONG *a, const NUMC_ULONG *b, NUMC_ULONG *out, size_t n);
/** @brief out[i] = a[i] * b[i] via partial-product emulation (same as signed). */
void muls_ulong_avx2(const NUMC_ULONG *a, const NUMC_ULONG *b, NUMC_ULONG *out, size_t n);
/** @brief out[i] = a[i] - b[i] using vpsubq (same as signed). */
void subs_ulong_avx2(const NUMC_ULONG *a, const NUMC_ULONG *b, NUMC_ULONG *out, size_t n);

// =============================================================================
//  Uint16 — 16 elements per 256-bit register
// =============================================================================

/** @brief out[i] = a[i] + b[i] using vpaddw (same as signed). */
void adds_ushort_avx2(const NUMC_USHORT *a, const NUMC_USHORT *b, NUMC_USHORT *out, size_t n);
/** @brief out[i] = a[i] * b[i] using vpmullw (low 16 bits, same as signed). */
void muls_ushort_avx2(const NUMC_USHORT *a, const NUMC_USHORT *b, NUMC_USHORT *out, size_t n);
/** @brief out[i] = a[i] - b[i] using vpsubw (same as signed). */
void subs_ushort_avx2(const NUMC_USHORT *a, const NUMC_USHORT *b, NUMC_USHORT *out, size_t n);
/** @brief out[i] = a[i] / b[i] via uint16→float promotion (cvtepu16), vdivps. */
void div_ushort_avx2(const NUMC_USHORT *a, const NUMC_USHORT *b, NUMC_USHORT *out, size_t n);

// =============================================================================
//  Uint8 — 32 elements per 256-bit register
// =============================================================================

/** @brief out[i] = a[i] + b[i] using vpaddb (same as signed). */
void adds_ubyte_avx2(const NUMC_UBYTE *a, const NUMC_UBYTE *b, NUMC_UBYTE *out, size_t n);
/** @brief out[i] = a[i] * b[i] via unpack→epi16 mul→unsigned pack. */
void muls_ubyte_avx2(const NUMC_UBYTE *a, const NUMC_UBYTE *b, NUMC_UBYTE *out, size_t n);
/** @brief out[i] = a[i] - b[i] using vpsubb (same as signed). */
void subs_ubyte_avx2(const NUMC_UBYTE *a, const NUMC_UBYTE *b, NUMC_UBYTE *out, size_t n);
/** @brief out[i] = a[i] / b[i] via uint8→float promotion (cvtepu8), vdivps. */
void div_ubyte_avx2(const NUMC_UBYTE *a, const NUMC_UBYTE *b, NUMC_UBYTE *out, size_t n);


// =============================================================================
//  Scalar Operations — out[i] = a[i] OP scalar
// =============================================================================

// Float (32-bit) — 8 elements per register
void adds_float_scalar(const NUMC_FLOAT *a, NUMC_FLOAT scalar, NUMC_FLOAT *out, size_t n);
void muls_float_scalar(const NUMC_FLOAT *a, NUMC_FLOAT scalar, NUMC_FLOAT *out, size_t n);
void subs_float_scalar(const NUMC_FLOAT *a, NUMC_FLOAT scalar, NUMC_FLOAT *out, size_t n);
void div_float_scalar(const NUMC_FLOAT *a, NUMC_FLOAT scalar, NUMC_FLOAT *out, size_t n);

// Double (64-bit) — 4 elements per register
void adds_double_scalar(const NUMC_DOUBLE *a, NUMC_DOUBLE scalar, NUMC_DOUBLE *out, size_t n);
void muls_double_scalar(const NUMC_DOUBLE *a, NUMC_DOUBLE scalar, NUMC_DOUBLE *out, size_t n);
void subs_double_scalar(const NUMC_DOUBLE *a, NUMC_DOUBLE scalar, NUMC_DOUBLE *out, size_t n);
void div_double_scalar(const NUMC_DOUBLE *a, NUMC_DOUBLE scalar, NUMC_DOUBLE *out, size_t n);

// Int32 (signed) — 8 elements per register
void adds_int_scalar(const NUMC_INT *a, NUMC_INT scalar, NUMC_INT *out, size_t n);
void muls_int_scalar(const NUMC_INT *a, NUMC_INT scalar, NUMC_INT *out, size_t n);
void subs_int_scalar(const NUMC_INT *a, NUMC_INT scalar, NUMC_INT *out, size_t n);
void div_int_scalar(const NUMC_INT *a, NUMC_INT scalar, NUMC_INT *out, size_t n);

// Int64 (signed) — 4 elements per register (no div: no SIMD 64-bit int div)
void adds_long_scalar(const NUMC_LONG *a, NUMC_LONG scalar, NUMC_LONG *out, size_t n);
void muls_long_scalar(const NUMC_LONG *a, NUMC_LONG scalar, NUMC_LONG *out, size_t n);
void subs_long_scalar(const NUMC_LONG *a, NUMC_LONG scalar, NUMC_LONG *out, size_t n);

// Int16 (signed) — 16 elements per register
void adds_short_scalar(const NUMC_SHORT *a, NUMC_SHORT scalar, NUMC_SHORT *out, size_t n);
void muls_short_scalar(const NUMC_SHORT *a, NUMC_SHORT scalar, NUMC_SHORT *out, size_t n);
void subs_short_scalar(const NUMC_SHORT *a, NUMC_SHORT scalar, NUMC_SHORT *out, size_t n);
void div_short_scalar(const NUMC_SHORT *a, NUMC_SHORT scalar, NUMC_SHORT *out, size_t n);

// Int8 (signed) — 32 elements per register
void adds_byte_scalar(const NUMC_BYTE *a, NUMC_BYTE scalar, NUMC_BYTE *out, size_t n);
void muls_byte_scalar(const NUMC_BYTE *a, NUMC_BYTE scalar, NUMC_BYTE *out, size_t n);
void subs_byte_scalar(const NUMC_BYTE *a, NUMC_BYTE scalar, NUMC_BYTE *out, size_t n);
void div_byte_scalar(const NUMC_BYTE *a, NUMC_BYTE scalar, NUMC_BYTE *out, size_t n);

// Uint32 — 8 elements per register
void adds_uint_scalar(const NUMC_UINT *a, NUMC_UINT scalar, NUMC_UINT *out, size_t n);
void muls_uint_scalar(const NUMC_UINT *a, NUMC_UINT scalar, NUMC_UINT *out, size_t n);
void subs_uint_scalar(const NUMC_UINT *a, NUMC_UINT scalar, NUMC_UINT *out, size_t n);
void div_uint_scalar(const NUMC_UINT *a, NUMC_UINT scalar, NUMC_UINT *out, size_t n);

// Uint64 — 4 elements per register (no div: no SIMD 64-bit int div)
void adds_ulong_scalar(const NUMC_ULONG *a, NUMC_ULONG scalar, NUMC_ULONG *out, size_t n);
void muls_ulong_scalar(const NUMC_ULONG *a, NUMC_ULONG scalar, NUMC_ULONG *out, size_t n);
void subs_ulong_scalar(const NUMC_ULONG *a, NUMC_ULONG scalar, NUMC_ULONG *out, size_t n);

// Uint16 — 16 elements per register
void adds_ushort_scalar(const NUMC_USHORT *a, NUMC_USHORT scalar, NUMC_USHORT *out, size_t n);
void muls_ushort_scalar(const NUMC_USHORT *a, NUMC_USHORT scalar, NUMC_USHORT *out, size_t n);
void subs_ushort_scalar(const NUMC_USHORT *a, NUMC_USHORT scalar, NUMC_USHORT *out, size_t n);
void div_ushort_scalar(const NUMC_USHORT *a, NUMC_USHORT scalar, NUMC_USHORT *out, size_t n);

// Uint8 — 32 elements per register
void adds_ubyte_scalar(const NUMC_UBYTE *a, NUMC_UBYTE scalar, NUMC_UBYTE *out, size_t n);
void muls_ubyte_scalar(const NUMC_UBYTE *a, NUMC_UBYTE scalar, NUMC_UBYTE *out, size_t n);
void subs_ubyte_scalar(const NUMC_UBYTE *a, NUMC_UBYTE scalar, NUMC_UBYTE *out, size_t n);
void div_ubyte_scalar(const NUMC_UBYTE *a, NUMC_UBYTE scalar, NUMC_UBYTE *out, size_t n);

// clang-format on

#endif
