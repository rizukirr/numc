/**
 * @file compare_avx2.h
 * @brief AVX2 uint8 comparison kernels using XOR-0x80 trick.
 *
 * AVX2 only provides _mm256_cmpgt_epi8 (signed compare). To compare
 * unsigned bytes, XOR both operands with 0x80 to map [0,255] to
 * [-128,127], then use the signed compare.
 */
#ifndef NUMC_COMPARE_AVX2_H
#define NUMC_COMPARE_AVX2_H

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

static inline void _cmp_eq_u8_avx2(const uint8_t *restrict a,
                                   const uint8_t *restrict b,
                                   uint8_t *restrict out, size_t n) {
  const __m256i one = _mm256_set1_epi8(1);
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
    __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
    __m256i r = _mm256_cmpeq_epi8(va, vb);
    /* cmpeq returns 0xFF for true, mask to 1 */
    r = _mm256_and_si256(r, one);
    _mm256_storeu_si256((__m256i *)(out + i), r);
  }
  for (; i < n; i++)
    out[i] = a[i] == b[i] ? 1 : 0;
}

static inline void _cmp_gt_u8_avx2(const uint8_t *restrict a,
                                   const uint8_t *restrict b,
                                   uint8_t *restrict out, size_t n) {
  const __m256i bias = _mm256_set1_epi8((char)0x80);
  const __m256i one = _mm256_set1_epi8(1);
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    __m256i va =
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + i)), bias);
    __m256i vb =
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(b + i)), bias);
    __m256i r = _mm256_and_si256(_mm256_cmpgt_epi8(va, vb), one);
    _mm256_storeu_si256((__m256i *)(out + i), r);
  }
  for (; i < n; i++)
    out[i] = a[i] > b[i] ? 1 : 0;
}

/* lt(a,b) = gt(b,a) */
static inline void _cmp_lt_u8_avx2(const uint8_t *restrict a,
                                   const uint8_t *restrict b,
                                   uint8_t *restrict out, size_t n) {
  _cmp_gt_u8_avx2(b, a, out, n);
}

/* ge(a,b) = eq(a,b) | gt(a,b) */
static inline void _cmp_ge_u8_avx2(const uint8_t *restrict a,
                                   const uint8_t *restrict b,
                                   uint8_t *restrict out, size_t n) {
  const __m256i bias = _mm256_set1_epi8((char)0x80);
  const __m256i one = _mm256_set1_epi8(1);
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
    __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
    __m256i eq = _mm256_cmpeq_epi8(va, vb);
    __m256i gt = _mm256_cmpgt_epi8(_mm256_xor_si256(va, bias),
                                   _mm256_xor_si256(vb, bias));
    __m256i r = _mm256_and_si256(_mm256_or_si256(eq, gt), one);
    _mm256_storeu_si256((__m256i *)(out + i), r);
  }
  for (; i < n; i++)
    out[i] = a[i] >= b[i] ? 1 : 0;
}

/* le(a,b) = ge(b,a) */
static inline void _cmp_le_u8_avx2(const uint8_t *restrict a,
                                   const uint8_t *restrict b,
                                   uint8_t *restrict out, size_t n) {
  _cmp_ge_u8_avx2(b, a, out, n);
}

#endif /* NUMC_COMPARE_AVX2_H */
