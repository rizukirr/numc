#include "../internal.h"
#include <immintrin.h>
#include <numc/dtype.h>
#include <stdint.h>

// multiplies 4x uint64_t lanes, returns low 64 bits
static inline __m256i _mm256_mul_epi64(__m256i a, __m256i b) {
  __m256i mask = _mm256_set1_epi64x(0xFFFFFFFF);

  // a_lo, b_lo
  __m256i a_lo = _mm256_and_si256(a, mask);
  __m256i b_lo = _mm256_and_si256(b, mask);

  // a_hi, b_hi
  __m256i a_hi = _mm256_srli_epi64(a, 32);
  __m256i b_hi = _mm256_srli_epi64(b, 32);

  // partial products
  __m256i p0 = _mm256_mul_epu32(a_lo, b_lo); // a_lo * b_lo
  __m256i p1 = _mm256_mul_epu32(a_lo, b_hi); // a_lo * b_hi
  __m256i p2 = _mm256_mul_epu32(a_hi, b_lo); // a_hi * b_lo

  // combine cross terms
  __m256i cross = _mm256_add_epi64(p1, p2);
  cross = _mm256_slli_epi64(cross, 32);

  // final low 64 bits
  return _mm256_add_epi64(p0, cross);
}

// FLOAT

void adds_float_avx2(const NUMC_FLOAT *a, const NUMC_FLOAT *b, NUMC_FLOAT *out,
                     size_t n) {
  size_t end = n & ~7ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 8) {
    __m256 load_a = _mm256_load_ps(&a[i]);
    __m256 load_b = _mm256_load_ps(&b[i]);
    __m256 add_ab = _mm256_add_ps(load_a, load_b);
    _mm256_store_ps(&out[i], add_ab);
  }

  for (size_t j = end; j < n; j++)
    out[j] = a[j] + b[j];
}

void muls_float_avx2(const NUMC_FLOAT *a, const NUMC_FLOAT *b, NUMC_FLOAT *out,
                     size_t n) {
  size_t end = n & ~7ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 8) {
    __m256 load_a = _mm256_load_ps(&a[i]);
    __m256 load_b = _mm256_load_ps(&b[i]);
    __m256 mul_ab = _mm256_mul_ps(load_a, load_b);
    _mm256_store_ps(&out[i], mul_ab);
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] * b[i];
}

void subs_float_avx2(const NUMC_FLOAT *a, const NUMC_FLOAT *b, NUMC_FLOAT *out,
                     size_t n) {
  size_t end = n & ~7ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 8) {
    __m256 load_a = _mm256_load_ps(&a[i]);
    __m256 load_b = _mm256_load_ps(&b[i]);
    __m256 sub_ab = _mm256_sub_ps(load_a, load_b);
    _mm256_store_ps(&out[i], sub_ab);
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] - b[i];
}

void div_float_avx2(const NUMC_FLOAT *a, const NUMC_FLOAT *b, NUMC_FLOAT *out,
                    size_t n) {
  size_t end = n & ~7ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 8) {
    __m256 load_a = _mm256_load_ps(&a[i]);
    __m256 load_b = _mm256_load_ps(&b[i]);
    __m256 div_ab = _mm256_div_ps(load_a, load_b);
    _mm256_store_ps(&out[i], div_ab);
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] / b[i];
}

// DOUBLE

void adds_double_avx2(const NUMC_DOUBLE *a, const NUMC_DOUBLE *b,
                      NUMC_DOUBLE *out, size_t n) {
  size_t end = n & ~3ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 4) {
    __m256d load_a = _mm256_load_pd(&a[i]);
    __m256d load_b = _mm256_load_pd(&b[i]);
    __m256d add_ab = _mm256_add_pd(load_a, load_b);
    _mm256_store_pd(&out[i], add_ab);
  }

  for (size_t j = end; j < n; j++)
    out[j] = a[j] + b[j];
}

void muls_double_avx2(const NUMC_DOUBLE *a, const NUMC_DOUBLE *b,
                      NUMC_DOUBLE *out, size_t n) {
  size_t end = n & ~3ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 4) {
    __m256d load_a = _mm256_load_pd(&a[i]);
    __m256d load_b = _mm256_load_pd(&b[i]);
    __m256d mul_ab = _mm256_mul_pd(load_a, load_b);
    _mm256_store_pd(&out[i], mul_ab);
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] * b[i];
}

void subs_double_avx2(const NUMC_DOUBLE *a, const NUMC_DOUBLE *b,
                      NUMC_DOUBLE *out, size_t n) {
  size_t end = n & ~3ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 4) {
    __m256d load_a = _mm256_load_pd(&a[i]);
    __m256d load_b = _mm256_load_pd(&b[i]);
    __m256d sub_ab = _mm256_sub_pd(load_a, load_b);
    _mm256_store_pd(&out[i], sub_ab);
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] - b[i];
}

void div_double_avx2(const NUMC_DOUBLE *a, const NUMC_DOUBLE *b,
                     NUMC_DOUBLE *out, size_t n) {
  size_t end = n & ~3ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 4) {
    __m256d load_a = _mm256_load_pd(&a[i]);
    __m256d load_b = _mm256_load_pd(&b[i]);
    __m256d div_ab = _mm256_div_pd(load_a, load_b);
    _mm256_store_pd(&out[i], div_ab);
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] / b[i];
}

// INT

void adds_int_avx2(const NUMC_INT *a, const NUMC_INT *b, NUMC_INT *out,
                   size_t n) {
  size_t end = n & ~7ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 8) {
    __m256i load_a = _mm256_load_si256((const __m256i *)&a[i]);
    __m256i load_b = _mm256_load_si256((const __m256i *)&b[i]);
    __m256i add_ab = _mm256_add_epi32(load_a, load_b);
    _mm256_store_si256((__m256i *)&out[i], add_ab);
  }

  for (size_t j = end; j < n; j++)
    out[j] = a[j] + b[j];
}

void muls_int_avx2(const NUMC_INT *a, const NUMC_INT *b, NUMC_INT *out,
                   size_t n) {
  size_t end = n & ~7ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 8) {
    __m256i load_a = _mm256_load_si256((const __m256i *)&a[i]);
    __m256i load_b = _mm256_load_si256((const __m256i *)&b[i]);
    __m256i mul_ab = _mm256_mullo_epi32(load_a, load_b);
    _mm256_store_si256((__m256i *)&out[i], mul_ab);
  }

  for (size_t j = end; j < n; j++)
    out[j] = a[j] * b[j];
}

void subs_int_avx2(const NUMC_INT *a, const NUMC_INT *b, NUMC_INT *out,
                   size_t n) {
  size_t end = n & ~7ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 8) {
    __m256i load_a = _mm256_load_si256((const __m256i *)&a[i]);
    __m256i load_b = _mm256_load_si256((const __m256i *)&b[i]);
    __m256i sub_ab = _mm256_sub_epi32(load_a, load_b);
    _mm256_store_si256((__m256i *)&out[i], sub_ab);
  }

  for (size_t j = end; j < n; j++)
    out[j] = a[j] - b[j];
}

void div_int_avx2(const NUMC_INT *a, const NUMC_INT *b, NUMC_INT *out,
                  size_t n) {
  size_t end = n & ~3ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 4) {
    __m128i load_a = _mm_load_si128((const __m128i *)&a[i]);
    __m128i load_b = _mm_load_si128((const __m128i *)&b[i]);

    __m256d a_lo = _mm256_cvtepi32_pd(load_a);
    __m256d b_lo = _mm256_cvtepi32_pd(load_b);

    __m256d div_ab = _mm256_div_pd(a_lo, b_lo);

    __m128i div_ab_128 = _mm256_cvtpd_epi32(div_ab);
    _mm_store_si128((__m128i *)&out[i], div_ab_128);
  }

  for (size_t i = end; i < n; i++)
    out[i] = (NUMC_INT)((NUMC_DOUBLE)a[i] / (NUMC_DOUBLE)b[i]);
}

// LONG

void adds_long_avx2(const NUMC_LONG *a, const NUMC_LONG *b, NUMC_LONG *out,
                    size_t n) {
  size_t end = n & ~3ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 4) {
    __m256i load_a = _mm256_load_si256((const __m256i *)&a[i]);
    __m256i load_b = _mm256_load_si256((const __m256i *)&b[i]);
    __m256i add_ab = _mm256_add_epi64(load_a, load_b);
    _mm256_store_si256((__m256i *)&out[i], add_ab);
  }

  for (size_t j = end; j < n; j++)
    out[j] = a[j] + b[j];
}

void muls_long_avx2(const NUMC_LONG *a, const NUMC_LONG *b, NUMC_LONG *out,
                    size_t n) {
  size_t end = n & ~3ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 4) {
    __m256i load_a = _mm256_load_si256((const __m256i *)&a[i]);
    __m256i load_b = _mm256_load_si256((const __m256i *)&b[i]);
    __m256i mul_ab = _mm256_mul_epi64(load_a, load_b);
    _mm256_store_si256((__m256i *)&out[i], mul_ab);
  }

  for (size_t j = end; j < n; j++)
    out[j] = a[j] * b[j];
}

void subs_long_avx2(const NUMC_LONG *a, const NUMC_LONG *b, NUMC_LONG *out,
                    size_t n) {
  size_t end = n & ~3ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 4) {
    __m256i load_a = _mm256_load_si256((const __m256i *)&a[i]);
    __m256i load_b = _mm256_load_si256((const __m256i *)&b[i]);
    __m256i sub_ab = _mm256_sub_epi64(load_a, load_b);
    _mm256_store_si256((__m256i *)&out[i], sub_ab);
  }

  for (size_t j = end; j < n; j++)
    out[j] = a[j] - b[j];
}

// SHORT

void adds_short_avx2(const NUMC_SHORT *a, const NUMC_SHORT *b, NUMC_SHORT *out,
                     size_t n) {
  size_t end = n & ~15ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 16) {
    __m256i load_a = _mm256_load_si256((const __m256i *)&a[i]);
    __m256i load_b = _mm256_load_si256((const __m256i *)&b[i]);
    __m256i add_ab = _mm256_add_epi16(load_a, load_b);
    _mm256_store_si256((__m256i *)&out[i], add_ab);
  }

  for (size_t j = end; j < n; j++)
    out[j] = a[j] + b[j];
}

void muls_short_avx2(const NUMC_SHORT *a, const NUMC_SHORT *b, NUMC_SHORT *out,
                     size_t n) {
  size_t end = n & ~15ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 16) {
    __m256i load_a = _mm256_load_si256((const __m256i *)&a[i]);
    __m256i load_b = _mm256_load_si256((const __m256i *)&b[i]);
    __m256i mul_ab = _mm256_mullo_epi16(load_a, load_b);
    _mm256_store_si256((__m256i *)&out[i], mul_ab);
  }

  for (size_t j = end; j < n; j++)
    out[j] = a[j] * b[j];
}

void subs_short_avx2(const NUMC_SHORT *a, const NUMC_SHORT *b, NUMC_SHORT *out,
                     size_t n) {
  size_t end = n & ~15ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 16) {
    __m256i load_a = _mm256_load_si256((const __m256i *)&a[i]);
    __m256i load_b = _mm256_load_si256((const __m256i *)&b[i]);
    __m256i sub_ab = _mm256_sub_epi16(load_a, load_b);
    _mm256_store_si256((__m256i *)&out[i], sub_ab);
  }

  for (size_t j = end; j < n; j++)
    out[j] = a[j] - b[j];
}

void div_short_avx2(const NUMC_SHORT *a, const NUMC_SHORT *b, NUMC_SHORT *out,
                    size_t n) {
  size_t end = n & ~7ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 8) {
    __m128i load_a = _mm_load_si128((const __m128i *)&a[i]);
    __m128i load_b = _mm_load_si128((const __m128i *)&b[i]);

    __m256i a_i32 = _mm256_cvtepi16_epi32(load_a);
    __m256i b_i32 = _mm256_cvtepi16_epi32(load_b);

    __m256 a_f32 = _mm256_cvtepi32_ps(a_i32);
    __m256 b_f32 = _mm256_cvtepi32_ps(b_i32);

    __m256 div_ab = _mm256_div_ps(a_f32, b_f32);

    __m256i res_i32 = _mm256_cvttps_epi32(div_ab);

    __m128i lo = _mm256_castsi256_si128(res_i32);
    __m128i hi = _mm256_extracti128_si256(res_i32, 1);
    __m128i res_i16 = _mm_packs_epi32(lo, hi);

    _mm_store_si128((__m128i *)&out[i], res_i16);
  }

  for (size_t j = end; j < n; j++)
    out[j] = (NUMC_SHORT)((NUMC_FLOAT)a[j] / (NUMC_FLOAT)b[j]);
}

// BYTE

void adds_byte_avx2(const NUMC_BYTE *a, const NUMC_BYTE *b, NUMC_BYTE *out,
                    size_t n) {
  size_t end = n & ~31ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 32) {
    __m256i load_a = _mm256_load_si256((const __m256i *)&a[i]);
    __m256i load_b = _mm256_load_si256((const __m256i *)&b[i]);
    __m256i add_ab = _mm256_add_epi8(load_a, load_b);
    _mm256_store_si256((__m256i *)&out[i], add_ab);
  }

  for (size_t j = end; j < n; j++)
    out[j] = a[j] + b[j];
}

void muls_byte_avx2(const NUMC_BYTE *a, const NUMC_BYTE *b, NUMC_BYTE *out,
                    size_t n) {
  size_t end = n & ~31ULL;
  __m256i zero = _mm256_setzero_si256();

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 32) {
    __m256i va = _mm256_load_si256((const __m256i *)&a[i]);
    __m256i vb = _mm256_load_si256((const __m256i *)&b[i]);

    __m256i a_lo = _mm256_unpacklo_epi8(va, zero);
    __m256i a_hi = _mm256_unpackhi_epi8(va, zero);
    __m256i b_lo = _mm256_unpacklo_epi8(vb, zero);
    __m256i b_hi = _mm256_unpackhi_epi8(vb, zero);

    __m256i mul_lo = _mm256_mullo_epi16(a_lo, b_lo);
    __m256i mul_hi = _mm256_mullo_epi16(a_hi, b_hi);

    __m256i packed = _mm256_packs_epi16(mul_lo, mul_hi);

    _mm256_store_si256((__m256i *)&out[i], packed);
  }

  for (size_t j = end; j < n; j++)
    out[j] = a[j] * b[j];
}

void subs_byte_avx2(const NUMC_BYTE *a, const NUMC_BYTE *b, NUMC_BYTE *out,
                    size_t n) {
  size_t end = n & ~31ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 32) {
    __m256i load_a = _mm256_load_si256((const __m256i *)&a[i]);
    __m256i load_b = _mm256_load_si256((const __m256i *)&b[i]);
    __m256i sub_ab = _mm256_sub_epi8(load_a, load_b);
    _mm256_store_si256((__m256i *)&out[i], sub_ab);
  }

  for (size_t j = end; j < n; j++)
    out[j] = a[j] - b[j];
}

void div_byte_avx2(const NUMC_BYTE *a, const NUMC_BYTE *b, NUMC_BYTE *out,
                   size_t n) {
  size_t end = n & ~7ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 8) {
    __m128i load_a = _mm_loadl_epi64((const __m128i *)&a[i]);
    __m128i load_b = _mm_loadl_epi64((const __m128i *)&b[i]);

    __m256i a_i32 = _mm256_cvtepi8_epi32(load_a);
    __m256i b_i32 = _mm256_cvtepi8_epi32(load_b);

    __m256 a_f32 = _mm256_cvtepi32_ps(a_i32);
    __m256 b_f32 = _mm256_cvtepi32_ps(b_i32);

    __m256 div_ab = _mm256_div_ps(a_f32, b_f32);

    __m256i res_i32 = _mm256_cvttps_epi32(div_ab);

    __m128i lo = _mm256_castsi256_si128(res_i32);
    __m128i hi = _mm256_extracti128_si256(res_i32, 1);
    __m128i res_i16 = _mm_packs_epi32(lo, hi);
    __m128i res_i8 = _mm_packs_epi16(res_i16, res_i16);

    _mm_storel_epi64((__m128i *)&out[i], res_i8);
  }

  for (size_t j = end; j < n; j++)
    out[j] = (NUMC_BYTE)((NUMC_FLOAT)a[j] / (NUMC_FLOAT)b[j]);
}

// UINT
// Add/sub/mul use the same epi32 instructions as signed INT.
// Division splits uint32 into hi16*65536+lo16 for correct unsigned→double
// conversion (AVX2 lacks a direct uint32→double intrinsic).

void adds_uint_avx2(const NUMC_UINT *a, const NUMC_UINT *b, NUMC_UINT *out,
                    size_t n) {
  size_t end = n & ~7ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 8) {
    __m256i load_a = _mm256_load_si256((const __m256i *)&a[i]);
    __m256i load_b = _mm256_load_si256((const __m256i *)&b[i]);
    __m256i add_ab = _mm256_add_epi32(load_a, load_b);
    _mm256_store_si256((__m256i *)&out[i], add_ab);
  }

  for (size_t j = end; j < n; j++)
    out[j] = a[j] + b[j];
}

void muls_uint_avx2(const NUMC_UINT *a, const NUMC_UINT *b, NUMC_UINT *out,
                    size_t n) {
  size_t end = n & ~7ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 8) {
    __m256i load_a = _mm256_load_si256((const __m256i *)&a[i]);
    __m256i load_b = _mm256_load_si256((const __m256i *)&b[i]);
    __m256i mul_ab = _mm256_mullo_epi32(load_a, load_b);
    _mm256_store_si256((__m256i *)&out[i], mul_ab);
  }

  for (size_t j = end; j < n; j++)
    out[j] = a[j] * b[j];
}

void subs_uint_avx2(const NUMC_UINT *a, const NUMC_UINT *b, NUMC_UINT *out,
                    size_t n) {
  size_t end = n & ~7ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 8) {
    __m256i load_a = _mm256_load_si256((const __m256i *)&a[i]);
    __m256i load_b = _mm256_load_si256((const __m256i *)&b[i]);
    __m256i sub_ab = _mm256_sub_epi32(load_a, load_b);
    _mm256_store_si256((__m256i *)&out[i], sub_ab);
  }

  for (size_t j = end; j < n; j++)
    out[j] = a[j] - b[j];
}

void div_uint_avx2(const NUMC_UINT *a, const NUMC_UINT *b, NUMC_UINT *out,
                   size_t n) {
  size_t end = n & ~3ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 4) {
    __m128i load_a = _mm_load_si128((const __m128i *)&a[i]);
    __m128i load_b = _mm_load_si128((const __m128i *)&b[i]);

    // uint32 → double: split into lo16 + hi16*65536
    // (AVX2 has no unsigned int32-to-double conversion)
    __m128i mask16 = _mm_set1_epi32(0xFFFF);

    __m256d a_d = _mm256_add_pd(
        _mm256_cvtepi32_pd(_mm_and_si128(load_a, mask16)),
        _mm256_mul_pd(_mm256_cvtepi32_pd(_mm_srli_epi32(load_a, 16)),
                      _mm256_set1_pd(65536.0)));

    __m256d b_d = _mm256_add_pd(
        _mm256_cvtepi32_pd(_mm_and_si128(load_b, mask16)),
        _mm256_mul_pd(_mm256_cvtepi32_pd(_mm_srli_epi32(load_b, 16)),
                      _mm256_set1_pd(65536.0)));

    __m256d div_ab = _mm256_div_pd(a_d, b_d);

    __m128i div_i32 = _mm256_cvttpd_epi32(div_ab);
    _mm_store_si128((__m128i *)&out[i], div_i32);
  }

  for (size_t i = end; i < n; i++)
    out[i] = (NUMC_UINT)((NUMC_DOUBLE)a[i] / (NUMC_DOUBLE)b[i]);
}

// ULONG
// Same epi64 instructions as signed LONG. No div (same as signed LONG —
// x86 has no SIMD 64-bit integer division and scalar fallback is used).

void adds_ulong_avx2(const NUMC_ULONG *a, const NUMC_ULONG *b, NUMC_ULONG *out,
                     size_t n) {
  size_t end = n & ~3ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 4) {
    __m256i load_a = _mm256_load_si256((const __m256i *)&a[i]);
    __m256i load_b = _mm256_load_si256((const __m256i *)&b[i]);
    __m256i add_ab = _mm256_add_epi64(load_a, load_b);
    _mm256_store_si256((__m256i *)&out[i], add_ab);
  }

  for (size_t j = end; j < n; j++)
    out[j] = a[j] + b[j];
}

void muls_ulong_avx2(const NUMC_ULONG *a, const NUMC_ULONG *b, NUMC_ULONG *out,
                     size_t n) {
  size_t end = n & ~3ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 4) {
    __m256i load_a = _mm256_load_si256((const __m256i *)&a[i]);
    __m256i load_b = _mm256_load_si256((const __m256i *)&b[i]);
    __m256i mul_ab = _mm256_mul_epi64(load_a, load_b);
    _mm256_store_si256((__m256i *)&out[i], mul_ab);
  }

  for (size_t j = end; j < n; j++)
    out[j] = a[j] * b[j];
}

void subs_ulong_avx2(const NUMC_ULONG *a, const NUMC_ULONG *b, NUMC_ULONG *out,
                     size_t n) {
  size_t end = n & ~3ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 4) {
    __m256i load_a = _mm256_load_si256((const __m256i *)&a[i]);
    __m256i load_b = _mm256_load_si256((const __m256i *)&b[i]);
    __m256i sub_ab = _mm256_sub_epi64(load_a, load_b);
    _mm256_store_si256((__m256i *)&out[i], sub_ab);
  }

  for (size_t j = end; j < n; j++)
    out[j] = a[j] - b[j];
}

// USHORT
// Add/sub/mul use the same epi16 instructions as signed SHORT.
// Division uses cvtepu16_epi32 (zero-extend) and packus_epi32 (unsigned
// saturation) instead of their signed counterparts.

void adds_ushort_avx2(const NUMC_USHORT *a, const NUMC_USHORT *b,
                      NUMC_USHORT *out, size_t n) {
  size_t end = n & ~15ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 16) {
    __m256i load_a = _mm256_load_si256((const __m256i *)&a[i]);
    __m256i load_b = _mm256_load_si256((const __m256i *)&b[i]);
    __m256i add_ab = _mm256_add_epi16(load_a, load_b);
    _mm256_store_si256((__m256i *)&out[i], add_ab);
  }

  for (size_t j = end; j < n; j++)
    out[j] = a[j] + b[j];
}

void muls_ushort_avx2(const NUMC_USHORT *a, const NUMC_USHORT *b,
                      NUMC_USHORT *out, size_t n) {
  size_t end = n & ~15ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 16) {
    __m256i load_a = _mm256_load_si256((const __m256i *)&a[i]);
    __m256i load_b = _mm256_load_si256((const __m256i *)&b[i]);
    __m256i mul_ab = _mm256_mullo_epi16(load_a, load_b);
    _mm256_store_si256((__m256i *)&out[i], mul_ab);
  }

  for (size_t j = end; j < n; j++)
    out[j] = a[j] * b[j];
}

void subs_ushort_avx2(const NUMC_USHORT *a, const NUMC_USHORT *b,
                      NUMC_USHORT *out, size_t n) {
  size_t end = n & ~15ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 16) {
    __m256i load_a = _mm256_load_si256((const __m256i *)&a[i]);
    __m256i load_b = _mm256_load_si256((const __m256i *)&b[i]);
    __m256i sub_ab = _mm256_sub_epi16(load_a, load_b);
    _mm256_store_si256((__m256i *)&out[i], sub_ab);
  }

  for (size_t j = end; j < n; j++)
    out[j] = a[j] - b[j];
}

void div_ushort_avx2(const NUMC_USHORT *a, const NUMC_USHORT *b,
                     NUMC_USHORT *out, size_t n) {
  size_t end = n & ~7ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 8) {
    __m128i load_a = _mm_load_si128((const __m128i *)&a[i]);
    __m128i load_b = _mm_load_si128((const __m128i *)&b[i]);

    __m256i a_i32 = _mm256_cvtepu16_epi32(load_a);
    __m256i b_i32 = _mm256_cvtepu16_epi32(load_b);

    __m256 a_f32 = _mm256_cvtepi32_ps(a_i32);
    __m256 b_f32 = _mm256_cvtepi32_ps(b_i32);

    __m256 div_ab = _mm256_div_ps(a_f32, b_f32);

    __m256i res_i32 = _mm256_cvttps_epi32(div_ab);

    __m128i lo = _mm256_castsi256_si128(res_i32);
    __m128i hi = _mm256_extracti128_si256(res_i32, 1);
    __m128i res_i16 = _mm_packus_epi32(lo, hi);

    _mm_store_si128((__m128i *)&out[i], res_i16);
  }

  for (size_t j = end; j < n; j++)
    out[j] = (NUMC_USHORT)((NUMC_FLOAT)a[j] / (NUMC_FLOAT)b[j]);
}

// UBYTE
// Add/sub use the same epi8 instructions as signed BYTE.
// Mul uses packus_epi16 (unsigned saturation) instead of packs_epi16.
// Division uses cvtepu8_epi32 (zero-extend) and packus variants.

void adds_ubyte_avx2(const NUMC_UBYTE *a, const NUMC_UBYTE *b, NUMC_UBYTE *out,
                     size_t n) {
  size_t end = n & ~31ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 32) {
    __m256i load_a = _mm256_load_si256((const __m256i *)&a[i]);
    __m256i load_b = _mm256_load_si256((const __m256i *)&b[i]);
    __m256i add_ab = _mm256_add_epi8(load_a, load_b);
    _mm256_store_si256((__m256i *)&out[i], add_ab);
  }

  for (size_t j = end; j < n; j++)
    out[j] = a[j] + b[j];
}

void muls_ubyte_avx2(const NUMC_UBYTE *a, const NUMC_UBYTE *b, NUMC_UBYTE *out,
                     size_t n) {
  size_t end = n & ~31ULL;
  __m256i zero = _mm256_setzero_si256();

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 32) {
    __m256i va = _mm256_load_si256((const __m256i *)&a[i]);
    __m256i vb = _mm256_load_si256((const __m256i *)&b[i]);

    __m256i a_lo = _mm256_unpacklo_epi8(va, zero);
    __m256i a_hi = _mm256_unpackhi_epi8(va, zero);
    __m256i b_lo = _mm256_unpacklo_epi8(vb, zero);
    __m256i b_hi = _mm256_unpackhi_epi8(vb, zero);

    __m256i mul_lo = _mm256_mullo_epi16(a_lo, b_lo);
    __m256i mul_hi = _mm256_mullo_epi16(a_hi, b_hi);

    __m256i packed = _mm256_packus_epi16(mul_lo, mul_hi);

    _mm256_store_si256((__m256i *)&out[i], packed);
  }

  for (size_t j = end; j < n; j++)
    out[j] = a[j] * b[j];
}

void subs_ubyte_avx2(const NUMC_UBYTE *a, const NUMC_UBYTE *b, NUMC_UBYTE *out,
                     size_t n) {
  size_t end = n & ~31ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 32) {
    __m256i load_a = _mm256_load_si256((const __m256i *)&a[i]);
    __m256i load_b = _mm256_load_si256((const __m256i *)&b[i]);
    __m256i sub_ab = _mm256_sub_epi8(load_a, load_b);
    _mm256_store_si256((__m256i *)&out[i], sub_ab);
  }

  for (size_t j = end; j < n; j++)
    out[j] = a[j] - b[j];
}

void div_ubyte_avx2(const NUMC_UBYTE *a, const NUMC_UBYTE *b, NUMC_UBYTE *out,
                    size_t n) {
  size_t end = n & ~7ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 8) {
    __m128i load_a = _mm_loadl_epi64((const __m128i *)&a[i]);
    __m128i load_b = _mm_loadl_epi64((const __m128i *)&b[i]);

    __m256i a_i32 = _mm256_cvtepu8_epi32(load_a);
    __m256i b_i32 = _mm256_cvtepu8_epi32(load_b);

    __m256 a_f32 = _mm256_cvtepi32_ps(a_i32);
    __m256 b_f32 = _mm256_cvtepi32_ps(b_i32);

    __m256 div_ab = _mm256_div_ps(a_f32, b_f32);

    __m256i res_i32 = _mm256_cvttps_epi32(div_ab);

    __m128i lo = _mm256_castsi256_si128(res_i32);
    __m128i hi = _mm256_extracti128_si256(res_i32, 1);
    __m128i res_i16 = _mm_packus_epi32(lo, hi);
    __m128i res_i8 = _mm_packus_epi16(res_i16, res_i16);

    _mm_storel_epi64((__m128i *)&out[i], res_i8);
  }

  for (size_t j = end; j < n; j++)
    out[j] = (NUMC_UBYTE)((NUMC_FLOAT)a[j] / (NUMC_FLOAT)b[j]);
}

// =============================================================================
//  Scalar Operations — out[i] = a[i] OP scalar
// =============================================================================

// FLOAT SCALAR

void adds_float_scalar(const NUMC_FLOAT *a, NUMC_FLOAT scalar,
                       NUMC_FLOAT *out, size_t n) {
  size_t end = n & ~7ULL;
  __m256 vs = _mm256_set1_ps(scalar);

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 8) {
    __m256 va = _mm256_load_ps(&a[i]);
    _mm256_store_ps(&out[i], _mm256_add_ps(va, vs));
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] + scalar;
}

void muls_float_scalar(const NUMC_FLOAT *a, NUMC_FLOAT scalar,
                       NUMC_FLOAT *out, size_t n) {
  size_t end = n & ~7ULL;
  __m256 vs = _mm256_set1_ps(scalar);

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 8) {
    __m256 va = _mm256_load_ps(&a[i]);
    _mm256_store_ps(&out[i], _mm256_mul_ps(va, vs));
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] * scalar;
}

void subs_float_scalar(const NUMC_FLOAT *a, NUMC_FLOAT scalar,
                       NUMC_FLOAT *out, size_t n) {
  size_t end = n & ~7ULL;
  __m256 vs = _mm256_set1_ps(scalar);

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 8) {
    __m256 va = _mm256_load_ps(&a[i]);
    _mm256_store_ps(&out[i], _mm256_sub_ps(va, vs));
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] - scalar;
}

void div_float_scalar(const NUMC_FLOAT *a, NUMC_FLOAT scalar,
                      NUMC_FLOAT *out, size_t n) {
  size_t end = n & ~7ULL;
  __m256 vs = _mm256_set1_ps(scalar);

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 8) {
    __m256 va = _mm256_load_ps(&a[i]);
    _mm256_store_ps(&out[i], _mm256_div_ps(va, vs));
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] / scalar;
}

// DOUBLE SCALAR

void adds_double_scalar(const NUMC_DOUBLE *a, NUMC_DOUBLE scalar,
                        NUMC_DOUBLE *out, size_t n) {
  size_t end = n & ~3ULL;
  __m256d vs = _mm256_set1_pd(scalar);

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 4) {
    __m256d va = _mm256_load_pd(&a[i]);
    _mm256_store_pd(&out[i], _mm256_add_pd(va, vs));
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] + scalar;
}

void muls_double_scalar(const NUMC_DOUBLE *a, NUMC_DOUBLE scalar,
                        NUMC_DOUBLE *out, size_t n) {
  size_t end = n & ~3ULL;
  __m256d vs = _mm256_set1_pd(scalar);

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 4) {
    __m256d va = _mm256_load_pd(&a[i]);
    _mm256_store_pd(&out[i], _mm256_mul_pd(va, vs));
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] * scalar;
}

void subs_double_scalar(const NUMC_DOUBLE *a, NUMC_DOUBLE scalar,
                        NUMC_DOUBLE *out, size_t n) {
  size_t end = n & ~3ULL;
  __m256d vs = _mm256_set1_pd(scalar);

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 4) {
    __m256d va = _mm256_load_pd(&a[i]);
    _mm256_store_pd(&out[i], _mm256_sub_pd(va, vs));
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] - scalar;
}

void div_double_scalar(const NUMC_DOUBLE *a, NUMC_DOUBLE scalar,
                       NUMC_DOUBLE *out, size_t n) {
  size_t end = n & ~3ULL;
  __m256d vs = _mm256_set1_pd(scalar);

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 4) {
    __m256d va = _mm256_load_pd(&a[i]);
    _mm256_store_pd(&out[i], _mm256_div_pd(va, vs));
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] / scalar;
}

// INT SCALAR

void adds_int_scalar(const NUMC_INT *a, NUMC_INT scalar, NUMC_INT *out,
                     size_t n) {
  size_t end = n & ~7ULL;
  __m256i vs = _mm256_set1_epi32(scalar);

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 8) {
    __m256i va = _mm256_load_si256((const __m256i *)&a[i]);
    _mm256_store_si256((__m256i *)&out[i], _mm256_add_epi32(va, vs));
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] + scalar;
}

void muls_int_scalar(const NUMC_INT *a, NUMC_INT scalar, NUMC_INT *out,
                     size_t n) {
  size_t end = n & ~7ULL;
  __m256i vs = _mm256_set1_epi32(scalar);

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 8) {
    __m256i va = _mm256_load_si256((const __m256i *)&a[i]);
    _mm256_store_si256((__m256i *)&out[i], _mm256_mullo_epi32(va, vs));
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] * scalar;
}

void subs_int_scalar(const NUMC_INT *a, NUMC_INT scalar, NUMC_INT *out,
                     size_t n) {
  size_t end = n & ~7ULL;
  __m256i vs = _mm256_set1_epi32(scalar);

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 8) {
    __m256i va = _mm256_load_si256((const __m256i *)&a[i]);
    _mm256_store_si256((__m256i *)&out[i], _mm256_sub_epi32(va, vs));
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] - scalar;
}

void div_int_scalar(const NUMC_INT *a, NUMC_INT scalar, NUMC_INT *out,
                    size_t n) {
  size_t end = n & ~3ULL;
  __m256d vs = _mm256_set1_pd((NUMC_DOUBLE)scalar);

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 4) {
    __m128i load_a = _mm_load_si128((const __m128i *)&a[i]);
    __m256d a_d = _mm256_cvtepi32_pd(load_a);
    __m256d div_as = _mm256_div_pd(a_d, vs);
    __m128i res = _mm256_cvtpd_epi32(div_as);
    _mm_store_si128((__m128i *)&out[i], res);
  }

  for (size_t i = end; i < n; i++)
    out[i] = (NUMC_INT)((NUMC_DOUBLE)a[i] / (NUMC_DOUBLE)scalar);
}

// LONG SCALAR

void adds_long_scalar(const NUMC_LONG *a, NUMC_LONG scalar, NUMC_LONG *out,
                      size_t n) {
  size_t end = n & ~3ULL;
  __m256i vs = _mm256_set1_epi64x(scalar);

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 4) {
    __m256i va = _mm256_load_si256((const __m256i *)&a[i]);
    _mm256_store_si256((__m256i *)&out[i], _mm256_add_epi64(va, vs));
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] + scalar;
}

void muls_long_scalar(const NUMC_LONG *a, NUMC_LONG scalar, NUMC_LONG *out,
                      size_t n) {
  size_t end = n & ~3ULL;
  __m256i vs = _mm256_set1_epi64x(scalar);

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 4) {
    __m256i va = _mm256_load_si256((const __m256i *)&a[i]);
    _mm256_store_si256((__m256i *)&out[i], _mm256_mul_epi64(va, vs));
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] * scalar;
}

void subs_long_scalar(const NUMC_LONG *a, NUMC_LONG scalar, NUMC_LONG *out,
                      size_t n) {
  size_t end = n & ~3ULL;
  __m256i vs = _mm256_set1_epi64x(scalar);

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 4) {
    __m256i va = _mm256_load_si256((const __m256i *)&a[i]);
    _mm256_store_si256((__m256i *)&out[i], _mm256_sub_epi64(va, vs));
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] - scalar;
}

// SHORT SCALAR

void adds_short_scalar(const NUMC_SHORT *a, NUMC_SHORT scalar,
                       NUMC_SHORT *out, size_t n) {
  size_t end = n & ~15ULL;
  __m256i vs = _mm256_set1_epi16(scalar);

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 16) {
    __m256i va = _mm256_load_si256((const __m256i *)&a[i]);
    _mm256_store_si256((__m256i *)&out[i], _mm256_add_epi16(va, vs));
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] + scalar;
}

void muls_short_scalar(const NUMC_SHORT *a, NUMC_SHORT scalar,
                       NUMC_SHORT *out, size_t n) {
  size_t end = n & ~15ULL;
  __m256i vs = _mm256_set1_epi16(scalar);

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 16) {
    __m256i va = _mm256_load_si256((const __m256i *)&a[i]);
    _mm256_store_si256((__m256i *)&out[i], _mm256_mullo_epi16(va, vs));
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] * scalar;
}

void subs_short_scalar(const NUMC_SHORT *a, NUMC_SHORT scalar,
                       NUMC_SHORT *out, size_t n) {
  size_t end = n & ~15ULL;
  __m256i vs = _mm256_set1_epi16(scalar);

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 16) {
    __m256i va = _mm256_load_si256((const __m256i *)&a[i]);
    _mm256_store_si256((__m256i *)&out[i], _mm256_sub_epi16(va, vs));
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] - scalar;
}

void div_short_scalar(const NUMC_SHORT *a, NUMC_SHORT scalar,
                      NUMC_SHORT *out, size_t n) {
  size_t end = n & ~7ULL;
  __m256 vs = _mm256_set1_ps((float)scalar);

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 8) {
    __m128i load_a = _mm_load_si128((const __m128i *)&a[i]);
    __m256i a_i32 = _mm256_cvtepi16_epi32(load_a);
    __m256 a_f32 = _mm256_cvtepi32_ps(a_i32);
    __m256 div_as = _mm256_div_ps(a_f32, vs);
    __m256i res_i32 = _mm256_cvttps_epi32(div_as);

    __m128i lo = _mm256_castsi256_si128(res_i32);
    __m128i hi = _mm256_extracti128_si256(res_i32, 1);
    __m128i res_i16 = _mm_packs_epi32(lo, hi);
    _mm_store_si128((__m128i *)&out[i], res_i16);
  }

  for (size_t i = end; i < n; i++)
    out[i] = (NUMC_SHORT)((NUMC_FLOAT)a[i] / (NUMC_FLOAT)scalar);
}

// BYTE SCALAR

void adds_byte_scalar(const NUMC_BYTE *a, NUMC_BYTE scalar, NUMC_BYTE *out,
                      size_t n) {
  size_t end = n & ~31ULL;
  __m256i vs = _mm256_set1_epi8(scalar);

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 32) {
    __m256i va = _mm256_load_si256((const __m256i *)&a[i]);
    _mm256_store_si256((__m256i *)&out[i], _mm256_add_epi8(va, vs));
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] + scalar;
}

void muls_byte_scalar(const NUMC_BYTE *a, NUMC_BYTE scalar, NUMC_BYTE *out,
                      size_t n) {
  size_t end = n & ~31ULL;
  __m256i zero = _mm256_setzero_si256();
  __m256i vs16 = _mm256_set1_epi16((int16_t)scalar);

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 32) {
    __m256i va = _mm256_load_si256((const __m256i *)&a[i]);

    __m256i a_lo = _mm256_unpacklo_epi8(va, zero);
    __m256i a_hi = _mm256_unpackhi_epi8(va, zero);

    __m256i mul_lo = _mm256_mullo_epi16(a_lo, vs16);
    __m256i mul_hi = _mm256_mullo_epi16(a_hi, vs16);

    __m256i packed = _mm256_packs_epi16(mul_lo, mul_hi);
    _mm256_store_si256((__m256i *)&out[i], packed);
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] * scalar;
}

void subs_byte_scalar(const NUMC_BYTE *a, NUMC_BYTE scalar, NUMC_BYTE *out,
                      size_t n) {
  size_t end = n & ~31ULL;
  __m256i vs = _mm256_set1_epi8(scalar);

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 32) {
    __m256i va = _mm256_load_si256((const __m256i *)&a[i]);
    _mm256_store_si256((__m256i *)&out[i], _mm256_sub_epi8(va, vs));
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] - scalar;
}

void div_byte_scalar(const NUMC_BYTE *a, NUMC_BYTE scalar, NUMC_BYTE *out,
                     size_t n) {
  size_t end = n & ~7ULL;
  __m256 vs = _mm256_set1_ps((float)scalar);

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 8) {
    __m128i load_a = _mm_loadl_epi64((const __m128i *)&a[i]);
    __m256i a_i32 = _mm256_cvtepi8_epi32(load_a);
    __m256 a_f32 = _mm256_cvtepi32_ps(a_i32);
    __m256 div_as = _mm256_div_ps(a_f32, vs);
    __m256i res_i32 = _mm256_cvttps_epi32(div_as);

    __m128i lo = _mm256_castsi256_si128(res_i32);
    __m128i hi = _mm256_extracti128_si256(res_i32, 1);
    __m128i res_i16 = _mm_packs_epi32(lo, hi);
    __m128i res_i8 = _mm_packs_epi16(res_i16, res_i16);
    _mm_storel_epi64((__m128i *)&out[i], res_i8);
  }

  for (size_t i = end; i < n; i++)
    out[i] = (NUMC_BYTE)((NUMC_FLOAT)a[i] / (NUMC_FLOAT)scalar);
}

// UINT SCALAR

void adds_uint_scalar(const NUMC_UINT *a, NUMC_UINT scalar, NUMC_UINT *out,
                      size_t n) {
  size_t end = n & ~7ULL;
  __m256i vs = _mm256_set1_epi32((int32_t)scalar);

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 8) {
    __m256i va = _mm256_load_si256((const __m256i *)&a[i]);
    _mm256_store_si256((__m256i *)&out[i], _mm256_add_epi32(va, vs));
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] + scalar;
}

void muls_uint_scalar(const NUMC_UINT *a, NUMC_UINT scalar, NUMC_UINT *out,
                      size_t n) {
  size_t end = n & ~7ULL;
  __m256i vs = _mm256_set1_epi32((int32_t)scalar);

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 8) {
    __m256i va = _mm256_load_si256((const __m256i *)&a[i]);
    _mm256_store_si256((__m256i *)&out[i], _mm256_mullo_epi32(va, vs));
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] * scalar;
}

void subs_uint_scalar(const NUMC_UINT *a, NUMC_UINT scalar, NUMC_UINT *out,
                      size_t n) {
  size_t end = n & ~7ULL;
  __m256i vs = _mm256_set1_epi32((int32_t)scalar);

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 8) {
    __m256i va = _mm256_load_si256((const __m256i *)&a[i]);
    _mm256_store_si256((__m256i *)&out[i], _mm256_sub_epi32(va, vs));
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] - scalar;
}

void div_uint_scalar(const NUMC_UINT *a, NUMC_UINT scalar, NUMC_UINT *out,
                     size_t n) {
  size_t end = n & ~3ULL;
  __m128i mask16 = _mm_set1_epi32(0xFFFF);
  __m128i s_raw = _mm_set1_epi32((int32_t)scalar);

  __m256d vs_d = _mm256_add_pd(
      _mm256_cvtepi32_pd(_mm_and_si128(s_raw, mask16)),
      _mm256_mul_pd(_mm256_cvtepi32_pd(_mm_srli_epi32(s_raw, 16)),
                    _mm256_set1_pd(65536.0)));

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 4) {
    __m128i load_a = _mm_load_si128((const __m128i *)&a[i]);

    __m256d a_d = _mm256_add_pd(
        _mm256_cvtepi32_pd(_mm_and_si128(load_a, mask16)),
        _mm256_mul_pd(_mm256_cvtepi32_pd(_mm_srli_epi32(load_a, 16)),
                      _mm256_set1_pd(65536.0)));

    __m256d div_as = _mm256_div_pd(a_d, vs_d);
    __m128i res = _mm256_cvttpd_epi32(div_as);
    _mm_store_si128((__m128i *)&out[i], res);
  }

  for (size_t i = end; i < n; i++)
    out[i] = (NUMC_UINT)((NUMC_DOUBLE)a[i] / (NUMC_DOUBLE)scalar);
}

// ULONG SCALAR

void adds_ulong_scalar(const NUMC_ULONG *a, NUMC_ULONG scalar,
                       NUMC_ULONG *out, size_t n) {
  size_t end = n & ~3ULL;
  __m256i vs = _mm256_set1_epi64x((int64_t)scalar);

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 4) {
    __m256i va = _mm256_load_si256((const __m256i *)&a[i]);
    _mm256_store_si256((__m256i *)&out[i], _mm256_add_epi64(va, vs));
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] + scalar;
}

void muls_ulong_scalar(const NUMC_ULONG *a, NUMC_ULONG scalar,
                       NUMC_ULONG *out, size_t n) {
  size_t end = n & ~3ULL;
  __m256i vs = _mm256_set1_epi64x((int64_t)scalar);

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 4) {
    __m256i va = _mm256_load_si256((const __m256i *)&a[i]);
    _mm256_store_si256((__m256i *)&out[i], _mm256_mul_epi64(va, vs));
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] * scalar;
}

void subs_ulong_scalar(const NUMC_ULONG *a, NUMC_ULONG scalar,
                       NUMC_ULONG *out, size_t n) {
  size_t end = n & ~3ULL;
  __m256i vs = _mm256_set1_epi64x((int64_t)scalar);

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 4) {
    __m256i va = _mm256_load_si256((const __m256i *)&a[i]);
    _mm256_store_si256((__m256i *)&out[i], _mm256_sub_epi64(va, vs));
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] - scalar;
}

// USHORT SCALAR

void adds_ushort_scalar(const NUMC_USHORT *a, NUMC_USHORT scalar,
                        NUMC_USHORT *out, size_t n) {
  size_t end = n & ~15ULL;
  __m256i vs = _mm256_set1_epi16((int16_t)scalar);

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 16) {
    __m256i va = _mm256_load_si256((const __m256i *)&a[i]);
    _mm256_store_si256((__m256i *)&out[i], _mm256_add_epi16(va, vs));
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] + scalar;
}

void muls_ushort_scalar(const NUMC_USHORT *a, NUMC_USHORT scalar,
                        NUMC_USHORT *out, size_t n) {
  size_t end = n & ~15ULL;
  __m256i vs = _mm256_set1_epi16((int16_t)scalar);

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 16) {
    __m256i va = _mm256_load_si256((const __m256i *)&a[i]);
    _mm256_store_si256((__m256i *)&out[i], _mm256_mullo_epi16(va, vs));
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] * scalar;
}

void subs_ushort_scalar(const NUMC_USHORT *a, NUMC_USHORT scalar,
                        NUMC_USHORT *out, size_t n) {
  size_t end = n & ~15ULL;
  __m256i vs = _mm256_set1_epi16((int16_t)scalar);

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 16) {
    __m256i va = _mm256_load_si256((const __m256i *)&a[i]);
    _mm256_store_si256((__m256i *)&out[i], _mm256_sub_epi16(va, vs));
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] - scalar;
}

void div_ushort_scalar(const NUMC_USHORT *a, NUMC_USHORT scalar,
                       NUMC_USHORT *out, size_t n) {
  size_t end = n & ~7ULL;
  __m256 vs = _mm256_set1_ps((float)scalar);

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 8) {
    __m128i load_a = _mm_load_si128((const __m128i *)&a[i]);
    __m256i a_i32 = _mm256_cvtepu16_epi32(load_a);
    __m256 a_f32 = _mm256_cvtepi32_ps(a_i32);
    __m256 div_as = _mm256_div_ps(a_f32, vs);
    __m256i res_i32 = _mm256_cvttps_epi32(div_as);

    __m128i lo = _mm256_castsi256_si128(res_i32);
    __m128i hi = _mm256_extracti128_si256(res_i32, 1);
    __m128i res_i16 = _mm_packus_epi32(lo, hi);
    _mm_store_si128((__m128i *)&out[i], res_i16);
  }

  for (size_t i = end; i < n; i++)
    out[i] = (NUMC_USHORT)((NUMC_FLOAT)a[i] / (NUMC_FLOAT)scalar);
}

// UBYTE SCALAR

void adds_ubyte_scalar(const NUMC_UBYTE *a, NUMC_UBYTE scalar,
                       NUMC_UBYTE *out, size_t n) {
  size_t end = n & ~31ULL;
  __m256i vs = _mm256_set1_epi8((char)scalar);

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 32) {
    __m256i va = _mm256_load_si256((const __m256i *)&a[i]);
    _mm256_store_si256((__m256i *)&out[i], _mm256_add_epi8(va, vs));
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] + scalar;
}

void muls_ubyte_scalar(const NUMC_UBYTE *a, NUMC_UBYTE scalar,
                       NUMC_UBYTE *out, size_t n) {
  size_t end = n & ~31ULL;
  __m256i zero = _mm256_setzero_si256();
  __m256i vs16 = _mm256_set1_epi16((int16_t)scalar);

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 32) {
    __m256i va = _mm256_load_si256((const __m256i *)&a[i]);

    __m256i a_lo = _mm256_unpacklo_epi8(va, zero);
    __m256i a_hi = _mm256_unpackhi_epi8(va, zero);

    __m256i mul_lo = _mm256_mullo_epi16(a_lo, vs16);
    __m256i mul_hi = _mm256_mullo_epi16(a_hi, vs16);

    __m256i packed = _mm256_packus_epi16(mul_lo, mul_hi);
    _mm256_store_si256((__m256i *)&out[i], packed);
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] * scalar;
}

void subs_ubyte_scalar(const NUMC_UBYTE *a, NUMC_UBYTE scalar,
                       NUMC_UBYTE *out, size_t n) {
  size_t end = n & ~31ULL;
  __m256i vs = _mm256_set1_epi8((char)scalar);

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 32) {
    __m256i va = _mm256_load_si256((const __m256i *)&a[i]);
    _mm256_store_si256((__m256i *)&out[i], _mm256_sub_epi8(va, vs));
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] - scalar;
}

void div_ubyte_scalar(const NUMC_UBYTE *a, NUMC_UBYTE scalar,
                      NUMC_UBYTE *out, size_t n) {
  size_t end = n & ~7ULL;
  __m256 vs = _mm256_set1_ps((float)scalar);

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 8) {
    __m128i load_a = _mm_loadl_epi64((const __m128i *)&a[i]);
    __m256i a_i32 = _mm256_cvtepu8_epi32(load_a);
    __m256 a_f32 = _mm256_cvtepi32_ps(a_i32);
    __m256 div_as = _mm256_div_ps(a_f32, vs);
    __m256i res_i32 = _mm256_cvttps_epi32(div_as);

    __m128i lo = _mm256_castsi256_si128(res_i32);
    __m128i hi = _mm256_extracti128_si256(res_i32, 1);
    __m128i res_i16 = _mm_packus_epi32(lo, hi);
    __m128i res_i8 = _mm_packus_epi16(res_i16, res_i16);
    _mm_storel_epi64((__m128i *)&out[i], res_i8);
  }

  for (size_t i = end; i < n; i++)
    out[i] = (NUMC_UBYTE)((NUMC_FLOAT)a[i] / (NUMC_FLOAT)scalar);
}
