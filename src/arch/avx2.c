#include "../internal.h"
#include <immintrin.h>
#include <numc/dtype.h>

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
    __m256 add_ab = _mm256_mul_ps(load_a, load_b);
    _mm256_store_ps(&out[i], add_ab);
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
    __m256 add_ab = _mm256_sub_ps(load_a, load_b);
    _mm256_store_ps(&out[i], add_ab);
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
  size_t end = n & ~7ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 4) {
    __m256 load_a = _mm256_load_pd(&a[i]);
    __m256 load_b = _mm256_load_pd(&b[i]);
    __m256 add_ab = _mm256_add_pd(load_a, load_b);
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
    __m256 load_a = _mm256_load_pd(&a[i]);
    __m256 load_b = _mm256_load_pd(&b[i]);
    __m256 add_ab = _mm256_mul_pd(load_a, load_b);
    _mm256_store_pd(&out[i], add_ab);
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] * b[i];
}

void subs_double_avx2(const NUMC_DOUBLE *a, const NUMC_DOUBLE *b,
                      NUMC_DOUBLE *out, size_t n) {
  size_t end = n & ~3ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 4) {
    __m256 load_a = _mm256_load_pd(&a[i]);
    __m256 load_b = _mm256_load_pd(&b[i]);
    __m256 add_ab = _mm256_sub_ps(load_a, load_b);
    _mm256_store_pd(&out[i], add_ab);
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] - b[i];
}

// TODO: IMPLEMENT DIV
void div_double_avx2(const NUMC_DOUBLE *a, const NUMC_DOUBLE *b,
                     NUMC_DOUBLE *out, size_t n) {}

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
    __m256i add_ab = _mm256_mul_epi32(load_a, load_b);
    _mm256_store_si256((__m256i *)&out[i], add_ab);
  }

  for (size_t j = end; j < n; j++)
    out[j] = a[j] + b[j];
}
void subs_int_avx2(const NUMC_INT *a, const NUMC_INT *b, NUMC_INT *out,
                   size_t n) {
  size_t end = n & ~7ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 8) {
    __m256i load_a = _mm256_load_si256((const __m256i *)&a[i]);
    __m256i load_b = _mm256_load_si256((const __m256i *)&b[i]);
    __m256i add_ab = _mm256_sub_epi32(load_a, load_b);
    _mm256_store_si256((__m256i *)&out[i], add_ab);
  }

  for (size_t j = end; j < n; j++)
    out[j] = a[j] * b[j];
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
    __m256i add_ab = _mm256_mul_epi64(load_a, load_b);
    _mm256_store_si256((__m256i *)&out[i], add_ab);
  }

  for (size_t j = end; j < n; j++)
    out[j] = a[j] + b[j];
}
void subs_long_avx2(const NUMC_LONG *a, const NUMC_LONG *b, NUMC_LONG *out,
                    size_t n) {
  size_t end = n & ~3ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 4) {
    __m256i load_a = _mm256_load_si256((const __m256i *)&a[i]);
    __m256i load_b = _mm256_load_si256((const __m256i *)&b[i]);
    __m256i add_ab = _mm256_sub_epi64(load_a, load_b);
    _mm256_store_si256((__m256i *)&out[i], add_ab);
  }

  for (size_t j = end; j < n; j++)
    out[j] = a[j] * b[j];
}

// SHORT

void adds_short_avx2(const NUMC_SHORT *a, const NUMC_SHORT *b, NUMC_SHORT *out,
                     size_t n) {
  size_t end = n & ~15ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 16) {
    __m256i load_a = _mm256_load_si256((const __m256i *)&a[i]);
    __m256i load_b = _mm256_load_si256((const __m256i *)&b[i]);
    __m256i add_ab = _mm256_add_epi32(load_a, load_b);
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
    __m256i add_ab = _mm256_mullo_epi16(load_a, load_b);
    _mm256_store_si256((__m256i *)&out[i], add_ab);
  }

  for (size_t j = end; j < n; j++)
    out[j] = a[j] + b[j];
}
void subs_short_avx2(const NUMC_SHORT *a, const NUMC_SHORT *b, NUMC_SHORT *out,
                     size_t n) {
  size_t end = n & ~15ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 16) {
    __m256i load_a = _mm256_load_si256((const __m256i *)&a[i]);
    __m256i load_b = _mm256_load_si256((const __m256i *)&b[i]);
    __m256i add_ab = _mm256_sub_epi16(load_a, load_b);
    _mm256_store_si256((__m256i *)&out[i], add_ab);
  }

  for (size_t j = end; j < n; j++)
    out[j] = a[j] * b[j];
}

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
  size_t end = n & ~31;
  __m256i zero = _mm256_setzero_si256();

  for (size_t i = 0; i < end; i += 32) {
    __m256i va = _mm256_loadu_si256((__m256i *)&a[i]);
    __m256i vb = _mm256_loadu_si256((__m256i *)&b[i]);

    __m256i a_lo = _mm256_unpacklo_epi8(va, zero);
    __m256i a_hi = _mm256_unpackhi_epi8(va, zero);
    __m256i b_lo = _mm256_unpacklo_epi8(vb, zero);
    __m256i b_hi = _mm256_unpackhi_epi8(vb, zero);

    __m256i mul_lo = _mm256_mullo_epi16(a_lo, b_lo);
    __m256i mul_hi = _mm256_mullo_epi16(a_hi, b_hi);

    __m256i packed = _mm256_packs_epi16(mul_lo, mul_hi);

    _mm256_storeu_si256((__m256i *)&out[i], packed);
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

// UNSIGNED INT

void adds_uint_avx2(const NUMC_UINT *a, const NUMC_UINT *b, NUMC_UINT *out,
                    size_t n) {
  size_t end = n & 7ULL;

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
  size_t end = n & 7ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 8) {
    __m256i load_a = _mm256_load_si256((const __m256i *)&a[i]);
    __m256i load_b = _mm256_load_si256((const __m256i *)&b[i]);
    __m256i mul_ab = _mm256_mul_epu32(load_a, load_b);
    _mm256_store_si256((__m256i *)&out[i], mul_ab);
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] * b[i];
}

void subs_uint_avx2(const NUMC_UINT *a, const NUMC_UINT *b, NUMC_UINT *out,
                    size_t n) {
  size_t end = n & 7ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 8) {
    __m256i load_a = _mm256_load_si256((const __m256i *)&a[i]);
    __m256i load_b = _mm256_load_si256((const __m256i *)&b[i]);
    __m256i sub_ab = _mm256_sub_epi32(load_a, load_b);
    _mm256_store_si256((__m256i *)&out[i], sub_ab);
  }

  for (size_t i = end; i < n; i++) {
    out[i] = a[i] - b[i];
  }
}

void adds_ulong_avx2(const NUMC_ULONG *a, const NUMC_ULONG *b, NUMC_ULONG *out,
                     size_t n) {
  size_t end = n & 3ULL;
#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 4) {
    __m256i load_a = _mm256_load_si256((const __m256i *)&a[i]);
    __m256i load_b = _mm256_load_si256((const __m256i *)&b[i]);
    __m256i add_ab = _mm256_add_epi64(load_a, load_b);
    _mm256_store_si256((__m256i *)&out[i], add_ab);
  }
}
void muls_ulong_avx2(const NUMC_ULONG *a, const NUMC_ULONG *b, NUMC_ULONG *out,
                     size_t n) {
  size_t end = n & 3ULL;

#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 4) {
    __m256i load_a = _mm256_load_si256((const __m256i *)&a[i]);
    __m256i load_b = _mm256_load_si256((const __m256i *)&b[i]);
    __m256i mul_ab = _mm256_mul_epi64(load_a, load_b);
    _mm256_store_si256((__m256i *)&out[i], mul_ab);
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] * b[i];
}

void subs_ulong_avx2(const NUMC_ULONG *a, const NUMC_ULONG *b, NUMC_ULONG *out,
                     size_t n) {
  size_t end = n & 3ULL;
#pragma omp parallel for schedule(static) if (n > NUMC_MAX_ELEMENT_LOOP)
  for (size_t i = 0; i < end; i += 4) {
    __m256i load_a = _mm256_load_si256((const __m256i *)&a[i]);
    __m256i load_b = _mm256_load_si256((const __m256i *)&b[i]);
    __m256i sub_ab = _mm256_sub_epi64(load_a, load_b);
    _mm256_store_si256((__m256i *)&out[i], sub_ab);
  }

  for (size_t i = end; i < n; i++)
    out[i] = a[i] - b[i];
}

void adds_ushort_avx2(const NUMC_USHORT *a, const NUMC_USHORT *b,
                      NUMC_USHORT *out, size_t n) {
  size_t end = n & 15ULL;

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
  size_t end = n & 15ULL;

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
  size_t end = n & 15ULL;

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
