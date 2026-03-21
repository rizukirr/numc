#ifndef NUMC_GEMM_AVX2_H
#define NUMC_GEMM_AVX2_H

#include <immintrin.h>
#include <stdint.h>
#include <string.h>

#define GEMM_MIN(a, b) ((a) < (b) ? (a) : (b))

/*
 * Cache-blocking parameters for AVX2 packed GEMM (BLIS-derived).
 *   - MC × KC panel of A resides in L2  (f32: 168×512×4 = 336KB)
 *                                        (f64:  72×256×8 = 144KB)
 *   - KC × NR sliver of B resides in L1 (f32: 512×16×4 = 32KB < 48KB)
 *                                        (f64: 256× 8×8 = 16KB < 48KB)
 *   - KC × NC panel of B resides in L3
 * MC=168 (f32) and MC=72 (f64) maximize L2 utilization. Thread utilization
 * is maintained via 2D IC×JR parallelism: tasks = ceil(M/MC) × ceil(N/NR),
 * giving many more work items than 1D IC-loop alone.
 */
#define GEMM_F32_MR 6
#define GEMM_F32_NR 16
#define GEMM_F32_MC 168
#define GEMM_F32_KC 512

#define GEMM_F64_MR 6
#define GEMM_F64_NR 8
#define GEMM_F64_MC 72
#define GEMM_F64_KC 256

#define GEMM_I32_MR 6
#define GEMM_I32_NR 16
#define GEMM_I32_MC 72
#define GEMM_I32_KC 256

#define GEMM_I16_MR 6
#define GEMM_I16_NR 32
#define GEMM_I16_MC 72
#define GEMM_I16_KC 512

#define GEMM_I64_MR 6
#define GEMM_I64_NR 8
#define GEMM_I64_MC 72
#define GEMM_I64_KC 64

/* i8/u8: promoted to i32 accumulators, KC-blocked packed GEMM */
#define GEMM_I8_MR 6
#define GEMM_I8_NR 16
#define GEMM_I8_MC 72
#define GEMM_I8_KC 512
#define GEMM_I8_NC 4080

/* GEMM OMP threshold on compute volume (M × K × N operations).
 * Threading pays off when compute dominates OMP fork cost (~20-50μs).
 * 256×256: 16.8M ops → ~330μs single-threaded → threading helps.
 * 128×128: 2.1M ops → ~44μs single-threaded → marginal, but numpy
 *          uses threading here too, so we match their strategy. */
#define GEMM_OMP_THRESHOLD (1 << 20)

/* N-dimension blocking for L3 residency (B panel: KC × NC elements) */
#define GEMM_F32_NC 4080
#define GEMM_F64_NC 4080
#define GEMM_I32_NC 4080
#define GEMM_I16_NC 4096

/* ─── Dedicated .S assembly micro-kernels (x86_64 only, SysV ABI) ─── */
#if (defined(__x86_64__) || defined(_M_X64)) && !defined(_MSC_VER)
extern void numc_gemm_ukernel_f32_6x16_avx2(const float *a, const float *b,
                                            float *c, uint64_t kc, int64_t rso,
                                            int first);
extern void numc_gemm_ukernel_f64_6x8_avx2(const double *a, const double *b,
                                           double *c, uint64_t kc, int64_t rso,
                                           int first);
extern void numc_gemm_ukernel_i32_6x16_avx2(const int32_t *a, const int32_t *b,
                                             int32_t *c, uint64_t kc,
                                             int64_t rso, int first);
extern void numc_gemm_ukernel_u32_6x16_avx2(const uint32_t *a, const uint32_t *b,
                                              uint32_t *c, uint64_t kc,
                                              int64_t rso, int first);
extern void numc_gemm_ukernel_i16_6x32_avx2(const int16_t *a, const int16_t *b,
                                              int16_t *c, uint64_t kc,
                                              int64_t rso, int first);
extern void numc_gemm_ukernel_u16_6x32_avx2(const uint16_t *a, const uint16_t *b,
                                              uint16_t *c, uint64_t kc,
                                              int64_t rso, int first);
extern void numc_gemm_ukernel_i8_6x16_avx2(const int8_t *a, const int8_t *b,
                                             int8_t *c, uint64_t kc,
                                             int64_t rso, int first);
extern void numc_gemm_ukernel_u8_6x16_avx2(const uint8_t *a, const uint8_t *b,
                                             uint8_t *c, uint64_t kc,
                                             int64_t rso, int first);
#define NUMC_HAVE_ASM_UKERNEL 1
#else
#define NUMC_HAVE_ASM_UKERNEL 0
#endif

static inline __m256i _gemm_mask_i32_lanes(size_t lanes) {
  static const int32_t mask_tbl[9][8] = {
      {0, 0, 0, 0, 0, 0, 0, 0},         {-1, 0, 0, 0, 0, 0, 0, 0},
      {-1, -1, 0, 0, 0, 0, 0, 0},       {-1, -1, -1, 0, 0, 0, 0, 0},
      {-1, -1, -1, -1, 0, 0, 0, 0},     {-1, -1, -1, -1, -1, 0, 0, 0},
      {-1, -1, -1, -1, -1, -1, 0, 0},   {-1, -1, -1, -1, -1, -1, -1, 0},
      {-1, -1, -1, -1, -1, -1, -1, -1},
  };
  return _mm256_loadu_si256((const __m256i *)mask_tbl[lanes]);
}

static inline __m256i _gemm_mask_i64_lanes(size_t lanes) {
  static const int64_t mask_tbl[5][4] = {
      {0, 0, 0, 0},    {-1, 0, 0, 0},    {-1, -1, 0, 0},
      {-1, -1, -1, 0}, {-1, -1, -1, -1},
  };
  return _mm256_loadu_si256((const __m256i *)mask_tbl[lanes]);
}

/* ── Float32 packing routines ──────────────────────────────────────────── */

/* Pack B[kc × nc] into NR-wide micropanels (row-panel format).
 * Layout: for each NR-panel at column jr, kc rows of NR contiguous elements.
 * Micro-kernel sees rsb = NR. */
static inline void gemm_pack_b_f32(const float *b, float *packed, size_t kc,
                                   size_t nc, intptr_t rsb) {
  size_t jr = 0;
  for (; jr + GEMM_F32_NR <= nc; jr += GEMM_F32_NR) {
    float *dest = packed + jr * kc;
    for (size_t p = 0; p < kc; p++) {
      const float *src = b + p * rsb + jr;
      _mm256_storeu_ps(dest + p * GEMM_F32_NR, _mm256_loadu_ps(src));
      _mm256_storeu_ps(dest + p * GEMM_F32_NR + 8, _mm256_loadu_ps(src + 8));
    }
  }
  if (jr < nc) {
    float *dest = packed + jr * kc;
    size_t rem = nc - jr;
    __m256 z = _mm256_setzero_ps();
    __m256i m0 = _gemm_mask_i32_lanes(rem < 8 ? rem : 8);
    __m256i m1 = _gemm_mask_i32_lanes(rem > 8 ? rem - 8 : 0);
    for (size_t p = 0; p < kc; p++) {
      const float *src = b + p * rsb + jr;
      float *d = dest + p * GEMM_F32_NR;
      _mm256_storeu_ps(d, _mm256_maskload_ps(src, m0));
      if (rem > 8) {
        _mm256_storeu_ps(d + 8, _mm256_maskload_ps(src + 8, m1));
      } else {
        _mm256_storeu_ps(d + 8, z);
      }
    }
  }
}

/* Pack a single NR-wide B strip for parallel packing. */
static inline void _gemm_pack_b_strip_f32(const float *b, float *dest,
                                          size_t kc, size_t nr_pack,
                                          intptr_t rsb) {
  if (nr_pack == GEMM_F32_NR) {
    for (size_t p = 0; p < kc; p++) {
      const float *src = b + p * rsb;
      _mm256_storeu_ps(dest + p * GEMM_F32_NR, _mm256_loadu_ps(src));
      _mm256_storeu_ps(dest + p * GEMM_F32_NR + 8, _mm256_loadu_ps(src + 8));
    }
  } else {
    __m256 z = _mm256_setzero_ps();
    __m256i m0 = _gemm_mask_i32_lanes(nr_pack < 8 ? nr_pack : 8);
    __m256i m1 = _gemm_mask_i32_lanes(nr_pack > 8 ? nr_pack - 8 : 0);
    for (size_t p = 0; p < kc; p++) {
      const float *src = b + p * rsb;
      float *d = dest + p * GEMM_F32_NR;
      _mm256_storeu_ps(d, _mm256_maskload_ps(src, m0));
      if (nr_pack > 8) {
        _mm256_storeu_ps(d + 8, _mm256_maskload_ps(src + 8, m1));
      } else {
        _mm256_storeu_ps(d + 8, z);
      }
    }
  }
}

/* Pack A[mc × kc] into MR-tall micropanels (column-panel format).
 * Layout: for each MR-panel at row ir, kc columns of MR contiguous elements.
 * Micro-kernel sees rsa = 1, csa = MR. */
static inline void gemm_pack_a_f32(const float *a, float *packed, size_t mc,
                                   size_t kc, intptr_t rsa, intptr_t csa) {
  size_t ir = 0;
  /* Fast path: csa=1 (row-major A) — gather MR rows with pointer arithmetic */
  if (csa == 1) {
    for (; ir + GEMM_F32_MR <= mc; ir += GEMM_F32_MR) {
      float *dest = packed + ir * kc;
      const float *r0 = a + (ir + 0) * rsa;
      const float *r1 = a + (ir + 1) * rsa;
      const float *r2 = a + (ir + 2) * rsa;
      const float *r3 = a + (ir + 3) * rsa;
      const float *r4 = a + (ir + 4) * rsa;
      const float *r5 = a + (ir + 5) * rsa;
      /* SIMD transpose: process 8 K-columns at a time.
         Load 8 floats from each of 6 rows, transpose rows 0-3 (4×4 per
         128-bit lane) and rows 4-5 (2×4 per lane), store 8 packed columns
         of 6 floats (MR=6, 24 bytes each). */
      size_t p = 0;
      for (; p + 8 <= kc; p += 8) {
        __m256 v0 = _mm256_loadu_ps(r0 + p);
        __m256 v1 = _mm256_loadu_ps(r1 + p);
        __m256 v2 = _mm256_loadu_ps(r2 + p);
        __m256 v3 = _mm256_loadu_ps(r3 + p);
        __m256 v4 = _mm256_loadu_ps(r4 + p);
        __m256 v5 = _mm256_loadu_ps(r5 + p);
        /* Standard 4×4 AVX2 in-lane transpose for rows 0-3.
         * t0[lo] = [r0[0],r1[0],r0[1],r1[1]], t0[hi] = [r0[4],r1[4],...]
         * t2[lo] = [r2[0],r3[0],r2[1],r3[1]], etc. */
        __m256 t0 = _mm256_unpacklo_ps(v0, v1);
        __m256 t1 = _mm256_unpackhi_ps(v0, v1);
        __m256 t2 = _mm256_unpacklo_ps(v2, v3);
        __m256 t3 = _mm256_unpackhi_ps(v2, v3);
        /* col k: [r0[k],r1[k],r2[k],r3[k]] via shuffle_ps */
        __m128 t0lo = _mm256_castps256_ps128(t0),
               t0hi = _mm256_extractf128_ps(t0, 1);
        __m128 t1lo = _mm256_castps256_ps128(t1),
               t1hi = _mm256_extractf128_ps(t1, 1);
        __m128 t2lo = _mm256_castps256_ps128(t2),
               t2hi = _mm256_extractf128_ps(t2, 1);
        __m128 t3lo = _mm256_castps256_ps128(t3),
               t3hi = _mm256_extractf128_ps(t3, 1);
        /* shuffle_ps(a,b,0x44)=[a[0],a[1],b[0],b[1]],
         * 0xEE=[a[2],a[3],b[2],b[3]] */
        __m128 c0 = _mm_shuffle_ps(t0lo, t2lo, 0x44); /* col 0, rows 0-3 */
        __m128 c1 = _mm_shuffle_ps(t0lo, t2lo, 0xEE); /* col 1, rows 0-3 */
        __m128 c2 = _mm_shuffle_ps(t1lo, t3lo, 0x44); /* col 2, rows 0-3 */
        __m128 c3 = _mm_shuffle_ps(t1lo, t3lo, 0xEE); /* col 3, rows 0-3 */
        __m128 c4 = _mm_shuffle_ps(t0hi, t2hi, 0x44); /* col 4, rows 0-3 */
        __m128 c5 = _mm_shuffle_ps(t0hi, t2hi, 0xEE); /* col 5, rows 0-3 */
        __m128 c6 = _mm_shuffle_ps(t1hi, t3hi, 0x44); /* col 6, rows 0-3 */
        __m128 c7 = _mm_shuffle_ps(t1hi, t3hi, 0xEE); /* col 7, rows 0-3 */
        /* Transpose rows 4-5: 2×8 partial — unpack gives pairs [r4[k],r5[k]] */
        __m256 u_lo = _mm256_unpacklo_ps(v4, v5);
        __m256 u_hi = _mm256_unpackhi_ps(v4, v5);
        __m128d u_lo_0 = _mm_castps_pd(_mm256_castps256_ps128(u_lo));
        __m128d u_lo_4 = _mm_castps_pd(_mm256_extractf128_ps(u_lo, 1));
        __m128d u_hi_0 = _mm_castps_pd(_mm256_castps256_ps128(u_hi));
        __m128d u_hi_4 = _mm_castps_pd(_mm256_extractf128_ps(u_hi, 1));
        /* Store 8 packed columns of 6 floats (24 bytes each).
         * Layout: [r0,r1,r2,r3] (4 floats) + [r4,r5] (lo/hi of u_*_*) */
        float *d = dest + p * GEMM_F32_MR;
        _mm_storeu_ps(d + 0 * 6, c0);
        _mm_storel_pd((double *)(d + 0 * 6 + 4), u_lo_0);
        _mm_storeu_ps(d + 1 * 6, c1);
        _mm_storeh_pd((double *)(d + 1 * 6 + 4), u_lo_0);
        _mm_storeu_ps(d + 2 * 6, c2);
        _mm_storel_pd((double *)(d + 2 * 6 + 4), u_hi_0);
        _mm_storeu_ps(d + 3 * 6, c3);
        _mm_storeh_pd((double *)(d + 3 * 6 + 4), u_hi_0);
        _mm_storeu_ps(d + 4 * 6, c4);
        _mm_storel_pd((double *)(d + 4 * 6 + 4), u_lo_4);
        _mm_storeu_ps(d + 5 * 6, c5);
        _mm_storeh_pd((double *)(d + 5 * 6 + 4), u_lo_4);
        _mm_storeu_ps(d + 6 * 6, c6);
        _mm_storel_pd((double *)(d + 6 * 6 + 4), u_hi_4);
        _mm_storeu_ps(d + 7 * 6, c7);
        _mm_storeh_pd((double *)(d + 7 * 6 + 4), u_hi_4);
      }
      /* Scalar cleanup for kc % 8 remainder */
      for (; p < kc; p++) {
        float *d = dest + p * GEMM_F32_MR;
        d[0] = r0[p];
        d[1] = r1[p];
        d[2] = r2[p];
        d[3] = r3[p];
        d[4] = r4[p];
        d[5] = r5[p];
      }
    }
    if (ir < mc) {
      float *dest = packed + ir * kc;
      size_t rem = mc - ir;
      __m256i m6 = _gemm_mask_i32_lanes(GEMM_F32_MR);
      for (size_t p = 0; p < kc; p++) {
        float v0 = rem > 0 ? a[(ir + 0) * rsa + p] : 0.0f;
        float v1 = rem > 1 ? a[(ir + 1) * rsa + p] : 0.0f;
        float v2 = rem > 2 ? a[(ir + 2) * rsa + p] : 0.0f;
        float v3 = rem > 3 ? a[(ir + 3) * rsa + p] : 0.0f;
        float v4 = rem > 4 ? a[(ir + 4) * rsa + p] : 0.0f;
        float v5 = rem > 5 ? a[(ir + 5) * rsa + p] : 0.0f;
        __m256 vv = _mm256_setr_ps(v0, v1, v2, v3, v4, v5, 0.0f, 0.0f);
        _mm256_maskstore_ps(dest + p * GEMM_F32_MR, m6, vv);
      }
    }
    return;
  }
  for (; ir + GEMM_F32_MR <= mc; ir += GEMM_F32_MR) {
    float *dest = packed + ir * kc;
    for (size_t p = 0; p < kc; p++) {
      for (size_t i = 0; i < GEMM_F32_MR; i++)
        dest[p * GEMM_F32_MR + i] = a[(ir + i) * rsa + p * csa];
    }
  }
  if (ir < mc) {
    float *dest = packed + ir * kc;
    size_t rem = mc - ir;
    __m256i m6 = _gemm_mask_i32_lanes(GEMM_F32_MR);
    for (size_t p = 0; p < kc; p++) {
      float v0 = rem > 0 ? a[(ir + 0) * rsa + p * csa] : 0.0f;
      float v1 = rem > 1 ? a[(ir + 1) * rsa + p * csa] : 0.0f;
      float v2 = rem > 2 ? a[(ir + 2) * rsa + p * csa] : 0.0f;
      float v3 = rem > 3 ? a[(ir + 3) * rsa + p * csa] : 0.0f;
      float v4 = rem > 4 ? a[(ir + 4) * rsa + p * csa] : 0.0f;
      float v5 = rem > 5 ? a[(ir + 5) * rsa + p * csa] : 0.0f;
      __m256 vv = _mm256_setr_ps(v0, v1, v2, v3, v4, v5, 0.0f, 0.0f);
      _mm256_maskstore_ps(dest + p * GEMM_F32_MR, m6, vv);
    }
  }
}

/* ── Float64 packing routines ──────────────────────────────────────────── */

static inline void gemm_pack_b_f64(const double *b, double *packed, size_t kc,
                                   size_t nc, intptr_t rsb) {
  size_t jr = 0;
  for (; jr + GEMM_F64_NR <= nc; jr += GEMM_F64_NR) {
    double *dest = packed + jr * kc;
    for (size_t p = 0; p < kc; p++) {
      const double *src = b + p * rsb + jr;
      _mm256_storeu_pd(dest + p * GEMM_F64_NR, _mm256_loadu_pd(src));
      _mm256_storeu_pd(dest + p * GEMM_F64_NR + 4, _mm256_loadu_pd(src + 4));
    }
  }
  if (jr < nc) {
    double *dest = packed + jr * kc;
    size_t rem = nc - jr;
    __m256d z = _mm256_setzero_pd();
    __m256i m0 = _gemm_mask_i64_lanes(rem < 4 ? rem : 4);
    __m256i m1 = _gemm_mask_i64_lanes(rem > 4 ? rem - 4 : 0);
    for (size_t p = 0; p < kc; p++) {
      const double *src = b + p * rsb + jr;
      double *d = dest + p * GEMM_F64_NR;
      _mm256_storeu_pd(d, _mm256_maskload_pd(src, m0));
      if (rem > 4) {
        _mm256_storeu_pd(d + 4, _mm256_maskload_pd(src + 4, m1));
      } else {
        _mm256_storeu_pd(d + 4, z);
      }
    }
  }
}

static inline void _gemm_pack_b_strip_f64(const double *b, double *dest,
                                          size_t kc, size_t nr_pack,
                                          intptr_t rsb) {
  if (nr_pack == GEMM_F64_NR) {
    for (size_t p = 0; p < kc; p++) {
      const double *src = b + p * rsb;
      _mm256_storeu_pd(dest + p * GEMM_F64_NR, _mm256_loadu_pd(src));
      _mm256_storeu_pd(dest + p * GEMM_F64_NR + 4, _mm256_loadu_pd(src + 4));
    }
  } else {
    __m256d z = _mm256_setzero_pd();
    __m256i m0 = _gemm_mask_i64_lanes(nr_pack < 4 ? nr_pack : 4);
    __m256i m1 = _gemm_mask_i64_lanes(nr_pack > 4 ? nr_pack - 4 : 0);
    for (size_t p = 0; p < kc; p++) {
      const double *src = b + p * rsb;
      double *d = dest + p * GEMM_F64_NR;
      _mm256_storeu_pd(d, _mm256_maskload_pd(src, m0));
      if (nr_pack > 4) {
        _mm256_storeu_pd(d + 4, _mm256_maskload_pd(src + 4, m1));
      } else {
        _mm256_storeu_pd(d + 4, z);
      }
    }
  }
}

static inline void gemm_pack_a_f64(const double *a, double *packed, size_t mc,
                                   size_t kc, intptr_t rsa, intptr_t csa) {
  size_t ir = 0;
  /* Fast path: csa=1 (row-major A) — gather MR rows with pointer arithmetic */
  if (csa == 1) {
    for (; ir + GEMM_F64_MR <= mc; ir += GEMM_F64_MR) {
      double *dest = packed + ir * kc;
      const double *r0 = a + (ir + 0) * rsa;
      const double *r1 = a + (ir + 1) * rsa;
      const double *r2 = a + (ir + 2) * rsa;
      const double *r3 = a + (ir + 3) * rsa;
      const double *r4 = a + (ir + 4) * rsa;
      const double *r5 = a + (ir + 5) * rsa;
      /* BLIS-style SIMD transpose: process 4 K-columns at a time.
         Load 4 doubles from each of 6 rows, transpose rows 0-3 (4×4)
         and rows 4-5 (2×4), store 4 packed columns of 6 doubles. */
      size_t p = 0;
      for (; p + 4 <= kc; p += 4) {
        /* Load 4 doubles from each row */
        __m256d v0 = _mm256_loadu_pd(r0 + p);
        __m256d v1 = _mm256_loadu_pd(r1 + p);
        __m256d v2 = _mm256_loadu_pd(r2 + p);
        __m256d v3 = _mm256_loadu_pd(r3 + p);
        __m256d v4 = _mm256_loadu_pd(r4 + p);
        __m256d v5 = _mm256_loadu_pd(r5 + p);
        /* Transpose rows 0-3: 4×4 double transpose */
        __m256d t0 = _mm256_unpacklo_pd(v0, v1); /* [r0_0,r1_0, r0_2,r1_2] */
        __m256d t1 = _mm256_unpackhi_pd(v0, v1); /* [r0_1,r1_1, r0_3,r1_3] */
        __m256d t2 = _mm256_unpacklo_pd(v2, v3); /* [r2_0,r3_0, r2_2,r3_2] */
        __m256d t3 = _mm256_unpackhi_pd(v2, v3); /* [r2_1,r3_1, r2_3,r3_3] */
        __m256d c0 = _mm256_insertf128_pd(t0, _mm256_castpd256_pd128(t2), 1);
        __m256d c1 = _mm256_insertf128_pd(t1, _mm256_castpd256_pd128(t3), 1);
        __m256d c2 = _mm256_permute2f128_pd(t0, t2, 0x31);
        __m256d c3 = _mm256_permute2f128_pd(t1, t3, 0x31);
        /* Transpose rows 4-5: 2×4 partial transpose */
        __m256d u0 = _mm256_unpacklo_pd(v4, v5);   /* [r4_0,r5_0, r4_2,r5_2] */
        __m256d u1 = _mm256_unpackhi_pd(v4, v5);   /* [r4_1,r5_1, r4_3,r5_3] */
        __m128d u2 = _mm256_extractf128_pd(u0, 1); /* [r4_2,r5_2] */
        __m128d u3 = _mm256_extractf128_pd(u1, 1); /* [r4_3,r5_3] */
        /* Store 4 packed columns of 6 doubles (48 bytes each) */
        double *d = dest + p * GEMM_F64_MR;
        _mm256_storeu_pd(d + 0 * 6, c0);
        _mm_storeu_pd(d + 0 * 6 + 4, _mm256_castpd256_pd128(u0));
        _mm256_storeu_pd(d + 1 * 6, c1);
        _mm_storeu_pd(d + 1 * 6 + 4, _mm256_castpd256_pd128(u1));
        _mm256_storeu_pd(d + 2 * 6, c2);
        _mm_storeu_pd(d + 2 * 6 + 4, u2);
        _mm256_storeu_pd(d + 3 * 6, c3);
        _mm_storeu_pd(d + 3 * 6 + 4, u3);
      }
      /* Scalar cleanup for kc % 4 remainder */
      for (; p < kc; p++) {
        double *d = dest + p * GEMM_F64_MR;
        d[0] = r0[p];
        d[1] = r1[p];
        d[2] = r2[p];
        d[3] = r3[p];
        d[4] = r4[p];
        d[5] = r5[p];
      }
    }
    if (ir < mc) {
      double *dest = packed + ir * kc;
      size_t rem = mc - ir;
      __m256i m2 = _gemm_mask_i64_lanes(2);
      for (size_t p = 0; p < kc; p++) {
        double v0 = rem > 0 ? a[(ir + 0) * rsa + p] : 0.0;
        double v1 = rem > 1 ? a[(ir + 1) * rsa + p] : 0.0;
        double v2 = rem > 2 ? a[(ir + 2) * rsa + p] : 0.0;
        double v3 = rem > 3 ? a[(ir + 3) * rsa + p] : 0.0;
        double v4 = rem > 4 ? a[(ir + 4) * rsa + p] : 0.0;
        double v5 = rem > 5 ? a[(ir + 5) * rsa + p] : 0.0;
        _mm256_storeu_pd(dest + p * GEMM_F64_MR,
                         _mm256_setr_pd(v0, v1, v2, v3));
        _mm256_maskstore_pd(dest + p * GEMM_F64_MR + 4, m2,
                            _mm256_setr_pd(v4, v5, 0.0, 0.0));
      }
    }
    return;
  }
  for (; ir + GEMM_F64_MR <= mc; ir += GEMM_F64_MR) {
    double *dest = packed + ir * kc;
    for (size_t p = 0; p < kc; p++) {
      for (size_t i = 0; i < GEMM_F64_MR; i++)
        dest[p * GEMM_F64_MR + i] = a[(ir + i) * rsa + p * csa];
    }
  }
  if (ir < mc) {
    double *dest = packed + ir * kc;
    size_t rem = mc - ir;
    __m256i m2 = _gemm_mask_i64_lanes(2);
    for (size_t p = 0; p < kc; p++) {
      double v0 = rem > 0 ? a[(ir + 0) * rsa + p * csa] : 0.0;
      double v1 = rem > 1 ? a[(ir + 1) * rsa + p * csa] : 0.0;
      double v2 = rem > 2 ? a[(ir + 2) * rsa + p * csa] : 0.0;
      double v3 = rem > 3 ? a[(ir + 3) * rsa + p * csa] : 0.0;
      double v4 = rem > 4 ? a[(ir + 4) * rsa + p * csa] : 0.0;
      double v5 = rem > 5 ? a[(ir + 5) * rsa + p * csa] : 0.0;
      _mm256_storeu_pd(dest + p * GEMM_F64_MR, _mm256_setr_pd(v0, v1, v2, v3));
      _mm256_maskstore_pd(dest + p * GEMM_F64_MR + 4, m2,
                          _mm256_setr_pd(v4, v5, 0.0, 0.0));
    }
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   Float32: 6×16 micro-kernel  (12 acc + 2 A broadcast + 2 B loads = 16 YMM)
   BLIS-style B pre-load: B vectors loaded at end of prev iter, ready for
   next iter. 2 alternating A broadcast registers for better ILP.
   ═══════════════════════════════════════════════════════════════════════════
 */

/* MSVC intrinsics K-iter: B pre-loaded in b0/b1 before call. */
#define GEMM_F32_K_ITER(ap, b0, b1)            \
  do {                                         \
    __m256 a0 = _mm256_broadcast_ss((ap) + 0); \
    __m256 a1 = _mm256_broadcast_ss((ap) + 1); \
    c00 = _mm256_fmadd_ps(a0, b0, c00);        \
    c01 = _mm256_fmadd_ps(a0, b1, c01);        \
    c10 = _mm256_fmadd_ps(a1, b0, c10);        \
    c11 = _mm256_fmadd_ps(a1, b1, c11);        \
    a0 = _mm256_broadcast_ss((ap) + 2);        \
    a1 = _mm256_broadcast_ss((ap) + 3);        \
    c20 = _mm256_fmadd_ps(a0, b0, c20);        \
    c21 = _mm256_fmadd_ps(a0, b1, c21);        \
    c30 = _mm256_fmadd_ps(a1, b0, c30);        \
    c31 = _mm256_fmadd_ps(a1, b1, c31);        \
    a0 = _mm256_broadcast_ss((ap) + 4);        \
    a1 = _mm256_broadcast_ss((ap) + 5);        \
    c40 = _mm256_fmadd_ps(a0, b0, c40);        \
    c41 = _mm256_fmadd_ps(a0, b1, c41);        \
    c50 = _mm256_fmadd_ps(a1, b0, c50);        \
    c51 = _mm256_fmadd_ps(a1, b1, c51);        \
  } while (0)

/* clang-format off */
/* BLIS-style ASM K-iter: B pre-loaded in ymm12/ymm13 from previous
 * iteration. Uses 2 A broadcast registers (ymm14/ymm15) for ILP.
 * Pre-loads next B at end for next iteration. */
#define GEMM_F32_ASM_K_ITER                      \
  "vbroadcastss (%[ap]), %%ymm14\n\t"            \
  "vbroadcastss 4(%[ap]), %%ymm15\n\t"           \
  "vfmadd231ps %%ymm14, %%ymm12, %%ymm0\n\t"    \
  "vfmadd231ps %%ymm14, %%ymm13, %%ymm1\n\t"    \
  "vfmadd231ps %%ymm15, %%ymm12, %%ymm2\n\t"    \
  "vfmadd231ps %%ymm15, %%ymm13, %%ymm3\n\t"    \
  "vbroadcastss 8(%[ap]), %%ymm14\n\t"           \
  "vbroadcastss 12(%[ap]), %%ymm15\n\t"          \
  "vfmadd231ps %%ymm14, %%ymm12, %%ymm4\n\t"    \
  "vfmadd231ps %%ymm14, %%ymm13, %%ymm5\n\t"    \
  "vfmadd231ps %%ymm15, %%ymm12, %%ymm6\n\t"    \
  "vfmadd231ps %%ymm15, %%ymm13, %%ymm7\n\t"    \
  "vbroadcastss 16(%[ap]), %%ymm14\n\t"          \
  "vbroadcastss 20(%[ap]), %%ymm15\n\t"          \
  "vfmadd231ps %%ymm14, %%ymm12, %%ymm8\n\t"    \
  "vfmadd231ps %%ymm14, %%ymm13, %%ymm9\n\t"    \
  "vfmadd231ps %%ymm15, %%ymm12, %%ymm10\n\t"   \
  "vfmadd231ps %%ymm15, %%ymm13, %%ymm11\n\t"   \
  "add %[csa_bytes], %[ap]\n\t"                  \
  "add %[rsb_bytes], %[bp]\n\t"                  \
  "vmovups (%[bp]), %%ymm12\n\t"                 \
  "vmovups 32(%[bp]), %%ymm13\n\t"
/* clang-format on */

#if defined(__GNUC__) || defined(__clang__)

/* Inline-asm f32 6×16 micro-kernel: 12 accumulators pinned to ymm0-ymm11,
 * no register spills. Interleaves vbroadcastss (port 5) with vfmadd231ps
 * (ports 0,1) for optimal port utilization. 8× unrolled K-loop with
 * 3 prefetches (A ahead, B ahead, A further) matching f64 kernel. */
static inline void gemm_ukernel_f32_6x16(const float *a, const float *b,
                                         float *c, size_t kc, intptr_t rsa,
                                         intptr_t csa, intptr_t rsb,
                                         intptr_t rso, int first) {
  (void)rsa;
  intptr_t csa_bytes = csa * (intptr_t)sizeof(float);
  intptr_t rsb_bytes = rsb * (intptr_t)sizeof(float);
  float *c1 = c + rso;
  float *c2 = c + 2 * rso;
  float *c3 = c + 3 * rso;
  float *c4 = c + 4 * rso;
  float *c5 = c + 5 * rso;

  const float *ap = a;
  const float *bp = b;

  __asm__ __volatile__(
      /* Prefetch C rows into L1 */
      "prefetcht0 (%[c0])\n\t"
      "prefetcht0 (%[c1])\n\t"
      "prefetcht0 (%[c2])\n\t"
      "prefetcht0 (%[c3])\n\t"
      "prefetcht0 (%[c4])\n\t"
      "prefetcht0 (%[c5])\n\t"
      /* Zero or load accumulators */
      "test %[first], %[first]\n\t"
      "jz 1f\n\t"
      /* first=1: zero all 12 accumulators */
      "vxorps %%ymm0, %%ymm0, %%ymm0\n\t"
      "vxorps %%ymm1, %%ymm1, %%ymm1\n\t"
      "vxorps %%ymm2, %%ymm2, %%ymm2\n\t"
      "vxorps %%ymm3, %%ymm3, %%ymm3\n\t"
      "vxorps %%ymm4, %%ymm4, %%ymm4\n\t"
      "vxorps %%ymm5, %%ymm5, %%ymm5\n\t"
      "vxorps %%ymm6, %%ymm6, %%ymm6\n\t"
      "vxorps %%ymm7, %%ymm7, %%ymm7\n\t"
      "vxorps %%ymm8, %%ymm8, %%ymm8\n\t"
      "vxorps %%ymm9, %%ymm9, %%ymm9\n\t"
      "vxorps %%ymm10, %%ymm10, %%ymm10\n\t"
      "vxorps %%ymm11, %%ymm11, %%ymm11\n\t"
      "jmp 2f\n\t"
      "1:\n\t"
      /* first=0: load existing C values */
      "vmovups (%[c0]), %%ymm0\n\t"
      "vmovups 32(%[c0]), %%ymm1\n\t"
      "vmovups (%[c1]), %%ymm2\n\t"
      "vmovups 32(%[c1]), %%ymm3\n\t"
      "vmovups (%[c2]), %%ymm4\n\t"
      "vmovups 32(%[c2]), %%ymm5\n\t"
      "vmovups (%[c3]), %%ymm6\n\t"
      "vmovups 32(%[c3]), %%ymm7\n\t"
      "vmovups (%[c4]), %%ymm8\n\t"
      "vmovups 32(%[c4]), %%ymm9\n\t"
      "vmovups (%[c5]), %%ymm10\n\t"
      "vmovups 32(%[c5]), %%ymm11\n\t"
      "2:\n\t"
      /* Pre-load first B vector pair (BLIS-style) */
      "vmovups (%[bp]), %%ymm12\n\t"
      "vmovups 32(%[bp]), %%ymm13\n\t"

      /* K-loop: k_iter = kc >> 3 (8× unrolled) */
      "mov %[kc], %%rcx\n\t"
      "shr $3, %%rcx\n\t"
      "jz 4f\n\t"

      ".p2align 5\n\t"
      "3:\n\t" GEMM_F32_ASM_K_ITER                    /* iter 0 */
      GEMM_F32_ASM_K_ITER                             /* iter 1 */
      "prefetcht0 384(%[ap])\n\t" GEMM_F32_ASM_K_ITER /* iter 2 */
      GEMM_F32_ASM_K_ITER                             /* iter 3 */
      "prefetcht0 512(%[bp])\n\t" GEMM_F32_ASM_K_ITER /* iter 4 */
      GEMM_F32_ASM_K_ITER                             /* iter 5 */
      "prefetcht0 768(%[ap])\n\t" GEMM_F32_ASM_K_ITER /* iter 6 */
      GEMM_F32_ASM_K_ITER                             /* iter 7 */

      "dec %%rcx\n\t"
      "jnz 3b\n\t"

      "4:\n\t"
      /* Handle k_left = kc & 7 */
      "mov %[kc], %%rcx\n\t"
      "and $7, %%rcx\n\t"
      "jz 6f\n\t"

      ".p2align 5\n\t"
      "5:\n\t" GEMM_F32_ASM_K_ITER "dec %%rcx\n\t"
      "jnz 5b\n\t"

      "6:\n\t"
      /* Refresh C prefetch — lines may have been evicted during
       * K-loop (A+B data exceeds L1 for large K) */
      "prefetcht0 (%[c0])\n\t"
      "prefetcht0 (%[c1])\n\t"
      "prefetcht0 (%[c2])\n\t"
      "prefetcht0 (%[c3])\n\t"
      "prefetcht0 (%[c4])\n\t"
      "prefetcht0 (%[c5])\n\t"
      /* Store accumulators to C */
      "vmovups %%ymm0, (%[c0])\n\t"
      "vmovups %%ymm1, 32(%[c0])\n\t"
      "vmovups %%ymm2, (%[c1])\n\t"
      "vmovups %%ymm3, 32(%[c1])\n\t"
      "vmovups %%ymm4, (%[c2])\n\t"
      "vmovups %%ymm5, 32(%[c2])\n\t"
      "vmovups %%ymm6, (%[c3])\n\t"
      "vmovups %%ymm7, 32(%[c3])\n\t"
      "vmovups %%ymm8, (%[c4])\n\t"
      "vmovups %%ymm9, 32(%[c4])\n\t"
      "vmovups %%ymm10, (%[c5])\n\t"
      "vmovups %%ymm11, 32(%[c5])\n\t"

      : [ap] "+r"(ap), [bp] "+r"(bp)
      : [c0] "r"(c), [c1] "r"(c1), [c2] "r"(c2), [c3] "r"(c3), [c4] "r"(c4),
        [c5] "r"(c5), [kc] "r"((uint64_t)kc), [first] "r"(first),
        [csa_bytes] "r"(csa_bytes), [rsb_bytes] "r"(rsb_bytes)
      : "rcx", "memory", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6",
        "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14",
        "ymm15");
}

#else /* MSVC fallback — C intrinsics version */

static inline void gemm_ukernel_f32_6x16(const float *a, const float *b,
                                         float *c, size_t kc, intptr_t rsa,
                                         intptr_t csa, intptr_t rsb,
                                         intptr_t rso, int first) {
  __m256 c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;

  /* Prefetch 6 rows of C into L1 (BLIS pattern) */
  _mm_prefetch((const char *)(c), _MM_HINT_T0);
  _mm_prefetch((const char *)(c + rso), _MM_HINT_T0);
  _mm_prefetch((const char *)(c + 2 * rso), _MM_HINT_T0);
  _mm_prefetch((const char *)(c + 3 * rso), _MM_HINT_T0);
  _mm_prefetch((const char *)(c + 4 * rso), _MM_HINT_T0);
  _mm_prefetch((const char *)(c + 5 * rso), _MM_HINT_T0);

  if (first) {
    c00 = _mm256_setzero_ps();
    c01 = _mm256_setzero_ps();
    c10 = _mm256_setzero_ps();
    c11 = _mm256_setzero_ps();
    c20 = _mm256_setzero_ps();
    c21 = _mm256_setzero_ps();
    c30 = _mm256_setzero_ps();
    c31 = _mm256_setzero_ps();
    c40 = _mm256_setzero_ps();
    c41 = _mm256_setzero_ps();
    c50 = _mm256_setzero_ps();
    c51 = _mm256_setzero_ps();
  } else {
    c00 = _mm256_loadu_ps(c);
    c01 = _mm256_loadu_ps(c + 8);
    c10 = _mm256_loadu_ps(c + rso);
    c11 = _mm256_loadu_ps(c + rso + 8);
    c20 = _mm256_loadu_ps(c + 2 * rso);
    c21 = _mm256_loadu_ps(c + 2 * rso + 8);
    c30 = _mm256_loadu_ps(c + 3 * rso);
    c31 = _mm256_loadu_ps(c + 3 * rso + 8);
    c40 = _mm256_loadu_ps(c + 4 * rso);
    c41 = _mm256_loadu_ps(c + 4 * rso + 8);
    c50 = _mm256_loadu_ps(c + 5 * rso);
    c51 = _mm256_loadu_ps(c + 5 * rso + 8);
  }

  const float *ap = a;
  const float *bp = b;
  size_t k_iter = kc / 8;
  size_t k_left = kc % 8;

  /* Pre-load first B vector pair (BLIS-style) */
  __m256 b0 = _mm256_loadu_ps(bp);
  __m256 b1 = _mm256_loadu_ps(bp + 8);

  for (size_t ki = 0; ki < k_iter; ki++) {
    GEMM_F32_K_ITER(ap, b0, b1);
    ap += csa;
    bp += rsb;
    b0 = _mm256_loadu_ps(bp);
    b1 = _mm256_loadu_ps(bp + 8);
    GEMM_F32_K_ITER(ap, b0, b1);
    ap += csa;
    bp += rsb;
    b0 = _mm256_loadu_ps(bp);
    b1 = _mm256_loadu_ps(bp + 8);
    GEMM_F32_K_ITER(ap, b0, b1);
    ap += csa;
    bp += rsb;
    b0 = _mm256_loadu_ps(bp);
    b1 = _mm256_loadu_ps(bp + 8);
    GEMM_F32_K_ITER(ap, b0, b1);
    ap += csa;
    bp += rsb;
    b0 = _mm256_loadu_ps(bp);
    b1 = _mm256_loadu_ps(bp + 8);
    GEMM_F32_K_ITER(ap, b0, b1);
    ap += csa;
    bp += rsb;
    b0 = _mm256_loadu_ps(bp);
    b1 = _mm256_loadu_ps(bp + 8);
    GEMM_F32_K_ITER(ap, b0, b1);
    ap += csa;
    bp += rsb;
    b0 = _mm256_loadu_ps(bp);
    b1 = _mm256_loadu_ps(bp + 8);
    GEMM_F32_K_ITER(ap, b0, b1);
    ap += csa;
    bp += rsb;
    b0 = _mm256_loadu_ps(bp);
    b1 = _mm256_loadu_ps(bp + 8);
    GEMM_F32_K_ITER(ap, b0, b1);
    ap += csa;
    bp += rsb;
    b0 = _mm256_loadu_ps(bp);
    b1 = _mm256_loadu_ps(bp + 8);
  }
  for (size_t ki = 0; ki < k_left; ki++) {
    GEMM_F32_K_ITER(ap, b0, b1);
    ap += csa;
    bp += rsb;
    b0 = _mm256_loadu_ps(bp);
    b1 = _mm256_loadu_ps(bp + 8);
  }

  /* Refresh C prefetch — lines may have been evicted during K-loop */
  _mm_prefetch((const char *)(c), _MM_HINT_T0);
  _mm_prefetch((const char *)(c + rso), _MM_HINT_T0);
  _mm_prefetch((const char *)(c + 2 * rso), _MM_HINT_T0);
  _mm_prefetch((const char *)(c + 3 * rso), _MM_HINT_T0);
  _mm_prefetch((const char *)(c + 4 * rso), _MM_HINT_T0);
  _mm_prefetch((const char *)(c + 5 * rso), _MM_HINT_T0);

  _mm256_storeu_ps(c, c00);
  _mm256_storeu_ps(c + 8, c01);
  _mm256_storeu_ps(c + rso, c10);
  _mm256_storeu_ps(c + rso + 8, c11);
  _mm256_storeu_ps(c + 2 * rso, c20);
  _mm256_storeu_ps(c + 2 * rso + 8, c21);
  _mm256_storeu_ps(c + 3 * rso, c30);
  _mm256_storeu_ps(c + 3 * rso + 8, c31);
  _mm256_storeu_ps(c + 4 * rso, c40);
  _mm256_storeu_ps(c + 4 * rso + 8, c41);
  _mm256_storeu_ps(c + 5 * rso, c50);
  _mm256_storeu_ps(c + 5 * rso + 8, c51);
}

#endif /* __GNUC__ || __clang__ */

#undef GEMM_F32_K_ITER

static inline void gemm_edge_f32(const float *a, const float *b, float *c,
                                 size_t mr, size_t nr, size_t kc, intptr_t rsa,
                                 intptr_t csa, intptr_t rsb, intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
    for (size_t p = 0; p < kc; p++) {
      float aip = a[i * rsa + p * csa];
      __m256 va = _mm256_set1_ps(aip);
      const float *brow = b + p * rsb;
      float *crow = c + i * rso;
      size_t j = 0;
      for (; j + 8 <= nr; j += 8) {
        __m256 vo = _mm256_loadu_ps(crow + j);
        vo = _mm256_fmadd_ps(va, _mm256_loadu_ps(brow + j), vo);
        _mm256_storeu_ps(crow + j, vo);
      }
      for (; j < nr; j++)
        crow[j] += aip * brow[j];
    }
  }
}

static inline void gemm_f32_avx2(const float *a, const float *b, float *out,
                                 size_t m_dim, size_t k_dim, size_t n_dim,
                                 intptr_t rsa, intptr_t csa, intptr_t rsb,
                                 intptr_t rso) {
  size_t nc_max = GEMM_MIN(GEMM_F32_NC, n_dim);
  float *packed_b = (float *)numc_malloc(
      32, GEMM_F32_KC * (nc_max + GEMM_F32_NR) * sizeof(float));
  if (!packed_b)
    return;

  for (size_t jc = 0; jc < n_dim; jc += GEMM_F32_NC) {
    size_t nc = GEMM_MIN(GEMM_F32_NC, n_dim - jc);
    size_t n_jr = (nc + GEMM_F32_NR - 1) / GEMM_F32_NR;

#ifdef _OPENMP
#pragma omp parallel if ((uint64_t)m_dim * k_dim * nc > GEMM_OMP_THRESHOLD)
    {
      NUMC_ALIGNAS(32) float packed_a[GEMM_F32_MC * GEMM_F32_KC];

      for (size_t pc = 0; pc < k_dim; pc += GEMM_F32_KC) {
        size_t kc = GEMM_MIN(GEMM_F32_KC, k_dim - pc);
        int first = (pc == 0);
        size_t last_ic = (size_t)-1;

#pragma omp for schedule(static)
        for (size_t jr_idx = 0; jr_idx < n_jr; jr_idx++) {
          size_t jj = jr_idx * GEMM_F32_NR;
          size_t nr_pack = GEMM_MIN(GEMM_F32_NR, nc - jj);
          _gemm_pack_b_strip_f32(b + pc * rsb + (jc + jj), packed_b + jj * kc,
                                 kc, nr_pack, rsb);
        }

        size_t n_ic = (m_dim + GEMM_F32_MC - 1) / GEMM_F32_MC;
        size_t n_tasks = n_ic * n_jr;

#pragma omp for schedule(static)
        for (size_t task = 0; task < n_tasks; task++) {
          size_t ic = (task / n_jr) * GEMM_F32_MC;
          size_t jr = (task % n_jr) * GEMM_F32_NR;
          size_t mc = GEMM_MIN(GEMM_F32_MC, m_dim - ic);
          size_t nr_cur = GEMM_MIN(GEMM_F32_NR, nc - jr);

          if (ic != last_ic) {
            gemm_pack_a_f32(a + ic * rsa + pc * csa, packed_a, mc, kc, rsa,
                            csa);
            last_ic = ic;
          }

          for (size_t ir = 0; ir < mc; ir += GEMM_F32_MR) {
            size_t mr_cur = GEMM_MIN(GEMM_F32_MR, mc - ir);
            if (mr_cur == GEMM_F32_MR && nr_cur == GEMM_F32_NR) {
#if NUMC_HAVE_ASM_UKERNEL
              numc_gemm_ukernel_f32_6x16_avx2(
                  packed_a + ir * kc, packed_b + jr * kc,
                  out + (ic + ir) * rso + (jc + jr), (uint64_t)kc, (int64_t)rso,
                  first);
#else
              gemm_ukernel_f32_6x16(packed_a + ir * kc, packed_b + jr * kc,
                                    out + (ic + ir) * rso + (jc + jr), kc, 1,
                                    GEMM_F32_MR, GEMM_F32_NR, rso, first);
#endif
            } else {
              NUMC_ALIGNAS(32) float tmp[GEMM_F32_MR * GEMM_F32_NR];
              gemm_ukernel_f32_6x16(packed_a + ir * kc, packed_b + jr * kc, tmp,
                                    kc, 1, GEMM_F32_MR, GEMM_F32_NR,
                                    GEMM_F32_NR, 1);
              float *dst = out + (ic + ir) * rso + (jc + jr);
              if (first) {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] = tmp[ii * GEMM_F32_NR + jj];
              } else {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] += tmp[ii * GEMM_F32_NR + jj];
              }
            }
          }
        }
      }
    }
#else
    {
      NUMC_ALIGNAS(32) float packed_a[GEMM_F32_MC * GEMM_F32_KC];

      for (size_t pc = 0; pc < k_dim; pc += GEMM_F32_KC) {
        size_t kc = GEMM_MIN(GEMM_F32_KC, k_dim - pc);
        int first = (pc == 0);

        gemm_pack_b_f32(b + pc * rsb + jc, packed_b, kc, nc, rsb);

        size_t n_ic = (m_dim + GEMM_F32_MC - 1) / GEMM_F32_MC;
        size_t n_tasks = n_ic * n_jr;

        for (size_t task = 0; task < n_tasks; task++) {
          size_t ic = (task / n_jr) * GEMM_F32_MC;
          size_t jr = (task % n_jr) * GEMM_F32_NR;
          size_t mc = GEMM_MIN(GEMM_F32_MC, m_dim - ic);
          size_t nr_cur = GEMM_MIN(GEMM_F32_NR, nc - jr);

          if (task % n_jr == 0)
            gemm_pack_a_f32(a + ic * rsa + pc * csa, packed_a, mc, kc, rsa,
                            csa);

          for (size_t ir = 0; ir < mc; ir += GEMM_F32_MR) {
            size_t mr_cur = GEMM_MIN(GEMM_F32_MR, mc - ir);
            if (mr_cur == GEMM_F32_MR && nr_cur == GEMM_F32_NR) {
#if NUMC_HAVE_ASM_UKERNEL
              numc_gemm_ukernel_f32_6x16_avx2(
                  packed_a + ir * kc, packed_b + jr * kc,
                  out + (ic + ir) * rso + (jc + jr), (uint64_t)kc, (int64_t)rso,
                  first);
#else
              gemm_ukernel_f32_6x16(packed_a + ir * kc, packed_b + jr * kc,
                                    out + (ic + ir) * rso + (jc + jr), kc, 1,
                                    GEMM_F32_MR, GEMM_F32_NR, rso, first);
#endif
            } else {
              NUMC_ALIGNAS(32) float tmp[GEMM_F32_MR * GEMM_F32_NR];
              gemm_ukernel_f32_6x16(packed_a + ir * kc, packed_b + jr * kc, tmp,
                                    kc, 1, GEMM_F32_MR, GEMM_F32_NR,
                                    GEMM_F32_NR, 1);
              float *dst = out + (ic + ir) * rso + (jc + jr);
              if (first) {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] = tmp[ii * GEMM_F32_NR + jj];
              } else {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] += tmp[ii * GEMM_F32_NR + jj];
              }
            }
          }
        }
      }
    }
#endif
  }

  numc_free(packed_b);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Float64: 6×8 micro-kernel  (12 acc + 2 A broadcast + 2 B loads = 16 YMM)
   BLIS-style B pre-load: B vectors loaded at end of prev iter, ready for
   next iter. 2 alternating A broadcast registers for better ILP.
   ═══════════════════════════════════════════════════════════════════════════
 */

/* MSVC intrinsics K-iter: B pre-loaded in b0/b1 before call. */
#define GEMM_F64_K_ITER(ap, b0, b1)             \
  do {                                          \
    __m256d a0 = _mm256_broadcast_sd((ap) + 0); \
    __m256d a1 = _mm256_broadcast_sd((ap) + 1); \
    c00 = _mm256_fmadd_pd(a0, b0, c00);         \
    c01 = _mm256_fmadd_pd(a0, b1, c01);         \
    c10 = _mm256_fmadd_pd(a1, b0, c10);         \
    c11 = _mm256_fmadd_pd(a1, b1, c11);         \
    a0 = _mm256_broadcast_sd((ap) + 2);         \
    a1 = _mm256_broadcast_sd((ap) + 3);         \
    c20 = _mm256_fmadd_pd(a0, b0, c20);         \
    c21 = _mm256_fmadd_pd(a0, b1, c21);         \
    c30 = _mm256_fmadd_pd(a1, b0, c30);         \
    c31 = _mm256_fmadd_pd(a1, b1, c31);         \
    a0 = _mm256_broadcast_sd((ap) + 4);         \
    a1 = _mm256_broadcast_sd((ap) + 5);         \
    c40 = _mm256_fmadd_pd(a0, b0, c40);         \
    c41 = _mm256_fmadd_pd(a0, b1, c41);         \
    c50 = _mm256_fmadd_pd(a1, b0, c50);         \
    c51 = _mm256_fmadd_pd(a1, b1, c51);         \
  } while (0)

#if defined(__GNUC__) || defined(__clang__)

/* Inline-asm f64 6×8 micro-kernel: 12 accumulators pinned to ymm0-ymm11,
 * no register spills. Interleaves vbroadcastsd (port 5) with vfmadd231pd
 * (ports 0,1) for optimal port utilization. 8× unrolled K-loop with
 * interleaved A and B prefetching for L1 residency. */
static inline void gemm_ukernel_f64_6x8(const double *a, const double *b,
                                        double *c, size_t kc, intptr_t rsa,
                                        intptr_t csa, intptr_t rsb,
                                        intptr_t rso, int first) {
  (void)rsa;
  intptr_t csa_bytes = csa * (intptr_t)sizeof(double);
  intptr_t rsb_bytes = rsb * (intptr_t)sizeof(double);
  double *c1 = c + rso;
  double *c2 = c + 2 * rso;
  double *c3 = c + 3 * rso;
  double *c4 = c + 4 * rso;
  double *c5 = c + 5 * rso;

  const double *ap = a;
  const double *bp = b;

  /* BLIS-style K-iteration: B pre-loaded in ymm12/ymm13 from previous
   * iteration (or initial pre-load). Uses 2 A broadcast registers
   * (ymm14/ymm15) for better ILP — allows second broadcast to overlap
   * with first FMA pair. Pre-loads next B at end for next iteration. */
#define GEMM_F64_ASM_K_ITER                   \
  "vbroadcastsd (%[ap]), %%ymm14\n\t"         \
  "vbroadcastsd 8(%[ap]), %%ymm15\n\t"        \
  "vfmadd231pd %%ymm14, %%ymm12, %%ymm0\n\t"  \
  "vfmadd231pd %%ymm14, %%ymm13, %%ymm1\n\t"  \
  "vfmadd231pd %%ymm15, %%ymm12, %%ymm2\n\t"  \
  "vfmadd231pd %%ymm15, %%ymm13, %%ymm3\n\t"  \
  "vbroadcastsd 16(%[ap]), %%ymm14\n\t"       \
  "vbroadcastsd 24(%[ap]), %%ymm15\n\t"       \
  "vfmadd231pd %%ymm14, %%ymm12, %%ymm4\n\t"  \
  "vfmadd231pd %%ymm14, %%ymm13, %%ymm5\n\t"  \
  "vfmadd231pd %%ymm15, %%ymm12, %%ymm6\n\t"  \
  "vfmadd231pd %%ymm15, %%ymm13, %%ymm7\n\t"  \
  "vbroadcastsd 32(%[ap]), %%ymm14\n\t"       \
  "vbroadcastsd 40(%[ap]), %%ymm15\n\t"       \
  "vfmadd231pd %%ymm14, %%ymm12, %%ymm8\n\t"  \
  "vfmadd231pd %%ymm14, %%ymm13, %%ymm9\n\t"  \
  "vfmadd231pd %%ymm15, %%ymm12, %%ymm10\n\t" \
  "vfmadd231pd %%ymm15, %%ymm13, %%ymm11\n\t" \
  "add %[csa_bytes], %[ap]\n\t"               \
  "add %[rsb_bytes], %[bp]\n\t"               \
  "vmovupd (%[bp]), %%ymm12\n\t"              \
  "vmovupd 32(%[bp]), %%ymm13\n\t"

  __asm__ __volatile__(
      /* Prefetch C rows into L1 */
      "prefetcht0 (%[c0])\n\t"
      "prefetcht0 (%[c1])\n\t"
      "prefetcht0 (%[c2])\n\t"
      "prefetcht0 (%[c3])\n\t"
      "prefetcht0 (%[c4])\n\t"
      "prefetcht0 (%[c5])\n\t"

      /* Zero or load accumulators */
      "test %[first], %[first]\n\t"
      "jz 1f\n\t"
      /* first=1: zero all 12 accumulators */
      "vxorpd %%ymm0, %%ymm0, %%ymm0\n\t"
      "vxorpd %%ymm1, %%ymm1, %%ymm1\n\t"
      "vxorpd %%ymm2, %%ymm2, %%ymm2\n\t"
      "vxorpd %%ymm3, %%ymm3, %%ymm3\n\t"
      "vxorpd %%ymm4, %%ymm4, %%ymm4\n\t"
      "vxorpd %%ymm5, %%ymm5, %%ymm5\n\t"
      "vxorpd %%ymm6, %%ymm6, %%ymm6\n\t"
      "vxorpd %%ymm7, %%ymm7, %%ymm7\n\t"
      "vxorpd %%ymm8, %%ymm8, %%ymm8\n\t"
      "vxorpd %%ymm9, %%ymm9, %%ymm9\n\t"
      "vxorpd %%ymm10, %%ymm10, %%ymm10\n\t"
      "vxorpd %%ymm11, %%ymm11, %%ymm11\n\t"
      "jmp 2f\n\t"

      "1:\n\t"
      /* first=0: load existing C values */
      "vmovupd (%[c0]), %%ymm0\n\t"
      "vmovupd 32(%[c0]), %%ymm1\n\t"
      "vmovupd (%[c1]), %%ymm2\n\t"
      "vmovupd 32(%[c1]), %%ymm3\n\t"
      "vmovupd (%[c2]), %%ymm4\n\t"
      "vmovupd 32(%[c2]), %%ymm5\n\t"
      "vmovupd (%[c3]), %%ymm6\n\t"
      "vmovupd 32(%[c3]), %%ymm7\n\t"
      "vmovupd (%[c4]), %%ymm8\n\t"
      "vmovupd 32(%[c4]), %%ymm9\n\t"
      "vmovupd (%[c5]), %%ymm10\n\t"
      "vmovupd 32(%[c5]), %%ymm11\n\t"

      "2:\n\t"
      /* Pre-load first B vector pair (BLIS-style) */
      "vmovupd (%[bp]), %%ymm12\n\t"
      "vmovupd 32(%[bp]), %%ymm13\n\t"

      /* K-loop: k_iter = kc >> 3 (8× unrolled) */
      "mov %[kc], %%rcx\n\t"
      "shr $3, %%rcx\n\t"
      "jz 4f\n\t"

      ".p2align 5\n\t"
      "3:\n\t" GEMM_F64_ASM_K_ITER                     /* iter 0 */
      GEMM_F64_ASM_K_ITER                              /* iter 1 */
      "prefetcht0 768(%[ap])\n\t" GEMM_F64_ASM_K_ITER  /* iter 2 */
      GEMM_F64_ASM_K_ITER                              /* iter 3 */
      "prefetcht0 512(%[bp])\n\t" GEMM_F64_ASM_K_ITER  /* iter 4 */
      GEMM_F64_ASM_K_ITER                              /* iter 5 */
      "prefetcht0 1536(%[ap])\n\t" GEMM_F64_ASM_K_ITER /* iter 6 */
      GEMM_F64_ASM_K_ITER                              /* iter 7 */

      "dec %%rcx\n\t"
      "jnz 3b\n\t"

      "4:\n\t"
      /* Handle k_left = kc & 7 */
      "mov %[kc], %%rcx\n\t"
      "and $7, %%rcx\n\t"
      "jz 6f\n\t"

      ".p2align 5\n\t"
      "5:\n\t" GEMM_F64_ASM_K_ITER "dec %%rcx\n\t"
      "jnz 5b\n\t"

      "6:\n\t"
      /* Refresh C prefetch — lines may have been evicted during
       * K-loop (A+B data exceeds L1 for large K) */
      "prefetcht0 (%[c0])\n\t"
      "prefetcht0 (%[c1])\n\t"
      "prefetcht0 (%[c2])\n\t"
      "prefetcht0 (%[c3])\n\t"
      "prefetcht0 (%[c4])\n\t"
      "prefetcht0 (%[c5])\n\t"
      /* Store accumulators to C */
      "vmovupd %%ymm0, (%[c0])\n\t"
      "vmovupd %%ymm1, 32(%[c0])\n\t"
      "vmovupd %%ymm2, (%[c1])\n\t"
      "vmovupd %%ymm3, 32(%[c1])\n\t"
      "vmovupd %%ymm4, (%[c2])\n\t"
      "vmovupd %%ymm5, 32(%[c2])\n\t"
      "vmovupd %%ymm6, (%[c3])\n\t"
      "vmovupd %%ymm7, 32(%[c3])\n\t"
      "vmovupd %%ymm8, (%[c4])\n\t"
      "vmovupd %%ymm9, 32(%[c4])\n\t"
      "vmovupd %%ymm10, (%[c5])\n\t"
      "vmovupd %%ymm11, 32(%[c5])\n\t"

      : [ap] "+r"(ap), [bp] "+r"(bp)
      : [c0] "r"(c), [c1] "r"(c1), [c2] "r"(c2), [c3] "r"(c3), [c4] "r"(c4),
        [c5] "r"(c5), [kc] "r"((uint64_t)kc), [first] "r"(first),
        [csa_bytes] "r"(csa_bytes), [rsb_bytes] "r"(rsb_bytes)
      : "rcx", "memory", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6",
        "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14",
        "ymm15");

#undef GEMM_F64_ASM_K_ITER
}

#else /* MSVC fallback — C intrinsics version */

static inline void gemm_ukernel_f64_6x8(const double *a, const double *b,
                                        double *c, size_t kc, intptr_t rsa,
                                        intptr_t csa, intptr_t rsb,
                                        intptr_t rso, int first) {
  __m256d c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;

  _mm_prefetch((const char *)(c), _MM_HINT_T0);
  _mm_prefetch((const char *)(c + rso), _MM_HINT_T0);
  _mm_prefetch((const char *)(c + 2 * rso), _MM_HINT_T0);
  _mm_prefetch((const char *)(c + 3 * rso), _MM_HINT_T0);
  _mm_prefetch((const char *)(c + 4 * rso), _MM_HINT_T0);
  _mm_prefetch((const char *)(c + 5 * rso), _MM_HINT_T0);

  if (first) {
    c00 = _mm256_setzero_pd();
    c01 = _mm256_setzero_pd();
    c10 = _mm256_setzero_pd();
    c11 = _mm256_setzero_pd();
    c20 = _mm256_setzero_pd();
    c21 = _mm256_setzero_pd();
    c30 = _mm256_setzero_pd();
    c31 = _mm256_setzero_pd();
    c40 = _mm256_setzero_pd();
    c41 = _mm256_setzero_pd();
    c50 = _mm256_setzero_pd();
    c51 = _mm256_setzero_pd();
  } else {
    c00 = _mm256_loadu_pd(c);
    c01 = _mm256_loadu_pd(c + 4);
    c10 = _mm256_loadu_pd(c + rso);
    c11 = _mm256_loadu_pd(c + rso + 4);
    c20 = _mm256_loadu_pd(c + 2 * rso);
    c21 = _mm256_loadu_pd(c + 2 * rso + 4);
    c30 = _mm256_loadu_pd(c + 3 * rso);
    c31 = _mm256_loadu_pd(c + 3 * rso + 4);
    c40 = _mm256_loadu_pd(c + 4 * rso);
    c41 = _mm256_loadu_pd(c + 4 * rso + 4);
    c50 = _mm256_loadu_pd(c + 5 * rso);
    c51 = _mm256_loadu_pd(c + 5 * rso + 4);
  }

  const double *ap = a;
  const double *bp = b;
  size_t k_iter = kc / 8;
  size_t k_left = kc % 8;

  /* Pre-load first B vector pair (BLIS-style) */
  __m256d b0 = _mm256_loadu_pd(bp);
  __m256d b1 = _mm256_loadu_pd(bp + 4);

  for (size_t ki = 0; ki < k_iter; ki++) {
    GEMM_F64_K_ITER(ap, b0, b1);
    ap += csa;
    bp += rsb;
    b0 = _mm256_loadu_pd(bp);
    b1 = _mm256_loadu_pd(bp + 4);
    GEMM_F64_K_ITER(ap, b0, b1);
    ap += csa;
    bp += rsb;
    b0 = _mm256_loadu_pd(bp);
    b1 = _mm256_loadu_pd(bp + 4);
    GEMM_F64_K_ITER(ap, b0, b1);
    ap += csa;
    bp += rsb;
    b0 = _mm256_loadu_pd(bp);
    b1 = _mm256_loadu_pd(bp + 4);
    GEMM_F64_K_ITER(ap, b0, b1);
    ap += csa;
    bp += rsb;
    b0 = _mm256_loadu_pd(bp);
    b1 = _mm256_loadu_pd(bp + 4);
    GEMM_F64_K_ITER(ap, b0, b1);
    ap += csa;
    bp += rsb;
    b0 = _mm256_loadu_pd(bp);
    b1 = _mm256_loadu_pd(bp + 4);
    GEMM_F64_K_ITER(ap, b0, b1);
    ap += csa;
    bp += rsb;
    b0 = _mm256_loadu_pd(bp);
    b1 = _mm256_loadu_pd(bp + 4);
    GEMM_F64_K_ITER(ap, b0, b1);
    ap += csa;
    bp += rsb;
    b0 = _mm256_loadu_pd(bp);
    b1 = _mm256_loadu_pd(bp + 4);
    GEMM_F64_K_ITER(ap, b0, b1);
    ap += csa;
    bp += rsb;
    b0 = _mm256_loadu_pd(bp);
    b1 = _mm256_loadu_pd(bp + 4);
  }
  for (size_t ki = 0; ki < k_left; ki++) {
    GEMM_F64_K_ITER(ap, b0, b1);
    ap += csa;
    bp += rsb;
    b0 = _mm256_loadu_pd(bp);
    b1 = _mm256_loadu_pd(bp + 4);
  }

  /* Refresh C prefetch — lines may have been evicted during K-loop */
  _mm_prefetch((const char *)(c), _MM_HINT_T0);
  _mm_prefetch((const char *)(c + rso), _MM_HINT_T0);
  _mm_prefetch((const char *)(c + 2 * rso), _MM_HINT_T0);
  _mm_prefetch((const char *)(c + 3 * rso), _MM_HINT_T0);
  _mm_prefetch((const char *)(c + 4 * rso), _MM_HINT_T0);
  _mm_prefetch((const char *)(c + 5 * rso), _MM_HINT_T0);

  _mm256_storeu_pd(c, c00);
  _mm256_storeu_pd(c + 4, c01);
  _mm256_storeu_pd(c + rso, c10);
  _mm256_storeu_pd(c + rso + 4, c11);
  _mm256_storeu_pd(c + 2 * rso, c20);
  _mm256_storeu_pd(c + 2 * rso + 4, c21);
  _mm256_storeu_pd(c + 3 * rso, c30);
  _mm256_storeu_pd(c + 3 * rso + 4, c31);
  _mm256_storeu_pd(c + 4 * rso, c40);
  _mm256_storeu_pd(c + 4 * rso + 4, c41);
  _mm256_storeu_pd(c + 5 * rso, c50);
  _mm256_storeu_pd(c + 5 * rso + 4, c51);
}

#endif /* __GNUC__ || __clang__ */

#undef GEMM_F64_K_ITER

static inline void gemm_edge_f64(const double *a, const double *b, double *c,
                                 size_t mr, size_t nr, size_t kc, intptr_t rsa,
                                 intptr_t csa, intptr_t rsb, intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
    for (size_t p = 0; p < kc; p++) {
      double aip = a[i * rsa + p * csa];
      __m256d va = _mm256_set1_pd(aip);
      const double *brow = b + p * rsb;
      double *crow = c + i * rso;
      size_t j = 0;
      for (; j + 4 <= nr; j += 4) {
        __m256d vo = _mm256_loadu_pd(crow + j);
        vo = _mm256_fmadd_pd(va, _mm256_loadu_pd(brow + j), vo);
        _mm256_storeu_pd(crow + j, vo);
      }
      for (; j < nr; j++)
        crow[j] += aip * brow[j];
    }
  }
}

static inline void gemm_f64_avx2(const double *a, const double *b, double *out,
                                 size_t m_dim, size_t k_dim, size_t n_dim,
                                 intptr_t rsa, intptr_t csa, intptr_t rsb,
                                 intptr_t rso) {
  size_t nc_max = GEMM_MIN(GEMM_F64_NC, n_dim);
  double *packed_b = (double *)numc_malloc(
      32, GEMM_F64_KC * (nc_max + GEMM_F64_NR) * sizeof(double));
  if (!packed_b)
    return;

  for (size_t jc = 0; jc < n_dim; jc += GEMM_F64_NC) {
    size_t nc = GEMM_MIN(GEMM_F64_NC, n_dim - jc);
    size_t n_jr = (nc + GEMM_F64_NR - 1) / GEMM_F64_NR;

#ifdef _OPENMP
#pragma omp parallel if ((uint64_t)m_dim * k_dim * nc > GEMM_OMP_THRESHOLD)
    {
      NUMC_ALIGNAS(32) double packed_a[GEMM_F64_MC * GEMM_F64_KC];

      for (size_t pc = 0; pc < k_dim; pc += GEMM_F64_KC) {
        size_t kc = GEMM_MIN(GEMM_F64_KC, k_dim - pc);
        int first = (pc == 0);
        size_t last_ic = (size_t)-1;

#pragma omp for schedule(static)
        for (size_t jr_idx = 0; jr_idx < n_jr; jr_idx++) {
          size_t jj = jr_idx * GEMM_F64_NR;
          size_t nr_pack = GEMM_MIN(GEMM_F64_NR, nc - jj);
          _gemm_pack_b_strip_f64(b + pc * rsb + (jc + jj), packed_b + jj * kc,
                                 kc, nr_pack, rsb);
        }

        size_t n_ic = (m_dim + GEMM_F64_MC - 1) / GEMM_F64_MC;
        size_t n_tasks = n_ic * n_jr;

#pragma omp for schedule(static)
        for (size_t task = 0; task < n_tasks; task++) {
          size_t ic = (task / n_jr) * GEMM_F64_MC;
          size_t jr = (task % n_jr) * GEMM_F64_NR;
          size_t mc = GEMM_MIN(GEMM_F64_MC, m_dim - ic);
          size_t nr_cur = GEMM_MIN(GEMM_F64_NR, nc - jr);

          if (ic != last_ic) {
            gemm_pack_a_f64(a + ic * rsa + pc * csa, packed_a, mc, kc, rsa,
                            csa);
            last_ic = ic;
          }

          for (size_t ir = 0; ir < mc; ir += GEMM_F64_MR) {
            size_t mr_cur = GEMM_MIN(GEMM_F64_MR, mc - ir);
            if (mr_cur == GEMM_F64_MR && nr_cur == GEMM_F64_NR) {
#if NUMC_HAVE_ASM_UKERNEL
              numc_gemm_ukernel_f64_6x8_avx2(packed_a + ir * kc,
                                             packed_b + jr * kc,
                                             out + (ic + ir) * rso + (jc + jr),
                                             (uint64_t)kc, (int64_t)rso, first);
#else
              gemm_ukernel_f64_6x8(packed_a + ir * kc, packed_b + jr * kc,
                                   out + (ic + ir) * rso + (jc + jr), kc, 1,
                                   GEMM_F64_MR, GEMM_F64_NR, rso, first);
#endif
            } else {
              NUMC_ALIGNAS(32) double tmp[GEMM_F64_MR * GEMM_F64_NR];
              gemm_ukernel_f64_6x8(packed_a + ir * kc, packed_b + jr * kc, tmp,
                                   kc, 1, GEMM_F64_MR, GEMM_F64_NR, GEMM_F64_NR,
                                   1);
              double *dst = out + (ic + ir) * rso + (jc + jr);
              if (first) {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] = tmp[ii * GEMM_F64_NR + jj];
              } else {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] += tmp[ii * GEMM_F64_NR + jj];
              }
            }
          }
        }
      }
    }
#else
    {
      NUMC_ALIGNAS(32) double packed_a[GEMM_F64_MC * GEMM_F64_KC];

      for (size_t pc = 0; pc < k_dim; pc += GEMM_F64_KC) {
        size_t kc = GEMM_MIN(GEMM_F64_KC, k_dim - pc);
        int first = (pc == 0);

        gemm_pack_b_f64(b + pc * rsb + jc, packed_b, kc, nc, rsb);

        size_t n_ic = (m_dim + GEMM_F64_MC - 1) / GEMM_F64_MC;
        size_t n_tasks = n_ic * n_jr;

        for (size_t task = 0; task < n_tasks; task++) {
          size_t ic = (task / n_jr) * GEMM_F64_MC;
          size_t jr = (task % n_jr) * GEMM_F64_NR;
          size_t mc = GEMM_MIN(GEMM_F64_MC, m_dim - ic);
          size_t nr_cur = GEMM_MIN(GEMM_F64_NR, nc - jr);

          if (task % n_jr == 0)
            gemm_pack_a_f64(a + ic * rsa + pc * csa, packed_a, mc, kc, rsa,
                            csa);

          for (size_t ir = 0; ir < mc; ir += GEMM_F64_MR) {
            size_t mr_cur = GEMM_MIN(GEMM_F64_MR, mc - ir);
            if (mr_cur == GEMM_F64_MR && nr_cur == GEMM_F64_NR) {
#if NUMC_HAVE_ASM_UKERNEL
              numc_gemm_ukernel_f64_6x8_avx2(packed_a + ir * kc,
                                             packed_b + jr * kc,
                                             out + (ic + ir) * rso + (jc + jr),
                                             (uint64_t)kc, (int64_t)rso, first);
#else
              gemm_ukernel_f64_6x8(packed_a + ir * kc, packed_b + jr * kc,
                                   out + (ic + ir) * rso + (jc + jr), kc, 1,
                                   GEMM_F64_MR, GEMM_F64_NR, rso, first);
#endif
            } else {
              NUMC_ALIGNAS(32) double tmp[GEMM_F64_MR * GEMM_F64_NR];
              gemm_ukernel_f64_6x8(packed_a + ir * kc, packed_b + jr * kc, tmp,
                                   kc, 1, GEMM_F64_MR, GEMM_F64_NR, GEMM_F64_NR,
                                   1);
              double *dst = out + (ic + ir) * rso + (jc + jr);
              if (first) {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] = tmp[ii * GEMM_F64_NR + jj];
              } else {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] += tmp[ii * GEMM_F64_NR + jj];
              }
            }
          }
        }
      }
    }
#endif
  }

  numc_free(packed_b);
}

/* ── Int32 packing routines ───────────────────────────────────────────── */

/* Pack B[kc × nc] into NR-wide micropanels (row-panel format).
 * Layout: for each NR-panel at column jr, kc rows of NR contiguous elements.
 * Micro-kernel sees rsb = NR. */
static inline void gemm_pack_b_i32(const int32_t *b, int32_t *packed, size_t kc,
                                   size_t nc, intptr_t rsb) {
  size_t jr = 0;
  for (; jr + GEMM_I32_NR <= nc; jr += GEMM_I32_NR) {
    int32_t *dest = packed + jr * kc;
    for (size_t p = 0; p < kc; p++) {
      const int32_t *src = b + p * rsb + jr;
      _mm256_storeu_si256((__m256i *)(dest + p * GEMM_I32_NR),
                          _mm256_loadu_si256((__m256i *)src));
      _mm256_storeu_si256((__m256i *)(dest + p * GEMM_I32_NR + 8),
                          _mm256_loadu_si256((__m256i *)(src + 8)));
    }
  }
  if (jr < nc) {
    int32_t *dest = packed + jr * kc;
    size_t rem = nc - jr;
    __m256i z = _mm256_setzero_si256();
    __m256i m0 = _gemm_mask_i32_lanes(rem < 8 ? rem : 8);
    __m256i m1 = _gemm_mask_i32_lanes(rem > 8 ? rem - 8 : 0);
    for (size_t p = 0; p < kc; p++) {
      const int32_t *src = b + p * rsb + jr;
      int32_t *d = dest + p * GEMM_I32_NR;
      _mm256_storeu_si256((__m256i *)d,
                          _mm256_maskload_epi32(src, m0));
      if (rem > 8) {
        _mm256_storeu_si256((__m256i *)(d + 8),
                            _mm256_maskload_epi32(src + 8, m1));
      } else {
        _mm256_storeu_si256((__m256i *)(d + 8), z);
      }
    }
  }
}

/* Pack a single NR-wide B strip for parallel packing. */
static inline void _gemm_pack_b_strip_i32(const int32_t *b, int32_t *dest,
                                           size_t kc, size_t nr_pack,
                                           intptr_t rsb) {
  if (nr_pack == GEMM_I32_NR) {
    for (size_t p = 0; p < kc; p++) {
      const int32_t *src = b + p * rsb;
      _mm256_storeu_si256((__m256i *)(dest + p * GEMM_I32_NR),
                          _mm256_loadu_si256((__m256i *)src));
      _mm256_storeu_si256((__m256i *)(dest + p * GEMM_I32_NR + 8),
                          _mm256_loadu_si256((__m256i *)(src + 8)));
    }
  } else {
    __m256i z = _mm256_setzero_si256();
    __m256i m0 = _gemm_mask_i32_lanes(nr_pack < 8 ? nr_pack : 8);
    __m256i m1 = _gemm_mask_i32_lanes(nr_pack > 8 ? nr_pack - 8 : 0);
    for (size_t p = 0; p < kc; p++) {
      const int32_t *src = b + p * rsb;
      int32_t *d = dest + p * GEMM_I32_NR;
      _mm256_storeu_si256((__m256i *)d,
                          _mm256_maskload_epi32(src, m0));
      if (nr_pack > 8) {
        _mm256_storeu_si256((__m256i *)(d + 8),
                            _mm256_maskload_epi32(src + 8, m1));
      } else {
        _mm256_storeu_si256((__m256i *)(d + 8), z);
      }
    }
  }
}

/* Pack A[mc × kc] into MR-tall micropanels (column-panel format).
 * Layout: for each MR-panel at row ir, kc columns of MR contiguous elements.
 * Micro-kernel sees rsa = 1, csa = MR. */
static inline void gemm_pack_a_i32(const int32_t *a, int32_t *packed, size_t mc,
                                   size_t kc, intptr_t rsa, intptr_t csa) {
  size_t ir = 0;
  /* Fast path: csa=1 (row-major A) — gather MR rows with SIMD transpose */
  if (csa == 1) {
    for (; ir + GEMM_I32_MR <= mc; ir += GEMM_I32_MR) {
      int32_t *dest = packed + ir * kc;
      const int32_t *r0 = a + (ir + 0) * rsa;
      const int32_t *r1 = a + (ir + 1) * rsa;
      const int32_t *r2 = a + (ir + 2) * rsa;
      const int32_t *r3 = a + (ir + 3) * rsa;
      const int32_t *r4 = a + (ir + 4) * rsa;
      const int32_t *r5 = a + (ir + 5) * rsa;
      /* Use f32 transpose path — bit patterns are identical for 32-bit types */
      size_t p = 0;
      for (; p + 8 <= kc; p += 8) {
        __m256 v0 = _mm256_loadu_ps((const float *)(r0 + p));
        __m256 v1 = _mm256_loadu_ps((const float *)(r1 + p));
        __m256 v2 = _mm256_loadu_ps((const float *)(r2 + p));
        __m256 v3 = _mm256_loadu_ps((const float *)(r3 + p));
        __m256 v4 = _mm256_loadu_ps((const float *)(r4 + p));
        __m256 v5 = _mm256_loadu_ps((const float *)(r5 + p));
        __m256 t0 = _mm256_unpacklo_ps(v0, v1);
        __m256 t1 = _mm256_unpackhi_ps(v0, v1);
        __m256 t2 = _mm256_unpacklo_ps(v2, v3);
        __m256 t3 = _mm256_unpackhi_ps(v2, v3);
        __m128 t0lo = _mm256_castps256_ps128(t0),
               t0hi = _mm256_extractf128_ps(t0, 1);
        __m128 t1lo = _mm256_castps256_ps128(t1),
               t1hi = _mm256_extractf128_ps(t1, 1);
        __m128 t2lo = _mm256_castps256_ps128(t2),
               t2hi = _mm256_extractf128_ps(t2, 1);
        __m128 t3lo = _mm256_castps256_ps128(t3),
               t3hi = _mm256_extractf128_ps(t3, 1);
        __m128 c0 = _mm_shuffle_ps(t0lo, t2lo, 0x44);
        __m128 c1 = _mm_shuffle_ps(t0lo, t2lo, 0xEE);
        __m128 c2 = _mm_shuffle_ps(t1lo, t3lo, 0x44);
        __m128 c3 = _mm_shuffle_ps(t1lo, t3lo, 0xEE);
        __m128 c4 = _mm_shuffle_ps(t0hi, t2hi, 0x44);
        __m128 c5 = _mm_shuffle_ps(t0hi, t2hi, 0xEE);
        __m128 c6 = _mm_shuffle_ps(t1hi, t3hi, 0x44);
        __m128 c7 = _mm_shuffle_ps(t1hi, t3hi, 0xEE);
        __m256 u_lo = _mm256_unpacklo_ps(v4, v5);
        __m256 u_hi = _mm256_unpackhi_ps(v4, v5);
        __m128d u_lo_0 = _mm_castps_pd(_mm256_castps256_ps128(u_lo));
        __m128d u_lo_4 = _mm_castps_pd(_mm256_extractf128_ps(u_lo, 1));
        __m128d u_hi_0 = _mm_castps_pd(_mm256_castps256_ps128(u_hi));
        __m128d u_hi_4 = _mm_castps_pd(_mm256_extractf128_ps(u_hi, 1));
        float *d = (float *)(dest + p * GEMM_I32_MR);
        _mm_storeu_ps(d + 0 * 6, c0);
        _mm_storel_pd((double *)(d + 0 * 6 + 4), u_lo_0);
        _mm_storeu_ps(d + 1 * 6, c1);
        _mm_storeh_pd((double *)(d + 1 * 6 + 4), u_lo_0);
        _mm_storeu_ps(d + 2 * 6, c2);
        _mm_storel_pd((double *)(d + 2 * 6 + 4), u_hi_0);
        _mm_storeu_ps(d + 3 * 6, c3);
        _mm_storeh_pd((double *)(d + 3 * 6 + 4), u_hi_0);
        _mm_storeu_ps(d + 4 * 6, c4);
        _mm_storel_pd((double *)(d + 4 * 6 + 4), u_lo_4);
        _mm_storeu_ps(d + 5 * 6, c5);
        _mm_storeh_pd((double *)(d + 5 * 6 + 4), u_lo_4);
        _mm_storeu_ps(d + 6 * 6, c6);
        _mm_storel_pd((double *)(d + 6 * 6 + 4), u_hi_4);
        _mm_storeu_ps(d + 7 * 6, c7);
        _mm_storeh_pd((double *)(d + 7 * 6 + 4), u_hi_4);
      }
      for (; p < kc; p++) {
        int32_t *d = dest + p * GEMM_I32_MR;
        d[0] = r0[p];
        d[1] = r1[p];
        d[2] = r2[p];
        d[3] = r3[p];
        d[4] = r4[p];
        d[5] = r5[p];
      }
    }
    if (ir < mc) {
      int32_t *dest = packed + ir * kc;
      size_t rem = mc - ir;
      for (size_t p = 0; p < kc; p++) {
        int32_t *d = dest + p * GEMM_I32_MR;
        d[0] = rem > 0 ? a[(ir + 0) * rsa + p] : 0;
        d[1] = rem > 1 ? a[(ir + 1) * rsa + p] : 0;
        d[2] = rem > 2 ? a[(ir + 2) * rsa + p] : 0;
        d[3] = rem > 3 ? a[(ir + 3) * rsa + p] : 0;
        d[4] = rem > 4 ? a[(ir + 4) * rsa + p] : 0;
        d[5] = rem > 5 ? a[(ir + 5) * rsa + p] : 0;
      }
    }
    return;
  }
  for (; ir + GEMM_I32_MR <= mc; ir += GEMM_I32_MR) {
    int32_t *dest = packed + ir * kc;
    for (size_t p = 0; p < kc; p++) {
      for (size_t i = 0; i < GEMM_I32_MR; i++)
        dest[p * GEMM_I32_MR + i] = a[(ir + i) * rsa + p * csa];
    }
  }
  if (ir < mc) {
    int32_t *dest = packed + ir * kc;
    size_t rem = mc - ir;
    for (size_t p = 0; p < kc; p++) {
      int32_t *d = dest + p * GEMM_I32_MR;
      d[0] = rem > 0 ? a[(ir + 0) * rsa + p * csa] : 0;
      d[1] = rem > 1 ? a[(ir + 1) * rsa + p * csa] : 0;
      d[2] = rem > 2 ? a[(ir + 2) * rsa + p * csa] : 0;
      d[3] = rem > 3 ? a[(ir + 3) * rsa + p * csa] : 0;
      d[4] = rem > 4 ? a[(ir + 4) * rsa + p * csa] : 0;
      d[5] = rem > 5 ? a[(ir + 5) * rsa + p * csa] : 0;
    }
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   Int32/Uint32: 6×16 micro-kernel  (mullo_epi32 + add_epi32)
   mullo_epi32 produces identical low 32 bits for signed and unsigned.
   ═══════════════════════════════════════════════════════════════════════════
 */

static inline void gemm_ukernel_i32_6x16(const int32_t *a, const int32_t *b,
                                         int32_t *c, size_t kc, intptr_t rsa,
                                         intptr_t csa, intptr_t rsb,
                                         intptr_t rso, int first) {
  __m256i c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
  if (first) {
    c00 = c01 = c10 = c11 = c20 = c21 = _mm256_setzero_si256();
    c30 = c31 = c40 = c41 = c50 = c51 = _mm256_setzero_si256();
  } else {
    c00 = _mm256_loadu_si256((__m256i *)(c));
    c01 = _mm256_loadu_si256((__m256i *)(c + 8));
    c10 = _mm256_loadu_si256((__m256i *)(c + rso));
    c11 = _mm256_loadu_si256((__m256i *)(c + rso + 8));
    c20 = _mm256_loadu_si256((__m256i *)(c + 2 * rso));
    c21 = _mm256_loadu_si256((__m256i *)(c + 2 * rso + 8));
    c30 = _mm256_loadu_si256((__m256i *)(c + 3 * rso));
    c31 = _mm256_loadu_si256((__m256i *)(c + 3 * rso + 8));
    c40 = _mm256_loadu_si256((__m256i *)(c + 4 * rso));
    c41 = _mm256_loadu_si256((__m256i *)(c + 4 * rso + 8));
    c50 = _mm256_loadu_si256((__m256i *)(c + 5 * rso));
    c51 = _mm256_loadu_si256((__m256i *)(c + 5 * rso + 8));
  }

  for (size_t p = 0; p < kc; p++) {
    const int32_t *bp = b + p * rsb;
    _mm_prefetch((const char *)(bp + rsb), _MM_HINT_T0);
    _mm_prefetch((const char *)(bp + rsb + 8), _MM_HINT_T0);
    __m256i b0 = _mm256_loadu_si256((__m256i *)bp),
            b1 = _mm256_loadu_si256((__m256i *)(bp + 8));
    __m256i av;
    av = _mm256_set1_epi32(a[0 * rsa + p * csa]);
    c00 = _mm256_add_epi32(c00, _mm256_mullo_epi32(av, b0));
    c01 = _mm256_add_epi32(c01, _mm256_mullo_epi32(av, b1));
    av = _mm256_set1_epi32(a[1 * rsa + p * csa]);
    c10 = _mm256_add_epi32(c10, _mm256_mullo_epi32(av, b0));
    c11 = _mm256_add_epi32(c11, _mm256_mullo_epi32(av, b1));
    av = _mm256_set1_epi32(a[2 * rsa + p * csa]);
    c20 = _mm256_add_epi32(c20, _mm256_mullo_epi32(av, b0));
    c21 = _mm256_add_epi32(c21, _mm256_mullo_epi32(av, b1));
    av = _mm256_set1_epi32(a[3 * rsa + p * csa]);
    c30 = _mm256_add_epi32(c30, _mm256_mullo_epi32(av, b0));
    c31 = _mm256_add_epi32(c31, _mm256_mullo_epi32(av, b1));
    av = _mm256_set1_epi32(a[4 * rsa + p * csa]);
    c40 = _mm256_add_epi32(c40, _mm256_mullo_epi32(av, b0));
    c41 = _mm256_add_epi32(c41, _mm256_mullo_epi32(av, b1));
    av = _mm256_set1_epi32(a[5 * rsa + p * csa]);
    c50 = _mm256_add_epi32(c50, _mm256_mullo_epi32(av, b0));
    c51 = _mm256_add_epi32(c51, _mm256_mullo_epi32(av, b1));
  }

  _mm256_storeu_si256((__m256i *)(c), c00);
  _mm256_storeu_si256((__m256i *)(c + 8), c01);
  _mm256_storeu_si256((__m256i *)(c + rso), c10);
  _mm256_storeu_si256((__m256i *)(c + rso + 8), c11);
  _mm256_storeu_si256((__m256i *)(c + 2 * rso), c20);
  _mm256_storeu_si256((__m256i *)(c + 2 * rso + 8), c21);
  _mm256_storeu_si256((__m256i *)(c + 3 * rso), c30);
  _mm256_storeu_si256((__m256i *)(c + 3 * rso + 8), c31);
  _mm256_storeu_si256((__m256i *)(c + 4 * rso), c40);
  _mm256_storeu_si256((__m256i *)(c + 4 * rso + 8), c41);
  _mm256_storeu_si256((__m256i *)(c + 5 * rso), c50);
  _mm256_storeu_si256((__m256i *)(c + 5 * rso + 8), c51);
}

static inline void gemm_edge_i32(const int32_t *a, const int32_t *b, int32_t *c,
                                 size_t mr, size_t nr, size_t kc, intptr_t rsa,
                                 intptr_t csa, intptr_t rsb, intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
    for (size_t p = 0; p < kc; p++) {
      int32_t aip = a[i * rsa + p * csa];
      __m256i va = _mm256_set1_epi32(aip);
      const int32_t *brow = b + p * rsb;
      int32_t *crow = c + i * rso;
      size_t j = 0;
      for (; j + 8 <= nr; j += 8) {
        __m256i vo = _mm256_loadu_si256((__m256i *)(crow + j));
        vo = _mm256_add_epi32(
            vo,
            _mm256_mullo_epi32(va, _mm256_loadu_si256((__m256i *)(brow + j))));
        _mm256_storeu_si256((__m256i *)(crow + j), vo);
      }
      for (; j < nr; j++)
        crow[j] += aip * brow[j];
    }
  }
}

static inline void gemm_i32_avx2(const int32_t *a, const int32_t *b,
                                 int32_t *out, size_t m_dim, size_t k_dim,
                                 size_t n_dim, intptr_t rsa, intptr_t csa,
                                 intptr_t rsb, intptr_t rso) {
  size_t nc_max = GEMM_MIN(GEMM_I32_NC, n_dim);
  int32_t *packed_b = (int32_t *)numc_malloc(
      32, GEMM_I32_KC * (nc_max + GEMM_I32_NR) * sizeof(int32_t));
  if (!packed_b)
    return;

  for (size_t jc = 0; jc < n_dim; jc += GEMM_I32_NC) {
    size_t nc = GEMM_MIN(GEMM_I32_NC, n_dim - jc);
    size_t n_jr = (nc + GEMM_I32_NR - 1) / GEMM_I32_NR;

#ifdef _OPENMP
#pragma omp parallel if ((uint64_t)m_dim * k_dim * nc > GEMM_OMP_THRESHOLD)
    {
      NUMC_ALIGNAS(32) int32_t packed_a[GEMM_I32_MC * GEMM_I32_KC];

      for (size_t pc = 0; pc < k_dim; pc += GEMM_I32_KC) {
        size_t kc = GEMM_MIN(GEMM_I32_KC, k_dim - pc);
        int first = (pc == 0);
        size_t last_ic = (size_t)-1;

#pragma omp for schedule(static)
        for (size_t jr_idx = 0; jr_idx < n_jr; jr_idx++) {
          size_t jj = jr_idx * GEMM_I32_NR;
          size_t nr_pack = GEMM_MIN(GEMM_I32_NR, nc - jj);
          _gemm_pack_b_strip_i32(b + pc * rsb + (jc + jj), packed_b + jj * kc,
                                 kc, nr_pack, rsb);
        }

        size_t n_ic = (m_dim + GEMM_I32_MC - 1) / GEMM_I32_MC;
        size_t n_tasks = n_ic * n_jr;

#pragma omp for schedule(static)
        for (size_t task = 0; task < n_tasks; task++) {
          size_t ic = (task / n_jr) * GEMM_I32_MC;
          size_t jr = (task % n_jr) * GEMM_I32_NR;
          size_t mc = GEMM_MIN(GEMM_I32_MC, m_dim - ic);
          size_t nr_cur = GEMM_MIN(GEMM_I32_NR, nc - jr);

          if (ic != last_ic) {
            gemm_pack_a_i32(a + ic * rsa + pc * csa, packed_a, mc, kc, rsa,
                            csa);
            last_ic = ic;
          }

          for (size_t ir = 0; ir < mc; ir += GEMM_I32_MR) {
            size_t mr_cur = GEMM_MIN(GEMM_I32_MR, mc - ir);
            if (mr_cur == GEMM_I32_MR && nr_cur == GEMM_I32_NR) {
#if NUMC_HAVE_ASM_UKERNEL
              numc_gemm_ukernel_i32_6x16_avx2(
                  packed_a + ir * kc, packed_b + jr * kc,
                  out + (ic + ir) * rso + (jc + jr), (uint64_t)kc, (int64_t)rso,
                  first);
#else
              gemm_ukernel_i32_6x16(packed_a + ir * kc, packed_b + jr * kc,
                                    out + (ic + ir) * rso + (jc + jr), kc, 1,
                                    GEMM_I32_MR, GEMM_I32_NR, rso, first);
#endif
            } else {
              NUMC_ALIGNAS(32) int32_t tmp[GEMM_I32_MR * GEMM_I32_NR];
              gemm_ukernel_i32_6x16(packed_a + ir * kc, packed_b + jr * kc, tmp,
                                    kc, 1, GEMM_I32_MR, GEMM_I32_NR,
                                    GEMM_I32_NR, 1);
              int32_t *dst = out + (ic + ir) * rso + (jc + jr);
              if (first) {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] = tmp[ii * GEMM_I32_NR + jj];
              } else {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] += tmp[ii * GEMM_I32_NR + jj];
              }
            }
          }
        }
      }
    }
#else
    {
      NUMC_ALIGNAS(32) int32_t packed_a[GEMM_I32_MC * GEMM_I32_KC];

      for (size_t pc = 0; pc < k_dim; pc += GEMM_I32_KC) {
        size_t kc = GEMM_MIN(GEMM_I32_KC, k_dim - pc);
        int first = (pc == 0);

        gemm_pack_b_i32(b + pc * rsb + jc, packed_b, kc, nc, rsb);

        size_t n_ic = (m_dim + GEMM_I32_MC - 1) / GEMM_I32_MC;
        size_t n_tasks = n_ic * n_jr;

        for (size_t task = 0; task < n_tasks; task++) {
          size_t ic = (task / n_jr) * GEMM_I32_MC;
          size_t jr = (task % n_jr) * GEMM_I32_NR;
          size_t mc = GEMM_MIN(GEMM_I32_MC, m_dim - ic);
          size_t nr_cur = GEMM_MIN(GEMM_I32_NR, nc - jr);

          if (task % n_jr == 0)
            gemm_pack_a_i32(a + ic * rsa + pc * csa, packed_a, mc, kc, rsa,
                            csa);

          for (size_t ir = 0; ir < mc; ir += GEMM_I32_MR) {
            size_t mr_cur = GEMM_MIN(GEMM_I32_MR, mc - ir);
            if (mr_cur == GEMM_I32_MR && nr_cur == GEMM_I32_NR) {
#if NUMC_HAVE_ASM_UKERNEL
              numc_gemm_ukernel_i32_6x16_avx2(
                  packed_a + ir * kc, packed_b + jr * kc,
                  out + (ic + ir) * rso + (jc + jr), (uint64_t)kc, (int64_t)rso,
                  first);
#else
              gemm_ukernel_i32_6x16(packed_a + ir * kc, packed_b + jr * kc,
                                    out + (ic + ir) * rso + (jc + jr), kc, 1,
                                    GEMM_I32_MR, GEMM_I32_NR, rso, first);
#endif
            } else {
              NUMC_ALIGNAS(32) int32_t tmp[GEMM_I32_MR * GEMM_I32_NR];
              gemm_ukernel_i32_6x16(packed_a + ir * kc, packed_b + jr * kc, tmp,
                                    kc, 1, GEMM_I32_MR, GEMM_I32_NR,
                                    GEMM_I32_NR, 1);
              int32_t *dst = out + (ic + ir) * rso + (jc + jr);
              if (first) {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] = tmp[ii * GEMM_I32_NR + jj];
              } else {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] += tmp[ii * GEMM_I32_NR + jj];
              }
            }
          }
        }
      }
    }
#endif
  }

  numc_free(packed_b);
}

/* Uint32: packed GEMM with dedicated ASM micro-kernel (vpmulld is sign-agnostic) */
static inline void gemm_u32_avx2(const uint32_t *a, const uint32_t *b,
                                 uint32_t *out, size_t m_dim, size_t k_dim,
                                 size_t n_dim, intptr_t rsa, intptr_t csa,
                                 intptr_t rsb, intptr_t rso) {
  size_t nc_max = GEMM_MIN(GEMM_I32_NC, n_dim);
  uint32_t *packed_b = (uint32_t *)numc_malloc(
      32, GEMM_I32_KC * (nc_max + GEMM_I32_NR) * sizeof(uint32_t));
  if (!packed_b)
    return;

  for (size_t jc = 0; jc < n_dim; jc += GEMM_I32_NC) {
    size_t nc = GEMM_MIN(GEMM_I32_NC, n_dim - jc);
    size_t n_jr = (nc + GEMM_I32_NR - 1) / GEMM_I32_NR;

#ifdef _OPENMP
#pragma omp parallel if ((uint64_t)m_dim * k_dim * nc > GEMM_OMP_THRESHOLD)
    {
      NUMC_ALIGNAS(32) uint32_t packed_a[GEMM_I32_MC * GEMM_I32_KC];

      for (size_t pc = 0; pc < k_dim; pc += GEMM_I32_KC) {
        size_t kc = GEMM_MIN(GEMM_I32_KC, k_dim - pc);
        int first = (pc == 0);
        size_t last_ic = (size_t)-1;

#pragma omp for schedule(static)
        for (size_t jr_idx = 0; jr_idx < n_jr; jr_idx++) {
          size_t jj = jr_idx * GEMM_I32_NR;
          size_t nr_pack = GEMM_MIN(GEMM_I32_NR, nc - jj);
          _gemm_pack_b_strip_i32((const int32_t *)(b + pc * rsb + (jc + jj)),
                                 (int32_t *)(packed_b + jj * kc),
                                 kc, nr_pack, rsb);
        }

        size_t n_ic = (m_dim + GEMM_I32_MC - 1) / GEMM_I32_MC;
        size_t n_tasks = n_ic * n_jr;

#pragma omp for schedule(static)
        for (size_t task = 0; task < n_tasks; task++) {
          size_t ic = (task / n_jr) * GEMM_I32_MC;
          size_t jr = (task % n_jr) * GEMM_I32_NR;
          size_t mc = GEMM_MIN(GEMM_I32_MC, m_dim - ic);
          size_t nr_cur = GEMM_MIN(GEMM_I32_NR, nc - jr);

          if (ic != last_ic) {
            gemm_pack_a_i32((const int32_t *)(a + ic * rsa + pc * csa),
                            (int32_t *)packed_a, mc, kc, rsa, csa);
            last_ic = ic;
          }

          for (size_t ir = 0; ir < mc; ir += GEMM_I32_MR) {
            size_t mr_cur = GEMM_MIN(GEMM_I32_MR, mc - ir);
            if (mr_cur == GEMM_I32_MR && nr_cur == GEMM_I32_NR) {
#if NUMC_HAVE_ASM_UKERNEL
              numc_gemm_ukernel_u32_6x16_avx2(
                  packed_a + ir * kc, packed_b + jr * kc,
                  out + (ic + ir) * rso + (jc + jr), (uint64_t)kc, (int64_t)rso,
                  first);
#else
              gemm_ukernel_i32_6x16((const int32_t *)(packed_a + ir * kc),
                                    (const int32_t *)(packed_b + jr * kc),
                                    (int32_t *)(out + (ic + ir) * rso + (jc + jr)),
                                    kc, 1, GEMM_I32_MR, GEMM_I32_NR, rso, first);
#endif
            } else {
              NUMC_ALIGNAS(32) uint32_t tmp[GEMM_I32_MR * GEMM_I32_NR];
              gemm_ukernel_i32_6x16((const int32_t *)(packed_a + ir * kc),
                                    (const int32_t *)(packed_b + jr * kc),
                                    (int32_t *)tmp,
                                    kc, 1, GEMM_I32_MR, GEMM_I32_NR,
                                    GEMM_I32_NR, 1);
              uint32_t *dst = out + (ic + ir) * rso + (jc + jr);
              if (first) {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] = tmp[ii * GEMM_I32_NR + jj];
              } else {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] += tmp[ii * GEMM_I32_NR + jj];
              }
            }
          }
        }
      }
    }
#else
    {
      NUMC_ALIGNAS(32) uint32_t packed_a[GEMM_I32_MC * GEMM_I32_KC];

      for (size_t pc = 0; pc < k_dim; pc += GEMM_I32_KC) {
        size_t kc = GEMM_MIN(GEMM_I32_KC, k_dim - pc);
        int first = (pc == 0);

        gemm_pack_b_i32((const int32_t *)(b + pc * rsb + jc),
                        (int32_t *)packed_b, kc, nc, rsb);

        size_t n_ic = (m_dim + GEMM_I32_MC - 1) / GEMM_I32_MC;
        size_t n_tasks = n_ic * n_jr;

        for (size_t task = 0; task < n_tasks; task++) {
          size_t ic = (task / n_jr) * GEMM_I32_MC;
          size_t jr = (task % n_jr) * GEMM_I32_NR;
          size_t mc = GEMM_MIN(GEMM_I32_MC, m_dim - ic);
          size_t nr_cur = GEMM_MIN(GEMM_I32_NR, nc - jr);

          if (task % n_jr == 0)
            gemm_pack_a_i32((const int32_t *)(a + ic * rsa + pc * csa),
                            (int32_t *)packed_a, mc, kc, rsa, csa);

          for (size_t ir = 0; ir < mc; ir += GEMM_I32_MR) {
            size_t mr_cur = GEMM_MIN(GEMM_I32_MR, mc - ir);
            if (mr_cur == GEMM_I32_MR && nr_cur == GEMM_I32_NR) {
#if NUMC_HAVE_ASM_UKERNEL
              numc_gemm_ukernel_u32_6x16_avx2(
                  packed_a + ir * kc, packed_b + jr * kc,
                  out + (ic + ir) * rso + (jc + jr), (uint64_t)kc, (int64_t)rso,
                  first);
#else
              gemm_ukernel_i32_6x16((const int32_t *)(packed_a + ir * kc),
                                    (const int32_t *)(packed_b + jr * kc),
                                    (int32_t *)(out + (ic + ir) * rso + (jc + jr)),
                                    kc, 1, GEMM_I32_MR, GEMM_I32_NR, rso, first);
#endif
            } else {
              NUMC_ALIGNAS(32) uint32_t tmp[GEMM_I32_MR * GEMM_I32_NR];
              gemm_ukernel_i32_6x16((const int32_t *)(packed_a + ir * kc),
                                    (const int32_t *)(packed_b + jr * kc),
                                    (int32_t *)tmp,
                                    kc, 1, GEMM_I32_MR, GEMM_I32_NR,
                                    GEMM_I32_NR, 1);
              uint32_t *dst = out + (ic + ir) * rso + (jc + jr);
              if (first) {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] = tmp[ii * GEMM_I32_NR + jj];
              } else {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] += tmp[ii * GEMM_I32_NR + jj];
              }
            }
          }
        }
      }
    }
#endif
  }

  numc_free(packed_b);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Int16/Uint16: 6×32 micro-kernel  (mullo_epi16 + add_epi16, 16 elem/reg)
   Same-width accumulation — matches the i32 overflow trade-off.
   ═══════════════════════════════════════════════════════════════════════════
 */

static inline void gemm_ukernel_i16_6x32(const int16_t *a, const int16_t *b,
                                         int16_t *c, size_t kc, intptr_t rsa,
                                         intptr_t csa, intptr_t rsb,
                                         intptr_t rso, int first) {
  __m256i c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
  if (first) {
    c00 = c01 = c10 = c11 = c20 = c21 = _mm256_setzero_si256();
    c30 = c31 = c40 = c41 = c50 = c51 = _mm256_setzero_si256();
  } else {
    c00 = _mm256_loadu_si256((__m256i *)(c));
    c01 = _mm256_loadu_si256((__m256i *)(c + 16));
    c10 = _mm256_loadu_si256((__m256i *)(c + rso));
    c11 = _mm256_loadu_si256((__m256i *)(c + rso + 16));
    c20 = _mm256_loadu_si256((__m256i *)(c + 2 * rso));
    c21 = _mm256_loadu_si256((__m256i *)(c + 2 * rso + 16));
    c30 = _mm256_loadu_si256((__m256i *)(c + 3 * rso));
    c31 = _mm256_loadu_si256((__m256i *)(c + 3 * rso + 16));
    c40 = _mm256_loadu_si256((__m256i *)(c + 4 * rso));
    c41 = _mm256_loadu_si256((__m256i *)(c + 4 * rso + 16));
    c50 = _mm256_loadu_si256((__m256i *)(c + 5 * rso));
    c51 = _mm256_loadu_si256((__m256i *)(c + 5 * rso + 16));
  }

  for (size_t p = 0; p < kc; p++) {
    const int16_t *bp = b + p * rsb;
    _mm_prefetch((const char *)(bp + rsb), _MM_HINT_T0);
    _mm_prefetch((const char *)(bp + rsb + 16), _MM_HINT_T0);
    __m256i b0 = _mm256_loadu_si256((__m256i *)bp),
            b1 = _mm256_loadu_si256((__m256i *)(bp + 16));
    __m256i av;
    av = _mm256_set1_epi16(a[0 * rsa + p * csa]);
    c00 = _mm256_add_epi16(c00, _mm256_mullo_epi16(av, b0));
    c01 = _mm256_add_epi16(c01, _mm256_mullo_epi16(av, b1));
    av = _mm256_set1_epi16(a[1 * rsa + p * csa]);
    c10 = _mm256_add_epi16(c10, _mm256_mullo_epi16(av, b0));
    c11 = _mm256_add_epi16(c11, _mm256_mullo_epi16(av, b1));
    av = _mm256_set1_epi16(a[2 * rsa + p * csa]);
    c20 = _mm256_add_epi16(c20, _mm256_mullo_epi16(av, b0));
    c21 = _mm256_add_epi16(c21, _mm256_mullo_epi16(av, b1));
    av = _mm256_set1_epi16(a[3 * rsa + p * csa]);
    c30 = _mm256_add_epi16(c30, _mm256_mullo_epi16(av, b0));
    c31 = _mm256_add_epi16(c31, _mm256_mullo_epi16(av, b1));
    av = _mm256_set1_epi16(a[4 * rsa + p * csa]);
    c40 = _mm256_add_epi16(c40, _mm256_mullo_epi16(av, b0));
    c41 = _mm256_add_epi16(c41, _mm256_mullo_epi16(av, b1));
    av = _mm256_set1_epi16(a[5 * rsa + p * csa]);
    c50 = _mm256_add_epi16(c50, _mm256_mullo_epi16(av, b0));
    c51 = _mm256_add_epi16(c51, _mm256_mullo_epi16(av, b1));
  }

  _mm256_storeu_si256((__m256i *)(c), c00);
  _mm256_storeu_si256((__m256i *)(c + 16), c01);
  _mm256_storeu_si256((__m256i *)(c + rso), c10);
  _mm256_storeu_si256((__m256i *)(c + rso + 16), c11);
  _mm256_storeu_si256((__m256i *)(c + 2 * rso), c20);
  _mm256_storeu_si256((__m256i *)(c + 2 * rso + 16), c21);
  _mm256_storeu_si256((__m256i *)(c + 3 * rso), c30);
  _mm256_storeu_si256((__m256i *)(c + 3 * rso + 16), c31);
  _mm256_storeu_si256((__m256i *)(c + 4 * rso), c40);
  _mm256_storeu_si256((__m256i *)(c + 4 * rso + 16), c41);
  _mm256_storeu_si256((__m256i *)(c + 5 * rso), c50);
  _mm256_storeu_si256((__m256i *)(c + 5 * rso + 16), c51);
}

static inline void gemm_edge_i16(const int16_t *a, const int16_t *b, int16_t *c,
                                 size_t mr, size_t nr, size_t kc, intptr_t rsa,
                                 intptr_t csa, intptr_t rsb, intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
    for (size_t p = 0; p < kc; p++) {
      int16_t aip = a[i * rsa + p * csa];
      __m256i va = _mm256_set1_epi16(aip);
      const int16_t *brow = b + p * rsb;
      int16_t *crow = c + i * rso;
      size_t j = 0;
      for (; j + 16 <= nr; j += 16) {
        __m256i vo = _mm256_loadu_si256((__m256i *)(crow + j));
        vo = _mm256_add_epi16(
            vo,
            _mm256_mullo_epi16(va, _mm256_loadu_si256((__m256i *)(brow + j))));
        _mm256_storeu_si256((__m256i *)(crow + j), vo);
      }
      for (; j < nr; j++)
        crow[j] = (int16_t)(crow[j] + aip * brow[j]);
    }
  }
}

/* Pack B[kc x nc] into NR-wide micropanels for i16.
 * Layout: for each NR-panel at column jr, kc rows of NR contiguous elements. */
static inline void gemm_pack_b_i16(const int16_t *b, int16_t *packed, size_t kc,
                                   size_t nc, intptr_t rsb) {
  size_t jr = 0;
  for (; jr + GEMM_I16_NR <= nc; jr += GEMM_I16_NR) {
    int16_t *dest = packed + jr * kc;
    for (size_t p = 0; p < kc; p++) {
      const int16_t *src = b + p * rsb + jr;
      _mm256_storeu_si256((__m256i *)(dest + p * GEMM_I16_NR),
                          _mm256_loadu_si256((__m256i *)src));
      _mm256_storeu_si256((__m256i *)(dest + p * GEMM_I16_NR + 16),
                          _mm256_loadu_si256((__m256i *)(src + 16)));
    }
  }
  if (jr < nc) {
    int16_t *dest = packed + jr * kc;
    size_t rem = nc - jr;
    __m256i z = _mm256_setzero_si256();
    for (size_t p = 0; p < kc; p++) {
      const int16_t *src = b + p * rsb + jr;
      int16_t *d = dest + p * GEMM_I16_NR;
      /* Scalar remainder copy with zero-padding */
      for (size_t j = 0; j < GEMM_I16_NR; j++)
        d[j] = j < rem ? src[j] : 0;
    }
  }
}

/* Pack a single NR-wide B strip for parallel packing (i16). */
static inline void _gemm_pack_b_strip_i16(const int16_t *b, int16_t *dest,
                                           size_t kc, size_t nr_pack,
                                           intptr_t rsb) {
  if (nr_pack == GEMM_I16_NR) {
    for (size_t p = 0; p < kc; p++) {
      const int16_t *src = b + p * rsb;
      _mm256_storeu_si256((__m256i *)(dest + p * GEMM_I16_NR),
                          _mm256_loadu_si256((__m256i *)src));
      _mm256_storeu_si256((__m256i *)(dest + p * GEMM_I16_NR + 16),
                          _mm256_loadu_si256((__m256i *)(src + 16)));
    }
  } else {
    for (size_t p = 0; p < kc; p++) {
      const int16_t *src = b + p * rsb;
      int16_t *d = dest + p * GEMM_I16_NR;
      for (size_t j = 0; j < GEMM_I16_NR; j++)
        d[j] = j < nr_pack ? src[j] : 0;
    }
  }
}

/* Pack A[mc x kc] into MR-tall micropanels for i16.
 * Layout: for each MR-panel at row ir, kc columns of MR contiguous elements. */
static inline void gemm_pack_a_i16(const int16_t *a, int16_t *packed, size_t mc,
                                   size_t kc, intptr_t rsa, intptr_t csa) {
  size_t ir = 0;
  for (; ir + GEMM_I16_MR <= mc; ir += GEMM_I16_MR) {
    int16_t *dest = packed + ir * kc;
    for (size_t p = 0; p < kc; p++) {
      int16_t *d = dest + p * GEMM_I16_MR;
      d[0] = a[(ir + 0) * rsa + p * csa];
      d[1] = a[(ir + 1) * rsa + p * csa];
      d[2] = a[(ir + 2) * rsa + p * csa];
      d[3] = a[(ir + 3) * rsa + p * csa];
      d[4] = a[(ir + 4) * rsa + p * csa];
      d[5] = a[(ir + 5) * rsa + p * csa];
    }
  }
  if (ir < mc) {
    int16_t *dest = packed + ir * kc;
    size_t rem = mc - ir;
    for (size_t p = 0; p < kc; p++) {
      int16_t *d = dest + p * GEMM_I16_MR;
      d[0] = rem > 0 ? a[(ir + 0) * rsa + p * csa] : 0;
      d[1] = rem > 1 ? a[(ir + 1) * rsa + p * csa] : 0;
      d[2] = rem > 2 ? a[(ir + 2) * rsa + p * csa] : 0;
      d[3] = rem > 3 ? a[(ir + 3) * rsa + p * csa] : 0;
      d[4] = rem > 4 ? a[(ir + 4) * rsa + p * csa] : 0;
      d[5] = rem > 5 ? a[(ir + 5) * rsa + p * csa] : 0;
    }
  }
}

static inline void gemm_i16_avx2(const int16_t *a, const int16_t *b,
                                 int16_t *out, size_t m_dim, size_t k_dim,
                                 size_t n_dim, intptr_t rsa, intptr_t csa,
                                 intptr_t rsb, intptr_t rso) {
  size_t nc_max = GEMM_MIN(GEMM_I16_NC, n_dim);
  int16_t *packed_b = (int16_t *)numc_malloc(
      32, GEMM_I16_KC * (nc_max + GEMM_I16_NR) * sizeof(int16_t));
  if (!packed_b)
    return;

  for (size_t jc = 0; jc < n_dim; jc += GEMM_I16_NC) {
    size_t nc = GEMM_MIN(GEMM_I16_NC, n_dim - jc);
    size_t n_jr = (nc + GEMM_I16_NR - 1) / GEMM_I16_NR;

#ifdef _OPENMP
#pragma omp parallel if ((uint64_t)m_dim * k_dim * nc > GEMM_OMP_THRESHOLD)
    {
      NUMC_ALIGNAS(32) int16_t packed_a[GEMM_I16_MC * GEMM_I16_KC];

      for (size_t pc = 0; pc < k_dim; pc += GEMM_I16_KC) {
        size_t kc = GEMM_MIN(GEMM_I16_KC, k_dim - pc);
        int first = (pc == 0);
        size_t last_ic = (size_t)-1;

#pragma omp for schedule(static)
        for (size_t jr_idx = 0; jr_idx < n_jr; jr_idx++) {
          size_t jj = jr_idx * GEMM_I16_NR;
          size_t nr_pack = GEMM_MIN(GEMM_I16_NR, nc - jj);
          _gemm_pack_b_strip_i16(b + pc * rsb + (jc + jj), packed_b + jj * kc,
                                 kc, nr_pack, rsb);
        }

        size_t n_ic = (m_dim + GEMM_I16_MC - 1) / GEMM_I16_MC;
        size_t n_tasks = n_ic * n_jr;

#pragma omp for schedule(static)
        for (size_t task = 0; task < n_tasks; task++) {
          size_t ic = (task / n_jr) * GEMM_I16_MC;
          size_t jr = (task % n_jr) * GEMM_I16_NR;
          size_t mc = GEMM_MIN(GEMM_I16_MC, m_dim - ic);
          size_t nr_cur = GEMM_MIN(GEMM_I16_NR, nc - jr);

          if (ic != last_ic) {
            gemm_pack_a_i16(a + ic * rsa + pc * csa, packed_a, mc, kc, rsa,
                            csa);
            last_ic = ic;
          }

          for (size_t ir = 0; ir < mc; ir += GEMM_I16_MR) {
            size_t mr_cur = GEMM_MIN(GEMM_I16_MR, mc - ir);
            if (mr_cur == GEMM_I16_MR && nr_cur == GEMM_I16_NR) {
#if NUMC_HAVE_ASM_UKERNEL
              numc_gemm_ukernel_i16_6x32_avx2(
                  packed_a + ir * kc, packed_b + jr * kc,
                  out + (ic + ir) * rso + (jc + jr), (uint64_t)kc, (int64_t)rso,
                  first);
#else
              gemm_ukernel_i16_6x32(packed_a + ir * kc, packed_b + jr * kc,
                                    out + (ic + ir) * rso + (jc + jr), kc, 1,
                                    GEMM_I16_MR, GEMM_I16_NR, rso, first);
#endif
            } else {
              NUMC_ALIGNAS(32) int16_t tmp[GEMM_I16_MR * GEMM_I16_NR];
              gemm_ukernel_i16_6x32(packed_a + ir * kc, packed_b + jr * kc, tmp,
                                    kc, 1, GEMM_I16_MR, GEMM_I16_NR,
                                    GEMM_I16_NR, 1);
              int16_t *dst = out + (ic + ir) * rso + (jc + jr);
              if (first) {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] = tmp[ii * GEMM_I16_NR + jj];
              } else {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] = (int16_t)(dst[ii * rso + jj] +
                                                   tmp[ii * GEMM_I16_NR + jj]);
              }
            }
          }
        }
      }
    }
#else
    {
      NUMC_ALIGNAS(32) int16_t packed_a[GEMM_I16_MC * GEMM_I16_KC];

      for (size_t pc = 0; pc < k_dim; pc += GEMM_I16_KC) {
        size_t kc = GEMM_MIN(GEMM_I16_KC, k_dim - pc);
        int first = (pc == 0);

        gemm_pack_b_i16(b + pc * rsb + jc, packed_b, kc, nc, rsb);

        size_t n_ic = (m_dim + GEMM_I16_MC - 1) / GEMM_I16_MC;
        size_t n_tasks = n_ic * n_jr;

        for (size_t task = 0; task < n_tasks; task++) {
          size_t ic = (task / n_jr) * GEMM_I16_MC;
          size_t jr = (task % n_jr) * GEMM_I16_NR;
          size_t mc = GEMM_MIN(GEMM_I16_MC, m_dim - ic);
          size_t nr_cur = GEMM_MIN(GEMM_I16_NR, nc - jr);

          if (task % n_jr == 0)
            gemm_pack_a_i16(a + ic * rsa + pc * csa, packed_a, mc, kc, rsa,
                            csa);

          for (size_t ir = 0; ir < mc; ir += GEMM_I16_MR) {
            size_t mr_cur = GEMM_MIN(GEMM_I16_MR, mc - ir);
            if (mr_cur == GEMM_I16_MR && nr_cur == GEMM_I16_NR) {
#if NUMC_HAVE_ASM_UKERNEL
              numc_gemm_ukernel_i16_6x32_avx2(
                  packed_a + ir * kc, packed_b + jr * kc,
                  out + (ic + ir) * rso + (jc + jr), (uint64_t)kc, (int64_t)rso,
                  first);
#else
              gemm_ukernel_i16_6x32(packed_a + ir * kc, packed_b + jr * kc,
                                    out + (ic + ir) * rso + (jc + jr), kc, 1,
                                    GEMM_I16_MR, GEMM_I16_NR, rso, first);
#endif
            } else {
              NUMC_ALIGNAS(32) int16_t tmp[GEMM_I16_MR * GEMM_I16_NR];
              gemm_ukernel_i16_6x32(packed_a + ir * kc, packed_b + jr * kc, tmp,
                                    kc, 1, GEMM_I16_MR, GEMM_I16_NR,
                                    GEMM_I16_NR, 1);
              int16_t *dst = out + (ic + ir) * rso + (jc + jr);
              if (first) {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] = tmp[ii * GEMM_I16_NR + jj];
              } else {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] = (int16_t)(dst[ii * rso + jj] +
                                                   tmp[ii * GEMM_I16_NR + jj]);
              }
            }
          }
        }
      }
    }
#endif
  }
  numc_free(packed_b);
}

static inline void gemm_u16_avx2(const uint16_t *a, const uint16_t *b,
                                 uint16_t *out, size_t m_dim, size_t k_dim,
                                 size_t n_dim, intptr_t rsa, intptr_t csa,
                                 intptr_t rsb, intptr_t rso) {
  gemm_i16_avx2((const int16_t *)a, (const int16_t *)b, (int16_t *)out, m_dim,
                k_dim, n_dim, rsa, csa, rsb, rso);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Int64/Uint64: 6×8 micro-kernel  (emulated 64-bit multiply via mul_epu32)
   No native mullo_epi64 in AVX2 — emulate: a*b = a_lo*b_lo + cross<<32
   ═══════════════════════════════════════════════════════════════════════════
 */

static inline __m256i gemm_mullo_epi64(__m256i a, __m256i b) {
  __m256i a_hi = _mm256_srli_epi64(a, 32);
  __m256i b_hi = _mm256_srli_epi64(b, 32);
  __m256i lo_lo = _mm256_mul_epu32(a, b);
  __m256i cross =
      _mm256_add_epi64(_mm256_mul_epu32(a, b_hi), _mm256_mul_epu32(a_hi, b));
  return _mm256_add_epi64(lo_lo, _mm256_slli_epi64(cross, 32));
}

static inline void gemm_ukernel_i64_6x8(const int64_t *a, const int64_t *b,
                                        int64_t *c, size_t kc, intptr_t rsa,
                                        intptr_t csa, intptr_t rsb,
                                        intptr_t rso) {
  __m256i c00 = _mm256_loadu_si256((__m256i *)(c)),
          c01 = _mm256_loadu_si256((__m256i *)(c + 4));
  __m256i c10 = _mm256_loadu_si256((__m256i *)(c + rso)),
          c11 = _mm256_loadu_si256((__m256i *)(c + rso + 4));
  __m256i c20 = _mm256_loadu_si256((__m256i *)(c + 2 * rso)),
          c21 = _mm256_loadu_si256((__m256i *)(c + 2 * rso + 4));
  __m256i c30 = _mm256_loadu_si256((__m256i *)(c + 3 * rso)),
          c31 = _mm256_loadu_si256((__m256i *)(c + 3 * rso + 4));
  __m256i c40 = _mm256_loadu_si256((__m256i *)(c + 4 * rso)),
          c41 = _mm256_loadu_si256((__m256i *)(c + 4 * rso + 4));
  __m256i c50 = _mm256_loadu_si256((__m256i *)(c + 5 * rso)),
          c51 = _mm256_loadu_si256((__m256i *)(c + 5 * rso + 4));

  for (size_t p = 0; p < kc; p++) {
    const int64_t *bp = b + p * rsb;
    _mm_prefetch((const char *)(bp + rsb), _MM_HINT_T0);
    __m256i b0 = _mm256_loadu_si256((__m256i *)bp),
            b1 = _mm256_loadu_si256((__m256i *)(bp + 4));
    __m256i av;
    av = _mm256_set1_epi64x(a[0 * rsa + p * csa]);
    c00 = _mm256_add_epi64(c00, gemm_mullo_epi64(av, b0));
    c01 = _mm256_add_epi64(c01, gemm_mullo_epi64(av, b1));
    av = _mm256_set1_epi64x(a[1 * rsa + p * csa]);
    c10 = _mm256_add_epi64(c10, gemm_mullo_epi64(av, b0));
    c11 = _mm256_add_epi64(c11, gemm_mullo_epi64(av, b1));
    av = _mm256_set1_epi64x(a[2 * rsa + p * csa]);
    c20 = _mm256_add_epi64(c20, gemm_mullo_epi64(av, b0));
    c21 = _mm256_add_epi64(c21, gemm_mullo_epi64(av, b1));
    av = _mm256_set1_epi64x(a[3 * rsa + p * csa]);
    c30 = _mm256_add_epi64(c30, gemm_mullo_epi64(av, b0));
    c31 = _mm256_add_epi64(c31, gemm_mullo_epi64(av, b1));
    av = _mm256_set1_epi64x(a[4 * rsa + p * csa]);
    c40 = _mm256_add_epi64(c40, gemm_mullo_epi64(av, b0));
    c41 = _mm256_add_epi64(c41, gemm_mullo_epi64(av, b1));
    av = _mm256_set1_epi64x(a[5 * rsa + p * csa]);
    c50 = _mm256_add_epi64(c50, gemm_mullo_epi64(av, b0));
    c51 = _mm256_add_epi64(c51, gemm_mullo_epi64(av, b1));
  }

  _mm256_storeu_si256((__m256i *)(c), c00);
  _mm256_storeu_si256((__m256i *)(c + 4), c01);
  _mm256_storeu_si256((__m256i *)(c + rso), c10);
  _mm256_storeu_si256((__m256i *)(c + rso + 4), c11);
  _mm256_storeu_si256((__m256i *)(c + 2 * rso), c20);
  _mm256_storeu_si256((__m256i *)(c + 2 * rso + 4), c21);
  _mm256_storeu_si256((__m256i *)(c + 3 * rso), c30);
  _mm256_storeu_si256((__m256i *)(c + 3 * rso + 4), c31);
  _mm256_storeu_si256((__m256i *)(c + 4 * rso), c40);
  _mm256_storeu_si256((__m256i *)(c + 4 * rso + 4), c41);
  _mm256_storeu_si256((__m256i *)(c + 5 * rso), c50);
  _mm256_storeu_si256((__m256i *)(c + 5 * rso + 4), c51);
}

static inline void gemm_edge_i64(const int64_t *a, const int64_t *b, int64_t *c,
                                 size_t mr, size_t nr, size_t kc, intptr_t rsa,
                                 intptr_t csa, intptr_t rsb, intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
    for (size_t p = 0; p < kc; p++) {
      int64_t aip = a[i * rsa + p * csa];
      const int64_t *brow = b + p * rsb;
      int64_t *crow = c + i * rso;
      for (size_t j = 0; j < nr; j++)
        crow[j] += aip * brow[j];
    }
  }
}

static inline void gemm_i64_avx2(const int64_t *a, const int64_t *b,
                                 int64_t *out, size_t m_dim, size_t k_dim,
                                 size_t n_dim, intptr_t rsa, intptr_t csa,
                                 intptr_t rsb, intptr_t rso) {
  for (size_t i = 0; i < m_dim; i++)
    memset(out + i * rso, 0, n_dim * sizeof(int64_t));

  for (size_t pc = 0; pc < k_dim; pc += GEMM_I64_KC) {
    size_t kc = GEMM_MIN(GEMM_I64_KC, k_dim - pc);
#ifdef _OPENMP
#pragma omp parallel for schedule( \
        static) if (m_dim * n_dim * sizeof(int64_t) > GEMM_OMP_THRESHOLD)
#endif
    for (size_t ic = 0; ic < m_dim; ic += GEMM_I64_MC) {
      size_t mc = GEMM_MIN(GEMM_I64_MC, m_dim - ic);
      size_t jr = 0;
      for (; jr + GEMM_I64_NR <= n_dim; jr += GEMM_I64_NR) {
        size_t ir = 0;
        for (; ir + GEMM_I64_MR <= mc; ir += GEMM_I64_MR)
          gemm_ukernel_i64_6x8(a + (ic + ir) * rsa + pc * csa,
                               b + pc * rsb + jr, out + (ic + ir) * rso + jr,
                               kc, rsa, csa, rsb, rso);
        if (ir < mc)
          gemm_edge_i64(a + (ic + ir) * rsa + pc * csa, b + pc * rsb + jr,
                        out + (ic + ir) * rso + jr, mc - ir, GEMM_I64_NR, kc,
                        rsa, csa, rsb, rso);
      }
      if (jr < n_dim)
        gemm_edge_i64(a + ic * rsa + pc * csa, b + pc * rsb + jr,
                      out + ic * rso + jr, mc, n_dim - jr, kc, rsa, csa, rsb,
                      rso);
    }
  }
}

static inline void gemm_u64_avx2(const uint64_t *a, const uint64_t *b,
                                 uint64_t *out, size_t m_dim, size_t k_dim,
                                 size_t n_dim, intptr_t rsa, intptr_t csa,
                                 intptr_t rsb, intptr_t rso) {
  gemm_i64_avx2((const int64_t *)a, (const int64_t *)b, (int64_t *)out, m_dim,
                k_dim, n_dim, rsa, csa, rsb, rso);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Int8/Uint8: 6×16 packed micro-kernel (widen to i32 accumulators, KC-blocked)
   Uses saturation packing for stores to match scalar truncation behavior.
   ═══════════════════════════════════════════════════════════════════════════
 */

/* ── i8 packing routines ── */

static inline void gemm_pack_b_i8(const int8_t *b, int8_t *packed, size_t kc,
                                   size_t nc, intptr_t rsb) {
  size_t jr = 0;
  for (; jr + GEMM_I8_NR <= nc; jr += GEMM_I8_NR) {
    int8_t *dest = packed + jr * kc;
    for (size_t p = 0; p < kc; p++) {
      const int8_t *src = b + p * rsb + jr;
      _mm_storeu_si128((__m128i *)(dest + p * GEMM_I8_NR),
                       _mm_loadu_si128((const __m128i *)src));
    }
  }
  if (jr < nc) {
    int8_t *dest = packed + jr * kc;
    size_t rem = nc - jr;
    for (size_t p = 0; p < kc; p++) {
      memset(dest + p * GEMM_I8_NR, 0, GEMM_I8_NR);
      memcpy(dest + p * GEMM_I8_NR, b + p * rsb + jr, rem);
    }
  }
}

static inline void _gemm_pack_b_strip_i8(const int8_t *b, int8_t *dest,
                                          size_t kc, size_t nr_pack,
                                          intptr_t rsb) {
  if (nr_pack == GEMM_I8_NR) {
    for (size_t p = 0; p < kc; p++) {
      const int8_t *src = b + p * rsb;
      _mm_storeu_si128((__m128i *)(dest + p * GEMM_I8_NR),
                       _mm_loadu_si128((const __m128i *)src));
    }
  } else {
    for (size_t p = 0; p < kc; p++) {
      memset(dest + p * GEMM_I8_NR, 0, GEMM_I8_NR);
      memcpy(dest + p * GEMM_I8_NR, b + p * rsb, nr_pack);
    }
  }
}

static inline void gemm_pack_a_i8(const int8_t *a, int8_t *packed, size_t mc,
                                   size_t kc, intptr_t rsa, intptr_t csa) {
  size_t ir = 0;
  for (; ir + GEMM_I8_MR <= mc; ir += GEMM_I8_MR) {
    int8_t *dest = packed + ir * kc;
    for (size_t p = 0; p < kc; p++) {
      for (size_t i = 0; i < GEMM_I8_MR; i++)
        dest[p * GEMM_I8_MR + i] = a[(ir + i) * rsa + p * csa];
    }
  }
  if (ir < mc) {
    int8_t *dest = packed + ir * kc;
    size_t rem = mc - ir;
    for (size_t p = 0; p < kc; p++) {
      int8_t *d = dest + p * GEMM_I8_MR;
      for (size_t i = 0; i < rem; i++)
        d[i] = a[(ir + i) * rsa + p * csa];
      for (size_t i = rem; i < GEMM_I8_MR; i++)
        d[i] = 0;
    }
  }
}

/* ── i8 6×16 packed micro-kernel ── */

static inline void gemm_ukernel_i8_6x16(const int8_t *a, const int8_t *b,
                                         int8_t *c, size_t kc, intptr_t rso,
                                         int first) {
  __m256i c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
  if (first) {
    c00 = c01 = c10 = c11 = c20 = c21 = _mm256_setzero_si256();
    c30 = c31 = c40 = c41 = c50 = c51 = _mm256_setzero_si256();
  } else {
    /* Load existing i8 from C, sign-extend to i32 accumulators */
    c00 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i const *)(c)));
    c01 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i const *)(c + 8)));
    c10 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i const *)(c + rso)));
    c11 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i const *)(c + rso + 8)));
    c20 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i const *)(c + 2 * rso)));
    c21 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i const *)(c + 2 * rso + 8)));
    c30 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i const *)(c + 3 * rso)));
    c31 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i const *)(c + 3 * rso + 8)));
    c40 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i const *)(c + 4 * rso)));
    c41 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i const *)(c + 4 * rso + 8)));
    c50 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i const *)(c + 5 * rso)));
    c51 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i const *)(c + 5 * rso + 8)));
  }

  const int8_t *ap = a;
  const int8_t *bp = b;
  for (size_t p = 0; p < kc; p++) {
    __m256i b0 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i const *)bp));
    __m256i b1 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i const *)(bp + 8)));
    __m256i av;
    av = _mm256_set1_epi32((int32_t)ap[0]);
    c00 = _mm256_add_epi32(c00, _mm256_mullo_epi32(av, b0));
    c01 = _mm256_add_epi32(c01, _mm256_mullo_epi32(av, b1));
    av = _mm256_set1_epi32((int32_t)ap[1]);
    c10 = _mm256_add_epi32(c10, _mm256_mullo_epi32(av, b0));
    c11 = _mm256_add_epi32(c11, _mm256_mullo_epi32(av, b1));
    av = _mm256_set1_epi32((int32_t)ap[2]);
    c20 = _mm256_add_epi32(c20, _mm256_mullo_epi32(av, b0));
    c21 = _mm256_add_epi32(c21, _mm256_mullo_epi32(av, b1));
    av = _mm256_set1_epi32((int32_t)ap[3]);
    c30 = _mm256_add_epi32(c30, _mm256_mullo_epi32(av, b0));
    c31 = _mm256_add_epi32(c31, _mm256_mullo_epi32(av, b1));
    av = _mm256_set1_epi32((int32_t)ap[4]);
    c40 = _mm256_add_epi32(c40, _mm256_mullo_epi32(av, b0));
    c41 = _mm256_add_epi32(c41, _mm256_mullo_epi32(av, b1));
    av = _mm256_set1_epi32((int32_t)ap[5]);
    c50 = _mm256_add_epi32(c50, _mm256_mullo_epi32(av, b0));
    c51 = _mm256_add_epi32(c51, _mm256_mullo_epi32(av, b1));
    ap += GEMM_I8_MR;
    bp += GEMM_I8_NR;
  }

  /* Truncate i32 → i8 via signed saturation packing and store NR=16 per row */
#define GEMM_STORE_I8_ROW_16(acc0, acc1, row)                              \
  do {                                                                     \
    __m128i lo0 = _mm256_castsi256_si128(acc0);                            \
    __m128i hi0 = _mm256_extracti128_si256(acc0, 1);                       \
    __m128i lo1 = _mm256_castsi256_si128(acc1);                            \
    __m128i hi1 = _mm256_extracti128_si256(acc1, 1);                       \
    __m128i p16a = _mm_packs_epi32(lo0, hi0);                              \
    __m128i p16b = _mm_packs_epi32(lo1, hi1);                              \
    __m128i p8 = _mm_packs_epi16(p16a, p16b);                              \
    _mm_storeu_si128((__m128i *)(c + (row) * rso), p8);                    \
  } while (0)
  GEMM_STORE_I8_ROW_16(c00, c01, 0);
  GEMM_STORE_I8_ROW_16(c10, c11, 1);
  GEMM_STORE_I8_ROW_16(c20, c21, 2);
  GEMM_STORE_I8_ROW_16(c30, c31, 3);
  GEMM_STORE_I8_ROW_16(c40, c41, 4);
  GEMM_STORE_I8_ROW_16(c50, c51, 5);
#undef GEMM_STORE_I8_ROW_16
}

static inline void gemm_i8_avx2(const int8_t *a, const int8_t *b, int8_t *out,
                                 size_t m_dim, size_t k_dim, size_t n_dim,
                                 intptr_t rsa, intptr_t csa, intptr_t rsb,
                                 intptr_t rso) {
  size_t nc_max = GEMM_MIN(GEMM_I8_NC, n_dim);
  int8_t *packed_b = (int8_t *)numc_malloc(
      32, GEMM_I8_KC * (nc_max + GEMM_I8_NR) * sizeof(int8_t));
  if (!packed_b)
    return;

  for (size_t jc = 0; jc < n_dim; jc += GEMM_I8_NC) {
    size_t nc = GEMM_MIN(GEMM_I8_NC, n_dim - jc);
    size_t n_jr = (nc + GEMM_I8_NR - 1) / GEMM_I8_NR;

#ifdef _OPENMP
#pragma omp parallel if ((uint64_t)m_dim * k_dim * nc > GEMM_OMP_THRESHOLD)
    {
      NUMC_ALIGNAS(32) int8_t packed_a[GEMM_I8_MC * GEMM_I8_KC];

      for (size_t pc = 0; pc < k_dim; pc += GEMM_I8_KC) {
        size_t kc = GEMM_MIN(GEMM_I8_KC, k_dim - pc);
        int first = (pc == 0);
        size_t last_ic = (size_t)-1;

#pragma omp for schedule(static)
        for (size_t jr_idx = 0; jr_idx < n_jr; jr_idx++) {
          size_t jj = jr_idx * GEMM_I8_NR;
          size_t nr_pack = GEMM_MIN(GEMM_I8_NR, nc - jj);
          _gemm_pack_b_strip_i8(b + pc * rsb + (jc + jj), packed_b + jj * kc,
                                kc, nr_pack, rsb);
        }

        size_t n_ic = (m_dim + GEMM_I8_MC - 1) / GEMM_I8_MC;
        size_t n_tasks = n_ic * n_jr;

#pragma omp for schedule(static)
        for (size_t task = 0; task < n_tasks; task++) {
          size_t ic = (task / n_jr) * GEMM_I8_MC;
          size_t jr = (task % n_jr) * GEMM_I8_NR;
          size_t mc = GEMM_MIN(GEMM_I8_MC, m_dim - ic);
          size_t nr_cur = GEMM_MIN(GEMM_I8_NR, nc - jr);

          if (ic != last_ic) {
            gemm_pack_a_i8(a + ic * rsa + pc * csa, packed_a, mc, kc, rsa,
                           csa);
            last_ic = ic;
          }

          for (size_t ir = 0; ir < mc; ir += GEMM_I8_MR) {
            size_t mr_cur = GEMM_MIN(GEMM_I8_MR, mc - ir);
            if (mr_cur == GEMM_I8_MR && nr_cur == GEMM_I8_NR) {
#if NUMC_HAVE_ASM_UKERNEL
              numc_gemm_ukernel_i8_6x16_avx2(
                  packed_a + ir * kc, packed_b + jr * kc,
                  out + (ic + ir) * rso + (jc + jr), (uint64_t)kc, (int64_t)rso,
                  first);
#else
              gemm_ukernel_i8_6x16(packed_a + ir * kc, packed_b + jr * kc,
                                   out + (ic + ir) * rso + (jc + jr), kc, rso,
                                   first);
#endif
            } else {
              NUMC_ALIGNAS(32) int8_t tmp[GEMM_I8_MR * GEMM_I8_NR];
              gemm_ukernel_i8_6x16(packed_a + ir * kc, packed_b + jr * kc, tmp,
                                   kc, GEMM_I8_NR, 1);
              int8_t *dst = out + (ic + ir) * rso + (jc + jr);
              if (first) {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] = tmp[ii * GEMM_I8_NR + jj];
              } else {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] = (int8_t)((int32_t)dst[ii * rso + jj] +
                                                  (int32_t)tmp[ii * GEMM_I8_NR + jj]);
              }
            }
          }
        }
      }
    }
#else
    {
      NUMC_ALIGNAS(32) int8_t packed_a[GEMM_I8_MC * GEMM_I8_KC];

      for (size_t pc = 0; pc < k_dim; pc += GEMM_I8_KC) {
        size_t kc = GEMM_MIN(GEMM_I8_KC, k_dim - pc);
        int first = (pc == 0);

        gemm_pack_b_i8(b + pc * rsb + jc, packed_b, kc, nc, rsb);

        size_t n_ic = (m_dim + GEMM_I8_MC - 1) / GEMM_I8_MC;
        size_t n_tasks = n_ic * n_jr;

        for (size_t task = 0; task < n_tasks; task++) {
          size_t ic = (task / n_jr) * GEMM_I8_MC;
          size_t jr = (task % n_jr) * GEMM_I8_NR;
          size_t mc = GEMM_MIN(GEMM_I8_MC, m_dim - ic);
          size_t nr_cur = GEMM_MIN(GEMM_I8_NR, nc - jr);

          if (task % n_jr == 0)
            gemm_pack_a_i8(a + ic * rsa + pc * csa, packed_a, mc, kc, rsa,
                           csa);

          for (size_t ir = 0; ir < mc; ir += GEMM_I8_MR) {
            size_t mr_cur = GEMM_MIN(GEMM_I8_MR, mc - ir);
            if (mr_cur == GEMM_I8_MR && nr_cur == GEMM_I8_NR) {
#if NUMC_HAVE_ASM_UKERNEL
              numc_gemm_ukernel_i8_6x16_avx2(
                  packed_a + ir * kc, packed_b + jr * kc,
                  out + (ic + ir) * rso + (jc + jr), (uint64_t)kc, (int64_t)rso,
                  first);
#else
              gemm_ukernel_i8_6x16(packed_a + ir * kc, packed_b + jr * kc,
                                   out + (ic + ir) * rso + (jc + jr), kc, rso,
                                   first);
#endif
            } else {
              NUMC_ALIGNAS(32) int8_t tmp[GEMM_I8_MR * GEMM_I8_NR];
              gemm_ukernel_i8_6x16(packed_a + ir * kc, packed_b + jr * kc, tmp,
                                   kc, GEMM_I8_NR, 1);
              int8_t *dst = out + (ic + ir) * rso + (jc + jr);
              if (first) {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] = tmp[ii * GEMM_I8_NR + jj];
              } else {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] = (int8_t)((int32_t)dst[ii * rso + jj] +
                                                  (int32_t)tmp[ii * GEMM_I8_NR + jj]);
              }
            }
          }
        }
      }
    }
#endif
  }

  numc_free(packed_b);
}

/* ── u8 packing routines (reuse i8 pack for B and A — byte-level identical) ── */

static inline void gemm_pack_b_u8(const uint8_t *b, uint8_t *packed, size_t kc,
                                   size_t nc, intptr_t rsb) {
  gemm_pack_b_i8((const int8_t *)b, (int8_t *)packed, kc, nc, rsb);
}

static inline void _gemm_pack_b_strip_u8(const uint8_t *b, uint8_t *dest,
                                          size_t kc, size_t nr_pack,
                                          intptr_t rsb) {
  _gemm_pack_b_strip_i8((const int8_t *)b, (int8_t *)dest, kc, nr_pack, rsb);
}

static inline void gemm_pack_a_u8(const uint8_t *a, uint8_t *packed, size_t mc,
                                   size_t kc, intptr_t rsa, intptr_t csa) {
  gemm_pack_a_i8((const int8_t *)a, (int8_t *)packed, mc, kc, rsa, csa);
}

/* ── u8 6×16 packed micro-kernel ── */

static inline void gemm_ukernel_u8_6x16(const uint8_t *a, const uint8_t *b,
                                         uint8_t *c, size_t kc, intptr_t rso,
                                         int first) {
  __m256i c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
  if (first) {
    c00 = c01 = c10 = c11 = c20 = c21 = _mm256_setzero_si256();
    c30 = c31 = c40 = c41 = c50 = c51 = _mm256_setzero_si256();
  } else {
    /* Load existing u8 from C, zero-extend to i32 accumulators */
    c00 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i const *)(c)));
    c01 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i const *)(c + 8)));
    c10 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i const *)(c + rso)));
    c11 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i const *)(c + rso + 8)));
    c20 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i const *)(c + 2 * rso)));
    c21 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i const *)(c + 2 * rso + 8)));
    c30 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i const *)(c + 3 * rso)));
    c31 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i const *)(c + 3 * rso + 8)));
    c40 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i const *)(c + 4 * rso)));
    c41 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i const *)(c + 4 * rso + 8)));
    c50 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i const *)(c + 5 * rso)));
    c51 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i const *)(c + 5 * rso + 8)));
  }

  const uint8_t *ap = a;
  const uint8_t *bp = b;
  for (size_t p = 0; p < kc; p++) {
    __m256i b0 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i const *)bp));
    __m256i b1 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i const *)(bp + 8)));
    __m256i av;
    av = _mm256_set1_epi32((int32_t)(uint32_t)ap[0]);
    c00 = _mm256_add_epi32(c00, _mm256_mullo_epi32(av, b0));
    c01 = _mm256_add_epi32(c01, _mm256_mullo_epi32(av, b1));
    av = _mm256_set1_epi32((int32_t)(uint32_t)ap[1]);
    c10 = _mm256_add_epi32(c10, _mm256_mullo_epi32(av, b0));
    c11 = _mm256_add_epi32(c11, _mm256_mullo_epi32(av, b1));
    av = _mm256_set1_epi32((int32_t)(uint32_t)ap[2]);
    c20 = _mm256_add_epi32(c20, _mm256_mullo_epi32(av, b0));
    c21 = _mm256_add_epi32(c21, _mm256_mullo_epi32(av, b1));
    av = _mm256_set1_epi32((int32_t)(uint32_t)ap[3]);
    c30 = _mm256_add_epi32(c30, _mm256_mullo_epi32(av, b0));
    c31 = _mm256_add_epi32(c31, _mm256_mullo_epi32(av, b1));
    av = _mm256_set1_epi32((int32_t)(uint32_t)ap[4]);
    c40 = _mm256_add_epi32(c40, _mm256_mullo_epi32(av, b0));
    c41 = _mm256_add_epi32(c41, _mm256_mullo_epi32(av, b1));
    av = _mm256_set1_epi32((int32_t)(uint32_t)ap[5]);
    c50 = _mm256_add_epi32(c50, _mm256_mullo_epi32(av, b0));
    c51 = _mm256_add_epi32(c51, _mm256_mullo_epi32(av, b1));
    ap += GEMM_I8_MR;
    bp += GEMM_I8_NR;
  }

  /* Truncate i32 → u8 via unsigned saturation packing and store NR=16 per row */
#define GEMM_STORE_U8_ROW_16(acc0, acc1, row)                              \
  do {                                                                     \
    __m128i lo0 = _mm256_castsi256_si128(acc0);                            \
    __m128i hi0 = _mm256_extracti128_si256(acc0, 1);                       \
    __m128i lo1 = _mm256_castsi256_si128(acc1);                            \
    __m128i hi1 = _mm256_extracti128_si256(acc1, 1);                       \
    __m128i p16a = _mm_packus_epi32(lo0, hi0);                             \
    __m128i p16b = _mm_packus_epi32(lo1, hi1);                             \
    __m128i p8 = _mm_packus_epi16(p16a, p16b);                             \
    _mm_storeu_si128((__m128i *)(c + (row) * rso), p8);                    \
  } while (0)
  GEMM_STORE_U8_ROW_16(c00, c01, 0);
  GEMM_STORE_U8_ROW_16(c10, c11, 1);
  GEMM_STORE_U8_ROW_16(c20, c21, 2);
  GEMM_STORE_U8_ROW_16(c30, c31, 3);
  GEMM_STORE_U8_ROW_16(c40, c41, 4);
  GEMM_STORE_U8_ROW_16(c50, c51, 5);
#undef GEMM_STORE_U8_ROW_16
}

static inline void gemm_u8_avx2(const uint8_t *a, const uint8_t *b,
                                 uint8_t *out, size_t m_dim, size_t k_dim,
                                 size_t n_dim, intptr_t rsa, intptr_t csa,
                                 intptr_t rsb, intptr_t rso) {
  size_t nc_max = GEMM_MIN(GEMM_I8_NC, n_dim);
  uint8_t *packed_b = (uint8_t *)numc_malloc(
      32, GEMM_I8_KC * (nc_max + GEMM_I8_NR) * sizeof(uint8_t));
  if (!packed_b)
    return;

  for (size_t jc = 0; jc < n_dim; jc += GEMM_I8_NC) {
    size_t nc = GEMM_MIN(GEMM_I8_NC, n_dim - jc);
    size_t n_jr = (nc + GEMM_I8_NR - 1) / GEMM_I8_NR;

#ifdef _OPENMP
#pragma omp parallel if ((uint64_t)m_dim * k_dim * nc > GEMM_OMP_THRESHOLD)
    {
      NUMC_ALIGNAS(32) uint8_t packed_a[GEMM_I8_MC * GEMM_I8_KC];

      for (size_t pc = 0; pc < k_dim; pc += GEMM_I8_KC) {
        size_t kc = GEMM_MIN(GEMM_I8_KC, k_dim - pc);
        int first = (pc == 0);
        size_t last_ic = (size_t)-1;

#pragma omp for schedule(static)
        for (size_t jr_idx = 0; jr_idx < n_jr; jr_idx++) {
          size_t jj = jr_idx * GEMM_I8_NR;
          size_t nr_pack = GEMM_MIN(GEMM_I8_NR, nc - jj);
          _gemm_pack_b_strip_u8(b + pc * rsb + (jc + jj), packed_b + jj * kc,
                                kc, nr_pack, rsb);
        }

        size_t n_ic = (m_dim + GEMM_I8_MC - 1) / GEMM_I8_MC;
        size_t n_tasks = n_ic * n_jr;

#pragma omp for schedule(static)
        for (size_t task = 0; task < n_tasks; task++) {
          size_t ic = (task / n_jr) * GEMM_I8_MC;
          size_t jr = (task % n_jr) * GEMM_I8_NR;
          size_t mc = GEMM_MIN(GEMM_I8_MC, m_dim - ic);
          size_t nr_cur = GEMM_MIN(GEMM_I8_NR, nc - jr);

          if (ic != last_ic) {
            gemm_pack_a_u8(a + ic * rsa + pc * csa, packed_a, mc, kc, rsa,
                           csa);
            last_ic = ic;
          }

          for (size_t ir = 0; ir < mc; ir += GEMM_I8_MR) {
            size_t mr_cur = GEMM_MIN(GEMM_I8_MR, mc - ir);
            if (mr_cur == GEMM_I8_MR && nr_cur == GEMM_I8_NR) {
#if NUMC_HAVE_ASM_UKERNEL
              numc_gemm_ukernel_u8_6x16_avx2(
                  packed_a + ir * kc, packed_b + jr * kc,
                  out + (ic + ir) * rso + (jc + jr), (uint64_t)kc, (int64_t)rso,
                  first);
#else
              gemm_ukernel_u8_6x16(packed_a + ir * kc, packed_b + jr * kc,
                                   out + (ic + ir) * rso + (jc + jr), kc, rso,
                                   first);
#endif
            } else {
              NUMC_ALIGNAS(32) uint8_t tmp[GEMM_I8_MR * GEMM_I8_NR];
              gemm_ukernel_u8_6x16(packed_a + ir * kc, packed_b + jr * kc, tmp,
                                   kc, GEMM_I8_NR, 1);
              uint8_t *dst = out + (ic + ir) * rso + (jc + jr);
              if (first) {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] = tmp[ii * GEMM_I8_NR + jj];
              } else {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] = (uint8_t)((uint32_t)dst[ii * rso + jj] +
                                                   (uint32_t)tmp[ii * GEMM_I8_NR + jj]);
              }
            }
          }
        }
      }
    }
#else
    {
      NUMC_ALIGNAS(32) uint8_t packed_a[GEMM_I8_MC * GEMM_I8_KC];

      for (size_t pc = 0; pc < k_dim; pc += GEMM_I8_KC) {
        size_t kc = GEMM_MIN(GEMM_I8_KC, k_dim - pc);
        int first = (pc == 0);

        gemm_pack_b_u8(b + pc * rsb + jc, packed_b, kc, nc, rsb);

        size_t n_ic = (m_dim + GEMM_I8_MC - 1) / GEMM_I8_MC;
        size_t n_tasks = n_ic * n_jr;

        for (size_t task = 0; task < n_tasks; task++) {
          size_t ic = (task / n_jr) * GEMM_I8_MC;
          size_t jr = (task % n_jr) * GEMM_I8_NR;
          size_t mc = GEMM_MIN(GEMM_I8_MC, m_dim - ic);
          size_t nr_cur = GEMM_MIN(GEMM_I8_NR, nc - jr);

          if (task % n_jr == 0)
            gemm_pack_a_u8(a + ic * rsa + pc * csa, packed_a, mc, kc, rsa,
                           csa);

          for (size_t ir = 0; ir < mc; ir += GEMM_I8_MR) {
            size_t mr_cur = GEMM_MIN(GEMM_I8_MR, mc - ir);
            if (mr_cur == GEMM_I8_MR && nr_cur == GEMM_I8_NR) {
#if NUMC_HAVE_ASM_UKERNEL
              numc_gemm_ukernel_u8_6x16_avx2(
                  packed_a + ir * kc, packed_b + jr * kc,
                  out + (ic + ir) * rso + (jc + jr), (uint64_t)kc, (int64_t)rso,
                  first);
#else
              gemm_ukernel_u8_6x16(packed_a + ir * kc, packed_b + jr * kc,
                                   out + (ic + ir) * rso + (jc + jr), kc, rso,
                                   first);
#endif
            } else {
              NUMC_ALIGNAS(32) uint8_t tmp[GEMM_I8_MR * GEMM_I8_NR];
              gemm_ukernel_u8_6x16(packed_a + ir * kc, packed_b + jr * kc, tmp,
                                   kc, GEMM_I8_NR, 1);
              uint8_t *dst = out + (ic + ir) * rso + (jc + jr);
              if (first) {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] = tmp[ii * GEMM_I8_NR + jj];
              } else {
                for (size_t ii = 0; ii < mr_cur; ii++)
                  for (size_t jj = 0; jj < nr_cur; jj++)
                    dst[ii * rso + jj] = (uint8_t)((uint32_t)dst[ii * rso + jj] +
                                                   (uint32_t)tmp[ii * GEMM_I8_NR + jj]);
              }
            }
          }
        }
      }
    }
#endif
  }

  numc_free(packed_b);
}

#undef GEMM_MIN

#endif
