#ifndef NUMC_GEMM_AVX2_H
#define NUMC_GEMM_AVX2_H

#include <immintrin.h>
#include <stdint.h>
#include <string.h>

#define GEMM_MIN(a, b) ((a) < (b) ? (a) : (b))

/*
 * Cache-blocking parameters for AVX2 packed GEMM (BLIS-derived).
 *   - MC × KC panel of A resides in L2  (f32: 168×256×4 = 172KB < 256KB)
 *   - KC × NR sliver of B resides in L1 (f32: 256×16×4 = 16KB < 32KB)
 *   - KC × NC panel of B resides in L3
 * MC=168 (f32) and MC=96 (f64) match BLIS L2 sizing. Thread utilization
 * is maintained via 2D IC×JR parallelism: tasks = ceil(M/MC) × ceil(N/NR),
 * giving many more work items than 1D IC-loop alone.
 */
#define GEMM_F32_MR 6
#define GEMM_F32_NR 16
#define GEMM_F32_MC 168
#define GEMM_F32_KC 256

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

/* i8/u8: promoted to i32 accumulators, no K-blocking */
#define GEMM_I8_MR 6
#define GEMM_I8_NR 8
#define GEMM_I8_MC 72

/* GEMM OMP threshold on compute volume (M × K × N operations).
 * Threading pays off when compute dominates OMP fork cost (~20-50μs).
 * 256×256: 16.8M ops → ~330μs single-threaded → threading helps.
 * 128×128: 2.1M ops → ~50μs single-threaded → overhead dominates. */
#define GEMM_OMP_THRESHOLD (1 << 23)

/* N-dimension blocking for L3 residency (B panel: KC × NC elements) */
#define GEMM_F32_NC 4080
#define GEMM_F64_NC 4080

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
    for (size_t p = 0; p < kc; p++) {
      const float *src = b + p * rsb + jr;
      size_t j = 0;
      for (; j < rem; j++)
        dest[p * GEMM_F32_NR + j] = src[j];
      for (; j < GEMM_F32_NR; j++)
        dest[p * GEMM_F32_NR + j] = 0.0f;
    }
  }
}

/* Pack A[mc × kc] into MR-tall micropanels (column-panel format).
 * Layout: for each MR-panel at row ir, kc columns of MR contiguous elements.
 * Micro-kernel sees rsa = 1, csa = MR. */
static inline void gemm_pack_a_f32(const float *a, float *packed, size_t mc,
                                   size_t kc, intptr_t rsa, intptr_t csa) {
  size_t ir = 0;
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
    for (size_t p = 0; p < kc; p++) {
      size_t i = 0;
      for (; i < rem; i++)
        dest[p * GEMM_F32_MR + i] = a[(ir + i) * rsa + p * csa];
      for (; i < GEMM_F32_MR; i++)
        dest[p * GEMM_F32_MR + i] = 0.0f;
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
    for (size_t p = 0; p < kc; p++) {
      const double *src = b + p * rsb + jr;
      size_t j = 0;
      for (; j < rem; j++)
        dest[p * GEMM_F64_NR + j] = src[j];
      for (; j < GEMM_F64_NR; j++)
        dest[p * GEMM_F64_NR + j] = 0.0;
    }
  }
}

static inline void gemm_pack_a_f64(const double *a, double *packed, size_t mc,
                                   size_t kc, intptr_t rsa, intptr_t csa) {
  size_t ir = 0;
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
    for (size_t p = 0; p < kc; p++) {
      size_t i = 0;
      for (; i < rem; i++)
        dest[p * GEMM_F64_MR + i] = a[(ir + i) * rsa + p * csa];
      for (; i < GEMM_F64_MR; i++)
        dest[p * GEMM_F64_MR + i] = 0.0;
    }
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   Float32: 6×16 micro-kernel  (12 acc + 1 A broadcast + 2 B loads = 15 YMM)
   ═══════════════════════════════════════════════════════════════════════════
 */

/* One K-iteration of the 6×16 micro-kernel body.
 * ap points to 6 packed A values (MR stride), bp points to 16 packed B values.
 * 12 FMA instructions: 6 broadcasts × 2 B-vector halves. */
#define GEMM_F32_K_ITER(ap, bp)            \
  do {                                     \
    __m256 b0 = _mm256_loadu_ps(bp);       \
    __m256 b1 = _mm256_loadu_ps((bp) + 8); \
    __m256 av;                             \
    av = _mm256_broadcast_ss((ap) + 0);    \
    c00 = _mm256_fmadd_ps(av, b0, c00);    \
    c01 = _mm256_fmadd_ps(av, b1, c01);    \
    av = _mm256_broadcast_ss((ap) + 1);    \
    c10 = _mm256_fmadd_ps(av, b0, c10);    \
    c11 = _mm256_fmadd_ps(av, b1, c11);    \
    av = _mm256_broadcast_ss((ap) + 2);    \
    c20 = _mm256_fmadd_ps(av, b0, c20);    \
    c21 = _mm256_fmadd_ps(av, b1, c21);    \
    av = _mm256_broadcast_ss((ap) + 3);    \
    c30 = _mm256_fmadd_ps(av, b0, c30);    \
    c31 = _mm256_fmadd_ps(av, b1, c31);    \
    av = _mm256_broadcast_ss((ap) + 4);    \
    c40 = _mm256_fmadd_ps(av, b0, c40);    \
    c41 = _mm256_fmadd_ps(av, b1, c41);    \
    av = _mm256_broadcast_ss((ap) + 5);    \
    c50 = _mm256_fmadd_ps(av, b0, c50);    \
    c51 = _mm256_fmadd_ps(av, b1, c51);    \
  } while (0)

#if defined(__GNUC__) || defined(__clang__)

/* Inline-asm f32 6×16 micro-kernel: 12 accumulators pinned to ymm0-ymm11,
 * no register spills. Interleaves vbroadcastss (port 5) with vfmadd231ps
 * (ports 0,1) for optimal port utilization. 4× unrolled K-loop. */
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
      /* K-loop: k_iter = kc >> 2 */
      "mov %[kc], %%rcx\n\t"
      "shr $2, %%rcx\n\t"
      "jz 4f\n\t"

      ".p2align 4\n\t"
      "3:\n\t"
      /* === K iteration 0 === */
      "vmovups (%[bp]), %%ymm12\n\t"
      "vmovups 32(%[bp]), %%ymm13\n\t"
      "vbroadcastss (%[ap]), %%ymm14\n\t"
      "vfmadd231ps %%ymm14, %%ymm12, %%ymm0\n\t"
      "vfmadd231ps %%ymm14, %%ymm13, %%ymm1\n\t"
      "vbroadcastss 4(%[ap]), %%ymm14\n\t"
      "vfmadd231ps %%ymm14, %%ymm12, %%ymm2\n\t"
      "vfmadd231ps %%ymm14, %%ymm13, %%ymm3\n\t"
      "vbroadcastss 8(%[ap]), %%ymm14\n\t"
      "vfmadd231ps %%ymm14, %%ymm12, %%ymm4\n\t"
      "vfmadd231ps %%ymm14, %%ymm13, %%ymm5\n\t"
      "vbroadcastss 12(%[ap]), %%ymm14\n\t"
      "vfmadd231ps %%ymm14, %%ymm12, %%ymm6\n\t"
      "vfmadd231ps %%ymm14, %%ymm13, %%ymm7\n\t"
      "vbroadcastss 16(%[ap]), %%ymm14\n\t"
      "vfmadd231ps %%ymm14, %%ymm12, %%ymm8\n\t"
      "vfmadd231ps %%ymm14, %%ymm13, %%ymm9\n\t"
      "vbroadcastss 20(%[ap]), %%ymm14\n\t"
      "vfmadd231ps %%ymm14, %%ymm12, %%ymm10\n\t"
      "vfmadd231ps %%ymm14, %%ymm13, %%ymm11\n\t"
      "add %[csa_bytes], %[ap]\n\t"
      "add %[rsb_bytes], %[bp]\n\t"

      /* === K iteration 1 === */
      "vmovups (%[bp]), %%ymm12\n\t"
      "vmovups 32(%[bp]), %%ymm13\n\t"
      "vbroadcastss (%[ap]), %%ymm14\n\t"
      "vfmadd231ps %%ymm14, %%ymm12, %%ymm0\n\t"
      "vfmadd231ps %%ymm14, %%ymm13, %%ymm1\n\t"
      "vbroadcastss 4(%[ap]), %%ymm14\n\t"
      "vfmadd231ps %%ymm14, %%ymm12, %%ymm2\n\t"
      "vfmadd231ps %%ymm14, %%ymm13, %%ymm3\n\t"
      "vbroadcastss 8(%[ap]), %%ymm14\n\t"
      "vfmadd231ps %%ymm14, %%ymm12, %%ymm4\n\t"
      "vfmadd231ps %%ymm14, %%ymm13, %%ymm5\n\t"
      "vbroadcastss 12(%[ap]), %%ymm14\n\t"
      "vfmadd231ps %%ymm14, %%ymm12, %%ymm6\n\t"
      "vfmadd231ps %%ymm14, %%ymm13, %%ymm7\n\t"
      "vbroadcastss 16(%[ap]), %%ymm14\n\t"
      "vfmadd231ps %%ymm14, %%ymm12, %%ymm8\n\t"
      "vfmadd231ps %%ymm14, %%ymm13, %%ymm9\n\t"
      "vbroadcastss 20(%[ap]), %%ymm14\n\t"
      "vfmadd231ps %%ymm14, %%ymm12, %%ymm10\n\t"
      "vfmadd231ps %%ymm14, %%ymm13, %%ymm11\n\t"
      "add %[csa_bytes], %[ap]\n\t"
      "add %[rsb_bytes], %[bp]\n\t"

      /* Prefetch next A block */
      "prefetcht0 256(%[ap])\n\t"

      /* === K iteration 2 === */
      "vmovups (%[bp]), %%ymm12\n\t"
      "vmovups 32(%[bp]), %%ymm13\n\t"
      "vbroadcastss (%[ap]), %%ymm14\n\t"
      "vfmadd231ps %%ymm14, %%ymm12, %%ymm0\n\t"
      "vfmadd231ps %%ymm14, %%ymm13, %%ymm1\n\t"
      "vbroadcastss 4(%[ap]), %%ymm14\n\t"
      "vfmadd231ps %%ymm14, %%ymm12, %%ymm2\n\t"
      "vfmadd231ps %%ymm14, %%ymm13, %%ymm3\n\t"
      "vbroadcastss 8(%[ap]), %%ymm14\n\t"
      "vfmadd231ps %%ymm14, %%ymm12, %%ymm4\n\t"
      "vfmadd231ps %%ymm14, %%ymm13, %%ymm5\n\t"
      "vbroadcastss 12(%[ap]), %%ymm14\n\t"
      "vfmadd231ps %%ymm14, %%ymm12, %%ymm6\n\t"
      "vfmadd231ps %%ymm14, %%ymm13, %%ymm7\n\t"
      "vbroadcastss 16(%[ap]), %%ymm14\n\t"
      "vfmadd231ps %%ymm14, %%ymm12, %%ymm8\n\t"
      "vfmadd231ps %%ymm14, %%ymm13, %%ymm9\n\t"
      "vbroadcastss 20(%[ap]), %%ymm14\n\t"
      "vfmadd231ps %%ymm14, %%ymm12, %%ymm10\n\t"
      "vfmadd231ps %%ymm14, %%ymm13, %%ymm11\n\t"
      "add %[csa_bytes], %[ap]\n\t"
      "add %[rsb_bytes], %[bp]\n\t"

      /* === K iteration 3 === */
      "vmovups (%[bp]), %%ymm12\n\t"
      "vmovups 32(%[bp]), %%ymm13\n\t"
      "vbroadcastss (%[ap]), %%ymm14\n\t"
      "vfmadd231ps %%ymm14, %%ymm12, %%ymm0\n\t"
      "vfmadd231ps %%ymm14, %%ymm13, %%ymm1\n\t"
      "vbroadcastss 4(%[ap]), %%ymm14\n\t"
      "vfmadd231ps %%ymm14, %%ymm12, %%ymm2\n\t"
      "vfmadd231ps %%ymm14, %%ymm13, %%ymm3\n\t"
      "vbroadcastss 8(%[ap]), %%ymm14\n\t"
      "vfmadd231ps %%ymm14, %%ymm12, %%ymm4\n\t"
      "vfmadd231ps %%ymm14, %%ymm13, %%ymm5\n\t"
      "vbroadcastss 12(%[ap]), %%ymm14\n\t"
      "vfmadd231ps %%ymm14, %%ymm12, %%ymm6\n\t"
      "vfmadd231ps %%ymm14, %%ymm13, %%ymm7\n\t"
      "vbroadcastss 16(%[ap]), %%ymm14\n\t"
      "vfmadd231ps %%ymm14, %%ymm12, %%ymm8\n\t"
      "vfmadd231ps %%ymm14, %%ymm13, %%ymm9\n\t"
      "vbroadcastss 20(%[ap]), %%ymm14\n\t"
      "vfmadd231ps %%ymm14, %%ymm12, %%ymm10\n\t"
      "vfmadd231ps %%ymm14, %%ymm13, %%ymm11\n\t"
      "add %[csa_bytes], %[ap]\n\t"
      "add %[rsb_bytes], %[bp]\n\t"

      "dec %%rcx\n\t"
      "jnz 3b\n\t"

      "4:\n\t"
      /* Handle k_left = kc & 3 */
      "mov %[kc], %%rcx\n\t"
      "and $3, %%rcx\n\t"
      "jz 6f\n\t"

      ".p2align 4\n\t"
      "5:\n\t"
      "vmovups (%[bp]), %%ymm12\n\t"
      "vmovups 32(%[bp]), %%ymm13\n\t"
      "vbroadcastss (%[ap]), %%ymm14\n\t"
      "vfmadd231ps %%ymm14, %%ymm12, %%ymm0\n\t"
      "vfmadd231ps %%ymm14, %%ymm13, %%ymm1\n\t"
      "vbroadcastss 4(%[ap]), %%ymm14\n\t"
      "vfmadd231ps %%ymm14, %%ymm12, %%ymm2\n\t"
      "vfmadd231ps %%ymm14, %%ymm13, %%ymm3\n\t"
      "vbroadcastss 8(%[ap]), %%ymm14\n\t"
      "vfmadd231ps %%ymm14, %%ymm12, %%ymm4\n\t"
      "vfmadd231ps %%ymm14, %%ymm13, %%ymm5\n\t"
      "vbroadcastss 12(%[ap]), %%ymm14\n\t"
      "vfmadd231ps %%ymm14, %%ymm12, %%ymm6\n\t"
      "vfmadd231ps %%ymm14, %%ymm13, %%ymm7\n\t"
      "vbroadcastss 16(%[ap]), %%ymm14\n\t"
      "vfmadd231ps %%ymm14, %%ymm12, %%ymm8\n\t"
      "vfmadd231ps %%ymm14, %%ymm13, %%ymm9\n\t"
      "vbroadcastss 20(%[ap]), %%ymm14\n\t"
      "vfmadd231ps %%ymm14, %%ymm12, %%ymm10\n\t"
      "vfmadd231ps %%ymm14, %%ymm13, %%ymm11\n\t"
      "add %[csa_bytes], %[ap]\n\t"
      "add %[rsb_bytes], %[bp]\n\t"
      "dec %%rcx\n\t"
      "jnz 5b\n\t"

      "6:\n\t"
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
        "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14");
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

  /* Pointer-based iteration through packed A (stride MR=6) and B (stride
   * NR=16). 4× unrolled K-loop (BLIS pattern: k_iter = kc/4, k_left = kc%4). */
  const float *ap = a;
  const float *bp = b;
  size_t k_iter = kc / 4;
  size_t k_left = kc % 4;

  for (size_t ki = 0; ki < k_iter; ki++) {
    GEMM_F32_K_ITER(ap, bp);
    ap += csa;
    bp += rsb;
    GEMM_F32_K_ITER(ap, bp);
    ap += csa;
    bp += rsb;
    _mm_prefetch((const char *)(ap + 64), _MM_HINT_T0);
    GEMM_F32_K_ITER(ap, bp);
    ap += csa;
    bp += rsb;
    GEMM_F32_K_ITER(ap, bp);
    ap += csa;
    bp += rsb;
  }
  for (size_t ki = 0; ki < k_left; ki++) {
    GEMM_F32_K_ITER(ap, bp);
    ap += csa;
    bp += rsb;
  }

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
  /* B packing buffer — shared across threads, fits L3 */
  size_t nc_max = GEMM_MIN(GEMM_F32_NC, n_dim);
  float *packed_b = (float *)numc_malloc(
      32, GEMM_F32_KC * (nc_max + GEMM_F32_NR) * sizeof(float));
  if (!packed_b)
    return;

  for (size_t jc = 0; jc < n_dim; jc += GEMM_F32_NC) {
    size_t nc = GEMM_MIN(GEMM_F32_NC, n_dim - jc);

    for (size_t pc = 0; pc < k_dim; pc += GEMM_F32_KC) {
      size_t kc = GEMM_MIN(GEMM_F32_KC, k_dim - pc);
      int first = (pc == 0);

      gemm_pack_b_f32(b + pc * rsb + jc, packed_b, kc, nc, rsb);

      /* 2D IC×JR parallelism: linearize (ic_idx, jr_idx) into flat tasks.
       * With schedule(static), contiguous tasks share the same IC block,
       * so each thread packs A at most once or twice (at IC boundaries). */
      size_t n_ic = (m_dim + GEMM_F32_MC - 1) / GEMM_F32_MC;
      size_t n_jr = (nc + GEMM_F32_NR - 1) / GEMM_F32_NR;
      size_t n_tasks = n_ic * n_jr;

#ifdef _OPENMP
#pragma omp parallel if ((uint64_t)m_dim * kc * nc > GEMM_OMP_THRESHOLD)
      {
        NUMC_ALIGNAS(32) float packed_a[GEMM_F32_MC * GEMM_F32_KC];
        size_t last_ic = (size_t)-1;

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
              gemm_ukernel_f32_6x16(packed_a + ir * kc, packed_b + jr * kc,
                                    out + (ic + ir) * rso + (jc + jr), kc, 1,
                                    GEMM_F32_MR, GEMM_F32_NR, rso, first);
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
#else
      {
        NUMC_ALIGNAS(32) float packed_a[GEMM_F32_MC * GEMM_F32_KC];

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
              gemm_ukernel_f32_6x16(packed_a + ir * kc, packed_b + jr * kc,
                                    out + (ic + ir) * rso + (jc + jr), kc, 1,
                                    GEMM_F32_MR, GEMM_F32_NR, rso, first);
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
#endif
    }
  }

  numc_free(packed_b);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Float64: 6×8 micro-kernel  (12 acc + 1 A broadcast + 2 B loads = 15 YMM)
   ═══════════════════════════════════════════════════════════════════════════
 */

#define GEMM_F64_K_ITER(ap, bp)             \
  do {                                      \
    __m256d b0 = _mm256_loadu_pd(bp);       \
    __m256d b1 = _mm256_loadu_pd((bp) + 4); \
    __m256d av;                             \
    av = _mm256_broadcast_sd((ap) + 0);     \
    c00 = _mm256_fmadd_pd(av, b0, c00);     \
    c01 = _mm256_fmadd_pd(av, b1, c01);     \
    av = _mm256_broadcast_sd((ap) + 1);     \
    c10 = _mm256_fmadd_pd(av, b0, c10);     \
    c11 = _mm256_fmadd_pd(av, b1, c11);     \
    av = _mm256_broadcast_sd((ap) + 2);     \
    c20 = _mm256_fmadd_pd(av, b0, c20);     \
    c21 = _mm256_fmadd_pd(av, b1, c21);     \
    av = _mm256_broadcast_sd((ap) + 3);     \
    c30 = _mm256_fmadd_pd(av, b0, c30);     \
    c31 = _mm256_fmadd_pd(av, b1, c31);     \
    av = _mm256_broadcast_sd((ap) + 4);     \
    c40 = _mm256_fmadd_pd(av, b0, c40);     \
    c41 = _mm256_fmadd_pd(av, b1, c41);     \
    av = _mm256_broadcast_sd((ap) + 5);     \
    c50 = _mm256_fmadd_pd(av, b0, c50);     \
    c51 = _mm256_fmadd_pd(av, b1, c51);     \
  } while (0)

#if defined(__GNUC__) || defined(__clang__)

/* Inline-asm f64 6×8 micro-kernel: 12 accumulators pinned to ymm0-ymm11,
 * no register spills. Interleaves vbroadcastsd (port 5) with vfmadd231pd
 * (ports 0,1) for optimal port utilization. 4× unrolled K-loop. */
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
      /* K-loop: k_iter = kc >> 2 */
      "mov %[kc], %%rcx\n\t"
      "shr $2, %%rcx\n\t"
      "jz 4f\n\t"

      ".p2align 4\n\t"
      "3:\n\t"
      /* === K iteration 0 === */
      "vmovupd (%[bp]), %%ymm12\n\t"
      "vmovupd 32(%[bp]), %%ymm13\n\t"
      "vbroadcastsd (%[ap]), %%ymm14\n\t"
      "vfmadd231pd %%ymm14, %%ymm12, %%ymm0\n\t"
      "vfmadd231pd %%ymm14, %%ymm13, %%ymm1\n\t"
      "vbroadcastsd 8(%[ap]), %%ymm14\n\t"
      "vfmadd231pd %%ymm14, %%ymm12, %%ymm2\n\t"
      "vfmadd231pd %%ymm14, %%ymm13, %%ymm3\n\t"
      "vbroadcastsd 16(%[ap]), %%ymm14\n\t"
      "vfmadd231pd %%ymm14, %%ymm12, %%ymm4\n\t"
      "vfmadd231pd %%ymm14, %%ymm13, %%ymm5\n\t"
      "vbroadcastsd 24(%[ap]), %%ymm14\n\t"
      "vfmadd231pd %%ymm14, %%ymm12, %%ymm6\n\t"
      "vfmadd231pd %%ymm14, %%ymm13, %%ymm7\n\t"
      "vbroadcastsd 32(%[ap]), %%ymm14\n\t"
      "vfmadd231pd %%ymm14, %%ymm12, %%ymm8\n\t"
      "vfmadd231pd %%ymm14, %%ymm13, %%ymm9\n\t"
      "vbroadcastsd 40(%[ap]), %%ymm14\n\t"
      "vfmadd231pd %%ymm14, %%ymm12, %%ymm10\n\t"
      "vfmadd231pd %%ymm14, %%ymm13, %%ymm11\n\t"
      "add %[csa_bytes], %[ap]\n\t"
      "add %[rsb_bytes], %[bp]\n\t"

      /* === K iteration 1 === */
      "vmovupd (%[bp]), %%ymm12\n\t"
      "vmovupd 32(%[bp]), %%ymm13\n\t"
      "vbroadcastsd (%[ap]), %%ymm14\n\t"
      "vfmadd231pd %%ymm14, %%ymm12, %%ymm0\n\t"
      "vfmadd231pd %%ymm14, %%ymm13, %%ymm1\n\t"
      "vbroadcastsd 8(%[ap]), %%ymm14\n\t"
      "vfmadd231pd %%ymm14, %%ymm12, %%ymm2\n\t"
      "vfmadd231pd %%ymm14, %%ymm13, %%ymm3\n\t"
      "vbroadcastsd 16(%[ap]), %%ymm14\n\t"
      "vfmadd231pd %%ymm14, %%ymm12, %%ymm4\n\t"
      "vfmadd231pd %%ymm14, %%ymm13, %%ymm5\n\t"
      "vbroadcastsd 24(%[ap]), %%ymm14\n\t"
      "vfmadd231pd %%ymm14, %%ymm12, %%ymm6\n\t"
      "vfmadd231pd %%ymm14, %%ymm13, %%ymm7\n\t"
      "vbroadcastsd 32(%[ap]), %%ymm14\n\t"
      "vfmadd231pd %%ymm14, %%ymm12, %%ymm8\n\t"
      "vfmadd231pd %%ymm14, %%ymm13, %%ymm9\n\t"
      "vbroadcastsd 40(%[ap]), %%ymm14\n\t"
      "vfmadd231pd %%ymm14, %%ymm12, %%ymm10\n\t"
      "vfmadd231pd %%ymm14, %%ymm13, %%ymm11\n\t"
      "add %[csa_bytes], %[ap]\n\t"
      "add %[rsb_bytes], %[bp]\n\t"

      /* Prefetch next A block */
      "prefetcht0 384(%[ap])\n\t"

      /* === K iteration 2 === */
      "vmovupd (%[bp]), %%ymm12\n\t"
      "vmovupd 32(%[bp]), %%ymm13\n\t"
      "vbroadcastsd (%[ap]), %%ymm14\n\t"
      "vfmadd231pd %%ymm14, %%ymm12, %%ymm0\n\t"
      "vfmadd231pd %%ymm14, %%ymm13, %%ymm1\n\t"
      "vbroadcastsd 8(%[ap]), %%ymm14\n\t"
      "vfmadd231pd %%ymm14, %%ymm12, %%ymm2\n\t"
      "vfmadd231pd %%ymm14, %%ymm13, %%ymm3\n\t"
      "vbroadcastsd 16(%[ap]), %%ymm14\n\t"
      "vfmadd231pd %%ymm14, %%ymm12, %%ymm4\n\t"
      "vfmadd231pd %%ymm14, %%ymm13, %%ymm5\n\t"
      "vbroadcastsd 24(%[ap]), %%ymm14\n\t"
      "vfmadd231pd %%ymm14, %%ymm12, %%ymm6\n\t"
      "vfmadd231pd %%ymm14, %%ymm13, %%ymm7\n\t"
      "vbroadcastsd 32(%[ap]), %%ymm14\n\t"
      "vfmadd231pd %%ymm14, %%ymm12, %%ymm8\n\t"
      "vfmadd231pd %%ymm14, %%ymm13, %%ymm9\n\t"
      "vbroadcastsd 40(%[ap]), %%ymm14\n\t"
      "vfmadd231pd %%ymm14, %%ymm12, %%ymm10\n\t"
      "vfmadd231pd %%ymm14, %%ymm13, %%ymm11\n\t"
      "add %[csa_bytes], %[ap]\n\t"
      "add %[rsb_bytes], %[bp]\n\t"

      /* === K iteration 3 === */
      "vmovupd (%[bp]), %%ymm12\n\t"
      "vmovupd 32(%[bp]), %%ymm13\n\t"
      "vbroadcastsd (%[ap]), %%ymm14\n\t"
      "vfmadd231pd %%ymm14, %%ymm12, %%ymm0\n\t"
      "vfmadd231pd %%ymm14, %%ymm13, %%ymm1\n\t"
      "vbroadcastsd 8(%[ap]), %%ymm14\n\t"
      "vfmadd231pd %%ymm14, %%ymm12, %%ymm2\n\t"
      "vfmadd231pd %%ymm14, %%ymm13, %%ymm3\n\t"
      "vbroadcastsd 16(%[ap]), %%ymm14\n\t"
      "vfmadd231pd %%ymm14, %%ymm12, %%ymm4\n\t"
      "vfmadd231pd %%ymm14, %%ymm13, %%ymm5\n\t"
      "vbroadcastsd 24(%[ap]), %%ymm14\n\t"
      "vfmadd231pd %%ymm14, %%ymm12, %%ymm6\n\t"
      "vfmadd231pd %%ymm14, %%ymm13, %%ymm7\n\t"
      "vbroadcastsd 32(%[ap]), %%ymm14\n\t"
      "vfmadd231pd %%ymm14, %%ymm12, %%ymm8\n\t"
      "vfmadd231pd %%ymm14, %%ymm13, %%ymm9\n\t"
      "vbroadcastsd 40(%[ap]), %%ymm14\n\t"
      "vfmadd231pd %%ymm14, %%ymm12, %%ymm10\n\t"
      "vfmadd231pd %%ymm14, %%ymm13, %%ymm11\n\t"
      "add %[csa_bytes], %[ap]\n\t"
      "add %[rsb_bytes], %[bp]\n\t"

      "dec %%rcx\n\t"
      "jnz 3b\n\t"

      "4:\n\t"
      /* Handle k_left = kc & 3 */
      "mov %[kc], %%rcx\n\t"
      "and $3, %%rcx\n\t"
      "jz 6f\n\t"

      ".p2align 4\n\t"
      "5:\n\t"
      "vmovupd (%[bp]), %%ymm12\n\t"
      "vmovupd 32(%[bp]), %%ymm13\n\t"
      "vbroadcastsd (%[ap]), %%ymm14\n\t"
      "vfmadd231pd %%ymm14, %%ymm12, %%ymm0\n\t"
      "vfmadd231pd %%ymm14, %%ymm13, %%ymm1\n\t"
      "vbroadcastsd 8(%[ap]), %%ymm14\n\t"
      "vfmadd231pd %%ymm14, %%ymm12, %%ymm2\n\t"
      "vfmadd231pd %%ymm14, %%ymm13, %%ymm3\n\t"
      "vbroadcastsd 16(%[ap]), %%ymm14\n\t"
      "vfmadd231pd %%ymm14, %%ymm12, %%ymm4\n\t"
      "vfmadd231pd %%ymm14, %%ymm13, %%ymm5\n\t"
      "vbroadcastsd 24(%[ap]), %%ymm14\n\t"
      "vfmadd231pd %%ymm14, %%ymm12, %%ymm6\n\t"
      "vfmadd231pd %%ymm14, %%ymm13, %%ymm7\n\t"
      "vbroadcastsd 32(%[ap]), %%ymm14\n\t"
      "vfmadd231pd %%ymm14, %%ymm12, %%ymm8\n\t"
      "vfmadd231pd %%ymm14, %%ymm13, %%ymm9\n\t"
      "vbroadcastsd 40(%[ap]), %%ymm14\n\t"
      "vfmadd231pd %%ymm14, %%ymm12, %%ymm10\n\t"
      "vfmadd231pd %%ymm14, %%ymm13, %%ymm11\n\t"
      "add %[csa_bytes], %[ap]\n\t"
      "add %[rsb_bytes], %[bp]\n\t"
      "dec %%rcx\n\t"
      "jnz 5b\n\t"

      "6:\n\t"
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
        "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14");
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
  size_t k_iter = kc / 4;
  size_t k_left = kc % 4;

  for (size_t ki = 0; ki < k_iter; ki++) {
    GEMM_F64_K_ITER(ap, bp);
    ap += csa;
    bp += rsb;
    GEMM_F64_K_ITER(ap, bp);
    ap += csa;
    bp += rsb;
    _mm_prefetch((const char *)(ap + 48), _MM_HINT_T0);
    GEMM_F64_K_ITER(ap, bp);
    ap += csa;
    bp += rsb;
    GEMM_F64_K_ITER(ap, bp);
    ap += csa;
    bp += rsb;
  }
  for (size_t ki = 0; ki < k_left; ki++) {
    GEMM_F64_K_ITER(ap, bp);
    ap += csa;
    bp += rsb;
  }

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

    for (size_t pc = 0; pc < k_dim; pc += GEMM_F64_KC) {
      size_t kc = GEMM_MIN(GEMM_F64_KC, k_dim - pc);
      int first = (pc == 0);

      gemm_pack_b_f64(b + pc * rsb + jc, packed_b, kc, nc, rsb);

      size_t n_ic = (m_dim + GEMM_F64_MC - 1) / GEMM_F64_MC;
      size_t n_jr = (nc + GEMM_F64_NR - 1) / GEMM_F64_NR;
      size_t n_tasks = n_ic * n_jr;

#ifdef _OPENMP
#pragma omp parallel if ((uint64_t)m_dim * kc * nc > GEMM_OMP_THRESHOLD)
      {
        NUMC_ALIGNAS(32) double packed_a[GEMM_F64_MC * GEMM_F64_KC];
        size_t last_ic = (size_t)-1;

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
              gemm_ukernel_f64_6x8(packed_a + ir * kc, packed_b + jr * kc,
                                   out + (ic + ir) * rso + (jc + jr), kc, 1,
                                   GEMM_F64_MR, GEMM_F64_NR, rso, first);
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
#else
      {
        NUMC_ALIGNAS(32) double packed_a[GEMM_F64_MC * GEMM_F64_KC];

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
              gemm_ukernel_f64_6x8(packed_a + ir * kc, packed_b + jr * kc,
                                   out + (ic + ir) * rso + (jc + jr), kc, 1,
                                   GEMM_F64_MR, GEMM_F64_NR, rso, first);
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
#endif
    }
  }

  numc_free(packed_b);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Int32/Uint32: 6×16 micro-kernel  (mullo_epi32 + add_epi32)
   mullo_epi32 produces identical low 32 bits for signed and unsigned.
   ═══════════════════════════════════════════════════════════════════════════
 */

static inline void gemm_ukernel_i32_6x16(const int32_t *a, const int32_t *b,
                                         int32_t *c, size_t kc, intptr_t rsa,
                                         intptr_t csa, intptr_t rsb,
                                         intptr_t rso) {
  __m256i c00 = _mm256_loadu_si256((__m256i *)(c)),
          c01 = _mm256_loadu_si256((__m256i *)(c + 8));
  __m256i c10 = _mm256_loadu_si256((__m256i *)(c + rso)),
          c11 = _mm256_loadu_si256((__m256i *)(c + rso + 8));
  __m256i c20 = _mm256_loadu_si256((__m256i *)(c + 2 * rso)),
          c21 = _mm256_loadu_si256((__m256i *)(c + 2 * rso + 8));
  __m256i c30 = _mm256_loadu_si256((__m256i *)(c + 3 * rso)),
          c31 = _mm256_loadu_si256((__m256i *)(c + 3 * rso + 8));
  __m256i c40 = _mm256_loadu_si256((__m256i *)(c + 4 * rso)),
          c41 = _mm256_loadu_si256((__m256i *)(c + 4 * rso + 8));
  __m256i c50 = _mm256_loadu_si256((__m256i *)(c + 5 * rso)),
          c51 = _mm256_loadu_si256((__m256i *)(c + 5 * rso + 8));

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
  for (size_t i = 0; i < m_dim; i++)
    memset(out + i * rso, 0, n_dim * sizeof(int32_t));

  for (size_t pc = 0; pc < k_dim; pc += GEMM_I32_KC) {
    size_t kc = GEMM_MIN(GEMM_I32_KC, k_dim - pc);
#ifdef _OPENMP
#pragma omp parallel for schedule( \
        static) if (m_dim * n_dim * sizeof(int32_t) > GEMM_OMP_THRESHOLD)
#endif
    for (size_t ic = 0; ic < m_dim; ic += GEMM_I32_MC) {
      size_t mc = GEMM_MIN(GEMM_I32_MC, m_dim - ic);
      size_t jr = 0;
      for (; jr + GEMM_I32_NR <= n_dim; jr += GEMM_I32_NR) {
        size_t ir = 0;
        for (; ir + GEMM_I32_MR <= mc; ir += GEMM_I32_MR)
          gemm_ukernel_i32_6x16(a + (ic + ir) * rsa + pc * csa,
                                b + pc * rsb + jr, out + (ic + ir) * rso + jr,
                                kc, rsa, csa, rsb, rso);
        if (ir < mc)
          gemm_edge_i32(a + (ic + ir) * rsa + pc * csa, b + pc * rsb + jr,
                        out + (ic + ir) * rso + jr, mc - ir, GEMM_I32_NR, kc,
                        rsa, csa, rsb, rso);
      }
      if (jr < n_dim)
        gemm_edge_i32(a + ic * rsa + pc * csa, b + pc * rsb + jr,
                      out + ic * rso + jr, mc, n_dim - jr, kc, rsa, csa, rsb,
                      rso);
    }
  }
}

/* Uint32: identical bit-level operations as int32 */
static inline void gemm_u32_avx2(const uint32_t *a, const uint32_t *b,
                                 uint32_t *out, size_t m_dim, size_t k_dim,
                                 size_t n_dim, intptr_t rsa, intptr_t csa,
                                 intptr_t rsb, intptr_t rso) {
  gemm_i32_avx2((const int32_t *)a, (const int32_t *)b, (int32_t *)out, m_dim,
                k_dim, n_dim, rsa, csa, rsb, rso);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Int16/Uint16: 6×32 micro-kernel  (mullo_epi16 + add_epi16, 16 elem/reg)
   Same-width accumulation — matches the i32 overflow trade-off.
   ═══════════════════════════════════════════════════════════════════════════
 */

static inline void gemm_ukernel_i16_6x32(const int16_t *a, const int16_t *b,
                                         int16_t *c, size_t kc, intptr_t rsa,
                                         intptr_t csa, intptr_t rsb,
                                         intptr_t rso) {
  __m256i c00 = _mm256_loadu_si256((__m256i *)(c)),
          c01 = _mm256_loadu_si256((__m256i *)(c + 16));
  __m256i c10 = _mm256_loadu_si256((__m256i *)(c + rso)),
          c11 = _mm256_loadu_si256((__m256i *)(c + rso + 16));
  __m256i c20 = _mm256_loadu_si256((__m256i *)(c + 2 * rso)),
          c21 = _mm256_loadu_si256((__m256i *)(c + 2 * rso + 16));
  __m256i c30 = _mm256_loadu_si256((__m256i *)(c + 3 * rso)),
          c31 = _mm256_loadu_si256((__m256i *)(c + 3 * rso + 16));
  __m256i c40 = _mm256_loadu_si256((__m256i *)(c + 4 * rso)),
          c41 = _mm256_loadu_si256((__m256i *)(c + 4 * rso + 16));
  __m256i c50 = _mm256_loadu_si256((__m256i *)(c + 5 * rso)),
          c51 = _mm256_loadu_si256((__m256i *)(c + 5 * rso + 16));

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

static inline void gemm_i16_avx2(const int16_t *a, const int16_t *b,
                                 int16_t *out, size_t m_dim, size_t k_dim,
                                 size_t n_dim, intptr_t rsa, intptr_t csa,
                                 intptr_t rsb, intptr_t rso) {
  for (size_t i = 0; i < m_dim; i++)
    memset(out + i * rso, 0, n_dim * sizeof(int16_t));

  for (size_t pc = 0; pc < k_dim; pc += GEMM_I16_KC) {
    size_t kc = GEMM_MIN(GEMM_I16_KC, k_dim - pc);
#ifdef _OPENMP
#pragma omp parallel for schedule( \
        static) if (m_dim * n_dim * sizeof(int16_t) > GEMM_OMP_THRESHOLD)
#endif
    for (size_t ic = 0; ic < m_dim; ic += GEMM_I16_MC) {
      size_t mc = GEMM_MIN(GEMM_I16_MC, m_dim - ic);
      size_t jr = 0;
      for (; jr + GEMM_I16_NR <= n_dim; jr += GEMM_I16_NR) {
        size_t ir = 0;
        for (; ir + GEMM_I16_MR <= mc; ir += GEMM_I16_MR)
          gemm_ukernel_i16_6x32(a + (ic + ir) * rsa + pc * csa,
                                b + pc * rsb + jr, out + (ic + ir) * rso + jr,
                                kc, rsa, csa, rsb, rso);
        if (ir < mc)
          gemm_edge_i16(a + (ic + ir) * rsa + pc * csa, b + pc * rsb + jr,
                        out + (ic + ir) * rso + jr, mc - ir, GEMM_I16_NR, kc,
                        rsa, csa, rsb, rso);
      }
      if (jr < n_dim)
        gemm_edge_i16(a + ic * rsa + pc * csa, b + pc * rsb + jr,
                      out + ic * rso + jr, mc, n_dim - jr, kc, rsa, csa, rsb,
                      rso);
    }
  }
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
   Int8/Uint8: 6×8 promoted micro-kernel  (widen to i32, full-K accumulation)
   Uses saturation packing for stores to match scalar behavior more efficiently.
   ═══════════════════════════════════════════════════════════════════════════
 */

static inline void gemm_ukernel_i8_6x8(const int8_t *a, const int8_t *b,
                                       int8_t *c, size_t k_dim, intptr_t rsa,
                                       intptr_t csa, intptr_t rsb,
                                       intptr_t rso) {
  __m256i c00 = _mm256_setzero_si256(), c10 = _mm256_setzero_si256(),
          c20 = _mm256_setzero_si256(), c30 = _mm256_setzero_si256(),
          c40 = _mm256_setzero_si256(), c50 = _mm256_setzero_si256();

  for (size_t p = 0; p < k_dim; p++) {
    const int8_t *bp = b + p * rsb;
    __m256i b0 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i const *)bp));
    __m256i av;
    av = _mm256_set1_epi32((int32_t)a[0 * rsa + p * csa]);
    c00 = _mm256_add_epi32(c00, _mm256_mullo_epi32(av, b0));
    av = _mm256_set1_epi32((int32_t)a[1 * rsa + p * csa]);
    c10 = _mm256_add_epi32(c10, _mm256_mullo_epi32(av, b0));
    av = _mm256_set1_epi32((int32_t)a[2 * rsa + p * csa]);
    c20 = _mm256_add_epi32(c20, _mm256_mullo_epi32(av, b0));
    av = _mm256_set1_epi32((int32_t)a[3 * rsa + p * csa]);
    c30 = _mm256_add_epi32(c30, _mm256_mullo_epi32(av, b0));
    av = _mm256_set1_epi32((int32_t)a[4 * rsa + p * csa]);
    c40 = _mm256_add_epi32(c40, _mm256_mullo_epi32(av, b0));
    av = _mm256_set1_epi32((int32_t)a[5 * rsa + p * csa]);
    c50 = _mm256_add_epi32(c50, _mm256_mullo_epi32(av, b0));
  }

#define GEMM_STORE_I8_ROW(acc, row)                          \
  do {                                                       \
    __m128i lo = _mm256_castsi256_si128(acc);                \
    __m128i hi = _mm256_extracti128_si256(acc, 1);           \
    __m128i p16 = _mm_packs_epi32(lo, hi);                   \
    __m128i p8 = _mm_packs_epi16(p16, p16);                  \
    *((int64_t *)(c + (row) * rso)) = _mm_cvtsi128_si64(p8); \
  } while (0)
  GEMM_STORE_I8_ROW(c00, 0);
  GEMM_STORE_I8_ROW(c10, 1);
  GEMM_STORE_I8_ROW(c20, 2);
  GEMM_STORE_I8_ROW(c30, 3);
  GEMM_STORE_I8_ROW(c40, 4);
  GEMM_STORE_I8_ROW(c50, 5);
#undef GEMM_STORE_I8_ROW
}

static inline void gemm_edge_i8(const int8_t *a, const int8_t *b, int8_t *c,
                                size_t mr, size_t nr, size_t k_dim,
                                intptr_t rsa, intptr_t csa, intptr_t rsb,
                                intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
    for (size_t j = 0; j < nr; j++) {
      int32_t acc = 0;
      for (size_t p = 0; p < k_dim; p++)
        acc += (int32_t)a[i * rsa + p * csa] * (int32_t)b[p * rsb + j];
      c[i * rso + j] = (int8_t)acc;
    }
  }
}

static inline void gemm_i8_avx2(const int8_t *a, const int8_t *b, int8_t *out,
                                size_t m_dim, size_t k_dim, size_t n_dim,
                                intptr_t rsa, intptr_t csa, intptr_t rsb,
                                intptr_t rso) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (m_dim * n_dim > \
                                                  GEMM_OMP_THRESHOLD)
#endif
  for (size_t ic = 0; ic < m_dim; ic += GEMM_I8_MC) {
    size_t mc = GEMM_MIN(GEMM_I8_MC, m_dim - ic);
    size_t jr = 0;
    for (; jr + GEMM_I8_NR <= n_dim; jr += GEMM_I8_NR) {
      size_t ir = 0;
      for (; ir + GEMM_I8_MR <= mc; ir += GEMM_I8_MR)
        gemm_ukernel_i8_6x8(a + (ic + ir) * rsa, b + jr,
                            out + (ic + ir) * rso + jr, k_dim, rsa, csa, rsb,
                            rso);
      if (ir < mc)
        gemm_edge_i8(a + (ic + ir) * rsa, b + jr, out + (ic + ir) * rso + jr,
                     mc - ir, GEMM_I8_NR, k_dim, rsa, csa, rsb, rso);
    }
    if (jr < n_dim)
      gemm_edge_i8(a + ic * rsa, b + jr, out + ic * rso + jr, mc, n_dim - jr,
                   k_dim, rsa, csa, rsb, rso);
  }
}

static inline void gemm_ukernel_u8_6x8(const uint8_t *a, const uint8_t *b,
                                       uint8_t *c, size_t k_dim, intptr_t rsa,
                                       intptr_t csa, intptr_t rsb,
                                       intptr_t rso) {
  __m256i c00 = _mm256_setzero_si256(), c10 = _mm256_setzero_si256(),
          c20 = _mm256_setzero_si256(), c30 = _mm256_setzero_si256(),
          c40 = _mm256_setzero_si256(), c50 = _mm256_setzero_si256();

  for (size_t p = 0; p < k_dim; p++) {
    const uint8_t *bp = b + p * rsb;
    __m256i b0 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i const *)bp));
    __m256i av;
    av = _mm256_set1_epi32((int32_t)(uint32_t)a[0 * rsa + p * csa]);
    c00 = _mm256_add_epi32(c00, _mm256_mullo_epi32(av, b0));
    av = _mm256_set1_epi32((int32_t)(uint32_t)a[1 * rsa + p * csa]);
    c10 = _mm256_add_epi32(c10, _mm256_mullo_epi32(av, b0));
    av = _mm256_set1_epi32((int32_t)(uint32_t)a[2 * rsa + p * csa]);
    c20 = _mm256_add_epi32(c20, _mm256_mullo_epi32(av, b0));
    av = _mm256_set1_epi32((int32_t)(uint32_t)a[3 * rsa + p * csa]);
    c30 = _mm256_add_epi32(c30, _mm256_mullo_epi32(av, b0));
    av = _mm256_set1_epi32((int32_t)(uint32_t)a[4 * rsa + p * csa]);
    c40 = _mm256_add_epi32(c40, _mm256_mullo_epi32(av, b0));
    av = _mm256_set1_epi32((int32_t)(uint32_t)a[5 * rsa + p * csa]);
    c50 = _mm256_add_epi32(c50, _mm256_mullo_epi32(av, b0));
  }

#define GEMM_STORE_U8_ROW(acc, row)                           \
  do {                                                        \
    __m128i lo = _mm256_castsi256_si128(acc);                 \
    __m128i hi = _mm256_extracti128_si256(acc, 1);            \
    __m128i p16 = _mm_packus_epi32(lo, hi);                   \
    __m128i p8 = _mm_packus_epi16(p16, p16);                  \
    *((uint64_t *)(c + (row) * rso)) = _mm_cvtsi128_si64(p8); \
  } while (0)
  GEMM_STORE_U8_ROW(c00, 0);
  GEMM_STORE_U8_ROW(c10, 1);
  GEMM_STORE_U8_ROW(c20, 2);
  GEMM_STORE_U8_ROW(c30, 3);
  GEMM_STORE_U8_ROW(c40, 4);
  GEMM_STORE_U8_ROW(c50, 5);
#undef GEMM_STORE_U8_ROW
}

static inline void gemm_edge_u8(const uint8_t *a, const uint8_t *b, uint8_t *c,
                                size_t mr, size_t nr, size_t k_dim,
                                intptr_t rsa, intptr_t csa, intptr_t rsb,
                                intptr_t rso) {
  for (size_t i = 0; i < mr; i++) {
    for (size_t j = 0; j < nr; j++) {
      uint32_t acc = 0;
      for (size_t p = 0; p < k_dim; p++)
        acc += (uint32_t)a[i * rsa + p * csa] * (uint32_t)b[p * rsb + j];
      c[i * rso + j] = (uint8_t)acc;
    }
  }
}

static inline void gemm_u8_avx2(const uint8_t *a, const uint8_t *b,
                                uint8_t *out, size_t m_dim, size_t k_dim,
                                size_t n_dim, intptr_t rsa, intptr_t csa,
                                intptr_t rsb, intptr_t rso) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (m_dim * n_dim > \
                                                  GEMM_OMP_THRESHOLD)
#endif
  for (size_t ic = 0; ic < m_dim; ic += GEMM_I8_MC) {
    size_t mc = GEMM_MIN(GEMM_I8_MC, m_dim - ic);
    size_t jr = 0;
    for (; jr + GEMM_I8_NR <= n_dim; jr += GEMM_I8_NR) {
      size_t ir = 0;
      for (; ir + GEMM_I8_MR <= mc; ir += GEMM_I8_MR)
        gemm_ukernel_u8_6x8(a + (ic + ir) * rsa, b + jr,
                            out + (ic + ir) * rso + jr, k_dim, rsa, csa, rsb,
                            rso);
      if (ir < mc)
        gemm_edge_u8(a + (ic + ir) * rsa, b + jr, out + (ic + ir) * rso + jr,
                     mc - ir, GEMM_I8_NR, k_dim, rsa, csa, rsb, rso);
    }
    if (jr < n_dim)
      gemm_edge_u8(a + ic * rsa, b + jr, out + ic * rso + jr, mc, n_dim - jr,
                   k_dim, rsa, csa, rsb, rso);
  }
}

#undef GEMM_MIN

#endif
