#ifndef NUMC_REDUCTION_KERNEL_H
#define NUMC_REDUCTION_KERNEL_H

#include "internal.h"

typedef void (*NumcReductionKernel)(const char *a, char *out, size_t n,
                                    intptr_t sa);

#define DEFINE_REDUCTION_KERNEL(OP_NAME, TYPE_ENUM, C_TYPE, INIT, EXPR,        \
                                OMP_OP)                                        \
  static void _kern_##OP_NAME##_##TYPE_ENUM(const char *a, char *out,          \
                                            size_t n, intptr_t sa) {           \
    if (n == 0) {                                                              \
      *(C_TYPE *)out = (INIT);                                                 \
      return;                                                                  \
    }                                                                          \
    C_TYPE acc = (INIT);                                                       \
    if (sa == (intptr_t)sizeof(C_TYPE)) {                                      \
      /* PATH 1: contiguous — indexed access, auto-vectorizes + OMP */         \
      const C_TYPE *restrict pa = (const C_TYPE *)a;                           \
      NUMC_OMP_REDUCE_FOR(                                                     \
          n, sizeof(C_TYPE), OMP_OP, acc, for (size_t i = 0; i < n; i++) {     \
            C_TYPE val = pa[i];                                                \
            acc = (EXPR);                                                      \
          });                                                                  \
    } else {                                                                   \
      /* PATH 2: strided — no OMP */                                           \
      for (size_t i = 0; i < n; i++) {                                         \
        C_TYPE val = *(const C_TYPE *)(a + i * sa);                            \
        acc = (EXPR);                                                          \
      }                                                                        \
    }                                                                          \
    *(C_TYPE *)out = acc;                                                      \
  }

/* Float reduction kernel with OMP-parallel vectorized helper.
 * Used for sum (pairwise), max, min — replaces hand-written copies.
 *
 * HELPER_FN:    vectorized helper, e.g. _pairwise_sum_f32, _vec_max_f64
 * OMP_OP:       OMP reduction clause operator (+, max, min)
 * COMBINE:      how per-thread result merges into global
 *               sum: global += local
 *               max: if (local > global) global = local
 *               min: if (local < global) global = local
 * STRIDED_EXPR: fallback expression for non-contiguous (uses acc, val)
 */
#define DEFINE_FLOAT_REDUCTION_KERNEL(OP_NAME, TYPE_ENUM, C_TYPE, INIT,        \
                                      HELPER_FN, OMP_OP, COMBINE,             \
                                      STRIDED_EXPR)                            \
  static void _kern_##OP_NAME##_##TYPE_ENUM(const char *a, char *out,          \
                                            size_t n, intptr_t sa) {           \
    if (n == 0) {                                                              \
      *(C_TYPE *)out = (INIT);                                                 \
      return;                                                                  \
    }                                                                          \
    if (sa == (intptr_t)sizeof(C_TYPE)) {                                      \
      const C_TYPE *pa = (const C_TYPE *)a;                                    \
      size_t total_bytes = n * sizeof(C_TYPE);                                 \
      if (total_bytes > NUMC_OMP_BYTE_THRESHOLD) {                             \
        int nt = (int)(total_bytes / NUMC_OMP_BYTES_PER_THREAD);               \
        if (nt < 1) nt = 1;                                                   \
        C_TYPE global = (INIT);                                                \
        NUMC_PRAGMA(omp parallel for reduction(OMP_OP:global)                  \
                        schedule(static) num_threads(nt))                      \
        for (int t = 0; t < nt; t++) {                                         \
          size_t start = (size_t)t * (n / nt);                                 \
          size_t end = (t == nt - 1) ? n : start + n / nt;                     \
          C_TYPE local = HELPER_FN(pa + start, end - start);                   \
          COMBINE;                                                             \
        }                                                                      \
        *(C_TYPE *)out = global;                                               \
      } else {                                                                 \
        *(C_TYPE *)out = HELPER_FN(pa, n);                                     \
      }                                                                        \
    } else {                                                                   \
      C_TYPE acc = (INIT);                                                     \
      for (size_t i = 0; i < n; i++) {                                         \
        C_TYPE val = *(const C_TYPE *)(a + i * sa);                            \
        acc = (STRIDED_EXPR);                                                  \
      }                                                                        \
      *(C_TYPE *)out = acc;                                                    \
    }                                                                          \
  }

/* Arg-reduction kernel: finds the INDEX of the max/min element.
 * Output is always int64_t regardless of input type.
 *
 * Two-pass algorithm for contiguous data:
 *   Pass 1: find extreme value (auto-vectorizes to vpmaxsb/vmaxps etc.)
 *   Pass 2: find first index matching that value (early-exit scan)
 * Data stays cache-hot between passes. Pass 1 is the bottleneck;
 * pass 2 exits on the first match.
 *
 * Integer variant: pass 1 ternary auto-vectorizes directly.
 * Float variant: uses multi-accumulator helper (same as max/min kernels)
 * because single-accumulator float ternary doesn't auto-vectorize. */
#define DEFINE_ARGREDUCTION_KERNEL(OP_NAME, TYPE_ENUM, C_TYPE, INIT, CMP)      \
  static void _kern_##OP_NAME##_##TYPE_ENUM(const char *a, char *out,          \
                                            size_t n, intptr_t sa) {           \
    if (n == 0) {                                                              \
      *(int64_t *)out = 0;                                                     \
      return;                                                                  \
    }                                                                          \
    if (sa == (intptr_t)sizeof(C_TYPE)) {                                      \
      const C_TYPE *restrict pa = (const C_TYPE *)a;                           \
      /* PASS 1: find extreme value (auto-vectorizes) */                       \
      C_TYPE best = (INIT);                                                    \
      for (size_t i = 0; i < n; i++) {                                         \
        C_TYPE v = pa[i];                                                      \
        best = v CMP best ? v : best;                                          \
      }                                                                        \
      /* PASS 2: find first matching index (early-exit) */                     \
      for (size_t i = 0; i < n; i++) {                                         \
        if (pa[i] == best) {                                                   \
          *(int64_t *)out = (int64_t)i;                                        \
          return;                                                              \
        }                                                                      \
      }                                                                        \
    } else {                                                                   \
      /* Strided: single-pass scalar (not vectorizable anyway) */              \
      C_TYPE best_val = (INIT);                                                \
      int64_t best_idx = 0;                                                    \
      for (size_t i = 0; i < n; i++) {                                         \
        C_TYPE val = *(const C_TYPE *)(a + i * sa);                            \
        if (val CMP best_val) {                                                \
          best_val = val;                                                      \
          best_idx = (int64_t)i;                                               \
        }                                                                      \
      }                                                                        \
      *(int64_t *)out = best_idx;                                              \
    }                                                                          \
  }

/* Float arg-reduction: pass 1 uses multi-accumulator helper (SLP-vectorizes
 * to vmaxps/vminps), pass 2 is same early-exit equality scan. */
#define DEFINE_FLOAT_ARGREDUCTION_KERNEL(OP_NAME, TYPE_ENUM, C_TYPE, INIT,     \
                                         HELPER_FN, CMP)                       \
  static void _kern_##OP_NAME##_##TYPE_ENUM(const char *a, char *out,          \
                                            size_t n, intptr_t sa) {           \
    if (n == 0) {                                                              \
      *(int64_t *)out = 0;                                                     \
      return;                                                                  \
    }                                                                          \
    if (sa == (intptr_t)sizeof(C_TYPE)) {                                      \
      const C_TYPE *restrict pa = (const C_TYPE *)a;                           \
      /* PASS 1: multi-accumulator max/min (SLP-vectorizes) */                 \
      C_TYPE best = HELPER_FN(pa, n);                                          \
      /* PASS 2: find first matching index (early-exit) */                     \
      for (size_t i = 0; i < n; i++) {                                         \
        if (pa[i] == best) {                                                   \
          *(int64_t *)out = (int64_t)i;                                        \
          return;                                                              \
        }                                                                      \
      }                                                                        \
      /* Fallback (e.g. all-NaN: best is INIT, no match) */                    \
      *(int64_t *)out = 0;                                                     \
    } else {                                                                   \
      /* Strided: single-pass scalar */                                        \
      C_TYPE best_val = (INIT);                                                \
      int64_t best_idx = 0;                                                    \
      for (size_t i = 0; i < n; i++) {                                         \
        C_TYPE val = *(const C_TYPE *)(a + i * sa);                            \
        if (val CMP best_val) {                                                \
          best_val = val;                                                      \
          best_idx = (int64_t)i;                                               \
        }                                                                      \
      }                                                                        \
      *(int64_t *)out = best_idx;                                              \
    }                                                                          \
  }

#endif
