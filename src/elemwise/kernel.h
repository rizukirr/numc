#ifndef NUMC_MATH_KERNEL_H
#define NUMC_MATH_KERNEL_H

#include "internal.h"
#include <numc/dtype.h>

/* ── Kernel function pointer typedefs ──────────────────────────────── */

typedef void (*NumcBinaryKernel)(const char *a, const char *b, char *out,
                                 size_t n, intptr_t sa, intptr_t sb,
                                 intptr_t so);

typedef void (*NumcUnaryKernel)(const char *a, char *out, size_t n, intptr_t sa,
                                intptr_t so);

typedef void (*NumcClipKernel)(const char *a, char *out, size_t n, intptr_t sa,
                               intptr_t so, double min, double max);

/* ── Stride-aware binary kernel macro ──────────────────────────────
 *
 * Three runtime paths:
 *   PATH 1 — Contiguous:       sa == sb == so == sizeof(T)
 *   PATH 2 — Scalar broadcast: sb == 0, a and out contiguous
 *   PATH 3 — Generic strided:  arbitrary sa, sb, so
 */

#define DEFINE_BINARY_KERNEL(OP_NAME, TYPE_ENUM, C_TYPE, EXPR)                 \
  static void _kern_##OP_NAME##_##TYPE_ENUM(const char *a, const char *b,      \
                                            char *out, size_t n, intptr_t sa,  \
                                            intptr_t sb, intptr_t so) {        \
    const intptr_t es = (intptr_t)sizeof(C_TYPE);                              \
    if (sa == es && sb == es && so == es) {                                    \
      if (a != out) {                                                          \
        /* PATH 1a: all contiguous, distinct buffers — restrict is valid */    \
        const C_TYPE *restrict pa = (const C_TYPE *)a;                         \
        const C_TYPE *restrict pb = (const C_TYPE *)b;                         \
        C_TYPE *restrict po = (C_TYPE *)out;                                   \
        NUMC_OMP_FOR(                                                          \
            n, sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {                \
              C_TYPE in1 = pa[i];                                              \
              C_TYPE in2 = pb[i];                                              \
              po[i] = (EXPR);                                                  \
            });                                                                \
      } else {                                                                 \
        /* PATH 1b: all contiguous, inplace (a == out) — no restrict */        \
        C_TYPE *p = (C_TYPE *)out;                                             \
        const C_TYPE *restrict pb = (const C_TYPE *)b;                         \
        NUMC_OMP_FOR(                                                          \
            n, sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {                \
              C_TYPE in1 = p[i];                                               \
              C_TYPE in2 = pb[i];                                              \
              p[i] = (EXPR);                                                   \
            });                                                                \
      }                                                                        \
    } else if (sb == 0 && sa == es && so == es) {                              \
      /* PATH 2: scalar broadcast */                                           \
      const C_TYPE in2 = *(const C_TYPE *)b;                                   \
      if (a == out) {                                                          \
        /* PATH 2a: inplace — single pointer, no aliasing check */             \
        C_TYPE *restrict p = (C_TYPE *)out;                                    \
        NUMC_OMP_FOR(                                                          \
            n, sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {                \
              C_TYPE in1 = p[i];                                               \
              p[i] = (EXPR);                                                   \
            });                                                                \
      } else {                                                                 \
        /* PATH 2b: separate src/dst */                                        \
        const C_TYPE *restrict pa = (const C_TYPE *)a;                         \
        C_TYPE *restrict po = (C_TYPE *)out;                                   \
        NUMC_OMP_FOR(                                                          \
            n, sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {                \
              C_TYPE in1 = pa[i];                                              \
              po[i] = (EXPR);                                                  \
            });                                                                \
      }                                                                        \
    } else {                                                                   \
      /* PATH 3: generic strided */                                            \
      for (size_t i = 0; i < n; i++) {                                         \
        C_TYPE in1 = *(const C_TYPE *)(a + i * sa);                            \
        C_TYPE in2 = *(const C_TYPE *)(b + i * sb);                            \
        *(C_TYPE *)(out + i * so) = (EXPR);                                    \
      }                                                                        \
    }                                                                          \
  }

/* ── Stride-aware unary kernel macro ───────────────────────────────
 *
 * Three runtime paths:
 *   PATH 1 — Contiguous, distinct buffers: sa == so == sizeof(T), a != out
 *   PATH 2 — Contiguous inplace:           a == out
 *   PATH 3 — Generic strided:              arbitrary sa, so
 */

#define DEFINE_UNARY_KERNEL(OP_NAME, TYPE_ENUM, C_TYPE, EXPR)                  \
  static void _kern_##OP_NAME##_##TYPE_ENUM(                                   \
      const char *a, char *out, size_t n, intptr_t sa, intptr_t so) {          \
    const intptr_t es = (intptr_t)sizeof(C_TYPE);                              \
    if (sa == es && so == es && a != out) {                                    \
      /* PATH 1: contiguous, distinct buffers — restrict is valid */           \
      const C_TYPE *restrict pa = (const C_TYPE *)a;                           \
      C_TYPE *restrict po = (C_TYPE *)out;                                     \
      NUMC_OMP_FOR(                                                            \
          n, sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {                  \
            C_TYPE in1 = pa[i];                                                \
            po[i] = (EXPR);                                                    \
          });                                                                  \
    } else if (sa == es && so == es) {                                         \
      /* PATH 2: contiguous inplace (a == out) — no restrict to avoid UB */    \
      C_TYPE *p = (C_TYPE *)a;                                                 \
      NUMC_OMP_FOR(                                                            \
          n, sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {                  \
            C_TYPE in1 = p[i];                                                 \
            p[i] = (EXPR);                                                     \
          });                                                                  \
    } else {                                                                   \
      /* PATH 3: generic strided */                                            \
      for (size_t i = 0; i < n; i++) {                                         \
        C_TYPE in1 = *(const C_TYPE *)(a + i * sa);                            \
        *(C_TYPE *)(out + i * so) = (EXPR);                                    \
      }                                                                        \
    }                                                                          \
  }

/* ── Stride-aware clip kernel macro ────────────────────────────────
 *
 * Same three paths as unary, but takes (double min, double max) params.
 */

#define DEFINE_CLIP_KERNEL(TE, CT)                                             \
  static void _kern_clip_##TE(const char *a, char *out, size_t n, intptr_t sa, \
                              intptr_t so, double min, double max) {           \
    const CT lo = (CT)min;                                                     \
    const CT hi = (CT)max;                                                     \
    const intptr_t es = (intptr_t)sizeof(CT);                                  \
    if (sa == es && so == es && a != out) {                                    \
      /* PATH 1: contiguous, distinct buffers — restrict is valid */           \
      const CT *restrict pa = (const CT *)a;                                   \
      CT *restrict po = (CT *)out;                                             \
      NUMC_OMP_FOR(                                                            \
          n, sizeof(CT), for (size_t i = 0; i < n; i++) {                      \
            CT v = pa[i];                                                      \
            po[i] = (v < lo) ? lo : (v > hi) ? hi : v;                         \
          });                                                                  \
    } else if (sa == es && so == es) {                                         \
      /* PATH 2: contiguous inplace (a == out) — no restrict to avoid UB */    \
      CT *p = (CT *)a;                                                         \
      NUMC_OMP_FOR(                                                            \
          n, sizeof(CT), for (size_t i = 0; i < n; i++) {                      \
            CT v = p[i];                                                       \
            p[i] = (v < lo) ? lo : (v > hi) ? hi : v;                          \
          });                                                                  \
    } else {                                                                   \
      /* PATH 3: generic strided */                                            \
      for (size_t i = 0; i < n; i++) {                                         \
        CT v = *(const CT *)(a + i * sa);                                      \
        *(CT *)(out + i * so) = (v < lo) ? lo : (v > hi) ? hi : v;             \
      }                                                                        \
    }                                                                          \
  }

/* ── Dispatch table entry helper ───────────────────────────────────── */

#define E(OP, TE) [TE] = _kern_##OP##_##TE

#endif
