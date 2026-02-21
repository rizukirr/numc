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
 * Five runtime paths:
 *   PATH 1 — Contiguous:              sa == sb == so == sizeof(T)
 *   PATH 2 — Right scalar broadcast:  sb == 0, a and out contiguous
 *   PATH 2.5 — Left scalar broadcast: sa == 0, b and out contiguous
 *   PATH 3 — Generic strided (tiled): arbitrary sa, sb, so
 */

#define NUMC_TILE_SIZE 256

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
      /* PATH 2: right scalar broadcast (b is scalar) */                       \
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
    } else if (sa == 0 && sb == es && so == es) {                              \
      /* PATH 2.5: left scalar broadcast (a is scalar, b and out contiguous)   \
       * Mirrors PATH 2 but for the opposite operand. Hits when outer          \
       * broadcast places a size-1 dim on the left after axis sorting. */      \
      const C_TYPE in1 = *(const C_TYPE *)a;                                   \
      const C_TYPE *restrict pb = (const C_TYPE *)b;                           \
      C_TYPE *restrict po = (C_TYPE *)out;                                     \
      NUMC_OMP_FOR(                                                            \
          n, sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {                  \
            C_TYPE in2 = pb[i];                                                \
            po[i] = (EXPR);                                                    \
          });                                                                  \
    } else {                                                                   \
      /* PATH 3: generic strided — tiled gather/compute/scatter.               \
       * Gathers elements into contiguous tile buffers so the compute          \
       * loop auto-vectorizes, then scatters results back. */                  \
      for (size_t base = 0; base < n; base += NUMC_TILE_SIZE) {               \
        size_t chunk = n - base < NUMC_TILE_SIZE ? n - base : NUMC_TILE_SIZE;  \
        C_TYPE abuf[NUMC_TILE_SIZE], bbuf[NUMC_TILE_SIZE],                     \
            obuf[NUMC_TILE_SIZE];                                              \
        for (size_t i = 0; i < chunk; i++)                                     \
          abuf[i] = *(const C_TYPE *)(a + (base + i) * sa);                    \
        for (size_t i = 0; i < chunk; i++)                                     \
          bbuf[i] = *(const C_TYPE *)(b + (base + i) * sb);                    \
        for (size_t i = 0; i < chunk; i++) {                                   \
          C_TYPE in1 = abuf[i];                                                \
          C_TYPE in2 = bbuf[i];                                                \
          obuf[i] = (EXPR);                                                    \
        }                                                                      \
        for (size_t i = 0; i < chunk; i++)                                     \
          *(C_TYPE *)(out + (base + i) * so) = obuf[i];                        \
      }                                                                        \
    }                                                                          \
  }

/* ── Stride-aware unary kernel macro ───────────────────────────────
 *
 * Three runtime paths:
 *   PATH 1 — Contiguous, distinct buffers: sa == so == sizeof(T), a != out
 *   PATH 2 — Contiguous inplace:           a == out
 *   PATH 3 — Generic strided (tiled):      arbitrary sa, so
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
      /* PATH 3: generic strided — tiled gather/compute/scatter */             \
      for (size_t base = 0; base < n; base += NUMC_TILE_SIZE) {               \
        size_t chunk = n - base < NUMC_TILE_SIZE ? n - base : NUMC_TILE_SIZE;  \
        C_TYPE abuf[NUMC_TILE_SIZE], obuf[NUMC_TILE_SIZE];                     \
        for (size_t i = 0; i < chunk; i++)                                     \
          abuf[i] = *(const C_TYPE *)(a + (base + i) * sa);                    \
        for (size_t i = 0; i < chunk; i++) {                                   \
          C_TYPE in1 = abuf[i];                                                \
          obuf[i] = (EXPR);                                                    \
        }                                                                      \
        for (size_t i = 0; i < chunk; i++)                                     \
          *(C_TYPE *)(out + (base + i) * so) = obuf[i];                        \
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
