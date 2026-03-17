/**
 * @file kernel.h
 * @brief Element-wise kernel typedefs and X-macro generators.
 *
 * Provides DEFINE_BINARY_KERNEL, DEFINE_UNARY_KERNEL, etc. that expand
 * via GENERATE_NUMC_TYPES to produce per-dtype kernel functions.
 */
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

typedef void (*NumcTernaryKernel)(const char *cond, const char *a,
                                  const char *b, char *out, size_t n,
                                  intptr_t sc, intptr_t sa, intptr_t sb,
                                  intptr_t so);

typedef void (*NumcQuaternaryKernel)(const char *a, const char *b,
                                     const char *c, char *out, size_t n,
                                     intptr_t sa, intptr_t sb, intptr_t sc,
                                     intptr_t so);

/* ── Stride-aware binary kernel macro ────────────────────────────── */

#define NUMC_TILE_SIZE 256

#define DEFINE_BINARY_KERNEL(OP_NAME, TYPE_ENUM, C_TYPE, EXPR)                \
  static void _kern_##OP_NAME##_##TYPE_ENUM(const char *a, const char *b,     \
                                            char *out, size_t n, intptr_t sa, \
                                            intptr_t sb, intptr_t so) {       \
    const intptr_t es = (intptr_t)sizeof(C_TYPE);                             \
    if (sa == es && sb == es && so == es) {                                   \
      if (a != out) {                                                         \
        const C_TYPE *restrict pa =                                           \
            (const C_TYPE *)__builtin_assume_aligned(a, 32);                  \
        const C_TYPE *restrict pb =                                           \
            (const C_TYPE *)__builtin_assume_aligned(b, 32);                  \
        C_TYPE *restrict po = (C_TYPE *)__builtin_assume_aligned(out, 32);    \
        NUMC_OMP_FOR(                                                         \
            n, sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {               \
              C_TYPE in1 = pa[i];                                             \
              C_TYPE in2 = pb[i];                                             \
              po[i] = (EXPR);                                                 \
            });                                                               \
      } else {                                                                \
        C_TYPE *restrict p = (C_TYPE *)__builtin_assume_aligned(out, 32);     \
        const C_TYPE *restrict pb =                                           \
            (const C_TYPE *)__builtin_assume_aligned(b, 32);                  \
        NUMC_OMP_FOR(                                                         \
            n, sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {               \
              C_TYPE in1 = p[i];                                              \
              C_TYPE in2 = pb[i];                                             \
              p[i] = (EXPR);                                                  \
            });                                                               \
      }                                                                       \
    } else if (sb == 0 && sa == es && so == es) {                             \
      const C_TYPE in2 = *(const C_TYPE *)b;                                  \
      if (a == out) {                                                         \
        C_TYPE *restrict p = (C_TYPE *)__builtin_assume_aligned(out, 32);     \
        NUMC_OMP_FOR(                                                         \
            n, sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {               \
              C_TYPE in1 = p[i];                                              \
              p[i] = (EXPR);                                                  \
            });                                                               \
      } else {                                                                \
        const C_TYPE *restrict pa =                                           \
            (const C_TYPE *)__builtin_assume_aligned(a, 32);                  \
        C_TYPE *restrict po = (C_TYPE *)__builtin_assume_aligned(out, 32);    \
        NUMC_OMP_FOR(                                                         \
            n, sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {               \
              C_TYPE in1 = pa[i];                                             \
              po[i] = (EXPR);                                                 \
            });                                                               \
      }                                                                       \
    } else if (sa == 0 && sb == es && so == es) {                             \
      const C_TYPE in1 = *(const C_TYPE *)a;                                  \
      const C_TYPE *restrict pb =                                             \
          (const C_TYPE *)__builtin_assume_aligned(b, 32);                    \
      C_TYPE *restrict po = (C_TYPE *)__builtin_assume_aligned(out, 32);      \
      NUMC_OMP_FOR(                                                           \
          n, sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {                 \
            C_TYPE in2 = pb[i];                                               \
            po[i] = (EXPR);                                                   \
          });                                                                 \
    } else if (sa == es && so == es) {                                        \
      const C_TYPE *restrict pa =                                             \
          (const C_TYPE *)__builtin_assume_aligned(a, 32);                    \
      C_TYPE *restrict po = (C_TYPE *)__builtin_assume_aligned(out, 32);      \
      NUMC_OMP_FOR(                                                           \
          n, sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {                 \
            __builtin_prefetch(b + (i + 16) * sb, 0, 3);                      \
            C_TYPE in1 = pa[i];                                               \
            C_TYPE in2 = *(const C_TYPE *)(b + i * sb);                       \
            po[i] = (EXPR);                                                   \
          });                                                                 \
    } else if (sb == es && so == es) {                                        \
      const C_TYPE *restrict pb =                                             \
          (const C_TYPE *)__builtin_assume_aligned(b, 32);                    \
      C_TYPE *restrict po = (C_TYPE *)__builtin_assume_aligned(out, 32);      \
      NUMC_OMP_FOR(                                                           \
          n, sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {                 \
            __builtin_prefetch(a + (i + 16) * sa, 0, 3);                      \
            C_TYPE in1 = *(const C_TYPE *)(a + i * sa);                       \
            C_TYPE in2 = pb[i];                                               \
            po[i] = (EXPR);                                                   \
          });                                                                 \
    } else if (sa == es && sb == es) {                                        \
      const C_TYPE *restrict pa =                                             \
          (const C_TYPE *)__builtin_assume_aligned(a, 32);                    \
      const C_TYPE *restrict pb =                                             \
          (const C_TYPE *)__builtin_assume_aligned(b, 32);                    \
      NUMC_OMP_FOR(                                                           \
          n, sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {                 \
            __builtin_prefetch(out + (i + 16) * so, 1, 3);                    \
            C_TYPE in1 = pa[i];                                               \
            C_TYPE in2 = pb[i];                                               \
            *(C_TYPE *)(out + i * so) = (EXPR);                               \
          });                                                                 \
    } else {                                                                  \
      for (size_t base = 0; base < n; base += NUMC_TILE_SIZE) {               \
        size_t chunk = n - base < NUMC_TILE_SIZE ? n - base : NUMC_TILE_SIZE; \
        C_TYPE abuf[NUMC_TILE_SIZE], bbuf[NUMC_TILE_SIZE],                    \
            obuf[NUMC_TILE_SIZE];                                             \
        for (size_t i = 0; i < chunk; i++) {                                  \
          __builtin_prefetch(a + (base + i + 16) * sa, 0, 3);                 \
          abuf[i] = *(const C_TYPE *)(a + (base + i) * sa);                   \
        }                                                                     \
        for (size_t i = 0; i < chunk; i++) {                                  \
          __builtin_prefetch(b + (base + i + 16) * sb, 0, 3);                 \
          bbuf[i] = *(const C_TYPE *)(b + (base + i) * sb);                   \
        }                                                                     \
        for (size_t i = 0; i < chunk; i++) {                                  \
          C_TYPE in1 = abuf[i];                                               \
          C_TYPE in2 = bbuf[i];                                               \
          obuf[i] = (EXPR);                                                   \
        }                                                                     \
        for (size_t i = 0; i < chunk; i++) {                                  \
          __builtin_prefetch(out + (base + i + 16) * so, 1, 3);               \
          *(C_TYPE *)(out + (base + i) * so) = obuf[i];                       \
        }                                                                     \
      }                                                                       \
    }                                                                         \
  }

#define DEFINE_BINARY_KERNEL_NOSIMD(OP_NAME, TYPE_ENUM, C_TYPE, EXPR)         \
  static void _kern_##OP_NAME##_##TYPE_ENUM(const char *a, const char *b,     \
                                            char *out, size_t n, intptr_t sa, \
                                            intptr_t sb, intptr_t so) {       \
    const intptr_t es = (intptr_t)sizeof(C_TYPE);                             \
    if (sa == es && sb == es && so == es) {                                   \
      if (a != out) {                                                         \
        const C_TYPE *pa = (const C_TYPE *)a;                                 \
        const C_TYPE *pb = (const C_TYPE *)b;                                 \
        C_TYPE *po = (C_TYPE *)out;                                           \
        NUMC_OMP_FOR_NOSIMD(                                                  \
            n, sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {               \
              C_TYPE in1 = pa[i];                                             \
              C_TYPE in2 = pb[i];                                             \
              po[i] = (EXPR);                                                 \
            });                                                               \
      } else {                                                                \
        C_TYPE *p = (C_TYPE *)out;                                            \
        const C_TYPE *pb = (const C_TYPE *)b;                                 \
        NUMC_OMP_FOR_NOSIMD(                                                  \
            n, sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {               \
              C_TYPE in1 = p[i];                                              \
              C_TYPE in2 = pb[i];                                             \
              p[i] = (EXPR);                                                  \
            });                                                               \
      }                                                                       \
    } else if (sb == 0 && sa == es && so == es) {                             \
      const C_TYPE in2 = *(const C_TYPE *)b;                                  \
      if (a == out) {                                                         \
        C_TYPE *p = (C_TYPE *)out;                                            \
        NUMC_OMP_FOR_NOSIMD(                                                  \
            n, sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {               \
              C_TYPE in1 = p[i];                                              \
              p[i] = (EXPR);                                                  \
            });                                                               \
      } else {                                                                \
        const C_TYPE *pa = (const C_TYPE *)a;                                 \
        C_TYPE *po = (C_TYPE *)out;                                           \
        NUMC_OMP_FOR_NOSIMD(                                                  \
            n, sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {               \
              C_TYPE in1 = pa[i];                                             \
              po[i] = (EXPR);                                                 \
            });                                                               \
      }                                                                       \
    } else {                                                                  \
      for (size_t i = 0; i < n; i++) {                                        \
        C_TYPE in1 = *(const C_TYPE *)(a + i * sa);                           \
        C_TYPE in2 = *(const C_TYPE *)(b + i * sb);                           \
        *(C_TYPE *)(out + i * so) = (EXPR);                                   \
      }                                                                       \
    }                                                                         \
  }

/* ── Optimized division common logic ────────────────────────────────── */

static inline bool _is_pow2(uint64_t n) {
  return n > 0 && (n & (n - 1)) == 0;
}
static inline int _log2_u64(uint64_t n) {
  return 63 - __builtin_clzll(n);
}

/* ── Integer division macro (with power-of-two optimization) ───────── */

#define DEFINE_INT_DIV_KERNEL(TYPE_ENUM, C_TYPE, IS_SIGNED)                    \
  static void _kern_div_##TYPE_ENUM(const char *a, const char *b, char *out,   \
                                    size_t n, intptr_t sa, intptr_t sb,        \
                                    intptr_t so) {                             \
    const intptr_t es = (intptr_t)sizeof(C_TYPE);                              \
    if (sa == es && sb == es && so == es) {                                    \
      const C_TYPE *restrict pa =                                              \
          (const C_TYPE *)__builtin_assume_aligned(a, 32);                     \
      const C_TYPE *restrict pb =                                              \
          (const C_TYPE *)__builtin_assume_aligned(b, 32);                     \
      C_TYPE *restrict po = (C_TYPE *)__builtin_assume_aligned(out, 32);       \
      if (sizeof(C_TYPE) <= 2) {                                               \
        NUMC_OMP_FOR(                                                          \
            n, 3 * sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {            \
              C_TYPE divisor = pb[i];                                          \
              po[i] = (divisor == 0)                                           \
                          ? 0                                                  \
                          : (C_TYPE)((float)pa[i] / (float)divisor);           \
            });                                                                \
      } else if (sizeof(C_TYPE) == 4) {                                        \
        NUMC_OMP_FOR(                                                          \
            n, 3 * sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {            \
              C_TYPE divisor = pb[i];                                          \
              po[i] = (divisor == 0)                                           \
                          ? 0                                                  \
                          : (C_TYPE)((double)pa[i] / (double)divisor);         \
            });                                                                \
      } else {                                                                 \
        NUMC_OMP_FOR(                                                          \
            n, 3 * sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {            \
              C_TYPE divisor = pb[i];                                          \
              po[i] = (divisor == 0) ? 0 : pa[i] / divisor;                    \
            });                                                                \
      }                                                                        \
    } else if (sb == 0 && sa == es && so == es) {                              \
      const C_TYPE in2 = *(const C_TYPE *)b;                                   \
      if (in2 != 0) {                                                          \
        uint64_t abs_d = (uint64_t)(in2 > 0 ? in2 : -in2);                     \
        if (_is_pow2(abs_d)) {                                                 \
          int shift = _log2_u64(abs_d);                                        \
          C_TYPE *restrict po = (C_TYPE *)__builtin_assume_aligned(out, 32);   \
          const C_TYPE *restrict pa =                                          \
              (const C_TYPE *)__builtin_assume_aligned(a, 32);                 \
          if (IS_SIGNED) {                                                     \
            NUMC_OMP_FOR(                                                      \
                n, sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {            \
                  C_TYPE x = pa[i];                                            \
                  C_TYPE bias = (x < 0) ? (C_TYPE)(abs_d - 1) : 0;             \
                  po[i] = (C_TYPE)((x + bias) >> shift);                       \
                });                                                            \
          } else {                                                             \
            NUMC_OMP_FOR(                                                      \
                n, sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {            \
                  po[i] = (C_TYPE)((uint64_t)pa[i] >> shift);                  \
                });                                                            \
          }                                                                    \
          return;                                                              \
        } else if (!(IS_SIGNED) && sizeof(C_TYPE) <= 4) {                      \
          /* Fast path: Magic number for non-power-of-two unsigned division */ \
          uint64_t shift_amt = sizeof(C_TYPE) * 8;                             \
          uint64_t inv = (1ULL << shift_amt) / in2 + 1;                        \
          if (a == out) {                                                      \
            C_TYPE *restrict p = (C_TYPE *)__builtin_assume_aligned(out, 32);  \
            NUMC_OMP_FOR(                                                      \
                n, sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {            \
                  p[i] = (C_TYPE)(((uint64_t)p[i] * inv) >> shift_amt);        \
                });                                                            \
          } else {                                                             \
            const C_TYPE *restrict pa =                                        \
                (const C_TYPE *)__builtin_assume_aligned(a, 32);               \
            C_TYPE *restrict po = (C_TYPE *)__builtin_assume_aligned(out, 32); \
            NUMC_OMP_FOR(                                                      \
                n, sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {            \
                  po[i] = (C_TYPE)(((uint64_t)pa[i] * inv) >> shift_amt);      \
                });                                                            \
          }                                                                    \
          return;                                                              \
        }                                                                      \
        float inv = 1.0f / (float)in2;                                         \
        C_TYPE *restrict po = (C_TYPE *)__builtin_assume_aligned(out, 32);     \
        const C_TYPE *restrict pa =                                            \
            (const C_TYPE *)__builtin_assume_aligned(a, 32);                   \
        NUMC_OMP_FOR(                                                          \
            n, sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {                \
              po[i] = (C_TYPE)((float)pa[i] * inv);                            \
            });                                                                \
      } else {                                                                 \
        C_TYPE *restrict po = (C_TYPE *)__builtin_assume_aligned(out, 32);     \
        for (size_t i = 0; i < n; i++)                                         \
          po[i] = 0;                                                           \
      }                                                                        \
    } else {                                                                   \
      for (size_t i = 0; i < n; i++) {                                         \
        __builtin_prefetch(a + (i + 16) * sa, 0, 3);                           \
        C_TYPE in1 = *(const C_TYPE *)(a + i * sa);                            \
        C_TYPE in2 = *(const C_TYPE *)(b + i * sb);                            \
        *(C_TYPE *)(out + i * so) = (in2 == 0) ? 0 : in1 / in2;                \
      }                                                                        \
    }                                                                          \
  }

/* ── Float division macro (with reciprocal optimization) ───────────── */

#define DEFINE_FLOAT_DIV_KERNEL(TYPE_ENUM, C_TYPE)                           \
  static void _kern_div_##TYPE_ENUM(const char *a, const char *b, char *out, \
                                    size_t n, intptr_t sa, intptr_t sb,      \
                                    intptr_t so) {                           \
    const intptr_t es = (intptr_t)sizeof(C_TYPE);                            \
    if (sa == es && sb == es && so == es) {                                  \
      const C_TYPE *restrict pa =                                            \
          (const C_TYPE *)__builtin_assume_aligned(a, 32);                   \
      const C_TYPE *restrict pb =                                            \
          (const C_TYPE *)__builtin_assume_aligned(b, 32);                   \
      C_TYPE *restrict po = (C_TYPE *)__builtin_assume_aligned(out, 32);     \
      NUMC_OMP_FOR(                                                          \
          n, sizeof(C_TYPE),                                                 \
          for (size_t i = 0; i < n; i++) { po[i] = pa[i] / pb[i]; });        \
    } else if (sb == 0 && sa == es && so == es) {                            \
      const C_TYPE in2 = *(const C_TYPE *)b;                                 \
      C_TYPE inv = (C_TYPE)1.0 / in2;                                        \
      const C_TYPE *restrict pa =                                            \
          (const C_TYPE *)__builtin_assume_aligned(a, 32);                   \
      C_TYPE *restrict po = (C_TYPE *)__builtin_assume_aligned(out, 32);     \
      NUMC_OMP_FOR(                                                          \
          n, sizeof(C_TYPE),                                                 \
          for (size_t i = 0; i < n; i++) { po[i] = pa[i] * inv; });          \
    } else {                                                                 \
      for (size_t i = 0; i < n; i++) {                                       \
        C_TYPE in1 = *(const C_TYPE *)(a + i * sa);                          \
        C_TYPE in2 = *(const C_TYPE *)(b + i * sb);                          \
        *(C_TYPE *)(out + i * so) = in1 / in2;                               \
      }                                                                      \
    }                                                                        \
  }

/* ── Stride-aware ternary kernel macro ───────────────────────────── */

#define DEFINE_TERNARY_KERNEL(OP_NAME, TYPE_ENUM, C_TYPE, EXPR)               \
  static void _kern_##OP_NAME##_##TYPE_ENUM(                                  \
      const char *cond, const char *a, const char *b, char *out, size_t n,    \
      intptr_t sc, intptr_t sa, intptr_t sb, intptr_t so) {                   \
    const intptr_t es = (intptr_t)sizeof(C_TYPE);                             \
    if (sc == es && sa == es && sb == es && so == es) {                       \
      const C_TYPE *restrict pc =                                             \
          (const C_TYPE *)__builtin_assume_aligned(cond, 32);                 \
      const C_TYPE *restrict pa =                                             \
          (const C_TYPE *)__builtin_assume_aligned(a, 32);                    \
      const C_TYPE *restrict pb =                                             \
          (const C_TYPE *)__builtin_assume_aligned(b, 32);                    \
      C_TYPE *restrict po = (C_TYPE *)__builtin_assume_aligned(out, 32);      \
      NUMC_OMP_FOR(                                                           \
          n, sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {                 \
            C_TYPE in_cond = pc[i];                                           \
            C_TYPE in_a = pa[i];                                              \
            C_TYPE in_b = pb[i];                                              \
            po[i] = (EXPR);                                                   \
          });                                                                 \
    } else if (sc == 0 && sa == es && sb == es && so == es) {                 \
      const C_TYPE in_cond = *(const C_TYPE *)cond;                           \
      const C_TYPE *restrict pa =                                             \
          (const C_TYPE *)__builtin_assume_aligned(a, 32);                    \
      const C_TYPE *restrict pb =                                             \
          (const C_TYPE *)__builtin_assume_aligned(b, 32);                    \
      C_TYPE *restrict po = (C_TYPE *)__builtin_assume_aligned(out, 32);      \
      NUMC_OMP_FOR(                                                           \
          n, sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {                 \
            C_TYPE in_a = pa[i];                                              \
            C_TYPE in_b = pb[i];                                              \
            po[i] = (EXPR);                                                   \
          });                                                                 \
    } else {                                                                  \
      for (size_t base = 0; base < n; base += NUMC_TILE_SIZE) {               \
        size_t chunk = n - base < NUMC_TILE_SIZE ? n - base : NUMC_TILE_SIZE; \
        C_TYPE cbuf[NUMC_TILE_SIZE], abuf[NUMC_TILE_SIZE],                    \
            bbuf[NUMC_TILE_SIZE], obuf[NUMC_TILE_SIZE];                       \
        for (size_t i = 0; i < chunk; i++) {                                  \
          __builtin_prefetch(cond + (base + i + 16) * sc, 0, 3);              \
          cbuf[i] = *(const C_TYPE *)(cond + (base + i) * sc);                \
        }                                                                     \
        for (size_t i = 0; i < chunk; i++) {                                  \
          __builtin_prefetch(a + (base + i + 16) * sa, 0, 3);                 \
          abuf[i] = *(const C_TYPE *)(a + (base + i) * sa);                   \
        }                                                                     \
        for (size_t i = 0; i < chunk; i++) {                                  \
          __builtin_prefetch(b + (base + i + 16) * sb, 0, 3);                 \
          bbuf[i] = *(const C_TYPE *)(b + (base + i) * sb);                   \
        }                                                                     \
        for (size_t i = 0; i < chunk; i++) {                                  \
          C_TYPE in_cond = cbuf[i];                                           \
          C_TYPE in_a = abuf[i];                                              \
          C_TYPE in_b = bbuf[i];                                              \
          obuf[i] = (EXPR);                                                   \
        }                                                                     \
        for (size_t i = 0; i < chunk; i++) {                                  \
          __builtin_prefetch(out + (base + i + 16) * so, 1, 3);               \
          *(C_TYPE *)(out + (base + i) * so) = obuf[i];                       \
        }                                                                     \
      }                                                                       \
    }                                                                         \
  }

/* ── Stride-aware unary kernel macro ─────────────────────────────── */

#define DEFINE_UNARY_KERNEL(OP_NAME, TYPE_ENUM, C_TYPE, EXPR)                 \
  static void _kern_##OP_NAME##_##TYPE_ENUM(                                  \
      const char *a, char *out, size_t n, intptr_t sa, intptr_t so) {         \
    const intptr_t es = (intptr_t)sizeof(C_TYPE);                             \
    if (sa == es && so == es && a != out) {                                   \
      const C_TYPE *restrict pa =                                             \
          (const C_TYPE *)__builtin_assume_aligned(a, 32);                    \
      C_TYPE *restrict po = (C_TYPE *)__builtin_assume_aligned(out, 32);      \
      NUMC_OMP_FOR(                                                           \
          n, sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {                 \
            C_TYPE in1 = pa[i];                                               \
            po[i] = (EXPR);                                                   \
          });                                                                 \
    } else if (sa == es && so == es) {                                        \
      C_TYPE *restrict p = (C_TYPE *)__builtin_assume_aligned(a, 32);         \
      NUMC_OMP_FOR(                                                           \
          n, sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {                 \
            C_TYPE in1 = p[i];                                                \
            p[i] = (EXPR);                                                    \
          });                                                                 \
    } else if (sa == es) {                                                    \
      const C_TYPE *restrict pa =                                             \
          (const C_TYPE *)__builtin_assume_aligned(a, 32);                    \
      NUMC_OMP_FOR(                                                           \
          n, sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {                 \
            C_TYPE in1 = pa[i];                                               \
            *(C_TYPE *)(out + i * so) = (EXPR);                               \
          });                                                                 \
    } else if (so == es) {                                                    \
      C_TYPE *restrict po = (C_TYPE *)__builtin_assume_aligned(out, 32);      \
      NUMC_OMP_FOR(                                                           \
          n, sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {                 \
            __builtin_prefetch(a + (i + 16) * sa, 0, 3);                      \
            C_TYPE in1 = *(const C_TYPE *)(a + i * sa);                       \
            po[i] = (EXPR);                                                   \
          });                                                                 \
    } else {                                                                  \
      for (size_t base = 0; base < n; base += NUMC_TILE_SIZE) {               \
        size_t chunk = n - base < NUMC_TILE_SIZE ? n - base : NUMC_TILE_SIZE; \
        C_TYPE abuf[NUMC_TILE_SIZE], obuf[NUMC_TILE_SIZE];                    \
        for (size_t i = 0; i < chunk; i++) {                                  \
          __builtin_prefetch(a + (base + i + 16) * sa, 0, 3);                 \
          abuf[i] = *(const C_TYPE *)(a + (base + i) * sa);                   \
        }                                                                     \
        for (size_t i = 0; i < chunk; i++) {                                  \
          C_TYPE in1 = abuf[i];                                               \
          obuf[i] = (EXPR);                                                   \
        }                                                                     \
        for (size_t i = 0; i < chunk; i++) {                                  \
          __builtin_prefetch(out + (base + i + 16) * so, 1, 3);               \
          *(C_TYPE *)(out + (base + i) * so) = obuf[i];                       \
        }                                                                     \
      }                                                                       \
    }                                                                         \
  }

#define DEFINE_UNARY_KERNEL_NOSIMD(OP_NAME, TYPE_ENUM, C_TYPE, EXPR)  \
  static void _kern_##OP_NAME##_##TYPE_ENUM(                          \
      const char *a, char *out, size_t n, intptr_t sa, intptr_t so) { \
    const intptr_t es = (intptr_t)sizeof(C_TYPE);                     \
    if (sa == es && so == es && a != out) {                           \
      const C_TYPE *pa = (const C_TYPE *)a;                           \
      C_TYPE *po = (C_TYPE *)out;                                     \
      NUMC_OMP_FOR_NOSIMD(                                            \
          n, sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {         \
            C_TYPE in1 = pa[i];                                       \
            po[i] = (EXPR);                                           \
          });                                                         \
    } else if (sa == es && so == es) {                                \
      C_TYPE *p = (C_TYPE *)out;                                      \
      NUMC_OMP_FOR_NOSIMD(                                            \
          n, sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {         \
            C_TYPE in1 = p[i];                                        \
            p[i] = (EXPR);                                            \
          });                                                         \
    } else {                                                          \
      for (size_t i = 0; i < n; i++) {                                \
        C_TYPE in1 = *(const C_TYPE *)(a + i * sa);                   \
        *(C_TYPE *)(out + i * so) = (EXPR);                           \
      }                                                               \
    }                                                                 \
  }

/* ── Stride-aware clip kernel macro ──────────────────────────────── */

#define DEFINE_CLIP_KERNEL(TE, CT)                                             \
  static void _kern_clip_##TE(const char *a, char *out, size_t n, intptr_t sa, \
                              intptr_t so, double min, double max) {           \
    const CT lo = (CT)min;                                                     \
    const CT hi = (CT)max;                                                     \
    const intptr_t es = (intptr_t)sizeof(CT);                                  \
    if (sa == es && so == es && a != out) {                                    \
      const CT *restrict pa = (const CT *)__builtin_assume_aligned(a, 32);     \
      CT *restrict po = (CT *)__builtin_assume_aligned(out, 32);               \
      NUMC_OMP_FOR(                                                            \
          n, sizeof(CT), for (size_t i = 0; i < n; i++) {                      \
            CT v = pa[i];                                                      \
            po[i] = (v < lo) ? lo : (v > hi) ? hi : v;                         \
          });                                                                  \
    } else if (sa == es && so == es) {                                         \
      CT *restrict p = (CT *)__builtin_assume_aligned(a, 32);                  \
      NUMC_OMP_FOR(                                                            \
          n, sizeof(CT), for (size_t i = 0; i < n; i++) {                      \
            CT v = p[i];                                                       \
            p[i] = (v < lo) ? lo : (v > hi) ? hi : v;                          \
          });                                                                  \
    } else if (sa == es) {                                                     \
      const CT *restrict pa = (const CT *)__builtin_assume_aligned(a, 32);     \
      NUMC_OMP_FOR(                                                            \
          n, sizeof(CT), for (size_t i = 0; i < n; i++) {                      \
            CT v = pa[i];                                                      \
            *(CT *)(out + i * so) = (v < lo) ? lo : (v > hi) ? hi : v;         \
          });                                                                  \
    } else if (so == es) {                                                     \
      CT *restrict po = (CT *)__builtin_assume_aligned(out, 32);               \
      NUMC_OMP_FOR(                                                            \
          n, sizeof(CT), for (size_t i = 0; i < n; i++) {                      \
            __builtin_prefetch(a + (i + 16) * sa, 0, 3);                       \
            CT v = *(const CT *)(a + i * sa);                                  \
            po[i] = (v < lo) ? lo : (v > hi) ? hi : v;                         \
          });                                                                  \
    } else {                                                                   \
      for (size_t i = 0; i < n; i++) {                                         \
        __builtin_prefetch(a + (i + 16) * sa, 0, 3);                           \
        CT v = *(const CT *)(a + i * sa);                                      \
        *(CT *)(out + i * so) = (v < lo) ? lo : (v > hi) ? hi : v;             \
      }                                                                        \
    }                                                                          \
  }

#define DEFINE_QUATERNARY_KERNEL(OP_NAME, TYPE_ENUM, C_TYPE, EXPR)            \
  static void _kern_##OP_NAME##_##TYPE_ENUM(                                  \
      const char *a, const char *b, const char *c, char *out, size_t n,       \
      intptr_t sa, intptr_t sb, intptr_t sc, intptr_t so) {                   \
    const intptr_t es = (intptr_t)sizeof(C_TYPE);                             \
    if (sa == es && sb == es && sc == es && so == es) {                       \
      const C_TYPE *restrict pa =                                             \
          (const C_TYPE *)__builtin_assume_aligned(a, 32);                    \
      const C_TYPE *restrict pb =                                             \
          (const C_TYPE *)__builtin_assume_aligned(b, 32);                    \
      const C_TYPE *restrict pc =                                             \
          (const C_TYPE *)__builtin_assume_aligned(c, 32);                    \
      C_TYPE *restrict po = (C_TYPE *)__builtin_assume_aligned(out, 32);      \
      NUMC_OMP_FOR(                                                           \
          n, sizeof(C_TYPE), for (size_t i = 0; i < n; i++) {                 \
            C_TYPE in_a = pa[i];                                              \
            C_TYPE in_b = pb[i];                                              \
            C_TYPE in_c = pc[i];                                              \
            po[i] = (EXPR);                                                   \
          });                                                                 \
    } else {                                                                  \
      for (size_t base = 0; base < n; base += NUMC_TILE_SIZE) {               \
        size_t chunk = n - base < NUMC_TILE_SIZE ? n - base : NUMC_TILE_SIZE; \
        C_TYPE abuf[NUMC_TILE_SIZE], bbuf[NUMC_TILE_SIZE],                    \
            cbuf[NUMC_TILE_SIZE], obuf[NUMC_TILE_SIZE];                       \
        for (size_t i = 0; i < chunk; i++) {                                  \
          __builtin_prefetch(a + (base + i + 16) * sa, 0, 3);                 \
          abuf[i] = *(const C_TYPE *)(a + (base + i) * sa);                   \
        }                                                                     \
        for (size_t i = 0; i < chunk; i++) {                                  \
          __builtin_prefetch(b + (base + i + 16) * sb, 0, 3);                 \
          bbuf[i] = *(const C_TYPE *)(b + (base + i) * sb);                   \
        }                                                                     \
        for (size_t i = 0; i < chunk; i++) {                                  \
          __builtin_prefetch(c + (base + i + 16) * sc, 0, 3);                 \
          cbuf[i] = *(const C_TYPE *)(c + (base + i) * sc);                   \
        }                                                                     \
        for (size_t i = 0; i < chunk; i++) {                                  \
          C_TYPE in_a = abuf[i];                                              \
          C_TYPE in_b = bbuf[i];                                              \
          C_TYPE in_c = cbuf[i];                                              \
          obuf[i] = (EXPR);                                                   \
        }                                                                     \
        for (size_t i = 0; i < chunk; i++) {                                  \
          __builtin_prefetch(out + (base + i + 16) * so, 1, 3);               \
          *(C_TYPE *)(out + (base + i) * so) = obuf[i];                       \
        }                                                                     \
      }                                                                       \
    }                                                                         \
  }

#define E(OP, TE) [TE] = _kern_##OP##_##TE

#endif
