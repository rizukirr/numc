#ifndef NUMC_ELEMWISE_UNARY_DISPATCH_H
#define NUMC_ELEMWISE_UNARY_DISPATCH_H

#include "arch_dispatch.h"
#include "dispatch.h"

#if NUMC_HAVE_AVX512 || NUMC_HAVE_AVX2 || NUMC_HAVE_SVE || NUMC_HAVE_NEON || \
    NUMC_HAVE_RVV

typedef void (*FastUnKern)(const void *, void *, size_t);

#if NUMC_HAVE_AVX512
#define FUN(OP, SFX) (FastUnKern) _fast_##OP##_##SFX##_avx512
#elif NUMC_HAVE_AVX2
#define FUN(OP, SFX) (FastUnKern) _fast_##OP##_##SFX##_avx2
#elif NUMC_HAVE_SVE
#define FUN(OP, SFX) (FastUnKern) _fast_##OP##_##SFX##_sve
#elif NUMC_HAVE_NEON
#define FUN(OP, SFX) (FastUnKern) _fast_##OP##_##SFX##_neon
#elif NUMC_HAVE_RVV
#define FUN(OP, SFX) (FastUnKern) _fast_##OP##_##SFX##_rvv
#endif

#define DEFINE_UNARY_SIMD(NAME, FAST_TABLE, FALLBACK_TABLE)                    \
  int numc_##NAME(NumcArray *a, NumcArray *out) {                              \
    int err = _check_unary(a, out);                                            \
    if (err)                                                                   \
      return err;                                                              \
    if (a->is_contiguous && out->is_contiguous) {                              \
      FastUnKern kern = FAST_TABLE[a->dtype];                                  \
      if (kern) {                                                              \
        size_t n = a->size, es = a->elem_size, total = n * es;                 \
        int nt = (int)(total / NUMC_OMP_BYTES_PER_THREAD);                     \
        NUMC_OMP_CAP_THREADS(nt);                                              \
        if (nt >= 2) {                                                         \
          size_t chunk = (n + (size_t)nt - 1) / (size_t)nt;                    \
          NUMC_PRAGMA(                                             \
              omp parallel for schedule(static) num_threads(nt))               \
          for (int t = 0; t < nt; t++) {                                       \
            size_t s = (size_t)t * chunk;                                      \
            size_t e = s + chunk;                                              \
            if (e > n)                                                         \
              e = n;                                                           \
            if (s < n)                                                         \
              kern((const char *)a->data + s * es, (char *)out->data + s * es, \
                   e - s);                                                     \
          }                                                                    \
        } else {                                                               \
          kern(a->data, out->data, n);                                         \
        }                                                                      \
        return 0;                                                              \
      }                                                                        \
    }                                                                          \
    return _unary_op(a, out, FALLBACK_TABLE);                                  \
  }                                                                            \
  int numc_##NAME##_inplace(NumcArray *a) {                                    \
    if (!a)                                                                    \
      return NUMC_ERR_NULL;                                                    \
    if (a->is_contiguous) {                                                    \
      FastUnKern kern = FAST_TABLE[a->dtype];                                  \
      if (kern) {                                                              \
        size_t n = a->size, es = a->elem_size, total = n * es;                 \
        int nt = (int)(total / NUMC_OMP_BYTES_PER_THREAD);                     \
        NUMC_OMP_CAP_THREADS(nt);                                              \
        if (nt >= 2) {                                                         \
          size_t chunk = (n + (size_t)nt - 1) / (size_t)nt;                    \
          NUMC_PRAGMA(                                             \
              omp parallel for schedule(static) num_threads(nt))               \
          for (int t = 0; t < nt; t++) {                                       \
            size_t s = (size_t)t * chunk;                                      \
            size_t e = s + chunk;                                              \
            if (e > n)                                                         \
              e = n;                                                           \
            if (s < n)                                                         \
              kern((const char *)a->data + s * es, (char *)a->data + s * es,   \
                   e - s);                                                     \
          }                                                                    \
        } else {                                                               \
          kern(a->data, a->data, n);                                           \
        }                                                                      \
        return 0;                                                              \
      }                                                                        \
    }                                                                          \
    return _unary_op_inplace(a, FALLBACK_TABLE);                               \
  }

#endif

#endif
