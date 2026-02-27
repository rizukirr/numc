#ifndef NUMC_MATH_MATMUL_KERNEL_H
#define NUMC_MATH_MATMUL_KERNEL_H

#include "internal.h"
#include <numc/dtype.h>

typedef void (*MatmulKernel)(const char *a, const char *b, char *out, size_t M,
                             size_t K, size_t N);

#define DEFINE_MATMUL_KERNEL(TE, CT)                                           \
  static void _matmul_naive_##TE(const char *a, const char *b, char *out,      \
                                 size_t M, size_t K, size_t N) {               \
    const CT *restrict av = (const CT *)a;                                     \
    const CT *restrict bv = (const CT *)b;                                     \
    CT *restrict ov = (CT *)out;                                               \
    NUMC_OMP_FOR(                                                              \
        M * N, sizeof(CT), for (size_t i = 0; i < M; i++) {                    \
          for (size_t k = 0; k < K; k++) {                                     \
            CT aik = av[i * K + k];                                            \
            for (size_t j = 0; j < N; j++) {                                   \
              ov[i * N + j] += aik * bv[k * N + j];                            \
            }                                                                  \
          }                                                                    \
        });                                                                    \
  }

#define DEFINE_MATMUL_KERNEL_PROMOTED(TE, CT, ACC_CT)                          \
  static void _matmul_naive_##TE(const char *a, const char *b, char *out,      \
                                 size_t M, size_t K, size_t N) {               \
    const CT *restrict av = (const CT *)a;                                     \
    const CT *restrict bv = (const CT *)b;                                     \
    CT *restrict ov = (CT *)out;                                               \
    NUMC_OMP_FOR(                                                              \
        M * N, sizeof(CT), for (size_t i = 0; i < M; i++) {                    \
          for (size_t k = 0; k < K; k++) {                                     \
            CT aik = av[i * K + k];                                            \
            for (size_t j = 0; j < N; j++) {                                   \
              ACC_CT acc = (ACC_CT)ov[i * N + j];                              \
              acc += (ACC_CT)aik * (ACC_CT)bv[k * N + j];                      \
              ov[i * N + j] = (CT)acc;                                         \
            }                                                                  \
          }                                                                    \
        });                                                                    \
  }

#define STAMP_MATMUL_KERNEL(TE, CT) DEFINE_MATMUL_KERNEL(TE, CT)
GENERATE_32BIT_NUMC_TYPES(STAMP_MATMUL_KERNEL)
GENERATE_64BIT_NUMC_TYPES(STAMP_MATMUL_KERNEL)
#undef STAMP_MATMUL_KERNEL

DEFINE_MATMUL_KERNEL_PROMOTED(NUMC_DTYPE_INT8, NUMC_INT8, NUMC_INT32)
DEFINE_MATMUL_KERNEL_PROMOTED(NUMC_DTYPE_UINT8, NUMC_UINT8, NUMC_UINT32)
DEFINE_MATMUL_KERNEL_PROMOTED(NUMC_DTYPE_INT16, NUMC_INT16, NUMC_INT64)
DEFINE_MATMUL_KERNEL_PROMOTED(NUMC_DTYPE_UINT16, NUMC_UINT16, NUMC_UINT64)

#endif
