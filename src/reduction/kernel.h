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

#endif
