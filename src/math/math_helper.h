#ifndef NUMC_MATH_HELPER_H
#define NUMC_MATH_HELPER_H

#ifdef _OPENMP
#include <omp.h>
#endif

#define NUMC_STR_(x) #x
#define NUMC_STR(x) NUMC_STR_(x)
#define NUMC_PRAGMA(x) _Pragma(NUMC_STR(x))

#define NUMC_OMP_BYTE_THRESHOLD (1 << 20) // 1 MB

#define NUMC_OMP_FOR(n, elem_size, loop)                                       \
  do {                                                                         \
    if ((n) * (elem_size) > NUMC_OMP_BYTE_THRESHOLD) {                         \
      NUMC_PRAGMA(omp parallel for schedule(static))                           \
      loop                                                                     \
    } else {                                                                   \
      loop                                                                     \
    }                                                                          \
  } while (0)

#endif
