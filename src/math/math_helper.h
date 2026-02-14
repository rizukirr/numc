#ifndef NUMC_MATH_HELPER_H
#define NUMC_MATH_HELPER_H

#ifdef _OPENMP
#include <omp.h>
#endif

#define NUMC_STR_(x) #x
#define NUMC_STR(x) NUMC_STR_(x)
#define NUMC_PRAGMA(x) _Pragma(NUMC_STR(x))

#define NUMC_OMP_BYTE_THRESHOLD (1 << 20)       // 1 MB total to enable OMP
#define NUMC_OMP_BYTES_PER_THREAD (1 << 20)     // 1 MB minimum work per thread

#define NUMC_OMP_FOR(n, elem_size, loop)                                       \
  do {                                                                         \
    size_t _total_bytes = (n) * (elem_size);                                   \
    if (_total_bytes > NUMC_OMP_BYTE_THRESHOLD) {                              \
      int _nthreads = (int)(_total_bytes / NUMC_OMP_BYTES_PER_THREAD);         \
      if (_nthreads < 1) _nthreads = 1;                                        \
      NUMC_PRAGMA(omp parallel for schedule(static) num_threads(_nthreads))    \
      loop                                                                     \
    } else {                                                                   \
      loop                                                                     \
    }                                                                          \
  } while (0)

#endif
