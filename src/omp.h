/**
 * @file omp.h
 * @brief Internal OpenMP configuration and convenience macros.
 */

#ifndef NUMC_OMP_H
#define NUMC_OMP_H

#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief Minimum element count before OpenMP parallelization kicks in.
 *
 * Below this threshold, thread creation overhead exceeds the parallelism
 * benefit. 100K elements is a conservative default (~0.4 MB for int32,
 * well into L2/L3 territory where parallelism helps).
 */
#define NUMC_OMP_THRESHOLD 100000

/** @brief Helper to stringify after macro expansion (two-level indirection). */
#define NUMC_STR_(x) #x
#define NUMC_STR(x) NUMC_STR_(x)
#define NUMC_PRAGMA(x) _Pragma(NUMC_STR(x))

// clang-format off
/** @brief OpenMP convenience macros — single-string _Pragma wrappers. */
#define NUMC_OMP_FOR \
  NUMC_PRAGMA(omp parallel for schedule(static) if(n > NUMC_OMP_THRESHOLD))
#define NUMC_OMP_REDUCE_SUM \
  NUMC_PRAGMA(omp parallel for simd reduction(+:acc) schedule(static) if(n > NUMC_OMP_THRESHOLD))
#define NUMC_OMP_REDUCE_MIN \
  NUMC_PRAGMA(omp parallel for simd reduction(min:m) schedule(static) if(n > NUMC_OMP_THRESHOLD))
#define NUMC_OMP_REDUCE_MAX \
  NUMC_PRAGMA(omp parallel for simd reduction(max:m) schedule(static) if(n > NUMC_OMP_THRESHOLD))
// clang-format on

#endif /* NUMC_OMP_H */
