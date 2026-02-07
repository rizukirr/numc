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

/**
 * @brief OpenMP parallel for with C-level size guard.
 *
 * OpenMP's if() clause still outlines the loop body into a separate function,
 * adding ~30 instructions of overhead (register saves, stack canary, runtime
 * calls) even when parallelism is skipped at runtime. This macro uses a
 * C-level if/else so the small-array path compiles to a direct inline loop
 * with zero OMP overhead.
 *
 * Usage:  NUMC_OMP_FOR(n, for (...) { body })
 */
#define NUMC_OMP_FOR(n, loop)                                                  \
  if ((n) > NUMC_OMP_THRESHOLD) {                                             \
    NUMC_PRAGMA(omp parallel for schedule(static))                             \
    loop                                                                       \
  } else {                                                                     \
    loop                                                                       \
  }

#endif /* NUMC_OMP_H */
