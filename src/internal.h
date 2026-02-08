/**
 * @file internal.h
 * @brief Internal utilities: aligned allocation, OpenMP helpers, constants.
 *
 * This header is NOT part of the public API. It merges the former alloc.h
 * and omp.h into a single private header used only by implementation files.
 */

#ifndef NUMC_INTERNAL_H
#define NUMC_INTERNAL_H

#include <stddef.h>

// =============================================================================
//  Internal Constants
// =============================================================================

/**
 * @brief Threshold for switching to SIMD/AVX2 code paths in binary ops.
 */
#define NUMC_MAX_ELEMENT_LOOP 1000000

// =============================================================================
//  Aligned Memory Allocation (cross-platform)
// =============================================================================

/**
 * @brief Default alignment for array data buffers (32 bytes for AVX).
 */
#define NUMC_ALIGN 32

/**
 * @brief Allocate aligned memory without initialization.
 *
 * Cross-platform wrapper for aligned allocation. Uses aligned_alloc()
 * on POSIX systems and _aligned_malloc() on Windows MSVC.
 * Memory is NOT zeroed - use numc_calloc() for zero-initialized memory.
 *
 * @param alignment Alignment in bytes (must be power of 2).
 * @param size      Size in bytes (must be multiple of alignment).
 * @return Pointer to aligned memory, or NULL on failure.
 */
void *numc_malloc(size_t alignment, size_t size);

/**
 * @brief Allocate aligned memory with zero initialization.
 *
 * Cross-platform wrapper for aligned allocation. Uses aligned_alloc()
 * on POSIX systems and _aligned_malloc() on Windows MSVC.
 *
 * @param alignment Alignment in bytes (must be power of 2).
 * @param size      Size in bytes (must be multiple of alignment).
 * @return Pointer to aligned memory, or NULL on failure.
 */
void *numc_calloc(size_t alignment, size_t size);

/**
 * @brief Free memory allocated by numc_malloc() or numc_calloc().
 *
 * Cross-platform wrapper for freeing aligned memory.
 *
 * @param ptr Pointer to memory allocated by numc_malloc() or numc_calloc().
 */
void numc_free(void *ptr);

/**
 * @brief Reallocate aligned memory while preserving alignment.
 *
 * Allocates new aligned memory, copies old data, and frees the old pointer.
 * There is no true aligned realloc on most platforms, so this allocates fresh.
 *
 * @param ptr       Pointer to existing aligned memory (or NULL).
 * @param alignment Alignment in bytes (must be power of 2).
 * @param old_size  Size of old allocation in bytes.
 * @param new_size  Size of new allocation in bytes.
 * @return Pointer to new aligned memory, or NULL on failure.
 */
void *numc_realloc(void *ptr, size_t alignment, size_t old_size,
                   size_t new_size);

// =============================================================================
//  OpenMP Helpers
// =============================================================================

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

#endif /* NUMC_INTERNAL_H */
