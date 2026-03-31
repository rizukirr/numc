/**
 * @file internal.h
 * @brief Internal struct definitions, OMP macros, and constants.
 *
 * Defines the NumcArray and NumcCtx structs, SIMD alignment constant,
 * and OpenMP convenience macros used throughout the library.
 */
#ifndef NUMC_INTERNAL_H
#define NUMC_INTERNAL_H

#include "arena.h"
#include <numc/dtype.h>
#include <numc/error.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* --- Constants --- */

/* Portable alignment specifier — C23 alignas is a keyword */
#define NUMC_ALIGNAS(N) alignas(N)

#define NUMC_MAX_DIMENSIONS  64
#define NUMC_MAX_INLINE_DIMS 8
#define NUMC_MAX_MEMORY      8388608 // 8MB
#define NUMC_SIMD_ALIGN      32

/* --- Opaque struct definitions (private) --- */

struct NumcCtx {
  Arena *arena;
};

struct NumcArray {
  struct NumcCtx *ctx;
  void *data;
  size_t *shape, *strides;
  size_t _shape_buf[NUMC_MAX_INLINE_DIMS], _strides_buf[NUMC_MAX_INLINE_DIMS];
  size_t dim, elem_size, size, capacity;
  bool is_contiguous;
  NumcDType dtype;
};

/* --- Aligned alloc (private) --- */

/**
 * @brief Allocate aligned memory.
 *
 * @param alignment Alignment in bytes (must be a power of two).
 * @param size      Number of bytes to allocate.
 * @return Pointer to the allocated memory, or NULL on failure.
 */
void *numc_malloc(size_t alignment, size_t size);

/**
 * @brief Free memory allocated by numc_malloc.
 *
 * @param ptr Pointer to the memory to be freed.
 */
void numc_free(void *ptr);

/* --- OMP macros (private) --- */

#if defined(_OPENMP) || defined(HAVE_OMP)
#include <omp.h>
#endif

#define NUMC_STR_(x)   #x
#define NUMC_STR(x)    NUMC_STR_(x)
#define NUMC_PRAGMA(x) _Pragma(NUMC_STR(x))

#if defined(__clang__)
#define NUMC_UNROLL(n) _Pragma(NUMC_STR(clang loop unroll_count(n)))
#elif defined(__GNUC__)
#define NUMC_UNROLL(n) _Pragma(NUMC_STR(GCC unroll n))
#else
#define NUMC_UNROLL(n)
#endif

#define NUMC_OMP_BYTE_THRESHOLD \
  (1 << 20) // 1 MB total (used by elemwise/random)
#define NUMC_OMP_BYTES_PER_THREAD (1 << 19) // 512 KB minimum work per thread

/* Reduction-specific thresholds: lower because reductions are
 * memory-bandwidth-bound — threading helps at smaller sizes. */
#define NUMC_OMP_REDUCE_BYTE_THRESHOLD   (1 << 18) // 256 KB
#define NUMC_OMP_REDUCE_BYTES_PER_THREAD (1 << 18) // 256 KB per thread

#if defined(__clang__)
#define NUMC_LOOP_SIMD \
  _Pragma("clang loop vectorize(enable) interleave(enable)")
#define NUMC_LOOP_NOSIMD _Pragma("clang loop vectorize(disable)")
#else
#define NUMC_LOOP_SIMD NUMC_PRAGMA(omp simd)
#define NUMC_LOOP_NOSIMD
#endif

/* Cap computed thread count at the runtime maximum to avoid
 * over-subscription on hybrid (P+E core) and smaller CPUs. */
#if defined(_OPENMP) || defined(HAVE_OMP)
#define NUMC_OMP_CAP_THREADS(nt)      \
  do {                                \
    int _max = omp_get_max_threads(); \
    if ((nt) > _max)                  \
      (nt) = _max;                    \
  } while (0)
#else
#define NUMC_OMP_CAP_THREADS(nt) ((void)(nt))
#endif

#define NUMC_OMP_FOR(n, elem_size, loop)                                    \
  do {                                                                      \
    size_t _total_bytes = (n) * (elem_size);                                \
    int _nthreads = (int)(_total_bytes / NUMC_OMP_BYTES_PER_THREAD);        \
    NUMC_OMP_CAP_THREADS(_nthreads);                                        \
    if (_nthreads >= 2) {                                                   \
      NUMC_PRAGMA(omp parallel for schedule(static) num_threads(_nthreads)) \
      loop                                                                  \
    } else {                                                                \
      NUMC_LOOP_SIMD                                                        \
      loop                                                                  \
    }                                                                       \
  } while (0)

#define NUMC_OMP_FOR_NOSIMD(n, elem_size, loop)                             \
  do {                                                                      \
    size_t _total_bytes = (n) * (elem_size);                                \
    int _nthreads = (int)(_total_bytes / NUMC_OMP_BYTES_PER_THREAD);        \
    NUMC_OMP_CAP_THREADS(_nthreads);                                        \
    if (_nthreads >= 2) {                                                   \
      NUMC_PRAGMA(omp parallel for schedule(static) num_threads(_nthreads)) \
      loop                                                                  \
    } else {                                                                \
      NUMC_LOOP_NOSIMD                                                      \
      loop                                                                  \
    }                                                                       \
  } while (0)

#define NUMC_OMP_REDUCE_FOR(n, elem_size, omp_op, acc, loop)         \
  do {                                                               \
    size_t _total_bytes = (n) * (elem_size);                         \
    int _nthreads = (int)(_total_bytes / NUMC_OMP_BYTES_PER_THREAD); \
    NUMC_OMP_CAP_THREADS(_nthreads);                                 \
    if (_nthreads >= 2) {                                            \
      NUMC_PRAGMA(omp parallel for reduction(omp_op : acc)                     \
                      schedule(static) num_threads(_nthreads))       \
      loop                                                           \
    } else {                                                         \
      NUMC_LOOP_SIMD                                                 \
      loop                                                           \
    }                                                                \
  } while (0)

#define NUMC_OMP_REDUCE_FOR_NOSIMD(n, elem_size, omp_op, acc, loop)  \
  do {                                                               \
    size_t _total_bytes = (n) * (elem_size);                         \
    int _nthreads = (int)(_total_bytes / NUMC_OMP_BYTES_PER_THREAD); \
    NUMC_OMP_CAP_THREADS(_nthreads);                                 \
    if (_nthreads >= 2) {                                            \
      NUMC_PRAGMA(omp parallel for reduction(omp_op : acc)                     \
                      schedule(static) num_threads(_nthreads))       \
      loop                                                           \
    } else {                                                         \
      NUMC_LOOP_NOSIMD                                               \
      loop                                                           \
    }                                                                \
  } while (0)

/* Reduction-specific OMP macros using lower thresholds. */
#define NUMC_OMP_REDUCE_FOR2(n, elem_size, omp_op, acc, loop)               \
  do {                                                                      \
    size_t _total_bytes = (n) * (elem_size);                                \
    int _nthreads = (int)(_total_bytes / NUMC_OMP_REDUCE_BYTES_PER_THREAD); \
    NUMC_OMP_CAP_THREADS(_nthreads);                                        \
    if (_nthreads >= 2) {                                                   \
      NUMC_PRAGMA(omp parallel for reduction(omp_op : acc)                     \
                      schedule(static) num_threads(_nthreads))              \
      loop                                                                  \
    } else {                                                                \
      NUMC_LOOP_SIMD                                                        \
      loop                                                                  \
    }                                                                       \
  } while (0)

/**
 * @brief Initialize runtime resources (BLIS, thread pools).
 *
 * Called during context creation to avoid first-call latency.
 */
void _numc_runtime_init(void);

#endif
