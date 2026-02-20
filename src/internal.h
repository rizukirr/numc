#ifndef NUMC_INTERNAL_H
#define NUMC_INTERNAL_H

#include "arena.h"
#include <numc/dtype.h>
#include <numc/error.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* --- Constants --- */

#define NUMC_MAX_DIMENSIONS 64
#define NUMC_MAX_MEMORY 8388608 // 8MB
#define NUMC_SIMD_ALIGN 32

/* --- Opaque struct definitions (private) --- */

struct NumcCtx {
  Arena *arena;
};

struct NumcArray {
  struct NumcCtx *ctx;
  void *data;
  size_t shape[NUMC_MAX_DIMENSIONS], strides[NUMC_MAX_DIMENSIONS];
  size_t dim, elem_size, size, capacity;
  bool is_contiguous;
  NumcDType dtype;
};

/* --- Aligned alloc (private) --- */

void *numc_malloc(size_t alignment, size_t size);
void numc_free(void *ptr);

/* --- OMP macros (private) --- */

#ifdef _OPENMP
#include <omp.h>
#endif

#define NUMC_STR_(x) #x
#define NUMC_STR(x) NUMC_STR_(x)
#define NUMC_PRAGMA(x) _Pragma(NUMC_STR(x))

#if defined(__clang__)
#define NUMC_UNROLL(n) _Pragma(NUMC_STR(clang loop unroll_count(n)))
#elif defined(__GNUC__)
#define NUMC_UNROLL(n) _Pragma(NUMC_STR(GCC unroll n))
#else
#define NUMC_UNROLL(n)
#endif

#define NUMC_OMP_BYTE_THRESHOLD (1 << 20)   // 1 MB total to enable OMP
#define NUMC_OMP_BYTES_PER_THREAD (1 << 20) // 1 MB minimum work per thread

#define NUMC_OMP_FOR(n, elem_size, loop)                                       \
  do {                                                                         \
    size_t _total_bytes = (n) * (elem_size);                                   \
    if (_total_bytes > NUMC_OMP_BYTE_THRESHOLD) {                              \
      int _nthreads = (int)(_total_bytes / NUMC_OMP_BYTES_PER_THREAD);         \
      if (_nthreads < 1)                                                       \
        _nthreads = 1;                                                         \
      NUMC_PRAGMA(omp parallel for schedule(static) num_threads(_nthreads))    \
      loop                                                                     \
    } else {                                                                   \
      loop                                                                     \
    }                                                                          \
  } while (0)

#endif
