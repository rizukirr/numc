#include "math_elemwise.h"
#include "../../../math_helper.h"
#include "../../simd_constants.h"
#include "../../simd_util.h"

#include <immintrin.h>

#define AVX_LOOP_ALIGNED(end, a, b, out, simd_op)                               \
  for (size_t i = 0; i < end; i += AVX_LEN_FLOAT) {                           \
    __m256 av = _mm256_load_ps(a + i);                                         \
    __m256 bv = _mm256_load_ps(b + i);                                         \
    _mm256_store_ps(out + i, simd_op(av, bv));                                 \
  }

#define AVX_LOOP_UNALIGNED(end, a, b, out, simd_op)                            \
  for (size_t i = 0; i < end; i += AVX_LEN_FLOAT) {                           \
    __m256 av = _mm256_loadu_ps(a + i);                                        \
    __m256 bv = _mm256_loadu_ps(b + i);                                        \
    _mm256_storeu_ps(out + i, simd_op(av, bv));                                \
  }

#define AVX_ELEMWISE_F32(name, simd_op, scalar_op)                             \
  void name(float *a, float *b, float *out, size_t n) {                        \
    size_t end = n & ~7;                                                       \
    size_t _total_bytes = n * sizeof(float);                                   \
    if (_total_bytes > NUMC_OMP_BYTE_THRESHOLD) {                              \
      int _nt = (int)(_total_bytes / NUMC_OMP_BYTES_PER_THREAD);              \
      if (_nt < 1) _nt = 1;                                                   \
      if (is_aligned((uintptr_t)a, (uintptr_t)b, (uintptr_t)out,              \
                     AVX_LEN_BYTES)) {                                         \
        NUMC_PRAGMA(omp parallel for schedule(static) num_threads(_nt))        \
        AVX_LOOP_ALIGNED(end, a, b, out, simd_op)                             \
      } else {                                                                 \
        NUMC_PRAGMA(omp parallel for schedule(static) num_threads(_nt))        \
        AVX_LOOP_UNALIGNED(end, a, b, out, simd_op)                           \
      }                                                                        \
    } else {                                                                   \
      if (is_aligned((uintptr_t)a, (uintptr_t)b, (uintptr_t)out,              \
                     AVX_LEN_BYTES)) {                                         \
        NUMC_UNROLL(4)                                                         \
        AVX_LOOP_ALIGNED(end, a, b, out, simd_op)                             \
      } else {                                                                 \
        NUMC_UNROLL(4)                                                         \
        AVX_LOOP_UNALIGNED(end, a, b, out, simd_op)                           \
      }                                                                        \
    }                                                                          \
    for (size_t i = end; i < n; i++) {                                         \
      out[i] = a[i] scalar_op b[i];                                            \
    }                                                                          \
  }

AVX_ELEMWISE_F32(array_add_f32_avx, _mm256_add_ps, +)
AVX_ELEMWISE_F32(array_mul_f32_avx, _mm256_mul_ps, *)
AVX_ELEMWISE_F32(array_sub_f32_avx, _mm256_sub_ps, -)
AVX_ELEMWISE_F32(array_div_f32_avx, _mm256_div_ps, /)
