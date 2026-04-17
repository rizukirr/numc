#include "dispatch.h"
#include "numc/dtype.h"
#include <numc/math.h>

#include "arch_dispatch.h"
#if NUMC_HAVE_AVX512
#include "intrinsics/elemwise_avx2.h"
#include "intrinsics/elemwise_avx512.h"
#include "intrinsics/elemwise_scalar_avx2.h"
#include "intrinsics/elemwise_scalar_avx512.h"
#elif NUMC_HAVE_AVX2
#include "intrinsics/elemwise_avx2.h"
#include "intrinsics/elemwise_scalar_avx2.h"
#elif NUMC_HAVE_SVE
#include "intrinsics/elemwise_sve.h"
#include "intrinsics/elemwise_scalar_sve.h"
#elif NUMC_HAVE_NEON
#include "intrinsics/elemwise_neon.h"
#include "intrinsics/elemwise_scalar_neon.h"
#endif
#if NUMC_HAVE_RVV
#include "intrinsics/elemwise_rvv.h"
#include "intrinsics/elemwise_scalar_rvv.h"
#endif

/* -- Stamp binary elem-wise arithmetic typed kernels --------------------*/

/* add: all 10 types, native + */
#define STAMP_ADD(TE, CT) DEFINE_BINARY_KERNEL(add, TE, CT, in1 + in2)
GENERATE_NUMC_TYPES(STAMP_ADD)
#undef STAMP_ADD

/* sub: all 10 types, native - */
#define STAMP_SUB(TE, CT) DEFINE_BINARY_KERNEL(sub, TE, CT, in1 - in2)
GENERATE_NUMC_TYPES(STAMP_SUB)
#undef STAMP_SUB

/* mul: all 10 types, native * */
#define STAMP_MUL(TE, CT) DEFINE_BINARY_KERNEL(mul, TE, CT, in1 *in2)
GENERATE_NUMC_TYPES(STAMP_MUL)
#undef STAMP_MUL

/* div: specialized kernel with reciprocal optimization for scalars */
#define STAMP_DIV_S(TE, CT) DEFINE_INT_DIV_KERNEL(TE, CT, true)
GENERATE_SIGNED_INT_NUMC_TYPES(STAMP_DIV_S)
#undef STAMP_DIV_S

#define STAMP_DIV_U(TE, CT) DEFINE_INT_DIV_KERNEL(TE, CT, false)
GENERATE_UNSIGNED_INT_NUMC_TYPES(STAMP_DIV_U)
#undef STAMP_DIV_U

#define STAMP_DIV_F(TE, CT) DEFINE_FLOAT_DIV_KERNEL(TE, CT)
GENERATE_FLOAT_NUMC_TYPES(STAMP_DIV_F)
#undef STAMP_DIV_F

/* -- Dispatch tables ----------------------------------------------- */

static const NumcBinaryKernel add_table[] = {
    E(add, NUMC_DTYPE_INT8),    E(add, NUMC_DTYPE_INT16),
    E(add, NUMC_DTYPE_INT32),   E(add, NUMC_DTYPE_INT64),
    E(add, NUMC_DTYPE_UINT8),   E(add, NUMC_DTYPE_UINT16),
    E(add, NUMC_DTYPE_UINT32),  E(add, NUMC_DTYPE_UINT64),
    E(add, NUMC_DTYPE_FLOAT32), E(add, NUMC_DTYPE_FLOAT64),
};

static const NumcBinaryKernel sub_table[] = {
    E(sub, NUMC_DTYPE_INT8),    E(sub, NUMC_DTYPE_INT16),
    E(sub, NUMC_DTYPE_INT32),   E(sub, NUMC_DTYPE_INT64),
    E(sub, NUMC_DTYPE_UINT8),   E(sub, NUMC_DTYPE_UINT16),
    E(sub, NUMC_DTYPE_UINT32),  E(sub, NUMC_DTYPE_UINT64),
    E(sub, NUMC_DTYPE_FLOAT32), E(sub, NUMC_DTYPE_FLOAT64),
};

static const NumcBinaryKernel mul_table[] = {
    E(mul, NUMC_DTYPE_INT8),    E(mul, NUMC_DTYPE_INT16),
    E(mul, NUMC_DTYPE_INT32),   E(mul, NUMC_DTYPE_INT64),
    E(mul, NUMC_DTYPE_UINT8),   E(mul, NUMC_DTYPE_UINT16),
    E(mul, NUMC_DTYPE_UINT32),  E(mul, NUMC_DTYPE_UINT64),
    E(mul, NUMC_DTYPE_FLOAT32), E(mul, NUMC_DTYPE_FLOAT64),
};

static const NumcBinaryKernel div_table[] = {
    E(div, NUMC_DTYPE_INT8),    E(div, NUMC_DTYPE_INT16),
    E(div, NUMC_DTYPE_INT32),   E(div, NUMC_DTYPE_INT64),
    E(div, NUMC_DTYPE_UINT8),   E(div, NUMC_DTYPE_UINT16),
    E(div, NUMC_DTYPE_UINT32),  E(div, NUMC_DTYPE_UINT64),
    E(div, NUMC_DTYPE_FLOAT32), E(div, NUMC_DTYPE_FLOAT64),
};

/* -- SIMD fast-path dispatch for binary ops ----------------------- */

#if NUMC_HAVE_AVX512 || NUMC_HAVE_AVX2 || NUMC_HAVE_SVE || NUMC_HAVE_NEON || \
    NUMC_HAVE_RVV

/* Use the shared typedef from kernel.h */
typedef NumcFastBinKern FastBinKern;

#if NUMC_HAVE_AVX512
#define FBIN(OP, SFX) (FastBinKern) _fast_##OP##_##SFX##_avx512
#elif NUMC_HAVE_AVX2
#define FBIN(OP, SFX) (FastBinKern) _fast_##OP##_##SFX##_avx2
#elif NUMC_HAVE_SVE
#define FBIN(OP, SFX) (FastBinKern) _fast_##OP##_##SFX##_sve
#elif NUMC_HAVE_NEON
#define FBIN(OP, SFX) (FastBinKern) _fast_##OP##_##SFX##_neon
#elif NUMC_HAVE_RVV
#define FBIN(OP, SFX) (FastBinKern) _fast_##OP##_##SFX##_rvv
#endif

#define FBIN_TABLE(OP)                           \
  static const FastBinKern OP##_fast_table[] = { \
      [NUMC_DTYPE_INT8] = FBIN(OP, i8),          \
      [NUMC_DTYPE_INT16] = FBIN(OP, i16),        \
      [NUMC_DTYPE_INT32] = FBIN(OP, i32),        \
      [NUMC_DTYPE_INT64] = FBIN(OP, i64),        \
      [NUMC_DTYPE_UINT8] = FBIN(OP, u8),         \
      [NUMC_DTYPE_UINT16] = FBIN(OP, u16),       \
      [NUMC_DTYPE_UINT32] = FBIN(OP, u32),       \
      [NUMC_DTYPE_UINT64] = FBIN(OP, u64),       \
      [NUMC_DTYPE_FLOAT32] = FBIN(OP, f32),      \
      [NUMC_DTYPE_FLOAT64] = FBIN(OP, f64),      \
  }

FBIN_TABLE(add);
FBIN_TABLE(sub);
FBIN_TABLE(mul);
#undef FBIN_TABLE
#undef FBIN

/* Dispatch: use SIMD kernel with OMP chunking for contiguous arrays,
   fall back to generic _binary_op for non-contiguous / broadcast */
#define DEFINE_BINARY_SIMD(NAME, FAST_TABLE, FALLBACK_TABLE)                   \
  int numc_##NAME(const NumcArray *a, const NumcArray *b, NumcArray *out) {    \
    int err = _check_binary(a, b, out);                                        \
    if (err)                                                                   \
      return err;                                                              \
    if (a->is_contiguous && b->is_contiguous && out->is_contiguous &&          \
        a->dim == b->dim) {                                                    \
      bool same_shape = true;                                                  \
      for (size_t d = 0; d < a->dim; d++)                                      \
        if (a->shape[d] != b->shape[d]) {                                      \
          same_shape = false;                                                  \
          break;                                                               \
        }                                                                      \
      if (same_shape) {                                                        \
        FastBinKern kern = FAST_TABLE[a->dtype];                               \
        size_t n = a->size, es = a->elem_size;                                 \
        int nt = (int)((n * es) / NUMC_OMP_BYTES_PER_THREAD);                  \
        NUMC_OMP_CAP_THREADS(nt);                                              \
        if (nt >= 2) {                                                         \
          size_t chunk = (n + (size_t)nt - 1) / (size_t)nt;                    \
          NUMC_PRAGMA(                                                      \
              omp parallel for schedule(static) num_threads(nt))               \
          for (int t = 0; t < nt; t++) {                                       \
            size_t s = (size_t)t * chunk;                                      \
            size_t e = s + chunk;                                              \
            if (e > n)                                                         \
              e = n;                                                           \
            if (s < n)                                                         \
              kern((const char *)a->data + s * es,                             \
                   (const char *)b->data + s * es, (char *)out->data + s * es, \
                   e - s);                                                     \
          }                                                                    \
        } else {                                                               \
          kern(a->data, b->data, out->data, n);                                \
        }                                                                      \
        return 0;                                                              \
      }                                                                        \
    }                                                                          \
    _binary_op_ex(a, b, out, FALLBACK_TABLE, FAST_TABLE[a->dtype]);            \
    return 0;                                                                  \
  }

DEFINE_BINARY_SIMD(add, add_fast_table, add_table)
DEFINE_BINARY_SIMD(sub, sub_fast_table, sub_table)
DEFINE_BINARY_SIMD(mul, mul_fast_table, mul_table)
#undef DEFINE_BINARY_SIMD

#endif /* SIMD available */

/* -- Public API ---------------------------------------------------- */

#define DEFINE_ELEMWISE_BINARY(NAME, TABLE)                                 \
  int numc_##NAME(const NumcArray *a, const NumcArray *b, NumcArray *out) { \
    int err = _check_binary(a, b, out);                                     \
    if (err)                                                                \
      return err;                                                           \
    _binary_op(a, b, out, TABLE);                                           \
    return 0;                                                               \
  }

#define DEFINE_ELEMWISE_SCALAR(NAME, TABLE)                       \
  int numc_##NAME##_scalar(const NumcArray *a, double scalar,     \
                           NumcArray *out) {                      \
    int err = _check_unary(a, out);                               \
    if (err)                                                      \
      return err;                                                 \
    char buf[8];                                                  \
    _double_to_dtype(scalar, a->dtype, buf);                      \
    _scalar_op(a, buf, out, TABLE);                               \
    return 0;                                                     \
  }                                                               \
  int numc_##NAME##_scalar_inplace(NumcArray *a, double scalar) { \
    return _scalar_op_inplace(a, scalar, TABLE);                  \
  }

#if !(NUMC_HAVE_AVX512 || NUMC_HAVE_AVX2 || NUMC_HAVE_SVE || NUMC_HAVE_NEON || \
      NUMC_HAVE_RVV)
DEFINE_ELEMWISE_BINARY(add, add_table)
DEFINE_ELEMWISE_BINARY(sub, sub_table)
DEFINE_ELEMWISE_BINARY(mul, mul_table)
#endif
DEFINE_ELEMWISE_BINARY(div, div_table)

/* -- SIMD scalar arithmetic dispatch -------------------------------- */

#if NUMC_HAVE_AVX512 || NUMC_HAVE_AVX2 || NUMC_HAVE_SVE || NUMC_HAVE_NEON || \
    NUMC_HAVE_RVV

typedef void (*FastScKern)(const void *restrict, const void *restrict,
                           void *restrict, size_t);

#if NUMC_HAVE_AVX512
#define FSC(OP, SFX) (FastScKern) _fast_##OP##_scalar_##SFX##_avx512
#elif NUMC_HAVE_AVX2
#define FSC(OP, SFX) (FastScKern) _fast_##OP##_scalar_##SFX##_avx2
#elif NUMC_HAVE_SVE
#define FSC(OP, SFX) (FastScKern) _fast_##OP##_scalar_##SFX##_sve
#elif NUMC_HAVE_NEON
#define FSC(OP, SFX) (FastScKern) _fast_##OP##_scalar_##SFX##_neon
#elif NUMC_HAVE_RVV
#define FSC(OP, SFX) (FastScKern) _fast_##OP##_scalar_##SFX##_rvv
#endif

#define FSC_TABLE(OP)                          \
  static const FastScKern OP##_sc_fast[] =     \
      {                                        \
          [NUMC_DTYPE_INT8] = FSC(OP, i8),     \
          [NUMC_DTYPE_INT16] = FSC(OP, i16),   \
          [NUMC_DTYPE_INT32] = FSC(OP, i32),   \
          [NUMC_DTYPE_INT64] = FSC(OP, i64),   \
          [NUMC_DTYPE_UINT8] = FSC(OP, u8),    \
          [NUMC_DTYPE_UINT16] = FSC(OP, u16),  \
          [NUMC_DTYPE_UINT32] = FSC(OP, u32),  \
          [NUMC_DTYPE_UINT64] = FSC(OP, u64),  \
          [NUMC_DTYPE_FLOAT32] = FSC(OP, f32), \
          [NUMC_DTYPE_FLOAT64] = FSC(OP, f64), \
  }

FSC_TABLE(add);
FSC_TABLE(sub);
FSC_TABLE(mul);
#undef FSC_TABLE
#undef FSC

#define DEFINE_SCALAR_SIMD(NAME, FAST_TABLE, FALLBACK_TABLE)      \
  int numc_##NAME##_scalar(const NumcArray *a, double scalar,     \
                           NumcArray *out) {                      \
    int err = _check_unary(a, out);                               \
    if (err)                                                      \
      return err;                                                 \
    char buf[8];                                                  \
    _double_to_dtype(scalar, a->dtype, buf);                      \
    if (a->is_contiguous && out->is_contiguous) {                 \
      FastScKern kern = FAST_TABLE[a->dtype];                     \
      size_t n = a->size, es = a->elem_size, total = n * 2 * es;  \
      int nt = (int)(total / NUMC_OMP_BYTES_PER_THREAD);          \
      NUMC_OMP_CAP_THREADS(nt);                                   \
      if (nt >= 2) {                                              \
        size_t chunk = (n + (size_t)nt - 1) / (size_t)nt;         \
        NUMC_PRAGMA(                                                   \
            omp parallel for schedule(static) num_threads(nt))    \
        for (int t = 0; t < nt; t++) {                            \
          size_t s = (size_t)t * chunk;                           \
          size_t e = s + chunk;                                   \
          if (e > n)                                              \
            e = n;                                                \
          if (s < n)                                              \
            kern((const char *)a->data + s * es, buf,             \
                 (char *)out->data + s * es, e - s);              \
        }                                                         \
      } else {                                                    \
        kern(a->data, buf, out->data, n);                         \
      }                                                           \
      return 0;                                                   \
    }                                                             \
    _scalar_op(a, buf, out, FALLBACK_TABLE);                      \
    return 0;                                                     \
  }                                                               \
  int numc_##NAME##_scalar_inplace(NumcArray *a, double scalar) { \
    if (!a)                                                       \
      return NUMC_ERR_NULL;                                       \
    char buf[8];                                                  \
    _double_to_dtype(scalar, a->dtype, buf);                      \
    if (a->is_contiguous) {                                       \
      FastScKern kern = FAST_TABLE[a->dtype];                     \
      size_t n = a->size, es = a->elem_size, total = n * 2 * es;  \
      int nt = (int)(total / NUMC_OMP_BYTES_PER_THREAD);          \
      NUMC_OMP_CAP_THREADS(nt);                                   \
      if (nt >= 2) {                                              \
        size_t chunk = (n + (size_t)nt - 1) / (size_t)nt;         \
        NUMC_PRAGMA(                                                   \
            omp parallel for schedule(static) num_threads(nt))    \
        for (int t = 0; t < nt; t++) {                            \
          size_t s = (size_t)t * chunk;                           \
          size_t e = s + chunk;                                   \
          if (e > n)                                              \
            e = n;                                                \
          if (s < n)                                              \
            kern((const char *)a->data + s * es, buf,             \
                 (char *)a->data + s * es, e - s);                \
        }                                                         \
      } else {                                                    \
        kern(a->data, buf, a->data, n);                           \
      }                                                           \
      return 0;                                                   \
    }                                                             \
    return _scalar_op_inplace(a, scalar, FALLBACK_TABLE);         \
  }

DEFINE_SCALAR_SIMD(add, add_sc_fast, add_table)
DEFINE_SCALAR_SIMD(sub, sub_sc_fast, sub_table)
DEFINE_SCALAR_SIMD(mul, mul_sc_fast, mul_table)
#undef DEFINE_SCALAR_SIMD

/* div: keep generic path (has power-of-2 / reciprocal optimizations) */
DEFINE_ELEMWISE_SCALAR(div, div_table)

#else /* No SIMD available */

DEFINE_ELEMWISE_SCALAR(add, add_table)
DEFINE_ELEMWISE_SCALAR(sub, sub_table)
DEFINE_ELEMWISE_SCALAR(mul, mul_table)
DEFINE_ELEMWISE_SCALAR(div, div_table)

#endif
