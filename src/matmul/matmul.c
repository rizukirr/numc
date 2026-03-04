#include "dispatch.h"
#include "kernel.h"
#include "numc/dtype.h"
#include "numc/math.h"
#include "internal.h"

#ifdef HAVE_BLAS
#include <blis.h>
#include <pthread.h>

static pthread_once_t blis_once = PTHREAD_ONCE_INIT;

/* libomp defaults to KMP_BLOCKTIME=0, immediately sleeping OpenMP
 * threads after each parallel region.  Waking them from OS idle
 * states costs 40-70 ms — catastrophic for sub-ms sgemm calls.
 * Setting KMP_BLOCKTIME=200 (the MKL default) keeps threads spinning
 * for 200 ms between calls.  Must run before the first OpenMP
 * parallel region; overwrite=0 respects user-set values.
 *
 * On Intel hybrid CPUs (P-core + E-core), users should additionally
 * set OMP_PLACES=cores and BLIS_NUM_THREADS=<P-core count> to avoid
 * scheduling BLIS threads on slower E-cores. */
#ifdef NUMC_BLIS_OPTIMIZED
__attribute__((constructor)) static void _numc_omp_init(void) {
  if (!getenv("KMP_BLOCKTIME") && !getenv("OMP_WAIT_POLICY")) {
    setenv("KMP_BLOCKTIME", "200", 0);
  }
}
#endif

static void _blis_init_once(void) { bli_init(); }
#endif

void _numc_runtime_init(void) {
#ifdef HAVE_BLAS
  pthread_once(&blis_once, _blis_init_once);
#endif

#ifdef HAVE_OMP
  /* Pre-warm OpenMP thread pool by running a dummy parallel loop.
   * This avoids the ~50ms 'cold start' penalty on the first math call. */
  #pragma omp parallel
  {
    (void)0; 
  }
#endif
}

#ifdef HAVE_BLAS
/**
 * @brief Set BLIS threading for the current call.
 *
 * Optimized BLIS: all calls here are >= 32k ops (dispatch threshold),
 *   so we always use full threading with BLIS auto-factorization.
 *
 * System/generic BLIS:
 *   - < 65k ops:  serial (avoids fork/join overhead).
 *   - >= 65k ops: all threads on IC loop.
 */
static void _blis_set_threading(size_t total_ops) {
  pthread_once(&blis_once, _blis_init_once);
#ifdef HAVE_OMP
  int nthreads = omp_get_max_threads();

#ifdef NUMC_BLIS_OPTIMIZED
  (void)total_ops;
  bli_thread_set_num_threads(nthreads);
#else
  if (total_ops < 65536) {
    bli_thread_set_ways(1, 1, 1, 1, 1);
  } else {
    bli_thread_set_ways(1, 1, nthreads, 1, 1);
  }
#endif /* NUMC_BLIS_OPTIMIZED */

#endif /* HAVE_OMP */
}

void _matmul_blis_f32(const struct NumcArray *a, const struct NumcArray *b,
                      struct NumcArray *out) {
  float alpha = 1.0f, beta = 0.0f;
  dim_t m = (dim_t)a->shape[0], k = (dim_t)a->shape[1], n = (dim_t)b->shape[1];

  /* BLIS Stride Support: allows zero-copy multiplication of views/slices.
   * Strides must be aligned to element size; guaranteed by arena allocator. */
  assert(a->strides[0] % sizeof(float) == 0 && "stride not aligned to float");
  assert(a->strides[1] % sizeof(float) == 0 && "stride not aligned to float");
  assert(b->strides[0] % sizeof(float) == 0 && "stride not aligned to float");
  assert(b->strides[1] % sizeof(float) == 0 && "stride not aligned to float");
  inc_t rs_a = (inc_t)(a->strides[0] / sizeof(float));
  inc_t cs_a = (inc_t)(a->strides[1] / sizeof(float));
  inc_t rs_b = (inc_t)(b->strides[0] / sizeof(float));
  inc_t cs_b = (inc_t)(b->strides[1] / sizeof(float));
  inc_t rs_c = (inc_t)(out->strides[0] / sizeof(float));
  inc_t cs_c = (inc_t)(out->strides[1] / sizeof(float));

  bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, m, n, k, &alpha,
            (float *)a->data, rs_a, cs_a, (float *)b->data, rs_b, cs_b, &beta,
            (float *)out->data, rs_c, cs_c);
}

void _matmul_blis_f64(const struct NumcArray *a, const struct NumcArray *b,
                      struct NumcArray *out) {
  double alpha = 1.0, beta = 0.0;
  dim_t m = (dim_t)a->shape[0], k = (dim_t)a->shape[1], n = (dim_t)b->shape[1];

  assert(a->strides[0] % sizeof(double) == 0 && "stride not aligned to double");
  assert(a->strides[1] % sizeof(double) == 0 && "stride not aligned to double");
  assert(b->strides[0] % sizeof(double) == 0 && "stride not aligned to double");
  assert(b->strides[1] % sizeof(double) == 0 && "stride not aligned to double");
  inc_t rs_a = (inc_t)(a->strides[0] / sizeof(double));
  inc_t cs_a = (inc_t)(a->strides[1] / sizeof(double));
  inc_t rs_b = (inc_t)(b->strides[0] / sizeof(double));
  inc_t cs_b = (inc_t)(b->strides[1] / sizeof(double));
  inc_t rs_c = (inc_t)(out->strides[0] / sizeof(double));
  inc_t cs_c = (inc_t)(out->strides[1] / sizeof(double));

  bli_dgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, m, n, k, &alpha,
            (double *)a->data, rs_a, cs_a, (double *)b->data, rs_b, cs_b, &beta,
            (double *)out->data, rs_c, cs_c);
}
#endif

/* ── Dispatch table for naive C23 kernels ────────────────────────── */

#define E(TE) [TE] = _matmul_naive_##TE
static const MatmulKernel matmul_table[] = {
    E(NUMC_DTYPE_INT8),    E(NUMC_DTYPE_INT16),  E(NUMC_DTYPE_INT32),
    E(NUMC_DTYPE_INT64),   E(NUMC_DTYPE_UINT8),  E(NUMC_DTYPE_UINT16),
    E(NUMC_DTYPE_UINT32),  E(NUMC_DTYPE_UINT64), E(NUMC_DTYPE_FLOAT32),
    E(NUMC_DTYPE_FLOAT64),
};
#undef E

int numc_matmul_naive(const NumcArray *a, const NumcArray *b, NumcArray *out) {
  int err = _check_matmul(a, b, out);
  if (err)
    return err;

  MatmulKernel kern = matmul_table[a->dtype];
  kern((const char *)a->data, (const char *)b->data, (char *)out->data,
       a->shape[0], b->shape[0], out->shape[1]);
  return 0;
}

/* ── Public Unified API ──────────────────────────────────────────── */

int numc_matmul(const NumcArray *a, const NumcArray *b, NumcArray *out) {
  int err = _check_matmul(a, b, out);
  if (err)
    return err;

#ifdef HAVE_BLAS
  /* Saturating multiply: overflow → SIZE_MAX (always dispatches to BLIS) */
  size_t total_ops;
  if (__builtin_mul_overflow(a->shape[0], a->shape[1], &total_ops) ||
      __builtin_mul_overflow(total_ops, b->shape[1], &total_ops))
    total_ops = SIZE_MAX;

#ifdef NUMC_BLIS_OPTIMIZED
  /*
   * Optimized BLIS (vendored, auto-configured with CPU kernels):
   * - sgemm for float32 >= 64k ops
   * - dgemm for float64 >= 64k ops
   * - Below 64k: falls through to naive kernels (no BLIS overhead)
   */
  if (total_ops >= 65536 && a->dtype == NUMC_DTYPE_FLOAT32) {
    _blis_set_threading(total_ops);
    _matmul_blis_f32(a, b, out);
    return 0;
  }
  if (total_ops >= 65536 && a->dtype == NUMC_DTYPE_FLOAT64) {
    _blis_set_threading(total_ops);
    _matmul_blis_f64(a, b, out);
    return 0;
  }
#else
  /*
   * System/generic BLIS:
   * - dgemm only for float64 >= 65k ops (generic sgemm is unreliable)
   * - IC-only threading
   */
  if (total_ops >= 65536 && a->dtype == NUMC_DTYPE_FLOAT64) {
    _blis_set_threading(total_ops);
    _matmul_blis_f64(a, b, out);
    return 0;
  }
#endif /* NUMC_BLIS_OPTIMIZED */

#endif /* HAVE_BLAS */

  /* Fallback to optimized naive kernels (C23 + OpenMP) */
  MatmulKernel kern = matmul_table[a->dtype];
  kern((const char *)a->data, (const char *)b->data, (char *)out->data,
       a->shape[0], b->shape[0], out->shape[1]);
  return 0;
}
