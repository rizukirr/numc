#include "dispatch.h"
#include "kernel.h"
#include "numc/dtype.h"
#include "numc/math.h"

#ifdef HAVE_BLAS

#include "internal.h"
#include <blis.h>

static bool blis_initialized = false;

/**
 * @brief Initialize BLIS and configure hierarchical threading.
 */
static void _ensure_blis_init(void) {
  if (!blis_initialized) {
    bli_init();
    blis_initialized = true;
  }
}

/**
 * @brief Set BLIS threading topology based on problem size.
 *
 * BLIS uses 5 nested loops (JC, PC, IC, JR, IR).
 * We map threads to the IC loop (L3 cache level) to maximize reuse
 * of the B matrix panel across cores sharing an L3.
 */
static void _blis_set_threading(size_t total_ops) {
  _ensure_blis_init();
#ifdef HAVE_OMP
  int nthreads = omp_get_max_threads();
  /* Bypassing multithreading for small matrices avoids sync overhead.
   * Modern BLIS "Supersonic" kernels are often faster in serial mode
   * for small problem sizes. */
  if (total_ops < 1000000) {
    bli_thread_set_ways(1, 1, 1, 1, 1);
  } else {
    /* Fat Multithreading: map all threads to IC loop. */
    bli_thread_set_ways(1, 1, nthreads, 1, 1);
  }
#endif
}

void _matmul_blis_f32(const struct NumcArray *a, const struct NumcArray *b,
                      struct NumcArray *out) {
  float alpha = 1.0f, beta = 0.0f;
  dim_t m = (dim_t)a->shape[0], k = (dim_t)a->shape[1], n = (dim_t)b->shape[1];

  /* BLIS Stride Support: allows zero-copy multiplication of views/slices */
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
  size_t total_ops =
      (size_t)a->shape[0] * (size_t)a->shape[1] * (size_t)b->shape[1];

  /* 
   * Performance Strategy:
   * 1. Use naive kernels for float32. Our C23 i-k-j loop is currently 
   *    more efficient than BLIS for common shapes in this environment.
   * 2. Use refined BLIS for large float64 (> 65k ops).
   * 3. Use naive kernels for ALL integer types.
   */
  if (total_ops >= 65536 && a->dtype == NUMC_DTYPE_FLOAT64) {
    _blis_set_threading(total_ops);
    _matmul_blis_f64(a, b, out);
    return 0;
  }
#endif

  /* Fallback to optimized naive kernels (C23 + OpenMP) */
  MatmulKernel kern = matmul_table[a->dtype];
  kern((const char *)a->data, (const char *)b->data, (char *)out->data,
       a->shape[0], b->shape[0], out->shape[1]);
  return 0;
}
