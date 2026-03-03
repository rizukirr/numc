#include "dispatch.h"
#include "kernel.h"
#include "numc/dtype.h"
#include "numc/math.h"

#ifdef HAVE_BLAS

#include "internal.h"
#include <blis/blis.h>

void _matmul_blis_f32(const struct NumcArray *a, const struct NumcArray *b,
                      struct NumcArray *out) {
  float alpha = 1.0f;
  float beta = 0.0f;

  dim_t m = a->shape[0];
  dim_t k = a->shape[1];
  dim_t n = b->shape[1];

  inc_t rs_a = a->strides[0] / sizeof(float);
  inc_t cs_a = a->strides[1] / sizeof(float);

  inc_t rs_b = b->strides[0] / sizeof(float);
  inc_t cs_b = b->strides[1] / sizeof(float);

  inc_t rs_c = out->strides[0] / sizeof(float);
  inc_t cs_c = out->strides[1] / sizeof(float);

  bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, m, n, k, &alpha,
            (float *)a->data, rs_a, cs_a, (float *)b->data, rs_b, cs_b, &beta,
            (float *)out->data, rs_c, cs_c);
}

void _matmul_blis_f64(const struct NumcArray *a, const struct NumcArray *b,
                      struct NumcArray *out) {
  double alpha = 1.0;
  double beta = 0.0;

  dim_t m = a->shape[0];
  dim_t k = a->shape[1];
  dim_t n = b->shape[1];

  inc_t rs_a = a->strides[0] / sizeof(double);
  inc_t cs_a = a->strides[1] / sizeof(double);

  inc_t rs_b = b->strides[0] / sizeof(double);
  inc_t cs_b = b->strides[1] / sizeof(double);

  inc_t rs_c = out->strides[0] / sizeof(double);
  inc_t cs_c = out->strides[1] / sizeof(double);

  bli_dgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, m, n, k, &alpha,
            (double *)a->data, rs_a, cs_a, (double *)b->data, rs_b, cs_b, &beta,
            (double *)out->data, rs_c, cs_c);
}

#endif

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

#ifdef HAVE_BLAS
static bool blis_initialized = false;
static void _ensure_blis_init(void) {
  if (!blis_initialized) {
    bli_init();
    blis_initialized = true;
  }
}

static void _blis_set_threading(size_t total_ops) {
  _ensure_blis_init();
#ifdef HAVE_OMP
  if (total_ops < 1000000) {
    bli_thread_set_ways(1, 1, 1, 1, 1);
  } else {
    /* Use Fat Multithreading: map threads to IC loop (L3 cache level) */
    bli_thread_set_ways(1, 1, omp_get_max_threads(), 1, 1);
  }
#endif
}
#endif

int numc_matmul(const NumcArray *a, const NumcArray *b, NumcArray *out) {
  int err = _check_matmul(a, b, out);
  if (err)
    return err;

#ifdef HAVE_BLAS
  size_t total_ops =
      (size_t)a->shape[0] * (size_t)a->shape[1] * (size_t)b->shape[1];
  /* Use BLIS only for float64 at reasonable sizes.
   * Our naive float32 i-k-j kernel is currently highly competitive
   * due to aggressive compiler vectorization and lower setup overhead. */
  if (total_ops >= 65536 && a->dtype == NUMC_DTYPE_FLOAT64) {
    _blis_set_threading(total_ops);
    _matmul_blis_f64(a, b, out);
    return 0;
  }
#endif

  MatmulKernel kern = matmul_table[a->dtype];
  kern((const char *)a->data, (const char *)b->data, (char *)out->data,
       a->shape[0], b->shape[0], out->shape[1]);
  return 0;
}
