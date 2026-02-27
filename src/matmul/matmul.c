#include "dispatch.h"
#include "kernel.h"
#include "numc/dtype.h"
#include "numc/math.h"

#define E(TE) [TE] = _matmul_naive_##TE

static const MatmulKernel _matmul_table[] = {
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

  MatmulKernel kern = _matmul_table[a->dtype];
  kern((const char *)a->data, (const char *)b->data, (char *)out->data,
       a->shape[0], b->shape[0], out->shape[1]);
  return 0;
}
