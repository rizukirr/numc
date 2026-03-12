/**
 * @file compare_rvv.h
 * @brief RVV uint8 comparison kernels.
 *
 * RVV has native unsigned byte comparisons and natural tail
 * handling via vsetvl.
 */
#ifndef NUMC_COMPARE_RVV_H
#define NUMC_COMPARE_RVV_H

#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

static inline void _cmp_eq_u8_rvv(const uint8_t *restrict a,
                                  const uint8_t *restrict b,
                                  uint8_t *restrict out, size_t n) {
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e8m4(n - i);
    vuint8m4_t va = __riscv_vle8_v_u8m4(a + i, vl);
    vuint8m4_t vb = __riscv_vle8_v_u8m4(b + i, vl);
    vbool2_t mask = __riscv_vmseq_vv_u8m4_b2(va, vb, vl);
    vuint8m4_t r =
        __riscv_vmerge_vxm_u8m4(__riscv_vmv_v_x_u8m4(0, vl), 1, mask, vl);
    __riscv_vse8_v_u8m4(out + i, r, vl);
  }
}

static inline void _cmp_gt_u8_rvv(const uint8_t *restrict a,
                                  const uint8_t *restrict b,
                                  uint8_t *restrict out, size_t n) {
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e8m4(n - i);
    vuint8m4_t va = __riscv_vle8_v_u8m4(a + i, vl);
    vuint8m4_t vb = __riscv_vle8_v_u8m4(b + i, vl);
    vbool2_t mask = __riscv_vmsgtu_vv_u8m4_b2(va, vb, vl);
    vuint8m4_t r =
        __riscv_vmerge_vxm_u8m4(__riscv_vmv_v_x_u8m4(0, vl), 1, mask, vl);
    __riscv_vse8_v_u8m4(out + i, r, vl);
  }
}

/* lt(a,b) = gt(b,a) */
static inline void _cmp_lt_u8_rvv(const uint8_t *restrict a,
                                  const uint8_t *restrict b,
                                  uint8_t *restrict out, size_t n) {
  _cmp_gt_u8_rvv(b, a, out, n);
}

static inline void _cmp_ge_u8_rvv(const uint8_t *restrict a,
                                  const uint8_t *restrict b,
                                  uint8_t *restrict out, size_t n) {
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e8m4(n - i);
    vuint8m4_t va = __riscv_vle8_v_u8m4(a + i, vl);
    vuint8m4_t vb = __riscv_vle8_v_u8m4(b + i, vl);
    vbool2_t mask = __riscv_vmsgeu_vv_u8m4_b2(va, vb, vl);
    vuint8m4_t r =
        __riscv_vmerge_vxm_u8m4(__riscv_vmv_v_x_u8m4(0, vl), 1, mask, vl);
    __riscv_vse8_v_u8m4(out + i, r, vl);
  }
}

/* le(a,b) = ge(b,a) */
static inline void _cmp_le_u8_rvv(const uint8_t *restrict a,
                                  const uint8_t *restrict b,
                                  uint8_t *restrict out, size_t n) {
  _cmp_ge_u8_rvv(b, a, out, n);
}

#endif /* NUMC_COMPARE_RVV_H */
