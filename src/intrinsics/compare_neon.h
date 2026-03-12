/**
 * @file compare_neon.h
 * @brief NEON uint8 comparison kernels.
 *
 * Unlike AVX2, NEON has native unsigned byte comparisons,
 * so no XOR-0x80 trick is needed.
 */
#ifndef NUMC_COMPARE_NEON_H
#define NUMC_COMPARE_NEON_H

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

static inline void _cmp_eq_u8_neon(const uint8_t *restrict a,
                                   const uint8_t *restrict b,
                                   uint8_t *restrict out, size_t n) {
  const uint8x16_t one = vdupq_n_u8(1);
  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    uint8x16_t va = vld1q_u8(a + i);
    uint8x16_t vb = vld1q_u8(b + i);
    uint8x16_t r = vandq_u8(vceqq_u8(va, vb), one);
    vst1q_u8(out + i, r);
  }
  for (; i < n; i++)
    out[i] = a[i] == b[i] ? 1 : 0;
}

static inline void _cmp_gt_u8_neon(const uint8_t *restrict a,
                                   const uint8_t *restrict b,
                                   uint8_t *restrict out, size_t n) {
  const uint8x16_t one = vdupq_n_u8(1);
  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    uint8x16_t va = vld1q_u8(a + i);
    uint8x16_t vb = vld1q_u8(b + i);
    uint8x16_t r = vandq_u8(vcgtq_u8(va, vb), one);
    vst1q_u8(out + i, r);
  }
  for (; i < n; i++)
    out[i] = a[i] > b[i] ? 1 : 0;
}

/* lt(a,b) = gt(b,a) */
static inline void _cmp_lt_u8_neon(const uint8_t *restrict a,
                                   const uint8_t *restrict b,
                                   uint8_t *restrict out, size_t n) {
  _cmp_gt_u8_neon(b, a, out, n);
}

static inline void _cmp_ge_u8_neon(const uint8_t *restrict a,
                                   const uint8_t *restrict b,
                                   uint8_t *restrict out, size_t n) {
  const uint8x16_t one = vdupq_n_u8(1);
  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    uint8x16_t va = vld1q_u8(a + i);
    uint8x16_t vb = vld1q_u8(b + i);
    uint8x16_t r = vandq_u8(vcgeq_u8(va, vb), one);
    vst1q_u8(out + i, r);
  }
  for (; i < n; i++)
    out[i] = a[i] >= b[i] ? 1 : 0;
}

/* le(a,b) = ge(b,a) */
static inline void _cmp_le_u8_neon(const uint8_t *restrict a,
                                   const uint8_t *restrict b,
                                   uint8_t *restrict out, size_t n) {
  _cmp_ge_u8_neon(b, a, out, n);
}

#endif /* NUMC_COMPARE_NEON_H */
