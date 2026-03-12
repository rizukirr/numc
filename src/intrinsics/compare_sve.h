/**
 * @file compare_sve.h
 * @brief SVE uint8 comparison kernels.
 *
 * SVE has native unsigned byte comparisons and predicated tail
 * handling via svwhilelt, so no separate tail loop is needed.
 */
#ifndef NUMC_COMPARE_SVE_H
#define NUMC_COMPARE_SVE_H

#include <arm_sve.h>
#include <stddef.h>
#include <stdint.h>

static inline void _cmp_eq_u8_sve(const uint8_t *restrict a,
                                  const uint8_t *restrict b,
                                  uint8_t *restrict out, size_t n) {
  size_t vl = svcntb();
  for (size_t i = 0; i < n; i += vl) {
    svbool_t pg = svwhilelt_b8((uint32_t)i, (uint32_t)n);
    svuint8_t va = svld1_u8(pg, a + i);
    svuint8_t vb = svld1_u8(pg, b + i);
    svbool_t pred = svcmpeq_u8(pg, va, vb);
    svuint8_t r = svsel_u8(pred, svdup_u8(1), svdup_u8(0));
    svst1_u8(pg, out + i, r);
  }
}

static inline void _cmp_gt_u8_sve(const uint8_t *restrict a,
                                  const uint8_t *restrict b,
                                  uint8_t *restrict out, size_t n) {
  size_t vl = svcntb();
  for (size_t i = 0; i < n; i += vl) {
    svbool_t pg = svwhilelt_b8((uint32_t)i, (uint32_t)n);
    svuint8_t va = svld1_u8(pg, a + i);
    svuint8_t vb = svld1_u8(pg, b + i);
    svbool_t pred = svcmpgt_u8(pg, va, vb);
    svuint8_t r = svsel_u8(pred, svdup_u8(1), svdup_u8(0));
    svst1_u8(pg, out + i, r);
  }
}

/* lt(a,b) = gt(b,a) */
static inline void _cmp_lt_u8_sve(const uint8_t *restrict a,
                                  const uint8_t *restrict b,
                                  uint8_t *restrict out, size_t n) {
  _cmp_gt_u8_sve(b, a, out, n);
}

static inline void _cmp_ge_u8_sve(const uint8_t *restrict a,
                                  const uint8_t *restrict b,
                                  uint8_t *restrict out, size_t n) {
  size_t vl = svcntb();
  for (size_t i = 0; i < n; i += vl) {
    svbool_t pg = svwhilelt_b8((uint32_t)i, (uint32_t)n);
    svuint8_t va = svld1_u8(pg, a + i);
    svuint8_t vb = svld1_u8(pg, b + i);
    svbool_t pred = svcmpge_u8(pg, va, vb);
    svuint8_t r = svsel_u8(pred, svdup_u8(1), svdup_u8(0));
    svst1_u8(pg, out + i, r);
  }
}

/* le(a,b) = ge(b,a) */
static inline void _cmp_le_u8_sve(const uint8_t *restrict a,
                                  const uint8_t *restrict b,
                                  uint8_t *restrict out, size_t n) {
  _cmp_ge_u8_sve(b, a, out, n);
}

#endif /* NUMC_COMPARE_SVE_H */
