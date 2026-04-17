/**
 * @file compare_sve.h
 * @brief SVE binary comparison kernels — uint8 output (0/1).
 *
 * All comparison functions output uint8_t* (NumPy-compatible bool).
 * SVE predicated comparisons return svbool_t; convert to 0/1 via svsel_u8
 * with a byte-width output predicate for the store.
 *
 * For types wider than 8 bits, the input predicate (b16/b32/b64) covers
 * the correct input elements, but we produce one uint8 per element.
 * We use svuzp1_b8 to convert wider predicates to byte predicates
 * for the output store, or just use svwhilelt_b8 on the output range.
 */
#ifndef NUMC_COMPARE_SVE_H
#define NUMC_COMPARE_SVE_H

#include <arm_sve.h>
#include <stddef.h>
#include <stdint.h>

/* ====================================================================
 * 8-bit integer comparisons (signed + unsigned): natural byte output
 * ================================================================ */

#define FAST_CMP_8_SVE(SFX, CT, SVT, LD, CMPEQ, CMPGT, CMPLT, CMPGE, CMPLE) \
  static inline void _fast_eq_##SFX##_sve(const void *restrict ap,          \
                                          const void *restrict bp,          \
                                          void *restrict op, size_t n) {    \
    const CT *a = (const CT *)ap;                                           \
    const CT *b = (const CT *)bp;                                           \
    uint8_t *out = (uint8_t *)op;                                           \
    size_t vl = svcntb();                                                   \
    for (size_t i = 0; i < n; i += vl) {                                    \
      svbool_t pg = svwhilelt_b8((uint32_t)i, (uint32_t)n);                 \
      SVT va = LD(pg, a + i);                                               \
      SVT vb = LD(pg, b + i);                                               \
      svbool_t cmp = CMPEQ(pg, va, vb);                                     \
      svst1_u8(pg, out + i, svsel_u8(cmp, svdup_u8(1), svdup_u8(0)));       \
    }                                                                       \
  }                                                                         \
  static inline void _fast_gt_##SFX##_sve(const void *restrict ap,          \
                                          const void *restrict bp,          \
                                          void *restrict op, size_t n) {    \
    const CT *a = (const CT *)ap;                                           \
    const CT *b = (const CT *)bp;                                           \
    uint8_t *out = (uint8_t *)op;                                           \
    size_t vl = svcntb();                                                   \
    for (size_t i = 0; i < n; i += vl) {                                    \
      svbool_t pg = svwhilelt_b8((uint32_t)i, (uint32_t)n);                 \
      SVT va = LD(pg, a + i);                                               \
      SVT vb = LD(pg, b + i);                                               \
      svbool_t cmp = CMPGT(pg, va, vb);                                     \
      svst1_u8(pg, out + i, svsel_u8(cmp, svdup_u8(1), svdup_u8(0)));       \
    }                                                                       \
  }                                                                         \
  static inline void _fast_lt_##SFX##_sve(const void *restrict ap,          \
                                          const void *restrict bp,          \
                                          void *restrict op, size_t n) {    \
    const CT *a = (const CT *)ap;                                           \
    const CT *b = (const CT *)bp;                                           \
    uint8_t *out = (uint8_t *)op;                                           \
    size_t vl = svcntb();                                                   \
    for (size_t i = 0; i < n; i += vl) {                                    \
      svbool_t pg = svwhilelt_b8((uint32_t)i, (uint32_t)n);                 \
      SVT va = LD(pg, a + i);                                               \
      SVT vb = LD(pg, b + i);                                               \
      svbool_t cmp = CMPLT(pg, va, vb);                                     \
      svst1_u8(pg, out + i, svsel_u8(cmp, svdup_u8(1), svdup_u8(0)));       \
    }                                                                       \
  }                                                                         \
  static inline void _fast_ge_##SFX##_sve(const void *restrict ap,          \
                                          const void *restrict bp,          \
                                          void *restrict op, size_t n) {    \
    const CT *a = (const CT *)ap;                                           \
    const CT *b = (const CT *)bp;                                           \
    uint8_t *out = (uint8_t *)op;                                           \
    size_t vl = svcntb();                                                   \
    for (size_t i = 0; i < n; i += vl) {                                    \
      svbool_t pg = svwhilelt_b8((uint32_t)i, (uint32_t)n);                 \
      SVT va = LD(pg, a + i);                                               \
      SVT vb = LD(pg, b + i);                                               \
      svbool_t cmp = CMPGE(pg, va, vb);                                     \
      svst1_u8(pg, out + i, svsel_u8(cmp, svdup_u8(1), svdup_u8(0)));       \
    }                                                                       \
  }                                                                         \
  static inline void _fast_le_##SFX##_sve(const void *restrict ap,          \
                                          const void *restrict bp,          \
                                          void *restrict op, size_t n) {    \
    const CT *a = (const CT *)ap;                                           \
    const CT *b = (const CT *)bp;                                           \
    uint8_t *out = (uint8_t *)op;                                           \
    size_t vl = svcntb();                                                   \
    for (size_t i = 0; i < n; i += vl) {                                    \
      svbool_t pg = svwhilelt_b8((uint32_t)i, (uint32_t)n);                 \
      SVT va = LD(pg, a + i);                                               \
      SVT vb = LD(pg, b + i);                                               \
      svbool_t cmp = CMPLE(pg, va, vb);                                     \
      svst1_u8(pg, out + i, svsel_u8(cmp, svdup_u8(1), svdup_u8(0)));       \
    }                                                                       \
  }

/* clang-format off */
FAST_CMP_8_SVE(i8,  int8_t,  svint8_t,  svld1_s8,
               svcmpeq_s8,  svcmpgt_s8,  svcmplt_s8,
               svcmpge_s8,  svcmple_s8)
FAST_CMP_8_SVE(u8,  uint8_t, svuint8_t, svld1_u8,
               svcmpeq_u8,  svcmpgt_u8,  svcmplt_u8,
               svcmpge_u8,  svcmple_u8)
/* clang-format on */
#undef FAST_CMP_8_SVE

/* ====================================================================
 * Wider types (16/32/64): input predicate at native width,
 * byte output via scalar tail. Use CNT for input stride, write
 * one byte per element to output.
 * ================================================================ */

#define FAST_CMP_WIDE_SVE(SFX, CT, SVT, LD, WHILELT, CNT, CMPEQ, CMPGT, CMPLT, \
                          CMPGE, CMPLE)                                        \
  static inline void _fast_eq_##SFX##_sve(const void *restrict ap,             \
                                          const void *restrict bp,             \
                                          void *restrict op, size_t n) {       \
    const CT *a = (const CT *)ap;                                              \
    const CT *b = (const CT *)bp;                                              \
    uint8_t *out = (uint8_t *)op;                                              \
    size_t vl = CNT();                                                         \
    for (size_t i = 0; i < n; i += vl) {                                       \
      svbool_t pg = WHILELT((uint32_t)i, (uint32_t)n);                         \
      SVT va = LD(pg, a + i);                                                  \
      SVT vb = LD(pg, b + i);                                                  \
      svbool_t cmp = CMPEQ(pg, va, vb);                                        \
      svbool_t pg8 = svwhilelt_b8((uint32_t)i, (uint32_t)n);                   \
      svst1_u8(pg8, out + i, svsel_u8(cmp, svdup_u8(1), svdup_u8(0)));         \
    }                                                                          \
  }                                                                            \
  static inline void _fast_gt_##SFX##_sve(const void *restrict ap,             \
                                          const void *restrict bp,             \
                                          void *restrict op, size_t n) {       \
    const CT *a = (const CT *)ap;                                              \
    const CT *b = (const CT *)bp;                                              \
    uint8_t *out = (uint8_t *)op;                                              \
    size_t vl = CNT();                                                         \
    for (size_t i = 0; i < n; i += vl) {                                       \
      svbool_t pg = WHILELT((uint32_t)i, (uint32_t)n);                         \
      SVT va = LD(pg, a + i);                                                  \
      SVT vb = LD(pg, b + i);                                                  \
      svbool_t cmp = CMPGT(pg, va, vb);                                        \
      svbool_t pg8 = svwhilelt_b8((uint32_t)i, (uint32_t)n);                   \
      svst1_u8(pg8, out + i, svsel_u8(cmp, svdup_u8(1), svdup_u8(0)));         \
    }                                                                          \
  }                                                                            \
  static inline void _fast_lt_##SFX##_sve(const void *restrict ap,             \
                                          const void *restrict bp,             \
                                          void *restrict op, size_t n) {       \
    const CT *a = (const CT *)ap;                                              \
    const CT *b = (const CT *)bp;                                              \
    uint8_t *out = (uint8_t *)op;                                              \
    size_t vl = CNT();                                                         \
    for (size_t i = 0; i < n; i += vl) {                                       \
      svbool_t pg = WHILELT((uint32_t)i, (uint32_t)n);                         \
      SVT va = LD(pg, a + i);                                                  \
      SVT vb = LD(pg, b + i);                                                  \
      svbool_t cmp = CMPLT(pg, va, vb);                                        \
      svbool_t pg8 = svwhilelt_b8((uint32_t)i, (uint32_t)n);                   \
      svst1_u8(pg8, out + i, svsel_u8(cmp, svdup_u8(1), svdup_u8(0)));         \
    }                                                                          \
  }                                                                            \
  static inline void _fast_ge_##SFX##_sve(const void *restrict ap,             \
                                          const void *restrict bp,             \
                                          void *restrict op, size_t n) {       \
    const CT *a = (const CT *)ap;                                              \
    const CT *b = (const CT *)bp;                                              \
    uint8_t *out = (uint8_t *)op;                                              \
    size_t vl = CNT();                                                         \
    for (size_t i = 0; i < n; i += vl) {                                       \
      svbool_t pg = WHILELT((uint32_t)i, (uint32_t)n);                         \
      SVT va = LD(pg, a + i);                                                  \
      SVT vb = LD(pg, b + i);                                                  \
      svbool_t cmp = CMPGE(pg, va, vb);                                        \
      svbool_t pg8 = svwhilelt_b8((uint32_t)i, (uint32_t)n);                   \
      svst1_u8(pg8, out + i, svsel_u8(cmp, svdup_u8(1), svdup_u8(0)));         \
    }                                                                          \
  }                                                                            \
  static inline void _fast_le_##SFX##_sve(const void *restrict ap,             \
                                          const void *restrict bp,             \
                                          void *restrict op, size_t n) {       \
    const CT *a = (const CT *)ap;                                              \
    const CT *b = (const CT *)bp;                                              \
    uint8_t *out = (uint8_t *)op;                                              \
    size_t vl = CNT();                                                         \
    for (size_t i = 0; i < n; i += vl) {                                       \
      svbool_t pg = WHILELT((uint32_t)i, (uint32_t)n);                         \
      SVT va = LD(pg, a + i);                                                  \
      SVT vb = LD(pg, b + i);                                                  \
      svbool_t cmp = CMPLE(pg, va, vb);                                        \
      svbool_t pg8 = svwhilelt_b8((uint32_t)i, (uint32_t)n);                   \
      svst1_u8(pg8, out + i, svsel_u8(cmp, svdup_u8(1), svdup_u8(0)));         \
    }                                                                          \
  }

/* -- Signed integer instantiations -------------------------------- */

/* clang-format off */
FAST_CMP_WIDE_SVE(i16, int16_t, svint16_t, svld1_s16, svwhilelt_b16, svcnth,
                   svcmpeq_s16, svcmpgt_s16, svcmplt_s16,
                   svcmpge_s16, svcmple_s16)
FAST_CMP_WIDE_SVE(i32, int32_t, svint32_t, svld1_s32, svwhilelt_b32, svcntw,
                   svcmpeq_s32, svcmpgt_s32, svcmplt_s32,
                   svcmpge_s32, svcmple_s32)
FAST_CMP_WIDE_SVE(i64, int64_t, svint64_t, svld1_s64, svwhilelt_b64, svcntd,
                   svcmpeq_s64, svcmpgt_s64, svcmplt_s64,
                   svcmpge_s64, svcmple_s64)

/* -- Unsigned integer instantiations ------------------------------ */

FAST_CMP_WIDE_SVE(u16, uint16_t, svuint16_t, svld1_u16, svwhilelt_b16, svcnth,
                   svcmpeq_u16, svcmpgt_u16, svcmplt_u16,
                   svcmpge_u16, svcmple_u16)
FAST_CMP_WIDE_SVE(u32, uint32_t, svuint32_t, svld1_u32, svwhilelt_b32, svcntw,
                   svcmpeq_u32, svcmpgt_u32, svcmplt_u32,
                   svcmpge_u32, svcmple_u32)
FAST_CMP_WIDE_SVE(u64, uint64_t, svuint64_t, svld1_u64, svwhilelt_b64, svcntd,
                   svcmpeq_u64, svcmpgt_u64, svcmplt_u64,
                   svcmpge_u64, svcmple_u64)

/* -- Floating-point instantiations -------------------------------- */

FAST_CMP_WIDE_SVE(f32, float, svfloat32_t, svld1_f32, svwhilelt_b32, svcntw,
                   svcmpeq_f32, svcmpgt_f32, svcmplt_f32,
                   svcmpge_f32, svcmple_f32)
FAST_CMP_WIDE_SVE(f64, double, svfloat64_t, svld1_f64, svwhilelt_b64, svcntd,
                   svcmpeq_f64, svcmpgt_f64, svcmplt_f64,
                   svcmpge_f64, svcmple_f64)
/* clang-format on */

#undef FAST_CMP_WIDE_SVE

#endif /* NUMC_COMPARE_SVE_H */
