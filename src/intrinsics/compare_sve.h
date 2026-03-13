/**
 * @file compare_sve.h
 * @brief SVE binary comparison kernels for all 10 types x 5 ops.
 *
 * Produces 0 or 1 (same type as input) per element.
 * SVE predicated comparisons return svbool_t; convert to 0/1 via svsel.
 * Tail handling uses svwhilelt — no scalar cleanup needed.
 */
#ifndef NUMC_COMPARE_SVE_H
#define NUMC_COMPARE_SVE_H

#include <arm_sve.h>
#include <stddef.h>
#include <stdint.h>

/* ════════════════════════════════════════════════════════════════════
 * Signed integer comparisons
 * ════════════════════════════════════════════════════════════════ */

#define FAST_CMP_SINT_SVE(SFX, CT, BITS, SVT, LD, ST, SEL, DUP,      \
                          CMPEQ, CMPGT, CMPLT, CMPGE, CMPLE, WHILELT, \
                          CNT)                                          \
  static inline void _fast_eq_##SFX##_sve(                             \
      const void *restrict ap, const void *restrict bp,                \
      void *restrict op, size_t n) {                                   \
    const CT *a = (const CT *)ap;                                      \
    const CT *b = (const CT *)bp;                                      \
    CT *out = (CT *)op;                                                \
    size_t vl = CNT();                                                 \
    for (size_t i = 0; i < n; i += vl) {                               \
      svbool_t pg = WHILELT((uint32_t)i, (uint32_t)n);                \
      SVT va = LD(pg, a + i);                                         \
      SVT vb = LD(pg, b + i);                                         \
      svbool_t cmp = CMPEQ(pg, va, vb);                               \
      ST(pg, out + i, SEL(cmp, DUP(1), DUP(0)));                      \
    }                                                                  \
  }                                                                    \
  static inline void _fast_gt_##SFX##_sve(                             \
      const void *restrict ap, const void *restrict bp,                \
      void *restrict op, size_t n) {                                   \
    const CT *a = (const CT *)ap;                                      \
    const CT *b = (const CT *)bp;                                      \
    CT *out = (CT *)op;                                                \
    size_t vl = CNT();                                                 \
    for (size_t i = 0; i < n; i += vl) {                               \
      svbool_t pg = WHILELT((uint32_t)i, (uint32_t)n);                \
      SVT va = LD(pg, a + i);                                         \
      SVT vb = LD(pg, b + i);                                         \
      svbool_t cmp = CMPGT(pg, va, vb);                               \
      ST(pg, out + i, SEL(cmp, DUP(1), DUP(0)));                      \
    }                                                                  \
  }                                                                    \
  static inline void _fast_lt_##SFX##_sve(                             \
      const void *restrict ap, const void *restrict bp,                \
      void *restrict op, size_t n) {                                   \
    const CT *a = (const CT *)ap;                                      \
    const CT *b = (const CT *)bp;                                      \
    CT *out = (CT *)op;                                                \
    size_t vl = CNT();                                                 \
    for (size_t i = 0; i < n; i += vl) {                               \
      svbool_t pg = WHILELT((uint32_t)i, (uint32_t)n);                \
      SVT va = LD(pg, a + i);                                         \
      SVT vb = LD(pg, b + i);                                         \
      svbool_t cmp = CMPLT(pg, va, vb);                               \
      ST(pg, out + i, SEL(cmp, DUP(1), DUP(0)));                      \
    }                                                                  \
  }                                                                    \
  static inline void _fast_ge_##SFX##_sve(                             \
      const void *restrict ap, const void *restrict bp,                \
      void *restrict op, size_t n) {                                   \
    const CT *a = (const CT *)ap;                                      \
    const CT *b = (const CT *)bp;                                      \
    CT *out = (CT *)op;                                                \
    size_t vl = CNT();                                                 \
    for (size_t i = 0; i < n; i += vl) {                               \
      svbool_t pg = WHILELT((uint32_t)i, (uint32_t)n);                \
      SVT va = LD(pg, a + i);                                         \
      SVT vb = LD(pg, b + i);                                         \
      svbool_t cmp = CMPGE(pg, va, vb);                               \
      ST(pg, out + i, SEL(cmp, DUP(1), DUP(0)));                      \
    }                                                                  \
  }                                                                    \
  static inline void _fast_le_##SFX##_sve(                             \
      const void *restrict ap, const void *restrict bp,                \
      void *restrict op, size_t n) {                                   \
    const CT *a = (const CT *)ap;                                      \
    const CT *b = (const CT *)bp;                                      \
    CT *out = (CT *)op;                                                \
    size_t vl = CNT();                                                 \
    for (size_t i = 0; i < n; i += vl) {                               \
      svbool_t pg = WHILELT((uint32_t)i, (uint32_t)n);                \
      SVT va = LD(pg, a + i);                                         \
      SVT vb = LD(pg, b + i);                                         \
      svbool_t cmp = CMPLE(pg, va, vb);                               \
      ST(pg, out + i, SEL(cmp, DUP(1), DUP(0)));                      \
    }                                                                  \
  }

/* clang-format off */
FAST_CMP_SINT_SVE(i8,  int8_t,  8,  svint8_t,  svld1_s8,  svst1_s8,
                   svsel_s8,  svdup_s8,
                   svcmpeq_s8,  svcmpgt_s8,  svcmplt_s8,
                   svcmpge_s8,  svcmple_s8,
                   svwhilelt_b8,  svcntb)
FAST_CMP_SINT_SVE(i16, int16_t, 16, svint16_t, svld1_s16, svst1_s16,
                   svsel_s16, svdup_s16,
                   svcmpeq_s16, svcmpgt_s16, svcmplt_s16,
                   svcmpge_s16, svcmple_s16,
                   svwhilelt_b16, svcnth)
FAST_CMP_SINT_SVE(i32, int32_t, 32, svint32_t, svld1_s32, svst1_s32,
                   svsel_s32, svdup_s32,
                   svcmpeq_s32, svcmpgt_s32, svcmplt_s32,
                   svcmpge_s32, svcmple_s32,
                   svwhilelt_b32, svcntw)
FAST_CMP_SINT_SVE(i64, int64_t, 64, svint64_t, svld1_s64, svst1_s64,
                   svsel_s64, svdup_s64,
                   svcmpeq_s64, svcmpgt_s64, svcmplt_s64,
                   svcmpge_s64, svcmple_s64,
                   svwhilelt_b64, svcntd)
/* clang-format on */

#undef FAST_CMP_SINT_SVE

/* ════════════════════════════════════════════════════════════════════
 * Unsigned integer comparisons
 * ════════════════════════════════════════════════════════════════ */

#define FAST_CMP_UINT_SVE(SFX, CT, BITS, SVT, LD, ST, SEL, DUP,      \
                          CMPEQ, CMPGT, CMPLT, CMPGE, CMPLE, WHILELT, \
                          CNT)                                          \
  static inline void _fast_eq_##SFX##_sve(                             \
      const void *restrict ap, const void *restrict bp,                \
      void *restrict op, size_t n) {                                   \
    const CT *a = (const CT *)ap;                                      \
    const CT *b = (const CT *)bp;                                      \
    CT *out = (CT *)op;                                                \
    size_t vl = CNT();                                                 \
    for (size_t i = 0; i < n; i += vl) {                               \
      svbool_t pg = WHILELT((uint32_t)i, (uint32_t)n);                \
      SVT va = LD(pg, a + i);                                         \
      SVT vb = LD(pg, b + i);                                         \
      svbool_t cmp = CMPEQ(pg, va, vb);                               \
      ST(pg, out + i, SEL(cmp, DUP(1), DUP(0)));                      \
    }                                                                  \
  }                                                                    \
  static inline void _fast_gt_##SFX##_sve(                             \
      const void *restrict ap, const void *restrict bp,                \
      void *restrict op, size_t n) {                                   \
    const CT *a = (const CT *)ap;                                      \
    const CT *b = (const CT *)bp;                                      \
    CT *out = (CT *)op;                                                \
    size_t vl = CNT();                                                 \
    for (size_t i = 0; i < n; i += vl) {                               \
      svbool_t pg = WHILELT((uint32_t)i, (uint32_t)n);                \
      SVT va = LD(pg, a + i);                                         \
      SVT vb = LD(pg, b + i);                                         \
      svbool_t cmp = CMPGT(pg, va, vb);                               \
      ST(pg, out + i, SEL(cmp, DUP(1), DUP(0)));                      \
    }                                                                  \
  }                                                                    \
  static inline void _fast_lt_##SFX##_sve(                             \
      const void *restrict ap, const void *restrict bp,                \
      void *restrict op, size_t n) {                                   \
    const CT *a = (const CT *)ap;                                      \
    const CT *b = (const CT *)bp;                                      \
    CT *out = (CT *)op;                                                \
    size_t vl = CNT();                                                 \
    for (size_t i = 0; i < n; i += vl) {                               \
      svbool_t pg = WHILELT((uint32_t)i, (uint32_t)n);                \
      SVT va = LD(pg, a + i);                                         \
      SVT vb = LD(pg, b + i);                                         \
      svbool_t cmp = CMPLT(pg, va, vb);                               \
      ST(pg, out + i, SEL(cmp, DUP(1), DUP(0)));                      \
    }                                                                  \
  }                                                                    \
  static inline void _fast_ge_##SFX##_sve(                             \
      const void *restrict ap, const void *restrict bp,                \
      void *restrict op, size_t n) {                                   \
    const CT *a = (const CT *)ap;                                      \
    const CT *b = (const CT *)bp;                                      \
    CT *out = (CT *)op;                                                \
    size_t vl = CNT();                                                 \
    for (size_t i = 0; i < n; i += vl) {                               \
      svbool_t pg = WHILELT((uint32_t)i, (uint32_t)n);                \
      SVT va = LD(pg, a + i);                                         \
      SVT vb = LD(pg, b + i);                                         \
      svbool_t cmp = CMPGE(pg, va, vb);                               \
      ST(pg, out + i, SEL(cmp, DUP(1), DUP(0)));                      \
    }                                                                  \
  }                                                                    \
  static inline void _fast_le_##SFX##_sve(                             \
      const void *restrict ap, const void *restrict bp,                \
      void *restrict op, size_t n) {                                   \
    const CT *a = (const CT *)ap;                                      \
    const CT *b = (const CT *)bp;                                      \
    CT *out = (CT *)op;                                                \
    size_t vl = CNT();                                                 \
    for (size_t i = 0; i < n; i += vl) {                               \
      svbool_t pg = WHILELT((uint32_t)i, (uint32_t)n);                \
      SVT va = LD(pg, a + i);                                         \
      SVT vb = LD(pg, b + i);                                         \
      svbool_t cmp = CMPLE(pg, va, vb);                               \
      ST(pg, out + i, SEL(cmp, DUP(1), DUP(0)));                      \
    }                                                                  \
  }

/* clang-format off */
FAST_CMP_UINT_SVE(u8,  uint8_t,  8,  svuint8_t,  svld1_u8,  svst1_u8,
                   svsel_u8,  svdup_u8,
                   svcmpeq_u8,  svcmpgt_u8,  svcmplt_u8,
                   svcmpge_u8,  svcmple_u8,
                   svwhilelt_b8,  svcntb)
FAST_CMP_UINT_SVE(u16, uint16_t, 16, svuint16_t, svld1_u16, svst1_u16,
                   svsel_u16, svdup_u16,
                   svcmpeq_u16, svcmpgt_u16, svcmplt_u16,
                   svcmpge_u16, svcmple_u16,
                   svwhilelt_b16, svcnth)
FAST_CMP_UINT_SVE(u32, uint32_t, 32, svuint32_t, svld1_u32, svst1_u32,
                   svsel_u32, svdup_u32,
                   svcmpeq_u32, svcmpgt_u32, svcmplt_u32,
                   svcmpge_u32, svcmple_u32,
                   svwhilelt_b32, svcntw)
FAST_CMP_UINT_SVE(u64, uint64_t, 64, svuint64_t, svld1_u64, svst1_u64,
                   svsel_u64, svdup_u64,
                   svcmpeq_u64, svcmpgt_u64, svcmplt_u64,
                   svcmpge_u64, svcmple_u64,
                   svwhilelt_b64, svcntd)
/* clang-format on */

#undef FAST_CMP_UINT_SVE

/* ════════════════════════════════════════════════════════════════════
 * Floating-point comparisons
 * ════════════════════════════════════════════════════════════════ */

#define FAST_CMP_FLOAT_SVE(SFX, CT, BITS, SVT, LD, ST, SEL, DUP,     \
                           CMPEQ, CMPGT, CMPLT, CMPGE, CMPLE,         \
                           WHILELT, CNT)                                \
  static inline void _fast_eq_##SFX##_sve(                             \
      const void *restrict ap, const void *restrict bp,                \
      void *restrict op, size_t n) {                                   \
    const CT *a = (const CT *)ap;                                      \
    const CT *b = (const CT *)bp;                                      \
    CT *out = (CT *)op;                                                \
    size_t vl = CNT();                                                 \
    for (size_t i = 0; i < n; i += vl) {                               \
      svbool_t pg = WHILELT((uint32_t)i, (uint32_t)n);                \
      SVT va = LD(pg, a + i);                                         \
      SVT vb = LD(pg, b + i);                                         \
      svbool_t cmp = CMPEQ(pg, va, vb);                               \
      ST(pg, out + i, SEL(cmp, DUP((CT)1), DUP((CT)0)));              \
    }                                                                  \
  }                                                                    \
  static inline void _fast_gt_##SFX##_sve(                             \
      const void *restrict ap, const void *restrict bp,                \
      void *restrict op, size_t n) {                                   \
    const CT *a = (const CT *)ap;                                      \
    const CT *b = (const CT *)bp;                                      \
    CT *out = (CT *)op;                                                \
    size_t vl = CNT();                                                 \
    for (size_t i = 0; i < n; i += vl) {                               \
      svbool_t pg = WHILELT((uint32_t)i, (uint32_t)n);                \
      SVT va = LD(pg, a + i);                                         \
      SVT vb = LD(pg, b + i);                                         \
      svbool_t cmp = CMPGT(pg, va, vb);                               \
      ST(pg, out + i, SEL(cmp, DUP((CT)1), DUP((CT)0)));              \
    }                                                                  \
  }                                                                    \
  static inline void _fast_lt_##SFX##_sve(                             \
      const void *restrict ap, const void *restrict bp,                \
      void *restrict op, size_t n) {                                   \
    const CT *a = (const CT *)ap;                                      \
    const CT *b = (const CT *)bp;                                      \
    CT *out = (CT *)op;                                                \
    size_t vl = CNT();                                                 \
    for (size_t i = 0; i < n; i += vl) {                               \
      svbool_t pg = WHILELT((uint32_t)i, (uint32_t)n);                \
      SVT va = LD(pg, a + i);                                         \
      SVT vb = LD(pg, b + i);                                         \
      svbool_t cmp = CMPLT(pg, va, vb);                               \
      ST(pg, out + i, SEL(cmp, DUP((CT)1), DUP((CT)0)));              \
    }                                                                  \
  }                                                                    \
  static inline void _fast_ge_##SFX##_sve(                             \
      const void *restrict ap, const void *restrict bp,                \
      void *restrict op, size_t n) {                                   \
    const CT *a = (const CT *)ap;                                      \
    const CT *b = (const CT *)bp;                                      \
    CT *out = (CT *)op;                                                \
    size_t vl = CNT();                                                 \
    for (size_t i = 0; i < n; i += vl) {                               \
      svbool_t pg = WHILELT((uint32_t)i, (uint32_t)n);                \
      SVT va = LD(pg, a + i);                                         \
      SVT vb = LD(pg, b + i);                                         \
      svbool_t cmp = CMPGE(pg, va, vb);                               \
      ST(pg, out + i, SEL(cmp, DUP((CT)1), DUP((CT)0)));              \
    }                                                                  \
  }                                                                    \
  static inline void _fast_le_##SFX##_sve(                             \
      const void *restrict ap, const void *restrict bp,                \
      void *restrict op, size_t n) {                                   \
    const CT *a = (const CT *)ap;                                      \
    const CT *b = (const CT *)bp;                                      \
    CT *out = (CT *)op;                                                \
    size_t vl = CNT();                                                 \
    for (size_t i = 0; i < n; i += vl) {                               \
      svbool_t pg = WHILELT((uint32_t)i, (uint32_t)n);                \
      SVT va = LD(pg, a + i);                                         \
      SVT vb = LD(pg, b + i);                                         \
      svbool_t cmp = CMPLE(pg, va, vb);                               \
      ST(pg, out + i, SEL(cmp, DUP((CT)1), DUP((CT)0)));              \
    }                                                                  \
  }

/* clang-format off */
FAST_CMP_FLOAT_SVE(f32, float,  32, svfloat32_t, svld1_f32, svst1_f32,
                    svsel_f32, svdup_f32,
                    svcmpeq_f32, svcmpgt_f32, svcmplt_f32,
                    svcmpge_f32, svcmple_f32,
                    svwhilelt_b32, svcntw)
FAST_CMP_FLOAT_SVE(f64, double, 64, svfloat64_t, svld1_f64, svst1_f64,
                    svsel_f64, svdup_f64,
                    svcmpeq_f64, svcmpgt_f64, svcmplt_f64,
                    svcmpge_f64, svcmple_f64,
                    svwhilelt_b64, svcntd)
/* clang-format on */

#undef FAST_CMP_FLOAT_SVE

#endif /* NUMC_COMPARE_SVE_H */
