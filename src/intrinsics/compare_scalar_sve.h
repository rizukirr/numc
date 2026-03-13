/**
 * @file compare_scalar_sve.h
 * @brief SVE scalar comparison kernels for all 10 types.
 *
 * SVE uses predicated loops with svwhilelt for natural tail handling.
 * Scalar-immediate compare variants (svcmpeq_n_*) avoid broadcast.
 */
#ifndef NUMC_COMPARE_SCALAR_SVE_H
#define NUMC_COMPARE_SCALAR_SVE_H

#include <arm_sve.h>
#include <stddef.h>
#include <stdint.h>

/* ── Signed integer macro ───────────────────────────────────────── */

#define STAMP_CMPSC_SINT_SVE(SFX, CT, W, CNT)                               \
  static inline void _cmpsc_eq_##SFX##_sve(const void *restrict ap,         \
                                           const void *restrict sp,         \
                                           void *restrict op, size_t n) {   \
    const CT *a = (const CT *)ap;                                           \
    const CT s = *(const CT *)sp;                                           \
    CT *out = (CT *)op;                                                     \
    size_t vl = CNT();                                                      \
    for (size_t i = 0; i < n; i += vl) {                                    \
      svbool_t pg = svwhilelt_b##W((uint32_t)i, (uint32_t)n);               \
      svint##W##_t va = svld1_s##W(pg, a + i);                              \
      svbool_t p = svcmpeq_n_s##W(pg, va, s);                               \
      svst1_s##W(pg, out + i, svsel_s##W(p, svdup_s##W(1), svdup_s##W(0))); \
    }                                                                       \
  }                                                                         \
  static inline void _cmpsc_gt_##SFX##_sve(const void *restrict ap,         \
                                           const void *restrict sp,         \
                                           void *restrict op, size_t n) {   \
    const CT *a = (const CT *)ap;                                           \
    const CT s = *(const CT *)sp;                                           \
    CT *out = (CT *)op;                                                     \
    size_t vl = CNT();                                                      \
    for (size_t i = 0; i < n; i += vl) {                                    \
      svbool_t pg = svwhilelt_b##W((uint32_t)i, (uint32_t)n);               \
      svint##W##_t va = svld1_s##W(pg, a + i);                              \
      svbool_t p = svcmpgt_n_s##W(pg, va, s);                               \
      svst1_s##W(pg, out + i, svsel_s##W(p, svdup_s##W(1), svdup_s##W(0))); \
    }                                                                       \
  }                                                                         \
  static inline void _cmpsc_lt_##SFX##_sve(const void *restrict ap,         \
                                           const void *restrict sp,         \
                                           void *restrict op, size_t n) {   \
    const CT *a = (const CT *)ap;                                           \
    const CT s = *(const CT *)sp;                                           \
    CT *out = (CT *)op;                                                     \
    size_t vl = CNT();                                                      \
    for (size_t i = 0; i < n; i += vl) {                                    \
      svbool_t pg = svwhilelt_b##W((uint32_t)i, (uint32_t)n);               \
      svint##W##_t va = svld1_s##W(pg, a + i);                              \
      svbool_t p = svcmplt_n_s##W(pg, va, s);                               \
      svst1_s##W(pg, out + i, svsel_s##W(p, svdup_s##W(1), svdup_s##W(0))); \
    }                                                                       \
  }                                                                         \
  static inline void _cmpsc_ge_##SFX##_sve(const void *restrict ap,         \
                                           const void *restrict sp,         \
                                           void *restrict op, size_t n) {   \
    const CT *a = (const CT *)ap;                                           \
    const CT s = *(const CT *)sp;                                           \
    CT *out = (CT *)op;                                                     \
    size_t vl = CNT();                                                      \
    for (size_t i = 0; i < n; i += vl) {                                    \
      svbool_t pg = svwhilelt_b##W((uint32_t)i, (uint32_t)n);               \
      svint##W##_t va = svld1_s##W(pg, a + i);                              \
      svbool_t p = svcmpge_n_s##W(pg, va, s);                               \
      svst1_s##W(pg, out + i, svsel_s##W(p, svdup_s##W(1), svdup_s##W(0))); \
    }                                                                       \
  }                                                                         \
  static inline void _cmpsc_le_##SFX##_sve(const void *restrict ap,         \
                                           const void *restrict sp,         \
                                           void *restrict op, size_t n) {   \
    const CT *a = (const CT *)ap;                                           \
    const CT s = *(const CT *)sp;                                           \
    CT *out = (CT *)op;                                                     \
    size_t vl = CNT();                                                      \
    for (size_t i = 0; i < n; i += vl) {                                    \
      svbool_t pg = svwhilelt_b##W((uint32_t)i, (uint32_t)n);               \
      svint##W##_t va = svld1_s##W(pg, a + i);                              \
      svbool_t p = svcmple_n_s##W(pg, va, s);                               \
      svst1_s##W(pg, out + i, svsel_s##W(p, svdup_s##W(1), svdup_s##W(0))); \
    }                                                                       \
  }

STAMP_CMPSC_SINT_SVE(i8, int8_t, 8, svcntb)
STAMP_CMPSC_SINT_SVE(i16, int16_t, 16, svcnth)
STAMP_CMPSC_SINT_SVE(i32, int32_t, 32, svcntw)
STAMP_CMPSC_SINT_SVE(i64, int64_t, 64, svcntd)
#undef STAMP_CMPSC_SINT_SVE

/* ── Unsigned integer macro ─────────────────────────────────────── */

#define STAMP_CMPSC_UINT_SVE(SFX, CT, W, CNT)                               \
  static inline void _cmpsc_eq_##SFX##_sve(const void *restrict ap,         \
                                           const void *restrict sp,         \
                                           void *restrict op, size_t n) {   \
    const CT *a = (const CT *)ap;                                           \
    const CT s = *(const CT *)sp;                                           \
    CT *out = (CT *)op;                                                     \
    size_t vl = CNT();                                                      \
    for (size_t i = 0; i < n; i += vl) {                                    \
      svbool_t pg = svwhilelt_b##W((uint32_t)i, (uint32_t)n);               \
      svuint##W##_t va = svld1_u##W(pg, a + i);                             \
      svbool_t p = svcmpeq_n_u##W(pg, va, s);                               \
      svst1_u##W(pg, out + i, svsel_u##W(p, svdup_u##W(1), svdup_u##W(0))); \
    }                                                                       \
  }                                                                         \
  static inline void _cmpsc_gt_##SFX##_sve(const void *restrict ap,         \
                                           const void *restrict sp,         \
                                           void *restrict op, size_t n) {   \
    const CT *a = (const CT *)ap;                                           \
    const CT s = *(const CT *)sp;                                           \
    CT *out = (CT *)op;                                                     \
    size_t vl = CNT();                                                      \
    for (size_t i = 0; i < n; i += vl) {                                    \
      svbool_t pg = svwhilelt_b##W((uint32_t)i, (uint32_t)n);               \
      svuint##W##_t va = svld1_u##W(pg, a + i);                             \
      svbool_t p = svcmpgt_n_u##W(pg, va, s);                               \
      svst1_u##W(pg, out + i, svsel_u##W(p, svdup_u##W(1), svdup_u##W(0))); \
    }                                                                       \
  }                                                                         \
  static inline void _cmpsc_lt_##SFX##_sve(const void *restrict ap,         \
                                           const void *restrict sp,         \
                                           void *restrict op, size_t n) {   \
    const CT *a = (const CT *)ap;                                           \
    const CT s = *(const CT *)sp;                                           \
    CT *out = (CT *)op;                                                     \
    size_t vl = CNT();                                                      \
    for (size_t i = 0; i < n; i += vl) {                                    \
      svbool_t pg = svwhilelt_b##W((uint32_t)i, (uint32_t)n);               \
      svuint##W##_t va = svld1_u##W(pg, a + i);                             \
      svbool_t p = svcmplt_n_u##W(pg, va, s);                               \
      svst1_u##W(pg, out + i, svsel_u##W(p, svdup_u##W(1), svdup_u##W(0))); \
    }                                                                       \
  }                                                                         \
  static inline void _cmpsc_ge_##SFX##_sve(const void *restrict ap,         \
                                           const void *restrict sp,         \
                                           void *restrict op, size_t n) {   \
    const CT *a = (const CT *)ap;                                           \
    const CT s = *(const CT *)sp;                                           \
    CT *out = (CT *)op;                                                     \
    size_t vl = CNT();                                                      \
    for (size_t i = 0; i < n; i += vl) {                                    \
      svbool_t pg = svwhilelt_b##W((uint32_t)i, (uint32_t)n);               \
      svuint##W##_t va = svld1_u##W(pg, a + i);                             \
      svbool_t p = svcmpge_n_u##W(pg, va, s);                               \
      svst1_u##W(pg, out + i, svsel_u##W(p, svdup_u##W(1), svdup_u##W(0))); \
    }                                                                       \
  }                                                                         \
  static inline void _cmpsc_le_##SFX##_sve(const void *restrict ap,         \
                                           const void *restrict sp,         \
                                           void *restrict op, size_t n) {   \
    const CT *a = (const CT *)ap;                                           \
    const CT s = *(const CT *)sp;                                           \
    CT *out = (CT *)op;                                                     \
    size_t vl = CNT();                                                      \
    for (size_t i = 0; i < n; i += vl) {                                    \
      svbool_t pg = svwhilelt_b##W((uint32_t)i, (uint32_t)n);               \
      svuint##W##_t va = svld1_u##W(pg, a + i);                             \
      svbool_t p = svcmple_n_u##W(pg, va, s);                               \
      svst1_u##W(pg, out + i, svsel_u##W(p, svdup_u##W(1), svdup_u##W(0))); \
    }                                                                       \
  }

STAMP_CMPSC_UINT_SVE(u8, uint8_t, 8, svcntb)
STAMP_CMPSC_UINT_SVE(u16, uint16_t, 16, svcnth)
STAMP_CMPSC_UINT_SVE(u32, uint32_t, 32, svcntw)
STAMP_CMPSC_UINT_SVE(u64, uint64_t, 64, svcntd)
#undef STAMP_CMPSC_UINT_SVE

/* ── Float macro ────────────────────────────────────────────────── */

#define STAMP_CMPSC_FLOAT_SVE(SFX, CT, W, CNT)                            \
  static inline void _cmpsc_eq_##SFX##_sve(const void *restrict ap,       \
                                           const void *restrict sp,       \
                                           void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                         \
    const CT s = *(const CT *)sp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl = CNT();                                                    \
    for (size_t i = 0; i < n; i += vl) {                                  \
      svbool_t pg = svwhilelt_b##W((uint32_t)i, (uint32_t)n);             \
      svfloat##W##_t va = svld1_f##W(pg, a + i);                          \
      svbool_t p = svcmpeq_n_f##W(pg, va, s);                             \
      svst1_f##W(pg, out + i,                                             \
                 svsel_f##W(p, svdup_f##W((CT)1), svdup_f##W((CT)0)));    \
    }                                                                     \
  }                                                                       \
  static inline void _cmpsc_gt_##SFX##_sve(const void *restrict ap,       \
                                           const void *restrict sp,       \
                                           void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                         \
    const CT s = *(const CT *)sp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl = CNT();                                                    \
    for (size_t i = 0; i < n; i += vl) {                                  \
      svbool_t pg = svwhilelt_b##W((uint32_t)i, (uint32_t)n);             \
      svfloat##W##_t va = svld1_f##W(pg, a + i);                          \
      svbool_t p = svcmpgt_n_f##W(pg, va, s);                             \
      svst1_f##W(pg, out + i,                                             \
                 svsel_f##W(p, svdup_f##W((CT)1), svdup_f##W((CT)0)));    \
    }                                                                     \
  }                                                                       \
  static inline void _cmpsc_lt_##SFX##_sve(const void *restrict ap,       \
                                           const void *restrict sp,       \
                                           void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                         \
    const CT s = *(const CT *)sp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl = CNT();                                                    \
    for (size_t i = 0; i < n; i += vl) {                                  \
      svbool_t pg = svwhilelt_b##W((uint32_t)i, (uint32_t)n);             \
      svfloat##W##_t va = svld1_f##W(pg, a + i);                          \
      svbool_t p = svcmplt_n_f##W(pg, va, s);                             \
      svst1_f##W(pg, out + i,                                             \
                 svsel_f##W(p, svdup_f##W((CT)1), svdup_f##W((CT)0)));    \
    }                                                                     \
  }                                                                       \
  static inline void _cmpsc_ge_##SFX##_sve(const void *restrict ap,       \
                                           const void *restrict sp,       \
                                           void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                         \
    const CT s = *(const CT *)sp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl = CNT();                                                    \
    for (size_t i = 0; i < n; i += vl) {                                  \
      svbool_t pg = svwhilelt_b##W((uint32_t)i, (uint32_t)n);             \
      svfloat##W##_t va = svld1_f##W(pg, a + i);                          \
      svbool_t p = svcmpge_n_f##W(pg, va, s);                             \
      svst1_f##W(pg, out + i,                                             \
                 svsel_f##W(p, svdup_f##W((CT)1), svdup_f##W((CT)0)));    \
    }                                                                     \
  }                                                                       \
  static inline void _cmpsc_le_##SFX##_sve(const void *restrict ap,       \
                                           const void *restrict sp,       \
                                           void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                         \
    const CT s = *(const CT *)sp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl = CNT();                                                    \
    for (size_t i = 0; i < n; i += vl) {                                  \
      svbool_t pg = svwhilelt_b##W((uint32_t)i, (uint32_t)n);             \
      svfloat##W##_t va = svld1_f##W(pg, a + i);                          \
      svbool_t p = svcmple_n_f##W(pg, va, s);                             \
      svst1_f##W(pg, out + i,                                             \
                 svsel_f##W(p, svdup_f##W((CT)1), svdup_f##W((CT)0)));    \
    }                                                                     \
  }

STAMP_CMPSC_FLOAT_SVE(f32, float, 32, svcntw)
STAMP_CMPSC_FLOAT_SVE(f64, double, 64, svcntd)
#undef STAMP_CMPSC_FLOAT_SVE

#endif /* NUMC_COMPARE_SCALAR_SVE_H */
