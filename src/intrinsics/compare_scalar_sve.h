/**
 * @file compare_scalar_sve.h
 * @brief SVE scalar comparison kernels — uint8 output (0/1).
 *
 * All comparison functions output uint8_t* (NumPy-compatible bool).
 * SVE uses predicated loops with svwhilelt for natural tail handling.
 * Scalar-immediate compare variants (svcmpeq_n_*) avoid broadcast.
 */
#ifndef NUMC_COMPARE_SCALAR_SVE_H
#define NUMC_COMPARE_SCALAR_SVE_H

#include <arm_sve.h>
#include <stddef.h>
#include <stdint.h>

/* ── 8-bit signed integer: natural byte output ──────────────────── */

#define STAMP_CMPSC_I8_SVE(FNAME, CMP_N)                                    \
  static inline void _cmpsc_##FNAME##_i8_sve(const void *restrict ap,       \
                                             const void *restrict sp,       \
                                             void *restrict op, size_t n) { \
    const int8_t *a = (const int8_t *)ap;                                   \
    const int8_t s = *(const int8_t *)sp;                                   \
    uint8_t *out = (uint8_t *)op;                                           \
    size_t vl = svcntb();                                                   \
    for (size_t i = 0; i < n; i += vl) {                                    \
      svbool_t pg = svwhilelt_b8((uint32_t)i, (uint32_t)n);                 \
      svint8_t va = svld1_s8(pg, a + i);                                    \
      svbool_t p = CMP_N(pg, va, s);                                        \
      svst1_u8(pg, out + i, svsel_u8(p, svdup_u8(1), svdup_u8(0)));         \
    }                                                                       \
  }

STAMP_CMPSC_I8_SVE(eq, svcmpeq_n_s8)
STAMP_CMPSC_I8_SVE(gt, svcmpgt_n_s8)
STAMP_CMPSC_I8_SVE(lt, svcmplt_n_s8)
STAMP_CMPSC_I8_SVE(ge, svcmpge_n_s8)
STAMP_CMPSC_I8_SVE(le, svcmple_n_s8)
#undef STAMP_CMPSC_I8_SVE

/* ── 8-bit unsigned integer: natural byte output ────────────────── */

#define STAMP_CMPSC_U8_SVE(FNAME, CMP_N)                                    \
  static inline void _cmpsc_##FNAME##_u8_sve(const void *restrict ap,       \
                                             const void *restrict sp,       \
                                             void *restrict op, size_t n) { \
    const uint8_t *a = (const uint8_t *)ap;                                 \
    const uint8_t s = *(const uint8_t *)sp;                                 \
    uint8_t *out = (uint8_t *)op;                                           \
    size_t vl = svcntb();                                                   \
    for (size_t i = 0; i < n; i += vl) {                                    \
      svbool_t pg = svwhilelt_b8((uint32_t)i, (uint32_t)n);                 \
      svuint8_t va = svld1_u8(pg, a + i);                                   \
      svbool_t p = CMP_N(pg, va, s);                                        \
      svst1_u8(pg, out + i, svsel_u8(p, svdup_u8(1), svdup_u8(0)));         \
    }                                                                       \
  }

STAMP_CMPSC_U8_SVE(eq, svcmpeq_n_u8)
STAMP_CMPSC_U8_SVE(gt, svcmpgt_n_u8)
STAMP_CMPSC_U8_SVE(lt, svcmplt_n_u8)
STAMP_CMPSC_U8_SVE(ge, svcmpge_n_u8)
STAMP_CMPSC_U8_SVE(le, svcmple_n_u8)
#undef STAMP_CMPSC_U8_SVE

/* ── Wider signed integer (16/32/64): byte output ───────────────── */

#define STAMP_CMPSC_SINT_WIDE_SVE(SFX, CT, W, CNT, CMP_N)              \
  static inline void _cmpsc_##SFX##_sve(const void *restrict ap,       \
                                        const void *restrict sp,       \
                                        void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                      \
    const CT s = *(const CT *)sp;                                      \
    uint8_t *out = (uint8_t *)op;                                      \
    size_t vl = CNT();                                                 \
    for (size_t i = 0; i < n; i += vl) {                               \
      svbool_t pg = svwhilelt_b##W((uint32_t)i, (uint32_t)n);          \
      svint##W##_t va = svld1_s##W(pg, a + i);                         \
      svbool_t p = CMP_N(pg, va, s);                                   \
      svbool_t pg8 = svwhilelt_b8((uint32_t)i, (uint32_t)n);           \
      svst1_u8(pg8, out + i, svsel_u8(p, svdup_u8(1), svdup_u8(0)));   \
    }                                                                  \
  }

/* clang-format off */
STAMP_CMPSC_SINT_WIDE_SVE(eq_i16, int16_t, 16, svcnth, svcmpeq_n_s16)
STAMP_CMPSC_SINT_WIDE_SVE(gt_i16, int16_t, 16, svcnth, svcmpgt_n_s16)
STAMP_CMPSC_SINT_WIDE_SVE(lt_i16, int16_t, 16, svcnth, svcmplt_n_s16)
STAMP_CMPSC_SINT_WIDE_SVE(ge_i16, int16_t, 16, svcnth, svcmpge_n_s16)
STAMP_CMPSC_SINT_WIDE_SVE(le_i16, int16_t, 16, svcnth, svcmple_n_s16)

STAMP_CMPSC_SINT_WIDE_SVE(eq_i32, int32_t, 32, svcntw, svcmpeq_n_s32)
STAMP_CMPSC_SINT_WIDE_SVE(gt_i32, int32_t, 32, svcntw, svcmpgt_n_s32)
STAMP_CMPSC_SINT_WIDE_SVE(lt_i32, int32_t, 32, svcntw, svcmplt_n_s32)
STAMP_CMPSC_SINT_WIDE_SVE(ge_i32, int32_t, 32, svcntw, svcmpge_n_s32)
STAMP_CMPSC_SINT_WIDE_SVE(le_i32, int32_t, 32, svcntw, svcmple_n_s32)

STAMP_CMPSC_SINT_WIDE_SVE(eq_i64, int64_t, 64, svcntd, svcmpeq_n_s64)
STAMP_CMPSC_SINT_WIDE_SVE(gt_i64, int64_t, 64, svcntd, svcmpgt_n_s64)
STAMP_CMPSC_SINT_WIDE_SVE(lt_i64, int64_t, 64, svcntd, svcmplt_n_s64)
STAMP_CMPSC_SINT_WIDE_SVE(ge_i64, int64_t, 64, svcntd, svcmpge_n_s64)
STAMP_CMPSC_SINT_WIDE_SVE(le_i64, int64_t, 64, svcntd, svcmple_n_s64)
/* clang-format on */
#undef STAMP_CMPSC_SINT_WIDE_SVE

/* ── Wider unsigned integer (16/32/64): byte output ─────────────── */

#define STAMP_CMPSC_UINT_WIDE_SVE(SFX, CT, W, CNT, CMP_N)              \
  static inline void _cmpsc_##SFX##_sve(const void *restrict ap,       \
                                        const void *restrict sp,       \
                                        void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                      \
    const CT s = *(const CT *)sp;                                      \
    uint8_t *out = (uint8_t *)op;                                      \
    size_t vl = CNT();                                                 \
    for (size_t i = 0; i < n; i += vl) {                               \
      svbool_t pg = svwhilelt_b##W((uint32_t)i, (uint32_t)n);          \
      svuint##W##_t va = svld1_u##W(pg, a + i);                        \
      svbool_t p = CMP_N(pg, va, s);                                   \
      svbool_t pg8 = svwhilelt_b8((uint32_t)i, (uint32_t)n);           \
      svst1_u8(pg8, out + i, svsel_u8(p, svdup_u8(1), svdup_u8(0)));   \
    }                                                                  \
  }

/* clang-format off */
STAMP_CMPSC_UINT_WIDE_SVE(eq_u16, uint16_t, 16, svcnth, svcmpeq_n_u16)
STAMP_CMPSC_UINT_WIDE_SVE(gt_u16, uint16_t, 16, svcnth, svcmpgt_n_u16)
STAMP_CMPSC_UINT_WIDE_SVE(lt_u16, uint16_t, 16, svcnth, svcmplt_n_u16)
STAMP_CMPSC_UINT_WIDE_SVE(ge_u16, uint16_t, 16, svcnth, svcmpge_n_u16)
STAMP_CMPSC_UINT_WIDE_SVE(le_u16, uint16_t, 16, svcnth, svcmple_n_u16)

STAMP_CMPSC_UINT_WIDE_SVE(eq_u32, uint32_t, 32, svcntw, svcmpeq_n_u32)
STAMP_CMPSC_UINT_WIDE_SVE(gt_u32, uint32_t, 32, svcntw, svcmpgt_n_u32)
STAMP_CMPSC_UINT_WIDE_SVE(lt_u32, uint32_t, 32, svcntw, svcmplt_n_u32)
STAMP_CMPSC_UINT_WIDE_SVE(ge_u32, uint32_t, 32, svcntw, svcmpge_n_u32)
STAMP_CMPSC_UINT_WIDE_SVE(le_u32, uint32_t, 32, svcntw, svcmple_n_u32)

STAMP_CMPSC_UINT_WIDE_SVE(eq_u64, uint64_t, 64, svcntd, svcmpeq_n_u64)
STAMP_CMPSC_UINT_WIDE_SVE(gt_u64, uint64_t, 64, svcntd, svcmpgt_n_u64)
STAMP_CMPSC_UINT_WIDE_SVE(lt_u64, uint64_t, 64, svcntd, svcmplt_n_u64)
STAMP_CMPSC_UINT_WIDE_SVE(ge_u64, uint64_t, 64, svcntd, svcmpge_n_u64)
STAMP_CMPSC_UINT_WIDE_SVE(le_u64, uint64_t, 64, svcntd, svcmple_n_u64)
/* clang-format on */
#undef STAMP_CMPSC_UINT_WIDE_SVE

/* ── Float (32/64): byte output ─────────────────────────────────── */

#define STAMP_CMPSC_FLOAT_SVE(SFX, CT, W, CNT, CMP_N)                  \
  static inline void _cmpsc_##SFX##_sve(const void *restrict ap,       \
                                        const void *restrict sp,       \
                                        void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                      \
    const CT s = *(const CT *)sp;                                      \
    uint8_t *out = (uint8_t *)op;                                      \
    size_t vl = CNT();                                                 \
    for (size_t i = 0; i < n; i += vl) {                               \
      svbool_t pg = svwhilelt_b##W((uint32_t)i, (uint32_t)n);          \
      svfloat##W##_t va = svld1_f##W(pg, a + i);                       \
      svbool_t p = CMP_N(pg, va, s);                                   \
      svbool_t pg8 = svwhilelt_b8((uint32_t)i, (uint32_t)n);           \
      svst1_u8(pg8, out + i, svsel_u8(p, svdup_u8(1), svdup_u8(0)));   \
    }                                                                  \
  }

/* clang-format off */
STAMP_CMPSC_FLOAT_SVE(eq_f32, float,  32, svcntw, svcmpeq_n_f32)
STAMP_CMPSC_FLOAT_SVE(gt_f32, float,  32, svcntw, svcmpgt_n_f32)
STAMP_CMPSC_FLOAT_SVE(lt_f32, float,  32, svcntw, svcmplt_n_f32)
STAMP_CMPSC_FLOAT_SVE(ge_f32, float,  32, svcntw, svcmpge_n_f32)
STAMP_CMPSC_FLOAT_SVE(le_f32, float,  32, svcntw, svcmple_n_f32)

STAMP_CMPSC_FLOAT_SVE(eq_f64, double, 64, svcntd, svcmpeq_n_f64)
STAMP_CMPSC_FLOAT_SVE(gt_f64, double, 64, svcntd, svcmpgt_n_f64)
STAMP_CMPSC_FLOAT_SVE(lt_f64, double, 64, svcntd, svcmplt_n_f64)
STAMP_CMPSC_FLOAT_SVE(ge_f64, double, 64, svcntd, svcmpge_n_f64)
STAMP_CMPSC_FLOAT_SVE(le_f64, double, 64, svcntd, svcmple_n_f64)
/* clang-format on */
#undef STAMP_CMPSC_FLOAT_SVE

#endif /* NUMC_COMPARE_SCALAR_SVE_H */
