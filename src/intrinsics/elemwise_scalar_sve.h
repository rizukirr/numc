/**
 * @file elemwise_scalar_sve.h
 * @brief SVE scalar arithmetic kernels for all 10 types.
 *
 * Operations: add_scalar, sub_scalar, mul_scalar
 *
 * SVE uses predicated loops with svwhilelt for natural tail handling.
 * Scalar-immediate variants (_n_) avoid explicit broadcast.
 */
#ifndef NUMC_ELEMWISE_SCALAR_SVE_H
#define NUMC_ELEMWISE_SCALAR_SVE_H

#include <arm_sve.h>
#include <stddef.h>
#include <stdint.h>

/* ====================================================================
 * Signed integer macro
 * ================================================================ */

#define FAST_SCAL_SINT_SVE(OP, SFX, CT, W, CNT, VEC_OP)                    \
  static inline void _fast_##OP##_scalar_##SFX##_sve(                      \
      const void *restrict ap, const void *restrict sp, void *restrict op, \
      size_t n) {                                                          \
    const CT *a = (const CT *)ap;                                          \
    const CT s = *(const CT *)sp;                                          \
    CT *out = (CT *)op;                                                    \
    size_t vl = CNT();                                                     \
    for (size_t i = 0; i < n; i += vl) {                                   \
      svbool_t pg = svwhilelt_b##W((uint32_t)i, (uint32_t)n);              \
      svint##W##_t va = svld1_s##W(pg, a + i);                             \
      svst1_s##W(pg, out + i, VEC_OP(pg, va, s));                          \
    }                                                                      \
  }

/* ====================================================================
 * Unsigned integer macro
 * ================================================================ */

#define FAST_SCAL_UINT_SVE(OP, SFX, CT, W, CNT, VEC_OP)                    \
  static inline void _fast_##OP##_scalar_##SFX##_sve(                      \
      const void *restrict ap, const void *restrict sp, void *restrict op, \
      size_t n) {                                                          \
    const CT *a = (const CT *)ap;                                          \
    const CT s = *(const CT *)sp;                                          \
    CT *out = (CT *)op;                                                    \
    size_t vl = CNT();                                                     \
    for (size_t i = 0; i < n; i += vl) {                                   \
      svbool_t pg = svwhilelt_b##W((uint32_t)i, (uint32_t)n);              \
      svuint##W##_t va = svld1_u##W(pg, a + i);                            \
      svst1_u##W(pg, out + i, VEC_OP(pg, va, s));                          \
    }                                                                      \
  }

/* ====================================================================
 * Float macro
 * ================================================================ */

#define FAST_SCAL_F32_SVE(OP, VEC_OP)                                      \
  static inline void _fast_##OP##_scalar_f32_sve(                          \
      const void *restrict ap, const void *restrict sp, void *restrict op, \
      size_t n) {                                                          \
    const float *a = (const float *)ap;                                    \
    const float s = *(const float *)sp;                                    \
    float *out = (float *)op;                                              \
    size_t vl = svcntw();                                                  \
    for (size_t i = 0; i < n; i += vl) {                                   \
      svbool_t pg = svwhilelt_b32((uint32_t)i, (uint32_t)n);               \
      svfloat32_t va = svld1_f32(pg, a + i);                               \
      svst1_f32(pg, out + i, VEC_OP(pg, va, s));                           \
    }                                                                      \
  }

#define FAST_SCAL_F64_SVE(OP, VEC_OP)                                      \
  static inline void _fast_##OP##_scalar_f64_sve(                          \
      const void *restrict ap, const void *restrict sp, void *restrict op, \
      size_t n) {                                                          \
    const double *a = (const double *)ap;                                  \
    const double s = *(const double *)sp;                                  \
    double *out = (double *)op;                                            \
    size_t vl = svcntd();                                                  \
    for (size_t i = 0; i < n; i += vl) {                                   \
      svbool_t pg = svwhilelt_b64((uint32_t)i, (uint32_t)n);               \
      svfloat64_t va = svld1_f64(pg, a + i);                               \
      svst1_f64(pg, out + i, VEC_OP(pg, va, s));                           \
    }                                                                      \
  }

/* -- Add scalar --------------------------------------------------- */

FAST_SCAL_SINT_SVE(add, i8, int8_t, 8, svcntb, svadd_n_s8_x)
FAST_SCAL_SINT_SVE(add, i16, int16_t, 16, svcnth, svadd_n_s16_x)
FAST_SCAL_SINT_SVE(add, i32, int32_t, 32, svcntw, svadd_n_s32_x)
FAST_SCAL_SINT_SVE(add, i64, int64_t, 64, svcntd, svadd_n_s64_x)
FAST_SCAL_UINT_SVE(add, u8, uint8_t, 8, svcntb, svadd_n_u8_x)
FAST_SCAL_UINT_SVE(add, u16, uint16_t, 16, svcnth, svadd_n_u16_x)
FAST_SCAL_UINT_SVE(add, u32, uint32_t, 32, svcntw, svadd_n_u32_x)
FAST_SCAL_UINT_SVE(add, u64, uint64_t, 64, svcntd, svadd_n_u64_x)
FAST_SCAL_F32_SVE(add, svadd_n_f32_x)
FAST_SCAL_F64_SVE(add, svadd_n_f64_x)

/* -- Sub scalar --------------------------------------------------- */

FAST_SCAL_SINT_SVE(sub, i8, int8_t, 8, svcntb, svsub_n_s8_x)
FAST_SCAL_SINT_SVE(sub, i16, int16_t, 16, svcnth, svsub_n_s16_x)
FAST_SCAL_SINT_SVE(sub, i32, int32_t, 32, svcntw, svsub_n_s32_x)
FAST_SCAL_SINT_SVE(sub, i64, int64_t, 64, svcntd, svsub_n_s64_x)
FAST_SCAL_UINT_SVE(sub, u8, uint8_t, 8, svcntb, svsub_n_u8_x)
FAST_SCAL_UINT_SVE(sub, u16, uint16_t, 16, svcnth, svsub_n_u16_x)
FAST_SCAL_UINT_SVE(sub, u32, uint32_t, 32, svcntw, svsub_n_u32_x)
FAST_SCAL_UINT_SVE(sub, u64, uint64_t, 64, svcntd, svsub_n_u64_x)
FAST_SCAL_F32_SVE(sub, svsub_n_f32_x)
FAST_SCAL_F64_SVE(sub, svsub_n_f64_x)

/* -- Mul scalar --------------------------------------------------- */

FAST_SCAL_SINT_SVE(mul, i8, int8_t, 8, svcntb, svmul_n_s8_x)
FAST_SCAL_SINT_SVE(mul, i16, int16_t, 16, svcnth, svmul_n_s16_x)
FAST_SCAL_SINT_SVE(mul, i32, int32_t, 32, svcntw, svmul_n_s32_x)
FAST_SCAL_SINT_SVE(mul, i64, int64_t, 64, svcntd, svmul_n_s64_x)
FAST_SCAL_UINT_SVE(mul, u8, uint8_t, 8, svcntb, svmul_n_u8_x)
FAST_SCAL_UINT_SVE(mul, u16, uint16_t, 16, svcnth, svmul_n_u16_x)
FAST_SCAL_UINT_SVE(mul, u32, uint32_t, 32, svcntw, svmul_n_u32_x)
FAST_SCAL_UINT_SVE(mul, u64, uint64_t, 64, svcntd, svmul_n_u64_x)
FAST_SCAL_F32_SVE(mul, svmul_n_f32_x)
FAST_SCAL_F64_SVE(mul, svmul_n_f64_x)

#undef FAST_SCAL_SINT_SVE
#undef FAST_SCAL_UINT_SVE
#undef FAST_SCAL_F32_SVE
#undef FAST_SCAL_F64_SVE

#endif /* NUMC_ELEMWISE_SCALAR_SVE_H */
