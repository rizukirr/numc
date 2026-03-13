/**
 * @file elemwise_sve.h
 * @brief SVE element-wise binary/unary kernels for all 10 types.
 *
 * Binary: sub, mul, maximum, minimum
 * Unary: neg, abs
 *
 * SVE has native support for all widths (8/16/32/64-bit) for all ops
 * including mul, max, min, abs. Predication handles tails automatically.
 */
#ifndef NUMC_ELEMWISE_SVE_H
#define NUMC_ELEMWISE_SVE_H

#include <arm_sve.h>
#include <stddef.h>
#include <stdint.h>

/* ════════════════════════════════════════════════════════════════════
 * Binary: signed integer macro
 * ════════════════════════════════════════════════════════════════ */

#define FAST_BIN_SINT_SVE(OP, SFX, CT, W, CNT, VEC_OP)                       \
  static inline void _fast_##OP##_##SFX##_sve(const void *restrict ap,       \
                                              const void *restrict bp,       \
                                              void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                            \
    const CT *b = (const CT *)bp;                                            \
    CT *out = (CT *)op;                                                      \
    size_t vl = CNT();                                                       \
    size_t i = 0;                                                            \
    for (; i < n; i += vl) {                                                 \
      svbool_t pg = svwhilelt_b##W((uint32_t)i, (uint32_t)n);                \
      svint##W##_t va = svld1_s##W(pg, a + i);                               \
      svint##W##_t vb = svld1_s##W(pg, b + i);                               \
      svst1_s##W(pg, out + i, VEC_OP(pg, va, vb));                           \
    }                                                                        \
  }

#define FAST_BIN_UINT_SVE(OP, SFX, CT, W, CNT, VEC_OP)                       \
  static inline void _fast_##OP##_##SFX##_sve(const void *restrict ap,       \
                                              const void *restrict bp,       \
                                              void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                            \
    const CT *b = (const CT *)bp;                                            \
    CT *out = (CT *)op;                                                      \
    size_t vl = CNT();                                                       \
    size_t i = 0;                                                            \
    for (; i < n; i += vl) {                                                 \
      svbool_t pg = svwhilelt_b##W((uint32_t)i, (uint32_t)n);                \
      svuint##W##_t va = svld1_u##W(pg, a + i);                              \
      svuint##W##_t vb = svld1_u##W(pg, b + i);                              \
      svst1_u##W(pg, out + i, VEC_OP(pg, va, vb));                           \
    }                                                                        \
  }

#define FAST_BIN_F32_SVE(OP, VEC_OP)                                     \
  static inline void _fast_##OP##_f32_sve(const void *restrict ap,       \
                                          const void *restrict bp,       \
                                          void *restrict op, size_t n) { \
    const float *a = (const float *)ap;                                  \
    const float *b = (const float *)bp;                                  \
    float *out = (float *)op;                                            \
    size_t vl = svcntw();                                                \
    size_t i = 0;                                                        \
    for (; i < n; i += vl) {                                             \
      svbool_t pg = svwhilelt_b32((uint32_t)i, (uint32_t)n);             \
      svfloat32_t va = svld1_f32(pg, a + i);                             \
      svfloat32_t vb = svld1_f32(pg, b + i);                             \
      svst1_f32(pg, out + i, VEC_OP(pg, va, vb));                        \
    }                                                                    \
  }

#define FAST_BIN_F64_SVE(OP, VEC_OP)                                     \
  static inline void _fast_##OP##_f64_sve(const void *restrict ap,       \
                                          const void *restrict bp,       \
                                          void *restrict op, size_t n) { \
    const double *a = (const double *)ap;                                \
    const double *b = (const double *)bp;                                \
    double *out = (double *)op;                                          \
    size_t vl = svcntd();                                                \
    size_t i = 0;                                                        \
    for (; i < n; i += vl) {                                             \
      svbool_t pg = svwhilelt_b64((uint32_t)i, (uint32_t)n);             \
      svfloat64_t va = svld1_f64(pg, a + i);                             \
      svfloat64_t vb = svld1_f64(pg, b + i);                             \
      svst1_f64(pg, out + i, VEC_OP(pg, va, vb));                        \
    }                                                                    \
  }

/* ── Add ─────────────────────────────────────────────────────────── */

FAST_BIN_SINT_SVE(add, i8, int8_t, 8, svcntb, svadd_s8_x)
FAST_BIN_SINT_SVE(add, i16, int16_t, 16, svcnth, svadd_s16_x)
FAST_BIN_SINT_SVE(add, i32, int32_t, 32, svcntw, svadd_s32_x)
FAST_BIN_SINT_SVE(add, i64, int64_t, 64, svcntd, svadd_s64_x)
FAST_BIN_UINT_SVE(add, u8, uint8_t, 8, svcntb, svadd_u8_x)
FAST_BIN_UINT_SVE(add, u16, uint16_t, 16, svcnth, svadd_u16_x)
FAST_BIN_UINT_SVE(add, u32, uint32_t, 32, svcntw, svadd_u32_x)
FAST_BIN_UINT_SVE(add, u64, uint64_t, 64, svcntd, svadd_u64_x)
FAST_BIN_F32_SVE(add, svadd_f32_x)
FAST_BIN_F64_SVE(add, svadd_f64_x)

/* ── Sub ─────────────────────────────────────────────────────────── */

FAST_BIN_SINT_SVE(sub, i8, int8_t, 8, svcntb, svsub_s8_x)
FAST_BIN_SINT_SVE(sub, i16, int16_t, 16, svcnth, svsub_s16_x)
FAST_BIN_SINT_SVE(sub, i32, int32_t, 32, svcntw, svsub_s32_x)
FAST_BIN_SINT_SVE(sub, i64, int64_t, 64, svcntd, svsub_s64_x)
FAST_BIN_UINT_SVE(sub, u8, uint8_t, 8, svcntb, svsub_u8_x)
FAST_BIN_UINT_SVE(sub, u16, uint16_t, 16, svcnth, svsub_u16_x)
FAST_BIN_UINT_SVE(sub, u32, uint32_t, 32, svcntw, svsub_u32_x)
FAST_BIN_UINT_SVE(sub, u64, uint64_t, 64, svcntd, svsub_u64_x)
FAST_BIN_F32_SVE(sub, svsub_f32_x)
FAST_BIN_F64_SVE(sub, svsub_f64_x)

/* ── Mul ─────────────────────────────────────────────────────────── */

FAST_BIN_SINT_SVE(mul, i8, int8_t, 8, svcntb, svmul_s8_x)
FAST_BIN_SINT_SVE(mul, i16, int16_t, 16, svcnth, svmul_s16_x)
FAST_BIN_SINT_SVE(mul, i32, int32_t, 32, svcntw, svmul_s32_x)
FAST_BIN_SINT_SVE(mul, i64, int64_t, 64, svcntd, svmul_s64_x)
FAST_BIN_UINT_SVE(mul, u8, uint8_t, 8, svcntb, svmul_u8_x)
FAST_BIN_UINT_SVE(mul, u16, uint16_t, 16, svcnth, svmul_u16_x)
FAST_BIN_UINT_SVE(mul, u32, uint32_t, 32, svcntw, svmul_u32_x)
FAST_BIN_UINT_SVE(mul, u64, uint64_t, 64, svcntd, svmul_u64_x)
FAST_BIN_F32_SVE(mul, svmul_f32_x)
FAST_BIN_F64_SVE(mul, svmul_f64_x)

/* ── Maximum ─────────────────────────────────────────────────────── */

FAST_BIN_SINT_SVE(maximum, i8, int8_t, 8, svcntb, svmax_s8_x)
FAST_BIN_SINT_SVE(maximum, i16, int16_t, 16, svcnth, svmax_s16_x)
FAST_BIN_SINT_SVE(maximum, i32, int32_t, 32, svcntw, svmax_s32_x)
FAST_BIN_SINT_SVE(maximum, i64, int64_t, 64, svcntd, svmax_s64_x)
FAST_BIN_UINT_SVE(maximum, u8, uint8_t, 8, svcntb, svmax_u8_x)
FAST_BIN_UINT_SVE(maximum, u16, uint16_t, 16, svcnth, svmax_u16_x)
FAST_BIN_UINT_SVE(maximum, u32, uint32_t, 32, svcntw, svmax_u32_x)
FAST_BIN_UINT_SVE(maximum, u64, uint64_t, 64, svcntd, svmax_u64_x)
FAST_BIN_F32_SVE(maximum, svmax_f32_x)
FAST_BIN_F64_SVE(maximum, svmax_f64_x)

/* ── Minimum ─────────────────────────────────────────────────────── */

FAST_BIN_SINT_SVE(minimum, i8, int8_t, 8, svcntb, svmin_s8_x)
FAST_BIN_SINT_SVE(minimum, i16, int16_t, 16, svcnth, svmin_s16_x)
FAST_BIN_SINT_SVE(minimum, i32, int32_t, 32, svcntw, svmin_s32_x)
FAST_BIN_SINT_SVE(minimum, i64, int64_t, 64, svcntd, svmin_s64_x)
FAST_BIN_UINT_SVE(minimum, u8, uint8_t, 8, svcntb, svmin_u8_x)
FAST_BIN_UINT_SVE(minimum, u16, uint16_t, 16, svcnth, svmin_u16_x)
FAST_BIN_UINT_SVE(minimum, u32, uint32_t, 32, svcntw, svmin_u32_x)
FAST_BIN_UINT_SVE(minimum, u64, uint64_t, 64, svcntd, svmin_u64_x)
FAST_BIN_F32_SVE(minimum, svmin_f32_x)
FAST_BIN_F64_SVE(minimum, svmin_f64_x)

#undef FAST_BIN_SINT_SVE
#undef FAST_BIN_UINT_SVE
#undef FAST_BIN_F32_SVE
#undef FAST_BIN_F64_SVE

/* ════════════════════════════════════════════════════════════════════
 * Unary operations
 * ════════════════════════════════════════════════════════════════ */

/* ── Neg (signed integers) ───────────────────────────────────────── */

#define FAST_NEG_SINT_SVE(SFX, CT, W, CNT)                                \
  static inline void _fast_neg_##SFX##_sve(const void *restrict ap,       \
                                           void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl = CNT();                                                    \
    size_t i = 0;                                                         \
    for (; i < n; i += vl) {                                              \
      svbool_t pg = svwhilelt_b##W((uint32_t)i, (uint32_t)n);             \
      svint##W##_t va = svld1_s##W(pg, a + i);                            \
      svst1_s##W(pg, out + i, svneg_s##W##_x(pg, va));                    \
    }                                                                     \
  }

FAST_NEG_SINT_SVE(i8, int8_t, 8, svcntb)
FAST_NEG_SINT_SVE(i16, int16_t, 16, svcnth)
FAST_NEG_SINT_SVE(i32, int32_t, 32, svcntw)
FAST_NEG_SINT_SVE(i64, int64_t, 64, svcntd)

#undef FAST_NEG_SINT_SVE

/* ── Neg (unsigned integers): 0 - val ────────────────────────────── */

#define FAST_NEG_UINT_SVE(SFX, CT, W, CNT)                                \
  static inline void _fast_neg_##SFX##_sve(const void *restrict ap,       \
                                           void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl = CNT();                                                    \
    size_t i = 0;                                                         \
    for (; i < n; i += vl) {                                              \
      svbool_t pg = svwhilelt_b##W((uint32_t)i, (uint32_t)n);             \
      svuint##W##_t va = svld1_u##W(pg, a + i);                           \
      svst1_u##W(pg, out + i, svsub_u##W##_x(pg, svdup_u##W(0), va));     \
    }                                                                     \
  }

FAST_NEG_UINT_SVE(u8, uint8_t, 8, svcntb)
FAST_NEG_UINT_SVE(u16, uint16_t, 16, svcnth)
FAST_NEG_UINT_SVE(u32, uint32_t, 32, svcntw)
FAST_NEG_UINT_SVE(u64, uint64_t, 64, svcntd)

#undef FAST_NEG_UINT_SVE

/* ── Neg (float) ─────────────────────────────────────────────────── */

static inline void _fast_neg_f32_sve(const void *restrict ap, void *restrict op,
                                     size_t n) {
  const float *a = (const float *)ap;
  float *out = (float *)op;
  size_t vl = svcntw();
  size_t i = 0;
  for (; i < n; i += vl) {
    svbool_t pg = svwhilelt_b32((uint32_t)i, (uint32_t)n);
    svfloat32_t va = svld1_f32(pg, a + i);
    svst1_f32(pg, out + i, svneg_f32_x(pg, va));
  }
}

static inline void _fast_neg_f64_sve(const void *restrict ap, void *restrict op,
                                     size_t n) {
  const double *a = (const double *)ap;
  double *out = (double *)op;
  size_t vl = svcntd();
  size_t i = 0;
  for (; i < n; i += vl) {
    svbool_t pg = svwhilelt_b64((uint32_t)i, (uint32_t)n);
    svfloat64_t va = svld1_f64(pg, a + i);
    svst1_f64(pg, out + i, svneg_f64_x(pg, va));
  }
}

/* ── Abs (signed integers) ───────────────────────────────────────── */

#define FAST_ABS_SINT_SVE(SFX, CT, W, CNT)                                \
  static inline void _fast_abs_##SFX##_sve(const void *restrict ap,       \
                                           void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl = CNT();                                                    \
    size_t i = 0;                                                         \
    for (; i < n; i += vl) {                                              \
      svbool_t pg = svwhilelt_b##W((uint32_t)i, (uint32_t)n);             \
      svint##W##_t va = svld1_s##W(pg, a + i);                            \
      svst1_s##W(pg, out + i, svabs_s##W##_x(pg, va));                    \
    }                                                                     \
  }

FAST_ABS_SINT_SVE(i8, int8_t, 8, svcntb)
FAST_ABS_SINT_SVE(i16, int16_t, 16, svcnth)
FAST_ABS_SINT_SVE(i32, int32_t, 32, svcntw)
FAST_ABS_SINT_SVE(i64, int64_t, 64, svcntd)

#undef FAST_ABS_SINT_SVE

/* ── Abs (float) ─────────────────────────────────────────────────── */

static inline void _fast_abs_f32_sve(const void *restrict ap, void *restrict op,
                                     size_t n) {
  const float *a = (const float *)ap;
  float *out = (float *)op;
  size_t vl = svcntw();
  size_t i = 0;
  for (; i < n; i += vl) {
    svbool_t pg = svwhilelt_b32((uint32_t)i, (uint32_t)n);
    svfloat32_t va = svld1_f32(pg, a + i);
    svst1_f32(pg, out + i, svabs_f32_x(pg, va));
  }
}

static inline void _fast_abs_f64_sve(const void *restrict ap, void *restrict op,
                                     size_t n) {
  const double *a = (const double *)ap;
  double *out = (double *)op;
  size_t vl = svcntd();
  size_t i = 0;
  for (; i < n; i += vl) {
    svbool_t pg = svwhilelt_b64((uint32_t)i, (uint32_t)n);
    svfloat64_t va = svld1_f64(pg, a + i);
    svst1_f64(pg, out + i, svabs_f64_x(pg, va));
  }
}

#endif /* NUMC_ELEMWISE_SVE_H */
