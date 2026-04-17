#ifndef NUMC_REDUCE_SVE_H
#define NUMC_REDUCE_SVE_H

#include <arm_sve.h>
#include <limits.h>
#include <stdint.h>

// clang-format off

/* -- scalar comparison helpers ------------------------------------- */

#define _RSMIN(a, b) ((a) < (b) ? (a) : (b))
#define _RSMAX(a, b) ((a) > (b) ? (a) : (b))

/* -- full array min/max reduction (SVE) ---------------------------- *
 *
 * 4-vector unrolled main loop, then predicated cleanup via svwhilelt.
 * SVE handles tails natively — no scalar tail loop needed.            */

#define DEFINE_REDUCE_FULL_SVE(NAME, CT, SVT, PTRUE, SVCNT, SVDUP,    \
                               SVLD1, SVOP_X, SVOP_M, SVOPV,          \
                               SVWHILELT)                              \
  static inline CT NAME(const CT *restrict a, size_t n) {              \
    size_t vl = SVCNT();                                               \
    svbool_t ptrue = PTRUE();                                          \
    SVT acc0 = SVLD1(SVWHILELT((uint64_t)0, (uint64_t)n), a);         \
    SVT acc1 = acc0, acc2 = acc0, acc3 = acc0;                         \
    size_t i = vl;                                                     \
    for (; i + 4 * vl <= n; i += 4 * vl) {                             \
      acc0 = SVOP_X(ptrue, acc0, SVLD1(ptrue, a + i));                 \
      acc1 = SVOP_X(ptrue, acc1, SVLD1(ptrue, a + i + vl));            \
      acc2 = SVOP_X(ptrue, acc2, SVLD1(ptrue, a + i + 2 * vl));        \
      acc3 = SVOP_X(ptrue, acc3, SVLD1(ptrue, a + i + 3 * vl));        \
    }                                                                  \
    acc0 = SVOP_X(ptrue, SVOP_X(ptrue, acc0, acc1),                    \
                         SVOP_X(ptrue, acc2, acc3));                    \
    for (; i < n;) {                                                   \
      svbool_t pg = SVWHILELT((uint64_t)i, (uint64_t)n);               \
      acc0 = SVOP_M(pg, acc0, SVLD1(pg, a + i));                       \
      i += vl;                                                         \
    }                                                                  \
    return SVOPV(ptrue, acc0);                                         \
  }

/* min reductions */
DEFINE_REDUCE_FULL_SVE(reduce_min_i8_sve,  int8_t,  svint8_t,
  svptrue_b8,  svcntb, svdup_s8,  svld1_s8,
  svmin_s8_x,  svmin_s8_m,  svminv_s8,  svwhilelt_b8)
DEFINE_REDUCE_FULL_SVE(reduce_min_u8_sve,  uint8_t, svuint8_t,
  svptrue_b8,  svcntb, svdup_u8,  svld1_u8,
  svmin_u8_x,  svmin_u8_m,  svminv_u8,  svwhilelt_b8)
DEFINE_REDUCE_FULL_SVE(reduce_min_i16_sve, int16_t, svint16_t,
  svptrue_b16, svcnth, svdup_s16, svld1_s16,
  svmin_s16_x, svmin_s16_m, svminv_s16, svwhilelt_b16)
DEFINE_REDUCE_FULL_SVE(reduce_min_u16_sve, uint16_t, svuint16_t,
  svptrue_b16, svcnth, svdup_u16, svld1_u16,
  svmin_u16_x, svmin_u16_m, svminv_u16, svwhilelt_b16)
DEFINE_REDUCE_FULL_SVE(reduce_min_i32_sve, int32_t, svint32_t,
  svptrue_b32, svcntw, svdup_s32, svld1_s32,
  svmin_s32_x, svmin_s32_m, svminv_s32, svwhilelt_b32)
DEFINE_REDUCE_FULL_SVE(reduce_min_u32_sve, uint32_t, svuint32_t,
  svptrue_b32, svcntw, svdup_u32, svld1_u32,
  svmin_u32_x, svmin_u32_m, svminv_u32, svwhilelt_b32)
DEFINE_REDUCE_FULL_SVE(reduce_min_i64_sve, int64_t, svint64_t,
  svptrue_b64, svcntd, svdup_s64, svld1_s64,
  svmin_s64_x, svmin_s64_m, svminv_s64, svwhilelt_b64)
DEFINE_REDUCE_FULL_SVE(reduce_min_u64_sve, uint64_t, svuint64_t,
  svptrue_b64, svcntd, svdup_u64, svld1_u64,
  svmin_u64_x, svmin_u64_m, svminv_u64, svwhilelt_b64)

/* max reductions */
DEFINE_REDUCE_FULL_SVE(reduce_max_i8_sve,  int8_t,  svint8_t,
  svptrue_b8,  svcntb, svdup_s8,  svld1_s8,
  svmax_s8_x,  svmax_s8_m,  svmaxv_s8,  svwhilelt_b8)
DEFINE_REDUCE_FULL_SVE(reduce_max_u8_sve,  uint8_t, svuint8_t,
  svptrue_b8,  svcntb, svdup_u8,  svld1_u8,
  svmax_u8_x,  svmax_u8_m,  svmaxv_u8,  svwhilelt_b8)
DEFINE_REDUCE_FULL_SVE(reduce_max_i16_sve, int16_t, svint16_t,
  svptrue_b16, svcnth, svdup_s16, svld1_s16,
  svmax_s16_x, svmax_s16_m, svmaxv_s16, svwhilelt_b16)
DEFINE_REDUCE_FULL_SVE(reduce_max_u16_sve, uint16_t, svuint16_t,
  svptrue_b16, svcnth, svdup_u16, svld1_u16,
  svmax_u16_x, svmax_u16_m, svmaxv_u16, svwhilelt_b16)
DEFINE_REDUCE_FULL_SVE(reduce_max_i32_sve, int32_t, svint32_t,
  svptrue_b32, svcntw, svdup_s32, svld1_s32,
  svmax_s32_x, svmax_s32_m, svmaxv_s32, svwhilelt_b32)
DEFINE_REDUCE_FULL_SVE(reduce_max_u32_sve, uint32_t, svuint32_t,
  svptrue_b32, svcntw, svdup_u32, svld1_u32,
  svmax_u32_x, svmax_u32_m, svmaxv_u32, svwhilelt_b32)
DEFINE_REDUCE_FULL_SVE(reduce_max_i64_sve, int64_t, svint64_t,
  svptrue_b64, svcntd, svdup_s64, svld1_s64,
  svmax_s64_x, svmax_s64_m, svmaxv_s64, svwhilelt_b64)
DEFINE_REDUCE_FULL_SVE(reduce_max_u64_sve, uint64_t, svuint64_t,
  svptrue_b64, svcntd, svdup_u64, svld1_u64,
  svmax_u64_x, svmax_u64_m, svmaxv_u64, svwhilelt_b64)

#undef DEFINE_REDUCE_FULL_SVE

/* -- fused row-reduce (axis-1 reduction, SVE) ---------------------- *
 *
 * Processes 4 rows at a time, vectorizing the inner column loop.
 * SVE predication handles column tails — no scalar cleanup needed.    */

#define DEFINE_FUSED_REDUCE_SVE(NAME, CT, SVT, PTRUE, SVCNT, SVLD1,   \
                                SVST1, SVOP_X, SVWHILELT)              \
  static inline void NAME(const char *restrict base,                   \
                           intptr_t row_stride, size_t nrows,          \
                           char *restrict dst, size_t ncols) {         \
    CT *restrict d = (CT *)dst;                                        \
    size_t vl = SVCNT();                                               \
    svbool_t ptrue = PTRUE();                                          \
    size_t r = 0;                                                      \
    for (; r + 4 <= nrows; r += 4) {                                   \
      const CT *restrict s0 =                                          \
          (const CT *)(base + r * row_stride);                         \
      const CT *restrict s1 =                                          \
          (const CT *)(base + (r + 1) * row_stride);                   \
      const CT *restrict s2 =                                          \
          (const CT *)(base + (r + 2) * row_stride);                   \
      const CT *restrict s3 =                                          \
          (const CT *)(base + (r + 3) * row_stride);                   \
      size_t i = 0;                                                    \
      for (; i + vl <= ncols; i += vl) {                               \
        SVT dv  = SVLD1(ptrue, d + i);                                 \
        SVT v01 = SVOP_X(ptrue, SVLD1(ptrue, s0 + i),                 \
                                SVLD1(ptrue, s1 + i));                 \
        SVT v23 = SVOP_X(ptrue, SVLD1(ptrue, s2 + i),                 \
                                SVLD1(ptrue, s3 + i));                 \
        SVST1(ptrue, d + i,                                            \
              SVOP_X(ptrue, dv, SVOP_X(ptrue, v01, v23)));             \
      }                                                                \
      if (i < ncols) {                                                 \
        svbool_t pg = SVWHILELT((uint64_t)i, (uint64_t)ncols);         \
        SVT dv  = SVLD1(pg, d + i);                                    \
        SVT v01 = SVOP_X(pg, SVLD1(pg, s0 + i),                       \
                              SVLD1(pg, s1 + i));                      \
        SVT v23 = SVOP_X(pg, SVLD1(pg, s2 + i),                       \
                              SVLD1(pg, s3 + i));                      \
        SVST1(pg, d + i,                                               \
              SVOP_X(pg, dv, SVOP_X(pg, v01, v23)));                   \
      }                                                                \
    }                                                                  \
    for (; r < nrows; r++) {                                           \
      const CT *restrict s =                                           \
          (const CT *)(base + r * row_stride);                         \
      size_t i = 0;                                                    \
      for (; i + vl <= ncols; i += vl)                                 \
        SVST1(ptrue, d + i,                                            \
              SVOP_X(ptrue, SVLD1(ptrue, d + i),                       \
                            SVLD1(ptrue, s + i)));                     \
      if (i < ncols) {                                                 \
        svbool_t pg = SVWHILELT((uint64_t)i, (uint64_t)ncols);         \
        SVST1(pg, d + i,                                               \
              SVOP_X(pg, SVLD1(pg, d + i), SVLD1(pg, s + i)));         \
      }                                                                \
    }                                                                  \
  }

/* min fused */
DEFINE_FUSED_REDUCE_SVE(_min_fused_i8_sve,  int8_t,  svint8_t,
  svptrue_b8,  svcntb, svld1_s8,  svst1_s8,  svmin_s8_x,  svwhilelt_b8)
DEFINE_FUSED_REDUCE_SVE(_min_fused_u8_sve,  uint8_t, svuint8_t,
  svptrue_b8,  svcntb, svld1_u8,  svst1_u8,  svmin_u8_x,  svwhilelt_b8)
DEFINE_FUSED_REDUCE_SVE(_min_fused_i16_sve, int16_t, svint16_t,
  svptrue_b16, svcnth, svld1_s16, svst1_s16, svmin_s16_x, svwhilelt_b16)
DEFINE_FUSED_REDUCE_SVE(_min_fused_u16_sve, uint16_t, svuint16_t,
  svptrue_b16, svcnth, svld1_u16, svst1_u16, svmin_u16_x, svwhilelt_b16)
DEFINE_FUSED_REDUCE_SVE(_min_fused_i32_sve, int32_t, svint32_t,
  svptrue_b32, svcntw, svld1_s32, svst1_s32, svmin_s32_x, svwhilelt_b32)
DEFINE_FUSED_REDUCE_SVE(_min_fused_u32_sve, uint32_t, svuint32_t,
  svptrue_b32, svcntw, svld1_u32, svst1_u32, svmin_u32_x, svwhilelt_b32)
DEFINE_FUSED_REDUCE_SVE(_min_fused_i64_sve, int64_t, svint64_t,
  svptrue_b64, svcntd, svld1_s64, svst1_s64, svmin_s64_x, svwhilelt_b64)
DEFINE_FUSED_REDUCE_SVE(_min_fused_u64_sve, uint64_t, svuint64_t,
  svptrue_b64, svcntd, svld1_u64, svst1_u64, svmin_u64_x, svwhilelt_b64)

/* max fused */
DEFINE_FUSED_REDUCE_SVE(_max_fused_i8_sve,  int8_t,  svint8_t,
  svptrue_b8,  svcntb, svld1_s8,  svst1_s8,  svmax_s8_x,  svwhilelt_b8)
DEFINE_FUSED_REDUCE_SVE(_max_fused_u8_sve,  uint8_t, svuint8_t,
  svptrue_b8,  svcntb, svld1_u8,  svst1_u8,  svmax_u8_x,  svwhilelt_b8)
DEFINE_FUSED_REDUCE_SVE(_max_fused_i16_sve, int16_t, svint16_t,
  svptrue_b16, svcnth, svld1_s16, svst1_s16, svmax_s16_x, svwhilelt_b16)
DEFINE_FUSED_REDUCE_SVE(_max_fused_u16_sve, uint16_t, svuint16_t,
  svptrue_b16, svcnth, svld1_u16, svst1_u16, svmax_u16_x, svwhilelt_b16)
DEFINE_FUSED_REDUCE_SVE(_max_fused_i32_sve, int32_t, svint32_t,
  svptrue_b32, svcntw, svld1_s32, svst1_s32, svmax_s32_x, svwhilelt_b32)
DEFINE_FUSED_REDUCE_SVE(_max_fused_u32_sve, uint32_t, svuint32_t,
  svptrue_b32, svcntw, svld1_u32, svst1_u32, svmax_u32_x, svwhilelt_b32)
DEFINE_FUSED_REDUCE_SVE(_max_fused_i64_sve, int64_t, svint64_t,
  svptrue_b64, svcntd, svld1_s64, svst1_s64, svmax_s64_x, svwhilelt_b64)
DEFINE_FUSED_REDUCE_SVE(_max_fused_u64_sve, uint64_t, svuint64_t,
  svptrue_b64, svcntd, svld1_u64, svst1_u64, svmax_u64_x, svwhilelt_b64)

#undef DEFINE_FUSED_REDUCE_SVE

// clang-format on

#undef _RSMIN
#undef _RSMAX

#endif
