/**
 * @file compare_scalar_neon.h
 * @brief NEON scalar comparison kernels — uint8 output (0/1).
 *
 * All comparison functions output uint8_t* (NumPy-compatible bool).
 * NEON comparison result (all-1s/all-0s) is narrowed to uint8 and AND-ed
 * with 1 before storing.
 */
#ifndef NUMC_COMPARE_SCALAR_NEON_H
#define NUMC_COMPARE_SCALAR_NEON_H

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

/* -- 8-bit signed: output is already byte-width ------------------- */

#define STAMP_CMPSC_I8_NEON(FNAME, CMP, SCALAR_OP)                           \
  static inline void _cmpsc_##FNAME##_i8_neon(const void *restrict ap,       \
                                              const void *restrict sp,       \
                                              void *restrict op, size_t n) { \
    const int8_t *a = (const int8_t *)ap;                                    \
    const int8_t s = *(const int8_t *)sp;                                    \
    uint8_t *out = (uint8_t *)op;                                            \
    uint8x16_t one = vdupq_n_u8(1);                                          \
    int8x16_t vs = vdupq_n_s8(s);                                            \
    size_t i = 0;                                                            \
    for (; i + 16 <= n; i += 16) {                                           \
      int8x16_t va = vld1q_s8(a + i);                                        \
      uint8x16_t r = vandq_u8(CMP(va, vs), one);                             \
      vst1q_u8(out + i, r);                                                  \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (uint8_t)(SCALAR_OP);                                         \
  }

STAMP_CMPSC_I8_NEON(eq, vceqq_s8, a[i] == s)
STAMP_CMPSC_I8_NEON(gt, vcgtq_s8, a[i] > s)
STAMP_CMPSC_I8_NEON(lt, vcltq_s8, a[i] < s)
STAMP_CMPSC_I8_NEON(ge, vcgeq_s8, a[i] >= s)
STAMP_CMPSC_I8_NEON(le, vcleq_s8, a[i] <= s)
#undef STAMP_CMPSC_I8_NEON

/* -- 16-bit signed: narrow 16→8 ----------------------------------- */

#define STAMP_CMPSC_I16_NEON(FNAME, CMP, SCALAR_OP)                           \
  static inline void _cmpsc_##FNAME##_i16_neon(const void *restrict ap,       \
                                               const void *restrict sp,       \
                                               void *restrict op, size_t n) { \
    const int16_t *a = (const int16_t *)ap;                                   \
    const int16_t s = *(const int16_t *)sp;                                   \
    uint8_t *out = (uint8_t *)op;                                             \
    uint8x16_t one = vdupq_n_u8(1);                                           \
    int16x8_t vs = vdupq_n_s16(s);                                            \
    size_t i = 0;                                                             \
    for (; i + 16 <= n; i += 16) {                                            \
      uint16x8_t c0 = CMP(vld1q_s16(a + i), vs);                              \
      uint16x8_t c1 = CMP(vld1q_s16(a + i + 8), vs);                          \
      uint8x8_t n0 = vmovn_u16(c0);                                           \
      uint8x8_t n1 = vmovn_u16(c1);                                           \
      uint8x16_t r = vandq_u8(vcombine_u8(n0, n1), one);                      \
      vst1q_u8(out + i, r);                                                   \
    }                                                                         \
    for (; i + 8 <= n; i += 8) {                                              \
      uint16x8_t c0 = CMP(vld1q_s16(a + i), vs);                              \
      uint8x8_t n0 = vmovn_u16(c0);                                           \
      uint8x8_t r = vand_u8(n0, vdup_n_u8(1));                                \
      vst1_u8(out + i, r);                                                    \
    }                                                                         \
    for (; i < n; i++)                                                        \
      out[i] = (uint8_t)(SCALAR_OP);                                          \
  }

STAMP_CMPSC_I16_NEON(eq, vceqq_s16, a[i] == s)
STAMP_CMPSC_I16_NEON(gt, vcgtq_s16, a[i] > s)
STAMP_CMPSC_I16_NEON(lt, vcltq_s16, a[i] < s)
STAMP_CMPSC_I16_NEON(ge, vcgeq_s16, a[i] >= s)
STAMP_CMPSC_I16_NEON(le, vcleq_s16, a[i] <= s)
#undef STAMP_CMPSC_I16_NEON

/* -- 32-bit signed: narrow 32→16→8 -------------------------------- */

#define STAMP_CMPSC_I32_NEON(FNAME, CMP, SCALAR_OP)                           \
  static inline void _cmpsc_##FNAME##_i32_neon(const void *restrict ap,       \
                                               const void *restrict sp,       \
                                               void *restrict op, size_t n) { \
    const int32_t *a = (const int32_t *)ap;                                   \
    const int32_t s = *(const int32_t *)sp;                                   \
    uint8_t *out = (uint8_t *)op;                                             \
    uint8x16_t one = vdupq_n_u8(1);                                           \
    int32x4_t vs = vdupq_n_s32(s);                                            \
    size_t i = 0;                                                             \
    for (; i + 16 <= n; i += 16) {                                            \
      uint32x4_t c0 = CMP(vld1q_s32(a + i), vs);                              \
      uint32x4_t c1 = CMP(vld1q_s32(a + i + 4), vs);                          \
      uint32x4_t c2 = CMP(vld1q_s32(a + i + 8), vs);                          \
      uint32x4_t c3 = CMP(vld1q_s32(a + i + 12), vs);                         \
      uint16x4_t h0 = vmovn_u32(c0);                                          \
      uint16x4_t h1 = vmovn_u32(c1);                                          \
      uint16x4_t h2 = vmovn_u32(c2);                                          \
      uint16x4_t h3 = vmovn_u32(c3);                                          \
      uint8x8_t b0 = vmovn_u16(vcombine_u16(h0, h1));                         \
      uint8x8_t b1 = vmovn_u16(vcombine_u16(h2, h3));                         \
      uint8x16_t r = vandq_u8(vcombine_u8(b0, b1), one);                      \
      vst1q_u8(out + i, r);                                                   \
    }                                                                         \
    for (; i + 4 <= n; i += 4) {                                              \
      uint32x4_t c0 = CMP(vld1q_s32(a + i), vs);                              \
      uint16x4_t h0 = vmovn_u32(c0);                                          \
      uint8x8_t b0 = vmovn_u16(vcombine_u16(h0, vdup_n_u16(0)));              \
      uint8x8_t r = vand_u8(b0, vdup_n_u8(1));                                \
      out[i] = vget_lane_u8(r, 0);                                            \
      out[i + 1] = vget_lane_u8(r, 1);                                        \
      out[i + 2] = vget_lane_u8(r, 2);                                        \
      out[i + 3] = vget_lane_u8(r, 3);                                        \
    }                                                                         \
    for (; i < n; i++)                                                        \
      out[i] = (uint8_t)(SCALAR_OP);                                          \
  }

STAMP_CMPSC_I32_NEON(eq, vceqq_s32, a[i] == s)
STAMP_CMPSC_I32_NEON(gt, vcgtq_s32, a[i] > s)
STAMP_CMPSC_I32_NEON(lt, vcltq_s32, a[i] < s)
STAMP_CMPSC_I32_NEON(ge, vcgeq_s32, a[i] >= s)
STAMP_CMPSC_I32_NEON(le, vcleq_s32, a[i] <= s)
#undef STAMP_CMPSC_I32_NEON

/* -- 64-bit signed: scalar extract (2 per vector) ----------------- */

#define STAMP_CMPSC_I64_NEON(FNAME, CMP, SCALAR_OP)                           \
  static inline void _cmpsc_##FNAME##_i64_neon(const void *restrict ap,       \
                                               const void *restrict sp,       \
                                               void *restrict op, size_t n) { \
    const int64_t *a = (const int64_t *)ap;                                   \
    const int64_t s = *(const int64_t *)sp;                                   \
    uint8_t *out = (uint8_t *)op;                                             \
    int64x2_t vs = vdupq_n_s64(s);                                            \
    size_t i = 0;                                                             \
    for (; i + 2 <= n; i += 2) {                                              \
      int64x2_t va = vld1q_s64(a + i);                                        \
      uint64x2_t c = CMP(va, vs);                                             \
      out[i] = (uint8_t)(vgetq_lane_u64(c, 0) & 1u);                          \
      out[i + 1] = (uint8_t)(vgetq_lane_u64(c, 1) & 1u);                      \
    }                                                                         \
    for (; i < n; i++)                                                        \
      out[i] = (uint8_t)(SCALAR_OP);                                          \
  }

STAMP_CMPSC_I64_NEON(eq, vceqq_s64, a[i] == s)
STAMP_CMPSC_I64_NEON(gt, vcgtq_s64, a[i] > s)
STAMP_CMPSC_I64_NEON(lt, vcltq_s64, a[i] < s)
STAMP_CMPSC_I64_NEON(ge, vcgeq_s64, a[i] >= s)
STAMP_CMPSC_I64_NEON(le, vcleq_s64, a[i] <= s)
#undef STAMP_CMPSC_I64_NEON

/* -- 8-bit unsigned: output is already byte-width ----------------- */

#define STAMP_CMPSC_U8_NEON(FNAME, CMP, SCALAR_OP)                           \
  static inline void _cmpsc_##FNAME##_u8_neon(const void *restrict ap,       \
                                              const void *restrict sp,       \
                                              void *restrict op, size_t n) { \
    const uint8_t *a = (const uint8_t *)ap;                                  \
    const uint8_t s = *(const uint8_t *)sp;                                  \
    uint8_t *out = (uint8_t *)op;                                            \
    uint8x16_t one = vdupq_n_u8(1);                                          \
    uint8x16_t vs = vdupq_n_u8(s);                                           \
    size_t i = 0;                                                            \
    for (; i + 16 <= n; i += 16) {                                           \
      uint8x16_t va = vld1q_u8(a + i);                                       \
      uint8x16_t r = vandq_u8(CMP(va, vs), one);                             \
      vst1q_u8(out + i, r);                                                  \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (uint8_t)(SCALAR_OP);                                         \
  }

STAMP_CMPSC_U8_NEON(eq, vceqq_u8, a[i] == s)
STAMP_CMPSC_U8_NEON(gt, vcgtq_u8, a[i] > s)
STAMP_CMPSC_U8_NEON(lt, vcltq_u8, a[i] < s)
STAMP_CMPSC_U8_NEON(ge, vcgeq_u8, a[i] >= s)
STAMP_CMPSC_U8_NEON(le, vcleq_u8, a[i] <= s)
#undef STAMP_CMPSC_U8_NEON

/* -- 16-bit unsigned: narrow 16→8 --------------------------------- */

#define STAMP_CMPSC_U16_NEON(FNAME, CMP, SCALAR_OP)                           \
  static inline void _cmpsc_##FNAME##_u16_neon(const void *restrict ap,       \
                                               const void *restrict sp,       \
                                               void *restrict op, size_t n) { \
    const uint16_t *a = (const uint16_t *)ap;                                 \
    const uint16_t s = *(const uint16_t *)sp;                                 \
    uint8_t *out = (uint8_t *)op;                                             \
    uint8x16_t one = vdupq_n_u8(1);                                           \
    uint16x8_t vs = vdupq_n_u16(s);                                           \
    size_t i = 0;                                                             \
    for (; i + 16 <= n; i += 16) {                                            \
      uint16x8_t c0 = CMP(vld1q_u16(a + i), vs);                              \
      uint16x8_t c1 = CMP(vld1q_u16(a + i + 8), vs);                          \
      uint8x8_t n0 = vmovn_u16(c0);                                           \
      uint8x8_t n1 = vmovn_u16(c1);                                           \
      uint8x16_t r = vandq_u8(vcombine_u8(n0, n1), one);                      \
      vst1q_u8(out + i, r);                                                   \
    }                                                                         \
    for (; i + 8 <= n; i += 8) {                                              \
      uint16x8_t c0 = CMP(vld1q_u16(a + i), vs);                              \
      uint8x8_t n0 = vmovn_u16(c0);                                           \
      uint8x8_t r = vand_u8(n0, vdup_n_u8(1));                                \
      vst1_u8(out + i, r);                                                    \
    }                                                                         \
    for (; i < n; i++)                                                        \
      out[i] = (uint8_t)(SCALAR_OP);                                          \
  }

STAMP_CMPSC_U16_NEON(eq, vceqq_u16, a[i] == s)
STAMP_CMPSC_U16_NEON(gt, vcgtq_u16, a[i] > s)
STAMP_CMPSC_U16_NEON(lt, vcltq_u16, a[i] < s)
STAMP_CMPSC_U16_NEON(ge, vcgeq_u16, a[i] >= s)
STAMP_CMPSC_U16_NEON(le, vcleq_u16, a[i] <= s)
#undef STAMP_CMPSC_U16_NEON

/* -- 32-bit unsigned: narrow 32→16→8 ------------------------------ */

#define STAMP_CMPSC_U32_NEON(FNAME, CMP, SCALAR_OP)                           \
  static inline void _cmpsc_##FNAME##_u32_neon(const void *restrict ap,       \
                                               const void *restrict sp,       \
                                               void *restrict op, size_t n) { \
    const uint32_t *a = (const uint32_t *)ap;                                 \
    const uint32_t s = *(const uint32_t *)sp;                                 \
    uint8_t *out = (uint8_t *)op;                                             \
    uint8x16_t one = vdupq_n_u8(1);                                           \
    uint32x4_t vs = vdupq_n_u32(s);                                           \
    size_t i = 0;                                                             \
    for (; i + 16 <= n; i += 16) {                                            \
      uint32x4_t c0 = CMP(vld1q_u32(a + i), vs);                              \
      uint32x4_t c1 = CMP(vld1q_u32(a + i + 4), vs);                          \
      uint32x4_t c2 = CMP(vld1q_u32(a + i + 8), vs);                          \
      uint32x4_t c3 = CMP(vld1q_u32(a + i + 12), vs);                         \
      uint16x4_t h0 = vmovn_u32(c0);                                          \
      uint16x4_t h1 = vmovn_u32(c1);                                          \
      uint16x4_t h2 = vmovn_u32(c2);                                          \
      uint16x4_t h3 = vmovn_u32(c3);                                          \
      uint8x8_t b0 = vmovn_u16(vcombine_u16(h0, h1));                         \
      uint8x8_t b1 = vmovn_u16(vcombine_u16(h2, h3));                         \
      uint8x16_t r = vandq_u8(vcombine_u8(b0, b1), one);                      \
      vst1q_u8(out + i, r);                                                   \
    }                                                                         \
    for (; i + 4 <= n; i += 4) {                                              \
      uint32x4_t c0 = CMP(vld1q_u32(a + i), vs);                              \
      uint16x4_t h0 = vmovn_u32(c0);                                          \
      uint8x8_t b0 = vmovn_u16(vcombine_u16(h0, vdup_n_u16(0)));              \
      uint8x8_t r = vand_u8(b0, vdup_n_u8(1));                                \
      out[i] = vget_lane_u8(r, 0);                                            \
      out[i + 1] = vget_lane_u8(r, 1);                                        \
      out[i + 2] = vget_lane_u8(r, 2);                                        \
      out[i + 3] = vget_lane_u8(r, 3);                                        \
    }                                                                         \
    for (; i < n; i++)                                                        \
      out[i] = (uint8_t)(SCALAR_OP);                                          \
  }

STAMP_CMPSC_U32_NEON(eq, vceqq_u32, a[i] == s)
STAMP_CMPSC_U32_NEON(gt, vcgtq_u32, a[i] > s)
STAMP_CMPSC_U32_NEON(lt, vcltq_u32, a[i] < s)
STAMP_CMPSC_U32_NEON(ge, vcgeq_u32, a[i] >= s)
STAMP_CMPSC_U32_NEON(le, vcleq_u32, a[i] <= s)
#undef STAMP_CMPSC_U32_NEON

/* -- 64-bit unsigned: scalar extract (2 per vector) --------------- */

#define STAMP_CMPSC_U64_NEON(FNAME, CMP, SCALAR_OP)                           \
  static inline void _cmpsc_##FNAME##_u64_neon(const void *restrict ap,       \
                                               const void *restrict sp,       \
                                               void *restrict op, size_t n) { \
    const uint64_t *a = (const uint64_t *)ap;                                 \
    const uint64_t s = *(const uint64_t *)sp;                                 \
    uint8_t *out = (uint8_t *)op;                                             \
    uint64x2_t vs = vdupq_n_u64(s);                                           \
    size_t i = 0;                                                             \
    for (; i + 2 <= n; i += 2) {                                              \
      uint64x2_t va = vld1q_u64(a + i);                                       \
      uint64x2_t c = CMP(va, vs);                                             \
      out[i] = (uint8_t)(vgetq_lane_u64(c, 0) & 1u);                          \
      out[i + 1] = (uint8_t)(vgetq_lane_u64(c, 1) & 1u);                      \
    }                                                                         \
    for (; i < n; i++)                                                        \
      out[i] = (uint8_t)(SCALAR_OP);                                          \
  }

STAMP_CMPSC_U64_NEON(eq, vceqq_u64, a[i] == s)
STAMP_CMPSC_U64_NEON(gt, vcgtq_u64, a[i] > s)
STAMP_CMPSC_U64_NEON(lt, vcltq_u64, a[i] < s)
STAMP_CMPSC_U64_NEON(ge, vcgeq_u64, a[i] >= s)
STAMP_CMPSC_U64_NEON(le, vcleq_u64, a[i] <= s)
#undef STAMP_CMPSC_U64_NEON

/* -- Float: f32 (4 per vector) ------------------------------------ */

/* Scalar helpers for tail loop */
static inline int eq_scalar_f32(float a, float b) {
  return a == b;
}
static inline int gt_scalar_f32(float a, float b) {
  return a > b;
}
static inline int lt_scalar_f32(float a, float b) {
  return a < b;
}
static inline int ge_scalar_f32(float a, float b) {
  return a >= b;
}
static inline int le_scalar_f32(float a, float b) {
  return a <= b;
}

#define CMPSC_F32_NEON(FNAME, CMP)                                            \
  static inline void _cmpsc_##FNAME##_f32_neon(const void *restrict ap,       \
                                               const void *restrict sp,       \
                                               void *restrict op, size_t n) { \
    const float *a = (const float *)ap;                                       \
    const float s = *(const float *)sp;                                       \
    uint8_t *out = (uint8_t *)op;                                             \
    float32x4_t vs = vdupq_n_f32(s);                                          \
    uint8x16_t one = vdupq_n_u8(1);                                           \
    size_t i = 0;                                                             \
    for (; i + 16 <= n; i += 16) {                                            \
      uint32x4_t c0 = CMP(vld1q_f32(a + i), vs);                              \
      uint32x4_t c1 = CMP(vld1q_f32(a + i + 4), vs);                          \
      uint32x4_t c2 = CMP(vld1q_f32(a + i + 8), vs);                          \
      uint32x4_t c3 = CMP(vld1q_f32(a + i + 12), vs);                         \
      uint16x4_t h0 = vmovn_u32(c0);                                          \
      uint16x4_t h1 = vmovn_u32(c1);                                          \
      uint16x4_t h2 = vmovn_u32(c2);                                          \
      uint16x4_t h3 = vmovn_u32(c3);                                          \
      uint8x8_t b0 = vmovn_u16(vcombine_u16(h0, h1));                         \
      uint8x8_t b1 = vmovn_u16(vcombine_u16(h2, h3));                         \
      uint8x16_t r = vandq_u8(vcombine_u8(b0, b1), one);                      \
      vst1q_u8(out + i, r);                                                   \
    }                                                                         \
    for (; i + 4 <= n; i += 4) {                                              \
      uint32x4_t c0 = CMP(vld1q_f32(a + i), vs);                              \
      uint16x4_t h0 = vmovn_u32(c0);                                          \
      uint8x8_t b0 = vmovn_u16(vcombine_u16(h0, vdup_n_u16(0)));              \
      uint8x8_t r = vand_u8(b0, vdup_n_u8(1));                                \
      out[i] = vget_lane_u8(r, 0);                                            \
      out[i + 1] = vget_lane_u8(r, 1);                                        \
      out[i + 2] = vget_lane_u8(r, 2);                                        \
      out[i + 3] = vget_lane_u8(r, 3);                                        \
    }                                                                         \
    for (; i < n; i++)                                                        \
      out[i] = (uint8_t)(FNAME##_scalar_f32(a[i], s));                        \
  }

CMPSC_F32_NEON(eq, vceqq_f32)
CMPSC_F32_NEON(gt, vcgtq_f32)
CMPSC_F32_NEON(lt, vcltq_f32)
CMPSC_F32_NEON(ge, vcgeq_f32)
CMPSC_F32_NEON(le, vcleq_f32)
#undef CMPSC_F32_NEON

/* -- Float: f64 (2 per vector) ------------------------------------ */

static inline int eq_scalar_f64(double a, double b) {
  return a == b;
}
static inline int gt_scalar_f64(double a, double b) {
  return a > b;
}
static inline int lt_scalar_f64(double a, double b) {
  return a < b;
}
static inline int ge_scalar_f64(double a, double b) {
  return a >= b;
}
static inline int le_scalar_f64(double a, double b) {
  return a <= b;
}

#define CMPSC_F64_NEON(FNAME, CMP)                                            \
  static inline void _cmpsc_##FNAME##_f64_neon(const void *restrict ap,       \
                                               const void *restrict sp,       \
                                               void *restrict op, size_t n) { \
    const double *a = (const double *)ap;                                     \
    const double s = *(const double *)sp;                                     \
    uint8_t *out = (uint8_t *)op;                                             \
    float64x2_t vs = vdupq_n_f64(s);                                          \
    size_t i = 0;                                                             \
    for (; i + 2 <= n; i += 2) {                                              \
      float64x2_t va = vld1q_f64(a + i);                                      \
      uint64x2_t c = CMP(va, vs);                                             \
      out[i] = (uint8_t)(vgetq_lane_u64(c, 0) & 1u);                          \
      out[i + 1] = (uint8_t)(vgetq_lane_u64(c, 1) & 1u);                      \
    }                                                                         \
    for (; i < n; i++)                                                        \
      out[i] = (uint8_t)(FNAME##_scalar_f64(a[i], s));                        \
  }

CMPSC_F64_NEON(eq, vceqq_f64)
CMPSC_F64_NEON(gt, vcgtq_f64)
CMPSC_F64_NEON(lt, vcltq_f64)
CMPSC_F64_NEON(ge, vcgeq_f64)
CMPSC_F64_NEON(le, vcleq_f64)
#undef CMPSC_F64_NEON

#endif /* NUMC_COMPARE_SCALAR_NEON_H */
