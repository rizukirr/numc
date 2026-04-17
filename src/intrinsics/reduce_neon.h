#ifndef NUMC_REDUCE_NEON_H
#define NUMC_REDUCE_NEON_H

#include <arm_neon.h>
#include <limits.h>
#include <stdint.h>

// clang-format off

#define _RSMIN(a, b) ((a) < (b) ? (a) : (b))
#define _RSMAX(a, b) ((a) > (b) ? (a) : (b))

/* -- 64-bit emulated min/max (NEON lacks native i64/u64 min/max) ---- */

static inline int64x2_t _neon_max_s64(int64x2_t a, int64x2_t b) {
  uint64x2_t gt = vcgtq_s64(a, b);
  return vbslq_s64(gt, a, b);
}
static inline int64x2_t _neon_min_s64(int64x2_t a, int64x2_t b) {
  uint64x2_t gt = vcgtq_s64(a, b);
  return vbslq_s64(gt, b, a);
}
static inline uint64x2_t _neon_max_u64(uint64x2_t a, uint64x2_t b) {
  uint64x2_t gt = vcgtq_u64(a, b);
  return vbslq_u64(gt, a, b);
}
static inline uint64x2_t _neon_min_u64(uint64x2_t a, uint64x2_t b) {
  uint64x2_t gt = vcgtq_u64(a, b);
  return vbslq_u64(gt, b, a);
}

/* -- 64-bit horizontal reduction helpers ---------------------------- */

static inline int64_t _hmax_s64_neon(int64x2_t v) {
  int64_t a = vgetq_lane_s64(v, 0), b = vgetq_lane_s64(v, 1);
  return a > b ? a : b;
}
static inline int64_t _hmin_s64_neon(int64x2_t v) {
  int64_t a = vgetq_lane_s64(v, 0), b = vgetq_lane_s64(v, 1);
  return a < b ? a : b;
}
static inline uint64_t _hmax_u64_neon(uint64x2_t v) {
  uint64_t a = vgetq_lane_u64(v, 0), b = vgetq_lane_u64(v, 1);
  return a > b ? a : b;
}
static inline uint64_t _hmin_u64_neon(uint64x2_t v) {
  uint64_t a = vgetq_lane_u64(v, 0), b = vgetq_lane_u64(v, 1);
  return a < b ? a : b;
}

/* -- full array min/max reduction ------------------------------------ *
 *
 * 4 vector accumulators, cleanup loop, scalar tail.
 * For 8-bit types: 16 elements per vector, 64 bytes/iteration.        */

#define DEFINE_REDUCE_FULL_NEON(NAME, CT, VT, EPV, INIT_VEC, LOAD, CMP, \
                                HREDUCE, SCMP)                           \
  static inline CT NAME(const CT *restrict a, size_t n) {                \
    VT a0 = INIT_VEC, a1 = INIT_VEC;                                    \
    VT a2 = INIT_VEC, a3 = INIT_VEC;                                    \
    size_t i = 0;                                                        \
    for (; i + 4 * EPV <= n; i += 4 * EPV) {                            \
      a0 = CMP(a0, LOAD(a + i));                                        \
      a1 = CMP(a1, LOAD(a + i + EPV));                                  \
      a2 = CMP(a2, LOAD(a + i + 2 * EPV));                              \
      a3 = CMP(a3, LOAD(a + i + 3 * EPV));                              \
    }                                                                    \
    a0 = CMP(CMP(a0, a1), CMP(a2, a3));                                 \
    for (; i + EPV <= n; i += EPV)                                       \
      a0 = CMP(a0, LOAD(a + i));                                        \
    CT result = HREDUCE(a0);                                             \
    for (; i < n; i++)                                                   \
      result = SCMP(a[i], result);                                       \
    return result;                                                       \
  }

/* -- min reductions (8/16/32-bit) ------------------------------------ */

DEFINE_REDUCE_FULL_NEON(reduce_min_i8_neon, int8_t, int8x16_t, 16,
  vdupq_n_s8(INT8_MAX),
  vld1q_s8, vminq_s8, vminvq_s8, _RSMIN)
DEFINE_REDUCE_FULL_NEON(reduce_min_u8_neon, uint8_t, uint8x16_t, 16,
  vdupq_n_u8(UINT8_MAX),
  vld1q_u8, vminq_u8, vminvq_u8, _RSMIN)
DEFINE_REDUCE_FULL_NEON(reduce_min_i16_neon, int16_t, int16x8_t, 8,
  vdupq_n_s16(INT16_MAX),
  vld1q_s16, vminq_s16, vminvq_s16, _RSMIN)
DEFINE_REDUCE_FULL_NEON(reduce_min_u16_neon, uint16_t, uint16x8_t, 8,
  vdupq_n_u16(UINT16_MAX),
  vld1q_u16, vminq_u16, vminvq_u16, _RSMIN)
DEFINE_REDUCE_FULL_NEON(reduce_min_i32_neon, int32_t, int32x4_t, 4,
  vdupq_n_s32(INT32_MAX),
  vld1q_s32, vminq_s32, vminvq_s32, _RSMIN)
DEFINE_REDUCE_FULL_NEON(reduce_min_u32_neon, uint32_t, uint32x4_t, 4,
  vdupq_n_u32(UINT32_MAX),
  vld1q_u32, vminq_u32, vminvq_u32, _RSMIN)

/* -- max reductions (8/16/32-bit) ------------------------------------ */

DEFINE_REDUCE_FULL_NEON(reduce_max_i8_neon, int8_t, int8x16_t, 16,
  vdupq_n_s8(INT8_MIN),
  vld1q_s8, vmaxq_s8, vmaxvq_s8, _RSMAX)
DEFINE_REDUCE_FULL_NEON(reduce_max_u8_neon, uint8_t, uint8x16_t, 16,
  vdupq_n_u8(0),
  vld1q_u8, vmaxq_u8, vmaxvq_u8, _RSMAX)
DEFINE_REDUCE_FULL_NEON(reduce_max_i16_neon, int16_t, int16x8_t, 8,
  vdupq_n_s16(INT16_MIN),
  vld1q_s16, vmaxq_s16, vmaxvq_s16, _RSMAX)
DEFINE_REDUCE_FULL_NEON(reduce_max_u16_neon, uint16_t, uint16x8_t, 8,
  vdupq_n_u16(0),
  vld1q_u16, vmaxq_u16, vmaxvq_u16, _RSMAX)
DEFINE_REDUCE_FULL_NEON(reduce_max_i32_neon, int32_t, int32x4_t, 4,
  vdupq_n_s32(INT32_MIN),
  vld1q_s32, vmaxq_s32, vmaxvq_s32, _RSMAX)
DEFINE_REDUCE_FULL_NEON(reduce_max_u32_neon, uint32_t, uint32x4_t, 4,
  vdupq_n_u32(0),
  vld1q_u32, vmaxq_u32, vmaxvq_u32, _RSMAX)

#undef DEFINE_REDUCE_FULL_NEON

/* -- 64-bit full array reductions ------------------------------------ */

#define DEFINE_REDUCE_FULL_64_NEON(NAME, CT, VT, INIT_VEC, LOAD, CMP, \
                                   HREDUCE, SCMP)                      \
  static inline CT NAME(const CT *restrict a, size_t n) {              \
    const size_t EPV = 2; /* 128 / 64 */                               \
    VT a0 = INIT_VEC, a1 = INIT_VEC;                                   \
    VT a2 = INIT_VEC, a3 = INIT_VEC;                                   \
    size_t i = 0;                                                      \
    for (; i + 4 * EPV <= n; i += 4 * EPV) {                           \
      a0 = CMP(a0, LOAD(a + i));                                       \
      a1 = CMP(a1, LOAD(a + i + EPV));                                 \
      a2 = CMP(a2, LOAD(a + i + 2 * EPV));                             \
      a3 = CMP(a3, LOAD(a + i + 3 * EPV));                             \
    }                                                                  \
    a0 = CMP(CMP(a0, a1), CMP(a2, a3));                                \
    for (; i + EPV <= n; i += EPV)                                     \
      a0 = CMP(a0, LOAD(a + i));                                       \
    CT result = HREDUCE(a0);                                           \
    for (; i < n; i++)                                                 \
      result = SCMP(a[i], result);                                     \
    return result;                                                     \
  }

DEFINE_REDUCE_FULL_64_NEON(reduce_min_i64_neon, int64_t, int64x2_t,
  vdupq_n_s64(INT64_MAX),
  vld1q_s64, _neon_min_s64, _hmin_s64_neon, _RSMIN)
DEFINE_REDUCE_FULL_64_NEON(reduce_min_u64_neon, uint64_t, uint64x2_t,
  vdupq_n_u64(UINT64_MAX),
  vld1q_u64, _neon_min_u64, _hmin_u64_neon, _RSMIN)
DEFINE_REDUCE_FULL_64_NEON(reduce_max_i64_neon, int64_t, int64x2_t,
  vdupq_n_s64(INT64_MIN),
  vld1q_s64, _neon_max_s64, _hmax_s64_neon, _RSMAX)
DEFINE_REDUCE_FULL_64_NEON(reduce_max_u64_neon, uint64_t, uint64x2_t,
  vdupq_n_u64(0),
  vld1q_u64, _neon_max_u64, _hmax_u64_neon, _RSMAX)

#undef DEFINE_REDUCE_FULL_64_NEON

/* -- fused row-reduce (axis-1 reduction) ----------------------------- *
 *
 * Processes 4 rows at a time, vectorizes inner column loop.
 * For int8: 16 columns per SIMD iteration.                            */

#define DEFINE_FUSED_REDUCE_NEON(NAME, CT, VT, EPV, LOAD, STORE, CMP, SCMP) \
  static inline void NAME(const char *restrict base, intptr_t row_stride,    \
                           size_t nrows, char *restrict dst,                  \
                           size_t ncols) {                                    \
    CT *restrict d = (CT *)dst;                                              \
    size_t r = 0;                                                            \
    for (; r + 4 <= nrows; r += 4) {                                         \
      const CT *restrict s0 = (const CT *)(base + r * row_stride);           \
      const CT *restrict s1 =                                                \
          (const CT *)(base + (r + 1) * row_stride);                         \
      const CT *restrict s2 =                                                \
          (const CT *)(base + (r + 2) * row_stride);                         \
      const CT *restrict s3 =                                                \
          (const CT *)(base + (r + 3) * row_stride);                         \
      size_t i = 0;                                                          \
      for (; i + EPV <= ncols; i += EPV) {                                   \
        VT dv  = LOAD(d + i);                                               \
        VT v01 = CMP(LOAD(s0 + i), LOAD(s1 + i));                           \
        VT v23 = CMP(LOAD(s2 + i), LOAD(s3 + i));                           \
        STORE(d + i, CMP(dv, CMP(v01, v23)));                               \
      }                                                                      \
      for (; i < ncols; i++) {                                               \
        CT v = SCMP(s0[i], s1[i]);                                           \
        v = SCMP(v, s2[i]);                                                  \
        v = SCMP(v, s3[i]);                                                  \
        d[i] = SCMP(v, d[i]);                                               \
      }                                                                      \
    }                                                                        \
    for (; r < nrows; r++) {                                                 \
      const CT *restrict s =                                                 \
          (const CT *)(base + r * row_stride);                               \
      size_t i = 0;                                                          \
      for (; i + EPV <= ncols; i += EPV)                                     \
        STORE(d + i, CMP(LOAD(d + i), LOAD(s + i)));                        \
      for (; i < ncols; i++)                                                 \
        d[i] = SCMP(s[i], d[i]);                                            \
    }                                                                        \
  }

/* min fused (8/16/32-bit) */
DEFINE_FUSED_REDUCE_NEON(_min_fused_i8_neon, int8_t, int8x16_t, 16,
  vld1q_s8, vst1q_s8, vminq_s8, _RSMIN)
DEFINE_FUSED_REDUCE_NEON(_min_fused_u8_neon, uint8_t, uint8x16_t, 16,
  vld1q_u8, vst1q_u8, vminq_u8, _RSMIN)
DEFINE_FUSED_REDUCE_NEON(_min_fused_i16_neon, int16_t, int16x8_t, 8,
  vld1q_s16, vst1q_s16, vminq_s16, _RSMIN)
DEFINE_FUSED_REDUCE_NEON(_min_fused_u16_neon, uint16_t, uint16x8_t, 8,
  vld1q_u16, vst1q_u16, vminq_u16, _RSMIN)
DEFINE_FUSED_REDUCE_NEON(_min_fused_i32_neon, int32_t, int32x4_t, 4,
  vld1q_s32, vst1q_s32, vminq_s32, _RSMIN)
DEFINE_FUSED_REDUCE_NEON(_min_fused_u32_neon, uint32_t, uint32x4_t, 4,
  vld1q_u32, vst1q_u32, vminq_u32, _RSMIN)

/* max fused (8/16/32-bit) */
DEFINE_FUSED_REDUCE_NEON(_max_fused_i8_neon, int8_t, int8x16_t, 16,
  vld1q_s8, vst1q_s8, vmaxq_s8, _RSMAX)
DEFINE_FUSED_REDUCE_NEON(_max_fused_u8_neon, uint8_t, uint8x16_t, 16,
  vld1q_u8, vst1q_u8, vmaxq_u8, _RSMAX)
DEFINE_FUSED_REDUCE_NEON(_max_fused_i16_neon, int16_t, int16x8_t, 8,
  vld1q_s16, vst1q_s16, vmaxq_s16, _RSMAX)
DEFINE_FUSED_REDUCE_NEON(_max_fused_u16_neon, uint16_t, uint16x8_t, 8,
  vld1q_u16, vst1q_u16, vmaxq_u16, _RSMAX)
DEFINE_FUSED_REDUCE_NEON(_max_fused_i32_neon, int32_t, int32x4_t, 4,
  vld1q_s32, vst1q_s32, vmaxq_s32, _RSMAX)
DEFINE_FUSED_REDUCE_NEON(_max_fused_u32_neon, uint32_t, uint32x4_t, 4,
  vld1q_u32, vst1q_u32, vmaxq_u32, _RSMAX)

#undef DEFINE_FUSED_REDUCE_NEON

/* -- 64-bit fused row-reduce (axis-1) -------------------------------- */

#define DEFINE_FUSED_64_NEON(NAME, CT, VT, LOAD, STORE, CMP, SCMP)          \
  static inline void NAME(const char *restrict base, intptr_t row_stride,    \
                           size_t nrows, char *restrict dst,                  \
                           size_t ncols) {                                    \
    CT *restrict d = (CT *)dst;                                              \
    const size_t EPV = 2;                                                    \
    size_t r = 0;                                                            \
    for (; r + 4 <= nrows; r += 4) {                                         \
      const CT *restrict s0 = (const CT *)(base + r * row_stride);           \
      const CT *restrict s1 =                                                \
          (const CT *)(base + (r + 1) * row_stride);                         \
      const CT *restrict s2 =                                                \
          (const CT *)(base + (r + 2) * row_stride);                         \
      const CT *restrict s3 =                                                \
          (const CT *)(base + (r + 3) * row_stride);                         \
      size_t i = 0;                                                          \
      for (; i + EPV <= ncols; i += EPV) {                                   \
        VT dv  = LOAD(d + i);                                               \
        VT v01 = CMP(LOAD(s0 + i), LOAD(s1 + i));                           \
        VT v23 = CMP(LOAD(s2 + i), LOAD(s3 + i));                           \
        STORE(d + i, CMP(dv, CMP(v01, v23)));                               \
      }                                                                      \
      for (; i < ncols; i++) {                                               \
        CT v = SCMP(s0[i], s1[i]);                                           \
        v = SCMP(v, s2[i]);                                                  \
        v = SCMP(v, s3[i]);                                                  \
        d[i] = SCMP(v, d[i]);                                               \
      }                                                                      \
    }                                                                        \
    for (; r < nrows; r++) {                                                 \
      const CT *restrict s =                                                 \
          (const CT *)(base + r * row_stride);                               \
      size_t i = 0;                                                          \
      for (; i + EPV <= ncols; i += EPV)                                     \
        STORE(d + i, CMP(LOAD(d + i), LOAD(s + i)));                        \
      for (; i < ncols; i++)                                                 \
        d[i] = SCMP(s[i], d[i]);                                            \
    }                                                                        \
  }

DEFINE_FUSED_64_NEON(_min_fused_i64_neon, int64_t, int64x2_t,
  vld1q_s64, vst1q_s64, _neon_min_s64, _RSMIN)
DEFINE_FUSED_64_NEON(_min_fused_u64_neon, uint64_t, uint64x2_t,
  vld1q_u64, vst1q_u64, _neon_min_u64, _RSMIN)
DEFINE_FUSED_64_NEON(_max_fused_i64_neon, int64_t, int64x2_t,
  vld1q_s64, vst1q_s64, _neon_max_s64, _RSMAX)
DEFINE_FUSED_64_NEON(_max_fused_u64_neon, uint64_t, uint64x2_t,
  vld1q_u64, vst1q_u64, _neon_max_u64, _RSMAX)

#undef DEFINE_FUSED_64_NEON

// clang-format on

#undef _RSMIN
#undef _RSMAX

#endif
