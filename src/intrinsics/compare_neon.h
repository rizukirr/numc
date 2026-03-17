/**
 * @file compare_neon.h
 * @brief NEON binary comparison kernels — uint8 output (0/1).
 *
 * All comparison functions output uint8_t* (NumPy-compatible bool).
 * NEON comparison intrinsics return all-ones masks in the element width;
 * we narrow the result down to uint8 and AND with 1 before storing.
 *
 * 8-bit:  compare → AND 1 → vst1q_u8 (16 elements)
 * 16-bit: compare 2×8 → vmovn_u16 each → vcombine → AND 1 → vst1q_u8 (16)
 * 32-bit: compare 4×4 → vmovn chain → AND 1 → vst1q_u8 (16)
 * 64-bit: compare → scalar tail (2 elems per vector, not worth narrowing)
 */
#ifndef NUMC_COMPARE_NEON_H
#define NUMC_COMPARE_NEON_H

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

/* ════════════════════════════════════════════════════════════════════
 * 8-bit signed integer: 16 elems → 16 uint8 output per vector
 * ════════════════════════════════════════════════════════════════ */

#define FAST_CMP_S8_NEON(NAME, CMP)                                        \
  static inline void _fast_##NAME##_i8_neon(const void *restrict ap,       \
                                            const void *restrict bp,       \
                                            void *restrict op, size_t n) { \
    const int8_t *a = (const int8_t *)ap;                                  \
    const int8_t *b = (const int8_t *)bp;                                  \
    uint8_t *out = (uint8_t *)op;                                          \
    const uint8x16_t one = vdupq_n_u8(1);                                  \
    size_t i = 0;                                                          \
    for (; i + 16 <= n; i += 16) {                                         \
      int8x16_t va = vld1q_s8(a + i);                                      \
      int8x16_t vb = vld1q_s8(b + i);                                      \
      uint8x16_t r = vandq_u8(CMP(va, vb), one);                           \
      vst1q_u8(out + i, r);                                                \
    }                                                                      \
    for (; i < n; i++)                                                     \
      out[i] = (uint8_t)(a[i] NAME##_OP b[i]);                             \
  }

/* We need scalar ops but can't put operators in macro name easily,
   so use dedicated macros per op */
#undef FAST_CMP_S8_NEON

#define FAST_CMP_I8_NEON(FNAME, CMP, SCALAR_OP)                             \
  static inline void _fast_##FNAME##_i8_neon(const void *restrict ap,       \
                                             const void *restrict bp,       \
                                             void *restrict op, size_t n) { \
    const int8_t *a = (const int8_t *)ap;                                   \
    const int8_t *b = (const int8_t *)bp;                                   \
    uint8_t *out = (uint8_t *)op;                                           \
    const uint8x16_t one = vdupq_n_u8(1);                                   \
    size_t i = 0;                                                           \
    for (; i + 16 <= n; i += 16) {                                          \
      int8x16_t va = vld1q_s8(a + i);                                       \
      int8x16_t vb = vld1q_s8(b + i);                                       \
      uint8x16_t r = vandq_u8(CMP(va, vb), one);                            \
      vst1q_u8(out + i, r);                                                 \
    }                                                                       \
    for (; i < n; i++)                                                      \
      out[i] = (uint8_t)(SCALAR_OP);                                        \
  }

FAST_CMP_I8_NEON(eq, vceqq_s8, a[i] == b[i])
FAST_CMP_I8_NEON(gt, vcgtq_s8, a[i] > b[i])
FAST_CMP_I8_NEON(lt, vcltq_s8, a[i] < b[i])
FAST_CMP_I8_NEON(ge, vcgeq_s8, a[i] >= b[i])
FAST_CMP_I8_NEON(le, vcleq_s8, a[i] <= b[i])
#undef FAST_CMP_I8_NEON

/* ════════════════════════════════════════════════════════════════════
 * 16-bit signed integer: 2×8 → narrow → 16 uint8 output
 * ════════════════════════════════════════════════════════════════ */

#define FAST_CMP_I16_NEON(FNAME, CMP, SCALAR_OP)                             \
  static inline void _fast_##FNAME##_i16_neon(const void *restrict ap,       \
                                              const void *restrict bp,       \
                                              void *restrict op, size_t n) { \
    const int16_t *a = (const int16_t *)ap;                                  \
    const int16_t *b = (const int16_t *)bp;                                  \
    uint8_t *out = (uint8_t *)op;                                            \
    const uint8x16_t one = vdupq_n_u8(1);                                    \
    size_t i = 0;                                                            \
    for (; i + 16 <= n; i += 16) {                                           \
      int16x8_t va0 = vld1q_s16(a + i);                                      \
      int16x8_t vb0 = vld1q_s16(b + i);                                      \
      int16x8_t va1 = vld1q_s16(a + i + 8);                                  \
      int16x8_t vb1 = vld1q_s16(b + i + 8);                                  \
      uint16x8_t c0 = CMP(va0, vb0);                                         \
      uint16x8_t c1 = CMP(va1, vb1);                                         \
      /* narrow 16→8: take low byte of each u16 */                           \
      uint8x8_t n0 = vmovn_u16(c0);                                          \
      uint8x8_t n1 = vmovn_u16(c1);                                          \
      uint8x16_t r = vandq_u8(vcombine_u8(n0, n1), one);                     \
      vst1q_u8(out + i, r);                                                  \
    }                                                                        \
    for (; i + 8 <= n; i += 8) {                                             \
      int16x8_t va0 = vld1q_s16(a + i);                                      \
      int16x8_t vb0 = vld1q_s16(b + i);                                      \
      uint16x8_t c0 = CMP(va0, vb0);                                         \
      uint8x8_t n0 = vmovn_u16(c0);                                          \
      uint8x8_t r = vand_u8(n0, vdup_n_u8(1));                               \
      vst1_u8(out + i, r);                                                   \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (uint8_t)(SCALAR_OP);                                         \
  }

FAST_CMP_I16_NEON(eq, vceqq_s16, a[i] == b[i])
FAST_CMP_I16_NEON(gt, vcgtq_s16, a[i] > b[i])
FAST_CMP_I16_NEON(lt, vcltq_s16, a[i] < b[i])
FAST_CMP_I16_NEON(ge, vcgeq_s16, a[i] >= b[i])
FAST_CMP_I16_NEON(le, vcleq_s16, a[i] <= b[i])
#undef FAST_CMP_I16_NEON

/* ════════════════════════════════════════════════════════════════════
 * 32-bit signed integer: 4×4 → narrow chain → 16 uint8 output
 * ════════════════════════════════════════════════════════════════ */

#define FAST_CMP_I32_NEON(FNAME, CMP, SCALAR_OP)                             \
  static inline void _fast_##FNAME##_i32_neon(const void *restrict ap,       \
                                              const void *restrict bp,       \
                                              void *restrict op, size_t n) { \
    const int32_t *a = (const int32_t *)ap;                                  \
    const int32_t *b = (const int32_t *)bp;                                  \
    uint8_t *out = (uint8_t *)op;                                            \
    const uint8x16_t one = vdupq_n_u8(1);                                    \
    size_t i = 0;                                                            \
    for (; i + 16 <= n; i += 16) {                                           \
      uint32x4_t c0 = CMP(vld1q_s32(a + i), vld1q_s32(b + i));               \
      uint32x4_t c1 = CMP(vld1q_s32(a + i + 4), vld1q_s32(b + i + 4));       \
      uint32x4_t c2 = CMP(vld1q_s32(a + i + 8), vld1q_s32(b + i + 8));       \
      uint32x4_t c3 = CMP(vld1q_s32(a + i + 12), vld1q_s32(b + i + 12));     \
      /* 32→16 */                                                            \
      uint16x4_t h0 = vmovn_u32(c0);                                         \
      uint16x4_t h1 = vmovn_u32(c1);                                         \
      uint16x4_t h2 = vmovn_u32(c2);                                         \
      uint16x4_t h3 = vmovn_u32(c3);                                         \
      /* 16→8 */                                                             \
      uint8x8_t b0 = vmovn_u16(vcombine_u16(h0, h1));                        \
      uint8x8_t b1 = vmovn_u16(vcombine_u16(h2, h3));                        \
      uint8x16_t r = vandq_u8(vcombine_u8(b0, b1), one);                     \
      vst1q_u8(out + i, r);                                                  \
    }                                                                        \
    for (; i + 4 <= n; i += 4) {                                             \
      uint32x4_t c0 = CMP(vld1q_s32(a + i), vld1q_s32(b + i));               \
      uint16x4_t h0 = vmovn_u32(c0);                                         \
      /* combine with zeros to form uint16x8, then narrow */                 \
      uint8x8_t b0 = vmovn_u16(vcombine_u16(h0, vdup_n_u16(0)));             \
      /* only lower 4 bytes valid */                                         \
      uint8x8_t r = vand_u8(b0, vdup_n_u8(1));                               \
      out[i] = vget_lane_u8(r, 0);                                           \
      out[i + 1] = vget_lane_u8(r, 1);                                       \
      out[i + 2] = vget_lane_u8(r, 2);                                       \
      out[i + 3] = vget_lane_u8(r, 3);                                       \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (uint8_t)(SCALAR_OP);                                         \
  }

FAST_CMP_I32_NEON(eq, vceqq_s32, a[i] == b[i])
FAST_CMP_I32_NEON(gt, vcgtq_s32, a[i] > b[i])
FAST_CMP_I32_NEON(lt, vcltq_s32, a[i] < b[i])
FAST_CMP_I32_NEON(ge, vcgeq_s32, a[i] >= b[i])
FAST_CMP_I32_NEON(le, vcleq_s32, a[i] <= b[i])
#undef FAST_CMP_I32_NEON

/* ════════════════════════════════════════════════════════════════════
 * 64-bit signed integer: scalar output (2 per vector, narrow not worth it)
 * ════════════════════════════════════════════════════════════════ */

#define FAST_CMP_I64_NEON(FNAME, CMP, SCALAR_OP)                             \
  static inline void _fast_##FNAME##_i64_neon(const void *restrict ap,       \
                                              const void *restrict bp,       \
                                              void *restrict op, size_t n) { \
    const int64_t *a = (const int64_t *)ap;                                  \
    const int64_t *b = (const int64_t *)bp;                                  \
    uint8_t *out = (uint8_t *)op;                                            \
    size_t i = 0;                                                            \
    for (; i + 2 <= n; i += 2) {                                             \
      int64x2_t va = vld1q_s64(a + i);                                       \
      int64x2_t vb = vld1q_s64(b + i);                                       \
      uint64x2_t c = CMP(va, vb);                                            \
      out[i] = (uint8_t)(vgetq_lane_u64(c, 0) & 1u);                         \
      out[i + 1] = (uint8_t)(vgetq_lane_u64(c, 1) & 1u);                     \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (uint8_t)(SCALAR_OP);                                         \
  }

FAST_CMP_I64_NEON(eq, vceqq_s64, a[i] == b[i])
FAST_CMP_I64_NEON(gt, vcgtq_s64, a[i] > b[i])
FAST_CMP_I64_NEON(lt, vcltq_s64, a[i] < b[i])
FAST_CMP_I64_NEON(ge, vcgeq_s64, a[i] >= b[i])
FAST_CMP_I64_NEON(le, vcleq_s64, a[i] <= b[i])
#undef FAST_CMP_I64_NEON

/* ════════════════════════════════════════════════════════════════════
 * 8-bit unsigned integer: 16 elems → 16 uint8 output per vector
 * ════════════════════════════════════════════════════════════════ */

#define FAST_CMP_U8_NEON(FNAME, CMP, SCALAR_OP)                             \
  static inline void _fast_##FNAME##_u8_neon(const void *restrict ap,       \
                                             const void *restrict bp,       \
                                             void *restrict op, size_t n) { \
    const uint8_t *a = (const uint8_t *)ap;                                 \
    const uint8_t *b = (const uint8_t *)bp;                                 \
    uint8_t *out = (uint8_t *)op;                                           \
    const uint8x16_t one = vdupq_n_u8(1);                                   \
    size_t i = 0;                                                           \
    for (; i + 16 <= n; i += 16) {                                          \
      uint8x16_t va = vld1q_u8(a + i);                                      \
      uint8x16_t vb = vld1q_u8(b + i);                                      \
      uint8x16_t r = vandq_u8(CMP(va, vb), one);                            \
      vst1q_u8(out + i, r);                                                 \
    }                                                                       \
    for (; i < n; i++)                                                      \
      out[i] = (uint8_t)(SCALAR_OP);                                        \
  }

FAST_CMP_U8_NEON(eq, vceqq_u8, a[i] == b[i])
FAST_CMP_U8_NEON(gt, vcgtq_u8, a[i] > b[i])
FAST_CMP_U8_NEON(lt, vcltq_u8, a[i] < b[i])
FAST_CMP_U8_NEON(ge, vcgeq_u8, a[i] >= b[i])
FAST_CMP_U8_NEON(le, vcleq_u8, a[i] <= b[i])
#undef FAST_CMP_U8_NEON

/* ════════════════════════════════════════════════════════════════════
 * 16-bit unsigned integer: 2×8 → narrow → 16 uint8 output
 * ════════════════════════════════════════════════════════════════ */

#define FAST_CMP_U16_NEON(FNAME, CMP, SCALAR_OP)                             \
  static inline void _fast_##FNAME##_u16_neon(const void *restrict ap,       \
                                              const void *restrict bp,       \
                                              void *restrict op, size_t n) { \
    const uint16_t *a = (const uint16_t *)ap;                                \
    const uint16_t *b = (const uint16_t *)bp;                                \
    uint8_t *out = (uint8_t *)op;                                            \
    const uint8x16_t one = vdupq_n_u8(1);                                    \
    size_t i = 0;                                                            \
    for (; i + 16 <= n; i += 16) {                                           \
      uint16x8_t va0 = vld1q_u16(a + i);                                     \
      uint16x8_t vb0 = vld1q_u16(b + i);                                     \
      uint16x8_t va1 = vld1q_u16(a + i + 8);                                 \
      uint16x8_t vb1 = vld1q_u16(b + i + 8);                                 \
      uint16x8_t c0 = CMP(va0, vb0);                                         \
      uint16x8_t c1 = CMP(va1, vb1);                                         \
      uint8x8_t n0 = vmovn_u16(c0);                                          \
      uint8x8_t n1 = vmovn_u16(c1);                                          \
      uint8x16_t r = vandq_u8(vcombine_u8(n0, n1), one);                     \
      vst1q_u8(out + i, r);                                                  \
    }                                                                        \
    for (; i + 8 <= n; i += 8) {                                             \
      uint16x8_t va0 = vld1q_u16(a + i);                                     \
      uint16x8_t vb0 = vld1q_u16(b + i);                                     \
      uint16x8_t c0 = CMP(va0, vb0);                                         \
      uint8x8_t n0 = vmovn_u16(c0);                                          \
      uint8x8_t r = vand_u8(n0, vdup_n_u8(1));                               \
      vst1_u8(out + i, r);                                                   \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (uint8_t)(SCALAR_OP);                                         \
  }

FAST_CMP_U16_NEON(eq, vceqq_u16, a[i] == b[i])
FAST_CMP_U16_NEON(gt, vcgtq_u16, a[i] > b[i])
FAST_CMP_U16_NEON(lt, vcltq_u16, a[i] < b[i])
FAST_CMP_U16_NEON(ge, vcgeq_u16, a[i] >= b[i])
FAST_CMP_U16_NEON(le, vcleq_u16, a[i] <= b[i])
#undef FAST_CMP_U16_NEON

/* ════════════════════════════════════════════════════════════════════
 * 32-bit unsigned integer: 4×4 → narrow chain → 16 uint8 output
 * ════════════════════════════════════════════════════════════════ */

#define FAST_CMP_U32_NEON(FNAME, CMP, SCALAR_OP)                             \
  static inline void _fast_##FNAME##_u32_neon(const void *restrict ap,       \
                                              const void *restrict bp,       \
                                              void *restrict op, size_t n) { \
    const uint32_t *a = (const uint32_t *)ap;                                \
    const uint32_t *b = (const uint32_t *)bp;                                \
    uint8_t *out = (uint8_t *)op;                                            \
    const uint8x16_t one = vdupq_n_u8(1);                                    \
    size_t i = 0;                                                            \
    for (; i + 16 <= n; i += 16) {                                           \
      uint32x4_t c0 = CMP(vld1q_u32(a + i), vld1q_u32(b + i));               \
      uint32x4_t c1 = CMP(vld1q_u32(a + i + 4), vld1q_u32(b + i + 4));       \
      uint32x4_t c2 = CMP(vld1q_u32(a + i + 8), vld1q_u32(b + i + 8));       \
      uint32x4_t c3 = CMP(vld1q_u32(a + i + 12), vld1q_u32(b + i + 12));     \
      uint16x4_t h0 = vmovn_u32(c0);                                         \
      uint16x4_t h1 = vmovn_u32(c1);                                         \
      uint16x4_t h2 = vmovn_u32(c2);                                         \
      uint16x4_t h3 = vmovn_u32(c3);                                         \
      uint8x8_t b0 = vmovn_u16(vcombine_u16(h0, h1));                        \
      uint8x8_t b1 = vmovn_u16(vcombine_u16(h2, h3));                        \
      uint8x16_t r = vandq_u8(vcombine_u8(b0, b1), one);                     \
      vst1q_u8(out + i, r);                                                  \
    }                                                                        \
    for (; i + 4 <= n; i += 4) {                                             \
      uint32x4_t c0 = CMP(vld1q_u32(a + i), vld1q_u32(b + i));               \
      uint16x4_t h0 = vmovn_u32(c0);                                         \
      uint8x8_t b0 = vmovn_u16(vcombine_u16(h0, vdup_n_u16(0)));             \
      uint8x8_t r = vand_u8(b0, vdup_n_u8(1));                               \
      out[i] = vget_lane_u8(r, 0);                                           \
      out[i + 1] = vget_lane_u8(r, 1);                                       \
      out[i + 2] = vget_lane_u8(r, 2);                                       \
      out[i + 3] = vget_lane_u8(r, 3);                                       \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (uint8_t)(SCALAR_OP);                                         \
  }

FAST_CMP_U32_NEON(eq, vceqq_u32, a[i] == b[i])
FAST_CMP_U32_NEON(gt, vcgtq_u32, a[i] > b[i])
FAST_CMP_U32_NEON(lt, vcltq_u32, a[i] < b[i])
FAST_CMP_U32_NEON(ge, vcgeq_u32, a[i] >= b[i])
FAST_CMP_U32_NEON(le, vcleq_u32, a[i] <= b[i])
#undef FAST_CMP_U32_NEON

/* ════════════════════════════════════════════════════════════════════
 * 64-bit unsigned integer: scalar output (2 per vector)
 * ════════════════════════════════════════════════════════════════ */

#define FAST_CMP_U64_NEON(FNAME, CMP, SCALAR_OP)                             \
  static inline void _fast_##FNAME##_u64_neon(const void *restrict ap,       \
                                              const void *restrict bp,       \
                                              void *restrict op, size_t n) { \
    const uint64_t *a = (const uint64_t *)ap;                                \
    const uint64_t *b = (const uint64_t *)bp;                                \
    uint8_t *out = (uint8_t *)op;                                            \
    size_t i = 0;                                                            \
    for (; i + 2 <= n; i += 2) {                                             \
      uint64x2_t va = vld1q_u64(a + i);                                      \
      uint64x2_t vb = vld1q_u64(b + i);                                      \
      uint64x2_t c = CMP(va, vb);                                            \
      out[i] = (uint8_t)(vgetq_lane_u64(c, 0) & 1u);                         \
      out[i + 1] = (uint8_t)(vgetq_lane_u64(c, 1) & 1u);                     \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (uint8_t)(SCALAR_OP);                                         \
  }

FAST_CMP_U64_NEON(eq, vceqq_u64, a[i] == b[i])
FAST_CMP_U64_NEON(gt, vcgtq_u64, a[i] > b[i])
FAST_CMP_U64_NEON(lt, vcltq_u64, a[i] < b[i])
FAST_CMP_U64_NEON(ge, vcgeq_u64, a[i] >= b[i])
FAST_CMP_U64_NEON(le, vcleq_u64, a[i] <= b[i])
#undef FAST_CMP_U64_NEON

/* ════════════════════════════════════════════════════════════════════
 * Float32: 4×4 → narrow chain → 16 uint8 output
 * ════════════════════════════════════════════════════════════════ */

#define FAST_CMP_F32_NEON(FNAME, CMP, SCALAR_OP)                             \
  static inline void _fast_##FNAME##_f32_neon(const void *restrict ap,       \
                                              const void *restrict bp,       \
                                              void *restrict op, size_t n) { \
    const float *a = (const float *)ap;                                      \
    const float *b = (const float *)bp;                                      \
    uint8_t *out = (uint8_t *)op;                                            \
    const uint8x16_t one = vdupq_n_u8(1);                                    \
    size_t i = 0;                                                            \
    for (; i + 16 <= n; i += 16) {                                           \
      uint32x4_t c0 = CMP(vld1q_f32(a + i), vld1q_f32(b + i));               \
      uint32x4_t c1 = CMP(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));       \
      uint32x4_t c2 = CMP(vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));       \
      uint32x4_t c3 = CMP(vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));     \
      uint16x4_t h0 = vmovn_u32(c0);                                         \
      uint16x4_t h1 = vmovn_u32(c1);                                         \
      uint16x4_t h2 = vmovn_u32(c2);                                         \
      uint16x4_t h3 = vmovn_u32(c3);                                         \
      uint8x8_t b0 = vmovn_u16(vcombine_u16(h0, h1));                        \
      uint8x8_t b1 = vmovn_u16(vcombine_u16(h2, h3));                        \
      uint8x16_t r = vandq_u8(vcombine_u8(b0, b1), one);                     \
      vst1q_u8(out + i, r);                                                  \
    }                                                                        \
    for (; i + 4 <= n; i += 4) {                                             \
      uint32x4_t c0 = CMP(vld1q_f32(a + i), vld1q_f32(b + i));               \
      uint16x4_t h0 = vmovn_u32(c0);                                         \
      uint8x8_t b0 = vmovn_u16(vcombine_u16(h0, vdup_n_u16(0)));             \
      uint8x8_t r = vand_u8(b0, vdup_n_u8(1));                               \
      out[i] = vget_lane_u8(r, 0);                                           \
      out[i + 1] = vget_lane_u8(r, 1);                                       \
      out[i + 2] = vget_lane_u8(r, 2);                                       \
      out[i + 3] = vget_lane_u8(r, 3);                                       \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (uint8_t)(SCALAR_OP);                                         \
  }

FAST_CMP_F32_NEON(eq, vceqq_f32, a[i] == b[i])
FAST_CMP_F32_NEON(gt, vcgtq_f32, a[i] > b[i])
FAST_CMP_F32_NEON(lt, vcltq_f32, a[i] < b[i])
FAST_CMP_F32_NEON(ge, vcgeq_f32, a[i] >= b[i])
FAST_CMP_F32_NEON(le, vcleq_f32, a[i] <= b[i])
#undef FAST_CMP_F32_NEON

/* ════════════════════════════════════════════════════════════════════
 * Float64: scalar output (2 per vector)
 * ════════════════════════════════════════════════════════════════ */

#define FAST_CMP_F64_NEON(FNAME, CMP, SCALAR_OP)                             \
  static inline void _fast_##FNAME##_f64_neon(const void *restrict ap,       \
                                              const void *restrict bp,       \
                                              void *restrict op, size_t n) { \
    const double *a = (const double *)ap;                                    \
    const double *b = (const double *)bp;                                    \
    uint8_t *out = (uint8_t *)op;                                            \
    size_t i = 0;                                                            \
    for (; i + 2 <= n; i += 2) {                                             \
      float64x2_t va = vld1q_f64(a + i);                                     \
      float64x2_t vb = vld1q_f64(b + i);                                     \
      uint64x2_t c = CMP(va, vb);                                            \
      out[i] = (uint8_t)(vgetq_lane_u64(c, 0) & 1u);                         \
      out[i + 1] = (uint8_t)(vgetq_lane_u64(c, 1) & 1u);                     \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (uint8_t)(SCALAR_OP);                                         \
  }

FAST_CMP_F64_NEON(eq, vceqq_f64, a[i] == b[i])
FAST_CMP_F64_NEON(gt, vcgtq_f64, a[i] > b[i])
FAST_CMP_F64_NEON(lt, vcltq_f64, a[i] < b[i])
FAST_CMP_F64_NEON(ge, vcgeq_f64, a[i] >= b[i])
FAST_CMP_F64_NEON(le, vcleq_f64, a[i] <= b[i])
#undef FAST_CMP_F64_NEON

/* ════════════════════════════════════════════════════════════════════
 * Legacy u8-only wrappers (old API: typed pointers, not void *)
 * ════════════════════════════════════════════════════════════════ */

static inline void _cmp_eq_u8_neon(const uint8_t *restrict a,
                                   const uint8_t *restrict b,
                                   uint8_t *restrict out, size_t n) {
  _fast_eq_u8_neon(a, b, out, n);
}
static inline void _cmp_gt_u8_neon(const uint8_t *restrict a,
                                   const uint8_t *restrict b,
                                   uint8_t *restrict out, size_t n) {
  _fast_gt_u8_neon(a, b, out, n);
}
static inline void _cmp_lt_u8_neon(const uint8_t *restrict a,
                                   const uint8_t *restrict b,
                                   uint8_t *restrict out, size_t n) {
  _fast_lt_u8_neon(a, b, out, n);
}
static inline void _cmp_ge_u8_neon(const uint8_t *restrict a,
                                   const uint8_t *restrict b,
                                   uint8_t *restrict out, size_t n) {
  _fast_ge_u8_neon(a, b, out, n);
}
static inline void _cmp_le_u8_neon(const uint8_t *restrict a,
                                   const uint8_t *restrict b,
                                   uint8_t *restrict out, size_t n) {
  _fast_le_u8_neon(a, b, out, n);
}

#endif /* NUMC_COMPARE_NEON_H */
