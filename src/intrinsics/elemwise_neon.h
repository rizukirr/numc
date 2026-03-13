/**
 * @file elemwise_neon.h
 * @brief NEON element-wise binary/unary kernels for all 10 types.
 *
 * Binary: sub, mul, maximum, minimum
 * Unary: neg, abs
 *
 * Special cases: i64/u64 mul (scalar, no native NEON instruction),
 * i64/u64 max/min (emulated via vcgtq + vbslq).
 */
#ifndef NUMC_ELEMWISE_NEON_H
#define NUMC_ELEMWISE_NEON_H

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

/* ════════════════════════════════════════════════════════════════════
 * Binary: generic macros for signed int, unsigned int, float
 * ════════════════════════════════════════════════════════════════ */

#define FAST_BIN_SINT_NEON(OP, SFX, CT, W, VPV, VEC_OP, TAIL_EXPR)    \
  static inline void _fast_##OP##_##SFX##_neon(                        \
      const void *restrict ap, const void *restrict bp,                \
      void *restrict op, size_t n) {                                   \
    const CT *a = (const CT *)ap;                                      \
    const CT *b = (const CT *)bp;                                      \
    CT *out = (CT *)op;                                                \
    size_t i = 0;                                                      \
    for (; i + (VPV) <= n; i += (VPV)) {                               \
      int##W##x##VPV##_t va = vld1q_s##W(a + i);                      \
      int##W##x##VPV##_t vb = vld1q_s##W(b + i);                      \
      vst1q_s##W(out + i, VEC_OP(va, vb));                             \
    }                                                                  \
    for (; i < n; i++)                                                 \
      out[i] = (CT)(TAIL_EXPR);                                       \
  }

#define FAST_BIN_UINT_NEON(OP, SFX, CT, W, VPV, VEC_OP, TAIL_EXPR)    \
  static inline void _fast_##OP##_##SFX##_neon(                        \
      const void *restrict ap, const void *restrict bp,                \
      void *restrict op, size_t n) {                                   \
    const CT *a = (const CT *)ap;                                      \
    const CT *b = (const CT *)bp;                                      \
    CT *out = (CT *)op;                                                \
    size_t i = 0;                                                      \
    for (; i + (VPV) <= n; i += (VPV)) {                               \
      uint##W##x##VPV##_t va = vld1q_u##W(a + i);                     \
      uint##W##x##VPV##_t vb = vld1q_u##W(b + i);                     \
      vst1q_u##W(out + i, VEC_OP(va, vb));                             \
    }                                                                  \
    for (; i < n; i++)                                                 \
      out[i] = (CT)(TAIL_EXPR);                                       \
  }

#define FAST_BIN_F32_NEON(OP, VEC_OP, TAIL_EXPR)                       \
  static inline void _fast_##OP##_f32_neon(                            \
      const void *restrict ap, const void *restrict bp,                \
      void *restrict op, size_t n) {                                   \
    const float *a = (const float *)ap;                                \
    const float *b = (const float *)bp;                                \
    float *out = (float *)op;                                          \
    size_t i = 0;                                                      \
    for (; i + 4 <= n; i += 4) {                                       \
      float32x4_t va = vld1q_f32(a + i);                               \
      float32x4_t vb = vld1q_f32(b + i);                               \
      vst1q_f32(out + i, VEC_OP(va, vb));                              \
    }                                                                  \
    for (; i < n; i++)                                                 \
      out[i] = (float)(TAIL_EXPR);                                    \
  }

#define FAST_BIN_F64_NEON(OP, VEC_OP, TAIL_EXPR)                       \
  static inline void _fast_##OP##_f64_neon(                            \
      const void *restrict ap, const void *restrict bp,                \
      void *restrict op, size_t n) {                                   \
    const double *a = (const double *)ap;                              \
    const double *b = (const double *)bp;                              \
    double *out = (double *)op;                                        \
    size_t i = 0;                                                      \
    for (; i + 2 <= n; i += 2) {                                       \
      float64x2_t va = vld1q_f64(a + i);                               \
      float64x2_t vb = vld1q_f64(b + i);                               \
      vst1q_f64(out + i, VEC_OP(va, vb));                              \
    }                                                                  \
    for (; i < n; i++)                                                 \
      out[i] = (double)(TAIL_EXPR);                                   \
  }

/* ── Add ─────────────────────────────────────────────────────────── */

FAST_BIN_SINT_NEON(add, i8, int8_t, 8, 16, vaddq_s8, a[i] + b[i])
FAST_BIN_SINT_NEON(add, i16, int16_t, 16, 8, vaddq_s16, a[i] + b[i])
FAST_BIN_SINT_NEON(add, i32, int32_t, 32, 4, vaddq_s32, a[i] + b[i])
FAST_BIN_SINT_NEON(add, i64, int64_t, 64, 2, vaddq_s64, a[i] + b[i])
FAST_BIN_UINT_NEON(add, u8, uint8_t, 8, 16, vaddq_u8, a[i] + b[i])
FAST_BIN_UINT_NEON(add, u16, uint16_t, 16, 8, vaddq_u16, a[i] + b[i])
FAST_BIN_UINT_NEON(add, u32, uint32_t, 32, 4, vaddq_u32, a[i] + b[i])
FAST_BIN_UINT_NEON(add, u64, uint64_t, 64, 2, vaddq_u64, a[i] + b[i])
FAST_BIN_F32_NEON(add, vaddq_f32, a[i] + b[i])
FAST_BIN_F64_NEON(add, vaddq_f64, a[i] + b[i])

/* ── Sub ─────────────────────────────────────────────────────────── */

FAST_BIN_SINT_NEON(sub, i8, int8_t, 8, 16, vsubq_s8, a[i] - b[i])
FAST_BIN_SINT_NEON(sub, i16, int16_t, 16, 8, vsubq_s16, a[i] - b[i])
FAST_BIN_SINT_NEON(sub, i32, int32_t, 32, 4, vsubq_s32, a[i] - b[i])
FAST_BIN_SINT_NEON(sub, i64, int64_t, 64, 2, vsubq_s64, a[i] - b[i])
FAST_BIN_UINT_NEON(sub, u8, uint8_t, 8, 16, vsubq_u8, a[i] - b[i])
FAST_BIN_UINT_NEON(sub, u16, uint16_t, 16, 8, vsubq_u16, a[i] - b[i])
FAST_BIN_UINT_NEON(sub, u32, uint32_t, 32, 4, vsubq_u32, a[i] - b[i])
FAST_BIN_UINT_NEON(sub, u64, uint64_t, 64, 2, vsubq_u64, a[i] - b[i])
FAST_BIN_F32_NEON(sub, vsubq_f32, a[i] - b[i])
FAST_BIN_F64_NEON(sub, vsubq_f64, a[i] - b[i])

/* ── Mul (8/16/32-bit: native) ───────────────────────────────────── */

FAST_BIN_SINT_NEON(mul, i8, int8_t, 8, 16, vmulq_s8, a[i] * b[i])
FAST_BIN_SINT_NEON(mul, i16, int16_t, 16, 8, vmulq_s16, a[i] * b[i])
FAST_BIN_SINT_NEON(mul, i32, int32_t, 32, 4, vmulq_s32, a[i] * b[i])
FAST_BIN_UINT_NEON(mul, u8, uint8_t, 8, 16, vmulq_u8, a[i] * b[i])
FAST_BIN_UINT_NEON(mul, u16, uint16_t, 16, 8, vmulq_u16, a[i] * b[i])
FAST_BIN_UINT_NEON(mul, u32, uint32_t, 32, 4, vmulq_u32, a[i] * b[i])

/* i64/u64 mul: scalar (no native NEON instruction) */

static inline void _fast_mul_i64_neon(const void *restrict ap,
                                      const void *restrict bp,
                                      void *restrict op, size_t n) {
  const int64_t *a = (const int64_t *)ap;
  const int64_t *b = (const int64_t *)bp;
  int64_t *out = (int64_t *)op;
  for (size_t i = 0; i < n; i++)
    out[i] = a[i] * b[i];
}

static inline void _fast_mul_u64_neon(const void *restrict ap,
                                      const void *restrict bp,
                                      void *restrict op, size_t n) {
  const uint64_t *a = (const uint64_t *)ap;
  const uint64_t *b = (const uint64_t *)bp;
  uint64_t *out = (uint64_t *)op;
  for (size_t i = 0; i < n; i++)
    out[i] = a[i] * b[i];
}

FAST_BIN_F32_NEON(mul, vmulq_f32, a[i] * b[i])
FAST_BIN_F64_NEON(mul, vmulq_f64, a[i] * b[i])

/* ── Maximum (signed, 8/16/32: native) ──────────────────────────── */

FAST_BIN_SINT_NEON(maximum, i8, int8_t, 8, 16, vmaxq_s8,
                   a[i] > b[i] ? a[i] : b[i])
FAST_BIN_SINT_NEON(maximum, i16, int16_t, 16, 8, vmaxq_s16,
                   a[i] > b[i] ? a[i] : b[i])
FAST_BIN_SINT_NEON(maximum, i32, int32_t, 32, 4, vmaxq_s32,
                   a[i] > b[i] ? a[i] : b[i])

/* i64 maximum: emulated via vcgtq_s64 + vbslq_s64 */
static inline void _fast_maximum_i64_neon(const void *restrict ap,
                                          const void *restrict bp,
                                          void *restrict op, size_t n) {
  const int64_t *a = (const int64_t *)ap;
  const int64_t *b = (const int64_t *)bp;
  int64_t *out = (int64_t *)op;
  size_t i = 0;
  for (; i + 2 <= n; i += 2) {
    int64x2_t va = vld1q_s64(a + i);
    int64x2_t vb = vld1q_s64(b + i);
    uint64x2_t gt = vcgtq_s64(va, vb);
    vst1q_s64(out + i, vbslq_s64(gt, va, vb));
  }
  for (; i < n; i++)
    out[i] = a[i] > b[i] ? a[i] : b[i];
}

/* ── Maximum (unsigned, 8/16/32: native) ────────────────────────── */

FAST_BIN_UINT_NEON(maximum, u8, uint8_t, 8, 16, vmaxq_u8,
                   a[i] > b[i] ? a[i] : b[i])
FAST_BIN_UINT_NEON(maximum, u16, uint16_t, 16, 8, vmaxq_u16,
                   a[i] > b[i] ? a[i] : b[i])
FAST_BIN_UINT_NEON(maximum, u32, uint32_t, 32, 4, vmaxq_u32,
                   a[i] > b[i] ? a[i] : b[i])

/* u64 maximum: emulated via vcgtq_u64 + vbslq_u64 */
static inline void _fast_maximum_u64_neon(const void *restrict ap,
                                          const void *restrict bp,
                                          void *restrict op, size_t n) {
  const uint64_t *a = (const uint64_t *)ap;
  const uint64_t *b = (const uint64_t *)bp;
  uint64_t *out = (uint64_t *)op;
  size_t i = 0;
  for (; i + 2 <= n; i += 2) {
    uint64x2_t va = vld1q_u64(a + i);
    uint64x2_t vb = vld1q_u64(b + i);
    uint64x2_t gt = vcgtq_u64(va, vb);
    vst1q_u64(out + i, vbslq_u64(gt, va, vb));
  }
  for (; i < n; i++)
    out[i] = a[i] > b[i] ? a[i] : b[i];
}

/* ── Maximum (float) ─────────────────────────────────────────────── */

FAST_BIN_F32_NEON(maximum, vmaxq_f32, a[i] > b[i] ? a[i] : b[i])
FAST_BIN_F64_NEON(maximum, vmaxq_f64, a[i] > b[i] ? a[i] : b[i])

/* ── Minimum (signed, 8/16/32: native) ──────────────────────────── */

FAST_BIN_SINT_NEON(minimum, i8, int8_t, 8, 16, vminq_s8,
                   a[i] < b[i] ? a[i] : b[i])
FAST_BIN_SINT_NEON(minimum, i16, int16_t, 16, 8, vminq_s16,
                   a[i] < b[i] ? a[i] : b[i])
FAST_BIN_SINT_NEON(minimum, i32, int32_t, 32, 4, vminq_s32,
                   a[i] < b[i] ? a[i] : b[i])

/* i64 minimum: vcgtq_s64(va, vb) then select vb where a>b */
static inline void _fast_minimum_i64_neon(const void *restrict ap,
                                          const void *restrict bp,
                                          void *restrict op, size_t n) {
  const int64_t *a = (const int64_t *)ap;
  const int64_t *b = (const int64_t *)bp;
  int64_t *out = (int64_t *)op;
  size_t i = 0;
  for (; i + 2 <= n; i += 2) {
    int64x2_t va = vld1q_s64(a + i);
    int64x2_t vb = vld1q_s64(b + i);
    uint64x2_t gt = vcgtq_s64(va, vb);
    vst1q_s64(out + i, vbslq_s64(gt, vb, va));
  }
  for (; i < n; i++)
    out[i] = a[i] < b[i] ? a[i] : b[i];
}

/* ── Minimum (unsigned, 8/16/32: native) ────────────────────────── */

FAST_BIN_UINT_NEON(minimum, u8, uint8_t, 8, 16, vminq_u8,
                   a[i] < b[i] ? a[i] : b[i])
FAST_BIN_UINT_NEON(minimum, u16, uint16_t, 16, 8, vminq_u16,
                   a[i] < b[i] ? a[i] : b[i])
FAST_BIN_UINT_NEON(minimum, u32, uint32_t, 32, 4, vminq_u32,
                   a[i] < b[i] ? a[i] : b[i])

/* u64 minimum: vcgtq_u64(va, vb) then select vb where a>b */
static inline void _fast_minimum_u64_neon(const void *restrict ap,
                                          const void *restrict bp,
                                          void *restrict op, size_t n) {
  const uint64_t *a = (const uint64_t *)ap;
  const uint64_t *b = (const uint64_t *)bp;
  uint64_t *out = (uint64_t *)op;
  size_t i = 0;
  for (; i + 2 <= n; i += 2) {
    uint64x2_t va = vld1q_u64(a + i);
    uint64x2_t vb = vld1q_u64(b + i);
    uint64x2_t gt = vcgtq_u64(va, vb);
    vst1q_u64(out + i, vbslq_u64(gt, vb, va));
  }
  for (; i < n; i++)
    out[i] = a[i] < b[i] ? a[i] : b[i];
}

/* ── Minimum (float) ─────────────────────────────────────────────── */

FAST_BIN_F32_NEON(minimum, vminq_f32, a[i] < b[i] ? a[i] : b[i])
FAST_BIN_F64_NEON(minimum, vminq_f64, a[i] < b[i] ? a[i] : b[i])

#undef FAST_BIN_SINT_NEON
#undef FAST_BIN_UINT_NEON
#undef FAST_BIN_F32_NEON
#undef FAST_BIN_F64_NEON

/* ════════════════════════════════════════════════════════════════════
 * Unary operations
 * ════════════════════════════════════════════════════════════════ */

#define FAST_UN_SINT_NEON(OP, SFX, CT, W, VPV, VEC_OP, TAIL_EXPR)     \
  static inline void _fast_##OP##_##SFX##_neon(                        \
      const void *restrict ap, void *restrict op, size_t n) {          \
    const CT *a = (const CT *)ap;                                      \
    CT *out = (CT *)op;                                                \
    size_t i = 0;                                                      \
    for (; i + (VPV) <= n; i += (VPV)) {                               \
      int##W##x##VPV##_t va = vld1q_s##W(a + i);                      \
      vst1q_s##W(out + i, VEC_OP);                                    \
    }                                                                  \
    for (; i < n; i++)                                                 \
      out[i] = (CT)(TAIL_EXPR);                                       \
  }

#define FAST_UN_UINT_NEON(OP, SFX, CT, W, VPV, VEC_OP, TAIL_EXPR)     \
  static inline void _fast_##OP##_##SFX##_neon(                        \
      const void *restrict ap, void *restrict op, size_t n) {          \
    const CT *a = (const CT *)ap;                                      \
    CT *out = (CT *)op;                                                \
    size_t i = 0;                                                      \
    for (; i + (VPV) <= n; i += (VPV)) {                               \
      uint##W##x##VPV##_t va = vld1q_u##W(a + i);                     \
      vst1q_u##W(out + i, VEC_OP);                                    \
    }                                                                  \
    for (; i < n; i++)                                                 \
      out[i] = (CT)(TAIL_EXPR);                                       \
  }

/* ── Neg (signed integers: native vnegq) ─────────────────────────── */

FAST_UN_SINT_NEON(neg, i8, int8_t, 8, 16, vnegq_s8(va), -a[i])
FAST_UN_SINT_NEON(neg, i16, int16_t, 16, 8, vnegq_s16(va), -a[i])
FAST_UN_SINT_NEON(neg, i32, int32_t, 32, 4, vnegq_s32(va), -a[i])
FAST_UN_SINT_NEON(neg, i64, int64_t, 64, 2, vnegq_s64(va), -a[i])

/* ── Neg (unsigned integers: 0 - val) ────────────────────────────── */

FAST_UN_UINT_NEON(neg, u8, uint8_t, 8, 16,
                  vsubq_u8(vdupq_n_u8(0), va),
                  (uint8_t)(-(int8_t)a[i]))
FAST_UN_UINT_NEON(neg, u16, uint16_t, 16, 8,
                  vsubq_u16(vdupq_n_u16(0), va),
                  (uint16_t)(-(int16_t)a[i]))
FAST_UN_UINT_NEON(neg, u32, uint32_t, 32, 4,
                  vsubq_u32(vdupq_n_u32(0), va),
                  (uint32_t)(-(int32_t)a[i]))
FAST_UN_UINT_NEON(neg, u64, uint64_t, 64, 2,
                  vsubq_u64(vdupq_n_u64(0), va),
                  (uint64_t)(-(int64_t)a[i]))

/* ── Neg (float) ─────────────────────────────────────────────────── */

static inline void _fast_neg_f32_neon(const void *restrict ap,
                                      void *restrict op, size_t n) {
  const float *a = (const float *)ap;
  float *out = (float *)op;
  size_t i = 0;
  for (; i + 4 <= n; i += 4)
    vst1q_f32(out + i, vnegq_f32(vld1q_f32(a + i)));
  for (; i < n; i++)
    out[i] = -a[i];
}

static inline void _fast_neg_f64_neon(const void *restrict ap,
                                      void *restrict op, size_t n) {
  const double *a = (const double *)ap;
  double *out = (double *)op;
  size_t i = 0;
  for (; i + 2 <= n; i += 2)
    vst1q_f64(out + i, vnegq_f64(vld1q_f64(a + i)));
  for (; i < n; i++)
    out[i] = -a[i];
}

/* ── Abs (signed integers: native vabsq) ─────────────────────────── */

FAST_UN_SINT_NEON(abs, i8, int8_t, 8, 16, vabsq_s8(va),
                  (int8_t)(a[i] < 0 ? -a[i] : a[i]))
FAST_UN_SINT_NEON(abs, i16, int16_t, 16, 8, vabsq_s16(va),
                  (int16_t)(a[i] < 0 ? -a[i] : a[i]))
FAST_UN_SINT_NEON(abs, i32, int32_t, 32, 4, vabsq_s32(va),
                  (int32_t)(a[i] < 0 ? -a[i] : a[i]))
FAST_UN_SINT_NEON(abs, i64, int64_t, 64, 2, vabsq_s64(va),
                  a[i] < 0 ? -a[i] : a[i])

/* ── Abs (float) ─────────────────────────────────────────────────── */

static inline void _fast_abs_f32_neon(const void *restrict ap,
                                      void *restrict op, size_t n) {
  const float *a = (const float *)ap;
  float *out = (float *)op;
  size_t i = 0;
  for (; i + 4 <= n; i += 4)
    vst1q_f32(out + i, vabsq_f32(vld1q_f32(a + i)));
  for (; i < n; i++)
    out[i] = a[i] < 0 ? -a[i] : a[i];
}

static inline void _fast_abs_f64_neon(const void *restrict ap,
                                      void *restrict op, size_t n) {
  const double *a = (const double *)ap;
  double *out = (double *)op;
  size_t i = 0;
  for (; i + 2 <= n; i += 2)
    vst1q_f64(out + i, vabsq_f64(vld1q_f64(a + i)));
  for (; i < n; i++)
    out[i] = a[i] < 0 ? -a[i] : a[i];
}

#undef FAST_UN_SINT_NEON
#undef FAST_UN_UINT_NEON

#endif /* NUMC_ELEMWISE_NEON_H */
