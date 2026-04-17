/**
 * @file elemwise_scalar_neon.h
 * @brief NEON scalar arithmetic kernels for all 10 types.
 *
 * Operations: add_scalar, sub_scalar, mul_scalar
 *
 * Special cases: i64/u64 mul_scalar (scalar fallback, no native NEON
 * 64-bit multiply instruction).
 */
#ifndef NUMC_ELEMWISE_SCALAR_NEON_H
#define NUMC_ELEMWISE_SCALAR_NEON_H

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

/* ====================================================================
 * Generic macros for signed int, unsigned int, float scalar ops
 * ================================================================ */

#define FAST_SCAL_SINT_NEON(OP, SFX, CT, W, VPV, VEC_OP, TAIL_EXPR)        \
  static inline void _fast_##OP##_scalar_##SFX##_neon(                     \
      const void *restrict ap, const void *restrict sp, void *restrict op, \
      size_t n) {                                                          \
    const CT *a = (const CT *)ap;                                          \
    const CT s = *(const CT *)sp;                                          \
    CT *out = (CT *)op;                                                    \
    int##W##x##VPV##_t vs = vdupq_n_s##W(s);                               \
    size_t i = 0;                                                          \
    for (; i + (VPV) <= n; i += (VPV)) {                                   \
      int##W##x##VPV##_t va = vld1q_s##W(a + i);                           \
      vst1q_s##W(out + i, VEC_OP(va, vs));                                 \
    }                                                                      \
    for (; i < n; i++)                                                     \
      out[i] = (CT)(TAIL_EXPR);                                            \
  }

#define FAST_SCAL_UINT_NEON(OP, SFX, CT, W, VPV, VEC_OP, TAIL_EXPR)        \
  static inline void _fast_##OP##_scalar_##SFX##_neon(                     \
      const void *restrict ap, const void *restrict sp, void *restrict op, \
      size_t n) {                                                          \
    const CT *a = (const CT *)ap;                                          \
    const CT s = *(const CT *)sp;                                          \
    CT *out = (CT *)op;                                                    \
    uint##W##x##VPV##_t vs = vdupq_n_u##W(s);                              \
    size_t i = 0;                                                          \
    for (; i + (VPV) <= n; i += (VPV)) {                                   \
      uint##W##x##VPV##_t va = vld1q_u##W(a + i);                          \
      vst1q_u##W(out + i, VEC_OP(va, vs));                                 \
    }                                                                      \
    for (; i < n; i++)                                                     \
      out[i] = (CT)(TAIL_EXPR);                                            \
  }

#define FAST_SCAL_F32_NEON(OP, VEC_OP, TAIL_EXPR)                          \
  static inline void _fast_##OP##_scalar_f32_neon(                         \
      const void *restrict ap, const void *restrict sp, void *restrict op, \
      size_t n) {                                                          \
    const float *a = (const float *)ap;                                    \
    const float s = *(const float *)sp;                                    \
    float *out = (float *)op;                                              \
    float32x4_t vs = vdupq_n_f32(s);                                       \
    size_t i = 0;                                                          \
    for (; i + 4 <= n; i += 4) {                                           \
      float32x4_t va = vld1q_f32(a + i);                                   \
      vst1q_f32(out + i, VEC_OP(va, vs));                                  \
    }                                                                      \
    for (; i < n; i++)                                                     \
      out[i] = (float)(TAIL_EXPR);                                         \
  }

#define FAST_SCAL_F64_NEON(OP, VEC_OP, TAIL_EXPR)                          \
  static inline void _fast_##OP##_scalar_f64_neon(                         \
      const void *restrict ap, const void *restrict sp, void *restrict op, \
      size_t n) {                                                          \
    const double *a = (const double *)ap;                                  \
    const double s = *(const double *)sp;                                  \
    double *out = (double *)op;                                            \
    float64x2_t vs = vdupq_n_f64(s);                                       \
    size_t i = 0;                                                          \
    for (; i + 2 <= n; i += 2) {                                           \
      float64x2_t va = vld1q_f64(a + i);                                   \
      vst1q_f64(out + i, VEC_OP(va, vs));                                  \
    }                                                                      \
    for (; i < n; i++)                                                     \
      out[i] = (double)(TAIL_EXPR);                                        \
  }

/* -- Add scalar ---------------------------------------------------- */

FAST_SCAL_SINT_NEON(add, i8, int8_t, 8, 16, vaddq_s8, a[i] + s)
FAST_SCAL_SINT_NEON(add, i16, int16_t, 16, 8, vaddq_s16, a[i] + s)
FAST_SCAL_SINT_NEON(add, i32, int32_t, 32, 4, vaddq_s32, a[i] + s)
FAST_SCAL_SINT_NEON(add, i64, int64_t, 64, 2, vaddq_s64, a[i] + s)
FAST_SCAL_UINT_NEON(add, u8, uint8_t, 8, 16, vaddq_u8, a[i] + s)
FAST_SCAL_UINT_NEON(add, u16, uint16_t, 16, 8, vaddq_u16, a[i] + s)
FAST_SCAL_UINT_NEON(add, u32, uint32_t, 32, 4, vaddq_u32, a[i] + s)
FAST_SCAL_UINT_NEON(add, u64, uint64_t, 64, 2, vaddq_u64, a[i] + s)
FAST_SCAL_F32_NEON(add, vaddq_f32, a[i] + s)
FAST_SCAL_F64_NEON(add, vaddq_f64, a[i] + s)

/* -- Sub scalar ---------------------------------------------------- */

FAST_SCAL_SINT_NEON(sub, i8, int8_t, 8, 16, vsubq_s8, a[i] - s)
FAST_SCAL_SINT_NEON(sub, i16, int16_t, 16, 8, vsubq_s16, a[i] - s)
FAST_SCAL_SINT_NEON(sub, i32, int32_t, 32, 4, vsubq_s32, a[i] - s)
FAST_SCAL_SINT_NEON(sub, i64, int64_t, 64, 2, vsubq_s64, a[i] - s)
FAST_SCAL_UINT_NEON(sub, u8, uint8_t, 8, 16, vsubq_u8, a[i] - s)
FAST_SCAL_UINT_NEON(sub, u16, uint16_t, 16, 8, vsubq_u16, a[i] - s)
FAST_SCAL_UINT_NEON(sub, u32, uint32_t, 32, 4, vsubq_u32, a[i] - s)
FAST_SCAL_UINT_NEON(sub, u64, uint64_t, 64, 2, vsubq_u64, a[i] - s)
FAST_SCAL_F32_NEON(sub, vsubq_f32, a[i] - s)
FAST_SCAL_F64_NEON(sub, vsubq_f64, a[i] - s)

/* -- Mul scalar (8/16/32-bit: native) ------------------------------ */

FAST_SCAL_SINT_NEON(mul, i8, int8_t, 8, 16, vmulq_s8, a[i] * s)
FAST_SCAL_SINT_NEON(mul, i16, int16_t, 16, 8, vmulq_s16, a[i] * s)
FAST_SCAL_SINT_NEON(mul, i32, int32_t, 32, 4, vmulq_s32, a[i] * s)
FAST_SCAL_UINT_NEON(mul, u8, uint8_t, 8, 16, vmulq_u8, a[i] * s)
FAST_SCAL_UINT_NEON(mul, u16, uint16_t, 16, 8, vmulq_u16, a[i] * s)
FAST_SCAL_UINT_NEON(mul, u32, uint32_t, 32, 4, vmulq_u32, a[i] * s)

/* i64/u64 mul_scalar: scalar fallback (no native NEON instruction) */

static inline void _fast_mul_scalar_i64_neon(const void *restrict ap,
                                             const void *restrict sp,
                                             void *restrict op, size_t n) {
  const int64_t *a = (const int64_t *)ap;
  const int64_t s = *(const int64_t *)sp;
  int64_t *out = (int64_t *)op;
  for (size_t i = 0; i < n; i++)
    out[i] = a[i] * s;
}

static inline void _fast_mul_scalar_u64_neon(const void *restrict ap,
                                             const void *restrict sp,
                                             void *restrict op, size_t n) {
  const uint64_t *a = (const uint64_t *)ap;
  const uint64_t s = *(const uint64_t *)sp;
  uint64_t *out = (uint64_t *)op;
  for (size_t i = 0; i < n; i++)
    out[i] = a[i] * s;
}

FAST_SCAL_F32_NEON(mul, vmulq_f32, a[i] * s)
FAST_SCAL_F64_NEON(mul, vmulq_f64, a[i] * s)

/* -- Clean up macros ----------------------------------------------- */

#undef FAST_SCAL_SINT_NEON
#undef FAST_SCAL_UINT_NEON
#undef FAST_SCAL_F32_NEON
#undef FAST_SCAL_F64_NEON

#endif /* NUMC_ELEMWISE_SCALAR_NEON_H */
