/**
 * @file compare_scalar_neon.h
 * @brief NEON scalar comparison kernels for all 10 types.
 *
 * NEON has native comparisons for all signed, unsigned, and float types.
 * Compare result (all-1s/all-0s) is AND-ed with 1 (or vbsl-ed for float).
 */
#ifndef NUMC_COMPARE_SCALAR_NEON_H
#define NUMC_COMPARE_SCALAR_NEON_H

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

/* ── Signed integer macro ───────────────────────────────────────── */

#define STAMP_CMPSC_SINT_NEON(SFX, CT, W, VPV)                             \
  static inline void _cmpsc_eq_##SFX##_neon(const void *restrict ap,       \
                                            const void *restrict sp,       \
                                            void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                          \
    const CT s = *(const CT *)sp;                                          \
    CT *out = (CT *)op;                                                    \
    uint##W##x##VPV##_t one = vdupq_n_u##W(1);                             \
    int##W##x##VPV##_t vs = vdupq_n_s##W(s);                               \
    size_t i = 0;                                                          \
    for (; i + (VPV) <= n; i += (VPV)) {                                   \
      int##W##x##VPV##_t va = vld1q_s##W(a + i);                           \
      uint##W##x##VPV##_t r = vandq_u##W(vceqq_s##W(va, vs), one);         \
      vst1q_u##W((uint##W##_t *)(out + i), r);                             \
    }                                                                      \
    for (; i < n; i++)                                                     \
      out[i] = (CT)(a[i] == s);                                            \
  }                                                                        \
  static inline void _cmpsc_gt_##SFX##_neon(const void *restrict ap,       \
                                            const void *restrict sp,       \
                                            void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                          \
    const CT s = *(const CT *)sp;                                          \
    CT *out = (CT *)op;                                                    \
    uint##W##x##VPV##_t one = vdupq_n_u##W(1);                             \
    int##W##x##VPV##_t vs = vdupq_n_s##W(s);                               \
    size_t i = 0;                                                          \
    for (; i + (VPV) <= n; i += (VPV)) {                                   \
      int##W##x##VPV##_t va = vld1q_s##W(a + i);                           \
      uint##W##x##VPV##_t r = vandq_u##W(vcgtq_s##W(va, vs), one);         \
      vst1q_u##W((uint##W##_t *)(out + i), r);                             \
    }                                                                      \
    for (; i < n; i++)                                                     \
      out[i] = (CT)(a[i] > s);                                             \
  }                                                                        \
  static inline void _cmpsc_lt_##SFX##_neon(const void *restrict ap,       \
                                            const void *restrict sp,       \
                                            void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                          \
    const CT s = *(const CT *)sp;                                          \
    CT *out = (CT *)op;                                                    \
    uint##W##x##VPV##_t one = vdupq_n_u##W(1);                             \
    int##W##x##VPV##_t vs = vdupq_n_s##W(s);                               \
    size_t i = 0;                                                          \
    for (; i + (VPV) <= n; i += (VPV)) {                                   \
      int##W##x##VPV##_t va = vld1q_s##W(a + i);                           \
      uint##W##x##VPV##_t r = vandq_u##W(vcltq_s##W(va, vs), one);         \
      vst1q_u##W((uint##W##_t *)(out + i), r);                             \
    }                                                                      \
    for (; i < n; i++)                                                     \
      out[i] = (CT)(a[i] < s);                                             \
  }                                                                        \
  static inline void _cmpsc_ge_##SFX##_neon(const void *restrict ap,       \
                                            const void *restrict sp,       \
                                            void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                          \
    const CT s = *(const CT *)sp;                                          \
    CT *out = (CT *)op;                                                    \
    uint##W##x##VPV##_t one = vdupq_n_u##W(1);                             \
    int##W##x##VPV##_t vs = vdupq_n_s##W(s);                               \
    size_t i = 0;                                                          \
    for (; i + (VPV) <= n; i += (VPV)) {                                   \
      int##W##x##VPV##_t va = vld1q_s##W(a + i);                           \
      uint##W##x##VPV##_t r = vandq_u##W(vcgeq_s##W(va, vs), one);         \
      vst1q_u##W((uint##W##_t *)(out + i), r);                             \
    }                                                                      \
    for (; i < n; i++)                                                     \
      out[i] = (CT)(a[i] >= s);                                            \
  }                                                                        \
  static inline void _cmpsc_le_##SFX##_neon(const void *restrict ap,       \
                                            const void *restrict sp,       \
                                            void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                          \
    const CT s = *(const CT *)sp;                                          \
    CT *out = (CT *)op;                                                    \
    uint##W##x##VPV##_t one = vdupq_n_u##W(1);                             \
    int##W##x##VPV##_t vs = vdupq_n_s##W(s);                               \
    size_t i = 0;                                                          \
    for (; i + (VPV) <= n; i += (VPV)) {                                   \
      int##W##x##VPV##_t va = vld1q_s##W(a + i);                           \
      uint##W##x##VPV##_t r = vandq_u##W(vcleq_s##W(va, vs), one);         \
      vst1q_u##W((uint##W##_t *)(out + i), r);                             \
    }                                                                      \
    for (; i < n; i++)                                                     \
      out[i] = (CT)(a[i] <= s);                                            \
  }

STAMP_CMPSC_SINT_NEON(i8, int8_t, 8, 16)
STAMP_CMPSC_SINT_NEON(i16, int16_t, 16, 8)
STAMP_CMPSC_SINT_NEON(i32, int32_t, 32, 4)
STAMP_CMPSC_SINT_NEON(i64, int64_t, 64, 2)
#undef STAMP_CMPSC_SINT_NEON

/* ── Unsigned integer macro ─────────────────────────────────────── */

#define STAMP_CMPSC_UINT_NEON(SFX, CT, W, VPV)                             \
  static inline void _cmpsc_eq_##SFX##_neon(const void *restrict ap,       \
                                            const void *restrict sp,       \
                                            void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                          \
    const CT s = *(const CT *)sp;                                          \
    CT *out = (CT *)op;                                                    \
    uint##W##x##VPV##_t one = vdupq_n_u##W(1);                             \
    uint##W##x##VPV##_t vs = vdupq_n_u##W(s);                              \
    size_t i = 0;                                                          \
    for (; i + (VPV) <= n; i += (VPV)) {                                   \
      uint##W##x##VPV##_t va = vld1q_u##W(a + i);                          \
      uint##W##x##VPV##_t r = vandq_u##W(vceqq_u##W(va, vs), one);         \
      vst1q_u##W(out + i, r);                                              \
    }                                                                      \
    for (; i < n; i++)                                                     \
      out[i] = (CT)(a[i] == s);                                            \
  }                                                                        \
  static inline void _cmpsc_gt_##SFX##_neon(const void *restrict ap,       \
                                            const void *restrict sp,       \
                                            void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                          \
    const CT s = *(const CT *)sp;                                          \
    CT *out = (CT *)op;                                                    \
    uint##W##x##VPV##_t one = vdupq_n_u##W(1);                             \
    uint##W##x##VPV##_t vs = vdupq_n_u##W(s);                              \
    size_t i = 0;                                                          \
    for (; i + (VPV) <= n; i += (VPV)) {                                   \
      uint##W##x##VPV##_t va = vld1q_u##W(a + i);                          \
      uint##W##x##VPV##_t r = vandq_u##W(vcgtq_u##W(va, vs), one);         \
      vst1q_u##W(out + i, r);                                              \
    }                                                                      \
    for (; i < n; i++)                                                     \
      out[i] = (CT)(a[i] > s);                                             \
  }                                                                        \
  static inline void _cmpsc_lt_##SFX##_neon(const void *restrict ap,       \
                                            const void *restrict sp,       \
                                            void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                          \
    const CT s = *(const CT *)sp;                                          \
    CT *out = (CT *)op;                                                    \
    uint##W##x##VPV##_t one = vdupq_n_u##W(1);                             \
    uint##W##x##VPV##_t vs = vdupq_n_u##W(s);                              \
    size_t i = 0;                                                          \
    for (; i + (VPV) <= n; i += (VPV)) {                                   \
      uint##W##x##VPV##_t va = vld1q_u##W(a + i);                          \
      uint##W##x##VPV##_t r = vandq_u##W(vcltq_u##W(va, vs), one);         \
      vst1q_u##W(out + i, r);                                              \
    }                                                                      \
    for (; i < n; i++)                                                     \
      out[i] = (CT)(a[i] < s);                                             \
  }                                                                        \
  static inline void _cmpsc_ge_##SFX##_neon(const void *restrict ap,       \
                                            const void *restrict sp,       \
                                            void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                          \
    const CT s = *(const CT *)sp;                                          \
    CT *out = (CT *)op;                                                    \
    uint##W##x##VPV##_t one = vdupq_n_u##W(1);                             \
    uint##W##x##VPV##_t vs = vdupq_n_u##W(s);                              \
    size_t i = 0;                                                          \
    for (; i + (VPV) <= n; i += (VPV)) {                                   \
      uint##W##x##VPV##_t va = vld1q_u##W(a + i);                          \
      uint##W##x##VPV##_t r = vandq_u##W(vcgeq_u##W(va, vs), one);         \
      vst1q_u##W(out + i, r);                                              \
    }                                                                      \
    for (; i < n; i++)                                                     \
      out[i] = (CT)(a[i] >= s);                                            \
  }                                                                        \
  static inline void _cmpsc_le_##SFX##_neon(const void *restrict ap,       \
                                            const void *restrict sp,       \
                                            void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                          \
    const CT s = *(const CT *)sp;                                          \
    CT *out = (CT *)op;                                                    \
    uint##W##x##VPV##_t one = vdupq_n_u##W(1);                             \
    uint##W##x##VPV##_t vs = vdupq_n_u##W(s);                              \
    size_t i = 0;                                                          \
    for (; i + (VPV) <= n; i += (VPV)) {                                   \
      uint##W##x##VPV##_t va = vld1q_u##W(a + i);                          \
      uint##W##x##VPV##_t r = vandq_u##W(vcleq_u##W(va, vs), one);         \
      vst1q_u##W(out + i, r);                                              \
    }                                                                      \
    for (; i < n; i++)                                                     \
      out[i] = (CT)(a[i] <= s);                                            \
  }

STAMP_CMPSC_UINT_NEON(u8, uint8_t, 8, 16)
STAMP_CMPSC_UINT_NEON(u16, uint16_t, 16, 8)
STAMP_CMPSC_UINT_NEON(u32, uint32_t, 32, 4)
STAMP_CMPSC_UINT_NEON(u64, uint64_t, 64, 2)
#undef STAMP_CMPSC_UINT_NEON

/* ── Float: f32 (4 per vector) ──────────────────────────────────── */

#define CMPSC_F32_NEON(NAME, CMP)                                            \
  static inline void _cmpsc_##NAME##_f32_neon(const void *restrict ap,       \
                                              const void *restrict sp,       \
                                              void *restrict op, size_t n) { \
    const float *a = (const float *)ap;                                      \
    const float s = *(const float *)sp;                                      \
    float *out = (float *)op;                                                \
    float32x4_t vs = vdupq_n_f32(s);                                         \
    float32x4_t one = vdupq_n_f32(1.0f);                                     \
    float32x4_t zero = vdupq_n_f32(0.0f);                                    \
    size_t i = 0;                                                            \
    for (; i + 4 <= n; i += 4) {                                             \
      float32x4_t va = vld1q_f32(a + i);                                     \
      uint32x4_t mask = CMP(va, vs);                                         \
      vst1q_f32(out + i, vbslq_f32(mask, one, zero));                        \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (float)(NAME##_scalar_f32(a[i], s));                          \
  }

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

CMPSC_F32_NEON(eq, vceqq_f32)
CMPSC_F32_NEON(gt, vcgtq_f32)
CMPSC_F32_NEON(lt, vcltq_f32)
CMPSC_F32_NEON(ge, vcgeq_f32)
CMPSC_F32_NEON(le, vcleq_f32)
#undef CMPSC_F32_NEON

/* ── Float: f64 (2 per vector) ──────────────────────────────────── */

#define CMPSC_F64_NEON(NAME, CMP)                                            \
  static inline void _cmpsc_##NAME##_f64_neon(const void *restrict ap,       \
                                              const void *restrict sp,       \
                                              void *restrict op, size_t n) { \
    const double *a = (const double *)ap;                                    \
    const double s = *(const double *)sp;                                    \
    double *out = (double *)op;                                              \
    float64x2_t vs = vdupq_n_f64(s);                                         \
    float64x2_t one = vdupq_n_f64(1.0);                                      \
    float64x2_t zero = vdupq_n_f64(0.0);                                     \
    size_t i = 0;                                                            \
    for (; i + 2 <= n; i += 2) {                                             \
      float64x2_t va = vld1q_f64(a + i);                                     \
      uint64x2_t mask = CMP(va, vs);                                         \
      vst1q_f64(out + i, vbslq_f64(mask, one, zero));                        \
    }                                                                        \
    for (; i < n; i++)                                                       \
      out[i] = (double)(NAME##_scalar_f64(a[i], s));                         \
  }

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

CMPSC_F64_NEON(eq, vceqq_f64)
CMPSC_F64_NEON(gt, vcgtq_f64)
CMPSC_F64_NEON(lt, vcltq_f64)
CMPSC_F64_NEON(ge, vcgeq_f64)
CMPSC_F64_NEON(le, vcleq_f64)
#undef CMPSC_F64_NEON

#endif /* NUMC_COMPARE_SCALAR_NEON_H */
