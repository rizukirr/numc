/**
 * @file compare_neon.h
 * @brief NEON binary comparison kernels for all 10 types × 5 ops.
 *
 * Produces 0 or 1 (same type as input) per element.
 * NEON has native signed, unsigned, and float comparisons.
 * Comparison intrinsics return all-ones for true, 0 for false;
 * we AND with 1 (integer) or use vbslq (float) to convert to 0/1.
 */
#ifndef NUMC_COMPARE_NEON_H
#define NUMC_COMPARE_NEON_H

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

/* ════════════════════════════════════════════════════════════════════
 * Signed integer comparisons
 * ════════════════════════════════════════════════════════════════ */

/*
 * NEON comparison intrinsics for signed types return unsigned result
 * types (e.g., vcgtq_s32 returns uint32x4_t). We AND in the unsigned
 * domain, then reinterpret back to signed for store.
 */
#define FAST_CMP_SINT_NEON(SFX, CT, VT, UT, VPV, LOAD, STORE,       \
                           UAND, USET1, REINTERP,                    \
                           CMPEQ, CMPGT, CMPLT, CMPGE, CMPLE)       \
  static inline void _fast_eq_##SFX##_neon(                           \
      const void *restrict ap, const void *restrict bp,               \
      void *restrict op, size_t n) {                                  \
    const CT *a = (const CT *)ap;                                     \
    const CT *b = (const CT *)bp;                                     \
    CT *out = (CT *)op;                                               \
    const UT one = USET1(1);                                          \
    size_t i = 0;                                                     \
    for (; i + (VPV) <= n; i += (VPV)) {                              \
      VT va = LOAD(a + i);                                            \
      VT vb = LOAD(b + i);                                            \
      STORE(out + i, REINTERP(UAND(CMPEQ(va, vb), one)));            \
    }                                                                 \
    for (; i < n; i++)                                                \
      out[i] = (CT)(a[i] == b[i]);                                    \
  }                                                                   \
  static inline void _fast_gt_##SFX##_neon(                           \
      const void *restrict ap, const void *restrict bp,               \
      void *restrict op, size_t n) {                                  \
    const CT *a = (const CT *)ap;                                     \
    const CT *b = (const CT *)bp;                                     \
    CT *out = (CT *)op;                                               \
    const UT one = USET1(1);                                          \
    size_t i = 0;                                                     \
    for (; i + (VPV) <= n; i += (VPV)) {                              \
      VT va = LOAD(a + i);                                            \
      VT vb = LOAD(b + i);                                            \
      STORE(out + i, REINTERP(UAND(CMPGT(va, vb), one)));            \
    }                                                                 \
    for (; i < n; i++)                                                \
      out[i] = (CT)(a[i] > b[i]);                                     \
  }                                                                   \
  static inline void _fast_lt_##SFX##_neon(                           \
      const void *restrict ap, const void *restrict bp,               \
      void *restrict op, size_t n) {                                  \
    const CT *a = (const CT *)ap;                                     \
    const CT *b = (const CT *)bp;                                     \
    CT *out = (CT *)op;                                               \
    const UT one = USET1(1);                                          \
    size_t i = 0;                                                     \
    for (; i + (VPV) <= n; i += (VPV)) {                              \
      VT va = LOAD(a + i);                                            \
      VT vb = LOAD(b + i);                                            \
      STORE(out + i, REINTERP(UAND(CMPLT(va, vb), one)));            \
    }                                                                 \
    for (; i < n; i++)                                                \
      out[i] = (CT)(a[i] < b[i]);                                     \
  }                                                                   \
  static inline void _fast_ge_##SFX##_neon(                           \
      const void *restrict ap, const void *restrict bp,               \
      void *restrict op, size_t n) {                                  \
    const CT *a = (const CT *)ap;                                     \
    const CT *b = (const CT *)bp;                                     \
    CT *out = (CT *)op;                                               \
    const UT one = USET1(1);                                          \
    size_t i = 0;                                                     \
    for (; i + (VPV) <= n; i += (VPV)) {                              \
      VT va = LOAD(a + i);                                            \
      VT vb = LOAD(b + i);                                            \
      STORE(out + i, REINTERP(UAND(CMPGE(va, vb), one)));            \
    }                                                                 \
    for (; i < n; i++)                                                \
      out[i] = (CT)(a[i] >= b[i]);                                    \
  }                                                                   \
  static inline void _fast_le_##SFX##_neon(                           \
      const void *restrict ap, const void *restrict bp,               \
      void *restrict op, size_t n) {                                  \
    const CT *a = (const CT *)ap;                                     \
    const CT *b = (const CT *)bp;                                     \
    CT *out = (CT *)op;                                               \
    const UT one = USET1(1);                                          \
    size_t i = 0;                                                     \
    for (; i + (VPV) <= n; i += (VPV)) {                              \
      VT va = LOAD(a + i);                                            \
      VT vb = LOAD(b + i);                                            \
      STORE(out + i, REINTERP(UAND(CMPLE(va, vb), one)));            \
    }                                                                 \
    for (; i < n; i++)                                                \
      out[i] = (CT)(a[i] <= b[i]);                                    \
  }

/* --- Signed integer instantiations --- */

/* clang-format off */
FAST_CMP_SINT_NEON(i8,  int8_t,  int8x16_t, uint8x16_t, 16,
                   vld1q_s8,  vst1q_s8,
                   vandq_u8,  vdupq_n_u8, vreinterpretq_s8_u8,
                   vceqq_s8,  vcgtq_s8,  vcltq_s8,
                   vcgeq_s8,  vcleq_s8)

FAST_CMP_SINT_NEON(i16, int16_t, int16x8_t, uint16x8_t, 8,
                   vld1q_s16, vst1q_s16,
                   vandq_u16, vdupq_n_u16, vreinterpretq_s16_u16,
                   vceqq_s16, vcgtq_s16, vcltq_s16,
                   vcgeq_s16, vcleq_s16)

FAST_CMP_SINT_NEON(i32, int32_t, int32x4_t, uint32x4_t, 4,
                   vld1q_s32, vst1q_s32,
                   vandq_u32, vdupq_n_u32, vreinterpretq_s32_u32,
                   vceqq_s32, vcgtq_s32, vcltq_s32,
                   vcgeq_s32, vcleq_s32)

FAST_CMP_SINT_NEON(i64, int64_t, int64x2_t, uint64x2_t, 2,
                   vld1q_s64, vst1q_s64,
                   vandq_u64, vdupq_n_u64, vreinterpretq_s64_u64,
                   vceqq_s64, vcgtq_s64, vcltq_s64,
                   vcgeq_s64, vcleq_s64)
/* clang-format on */

#undef FAST_CMP_SINT_NEON

/* ════════════════════════════════════════════════════════════════════
 * Unsigned integer comparisons
 * ════════════════════════════════════════════════════════════════ */

#define FAST_CMP_UINT_NEON(SFX, CT, VT, VPV, LOAD, STORE, AND,       \
                           CMPEQ, CMPGT, CMPLT, CMPGE, CMPLE, SET1)  \
  static inline void _fast_eq_##SFX##_neon(                           \
      const void *restrict ap, const void *restrict bp,               \
      void *restrict op, size_t n) {                                  \
    const CT *a = (const CT *)ap;                                     \
    const CT *b = (const CT *)bp;                                     \
    CT *out = (CT *)op;                                               \
    const VT one = SET1((CT)1);                                       \
    size_t i = 0;                                                     \
    for (; i + (VPV) <= n; i += (VPV)) {                              \
      VT va = LOAD(a + i);                                            \
      VT vb = LOAD(b + i);                                            \
      STORE(out + i, AND(CMPEQ(va, vb), one));                        \
    }                                                                 \
    for (; i < n; i++)                                                \
      out[i] = (CT)(a[i] == b[i]);                                    \
  }                                                                   \
  static inline void _fast_gt_##SFX##_neon(                           \
      const void *restrict ap, const void *restrict bp,               \
      void *restrict op, size_t n) {                                  \
    const CT *a = (const CT *)ap;                                     \
    const CT *b = (const CT *)bp;                                     \
    CT *out = (CT *)op;                                               \
    const VT one = SET1((CT)1);                                       \
    size_t i = 0;                                                     \
    for (; i + (VPV) <= n; i += (VPV)) {                              \
      VT va = LOAD(a + i);                                            \
      VT vb = LOAD(b + i);                                            \
      STORE(out + i, AND(CMPGT(va, vb), one));                        \
    }                                                                 \
    for (; i < n; i++)                                                \
      out[i] = (CT)(a[i] > b[i]);                                     \
  }                                                                   \
  static inline void _fast_lt_##SFX##_neon(                           \
      const void *restrict ap, const void *restrict bp,               \
      void *restrict op, size_t n) {                                  \
    const CT *a = (const CT *)ap;                                     \
    const CT *b = (const CT *)bp;                                     \
    CT *out = (CT *)op;                                               \
    const VT one = SET1((CT)1);                                       \
    size_t i = 0;                                                     \
    for (; i + (VPV) <= n; i += (VPV)) {                              \
      VT va = LOAD(a + i);                                            \
      VT vb = LOAD(b + i);                                            \
      STORE(out + i, AND(CMPLT(va, vb), one));                        \
    }                                                                 \
    for (; i < n; i++)                                                \
      out[i] = (CT)(a[i] < b[i]);                                     \
  }                                                                   \
  static inline void _fast_ge_##SFX##_neon(                           \
      const void *restrict ap, const void *restrict bp,               \
      void *restrict op, size_t n) {                                  \
    const CT *a = (const CT *)ap;                                     \
    const CT *b = (const CT *)bp;                                     \
    CT *out = (CT *)op;                                               \
    const VT one = SET1((CT)1);                                       \
    size_t i = 0;                                                     \
    for (; i + (VPV) <= n; i += (VPV)) {                              \
      VT va = LOAD(a + i);                                            \
      VT vb = LOAD(b + i);                                            \
      STORE(out + i, AND(CMPGE(va, vb), one));                        \
    }                                                                 \
    for (; i < n; i++)                                                \
      out[i] = (CT)(a[i] >= b[i]);                                    \
  }                                                                   \
  static inline void _fast_le_##SFX##_neon(                           \
      const void *restrict ap, const void *restrict bp,               \
      void *restrict op, size_t n) {                                  \
    const CT *a = (const CT *)ap;                                     \
    const CT *b = (const CT *)bp;                                     \
    CT *out = (CT *)op;                                               \
    const VT one = SET1((CT)1);                                       \
    size_t i = 0;                                                     \
    for (; i + (VPV) <= n; i += (VPV)) {                              \
      VT va = LOAD(a + i);                                            \
      VT vb = LOAD(b + i);                                            \
      STORE(out + i, AND(CMPLE(va, vb), one));                        \
    }                                                                 \
    for (; i < n; i++)                                                \
      out[i] = (CT)(a[i] <= b[i]);                                    \
  }

/* --- Unsigned integer instantiations --- */

/* clang-format off */
FAST_CMP_UINT_NEON(u8,  uint8_t,  uint8x16_t, 16,
                   vld1q_u8,  vst1q_u8,  vandq_u8,
                   vceqq_u8,  vcgtq_u8,  vcltq_u8,
                   vcgeq_u8,  vcleq_u8,  vdupq_n_u8)

FAST_CMP_UINT_NEON(u16, uint16_t, uint16x8_t, 8,
                   vld1q_u16, vst1q_u16, vandq_u16,
                   vceqq_u16, vcgtq_u16, vcltq_u16,
                   vcgeq_u16, vcleq_u16, vdupq_n_u16)

FAST_CMP_UINT_NEON(u32, uint32_t, uint32x4_t, 4,
                   vld1q_u32, vst1q_u32, vandq_u32,
                   vceqq_u32, vcgtq_u32, vcltq_u32,
                   vcgeq_u32, vcleq_u32, vdupq_n_u32)

FAST_CMP_UINT_NEON(u64, uint64_t, uint64x2_t, 2,
                   vld1q_u64, vst1q_u64, vandq_u64,
                   vceqq_u64, vcgtq_u64, vcltq_u64,
                   vcgeq_u64, vcleq_u64, vdupq_n_u64)
/* clang-format on */

#undef FAST_CMP_UINT_NEON

/* ════════════════════════════════════════════════════════════════════
 * Float comparisons
 *
 * NEON float comparisons return uint32x4/uint64x2 bitmasks.
 * Use vbslq to select 1.0 or 0.0 based on the mask.
 * ════════════════════════════════════════════════════════════════ */

#define FAST_CMP_FLOAT_NEON(SFX, CT, VT, UT, VPV, LOAD, STORE,       \
                            CMPEQ, CMPGT, CMPLT, CMPGE, CMPLE,       \
                            SET1, BSL)                                \
  static inline void _fast_eq_##SFX##_neon(                           \
      const void *restrict ap, const void *restrict bp,               \
      void *restrict op, size_t n) {                                  \
    const CT *a = (const CT *)ap;                                     \
    const CT *b = (const CT *)bp;                                     \
    CT *out = (CT *)op;                                               \
    const VT one = SET1((CT)1.0);                                     \
    const VT zero = SET1((CT)0.0);                                    \
    size_t i = 0;                                                     \
    for (; i + (VPV) <= n; i += (VPV)) {                              \
      VT va = LOAD(a + i);                                            \
      VT vb = LOAD(b + i);                                            \
      UT mask = CMPEQ(va, vb);                                        \
      STORE(out + i, BSL(mask, one, zero));                           \
    }                                                                 \
    for (; i < n; i++)                                                \
      out[i] = (CT)(a[i] == b[i]);                                    \
  }                                                                   \
  static inline void _fast_gt_##SFX##_neon(                           \
      const void *restrict ap, const void *restrict bp,               \
      void *restrict op, size_t n) {                                  \
    const CT *a = (const CT *)ap;                                     \
    const CT *b = (const CT *)bp;                                     \
    CT *out = (CT *)op;                                               \
    const VT one = SET1((CT)1.0);                                     \
    const VT zero = SET1((CT)0.0);                                    \
    size_t i = 0;                                                     \
    for (; i + (VPV) <= n; i += (VPV)) {                              \
      VT va = LOAD(a + i);                                            \
      VT vb = LOAD(b + i);                                            \
      UT mask = CMPGT(va, vb);                                        \
      STORE(out + i, BSL(mask, one, zero));                           \
    }                                                                 \
    for (; i < n; i++)                                                \
      out[i] = (CT)(a[i] > b[i]);                                     \
  }                                                                   \
  static inline void _fast_lt_##SFX##_neon(                           \
      const void *restrict ap, const void *restrict bp,               \
      void *restrict op, size_t n) {                                  \
    const CT *a = (const CT *)ap;                                     \
    const CT *b = (const CT *)bp;                                     \
    CT *out = (CT *)op;                                               \
    const VT one = SET1((CT)1.0);                                     \
    const VT zero = SET1((CT)0.0);                                    \
    size_t i = 0;                                                     \
    for (; i + (VPV) <= n; i += (VPV)) {                              \
      VT va = LOAD(a + i);                                            \
      VT vb = LOAD(b + i);                                            \
      UT mask = CMPLT(va, vb);                                        \
      STORE(out + i, BSL(mask, one, zero));                           \
    }                                                                 \
    for (; i < n; i++)                                                \
      out[i] = (CT)(a[i] < b[i]);                                     \
  }                                                                   \
  static inline void _fast_ge_##SFX##_neon(                           \
      const void *restrict ap, const void *restrict bp,               \
      void *restrict op, size_t n) {                                  \
    const CT *a = (const CT *)ap;                                     \
    const CT *b = (const CT *)bp;                                     \
    CT *out = (CT *)op;                                               \
    const VT one = SET1((CT)1.0);                                     \
    const VT zero = SET1((CT)0.0);                                    \
    size_t i = 0;                                                     \
    for (; i + (VPV) <= n; i += (VPV)) {                              \
      VT va = LOAD(a + i);                                            \
      VT vb = LOAD(b + i);                                            \
      UT mask = CMPGE(va, vb);                                        \
      STORE(out + i, BSL(mask, one, zero));                           \
    }                                                                 \
    for (; i < n; i++)                                                \
      out[i] = (CT)(a[i] >= b[i]);                                    \
  }                                                                   \
  static inline void _fast_le_##SFX##_neon(                           \
      const void *restrict ap, const void *restrict bp,               \
      void *restrict op, size_t n) {                                  \
    const CT *a = (const CT *)ap;                                     \
    const CT *b = (const CT *)bp;                                     \
    CT *out = (CT *)op;                                               \
    const VT one = SET1((CT)1.0);                                     \
    const VT zero = SET1((CT)0.0);                                    \
    size_t i = 0;                                                     \
    for (; i + (VPV) <= n; i += (VPV)) {                              \
      VT va = LOAD(a + i);                                            \
      VT vb = LOAD(b + i);                                            \
      UT mask = CMPLE(va, vb);                                        \
      STORE(out + i, BSL(mask, one, zero));                           \
    }                                                                 \
    for (; i < n; i++)                                                \
      out[i] = (CT)(a[i] <= b[i]);                                    \
  }

/* --- Float instantiations --- */

/* clang-format off */
FAST_CMP_FLOAT_NEON(f32, float,  float32x4_t, uint32x4_t, 4,
                    vld1q_f32, vst1q_f32,
                    vceqq_f32, vcgtq_f32, vcltq_f32,
                    vcgeq_f32, vcleq_f32,
                    vdupq_n_f32, vbslq_f32)

FAST_CMP_FLOAT_NEON(f64, double, float64x2_t, uint64x2_t, 2,
                    vld1q_f64, vst1q_f64,
                    vceqq_f64, vcgtq_f64, vcltq_f64,
                    vcgeq_f64, vcleq_f64,
                    vdupq_n_f64, vbslq_f64)
/* clang-format on */

#undef FAST_CMP_FLOAT_NEON

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
