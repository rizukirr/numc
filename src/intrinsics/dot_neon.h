#ifndef NUMC_DOT_NEON_H
#define NUMC_DOT_NEON_H

#include <arm_neon.h>
#include <stdint.h>

// clang-format off

/* -- helpers ----------------------------------------------------------- */

#define PREFETCH_NEON(p, off) __builtin_prefetch((const char *)((p) + (off)), 0, 3)
#define PREFETCH2_NEON(a, b, off) PREFETCH_NEON(a, off); PREFETCH_NEON(b, off)

#define REDUCE_ACC4(add_fn, a0, a1, a2, a3) \
  do {                                       \
    a0 = add_fn(a0, a1);                     \
    a2 = add_fn(a2, a3);                     \
    a0 = add_fn(a0, a2);                     \
  } while (0)

/* NEON has no 64-bit integer multiply — emulate with lane extract */
static inline int64x2_t _vmul_s64(int64x2_t a, int64x2_t b) {
  return vcombine_s64(
      vdup_n_s64(vgetq_lane_s64(a, 0) * vgetq_lane_s64(b, 0)),
      vdup_n_s64(vgetq_lane_s64(a, 1) * vgetq_lane_s64(b, 1)));
}

static inline uint64x2_t _vmul_u64(uint64x2_t a, uint64x2_t b) {
  return vcombine_u64(
      vdup_n_u64(vgetq_lane_u64(a, 0) * vgetq_lane_u64(b, 0)),
      vdup_n_u64(vgetq_lane_u64(a, 1) * vgetq_lane_u64(b, 1)));
}

static inline int64_t _hsum_s64(int64x2_t v) {
  return vgetq_lane_s64(v, 0) + vgetq_lane_s64(v, 1);
}

static inline uint64_t _hsum_u64(uint64x2_t v) {
  return vgetq_lane_u64(v, 0) + vgetq_lane_u64(v, 1);
}

/* -- float dot products ------------------------------------------------ */

static inline void dot_f32u_neon(const float *a, const float *b, size_t n,
                                 float *dest) {
  float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0),
              acc2 = vdupq_n_f32(0), acc3 = vdupq_n_f32(0);

  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    PREFETCH2_NEON(a, b, i + 64);
    acc0 = vfmaq_f32(acc0, vld1q_f32(a+i),    vld1q_f32(b+i));
    acc1 = vfmaq_f32(acc1, vld1q_f32(a+i+4),  vld1q_f32(b+i+4));
    acc2 = vfmaq_f32(acc2, vld1q_f32(a+i+8),  vld1q_f32(b+i+8));
    acc3 = vfmaq_f32(acc3, vld1q_f32(a+i+12), vld1q_f32(b+i+12));
  }
  REDUCE_ACC4(vaddq_f32, acc0, acc1, acc2, acc3);

  float result = vaddvq_f32(acc0);
  float tail = 0.0f;
  for (; i < n; i++) tail += a[i] * b[i];
  *dest = result + tail;
}

static inline void dot_f64u_neon(const double *a, const double *b, size_t n,
                                 double *dest) {
  float64x2_t acc0 = vdupq_n_f64(0), acc1 = vdupq_n_f64(0),
              acc2 = vdupq_n_f64(0), acc3 = vdupq_n_f64(0);

  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    PREFETCH2_NEON(a, b, i + 32);
    acc0 = vfmaq_f64(acc0, vld1q_f64(a+i),   vld1q_f64(b+i));
    acc1 = vfmaq_f64(acc1, vld1q_f64(a+i+2), vld1q_f64(b+i+2));
    acc2 = vfmaq_f64(acc2, vld1q_f64(a+i+4), vld1q_f64(b+i+4));
    acc3 = vfmaq_f64(acc3, vld1q_f64(a+i+6), vld1q_f64(b+i+6));
  }
  REDUCE_ACC4(vaddq_f64, acc0, acc1, acc2, acc3);

  double result = vaddvq_f64(acc0);
  double tail = 0.0;
  for (; i < n; i++) tail += a[i] * b[i];
  *dest = result + tail;
}

/* -- 32-bit integer dot products --------------------------------------- */

static inline void dot_i32_neon(const int32_t *a, const int32_t *b, size_t n,
                                int32_t *dest) {
  int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0),
            acc2 = vdupq_n_s32(0), acc3 = vdupq_n_s32(0);

  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    PREFETCH2_NEON(a, b, i + 64);
    acc0 = vmlaq_s32(acc0, vld1q_s32(a+i),    vld1q_s32(b+i));
    acc1 = vmlaq_s32(acc1, vld1q_s32(a+i+4),  vld1q_s32(b+i+4));
    acc2 = vmlaq_s32(acc2, vld1q_s32(a+i+8),  vld1q_s32(b+i+8));
    acc3 = vmlaq_s32(acc3, vld1q_s32(a+i+12), vld1q_s32(b+i+12));
  }
  REDUCE_ACC4(vaddq_s32, acc0, acc1, acc2, acc3);

  int32_t result = vaddvq_s32(acc0);
  int32_t tail = 0;
  for (; i < n; i++) tail += a[i] * b[i];
  *dest = result + tail;
}

static inline void dot_u32_neon(const uint32_t *a, const uint32_t *b, size_t n,
                                uint32_t *dest) {
  uint32x4_t acc0 = vdupq_n_u32(0), acc1 = vdupq_n_u32(0),
             acc2 = vdupq_n_u32(0), acc3 = vdupq_n_u32(0);

  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    PREFETCH2_NEON(a, b, i + 64);
    acc0 = vmlaq_u32(acc0, vld1q_u32(a+i),    vld1q_u32(b+i));
    acc1 = vmlaq_u32(acc1, vld1q_u32(a+i+4),  vld1q_u32(b+i+4));
    acc2 = vmlaq_u32(acc2, vld1q_u32(a+i+8),  vld1q_u32(b+i+8));
    acc3 = vmlaq_u32(acc3, vld1q_u32(a+i+12), vld1q_u32(b+i+12));
  }
  REDUCE_ACC4(vaddq_u32, acc0, acc1, acc2, acc3);

  uint32_t result = vaddvq_u32(acc0);
  uint32_t tail = 0;
  for (; i < n; i++) tail += a[i] * b[i];
  *dest = result + tail;
}

/* -- 64-bit integer dot products --------------------------------------- */

static inline void dot_i64_neon(const int64_t *a, const int64_t *b, size_t n,
                                int64_t *dest) {
  int64x2_t acc0 = vdupq_n_s64(0), acc1 = vdupq_n_s64(0),
            acc2 = vdupq_n_s64(0), acc3 = vdupq_n_s64(0);

  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    PREFETCH2_NEON(a, b, i + 32);
    acc0 = vaddq_s64(acc0, _vmul_s64(vld1q_s64(a+i),   vld1q_s64(b+i)));
    acc1 = vaddq_s64(acc1, _vmul_s64(vld1q_s64(a+i+2), vld1q_s64(b+i+2)));
    acc2 = vaddq_s64(acc2, _vmul_s64(vld1q_s64(a+i+4), vld1q_s64(b+i+4)));
    acc3 = vaddq_s64(acc3, _vmul_s64(vld1q_s64(a+i+6), vld1q_s64(b+i+6)));
  }
  REDUCE_ACC4(vaddq_s64, acc0, acc1, acc2, acc3);

  int64_t result = _hsum_s64(acc0);
  int64_t tail = 0;
  for (; i < n; i++) tail += a[i] * b[i];
  *dest = result + tail;
}

static inline void dot_u64_neon(const uint64_t *a, const uint64_t *b, size_t n,
                                uint64_t *dest) {
  uint64x2_t acc0 = vdupq_n_u64(0), acc1 = vdupq_n_u64(0),
             acc2 = vdupq_n_u64(0), acc3 = vdupq_n_u64(0);

  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    PREFETCH2_NEON(a, b, i + 32);
    acc0 = vaddq_u64(acc0, _vmul_u64(vld1q_u64(a+i),   vld1q_u64(b+i)));
    acc1 = vaddq_u64(acc1, _vmul_u64(vld1q_u64(a+i+2), vld1q_u64(b+i+2)));
    acc2 = vaddq_u64(acc2, _vmul_u64(vld1q_u64(a+i+4), vld1q_u64(b+i+4)));
    acc3 = vaddq_u64(acc3, _vmul_u64(vld1q_u64(a+i+6), vld1q_u64(b+i+6)));
  }
  REDUCE_ACC4(vaddq_u64, acc0, acc1, acc2, acc3);

  uint64_t result = _hsum_u64(acc0);
  uint64_t tail = 0;
  for (; i < n; i++) tail += a[i] * b[i];
  *dest = result + tail;
}

/* -- 8-bit dot (widen i8→i16, multiply, widen-accumulate i16→i32) ------ */

static inline void dot_i8_neon(const int8_t *a, const int8_t *b, size_t n,
                               int8_t *dest) {
  int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0),
            acc2 = vdupq_n_s32(0), acc3 = vdupq_n_s32(0);

  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    PREFETCH2_NEON(a, b, i + 64);
    /* vmull_s8: 8×i8 × 8×i8 → 8×i16, then vaddw halves into i32 */
    int16x8_t p0 = vmull_s8(vld1_s8(a+i),    vld1_s8(b+i));
    int16x8_t p1 = vmull_s8(vld1_s8(a+i+8),  vld1_s8(b+i+8));
    int16x8_t p2 = vmull_s8(vld1_s8(a+i+16), vld1_s8(b+i+16));
    int16x8_t p3 = vmull_s8(vld1_s8(a+i+24), vld1_s8(b+i+24));
    acc0 = vpadalq_s16(acc0, p0);
    acc1 = vpadalq_s16(acc1, p1);
    acc2 = vpadalq_s16(acc2, p2);
    acc3 = vpadalq_s16(acc3, p3);
  }
  REDUCE_ACC4(vaddq_s32, acc0, acc1, acc2, acc3);

  int result = vaddvq_s32(acc0);
  int tail = 0;
  for (; i < n; i++) tail += a[i] * b[i];
  *dest = (int8_t)(result + tail);
}

static inline void dot_u8_neon(const uint8_t *a, const uint8_t *b, size_t n,
                               uint8_t *dest) {
  uint32x4_t acc0 = vdupq_n_u32(0), acc1 = vdupq_n_u32(0),
             acc2 = vdupq_n_u32(0), acc3 = vdupq_n_u32(0);

  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    PREFETCH2_NEON(a, b, i + 64);
    uint16x8_t p0 = vmull_u8(vld1_u8(a+i),    vld1_u8(b+i));
    uint16x8_t p1 = vmull_u8(vld1_u8(a+i+8),  vld1_u8(b+i+8));
    uint16x8_t p2 = vmull_u8(vld1_u8(a+i+16), vld1_u8(b+i+16));
    uint16x8_t p3 = vmull_u8(vld1_u8(a+i+24), vld1_u8(b+i+24));
    acc0 = vpadalq_u16(acc0, p0);
    acc1 = vpadalq_u16(acc1, p1);
    acc2 = vpadalq_u16(acc2, p2);
    acc3 = vpadalq_u16(acc3, p3);
  }
  REDUCE_ACC4(vaddq_u32, acc0, acc1, acc2, acc3);

  uint32_t result = vaddvq_u32(acc0);
  uint8_t tail = 0;
  for (; i < n; i++) tail += (uint8_t)(a[i] * b[i]);
  *dest = (uint8_t)(result + tail);
}

/* -- 16-bit dot (vmlal: i16→i32 widening multiply-accumulate) ---------- */

static inline void dot_i16_neon(const int16_t *a, const int16_t *b, size_t n,
                                int16_t *dest) {
  int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0),
            acc2 = vdupq_n_s32(0), acc3 = vdupq_n_s32(0);

  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    PREFETCH2_NEON(a, b, i + 32);
    acc0 = vmlal_s16(acc0, vld1_s16(a+i),    vld1_s16(b+i));
    acc1 = vmlal_s16(acc1, vld1_s16(a+i+4),  vld1_s16(b+i+4));
    acc2 = vmlal_s16(acc2, vld1_s16(a+i+8),  vld1_s16(b+i+8));
    acc3 = vmlal_s16(acc3, vld1_s16(a+i+12), vld1_s16(b+i+12));
  }
  REDUCE_ACC4(vaddq_s32, acc0, acc1, acc2, acc3);

  int result = vaddvq_s32(acc0);
  int tail = 0;
  for (; i < n; i++) tail += a[i] * b[i];
  *dest = (int16_t)(result + tail);
}

static inline void dot_u16_neon(const uint16_t *a, const uint16_t *b, size_t n,
                                uint16_t *dest) {
  uint32x4_t acc0 = vdupq_n_u32(0), acc1 = vdupq_n_u32(0),
             acc2 = vdupq_n_u32(0), acc3 = vdupq_n_u32(0);

  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    PREFETCH2_NEON(a, b, i + 32);
    acc0 = vmlal_u16(acc0, vld1_u16(a+i),    vld1_u16(b+i));
    acc1 = vmlal_u16(acc1, vld1_u16(a+i+4),  vld1_u16(b+i+4));
    acc2 = vmlal_u16(acc2, vld1_u16(a+i+8),  vld1_u16(b+i+8));
    acc3 = vmlal_u16(acc3, vld1_u16(a+i+12), vld1_u16(b+i+12));
  }
  REDUCE_ACC4(vaddq_u32, acc0, acc1, acc2, acc3);

  uint32_t result = vaddvq_u32(acc0);
  uint16_t tail = 0;
  for (; i < n; i++) tail += (uint16_t)(a[i] * b[i]);
  *dest = (uint16_t)(result + tail);
}

// clang-format on

#undef PREFETCH_NEON
#undef PREFETCH2_NEON
#undef REDUCE_ACC4

#endif
