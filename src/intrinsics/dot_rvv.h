#ifndef NUMC_DOT_RVV_H
#define NUMC_DOT_RVV_H

#include <riscv_vector.h>
#include <stdint.h>

// clang-format off

/*
 * RISC-V Vector (RVV 1.0) dot products — scalable vector length.
 *
 * RVV uses vsetvl to dynamically set vector length each iteration.
 * The hardware decides how many elements fit; the loop naturally
 * handles tails with no scalar fallback needed.
 */

/* ── float dot products ──────────────────────────────────────────────── */

static inline void dot_f32u_rvv(const float *a, const float *b, size_t n,
                                float *dest) {
  size_t vl;
  vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1());

  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m4(n - i);
    vfloat32m4_t va = __riscv_vle32_v_f32m4(a + i, vl);
    vfloat32m4_t vb = __riscv_vle32_v_f32m4(b + i, vl);
    acc = __riscv_vfredusum_vs_f32m4_f32m1(
        __riscv_vfmul_vv_f32m4(va, vb, vl), acc, vl);
  }

  *dest = __riscv_vfmv_f_s_f32m1_f32(acc);
}

static inline void dot_f64u_rvv(const double *a, const double *b, size_t n,
                                double *dest) {
  size_t vl;
  vfloat64m1_t acc = __riscv_vfmv_v_f_f64m1(0.0, __riscv_vsetvlmax_e64m1());

  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e64m4(n - i);
    vfloat64m4_t va = __riscv_vle64_v_f64m4(a + i, vl);
    vfloat64m4_t vb = __riscv_vle64_v_f64m4(b + i, vl);
    acc = __riscv_vfredusum_vs_f64m4_f64m1(
        __riscv_vfmul_vv_f64m4(va, vb, vl), acc, vl);
  }

  *dest = __riscv_vfmv_f_s_f64m1_f64(acc);
}

/* ── 32-bit integer dot products ─────────────────────────────────────── */

static inline void dot_i32_rvv(const int32_t *a, const int32_t *b, size_t n,
                               int32_t *dest) {
  size_t vl;
  vint32m1_t acc = __riscv_vmv_v_x_i32m1(0, __riscv_vsetvlmax_e32m1());

  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m4(n - i);
    vint32m4_t va = __riscv_vle32_v_i32m4(a + i, vl);
    vint32m4_t vb = __riscv_vle32_v_i32m4(b + i, vl);
    acc = __riscv_vredsum_vs_i32m4_i32m1(
        __riscv_vmul_vv_i32m4(va, vb, vl), acc, vl);
  }

  *dest = __riscv_vmv_x_s_i32m1_i32(acc);
}

static inline void dot_u32_rvv(const uint32_t *a, const uint32_t *b, size_t n,
                               uint32_t *dest) {
  size_t vl;
  vuint32m1_t acc = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvlmax_e32m1());

  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m4(n - i);
    vuint32m4_t va = __riscv_vle32_v_u32m4(a + i, vl);
    vuint32m4_t vb = __riscv_vle32_v_u32m4(b + i, vl);
    acc = __riscv_vredsum_vs_u32m4_u32m1(
        __riscv_vmul_vv_u32m4(va, vb, vl), acc, vl);
  }

  *dest = __riscv_vmv_x_s_u32m1_u32(acc);
}

/* ── 64-bit integer dot products ─────────────────────────────────────── */

static inline void dot_i64_rvv(const int64_t *a, const int64_t *b, size_t n,
                               int64_t *dest) {
  size_t vl;
  vint64m1_t acc = __riscv_vmv_v_x_i64m1(0, __riscv_vsetvlmax_e64m1());

  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e64m4(n - i);
    vint64m4_t va = __riscv_vle64_v_i64m4(a + i, vl);
    vint64m4_t vb = __riscv_vle64_v_i64m4(b + i, vl);
    acc = __riscv_vredsum_vs_i64m4_i64m1(
        __riscv_vmul_vv_i64m4(va, vb, vl), acc, vl);
  }

  *dest = __riscv_vmv_x_s_i64m1_i64(acc);
}

static inline void dot_u64_rvv(const uint64_t *a, const uint64_t *b, size_t n,
                               uint64_t *dest) {
  size_t vl;
  vuint64m1_t acc = __riscv_vmv_v_x_u64m1(0, __riscv_vsetvlmax_e64m1());

  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e64m4(n - i);
    vuint64m4_t va = __riscv_vle64_v_u64m4(a + i, vl);
    vuint64m4_t vb = __riscv_vle64_v_u64m4(b + i, vl);
    acc = __riscv_vredsum_vs_u64m4_u64m1(
        __riscv_vmul_vv_u64m4(va, vb, vl), acc, vl);
  }

  *dest = __riscv_vmv_x_s_u64m1_u64(acc);
}

/* ── 8-bit dot (widen i8→i16, multiply, widen-reduce into i32) ───────── */

static inline void dot_i8_rvv(const int8_t *a, const int8_t *b, size_t n,
                              int8_t *dest) {
  size_t vl;
  vint32m1_t acc = __riscv_vmv_v_x_i32m1(0, __riscv_vsetvlmax_e32m1());

  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e8m2(n - i);
    vint8m2_t va = __riscv_vle8_v_i8m2(a + i, vl);
    vint8m2_t vb = __riscv_vle8_v_i8m2(b + i, vl);
    /* widen i8→i16, multiply */
    vint16m4_t prod = __riscv_vwmul_vv_i16m4(va, vb, vl);
    /* widen i16→i32 for reduction */
    vint32m8_t prod32 = __riscv_vsext_vf2_i32m8(prod, vl);
    acc = __riscv_vredsum_vs_i32m8_i32m1(prod32, acc, vl);
  }

  *dest = (int8_t)__riscv_vmv_x_s_i32m1_i32(acc);
}

static inline void dot_u8_rvv(const uint8_t *a, const uint8_t *b, size_t n,
                              uint8_t *dest) {
  size_t vl;
  vuint32m1_t acc = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvlmax_e32m1());

  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e8m2(n - i);
    vuint8m2_t va = __riscv_vle8_v_u8m2(a + i, vl);
    vuint8m2_t vb = __riscv_vle8_v_u8m2(b + i, vl);
    vuint16m4_t prod = __riscv_vwmulu_vv_u16m4(va, vb, vl);
    vuint32m8_t prod32 = __riscv_vzext_vf2_u32m8(prod, vl);
    acc = __riscv_vredsum_vs_u32m8_u32m1(prod32, acc, vl);
  }

  *dest = (uint8_t)__riscv_vmv_x_s_u32m1_u32(acc);
}

/* ── 16-bit dot (widen i16→i32, multiply-reduce) ─────────────────────── */

static inline void dot_i16_rvv(const int16_t *a, const int16_t *b, size_t n,
                               int16_t *dest) {
  size_t vl;
  vint32m1_t acc = __riscv_vmv_v_x_i32m1(0, __riscv_vsetvlmax_e32m1());

  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m2(n - i);
    vint16m2_t va = __riscv_vle16_v_i16m2(a + i, vl);
    vint16m2_t vb = __riscv_vle16_v_i16m2(b + i, vl);
    /* widening multiply i16×i16→i32, then reduce */
    vint32m4_t prod = __riscv_vwmul_vv_i32m4(va, vb, vl);
    acc = __riscv_vredsum_vs_i32m4_i32m1(prod, acc, vl);
  }

  *dest = (int16_t)__riscv_vmv_x_s_i32m1_i32(acc);
}

static inline void dot_u16_rvv(const uint16_t *a, const uint16_t *b, size_t n,
                               uint16_t *dest) {
  size_t vl;
  vuint32m1_t acc = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvlmax_e32m1());

  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m2(n - i);
    vuint16m2_t va = __riscv_vle16_v_u16m2(a + i, vl);
    vuint16m2_t vb = __riscv_vle16_v_u16m2(b + i, vl);
    vuint32m4_t prod = __riscv_vwmulu_vv_u32m4(va, vb, vl);
    acc = __riscv_vredsum_vs_u32m4_u32m1(prod, acc, vl);
  }

  *dest = (uint16_t)__riscv_vmv_x_s_u32m1_u32(acc);
}

// clang-format on

#endif
