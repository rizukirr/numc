#ifndef NUMC_DOT_SVE_H
#define NUMC_DOT_SVE_H

#include <arm_sve.h>
#include <stdint.h>
#include <stddef.h>

// clang-format off

/*
 * SVE dot products — scalable vector length.
 *
 * SVE vectors have hardware-determined length (128–2048 bits).
 * svcntw() returns elements per vector for 32-bit, etc.
 * All loops use predicated operations with svwhilelt for clean tail handling.
 */

/* ── float dot products ──────────────────────────────────────────────── */

static inline void dot_f32u_sve(const float *a, const float *b, size_t n,
                                float *dest) {
  svfloat32_t acc0 = svdup_f32(0), acc1 = svdup_f32(0),
              acc2 = svdup_f32(0), acc3 = svdup_f32(0);
  size_t vl = svcntw();
  size_t stride = vl * 4;

  size_t i = 0;
  for (; i + stride <= n; i += stride) {
    svbool_t pt = svptrue_b32();
    acc0 = svmla_f32_x(pt, acc0, svld1_f32(pt, a+i),        svld1_f32(pt, b+i));
    acc1 = svmla_f32_x(pt, acc1, svld1_f32(pt, a+i+vl),     svld1_f32(pt, b+i+vl));
    acc2 = svmla_f32_x(pt, acc2, svld1_f32(pt, a+i+vl*2),   svld1_f32(pt, b+i+vl*2));
    acc3 = svmla_f32_x(pt, acc3, svld1_f32(pt, a+i+vl*3),   svld1_f32(pt, b+i+vl*3));
  }
  acc0 = svadd_f32_x(svptrue_b32(), acc0, acc1);
  acc2 = svadd_f32_x(svptrue_b32(), acc2, acc3);
  acc0 = svadd_f32_x(svptrue_b32(), acc0, acc2);

  /* Predicated tail */
  for (; i < n; i += vl) {
    svbool_t pg = svwhilelt_b32((uint32_t)i, (uint32_t)n);
    acc0 = svmla_f32_m(pg, acc0, svld1_f32(pg, a+i), svld1_f32(pg, b+i));
  }

  *dest = svaddv_f32(svptrue_b32(), acc0);
}

static inline void dot_f64u_sve(const double *a, const double *b, size_t n,
                                double *dest) {
  svfloat64_t acc0 = svdup_f64(0), acc1 = svdup_f64(0),
              acc2 = svdup_f64(0), acc3 = svdup_f64(0);
  size_t vl = svcntd();
  size_t stride = vl * 4;

  size_t i = 0;
  for (; i + stride <= n; i += stride) {
    svbool_t pt = svptrue_b64();
    acc0 = svmla_f64_x(pt, acc0, svld1_f64(pt, a+i),        svld1_f64(pt, b+i));
    acc1 = svmla_f64_x(pt, acc1, svld1_f64(pt, a+i+vl),     svld1_f64(pt, b+i+vl));
    acc2 = svmla_f64_x(pt, acc2, svld1_f64(pt, a+i+vl*2),   svld1_f64(pt, b+i+vl*2));
    acc3 = svmla_f64_x(pt, acc3, svld1_f64(pt, a+i+vl*3),   svld1_f64(pt, b+i+vl*3));
  }
  acc0 = svadd_f64_x(svptrue_b64(), acc0, acc1);
  acc2 = svadd_f64_x(svptrue_b64(), acc2, acc3);
  acc0 = svadd_f64_x(svptrue_b64(), acc0, acc2);

  for (; i < n; i += vl) {
    svbool_t pg = svwhilelt_b64((uint32_t)i, (uint32_t)n);
    acc0 = svmla_f64_m(pg, acc0, svld1_f64(pg, a+i), svld1_f64(pg, b+i));
  }

  *dest = svaddv_f64(svptrue_b64(), acc0);
}

/* ── 32-bit integer dot products ─────────────────────────────────────── */

static inline void dot_i32_sve(const int32_t *a, const int32_t *b, size_t n,
                               int32_t *dest) {
  svint32_t acc0 = svdup_s32(0), acc1 = svdup_s32(0),
            acc2 = svdup_s32(0), acc3 = svdup_s32(0);
  size_t vl = svcntw();
  size_t stride = vl * 4;

  size_t i = 0;
  for (; i + stride <= n; i += stride) {
    svbool_t pt = svptrue_b32();
    acc0 = svmla_s32_x(pt, acc0, svld1_s32(pt, a+i),        svld1_s32(pt, b+i));
    acc1 = svmla_s32_x(pt, acc1, svld1_s32(pt, a+i+vl),     svld1_s32(pt, b+i+vl));
    acc2 = svmla_s32_x(pt, acc2, svld1_s32(pt, a+i+vl*2),   svld1_s32(pt, b+i+vl*2));
    acc3 = svmla_s32_x(pt, acc3, svld1_s32(pt, a+i+vl*3),   svld1_s32(pt, b+i+vl*3));
  }
  acc0 = svadd_s32_x(svptrue_b32(), acc0, acc1);
  acc2 = svadd_s32_x(svptrue_b32(), acc2, acc3);
  acc0 = svadd_s32_x(svptrue_b32(), acc0, acc2);

  for (; i < n; i += vl) {
    svbool_t pg = svwhilelt_b32((uint32_t)i, (uint32_t)n);
    acc0 = svmla_s32_m(pg, acc0, svld1_s32(pg, a+i), svld1_s32(pg, b+i));
  }

  *dest = (int32_t)svaddv_s32(svptrue_b32(), acc0);
}

static inline void dot_u32_sve(const uint32_t *a, const uint32_t *b, size_t n,
                               uint32_t *dest) {
  svuint32_t acc0 = svdup_u32(0), acc1 = svdup_u32(0),
             acc2 = svdup_u32(0), acc3 = svdup_u32(0);
  size_t vl = svcntw();
  size_t stride = vl * 4;

  size_t i = 0;
  for (; i + stride <= n; i += stride) {
    svbool_t pt = svptrue_b32();
    acc0 = svmla_u32_x(pt, acc0, svld1_u32(pt, a+i),        svld1_u32(pt, b+i));
    acc1 = svmla_u32_x(pt, acc1, svld1_u32(pt, a+i+vl),     svld1_u32(pt, b+i+vl));
    acc2 = svmla_u32_x(pt, acc2, svld1_u32(pt, a+i+vl*2),   svld1_u32(pt, b+i+vl*2));
    acc3 = svmla_u32_x(pt, acc3, svld1_u32(pt, a+i+vl*3),   svld1_u32(pt, b+i+vl*3));
  }
  acc0 = svadd_u32_x(svptrue_b32(), acc0, acc1);
  acc2 = svadd_u32_x(svptrue_b32(), acc2, acc3);
  acc0 = svadd_u32_x(svptrue_b32(), acc0, acc2);

  for (; i < n; i += vl) {
    svbool_t pg = svwhilelt_b32((uint32_t)i, (uint32_t)n);
    acc0 = svmla_u32_m(pg, acc0, svld1_u32(pg, a+i), svld1_u32(pg, b+i));
  }

  *dest = (uint32_t)svaddv_u32(svptrue_b32(), acc0);
}

/* ── 64-bit integer dot products ─────────────────────────────────────── */

static inline void dot_i64_sve(const int64_t *a, const int64_t *b, size_t n,
                               int64_t *dest) {
  svint64_t acc0 = svdup_s64(0), acc1 = svdup_s64(0),
            acc2 = svdup_s64(0), acc3 = svdup_s64(0);
  size_t vl = svcntd();
  size_t stride = vl * 4;

  size_t i = 0;
  for (; i + stride <= n; i += stride) {
    svbool_t pt = svptrue_b64();
    acc0 = svmla_s64_x(pt, acc0, svld1_s64(pt, a+i),        svld1_s64(pt, b+i));
    acc1 = svmla_s64_x(pt, acc1, svld1_s64(pt, a+i+vl),     svld1_s64(pt, b+i+vl));
    acc2 = svmla_s64_x(pt, acc2, svld1_s64(pt, a+i+vl*2),   svld1_s64(pt, b+i+vl*2));
    acc3 = svmla_s64_x(pt, acc3, svld1_s64(pt, a+i+vl*3),   svld1_s64(pt, b+i+vl*3));
  }
  acc0 = svadd_s64_x(svptrue_b64(), acc0, acc1);
  acc2 = svadd_s64_x(svptrue_b64(), acc2, acc3);
  acc0 = svadd_s64_x(svptrue_b64(), acc0, acc2);

  for (; i < n; i += vl) {
    svbool_t pg = svwhilelt_b64((uint32_t)i, (uint32_t)n);
    acc0 = svmla_s64_m(pg, acc0, svld1_s64(pg, a+i), svld1_s64(pg, b+i));
  }

  *dest = svaddv_s64(svptrue_b64(), acc0);
}

static inline void dot_u64_sve(const uint64_t *a, const uint64_t *b, size_t n,
                               uint64_t *dest) {
  svuint64_t acc0 = svdup_u64(0), acc1 = svdup_u64(0),
             acc2 = svdup_u64(0), acc3 = svdup_u64(0);
  size_t vl = svcntd();
  size_t stride = vl * 4;

  size_t i = 0;
  for (; i + stride <= n; i += stride) {
    svbool_t pt = svptrue_b64();
    acc0 = svmla_u64_x(pt, acc0, svld1_u64(pt, a+i),        svld1_u64(pt, b+i));
    acc1 = svmla_u64_x(pt, acc1, svld1_u64(pt, a+i+vl),     svld1_u64(pt, b+i+vl));
    acc2 = svmla_u64_x(pt, acc2, svld1_u64(pt, a+i+vl*2),   svld1_u64(pt, b+i+vl*2));
    acc3 = svmla_u64_x(pt, acc3, svld1_u64(pt, a+i+vl*3),   svld1_u64(pt, b+i+vl*3));
  }
  acc0 = svadd_u64_x(svptrue_b64(), acc0, acc1);
  acc2 = svadd_u64_x(svptrue_b64(), acc2, acc3);
  acc0 = svadd_u64_x(svptrue_b64(), acc0, acc2);

  for (; i < n; i += vl) {
    svbool_t pg = svwhilelt_b64((uint32_t)i, (uint32_t)n);
    acc0 = svmla_u64_m(pg, acc0, svld1_u64(pg, a+i), svld1_u64(pg, b+i));
  }

  *dest = svaddv_u64(svptrue_b64(), acc0);
}

/* ── 8-bit dot (widen i8→i16, multiply, pairwise-add into i32) ───────── */

static inline void dot_i8_sve(const int8_t *a, const int8_t *b, size_t n,
                              int8_t *dest) {
  svint32_t acc = svdup_s32(0);
  size_t vl8 = svcntb();  /* elements per i8 vector */

  size_t i = 0;
  for (; i + vl8 <= n; i += vl8) {
    svbool_t pt = svptrue_b8();
    svint8_t va = svld1_s8(pt, a+i);
    svint8_t vb = svld1_s8(pt, b+i);
    /* widen low half: i8→i16 */
    svint16_t va_lo = svunpklo_s16(va);
    svint16_t vb_lo = svunpklo_s16(vb);
    svint16_t va_hi = svunpkhi_s16(va);
    svint16_t vb_hi = svunpkhi_s16(vb);
    /* multiply i16×i16→i16 (low bits), then widen-add into i32 */
    svint16_t p_lo = svmul_s16_x(svptrue_b16(), va_lo, vb_lo);
    svint16_t p_hi = svmul_s16_x(svptrue_b16(), va_hi, vb_hi);
    /* pairwise widening add: i16→i32 */
    acc = svadd_s32_x(svptrue_b32(), acc, svunpklo_s32(p_lo));
    acc = svadd_s32_x(svptrue_b32(), acc, svunpkhi_s32(p_lo));
    acc = svadd_s32_x(svptrue_b32(), acc, svunpklo_s32(p_hi));
    acc = svadd_s32_x(svptrue_b32(), acc, svunpkhi_s32(p_hi));
  }

  int result = (int)svaddv_s32(svptrue_b32(), acc);
  int tail = 0;
  for (; i < n; i++) tail += a[i] * b[i];
  *dest = (int8_t)(result + tail);
}

static inline void dot_u8_sve(const uint8_t *a, const uint8_t *b, size_t n,
                              uint8_t *dest) {
  svuint32_t acc = svdup_u32(0);
  size_t vl8 = svcntb();

  size_t i = 0;
  for (; i + vl8 <= n; i += vl8) {
    svbool_t pt = svptrue_b8();
    svuint8_t va = svld1_u8(pt, a+i);
    svuint8_t vb = svld1_u8(pt, b+i);
    svuint16_t va_lo = svunpklo_u16(va);
    svuint16_t vb_lo = svunpklo_u16(vb);
    svuint16_t va_hi = svunpkhi_u16(va);
    svuint16_t vb_hi = svunpkhi_u16(vb);
    svuint16_t p_lo = svmul_u16_x(svptrue_b16(), va_lo, vb_lo);
    svuint16_t p_hi = svmul_u16_x(svptrue_b16(), va_hi, vb_hi);
    acc = svadd_u32_x(svptrue_b32(), acc, svunpklo_u32(p_lo));
    acc = svadd_u32_x(svptrue_b32(), acc, svunpkhi_u32(p_lo));
    acc = svadd_u32_x(svptrue_b32(), acc, svunpklo_u32(p_hi));
    acc = svadd_u32_x(svptrue_b32(), acc, svunpkhi_u32(p_hi));
  }

  uint32_t result = (uint32_t)svaddv_u32(svptrue_b32(), acc);
  uint8_t tail = 0;
  for (; i < n; i++) tail += (uint8_t)(a[i] * b[i]);
  *dest = (uint8_t)(result + tail);
}

/* ── 16-bit dot (widen i16→i32, multiply-accumulate) ─────────────────── */

static inline void dot_i16_sve(const int16_t *a, const int16_t *b, size_t n,
                               int16_t *dest) {
  svint32_t acc0 = svdup_s32(0), acc1 = svdup_s32(0);
  size_t vl16 = svcnth();  /* elements per i16 vector */

  size_t i = 0;
  for (; i + vl16 <= n; i += vl16) {
    svbool_t pt = svptrue_b16();
    svint16_t va = svld1_s16(pt, a+i);
    svint16_t vb = svld1_s16(pt, b+i);
    /* widen halves to i32, multiply-accumulate */
    acc0 = svmla_s32_x(svptrue_b32(), acc0,
                        svmovlb_s32(va), svmovlb_s32(vb));
    acc1 = svmla_s32_x(svptrue_b32(), acc1,
                        svmovlt_s32(va), svmovlt_s32(vb));
  }
  acc0 = svadd_s32_x(svptrue_b32(), acc0, acc1);

  int result = (int)svaddv_s32(svptrue_b32(), acc0);
  int tail = 0;
  for (; i < n; i++) tail += a[i] * b[i];
  *dest = (int16_t)(result + tail);
}

static inline void dot_u16_sve(const uint16_t *a, const uint16_t *b, size_t n,
                               uint16_t *dest) {
  svuint32_t acc0 = svdup_u32(0), acc1 = svdup_u32(0);
  size_t vl16 = svcnth();

  size_t i = 0;
  for (; i + vl16 <= n; i += vl16) {
    svbool_t pt = svptrue_b16();
    svuint16_t va = svld1_u16(pt, a+i);
    svuint16_t vb = svld1_u16(pt, b+i);
    acc0 = svmla_u32_x(svptrue_b32(), acc0,
                        svmovlb_u32(va), svmovlb_u32(vb));
    acc1 = svmla_u32_x(svptrue_b32(), acc1,
                        svmovlt_u32(va), svmovlt_u32(vb));
  }
  acc0 = svadd_u32_x(svptrue_b32(), acc0, acc1);

  uint32_t result = (uint32_t)svaddv_u32(svptrue_b32(), acc0);
  uint16_t tail = 0;
  for (; i < n; i++) tail += (uint16_t)(a[i] * b[i]);
  *dest = (uint16_t)(result + tail);
}

// clang-format on

#endif
