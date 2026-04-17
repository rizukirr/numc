#ifndef NUMC_DOT_SVE2_H
#define NUMC_DOT_SVE2_H

#include <arm_sve.h>
#include <stdint.h>
#include <stddef.h>

// clang-format off

/*
 * SVE2 dot products — leverages SVE2 widening multiply-accumulate instructions.
 *
 * Key SVE2 advantages over SVE:
 * - svmlalb/svmlalt: widening multiply-accumulate (bottom/top interleaved)
 * - Better 8-bit/16-bit pipelines
 *
 * For 32/64-bit and float types, SVE2 uses the same instructions as SVE,
 * so those are thin wrappers. The real wins are in 8-bit and 16-bit paths.
 */

/* -- float dot products (same as SVE) ---------------------------------- */

static inline void dot_f32u_sve2(const float *a, const float *b, size_t n,
                                 float *dest) {
  svfloat32_t acc0 = svdup_f32(0), acc1 = svdup_f32(0),
              acc2 = svdup_f32(0), acc3 = svdup_f32(0);
  size_t vl = svcntw();
  size_t stride = vl * 4;

  size_t i = 0;
  for (; i + stride <= n; i += stride) {
    svbool_t pt = svptrue_b32();
    acc0 = svmla_f32_x(pt, acc0, svld1_f32(pt, a+i),      svld1_f32(pt, b+i));
    acc1 = svmla_f32_x(pt, acc1, svld1_f32(pt, a+i+vl),   svld1_f32(pt, b+i+vl));
    acc2 = svmla_f32_x(pt, acc2, svld1_f32(pt, a+i+vl*2), svld1_f32(pt, b+i+vl*2));
    acc3 = svmla_f32_x(pt, acc3, svld1_f32(pt, a+i+vl*3), svld1_f32(pt, b+i+vl*3));
  }
  acc0 = svadd_f32_x(svptrue_b32(), acc0, acc1);
  acc2 = svadd_f32_x(svptrue_b32(), acc2, acc3);
  acc0 = svadd_f32_x(svptrue_b32(), acc0, acc2);

  for (; i < n; i += vl) {
    svbool_t pg = svwhilelt_b32((uint32_t)i, (uint32_t)n);
    acc0 = svmla_f32_m(pg, acc0, svld1_f32(pg, a+i), svld1_f32(pg, b+i));
  }

  *dest = svaddv_f32(svptrue_b32(), acc0);
}

static inline void dot_f64u_sve2(const double *a, const double *b, size_t n,
                                 double *dest) {
  svfloat64_t acc0 = svdup_f64(0), acc1 = svdup_f64(0),
              acc2 = svdup_f64(0), acc3 = svdup_f64(0);
  size_t vl = svcntd();
  size_t stride = vl * 4;

  size_t i = 0;
  for (; i + stride <= n; i += stride) {
    svbool_t pt = svptrue_b64();
    acc0 = svmla_f64_x(pt, acc0, svld1_f64(pt, a+i),      svld1_f64(pt, b+i));
    acc1 = svmla_f64_x(pt, acc1, svld1_f64(pt, a+i+vl),   svld1_f64(pt, b+i+vl));
    acc2 = svmla_f64_x(pt, acc2, svld1_f64(pt, a+i+vl*2), svld1_f64(pt, b+i+vl*2));
    acc3 = svmla_f64_x(pt, acc3, svld1_f64(pt, a+i+vl*3), svld1_f64(pt, b+i+vl*3));
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

/* -- 32-bit integer dot products (same as SVE) ------------------------- */

static inline void dot_i32_sve2(const int32_t *a, const int32_t *b, size_t n,
                                int32_t *dest) {
  svint32_t acc0 = svdup_s32(0), acc1 = svdup_s32(0),
            acc2 = svdup_s32(0), acc3 = svdup_s32(0);
  size_t vl = svcntw();
  size_t stride = vl * 4;

  size_t i = 0;
  for (; i + stride <= n; i += stride) {
    svbool_t pt = svptrue_b32();
    acc0 = svmla_s32_x(pt, acc0, svld1_s32(pt, a+i),      svld1_s32(pt, b+i));
    acc1 = svmla_s32_x(pt, acc1, svld1_s32(pt, a+i+vl),   svld1_s32(pt, b+i+vl));
    acc2 = svmla_s32_x(pt, acc2, svld1_s32(pt, a+i+vl*2), svld1_s32(pt, b+i+vl*2));
    acc3 = svmla_s32_x(pt, acc3, svld1_s32(pt, a+i+vl*3), svld1_s32(pt, b+i+vl*3));
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

static inline void dot_u32_sve2(const uint32_t *a, const uint32_t *b, size_t n,
                                uint32_t *dest) {
  svuint32_t acc0 = svdup_u32(0), acc1 = svdup_u32(0),
             acc2 = svdup_u32(0), acc3 = svdup_u32(0);
  size_t vl = svcntw();
  size_t stride = vl * 4;

  size_t i = 0;
  for (; i + stride <= n; i += stride) {
    svbool_t pt = svptrue_b32();
    acc0 = svmla_u32_x(pt, acc0, svld1_u32(pt, a+i),      svld1_u32(pt, b+i));
    acc1 = svmla_u32_x(pt, acc1, svld1_u32(pt, a+i+vl),   svld1_u32(pt, b+i+vl));
    acc2 = svmla_u32_x(pt, acc2, svld1_u32(pt, a+i+vl*2), svld1_u32(pt, b+i+vl*2));
    acc3 = svmla_u32_x(pt, acc3, svld1_u32(pt, a+i+vl*3), svld1_u32(pt, b+i+vl*3));
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

/* -- 64-bit integer dot products (same as SVE) ------------------------- */

static inline void dot_i64_sve2(const int64_t *a, const int64_t *b, size_t n,
                                int64_t *dest) {
  svint64_t acc0 = svdup_s64(0), acc1 = svdup_s64(0),
            acc2 = svdup_s64(0), acc3 = svdup_s64(0);
  size_t vl = svcntd();
  size_t stride = vl * 4;

  size_t i = 0;
  for (; i + stride <= n; i += stride) {
    svbool_t pt = svptrue_b64();
    acc0 = svmla_s64_x(pt, acc0, svld1_s64(pt, a+i),      svld1_s64(pt, b+i));
    acc1 = svmla_s64_x(pt, acc1, svld1_s64(pt, a+i+vl),   svld1_s64(pt, b+i+vl));
    acc2 = svmla_s64_x(pt, acc2, svld1_s64(pt, a+i+vl*2), svld1_s64(pt, b+i+vl*2));
    acc3 = svmla_s64_x(pt, acc3, svld1_s64(pt, a+i+vl*3), svld1_s64(pt, b+i+vl*3));
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

static inline void dot_u64_sve2(const uint64_t *a, const uint64_t *b, size_t n,
                                uint64_t *dest) {
  svuint64_t acc0 = svdup_u64(0), acc1 = svdup_u64(0),
             acc2 = svdup_u64(0), acc3 = svdup_u64(0);
  size_t vl = svcntd();
  size_t stride = vl * 4;

  size_t i = 0;
  for (; i + stride <= n; i += stride) {
    svbool_t pt = svptrue_b64();
    acc0 = svmla_u64_x(pt, acc0, svld1_u64(pt, a+i),      svld1_u64(pt, b+i));
    acc1 = svmla_u64_x(pt, acc1, svld1_u64(pt, a+i+vl),   svld1_u64(pt, b+i+vl));
    acc2 = svmla_u64_x(pt, acc2, svld1_u64(pt, a+i+vl*2), svld1_u64(pt, b+i+vl*2));
    acc3 = svmla_u64_x(pt, acc3, svld1_u64(pt, a+i+vl*3), svld1_u64(pt, b+i+vl*3));
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

/* -- 8-bit dot (SVE2 svmlalb/svmlalt: widening multiply-accumulate) -- */

static inline void dot_i8_sve2(const int8_t *a, const int8_t *b, size_t n,
                               int8_t *dest) {
  svint32_t acc0 = svdup_s32(0), acc1 = svdup_s32(0);
  size_t vl8 = svcntb();

  size_t i = 0;
  for (; i + vl8 <= n; i += vl8) {
    svbool_t pt = svptrue_b8();
    svint8_t va = svld1_s8(pt, a+i);
    svint8_t vb = svld1_s8(pt, b+i);
    /* SVE2 svmullb/svmullt: widening multiply i8→i16, even/odd elements */
    svint16_t p_lo = svmullb_s16(va, vb);
    svint16_t p_hi = svmullt_s16(va, vb);
    /* widen i16→i32 via unpack, accumulate */
    acc0 = svadd_s32_x(svptrue_b32(), acc0, svunpklo_s32(p_lo));
    acc0 = svadd_s32_x(svptrue_b32(), acc0, svunpkhi_s32(p_lo));
    acc1 = svadd_s32_x(svptrue_b32(), acc1, svunpklo_s32(p_hi));
    acc1 = svadd_s32_x(svptrue_b32(), acc1, svunpkhi_s32(p_hi));
  }
  acc0 = svadd_s32_x(svptrue_b32(), acc0, acc1);

  int result = (int)svaddv_s32(svptrue_b32(), acc0);
  int tail = 0;
  for (; i < n; i++) tail += a[i] * b[i];
  *dest = (int8_t)(result + tail);
}

static inline void dot_u8_sve2(const uint8_t *a, const uint8_t *b, size_t n,
                               uint8_t *dest) {
  svuint32_t acc0 = svdup_u32(0), acc1 = svdup_u32(0);
  size_t vl8 = svcntb();

  size_t i = 0;
  for (; i + vl8 <= n; i += vl8) {
    svbool_t pt = svptrue_b8();
    svuint8_t va = svld1_u8(pt, a+i);
    svuint8_t vb = svld1_u8(pt, b+i);
    svuint16_t p_lo = svmullb_u16(va, vb);
    svuint16_t p_hi = svmullt_u16(va, vb);
    acc0 = svadd_u32_x(svptrue_b32(), acc0, svunpklo_u32(p_lo));
    acc0 = svadd_u32_x(svptrue_b32(), acc0, svunpkhi_u32(p_lo));
    acc1 = svadd_u32_x(svptrue_b32(), acc1, svunpklo_u32(p_hi));
    acc1 = svadd_u32_x(svptrue_b32(), acc1, svunpkhi_u32(p_hi));
  }
  acc0 = svadd_u32_x(svptrue_b32(), acc0, acc1);

  uint32_t result = (uint32_t)svaddv_u32(svptrue_b32(), acc0);
  uint8_t tail = 0;
  for (; i < n; i++) tail += (uint8_t)(a[i] * b[i]);
  *dest = (uint8_t)(result + tail);
}

/* -- 16-bit dot (SVE2 svmlalb/svmlalt: i16→i32 widening MLA) ---------- */

static inline void dot_i16_sve2(const int16_t *a, const int16_t *b, size_t n,
                                int16_t *dest) {
  svint32_t acc0 = svdup_s32(0), acc1 = svdup_s32(0);
  size_t vl16 = svcnth();

  size_t i = 0;
  for (; i + vl16 <= n; i += vl16) {
    svbool_t pt = svptrue_b16();
    svint16_t va = svld1_s16(pt, a+i);
    svint16_t vb = svld1_s16(pt, b+i);
    /* SVE2 widening MLA: i16×i16→i32, bottom and top interleaved */
    acc0 = svmlalb_s32(acc0, va, vb);
    acc1 = svmlalt_s32(acc1, va, vb);
  }
  acc0 = svadd_s32_x(svptrue_b32(), acc0, acc1);

  int result = (int)svaddv_s32(svptrue_b32(), acc0);
  int tail = 0;
  for (; i < n; i++) tail += a[i] * b[i];
  *dest = (int16_t)(result + tail);
}

static inline void dot_u16_sve2(const uint16_t *a, const uint16_t *b, size_t n,
                                uint16_t *dest) {
  svuint32_t acc0 = svdup_u32(0), acc1 = svdup_u32(0);
  size_t vl16 = svcnth();

  size_t i = 0;
  for (; i + vl16 <= n; i += vl16) {
    svbool_t pt = svptrue_b16();
    svuint16_t va = svld1_u16(pt, a+i);
    svuint16_t vb = svld1_u16(pt, b+i);
    acc0 = svmlalb_u32(acc0, va, vb);
    acc1 = svmlalt_u32(acc1, va, vb);
  }
  acc0 = svadd_u32_x(svptrue_b32(), acc0, acc1);

  uint32_t result = (uint32_t)svaddv_u32(svptrue_b32(), acc0);
  uint16_t tail = 0;
  for (; i < n; i++) tail += (uint16_t)(a[i] * b[i]);
  *dest = (uint16_t)(result + tail);
}

// clang-format on

#endif
