/**
 * @file elemwise_rvv.h
 * @brief RVV element-wise binary/unary kernels for all 10 types.
 *
 * Binary: sub, mul, maximum, minimum
 * Unary: neg, abs
 *
 * Uses LMUL=m4 for throughput. RVV vsetvl handles tails naturally,
 * so no scalar cleanup loop is needed.
 */
#ifndef NUMC_ELEMWISE_RVV_H
#define NUMC_ELEMWISE_RVV_H

#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

/* ====================================================================
 * Binary: signed integer macro
 * ==================================================================== */

#define FAST_BIN_SINT_RVV(OP, SFX, CT, SEW, VEC_OP)                          \
  static inline void _fast_##OP##_##SFX##_rvv(const void *restrict ap,       \
                                              const void *restrict bp,       \
                                              void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                            \
    const CT *b = (const CT *)bp;                                            \
    CT *out = (CT *)op;                                                      \
    size_t vl;                                                               \
    for (size_t i = 0; i < n; i += vl) {                                     \
      vl = __riscv_vsetvl_e##SEW##m4(n - i);                                 \
      vint##SEW##m4_t va = __riscv_vle##SEW##_v_i##SEW##m4(a + i, vl);       \
      vint##SEW##m4_t vb = __riscv_vle##SEW##_v_i##SEW##m4(b + i, vl);       \
      __riscv_vse##SEW##_v_i##SEW##m4(out + i, VEC_OP(va, vb, vl), vl);      \
    }                                                                        \
  }

#define FAST_BIN_UINT_RVV(OP, SFX, CT, SEW, VEC_OP)                          \
  static inline void _fast_##OP##_##SFX##_rvv(const void *restrict ap,       \
                                              const void *restrict bp,       \
                                              void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                            \
    const CT *b = (const CT *)bp;                                            \
    CT *out = (CT *)op;                                                      \
    size_t vl;                                                               \
    for (size_t i = 0; i < n; i += vl) {                                     \
      vl = __riscv_vsetvl_e##SEW##m4(n - i);                                 \
      vuint##SEW##m4_t va = __riscv_vle##SEW##_v_u##SEW##m4(a + i, vl);      \
      vuint##SEW##m4_t vb = __riscv_vle##SEW##_v_u##SEW##m4(b + i, vl);      \
      __riscv_vse##SEW##_v_u##SEW##m4(out + i, VEC_OP(va, vb, vl), vl);      \
    }                                                                        \
  }

#define FAST_BIN_F32_RVV(OP, VEC_OP)                                     \
  static inline void _fast_##OP##_f32_rvv(const void *restrict ap,       \
                                          const void *restrict bp,       \
                                          void *restrict op, size_t n) { \
    const float *a = (const float *)ap;                                  \
    const float *b = (const float *)bp;                                  \
    float *out = (float *)op;                                            \
    size_t vl;                                                           \
    for (size_t i = 0; i < n; i += vl) {                                 \
      vl = __riscv_vsetvl_e32m4(n - i);                                  \
      vfloat32m4_t va = __riscv_vle32_v_f32m4(a + i, vl);                \
      vfloat32m4_t vb = __riscv_vle32_v_f32m4(b + i, vl);                \
      __riscv_vse32_v_f32m4(out + i, VEC_OP(va, vb, vl), vl);            \
    }                                                                    \
  }

#define FAST_BIN_F64_RVV(OP, VEC_OP)                                     \
  static inline void _fast_##OP##_f64_rvv(const void *restrict ap,       \
                                          const void *restrict bp,       \
                                          void *restrict op, size_t n) { \
    const double *a = (const double *)ap;                                \
    const double *b = (const double *)bp;                                \
    double *out = (double *)op;                                          \
    size_t vl;                                                           \
    for (size_t i = 0; i < n; i += vl) {                                 \
      vl = __riscv_vsetvl_e64m4(n - i);                                  \
      vfloat64m4_t va = __riscv_vle64_v_f64m4(a + i, vl);                \
      vfloat64m4_t vb = __riscv_vle64_v_f64m4(b + i, vl);                \
      __riscv_vse64_v_f64m4(out + i, VEC_OP(va, vb, vl), vl);            \
    }                                                                    \
  }

/* ── Add ──────────────────────────────────────────────────────────── */

FAST_BIN_SINT_RVV(add, i8, int8_t, 8, __riscv_vadd_vv_i8m4)
FAST_BIN_SINT_RVV(add, i16, int16_t, 16, __riscv_vadd_vv_i16m4)
FAST_BIN_SINT_RVV(add, i32, int32_t, 32, __riscv_vadd_vv_i32m4)
FAST_BIN_SINT_RVV(add, i64, int64_t, 64, __riscv_vadd_vv_i64m4)
FAST_BIN_UINT_RVV(add, u8, uint8_t, 8, __riscv_vadd_vv_u8m4)
FAST_BIN_UINT_RVV(add, u16, uint16_t, 16, __riscv_vadd_vv_u16m4)
FAST_BIN_UINT_RVV(add, u32, uint32_t, 32, __riscv_vadd_vv_u32m4)
FAST_BIN_UINT_RVV(add, u64, uint64_t, 64, __riscv_vadd_vv_u64m4)
FAST_BIN_F32_RVV(add, __riscv_vfadd_vv_f32m4)
FAST_BIN_F64_RVV(add, __riscv_vfadd_vv_f64m4)

/* ── Sub ──────────────────────────────────────────────────────────── */

FAST_BIN_SINT_RVV(sub, i8, int8_t, 8, __riscv_vsub_vv_i8m4)
FAST_BIN_SINT_RVV(sub, i16, int16_t, 16, __riscv_vsub_vv_i16m4)
FAST_BIN_SINT_RVV(sub, i32, int32_t, 32, __riscv_vsub_vv_i32m4)
FAST_BIN_SINT_RVV(sub, i64, int64_t, 64, __riscv_vsub_vv_i64m4)
FAST_BIN_UINT_RVV(sub, u8, uint8_t, 8, __riscv_vsub_vv_u8m4)
FAST_BIN_UINT_RVV(sub, u16, uint16_t, 16, __riscv_vsub_vv_u16m4)
FAST_BIN_UINT_RVV(sub, u32, uint32_t, 32, __riscv_vsub_vv_u32m4)
FAST_BIN_UINT_RVV(sub, u64, uint64_t, 64, __riscv_vsub_vv_u64m4)
FAST_BIN_F32_RVV(sub, __riscv_vfsub_vv_f32m4)
FAST_BIN_F64_RVV(sub, __riscv_vfsub_vv_f64m4)

/* ── Mul ──────────────────────────────────────────────────────────── */

FAST_BIN_SINT_RVV(mul, i8, int8_t, 8, __riscv_vmul_vv_i8m4)
FAST_BIN_SINT_RVV(mul, i16, int16_t, 16, __riscv_vmul_vv_i16m4)
FAST_BIN_SINT_RVV(mul, i32, int32_t, 32, __riscv_vmul_vv_i32m4)
FAST_BIN_SINT_RVV(mul, i64, int64_t, 64, __riscv_vmul_vv_i64m4)
FAST_BIN_UINT_RVV(mul, u8, uint8_t, 8, __riscv_vmul_vv_u8m4)
FAST_BIN_UINT_RVV(mul, u16, uint16_t, 16, __riscv_vmul_vv_u16m4)
FAST_BIN_UINT_RVV(mul, u32, uint32_t, 32, __riscv_vmul_vv_u32m4)
FAST_BIN_UINT_RVV(mul, u64, uint64_t, 64, __riscv_vmul_vv_u64m4)
FAST_BIN_F32_RVV(mul, __riscv_vfmul_vv_f32m4)
FAST_BIN_F64_RVV(mul, __riscv_vfmul_vv_f64m4)

/* ── Maximum (signed) ─────────────────────────────────────────────── */

FAST_BIN_SINT_RVV(maximum, i8, int8_t, 8, __riscv_vmax_vv_i8m4)
FAST_BIN_SINT_RVV(maximum, i16, int16_t, 16, __riscv_vmax_vv_i16m4)
FAST_BIN_SINT_RVV(maximum, i32, int32_t, 32, __riscv_vmax_vv_i32m4)
FAST_BIN_SINT_RVV(maximum, i64, int64_t, 64, __riscv_vmax_vv_i64m4)

/* ── Maximum (unsigned) ───────────────────────────────────────────── */

FAST_BIN_UINT_RVV(maximum, u8, uint8_t, 8, __riscv_vmaxu_vv_u8m4)
FAST_BIN_UINT_RVV(maximum, u16, uint16_t, 16, __riscv_vmaxu_vv_u16m4)
FAST_BIN_UINT_RVV(maximum, u32, uint32_t, 32, __riscv_vmaxu_vv_u32m4)
FAST_BIN_UINT_RVV(maximum, u64, uint64_t, 64, __riscv_vmaxu_vv_u64m4)

/* ── Maximum (float) ──────────────────────────────────────────────── */

FAST_BIN_F32_RVV(maximum, __riscv_vfmax_vv_f32m4)
FAST_BIN_F64_RVV(maximum, __riscv_vfmax_vv_f64m4)

/* ── Minimum (signed) ─────────────────────────────────────────────── */

FAST_BIN_SINT_RVV(minimum, i8, int8_t, 8, __riscv_vmin_vv_i8m4)
FAST_BIN_SINT_RVV(minimum, i16, int16_t, 16, __riscv_vmin_vv_i16m4)
FAST_BIN_SINT_RVV(minimum, i32, int32_t, 32, __riscv_vmin_vv_i32m4)
FAST_BIN_SINT_RVV(minimum, i64, int64_t, 64, __riscv_vmin_vv_i64m4)

/* ── Minimum (unsigned) ───────────────────────────────────────────── */

FAST_BIN_UINT_RVV(minimum, u8, uint8_t, 8, __riscv_vminu_vv_u8m4)
FAST_BIN_UINT_RVV(minimum, u16, uint16_t, 16, __riscv_vminu_vv_u16m4)
FAST_BIN_UINT_RVV(minimum, u32, uint32_t, 32, __riscv_vminu_vv_u32m4)
FAST_BIN_UINT_RVV(minimum, u64, uint64_t, 64, __riscv_vminu_vv_u64m4)

/* ── Minimum (float) ──────────────────────────────────────────────── */

FAST_BIN_F32_RVV(minimum, __riscv_vfmin_vv_f32m4)
FAST_BIN_F64_RVV(minimum, __riscv_vfmin_vv_f64m4)

#undef FAST_BIN_SINT_RVV
#undef FAST_BIN_UINT_RVV
#undef FAST_BIN_F32_RVV
#undef FAST_BIN_F64_RVV

/* ====================================================================
 * Unary operations
 * ==================================================================== */

/* ── Neg (signed integer) ─────────────────────────────────────────── */

#define FAST_NEG_SINT_RVV(SFX, CT, SEW)                                       \
  static inline void _fast_neg_##SFX##_rvv(const void *restrict ap,           \
                                           void *restrict op, size_t n) {     \
    const CT *a = (const CT *)ap;                                             \
    CT *out = (CT *)op;                                                       \
    size_t vl;                                                                \
    for (size_t i = 0; i < n; i += vl) {                                      \
      vl = __riscv_vsetvl_e##SEW##m4(n - i);                                  \
      vint##SEW##m4_t va = __riscv_vle##SEW##_v_i##SEW##m4(a + i, vl);        \
      __riscv_vse##SEW##_v_i##SEW##m4(out + i,                                \
                                      __riscv_vneg_v_i##SEW##m4(va, vl), vl); \
    }                                                                         \
  }

FAST_NEG_SINT_RVV(i8, int8_t, 8)
FAST_NEG_SINT_RVV(i16, int16_t, 16)
FAST_NEG_SINT_RVV(i32, int32_t, 32)
FAST_NEG_SINT_RVV(i64, int64_t, 64)

#undef FAST_NEG_SINT_RVV

/* ── Neg (unsigned integer): 0 - x ───────────────────────────────── */

#define FAST_NEG_UINT_RVV(SFX, CT, SEW)                                   \
  static inline void _fast_neg_##SFX##_rvv(const void *restrict ap,       \
                                           void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl;                                                            \
    for (size_t i = 0; i < n; i += vl) {                                  \
      vl = __riscv_vsetvl_e##SEW##m4(n - i);                              \
      vuint##SEW##m4_t va = __riscv_vle##SEW##_v_u##SEW##m4(a + i, vl);   \
      vuint##SEW##m4_t zero = __riscv_vmv_v_x_u##SEW##m4(0, vl);          \
      __riscv_vse##SEW##_v_u##SEW##m4(                                    \
          out + i, __riscv_vsub_vv_u##SEW##m4(zero, va, vl), vl);         \
    }                                                                     \
  }

FAST_NEG_UINT_RVV(u8, uint8_t, 8)
FAST_NEG_UINT_RVV(u16, uint16_t, 16)
FAST_NEG_UINT_RVV(u32, uint32_t, 32)
FAST_NEG_UINT_RVV(u64, uint64_t, 64)

#undef FAST_NEG_UINT_RVV

/* ── Neg (float) ──────────────────────────────────────────────────── */

static inline void _fast_neg_f32_rvv(const void *restrict ap, void *restrict op,
                                     size_t n) {
  const float *a = (const float *)ap;
  float *out = (float *)op;
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m4(n - i);
    vfloat32m4_t va = __riscv_vle32_v_f32m4(a + i, vl);
    __riscv_vse32_v_f32m4(out + i, __riscv_vfneg_v_f32m4(va, vl), vl);
  }
}

static inline void _fast_neg_f64_rvv(const void *restrict ap, void *restrict op,
                                     size_t n) {
  const double *a = (const double *)ap;
  double *out = (double *)op;
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e64m4(n - i);
    vfloat64m4_t va = __riscv_vle64_v_f64m4(a + i, vl);
    __riscv_vse64_v_f64m4(out + i, __riscv_vfneg_v_f64m4(va, vl), vl);
  }
}

/* ── Abs (signed integer): max(x, -x) ────────────────────────────── */

#define FAST_ABS_SINT_RVV(SFX, CT, SEW)                                   \
  static inline void _fast_abs_##SFX##_rvv(const void *restrict ap,       \
                                           void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl;                                                            \
    for (size_t i = 0; i < n; i += vl) {                                  \
      vl = __riscv_vsetvl_e##SEW##m4(n - i);                              \
      vint##SEW##m4_t va = __riscv_vle##SEW##_v_i##SEW##m4(a + i, vl);    \
      vint##SEW##m4_t neg = __riscv_vneg_v_i##SEW##m4(va, vl);            \
      __riscv_vse##SEW##_v_i##SEW##m4(                                    \
          out + i, __riscv_vmax_vv_i##SEW##m4(va, neg, vl), vl);          \
    }                                                                     \
  }

FAST_ABS_SINT_RVV(i8, int8_t, 8)
FAST_ABS_SINT_RVV(i16, int16_t, 16)
FAST_ABS_SINT_RVV(i32, int32_t, 32)
FAST_ABS_SINT_RVV(i64, int64_t, 64)

#undef FAST_ABS_SINT_RVV

/* ── Abs (float) ──────────────────────────────────────────────────── */

static inline void _fast_abs_f32_rvv(const void *restrict ap, void *restrict op,
                                     size_t n) {
  const float *a = (const float *)ap;
  float *out = (float *)op;
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m4(n - i);
    vfloat32m4_t va = __riscv_vle32_v_f32m4(a + i, vl);
    __riscv_vse32_v_f32m4(out + i, __riscv_vfabs_v_f32m4(va, vl), vl);
  }
}

static inline void _fast_abs_f64_rvv(const void *restrict ap, void *restrict op,
                                     size_t n) {
  const double *a = (const double *)ap;
  double *out = (double *)op;
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e64m4(n - i);
    vfloat64m4_t va = __riscv_vle64_v_f64m4(a + i, vl);
    __riscv_vse64_v_f64m4(out + i, __riscv_vfabs_v_f64m4(va, vl), vl);
  }
}

#endif /* NUMC_ELEMWISE_RVV_H */
