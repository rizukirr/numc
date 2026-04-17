/**
 * @file elemwise_scalar_rvv.h
 * @brief RVV scalar arithmetic kernels for all 10 types.
 *
 * Operations: add_scalar, sub_scalar, mul_scalar
 *
 * Uses LMUL=m4 for throughput. RVV vsetvl handles tails naturally,
 * so no scalar cleanup loop is needed. Uses vector-scalar intrinsics
 * (_vx for integer, _vf for float) to avoid explicit broadcast.
 */
#ifndef NUMC_ELEMWISE_SCALAR_RVV_H
#define NUMC_ELEMWISE_SCALAR_RVV_H

#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

/* ====================================================================
 * Scalar binary: signed integer macro
 * ==================================================================== */

#define FAST_SCAL_SINT_RVV(OP, SFX, CT, SEW, VEC_OP)                       \
  static inline void _fast_##OP##_scalar_##SFX##_rvv(                      \
      const void *restrict ap, const void *restrict sp, void *restrict op, \
      size_t n) {                                                          \
    const CT *a = (const CT *)ap;                                          \
    const CT s = *(const CT *)sp;                                          \
    CT *out = (CT *)op;                                                    \
    size_t vl;                                                             \
    for (size_t i = 0; i < n; i += vl) {                                   \
      vl = __riscv_vsetvl_e##SEW##m4(n - i);                               \
      vint##SEW##m4_t va = __riscv_vle##SEW##_v_i##SEW##m4(a + i, vl);     \
      __riscv_vse##SEW##_v_i##SEW##m4(out + i, VEC_OP(va, s, vl), vl);     \
    }                                                                      \
  }

/* ====================================================================
 * Scalar binary: unsigned integer macro
 * ==================================================================== */

#define FAST_SCAL_UINT_RVV(OP, SFX, CT, SEW, VEC_OP)                       \
  static inline void _fast_##OP##_scalar_##SFX##_rvv(                      \
      const void *restrict ap, const void *restrict sp, void *restrict op, \
      size_t n) {                                                          \
    const CT *a = (const CT *)ap;                                          \
    const CT s = *(const CT *)sp;                                          \
    CT *out = (CT *)op;                                                    \
    size_t vl;                                                             \
    for (size_t i = 0; i < n; i += vl) {                                   \
      vl = __riscv_vsetvl_e##SEW##m4(n - i);                               \
      vuint##SEW##m4_t va = __riscv_vle##SEW##_v_u##SEW##m4(a + i, vl);    \
      __riscv_vse##SEW##_v_u##SEW##m4(out + i, VEC_OP(va, s, vl), vl);     \
    }                                                                      \
  }

/* ====================================================================
 * Scalar binary: float macros
 * ==================================================================== */

#define FAST_SCAL_F32_RVV(OP, VEC_OP)                                      \
  static inline void _fast_##OP##_scalar_f32_rvv(                          \
      const void *restrict ap, const void *restrict sp, void *restrict op, \
      size_t n) {                                                          \
    const float *a = (const float *)ap;                                    \
    const float s = *(const float *)sp;                                    \
    float *out = (float *)op;                                              \
    size_t vl;                                                             \
    for (size_t i = 0; i < n; i += vl) {                                   \
      vl = __riscv_vsetvl_e32m4(n - i);                                    \
      vfloat32m4_t va = __riscv_vle32_v_f32m4(a + i, vl);                  \
      __riscv_vse32_v_f32m4(out + i, VEC_OP(va, s, vl), vl);               \
    }                                                                      \
  }

#define FAST_SCAL_F64_RVV(OP, VEC_OP)                                      \
  static inline void _fast_##OP##_scalar_f64_rvv(                          \
      const void *restrict ap, const void *restrict sp, void *restrict op, \
      size_t n) {                                                          \
    const double *a = (const double *)ap;                                  \
    const double s = *(const double *)sp;                                  \
    double *out = (double *)op;                                            \
    size_t vl;                                                             \
    for (size_t i = 0; i < n; i += vl) {                                   \
      vl = __riscv_vsetvl_e64m4(n - i);                                    \
      vfloat64m4_t va = __riscv_vle64_v_f64m4(a + i, vl);                  \
      __riscv_vse64_v_f64m4(out + i, VEC_OP(va, s, vl), vl);               \
    }                                                                      \
  }

/* -- Add scalar ---------------------------------------------------- */

FAST_SCAL_SINT_RVV(add, i8, int8_t, 8, __riscv_vadd_vx_i8m4)
FAST_SCAL_SINT_RVV(add, i16, int16_t, 16, __riscv_vadd_vx_i16m4)
FAST_SCAL_SINT_RVV(add, i32, int32_t, 32, __riscv_vadd_vx_i32m4)
FAST_SCAL_SINT_RVV(add, i64, int64_t, 64, __riscv_vadd_vx_i64m4)
FAST_SCAL_UINT_RVV(add, u8, uint8_t, 8, __riscv_vadd_vx_u8m4)
FAST_SCAL_UINT_RVV(add, u16, uint16_t, 16, __riscv_vadd_vx_u16m4)
FAST_SCAL_UINT_RVV(add, u32, uint32_t, 32, __riscv_vadd_vx_u32m4)
FAST_SCAL_UINT_RVV(add, u64, uint64_t, 64, __riscv_vadd_vx_u64m4)
FAST_SCAL_F32_RVV(add, __riscv_vfadd_vf_f32m4)
FAST_SCAL_F64_RVV(add, __riscv_vfadd_vf_f64m4)

/* -- Sub scalar ---------------------------------------------------- */

FAST_SCAL_SINT_RVV(sub, i8, int8_t, 8, __riscv_vsub_vx_i8m4)
FAST_SCAL_SINT_RVV(sub, i16, int16_t, 16, __riscv_vsub_vx_i16m4)
FAST_SCAL_SINT_RVV(sub, i32, int32_t, 32, __riscv_vsub_vx_i32m4)
FAST_SCAL_SINT_RVV(sub, i64, int64_t, 64, __riscv_vsub_vx_i64m4)
FAST_SCAL_UINT_RVV(sub, u8, uint8_t, 8, __riscv_vsub_vx_u8m4)
FAST_SCAL_UINT_RVV(sub, u16, uint16_t, 16, __riscv_vsub_vx_u16m4)
FAST_SCAL_UINT_RVV(sub, u32, uint32_t, 32, __riscv_vsub_vx_u32m4)
FAST_SCAL_UINT_RVV(sub, u64, uint64_t, 64, __riscv_vsub_vx_u64m4)
FAST_SCAL_F32_RVV(sub, __riscv_vfsub_vf_f32m4)
FAST_SCAL_F64_RVV(sub, __riscv_vfsub_vf_f64m4)

/* -- Mul scalar ---------------------------------------------------- */

FAST_SCAL_SINT_RVV(mul, i8, int8_t, 8, __riscv_vmul_vx_i8m4)
FAST_SCAL_SINT_RVV(mul, i16, int16_t, 16, __riscv_vmul_vx_i16m4)
FAST_SCAL_SINT_RVV(mul, i32, int32_t, 32, __riscv_vmul_vx_i32m4)
FAST_SCAL_SINT_RVV(mul, i64, int64_t, 64, __riscv_vmul_vx_i64m4)
FAST_SCAL_UINT_RVV(mul, u8, uint8_t, 8, __riscv_vmul_vx_u8m4)
FAST_SCAL_UINT_RVV(mul, u16, uint16_t, 16, __riscv_vmul_vx_u16m4)
FAST_SCAL_UINT_RVV(mul, u32, uint32_t, 32, __riscv_vmul_vx_u32m4)
FAST_SCAL_UINT_RVV(mul, u64, uint64_t, 64, __riscv_vmul_vx_u64m4)
FAST_SCAL_F32_RVV(mul, __riscv_vfmul_vf_f32m4)
FAST_SCAL_F64_RVV(mul, __riscv_vfmul_vf_f64m4)

/* -- Cleanup ------------------------------------------------------- */

#undef FAST_SCAL_SINT_RVV
#undef FAST_SCAL_UINT_RVV
#undef FAST_SCAL_F32_RVV
#undef FAST_SCAL_F64_RVV

#endif /* NUMC_ELEMWISE_SCALAR_RVV_H */
