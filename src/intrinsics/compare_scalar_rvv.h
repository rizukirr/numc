/**
 * @file compare_scalar_rvv.h
 * @brief RVV scalar comparison kernels — uint8 output (0/1).
 *
 * All comparison functions output uint8_t* (NumPy-compatible bool).
 * Uses vector-scalar compare intrinsics (_vx/_vf) for efficiency.
 * LMUL=m4 for throughput. Natural tail handling via vsetvl.
 *
 * For wider types, the mask ratio must match the u8 output LMUL:
 *   e16m4 (vbool4) → u8m2
 *   e32m4 (vbool8) → u8m1
 *   e64m4 (vbool16) → u8mf2
 */
#ifndef NUMC_COMPARE_SCALAR_RVV_H
#define NUMC_COMPARE_SCALAR_RVV_H

#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

/* -- 8-bit signed: input and output both e8m4 --------------------- */

#define STAMP_CMPSC_I8_RVV(FNAME, CMP_VX)                                    \
  static inline void _cmpsc_##FNAME##_i8_rvv(const void *restrict ap,        \
                                             const void *restrict sp,        \
                                             void *restrict op, size_t n) {  \
    const int8_t *a = (const int8_t *)ap;                                    \
    const int8_t s = *(const int8_t *)sp;                                    \
    uint8_t *out = (uint8_t *)op;                                            \
    size_t vl;                                                               \
    for (size_t i = 0; i < n; i += vl) {                                     \
      vl = __riscv_vsetvl_e8m4(n - i);                                       \
      vint8m4_t va = __riscv_vle8_v_i8m4(a + i, vl);                         \
      vbool2_t mask = CMP_VX(va, s, vl);                                     \
      vuint8m4_t r =                                                         \
          __riscv_vmerge_vxm_u8m4(__riscv_vmv_v_x_u8m4(0, vl), 1, mask, vl); \
      __riscv_vse8_v_u8m4(out + i, r, vl);                                   \
    }                                                                        \
  }

STAMP_CMPSC_I8_RVV(eq, __riscv_vmseq_vx_i8m4_b2)
STAMP_CMPSC_I8_RVV(gt, __riscv_vmsgt_vx_i8m4_b2)
STAMP_CMPSC_I8_RVV(lt, __riscv_vmslt_vx_i8m4_b2)
STAMP_CMPSC_I8_RVV(ge, __riscv_vmsge_vx_i8m4_b2)
STAMP_CMPSC_I8_RVV(le, __riscv_vmsle_vx_i8m4_b2)
#undef STAMP_CMPSC_I8_RVV

/* -- 8-bit unsigned: input and output both e8m4 ------------------- */

#define STAMP_CMPSC_U8_RVV(FNAME, CMP_VX)                                    \
  static inline void _cmpsc_##FNAME##_u8_rvv(const void *restrict ap,        \
                                             const void *restrict sp,        \
                                             void *restrict op, size_t n) {  \
    const uint8_t *a = (const uint8_t *)ap;                                  \
    const uint8_t s = *(const uint8_t *)sp;                                  \
    uint8_t *out = (uint8_t *)op;                                            \
    size_t vl;                                                               \
    for (size_t i = 0; i < n; i += vl) {                                     \
      vl = __riscv_vsetvl_e8m4(n - i);                                       \
      vuint8m4_t va = __riscv_vle8_v_u8m4(a + i, vl);                        \
      vbool2_t mask = CMP_VX(va, s, vl);                                     \
      vuint8m4_t r =                                                         \
          __riscv_vmerge_vxm_u8m4(__riscv_vmv_v_x_u8m4(0, vl), 1, mask, vl); \
      __riscv_vse8_v_u8m4(out + i, r, vl);                                   \
    }                                                                        \
  }

STAMP_CMPSC_U8_RVV(eq, __riscv_vmseq_vx_u8m4_b2)
STAMP_CMPSC_U8_RVV(gt, __riscv_vmsgtu_vx_u8m4_b2)
STAMP_CMPSC_U8_RVV(lt, __riscv_vmsltu_vx_u8m4_b2)
STAMP_CMPSC_U8_RVV(ge, __riscv_vmsgeu_vx_u8m4_b2)
STAMP_CMPSC_U8_RVV(le, __riscv_vmsleu_vx_u8m4_b2)
#undef STAMP_CMPSC_U8_RVV

/* -- Wider signed integer (16/32/64): byte output ----------------- */
/* OLMU = output LMUL suffix: m2 for 16-bit, m1 for 32-bit, mf2 for 64-bit */

#define STAMP_CMPSC_SINT_WIDE_RVV(SFX, CT, SEW, LMUL, RATIO, OLMU, CMP_VX) \
  static inline void _cmpsc_##SFX##_rvv(const void *restrict ap,           \
                                        const void *restrict sp,           \
                                        void *restrict op, size_t n) {     \
    const CT *a = (const CT *)ap;                                          \
    const CT s = *(const CT *)sp;                                          \
    uint8_t *out = (uint8_t *)op;                                          \
    size_t vl;                                                             \
    for (size_t i = 0; i < n; i += vl) {                                   \
      vl = __riscv_vsetvl_e##SEW##m##LMUL(n - i);                          \
      vint##SEW##m##LMUL##_t va =                                          \
          __riscv_vle##SEW##_v_i##SEW##m##LMUL(a + i, vl);                 \
      vbool##RATIO##_t mask = CMP_VX(va, s, vl);                           \
      vuint8##OLMU##_t r = __riscv_vmerge_vxm_u8##OLMU(                    \
          __riscv_vmv_v_x_u8##OLMU(0, vl), 1, mask, vl);                   \
      __riscv_vse8_v_u8##OLMU(out + i, r, vl);                             \
    }                                                                      \
  }

/* clang-format off */
STAMP_CMPSC_SINT_WIDE_RVV(eq_i16, int16_t, 16, 4, 4, m2, __riscv_vmseq_vx_i16m4_b4)
STAMP_CMPSC_SINT_WIDE_RVV(gt_i16, int16_t, 16, 4, 4, m2, __riscv_vmsgt_vx_i16m4_b4)
STAMP_CMPSC_SINT_WIDE_RVV(lt_i16, int16_t, 16, 4, 4, m2, __riscv_vmslt_vx_i16m4_b4)
STAMP_CMPSC_SINT_WIDE_RVV(ge_i16, int16_t, 16, 4, 4, m2, __riscv_vmsge_vx_i16m4_b4)
STAMP_CMPSC_SINT_WIDE_RVV(le_i16, int16_t, 16, 4, 4, m2, __riscv_vmsle_vx_i16m4_b4)

STAMP_CMPSC_SINT_WIDE_RVV(eq_i32, int32_t, 32, 4, 8, m1, __riscv_vmseq_vx_i32m4_b8)
STAMP_CMPSC_SINT_WIDE_RVV(gt_i32, int32_t, 32, 4, 8, m1, __riscv_vmsgt_vx_i32m4_b8)
STAMP_CMPSC_SINT_WIDE_RVV(lt_i32, int32_t, 32, 4, 8, m1, __riscv_vmslt_vx_i32m4_b8)
STAMP_CMPSC_SINT_WIDE_RVV(ge_i32, int32_t, 32, 4, 8, m1, __riscv_vmsge_vx_i32m4_b8)
STAMP_CMPSC_SINT_WIDE_RVV(le_i32, int32_t, 32, 4, 8, m1, __riscv_vmsle_vx_i32m4_b8)

STAMP_CMPSC_SINT_WIDE_RVV(eq_i64, int64_t, 64, 4, 16, mf2, __riscv_vmseq_vx_i64m4_b16)
STAMP_CMPSC_SINT_WIDE_RVV(gt_i64, int64_t, 64, 4, 16, mf2, __riscv_vmsgt_vx_i64m4_b16)
STAMP_CMPSC_SINT_WIDE_RVV(lt_i64, int64_t, 64, 4, 16, mf2, __riscv_vmslt_vx_i64m4_b16)
STAMP_CMPSC_SINT_WIDE_RVV(ge_i64, int64_t, 64, 4, 16, mf2, __riscv_vmsge_vx_i64m4_b16)
STAMP_CMPSC_SINT_WIDE_RVV(le_i64, int64_t, 64, 4, 16, mf2, __riscv_vmsle_vx_i64m4_b16)
/* clang-format on */
#undef STAMP_CMPSC_SINT_WIDE_RVV

/* -- Wider unsigned integer (16/32/64): byte output --------------- */

#define STAMP_CMPSC_UINT_WIDE_RVV(SFX, CT, SEW, LMUL, RATIO, OLMU, CMP_VX) \
  static inline void _cmpsc_##SFX##_rvv(const void *restrict ap,           \
                                        const void *restrict sp,           \
                                        void *restrict op, size_t n) {     \
    const CT *a = (const CT *)ap;                                          \
    const CT s = *(const CT *)sp;                                          \
    uint8_t *out = (uint8_t *)op;                                          \
    size_t vl;                                                             \
    for (size_t i = 0; i < n; i += vl) {                                   \
      vl = __riscv_vsetvl_e##SEW##m##LMUL(n - i);                          \
      vuint##SEW##m##LMUL##_t va =                                         \
          __riscv_vle##SEW##_v_u##SEW##m##LMUL(a + i, vl);                 \
      vbool##RATIO##_t mask = CMP_VX(va, s, vl);                           \
      vuint8##OLMU##_t r = __riscv_vmerge_vxm_u8##OLMU(                    \
          __riscv_vmv_v_x_u8##OLMU(0, vl), 1, mask, vl);                   \
      __riscv_vse8_v_u8##OLMU(out + i, r, vl);                             \
    }                                                                      \
  }

/* clang-format off */
STAMP_CMPSC_UINT_WIDE_RVV(eq_u16, uint16_t, 16, 4, 4, m2, __riscv_vmseq_vx_u16m4_b4)
STAMP_CMPSC_UINT_WIDE_RVV(gt_u16, uint16_t, 16, 4, 4, m2, __riscv_vmsgtu_vx_u16m4_b4)
STAMP_CMPSC_UINT_WIDE_RVV(lt_u16, uint16_t, 16, 4, 4, m2, __riscv_vmsltu_vx_u16m4_b4)
STAMP_CMPSC_UINT_WIDE_RVV(ge_u16, uint16_t, 16, 4, 4, m2, __riscv_vmsgeu_vx_u16m4_b4)
STAMP_CMPSC_UINT_WIDE_RVV(le_u16, uint16_t, 16, 4, 4, m2, __riscv_vmsleu_vx_u16m4_b4)

STAMP_CMPSC_UINT_WIDE_RVV(eq_u32, uint32_t, 32, 4, 8, m1, __riscv_vmseq_vx_u32m4_b8)
STAMP_CMPSC_UINT_WIDE_RVV(gt_u32, uint32_t, 32, 4, 8, m1, __riscv_vmsgtu_vx_u32m4_b8)
STAMP_CMPSC_UINT_WIDE_RVV(lt_u32, uint32_t, 32, 4, 8, m1, __riscv_vmsltu_vx_u32m4_b8)
STAMP_CMPSC_UINT_WIDE_RVV(ge_u32, uint32_t, 32, 4, 8, m1, __riscv_vmsgeu_vx_u32m4_b8)
STAMP_CMPSC_UINT_WIDE_RVV(le_u32, uint32_t, 32, 4, 8, m1, __riscv_vmsleu_vx_u32m4_b8)

STAMP_CMPSC_UINT_WIDE_RVV(eq_u64, uint64_t, 64, 4, 16, mf2, __riscv_vmseq_vx_u64m4_b16)
STAMP_CMPSC_UINT_WIDE_RVV(gt_u64, uint64_t, 64, 4, 16, mf2, __riscv_vmsgtu_vx_u64m4_b16)
STAMP_CMPSC_UINT_WIDE_RVV(lt_u64, uint64_t, 64, 4, 16, mf2, __riscv_vmsltu_vx_u64m4_b16)
STAMP_CMPSC_UINT_WIDE_RVV(ge_u64, uint64_t, 64, 4, 16, mf2, __riscv_vmsgeu_vx_u64m4_b16)
STAMP_CMPSC_UINT_WIDE_RVV(le_u64, uint64_t, 64, 4, 16, mf2, __riscv_vmsleu_vx_u64m4_b16)
/* clang-format on */
#undef STAMP_CMPSC_UINT_WIDE_RVV

/* -- Float (32/64): byte output ----------------------------------- */

#define STAMP_CMPSC_FLOAT_RVV(SFX, CT, SEW, LMUL, RATIO, OLMU, CMP_VF) \
  static inline void _cmpsc_##SFX##_rvv(const void *restrict ap,       \
                                        const void *restrict sp,       \
                                        void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                      \
    const CT s = *(const CT *)sp;                                      \
    uint8_t *out = (uint8_t *)op;                                      \
    size_t vl;                                                         \
    for (size_t i = 0; i < n; i += vl) {                               \
      vl = __riscv_vsetvl_e##SEW##m##LMUL(n - i);                      \
      vfloat##SEW##m##LMUL##_t va =                                    \
          __riscv_vle##SEW##_v_f##SEW##m##LMUL(a + i, vl);             \
      vbool##RATIO##_t mask = CMP_VF(va, s, vl);                       \
      vuint8##OLMU##_t r = __riscv_vmerge_vxm_u8##OLMU(                \
          __riscv_vmv_v_x_u8##OLMU(0, vl), 1, mask, vl);               \
      __riscv_vse8_v_u8##OLMU(out + i, r, vl);                         \
    }                                                                  \
  }

/* clang-format off */
STAMP_CMPSC_FLOAT_RVV(eq_f32, float,  32, 4, 8, m1, __riscv_vmfeq_vf_f32m4_b8)
STAMP_CMPSC_FLOAT_RVV(gt_f32, float,  32, 4, 8, m1, __riscv_vmfgt_vf_f32m4_b8)
STAMP_CMPSC_FLOAT_RVV(lt_f32, float,  32, 4, 8, m1, __riscv_vmflt_vf_f32m4_b8)
STAMP_CMPSC_FLOAT_RVV(ge_f32, float,  32, 4, 8, m1, __riscv_vmfge_vf_f32m4_b8)
STAMP_CMPSC_FLOAT_RVV(le_f32, float,  32, 4, 8, m1, __riscv_vmfle_vf_f32m4_b8)

STAMP_CMPSC_FLOAT_RVV(eq_f64, double, 64, 4, 16, mf2, __riscv_vmfeq_vf_f64m4_b16)
STAMP_CMPSC_FLOAT_RVV(gt_f64, double, 64, 4, 16, mf2, __riscv_vmfgt_vf_f64m4_b16)
STAMP_CMPSC_FLOAT_RVV(lt_f64, double, 64, 4, 16, mf2, __riscv_vmflt_vf_f64m4_b16)
STAMP_CMPSC_FLOAT_RVV(ge_f64, double, 64, 4, 16, mf2, __riscv_vmfge_vf_f64m4_b16)
STAMP_CMPSC_FLOAT_RVV(le_f64, double, 64, 4, 16, mf2, __riscv_vmfle_vf_f64m4_b16)
/* clang-format on */
#undef STAMP_CMPSC_FLOAT_RVV

#endif /* NUMC_COMPARE_SCALAR_RVV_H */
