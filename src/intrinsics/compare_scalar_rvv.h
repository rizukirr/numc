/**
 * @file compare_scalar_rvv.h
 * @brief RVV scalar comparison kernels for all 10 types.
 *
 * Uses vector-scalar compare intrinsics (_vx/_vf) for efficiency.
 * LMUL=m4 for throughput. Natural tail handling via vsetvl.
 */
#ifndef NUMC_COMPARE_SCALAR_RVV_H
#define NUMC_COMPARE_SCALAR_RVV_H

#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

/* ── Signed integer macro ───────────────────────────────────────── */

#define STAMP_CMPSC_SINT_RVV(SFX, CT, SEW, LMUL, RATIO)                   \
  static inline void _cmpsc_eq_##SFX##_rvv(const void *restrict ap,       \
                                           const void *restrict sp,       \
                                           void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                         \
    const CT s = *(const CT *)sp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl;                                                            \
    for (size_t i = 0; i < n; i += vl) {                                  \
      vl = __riscv_vsetvl_e##SEW##m##LMUL(n - i);                         \
      vint##SEW##m##LMUL##_t va =                                         \
          __riscv_vle##SEW##_v_i##SEW##m##LMUL(a + i, vl);                \
      vbool##RATIO##_t mask =                                             \
          __riscv_vmseq_vx_i##SEW##m##LMUL##_b##RATIO(va, s, vl);         \
      vint##SEW##m##LMUL##_t r = __riscv_vmerge_vxm_i##SEW##m##LMUL(      \
          __riscv_vmv_v_x_i##SEW##m##LMUL(0, vl), 1, mask, vl);           \
      __riscv_vse##SEW##_v_i##SEW##m##LMUL(out + i, r, vl);               \
    }                                                                     \
  }                                                                       \
  static inline void _cmpsc_gt_##SFX##_rvv(const void *restrict ap,       \
                                           const void *restrict sp,       \
                                           void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                         \
    const CT s = *(const CT *)sp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl;                                                            \
    for (size_t i = 0; i < n; i += vl) {                                  \
      vl = __riscv_vsetvl_e##SEW##m##LMUL(n - i);                         \
      vint##SEW##m##LMUL##_t va =                                         \
          __riscv_vle##SEW##_v_i##SEW##m##LMUL(a + i, vl);                \
      vbool##RATIO##_t mask =                                             \
          __riscv_vmsgt_vx_i##SEW##m##LMUL##_b##RATIO(va, s, vl);         \
      vint##SEW##m##LMUL##_t r = __riscv_vmerge_vxm_i##SEW##m##LMUL(      \
          __riscv_vmv_v_x_i##SEW##m##LMUL(0, vl), 1, mask, vl);           \
      __riscv_vse##SEW##_v_i##SEW##m##LMUL(out + i, r, vl);               \
    }                                                                     \
  }                                                                       \
  static inline void _cmpsc_lt_##SFX##_rvv(const void *restrict ap,       \
                                           const void *restrict sp,       \
                                           void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                         \
    const CT s = *(const CT *)sp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl;                                                            \
    for (size_t i = 0; i < n; i += vl) {                                  \
      vl = __riscv_vsetvl_e##SEW##m##LMUL(n - i);                         \
      vint##SEW##m##LMUL##_t va =                                         \
          __riscv_vle##SEW##_v_i##SEW##m##LMUL(a + i, vl);                \
      vbool##RATIO##_t mask =                                             \
          __riscv_vmslt_vx_i##SEW##m##LMUL##_b##RATIO(va, s, vl);         \
      vint##SEW##m##LMUL##_t r = __riscv_vmerge_vxm_i##SEW##m##LMUL(      \
          __riscv_vmv_v_x_i##SEW##m##LMUL(0, vl), 1, mask, vl);           \
      __riscv_vse##SEW##_v_i##SEW##m##LMUL(out + i, r, vl);               \
    }                                                                     \
  }                                                                       \
  static inline void _cmpsc_ge_##SFX##_rvv(const void *restrict ap,       \
                                           const void *restrict sp,       \
                                           void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                         \
    const CT s = *(const CT *)sp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl;                                                            \
    for (size_t i = 0; i < n; i += vl) {                                  \
      vl = __riscv_vsetvl_e##SEW##m##LMUL(n - i);                         \
      vint##SEW##m##LMUL##_t va =                                         \
          __riscv_vle##SEW##_v_i##SEW##m##LMUL(a + i, vl);                \
      vbool##RATIO##_t mask =                                             \
          __riscv_vmsge_vx_i##SEW##m##LMUL##_b##RATIO(va, s, vl);         \
      vint##SEW##m##LMUL##_t r = __riscv_vmerge_vxm_i##SEW##m##LMUL(      \
          __riscv_vmv_v_x_i##SEW##m##LMUL(0, vl), 1, mask, vl);           \
      __riscv_vse##SEW##_v_i##SEW##m##LMUL(out + i, r, vl);               \
    }                                                                     \
  }                                                                       \
  static inline void _cmpsc_le_##SFX##_rvv(const void *restrict ap,       \
                                           const void *restrict sp,       \
                                           void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                         \
    const CT s = *(const CT *)sp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl;                                                            \
    for (size_t i = 0; i < n; i += vl) {                                  \
      vl = __riscv_vsetvl_e##SEW##m##LMUL(n - i);                         \
      vint##SEW##m##LMUL##_t va =                                         \
          __riscv_vle##SEW##_v_i##SEW##m##LMUL(a + i, vl);                \
      vbool##RATIO##_t mask =                                             \
          __riscv_vmsle_vx_i##SEW##m##LMUL##_b##RATIO(va, s, vl);         \
      vint##SEW##m##LMUL##_t r = __riscv_vmerge_vxm_i##SEW##m##LMUL(      \
          __riscv_vmv_v_x_i##SEW##m##LMUL(0, vl), 1, mask, vl);           \
      __riscv_vse##SEW##_v_i##SEW##m##LMUL(out + i, r, vl);               \
    }                                                                     \
  }

STAMP_CMPSC_SINT_RVV(i8, int8_t, 8, 4, 2)
STAMP_CMPSC_SINT_RVV(i16, int16_t, 16, 4, 4)
STAMP_CMPSC_SINT_RVV(i32, int32_t, 32, 4, 8)
STAMP_CMPSC_SINT_RVV(i64, int64_t, 64, 4, 16)
#undef STAMP_CMPSC_SINT_RVV

/* ── Unsigned integer macro ─────────────────────────────────────── */

#define STAMP_CMPSC_UINT_RVV(SFX, CT, SEW, LMUL, RATIO)                   \
  static inline void _cmpsc_eq_##SFX##_rvv(const void *restrict ap,       \
                                           const void *restrict sp,       \
                                           void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                         \
    const CT s = *(const CT *)sp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl;                                                            \
    for (size_t i = 0; i < n; i += vl) {                                  \
      vl = __riscv_vsetvl_e##SEW##m##LMUL(n - i);                         \
      vuint##SEW##m##LMUL##_t va =                                        \
          __riscv_vle##SEW##_v_u##SEW##m##LMUL(a + i, vl);                \
      vbool##RATIO##_t mask =                                             \
          __riscv_vmseq_vx_u##SEW##m##LMUL##_b##RATIO(va, s, vl);         \
      vuint##SEW##m##LMUL##_t r = __riscv_vmerge_vxm_u##SEW##m##LMUL(     \
          __riscv_vmv_v_x_u##SEW##m##LMUL(0, vl), 1, mask, vl);           \
      __riscv_vse##SEW##_v_u##SEW##m##LMUL(out + i, r, vl);               \
    }                                                                     \
  }                                                                       \
  static inline void _cmpsc_gt_##SFX##_rvv(const void *restrict ap,       \
                                           const void *restrict sp,       \
                                           void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                         \
    const CT s = *(const CT *)sp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl;                                                            \
    for (size_t i = 0; i < n; i += vl) {                                  \
      vl = __riscv_vsetvl_e##SEW##m##LMUL(n - i);                         \
      vuint##SEW##m##LMUL##_t va =                                        \
          __riscv_vle##SEW##_v_u##SEW##m##LMUL(a + i, vl);                \
      vbool##RATIO##_t mask =                                             \
          __riscv_vmsgtu_vx_u##SEW##m##LMUL##_b##RATIO(va, s, vl);        \
      vuint##SEW##m##LMUL##_t r = __riscv_vmerge_vxm_u##SEW##m##LMUL(     \
          __riscv_vmv_v_x_u##SEW##m##LMUL(0, vl), 1, mask, vl);           \
      __riscv_vse##SEW##_v_u##SEW##m##LMUL(out + i, r, vl);               \
    }                                                                     \
  }                                                                       \
  static inline void _cmpsc_lt_##SFX##_rvv(const void *restrict ap,       \
                                           const void *restrict sp,       \
                                           void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                         \
    const CT s = *(const CT *)sp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl;                                                            \
    for (size_t i = 0; i < n; i += vl) {                                  \
      vl = __riscv_vsetvl_e##SEW##m##LMUL(n - i);                         \
      vuint##SEW##m##LMUL##_t va =                                        \
          __riscv_vle##SEW##_v_u##SEW##m##LMUL(a + i, vl);                \
      vbool##RATIO##_t mask =                                             \
          __riscv_vmsltu_vx_u##SEW##m##LMUL##_b##RATIO(va, s, vl);        \
      vuint##SEW##m##LMUL##_t r = __riscv_vmerge_vxm_u##SEW##m##LMUL(     \
          __riscv_vmv_v_x_u##SEW##m##LMUL(0, vl), 1, mask, vl);           \
      __riscv_vse##SEW##_v_u##SEW##m##LMUL(out + i, r, vl);               \
    }                                                                     \
  }                                                                       \
  static inline void _cmpsc_ge_##SFX##_rvv(const void *restrict ap,       \
                                           const void *restrict sp,       \
                                           void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                         \
    const CT s = *(const CT *)sp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl;                                                            \
    for (size_t i = 0; i < n; i += vl) {                                  \
      vl = __riscv_vsetvl_e##SEW##m##LMUL(n - i);                         \
      vuint##SEW##m##LMUL##_t va =                                        \
          __riscv_vle##SEW##_v_u##SEW##m##LMUL(a + i, vl);                \
      vbool##RATIO##_t mask =                                             \
          __riscv_vmsgeu_vx_u##SEW##m##LMUL##_b##RATIO(va, s, vl);        \
      vuint##SEW##m##LMUL##_t r = __riscv_vmerge_vxm_u##SEW##m##LMUL(     \
          __riscv_vmv_v_x_u##SEW##m##LMUL(0, vl), 1, mask, vl);           \
      __riscv_vse##SEW##_v_u##SEW##m##LMUL(out + i, r, vl);               \
    }                                                                     \
  }                                                                       \
  static inline void _cmpsc_le_##SFX##_rvv(const void *restrict ap,       \
                                           const void *restrict sp,       \
                                           void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                         \
    const CT s = *(const CT *)sp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl;                                                            \
    for (size_t i = 0; i < n; i += vl) {                                  \
      vl = __riscv_vsetvl_e##SEW##m##LMUL(n - i);                         \
      vuint##SEW##m##LMUL##_t va =                                        \
          __riscv_vle##SEW##_v_u##SEW##m##LMUL(a + i, vl);                \
      vbool##RATIO##_t mask =                                             \
          __riscv_vmsleu_vx_u##SEW##m##LMUL##_b##RATIO(va, s, vl);        \
      vuint##SEW##m##LMUL##_t r = __riscv_vmerge_vxm_u##SEW##m##LMUL(     \
          __riscv_vmv_v_x_u##SEW##m##LMUL(0, vl), 1, mask, vl);           \
      __riscv_vse##SEW##_v_u##SEW##m##LMUL(out + i, r, vl);               \
    }                                                                     \
  }

STAMP_CMPSC_UINT_RVV(u8, uint8_t, 8, 4, 2)
STAMP_CMPSC_UINT_RVV(u16, uint16_t, 16, 4, 4)
STAMP_CMPSC_UINT_RVV(u32, uint32_t, 32, 4, 8)
STAMP_CMPSC_UINT_RVV(u64, uint64_t, 64, 4, 16)
#undef STAMP_CMPSC_UINT_RVV

/* ── Float macro ────────────────────────────────────────────────── */

#define STAMP_CMPSC_FLOAT_RVV(SFX, CT, SEW, LMUL, RATIO)                  \
  static inline void _cmpsc_eq_##SFX##_rvv(const void *restrict ap,       \
                                           const void *restrict sp,       \
                                           void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                         \
    const CT s = *(const CT *)sp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl;                                                            \
    for (size_t i = 0; i < n; i += vl) {                                  \
      vl = __riscv_vsetvl_e##SEW##m##LMUL(n - i);                         \
      vfloat##SEW##m##LMUL##_t va =                                       \
          __riscv_vle##SEW##_v_f##SEW##m##LMUL(a + i, vl);                \
      vbool##RATIO##_t mask =                                             \
          __riscv_vmfeq_vf_f##SEW##m##LMUL##_b##RATIO(va, s, vl);         \
      vfloat##SEW##m##LMUL##_t r = __riscv_vfmerge_vfm_f##SEW##m##LMUL(   \
          __riscv_vfmv_v_f_f##SEW##m##LMUL((CT)0, vl), (CT)1, mask, vl);  \
      __riscv_vse##SEW##_v_f##SEW##m##LMUL(out + i, r, vl);               \
    }                                                                     \
  }                                                                       \
  static inline void _cmpsc_gt_##SFX##_rvv(const void *restrict ap,       \
                                           const void *restrict sp,       \
                                           void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                         \
    const CT s = *(const CT *)sp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl;                                                            \
    for (size_t i = 0; i < n; i += vl) {                                  \
      vl = __riscv_vsetvl_e##SEW##m##LMUL(n - i);                         \
      vfloat##SEW##m##LMUL##_t va =                                       \
          __riscv_vle##SEW##_v_f##SEW##m##LMUL(a + i, vl);                \
      vbool##RATIO##_t mask =                                             \
          __riscv_vmfgt_vf_f##SEW##m##LMUL##_b##RATIO(va, s, vl);         \
      vfloat##SEW##m##LMUL##_t r = __riscv_vfmerge_vfm_f##SEW##m##LMUL(   \
          __riscv_vfmv_v_f_f##SEW##m##LMUL((CT)0, vl), (CT)1, mask, vl);  \
      __riscv_vse##SEW##_v_f##SEW##m##LMUL(out + i, r, vl);               \
    }                                                                     \
  }                                                                       \
  static inline void _cmpsc_lt_##SFX##_rvv(const void *restrict ap,       \
                                           const void *restrict sp,       \
                                           void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                         \
    const CT s = *(const CT *)sp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl;                                                            \
    for (size_t i = 0; i < n; i += vl) {                                  \
      vl = __riscv_vsetvl_e##SEW##m##LMUL(n - i);                         \
      vfloat##SEW##m##LMUL##_t va =                                       \
          __riscv_vle##SEW##_v_f##SEW##m##LMUL(a + i, vl);                \
      vbool##RATIO##_t mask =                                             \
          __riscv_vmflt_vf_f##SEW##m##LMUL##_b##RATIO(va, s, vl);         \
      vfloat##SEW##m##LMUL##_t r = __riscv_vfmerge_vfm_f##SEW##m##LMUL(   \
          __riscv_vfmv_v_f_f##SEW##m##LMUL((CT)0, vl), (CT)1, mask, vl);  \
      __riscv_vse##SEW##_v_f##SEW##m##LMUL(out + i, r, vl);               \
    }                                                                     \
  }                                                                       \
  static inline void _cmpsc_ge_##SFX##_rvv(const void *restrict ap,       \
                                           const void *restrict sp,       \
                                           void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                         \
    const CT s = *(const CT *)sp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl;                                                            \
    for (size_t i = 0; i < n; i += vl) {                                  \
      vl = __riscv_vsetvl_e##SEW##m##LMUL(n - i);                         \
      vfloat##SEW##m##LMUL##_t va =                                       \
          __riscv_vle##SEW##_v_f##SEW##m##LMUL(a + i, vl);                \
      vbool##RATIO##_t mask =                                             \
          __riscv_vmfge_vf_f##SEW##m##LMUL##_b##RATIO(va, s, vl);         \
      vfloat##SEW##m##LMUL##_t r = __riscv_vfmerge_vfm_f##SEW##m##LMUL(   \
          __riscv_vfmv_v_f_f##SEW##m##LMUL((CT)0, vl), (CT)1, mask, vl);  \
      __riscv_vse##SEW##_v_f##SEW##m##LMUL(out + i, r, vl);               \
    }                                                                     \
  }                                                                       \
  static inline void _cmpsc_le_##SFX##_rvv(const void *restrict ap,       \
                                           const void *restrict sp,       \
                                           void *restrict op, size_t n) { \
    const CT *a = (const CT *)ap;                                         \
    const CT s = *(const CT *)sp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl;                                                            \
    for (size_t i = 0; i < n; i += vl) {                                  \
      vl = __riscv_vsetvl_e##SEW##m##LMUL(n - i);                         \
      vfloat##SEW##m##LMUL##_t va =                                       \
          __riscv_vle##SEW##_v_f##SEW##m##LMUL(a + i, vl);                \
      vbool##RATIO##_t mask =                                             \
          __riscv_vmfle_vf_f##SEW##m##LMUL##_b##RATIO(va, s, vl);         \
      vfloat##SEW##m##LMUL##_t r = __riscv_vfmerge_vfm_f##SEW##m##LMUL(   \
          __riscv_vfmv_v_f_f##SEW##m##LMUL((CT)0, vl), (CT)1, mask, vl);  \
      __riscv_vse##SEW##_v_f##SEW##m##LMUL(out + i, r, vl);               \
    }                                                                     \
  }

STAMP_CMPSC_FLOAT_RVV(f32, float, 32, 4, 8)
STAMP_CMPSC_FLOAT_RVV(f64, double, 64, 4, 16)
#undef STAMP_CMPSC_FLOAT_RVV

#endif /* NUMC_COMPARE_SCALAR_RVV_H */
