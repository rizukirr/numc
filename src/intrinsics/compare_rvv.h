/**
 * @file compare_rvv.h
 * @brief RVV binary comparison kernels for all 10 types x 5 ops.
 *
 * Produces 0 or 1 (same type as input) per element.
 * RVV comparisons produce vbool masks, then convert to 0/1 via vmerge.
 * LMUL=m4 throughout; vsetvl handles tails — no scalar cleanup needed.
 */
#ifndef NUMC_COMPARE_RVV_H
#define NUMC_COMPARE_RVV_H

#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

/* ════════════════════════════════════════════════════════════════════
 * Signed integer comparisons
 * ════════════════════════════════════════════════════════════════ */

#define FAST_CMP_SINT_RVV(SFX, CT, BITS, BOOLBITS, VTYPE, SETVL, LOAD,  \
                          STORE, MVV, MERGE, CMPEQ, CMPGT, CMPLT, CMPGE, \
                          CMPLE)                                          \
  static inline void _fast_eq_##SFX##_rvv(                               \
      const void *restrict ap, const void *restrict bp,                   \
      void *restrict op, size_t n) {                                      \
    const CT *a = (const CT *)ap;                                         \
    const CT *b = (const CT *)bp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl;                                                            \
    for (size_t i = 0; i < n; i += vl) {                                  \
      vl = SETVL(n - i);                                                  \
      VTYPE va = LOAD(a + i, vl);                                        \
      VTYPE vb = LOAD(b + i, vl);                                        \
      vbool##BOOLBITS##_t mask = CMPEQ(va, vb, vl);                      \
      VTYPE zeros = MVV(0, vl);                                          \
      VTYPE result = MERGE(zeros, 1, mask, vl);                          \
      STORE(out + i, result, vl);                                        \
    }                                                                     \
  }                                                                       \
  static inline void _fast_gt_##SFX##_rvv(                               \
      const void *restrict ap, const void *restrict bp,                   \
      void *restrict op, size_t n) {                                      \
    const CT *a = (const CT *)ap;                                         \
    const CT *b = (const CT *)bp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl;                                                            \
    for (size_t i = 0; i < n; i += vl) {                                  \
      vl = SETVL(n - i);                                                  \
      VTYPE va = LOAD(a + i, vl);                                        \
      VTYPE vb = LOAD(b + i, vl);                                        \
      vbool##BOOLBITS##_t mask = CMPGT(va, vb, vl);                      \
      VTYPE zeros = MVV(0, vl);                                          \
      VTYPE result = MERGE(zeros, 1, mask, vl);                          \
      STORE(out + i, result, vl);                                        \
    }                                                                     \
  }                                                                       \
  static inline void _fast_lt_##SFX##_rvv(                               \
      const void *restrict ap, const void *restrict bp,                   \
      void *restrict op, size_t n) {                                      \
    const CT *a = (const CT *)ap;                                         \
    const CT *b = (const CT *)bp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl;                                                            \
    for (size_t i = 0; i < n; i += vl) {                                  \
      vl = SETVL(n - i);                                                  \
      VTYPE va = LOAD(a + i, vl);                                        \
      VTYPE vb = LOAD(b + i, vl);                                        \
      vbool##BOOLBITS##_t mask = CMPLT(va, vb, vl);                      \
      VTYPE zeros = MVV(0, vl);                                          \
      VTYPE result = MERGE(zeros, 1, mask, vl);                          \
      STORE(out + i, result, vl);                                        \
    }                                                                     \
  }                                                                       \
  static inline void _fast_ge_##SFX##_rvv(                               \
      const void *restrict ap, const void *restrict bp,                   \
      void *restrict op, size_t n) {                                      \
    const CT *a = (const CT *)ap;                                         \
    const CT *b = (const CT *)bp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl;                                                            \
    for (size_t i = 0; i < n; i += vl) {                                  \
      vl = SETVL(n - i);                                                  \
      VTYPE va = LOAD(a + i, vl);                                        \
      VTYPE vb = LOAD(b + i, vl);                                        \
      vbool##BOOLBITS##_t mask = CMPGE(va, vb, vl);                      \
      VTYPE zeros = MVV(0, vl);                                          \
      VTYPE result = MERGE(zeros, 1, mask, vl);                          \
      STORE(out + i, result, vl);                                        \
    }                                                                     \
  }                                                                       \
  static inline void _fast_le_##SFX##_rvv(                               \
      const void *restrict ap, const void *restrict bp,                   \
      void *restrict op, size_t n) {                                      \
    const CT *a = (const CT *)ap;                                         \
    const CT *b = (const CT *)bp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl;                                                            \
    for (size_t i = 0; i < n; i += vl) {                                  \
      vl = SETVL(n - i);                                                  \
      VTYPE va = LOAD(a + i, vl);                                        \
      VTYPE vb = LOAD(b + i, vl);                                        \
      vbool##BOOLBITS##_t mask = CMPLE(va, vb, vl);                      \
      VTYPE zeros = MVV(0, vl);                                          \
      VTYPE result = MERGE(zeros, 1, mask, vl);                          \
      STORE(out + i, result, vl);                                        \
    }                                                                     \
  }

/* clang-format off */

/* i8: e8m4, vbool2 */
FAST_CMP_SINT_RVV(i8, int8_t, 8, 2, vint8m4_t,
  __riscv_vsetvl_e8m4,
  __riscv_vle8_v_i8m4, __riscv_vse8_v_i8m4,
  __riscv_vmv_v_x_i8m4, __riscv_vmerge_vxm_i8m4,
  __riscv_vmseq_vv_i8m4_b2, __riscv_vmsgt_vv_i8m4_b2,
  __riscv_vmslt_vv_i8m4_b2, __riscv_vmsge_vv_i8m4_b2,
  __riscv_vmsle_vv_i8m4_b2)

/* i16: e16m4, vbool4 */
FAST_CMP_SINT_RVV(i16, int16_t, 16, 4, vint16m4_t,
  __riscv_vsetvl_e16m4,
  __riscv_vle16_v_i16m4, __riscv_vse16_v_i16m4,
  __riscv_vmv_v_x_i16m4, __riscv_vmerge_vxm_i16m4,
  __riscv_vmseq_vv_i16m4_b4, __riscv_vmsgt_vv_i16m4_b4,
  __riscv_vmslt_vv_i16m4_b4, __riscv_vmsge_vv_i16m4_b4,
  __riscv_vmsle_vv_i16m4_b4)

/* i32: e32m4, vbool8 */
FAST_CMP_SINT_RVV(i32, int32_t, 32, 8, vint32m4_t,
  __riscv_vsetvl_e32m4,
  __riscv_vle32_v_i32m4, __riscv_vse32_v_i32m4,
  __riscv_vmv_v_x_i32m4, __riscv_vmerge_vxm_i32m4,
  __riscv_vmseq_vv_i32m4_b8, __riscv_vmsgt_vv_i32m4_b8,
  __riscv_vmslt_vv_i32m4_b8, __riscv_vmsge_vv_i32m4_b8,
  __riscv_vmsle_vv_i32m4_b8)

/* i64: e64m4, vbool16 */
FAST_CMP_SINT_RVV(i64, int64_t, 64, 16, vint64m4_t,
  __riscv_vsetvl_e64m4,
  __riscv_vle64_v_i64m4, __riscv_vse64_v_i64m4,
  __riscv_vmv_v_x_i64m4, __riscv_vmerge_vxm_i64m4,
  __riscv_vmseq_vv_i64m4_b16, __riscv_vmsgt_vv_i64m4_b16,
  __riscv_vmslt_vv_i64m4_b16, __riscv_vmsge_vv_i64m4_b16,
  __riscv_vmsle_vv_i64m4_b16)

/* clang-format on */

#undef FAST_CMP_SINT_RVV

/* ════════════════════════════════════════════════════════════════════
 * Unsigned integer comparisons
 * ════════════════════════════════════════════════════════════════ */

#define FAST_CMP_UINT_RVV(SFX, CT, BITS, BOOLBITS, VTYPE, SETVL, LOAD,  \
                          STORE, MVV, MERGE, CMPEQ, CMPGT, CMPLT, CMPGE, \
                          CMPLE)                                          \
  static inline void _fast_eq_##SFX##_rvv(                               \
      const void *restrict ap, const void *restrict bp,                   \
      void *restrict op, size_t n) {                                      \
    const CT *a = (const CT *)ap;                                         \
    const CT *b = (const CT *)bp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl;                                                            \
    for (size_t i = 0; i < n; i += vl) {                                  \
      vl = SETVL(n - i);                                                  \
      VTYPE va = LOAD(a + i, vl);                                        \
      VTYPE vb = LOAD(b + i, vl);                                        \
      vbool##BOOLBITS##_t mask = CMPEQ(va, vb, vl);                      \
      VTYPE zeros = MVV(0, vl);                                          \
      VTYPE result = MERGE(zeros, 1, mask, vl);                          \
      STORE(out + i, result, vl);                                        \
    }                                                                     \
  }                                                                       \
  static inline void _fast_gt_##SFX##_rvv(                               \
      const void *restrict ap, const void *restrict bp,                   \
      void *restrict op, size_t n) {                                      \
    const CT *a = (const CT *)ap;                                         \
    const CT *b = (const CT *)bp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl;                                                            \
    for (size_t i = 0; i < n; i += vl) {                                  \
      vl = SETVL(n - i);                                                  \
      VTYPE va = LOAD(a + i, vl);                                        \
      VTYPE vb = LOAD(b + i, vl);                                        \
      vbool##BOOLBITS##_t mask = CMPGT(va, vb, vl);                      \
      VTYPE zeros = MVV(0, vl);                                          \
      VTYPE result = MERGE(zeros, 1, mask, vl);                          \
      STORE(out + i, result, vl);                                        \
    }                                                                     \
  }                                                                       \
  static inline void _fast_lt_##SFX##_rvv(                               \
      const void *restrict ap, const void *restrict bp,                   \
      void *restrict op, size_t n) {                                      \
    const CT *a = (const CT *)ap;                                         \
    const CT *b = (const CT *)bp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl;                                                            \
    for (size_t i = 0; i < n; i += vl) {                                  \
      vl = SETVL(n - i);                                                  \
      VTYPE va = LOAD(a + i, vl);                                        \
      VTYPE vb = LOAD(b + i, vl);                                        \
      vbool##BOOLBITS##_t mask = CMPLT(va, vb, vl);                      \
      VTYPE zeros = MVV(0, vl);                                          \
      VTYPE result = MERGE(zeros, 1, mask, vl);                          \
      STORE(out + i, result, vl);                                        \
    }                                                                     \
  }                                                                       \
  static inline void _fast_ge_##SFX##_rvv(                               \
      const void *restrict ap, const void *restrict bp,                   \
      void *restrict op, size_t n) {                                      \
    const CT *a = (const CT *)ap;                                         \
    const CT *b = (const CT *)bp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl;                                                            \
    for (size_t i = 0; i < n; i += vl) {                                  \
      vl = SETVL(n - i);                                                  \
      VTYPE va = LOAD(a + i, vl);                                        \
      VTYPE vb = LOAD(b + i, vl);                                        \
      vbool##BOOLBITS##_t mask = CMPGE(va, vb, vl);                      \
      VTYPE zeros = MVV(0, vl);                                          \
      VTYPE result = MERGE(zeros, 1, mask, vl);                          \
      STORE(out + i, result, vl);                                        \
    }                                                                     \
  }                                                                       \
  static inline void _fast_le_##SFX##_rvv(                               \
      const void *restrict ap, const void *restrict bp,                   \
      void *restrict op, size_t n) {                                      \
    const CT *a = (const CT *)ap;                                         \
    const CT *b = (const CT *)bp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl;                                                            \
    for (size_t i = 0; i < n; i += vl) {                                  \
      vl = SETVL(n - i);                                                  \
      VTYPE va = LOAD(a + i, vl);                                        \
      VTYPE vb = LOAD(b + i, vl);                                        \
      vbool##BOOLBITS##_t mask = CMPLE(va, vb, vl);                      \
      VTYPE zeros = MVV(0, vl);                                          \
      VTYPE result = MERGE(zeros, 1, mask, vl);                          \
      STORE(out + i, result, vl);                                        \
    }                                                                     \
  }

/* clang-format off */

/* u8: e8m4, vbool2 */
FAST_CMP_UINT_RVV(u8, uint8_t, 8, 2, vuint8m4_t,
  __riscv_vsetvl_e8m4,
  __riscv_vle8_v_u8m4, __riscv_vse8_v_u8m4,
  __riscv_vmv_v_x_u8m4, __riscv_vmerge_vxm_u8m4,
  __riscv_vmseq_vv_u8m4_b2, __riscv_vmsgtu_vv_u8m4_b2,
  __riscv_vmsltu_vv_u8m4_b2, __riscv_vmsgeu_vv_u8m4_b2,
  __riscv_vmsleu_vv_u8m4_b2)

/* u16: e16m4, vbool4 */
FAST_CMP_UINT_RVV(u16, uint16_t, 16, 4, vuint16m4_t,
  __riscv_vsetvl_e16m4,
  __riscv_vle16_v_u16m4, __riscv_vse16_v_u16m4,
  __riscv_vmv_v_x_u16m4, __riscv_vmerge_vxm_u16m4,
  __riscv_vmseq_vv_u16m4_b4, __riscv_vmsgtu_vv_u16m4_b4,
  __riscv_vmsltu_vv_u16m4_b4, __riscv_vmsgeu_vv_u16m4_b4,
  __riscv_vmsleu_vv_u16m4_b4)

/* u32: e32m4, vbool8 */
FAST_CMP_UINT_RVV(u32, uint32_t, 32, 8, vuint32m4_t,
  __riscv_vsetvl_e32m4,
  __riscv_vle32_v_u32m4, __riscv_vse32_v_u32m4,
  __riscv_vmv_v_x_u32m4, __riscv_vmerge_vxm_u32m4,
  __riscv_vmseq_vv_u32m4_b8, __riscv_vmsgtu_vv_u32m4_b8,
  __riscv_vmsltu_vv_u32m4_b8, __riscv_vmsgeu_vv_u32m4_b8,
  __riscv_vmsleu_vv_u32m4_b8)

/* u64: e64m4, vbool16 */
FAST_CMP_UINT_RVV(u64, uint64_t, 64, 16, vuint64m4_t,
  __riscv_vsetvl_e64m4,
  __riscv_vle64_v_u64m4, __riscv_vse64_v_u64m4,
  __riscv_vmv_v_x_u64m4, __riscv_vmerge_vxm_u64m4,
  __riscv_vmseq_vv_u64m4_b16, __riscv_vmsgtu_vv_u64m4_b16,
  __riscv_vmsltu_vv_u64m4_b16, __riscv_vmsgeu_vv_u64m4_b16,
  __riscv_vmsleu_vv_u64m4_b16)

/* clang-format on */

#undef FAST_CMP_UINT_RVV

/* ════════════════════════════════════════════════════════════════════
 * Float comparisons (vfmerge_vfm for float merge)
 * ════════════════════════════════════════════════════════════════ */

#define FAST_CMP_FLOAT_RVV(SFX, CT, FONE, BITS, BOOLBITS, VTYPE, SETVL, \
                           LOAD, STORE, FMVV, FMERGE, CMPEQ, CMPGT,     \
                           CMPLT, CMPGE, CMPLE)                          \
  static inline void _fast_eq_##SFX##_rvv(                               \
      const void *restrict ap, const void *restrict bp,                   \
      void *restrict op, size_t n) {                                      \
    const CT *a = (const CT *)ap;                                         \
    const CT *b = (const CT *)bp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl;                                                            \
    for (size_t i = 0; i < n; i += vl) {                                  \
      vl = SETVL(n - i);                                                  \
      VTYPE va = LOAD(a + i, vl);                                        \
      VTYPE vb = LOAD(b + i, vl);                                        \
      vbool##BOOLBITS##_t mask = CMPEQ(va, vb, vl);                      \
      VTYPE zeros = FMVV(0, vl);                                         \
      VTYPE result = FMERGE(zeros, FONE, mask, vl);                      \
      STORE(out + i, result, vl);                                        \
    }                                                                     \
  }                                                                       \
  static inline void _fast_gt_##SFX##_rvv(                               \
      const void *restrict ap, const void *restrict bp,                   \
      void *restrict op, size_t n) {                                      \
    const CT *a = (const CT *)ap;                                         \
    const CT *b = (const CT *)bp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl;                                                            \
    for (size_t i = 0; i < n; i += vl) {                                  \
      vl = SETVL(n - i);                                                  \
      VTYPE va = LOAD(a + i, vl);                                        \
      VTYPE vb = LOAD(b + i, vl);                                        \
      vbool##BOOLBITS##_t mask = CMPGT(va, vb, vl);                      \
      VTYPE zeros = FMVV(0, vl);                                         \
      VTYPE result = FMERGE(zeros, FONE, mask, vl);                      \
      STORE(out + i, result, vl);                                        \
    }                                                                     \
  }                                                                       \
  static inline void _fast_lt_##SFX##_rvv(                               \
      const void *restrict ap, const void *restrict bp,                   \
      void *restrict op, size_t n) {                                      \
    const CT *a = (const CT *)ap;                                         \
    const CT *b = (const CT *)bp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl;                                                            \
    for (size_t i = 0; i < n; i += vl) {                                  \
      vl = SETVL(n - i);                                                  \
      VTYPE va = LOAD(a + i, vl);                                        \
      VTYPE vb = LOAD(b + i, vl);                                        \
      vbool##BOOLBITS##_t mask = CMPLT(va, vb, vl);                      \
      VTYPE zeros = FMVV(0, vl);                                         \
      VTYPE result = FMERGE(zeros, FONE, mask, vl);                      \
      STORE(out + i, result, vl);                                        \
    }                                                                     \
  }                                                                       \
  static inline void _fast_ge_##SFX##_rvv(                               \
      const void *restrict ap, const void *restrict bp,                   \
      void *restrict op, size_t n) {                                      \
    const CT *a = (const CT *)ap;                                         \
    const CT *b = (const CT *)bp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl;                                                            \
    for (size_t i = 0; i < n; i += vl) {                                  \
      vl = SETVL(n - i);                                                  \
      VTYPE va = LOAD(a + i, vl);                                        \
      VTYPE vb = LOAD(b + i, vl);                                        \
      vbool##BOOLBITS##_t mask = CMPGE(va, vb, vl);                      \
      VTYPE zeros = FMVV(0, vl);                                         \
      VTYPE result = FMERGE(zeros, FONE, mask, vl);                      \
      STORE(out + i, result, vl);                                        \
    }                                                                     \
  }                                                                       \
  static inline void _fast_le_##SFX##_rvv(                               \
      const void *restrict ap, const void *restrict bp,                   \
      void *restrict op, size_t n) {                                      \
    const CT *a = (const CT *)ap;                                         \
    const CT *b = (const CT *)bp;                                         \
    CT *out = (CT *)op;                                                   \
    size_t vl;                                                            \
    for (size_t i = 0; i < n; i += vl) {                                  \
      vl = SETVL(n - i);                                                  \
      VTYPE va = LOAD(a + i, vl);                                        \
      VTYPE vb = LOAD(b + i, vl);                                        \
      vbool##BOOLBITS##_t mask = CMPLE(va, vb, vl);                      \
      VTYPE zeros = FMVV(0, vl);                                         \
      VTYPE result = FMERGE(zeros, FONE, mask, vl);                      \
      STORE(out + i, result, vl);                                        \
    }                                                                     \
  }

/* clang-format off */

/* f32: e32m4, vbool8 */
FAST_CMP_FLOAT_RVV(f32, float, 1.0f, 32, 8, vfloat32m4_t,
  __riscv_vsetvl_e32m4,
  __riscv_vle32_v_f32m4, __riscv_vse32_v_f32m4,
  __riscv_vfmv_v_f_f32m4, __riscv_vfmerge_vfm_f32m4,
  __riscv_vmfeq_vv_f32m4_b8, __riscv_vmfgt_vv_f32m4_b8,
  __riscv_vmflt_vv_f32m4_b8, __riscv_vmfge_vv_f32m4_b8,
  __riscv_vmfle_vv_f32m4_b8)

/* f64: e64m4, vbool16 */
FAST_CMP_FLOAT_RVV(f64, double, 1.0, 64, 16, vfloat64m4_t,
  __riscv_vsetvl_e64m4,
  __riscv_vle64_v_f64m4, __riscv_vse64_v_f64m4,
  __riscv_vfmv_v_f_f64m4, __riscv_vfmerge_vfm_f64m4,
  __riscv_vmfeq_vv_f64m4_b16, __riscv_vmfgt_vv_f64m4_b16,
  __riscv_vmflt_vv_f64m4_b16, __riscv_vmfge_vv_f64m4_b16,
  __riscv_vmfle_vv_f64m4_b16)

/* clang-format on */

#undef FAST_CMP_FLOAT_RVV

#endif /* NUMC_COMPARE_RVV_H */
