/**
 * @file compare_rvv.h
 * @brief RVV binary comparison kernels — uint8 output (0/1).
 *
 * All comparison functions output uint8_t* (NumPy-compatible bool).
 * RVV comparisons produce vbool masks, then convert to 0/1 uint8
 * via vmerge and store with vse8.
 *
 * For 8-bit types, input and output are both e8m4 — straightforward.
 * For wider types, the mask from a wider compare (e.g. vbool4 from e16m4)
 * must be used with a matching-ratio u8 LMUL:
 *   e16m4 (vbool4) → u8m2 (vbool4)
 *   e32m4 (vbool8) → u8m1 (vbool8)
 *   e64m4 (vbool16) → u8mf2 (vbool16)
 */
#ifndef NUMC_COMPARE_RVV_H
#define NUMC_COMPARE_RVV_H

#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

/* ════════════════════════════════════════════════════════════════════
 * 8-bit: input and output both e8m4 — direct path
 * ════════════════════════════════════════════════════════════════ */

#define FAST_CMP_8_RVV(SFX, CT, VTYPE, SETVL, LOAD, BOOLBITS, CMPEQ, CMPGT, \
                       CMPLT, CMPGE, CMPLE)                                 \
  static inline void _fast_eq_##SFX##_rvv(const void *restrict ap,          \
                                          const void *restrict bp,          \
                                          void *restrict op, size_t n) {    \
    const CT *a = (const CT *)ap;                                           \
    const CT *b = (const CT *)bp;                                           \
    uint8_t *out = (uint8_t *)op;                                           \
    size_t vl;                                                              \
    for (size_t i = 0; i < n; i += vl) {                                    \
      vl = SETVL(n - i);                                                    \
      VTYPE va = LOAD(a + i, vl);                                           \
      VTYPE vb = LOAD(b + i, vl);                                           \
      vbool##BOOLBITS##_t mask = CMPEQ(va, vb, vl);                         \
      vuint8m4_t zeros = __riscv_vmv_v_x_u8m4(0, vl);                       \
      vuint8m4_t result = __riscv_vmerge_vxm_u8m4(zeros, 1, mask, vl);      \
      __riscv_vse8_v_u8m4(out + i, result, vl);                             \
    }                                                                       \
  }                                                                         \
  static inline void _fast_gt_##SFX##_rvv(const void *restrict ap,          \
                                          const void *restrict bp,          \
                                          void *restrict op, size_t n) {    \
    const CT *a = (const CT *)ap;                                           \
    const CT *b = (const CT *)bp;                                           \
    uint8_t *out = (uint8_t *)op;                                           \
    size_t vl;                                                              \
    for (size_t i = 0; i < n; i += vl) {                                    \
      vl = SETVL(n - i);                                                    \
      VTYPE va = LOAD(a + i, vl);                                           \
      VTYPE vb = LOAD(b + i, vl);                                           \
      vbool##BOOLBITS##_t mask = CMPGT(va, vb, vl);                         \
      vuint8m4_t zeros = __riscv_vmv_v_x_u8m4(0, vl);                       \
      vuint8m4_t result = __riscv_vmerge_vxm_u8m4(zeros, 1, mask, vl);      \
      __riscv_vse8_v_u8m4(out + i, result, vl);                             \
    }                                                                       \
  }                                                                         \
  static inline void _fast_lt_##SFX##_rvv(const void *restrict ap,          \
                                          const void *restrict bp,          \
                                          void *restrict op, size_t n) {    \
    const CT *a = (const CT *)ap;                                           \
    const CT *b = (const CT *)bp;                                           \
    uint8_t *out = (uint8_t *)op;                                           \
    size_t vl;                                                              \
    for (size_t i = 0; i < n; i += vl) {                                    \
      vl = SETVL(n - i);                                                    \
      VTYPE va = LOAD(a + i, vl);                                           \
      VTYPE vb = LOAD(b + i, vl);                                           \
      vbool##BOOLBITS##_t mask = CMPLT(va, vb, vl);                         \
      vuint8m4_t zeros = __riscv_vmv_v_x_u8m4(0, vl);                       \
      vuint8m4_t result = __riscv_vmerge_vxm_u8m4(zeros, 1, mask, vl);      \
      __riscv_vse8_v_u8m4(out + i, result, vl);                             \
    }                                                                       \
  }                                                                         \
  static inline void _fast_ge_##SFX##_rvv(const void *restrict ap,          \
                                          const void *restrict bp,          \
                                          void *restrict op, size_t n) {    \
    const CT *a = (const CT *)ap;                                           \
    const CT *b = (const CT *)bp;                                           \
    uint8_t *out = (uint8_t *)op;                                           \
    size_t vl;                                                              \
    for (size_t i = 0; i < n; i += vl) {                                    \
      vl = SETVL(n - i);                                                    \
      VTYPE va = LOAD(a + i, vl);                                           \
      VTYPE vb = LOAD(b + i, vl);                                           \
      vbool##BOOLBITS##_t mask = CMPGE(va, vb, vl);                         \
      vuint8m4_t zeros = __riscv_vmv_v_x_u8m4(0, vl);                       \
      vuint8m4_t result = __riscv_vmerge_vxm_u8m4(zeros, 1, mask, vl);      \
      __riscv_vse8_v_u8m4(out + i, result, vl);                             \
    }                                                                       \
  }                                                                         \
  static inline void _fast_le_##SFX##_rvv(const void *restrict ap,          \
                                          const void *restrict bp,          \
                                          void *restrict op, size_t n) {    \
    const CT *a = (const CT *)ap;                                           \
    const CT *b = (const CT *)bp;                                           \
    uint8_t *out = (uint8_t *)op;                                           \
    size_t vl;                                                              \
    for (size_t i = 0; i < n; i += vl) {                                    \
      vl = SETVL(n - i);                                                    \
      VTYPE va = LOAD(a + i, vl);                                           \
      VTYPE vb = LOAD(b + i, vl);                                           \
      vbool##BOOLBITS##_t mask = CMPLE(va, vb, vl);                         \
      vuint8m4_t zeros = __riscv_vmv_v_x_u8m4(0, vl);                       \
      vuint8m4_t result = __riscv_vmerge_vxm_u8m4(zeros, 1, mask, vl);      \
      __riscv_vse8_v_u8m4(out + i, result, vl);                             \
    }                                                                       \
  }

/* clang-format off */

/* i8: e8m4, vbool2 */
FAST_CMP_8_RVV(i8, int8_t, vint8m4_t,
  __riscv_vsetvl_e8m4,
  __riscv_vle8_v_i8m4, 2,
  __riscv_vmseq_vv_i8m4_b2, __riscv_vmsgt_vv_i8m4_b2,
  __riscv_vmslt_vv_i8m4_b2, __riscv_vmsge_vv_i8m4_b2,
  __riscv_vmsle_vv_i8m4_b2)

/* u8: e8m4, vbool2 */
FAST_CMP_8_RVV(u8, uint8_t, vuint8m4_t,
  __riscv_vsetvl_e8m4,
  __riscv_vle8_v_u8m4, 2,
  __riscv_vmseq_vv_u8m4_b2, __riscv_vmsgtu_vv_u8m4_b2,
  __riscv_vmsltu_vv_u8m4_b2, __riscv_vmsgeu_vv_u8m4_b2,
  __riscv_vmsleu_vv_u8m4_b2)

/* clang-format on */
#undef FAST_CMP_8_RVV

/* ════════════════════════════════════════════════════════════════════
 * Wider types (16/32/64): load at native width, compare, produce
 * uint8 output via matching-ratio u8 LMUL merge + vse8.
 *
 * The mask type must match between the compare and the merge:
 *   e16m4 → vbool4 → use u8m2 (ratio 4) for merge/store
 *   e32m4 → vbool8 → use u8m1 (ratio 8) for merge/store
 *   e64m4 → vbool16 → use u8mf2 (ratio 16) for merge/store
 *
 * OLMU = output LMUL suffix for u8 (m2, m1, mf2)
 * ════════════════════════════════════════════════════════════════ */

#define FAST_CMP_WIDE_RVV(SFX, CT, VTYPE, SETVL, LOAD, BOOLBITS, OLMU, CMPEQ, \
                          CMPGT, CMPLT, CMPGE, CMPLE)                         \
  static inline void _fast_eq_##SFX##_rvv(const void *restrict ap,            \
                                          const void *restrict bp,            \
                                          void *restrict op, size_t n) {      \
    const CT *a = (const CT *)ap;                                             \
    const CT *b = (const CT *)bp;                                             \
    uint8_t *out = (uint8_t *)op;                                             \
    size_t vl;                                                                \
    for (size_t i = 0; i < n; i += vl) {                                      \
      vl = SETVL(n - i);                                                      \
      VTYPE va = LOAD(a + i, vl);                                             \
      VTYPE vb = LOAD(b + i, vl);                                             \
      vbool##BOOLBITS##_t mask = CMPEQ(va, vb, vl);                           \
      vuint8##OLMU##_t zeros = __riscv_vmv_v_x_u8##OLMU(0, vl);               \
      vuint8##OLMU##_t result =                                               \
          __riscv_vmerge_vxm_u8##OLMU(zeros, 1, mask, vl);                    \
      __riscv_vse8_v_u8##OLMU(out + i, result, vl);                           \
    }                                                                         \
  }                                                                           \
  static inline void _fast_gt_##SFX##_rvv(const void *restrict ap,            \
                                          const void *restrict bp,            \
                                          void *restrict op, size_t n) {      \
    const CT *a = (const CT *)ap;                                             \
    const CT *b = (const CT *)bp;                                             \
    uint8_t *out = (uint8_t *)op;                                             \
    size_t vl;                                                                \
    for (size_t i = 0; i < n; i += vl) {                                      \
      vl = SETVL(n - i);                                                      \
      VTYPE va = LOAD(a + i, vl);                                             \
      VTYPE vb = LOAD(b + i, vl);                                             \
      vbool##BOOLBITS##_t mask = CMPGT(va, vb, vl);                           \
      vuint8##OLMU##_t zeros = __riscv_vmv_v_x_u8##OLMU(0, vl);               \
      vuint8##OLMU##_t result =                                               \
          __riscv_vmerge_vxm_u8##OLMU(zeros, 1, mask, vl);                    \
      __riscv_vse8_v_u8##OLMU(out + i, result, vl);                           \
    }                                                                         \
  }                                                                           \
  static inline void _fast_lt_##SFX##_rvv(const void *restrict ap,            \
                                          const void *restrict bp,            \
                                          void *restrict op, size_t n) {      \
    const CT *a = (const CT *)ap;                                             \
    const CT *b = (const CT *)bp;                                             \
    uint8_t *out = (uint8_t *)op;                                             \
    size_t vl;                                                                \
    for (size_t i = 0; i < n; i += vl) {                                      \
      vl = SETVL(n - i);                                                      \
      VTYPE va = LOAD(a + i, vl);                                             \
      VTYPE vb = LOAD(b + i, vl);                                             \
      vbool##BOOLBITS##_t mask = CMPLT(va, vb, vl);                           \
      vuint8##OLMU##_t zeros = __riscv_vmv_v_x_u8##OLMU(0, vl);               \
      vuint8##OLMU##_t result =                                               \
          __riscv_vmerge_vxm_u8##OLMU(zeros, 1, mask, vl);                    \
      __riscv_vse8_v_u8##OLMU(out + i, result, vl);                           \
    }                                                                         \
  }                                                                           \
  static inline void _fast_ge_##SFX##_rvv(const void *restrict ap,            \
                                          const void *restrict bp,            \
                                          void *restrict op, size_t n) {      \
    const CT *a = (const CT *)ap;                                             \
    const CT *b = (const CT *)bp;                                             \
    uint8_t *out = (uint8_t *)op;                                             \
    size_t vl;                                                                \
    for (size_t i = 0; i < n; i += vl) {                                      \
      vl = SETVL(n - i);                                                      \
      VTYPE va = LOAD(a + i, vl);                                             \
      VTYPE vb = LOAD(b + i, vl);                                             \
      vbool##BOOLBITS##_t mask = CMPGE(va, vb, vl);                           \
      vuint8##OLMU##_t zeros = __riscv_vmv_v_x_u8##OLMU(0, vl);               \
      vuint8##OLMU##_t result =                                               \
          __riscv_vmerge_vxm_u8##OLMU(zeros, 1, mask, vl);                    \
      __riscv_vse8_v_u8##OLMU(out + i, result, vl);                           \
    }                                                                         \
  }                                                                           \
  static inline void _fast_le_##SFX##_rvv(const void *restrict ap,            \
                                          const void *restrict bp,            \
                                          void *restrict op, size_t n) {      \
    const CT *a = (const CT *)ap;                                             \
    const CT *b = (const CT *)bp;                                             \
    uint8_t *out = (uint8_t *)op;                                             \
    size_t vl;                                                                \
    for (size_t i = 0; i < n; i += vl) {                                      \
      vl = SETVL(n - i);                                                      \
      VTYPE va = LOAD(a + i, vl);                                             \
      VTYPE vb = LOAD(b + i, vl);                                             \
      vbool##BOOLBITS##_t mask = CMPLE(va, vb, vl);                           \
      vuint8##OLMU##_t zeros = __riscv_vmv_v_x_u8##OLMU(0, vl);               \
      vuint8##OLMU##_t result =                                               \
          __riscv_vmerge_vxm_u8##OLMU(zeros, 1, mask, vl);                    \
      __riscv_vse8_v_u8##OLMU(out + i, result, vl);                           \
    }                                                                         \
  }

/* clang-format off */

/* ── Signed wider types ─────────────────────────────────────────── */

/* i16: e16m4, vbool4 → u8m2 */
FAST_CMP_WIDE_RVV(i16, int16_t, vint16m4_t,
  __riscv_vsetvl_e16m4,
  __riscv_vle16_v_i16m4, 4, m2,
  __riscv_vmseq_vv_i16m4_b4, __riscv_vmsgt_vv_i16m4_b4,
  __riscv_vmslt_vv_i16m4_b4, __riscv_vmsge_vv_i16m4_b4,
  __riscv_vmsle_vv_i16m4_b4)

/* i32: e32m4, vbool8 → u8m1 */
FAST_CMP_WIDE_RVV(i32, int32_t, vint32m4_t,
  __riscv_vsetvl_e32m4,
  __riscv_vle32_v_i32m4, 8, m1,
  __riscv_vmseq_vv_i32m4_b8, __riscv_vmsgt_vv_i32m4_b8,
  __riscv_vmslt_vv_i32m4_b8, __riscv_vmsge_vv_i32m4_b8,
  __riscv_vmsle_vv_i32m4_b8)

/* i64: e64m4, vbool16 → u8mf2 */
FAST_CMP_WIDE_RVV(i64, int64_t, vint64m4_t,
  __riscv_vsetvl_e64m4,
  __riscv_vle64_v_i64m4, 16, mf2,
  __riscv_vmseq_vv_i64m4_b16, __riscv_vmsgt_vv_i64m4_b16,
  __riscv_vmslt_vv_i64m4_b16, __riscv_vmsge_vv_i64m4_b16,
  __riscv_vmsle_vv_i64m4_b16)

/* ── Unsigned wider types ───────────────────────────────────────── */

/* u16: e16m4, vbool4 → u8m2 */
FAST_CMP_WIDE_RVV(u16, uint16_t, vuint16m4_t,
  __riscv_vsetvl_e16m4,
  __riscv_vle16_v_u16m4, 4, m2,
  __riscv_vmseq_vv_u16m4_b4, __riscv_vmsgtu_vv_u16m4_b4,
  __riscv_vmsltu_vv_u16m4_b4, __riscv_vmsgeu_vv_u16m4_b4,
  __riscv_vmsleu_vv_u16m4_b4)

/* u32: e32m4, vbool8 → u8m1 */
FAST_CMP_WIDE_RVV(u32, uint32_t, vuint32m4_t,
  __riscv_vsetvl_e32m4,
  __riscv_vle32_v_u32m4, 8, m1,
  __riscv_vmseq_vv_u32m4_b8, __riscv_vmsgtu_vv_u32m4_b8,
  __riscv_vmsltu_vv_u32m4_b8, __riscv_vmsgeu_vv_u32m4_b8,
  __riscv_vmsleu_vv_u32m4_b8)

/* u64: e64m4, vbool16 → u8mf2 */
FAST_CMP_WIDE_RVV(u64, uint64_t, vuint64m4_t,
  __riscv_vsetvl_e64m4,
  __riscv_vle64_v_u64m4, 16, mf2,
  __riscv_vmseq_vv_u64m4_b16, __riscv_vmsgtu_vv_u64m4_b16,
  __riscv_vmsltu_vv_u64m4_b16, __riscv_vmsgeu_vv_u64m4_b16,
  __riscv_vmsleu_vv_u64m4_b16)

/* ── Float types ────────────────────────────────────────────────── */

/* f32: e32m4, vbool8 → u8m1 */
FAST_CMP_WIDE_RVV(f32, float, vfloat32m4_t,
  __riscv_vsetvl_e32m4,
  __riscv_vle32_v_f32m4, 8, m1,
  __riscv_vmfeq_vv_f32m4_b8, __riscv_vmfgt_vv_f32m4_b8,
  __riscv_vmflt_vv_f32m4_b8, __riscv_vmfge_vv_f32m4_b8,
  __riscv_vmfle_vv_f32m4_b8)

/* f64: e64m4, vbool16 → u8mf2 */
FAST_CMP_WIDE_RVV(f64, double, vfloat64m4_t,
  __riscv_vsetvl_e64m4,
  __riscv_vle64_v_f64m4, 16, mf2,
  __riscv_vmfeq_vv_f64m4_b16, __riscv_vmfgt_vv_f64m4_b16,
  __riscv_vmflt_vv_f64m4_b16, __riscv_vmfge_vv_f64m4_b16,
  __riscv_vmfle_vv_f64m4_b16)

/* clang-format on */

#undef FAST_CMP_WIDE_RVV

#endif /* NUMC_COMPARE_RVV_H */
