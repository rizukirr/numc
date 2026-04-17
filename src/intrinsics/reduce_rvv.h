#ifndef NUMC_REDUCE_RVV_H
#define NUMC_REDUCE_RVV_H

#include <limits.h>
#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

// clang-format off

/* -- full array min/max reduction ------------------------------------ *
 *
 * RVV vsetvl handles tails naturally — no scalar cleanup needed.
 * Single accumulator loop, then horizontal vredmin/vredmax.          */

#define DEFINE_REDUCE_FULL_RVV(NAME, CT, VT, VSETVL, VSETVLMAX, VLE,      \
                                VCMP, VREDUCE, VMV_VX, VMV_XS, INIT)       \
  static inline CT NAME(const CT *restrict a, size_t n) {                  \
    size_t vlmax = VSETVLMAX();                                            \
    VT acc = VMV_VX(INIT, vlmax);                                          \
    size_t i = 0;                                                          \
    while (i < n) {                                                        \
      size_t vl = VSETVL(n - i);                                           \
      VT v = VLE(a + i, vl);                                               \
      acc = VCMP(acc, v, vl);                                              \
      i += vl;                                                             \
    }                                                                      \
    VT sc = VMV_VX(INIT, 1);                                               \
    VT result = VREDUCE(acc, sc, vlmax);                                   \
    return VMV_XS(result);                                                 \
  }

/* min reductions — signed */
DEFINE_REDUCE_FULL_RVV(reduce_min_i8_rvv, int8_t, vint8m1_t,
  __riscv_vsetvl_e8m1, __riscv_vsetvlmax_e8m1,
  __riscv_vle8_v_i8m1, __riscv_vmin_vv_i8m1,
  __riscv_vredmin_vs_i8m1_i8m1, __riscv_vmv_v_x_i8m1,
  __riscv_vmv_x_s_i8m1_i8, INT8_MAX)

DEFINE_REDUCE_FULL_RVV(reduce_min_i16_rvv, int16_t, vint16m1_t,
  __riscv_vsetvl_e16m1, __riscv_vsetvlmax_e16m1,
  __riscv_vle16_v_i16m1, __riscv_vmin_vv_i16m1,
  __riscv_vredmin_vs_i16m1_i16m1, __riscv_vmv_v_x_i16m1,
  __riscv_vmv_x_s_i16m1_i16, INT16_MAX)

DEFINE_REDUCE_FULL_RVV(reduce_min_i32_rvv, int32_t, vint32m1_t,
  __riscv_vsetvl_e32m1, __riscv_vsetvlmax_e32m1,
  __riscv_vle32_v_i32m1, __riscv_vmin_vv_i32m1,
  __riscv_vredmin_vs_i32m1_i32m1, __riscv_vmv_v_x_i32m1,
  __riscv_vmv_x_s_i32m1_i32, INT32_MAX)

DEFINE_REDUCE_FULL_RVV(reduce_min_i64_rvv, int64_t, vint64m1_t,
  __riscv_vsetvl_e64m1, __riscv_vsetvlmax_e64m1,
  __riscv_vle64_v_i64m1, __riscv_vmin_vv_i64m1,
  __riscv_vredmin_vs_i64m1_i64m1, __riscv_vmv_v_x_i64m1,
  __riscv_vmv_x_s_i64m1_i64, INT64_MAX)

/* min reductions — unsigned */
DEFINE_REDUCE_FULL_RVV(reduce_min_u8_rvv, uint8_t, vuint8m1_t,
  __riscv_vsetvl_e8m1, __riscv_vsetvlmax_e8m1,
  __riscv_vle8_v_u8m1, __riscv_vminu_vv_u8m1,
  __riscv_vredminu_vs_u8m1_u8m1, __riscv_vmv_v_x_u8m1,
  __riscv_vmv_x_s_u8m1_u8, UINT8_MAX)

DEFINE_REDUCE_FULL_RVV(reduce_min_u16_rvv, uint16_t, vuint16m1_t,
  __riscv_vsetvl_e16m1, __riscv_vsetvlmax_e16m1,
  __riscv_vle16_v_u16m1, __riscv_vminu_vv_u16m1,
  __riscv_vredminu_vs_u16m1_u16m1, __riscv_vmv_v_x_u16m1,
  __riscv_vmv_x_s_u16m1_u16, UINT16_MAX)

DEFINE_REDUCE_FULL_RVV(reduce_min_u32_rvv, uint32_t, vuint32m1_t,
  __riscv_vsetvl_e32m1, __riscv_vsetvlmax_e32m1,
  __riscv_vle32_v_u32m1, __riscv_vminu_vv_u32m1,
  __riscv_vredminu_vs_u32m1_u32m1, __riscv_vmv_v_x_u32m1,
  __riscv_vmv_x_s_u32m1_u32, UINT32_MAX)

DEFINE_REDUCE_FULL_RVV(reduce_min_u64_rvv, uint64_t, vuint64m1_t,
  __riscv_vsetvl_e64m1, __riscv_vsetvlmax_e64m1,
  __riscv_vle64_v_u64m1, __riscv_vminu_vv_u64m1,
  __riscv_vredminu_vs_u64m1_u64m1, __riscv_vmv_v_x_u64m1,
  __riscv_vmv_x_s_u64m1_u64, UINT64_MAX)

/* max reductions — signed */
DEFINE_REDUCE_FULL_RVV(reduce_max_i8_rvv, int8_t, vint8m1_t,
  __riscv_vsetvl_e8m1, __riscv_vsetvlmax_e8m1,
  __riscv_vle8_v_i8m1, __riscv_vmax_vv_i8m1,
  __riscv_vredmax_vs_i8m1_i8m1, __riscv_vmv_v_x_i8m1,
  __riscv_vmv_x_s_i8m1_i8, INT8_MIN)

DEFINE_REDUCE_FULL_RVV(reduce_max_i16_rvv, int16_t, vint16m1_t,
  __riscv_vsetvl_e16m1, __riscv_vsetvlmax_e16m1,
  __riscv_vle16_v_i16m1, __riscv_vmax_vv_i16m1,
  __riscv_vredmax_vs_i16m1_i16m1, __riscv_vmv_v_x_i16m1,
  __riscv_vmv_x_s_i16m1_i16, INT16_MIN)

DEFINE_REDUCE_FULL_RVV(reduce_max_i32_rvv, int32_t, vint32m1_t,
  __riscv_vsetvl_e32m1, __riscv_vsetvlmax_e32m1,
  __riscv_vle32_v_i32m1, __riscv_vmax_vv_i32m1,
  __riscv_vredmax_vs_i32m1_i32m1, __riscv_vmv_v_x_i32m1,
  __riscv_vmv_x_s_i32m1_i32, INT32_MIN)

DEFINE_REDUCE_FULL_RVV(reduce_max_i64_rvv, int64_t, vint64m1_t,
  __riscv_vsetvl_e64m1, __riscv_vsetvlmax_e64m1,
  __riscv_vle64_v_i64m1, __riscv_vmax_vv_i64m1,
  __riscv_vredmax_vs_i64m1_i64m1, __riscv_vmv_v_x_i64m1,
  __riscv_vmv_x_s_i64m1_i64, INT64_MIN)

/* max reductions — unsigned */
DEFINE_REDUCE_FULL_RVV(reduce_max_u8_rvv, uint8_t, vuint8m1_t,
  __riscv_vsetvl_e8m1, __riscv_vsetvlmax_e8m1,
  __riscv_vle8_v_u8m1, __riscv_vmaxu_vv_u8m1,
  __riscv_vredmaxu_vs_u8m1_u8m1, __riscv_vmv_v_x_u8m1,
  __riscv_vmv_x_s_u8m1_u8, 0)

DEFINE_REDUCE_FULL_RVV(reduce_max_u16_rvv, uint16_t, vuint16m1_t,
  __riscv_vsetvl_e16m1, __riscv_vsetvlmax_e16m1,
  __riscv_vle16_v_u16m1, __riscv_vmaxu_vv_u16m1,
  __riscv_vredmaxu_vs_u16m1_u16m1, __riscv_vmv_v_x_u16m1,
  __riscv_vmv_x_s_u16m1_u16, 0)

DEFINE_REDUCE_FULL_RVV(reduce_max_u32_rvv, uint32_t, vuint32m1_t,
  __riscv_vsetvl_e32m1, __riscv_vsetvlmax_e32m1,
  __riscv_vle32_v_u32m1, __riscv_vmaxu_vv_u32m1,
  __riscv_vredmaxu_vs_u32m1_u32m1, __riscv_vmv_v_x_u32m1,
  __riscv_vmv_x_s_u32m1_u32, 0)

DEFINE_REDUCE_FULL_RVV(reduce_max_u64_rvv, uint64_t, vuint64m1_t,
  __riscv_vsetvl_e64m1, __riscv_vsetvlmax_e64m1,
  __riscv_vle64_v_u64m1, __riscv_vmaxu_vv_u64m1,
  __riscv_vredmaxu_vs_u64m1_u64m1, __riscv_vmv_v_x_u64m1,
  __riscv_vmv_x_s_u64m1_u64, 0)

#undef DEFINE_REDUCE_FULL_RVV

/* -- fused row-reduce (axis-1 reduction) ----------------------------- *
 *
 * Processes 4 rows at a time, vectorizing the inner column loop.
 * RVV vsetvl handles tails — no scalar cleanup needed.               */

#define DEFINE_FUSED_REDUCE_RVV(NAME, CT, VT, VSETVL, VLE, VSE, VCMP)     \
  static inline void NAME(const char *restrict base, intptr_t row_stride,  \
                           size_t nrows, char *restrict dst,               \
                           size_t ncols) {                                 \
    CT *restrict d = (CT *)dst;                                            \
    size_t r = 0;                                                          \
    for (; r + 4 <= nrows; r += 4) {                                       \
      const CT *restrict s0 =                                              \
          (const CT *)(base + r * row_stride);                             \
      const CT *restrict s1 =                                              \
          (const CT *)(base + (r + 1) * row_stride);                       \
      const CT *restrict s2 =                                              \
          (const CT *)(base + (r + 2) * row_stride);                       \
      const CT *restrict s3 =                                              \
          (const CT *)(base + (r + 3) * row_stride);                       \
      size_t i = 0;                                                        \
      while (i < ncols) {                                                  \
        size_t vl = VSETVL(ncols - i);                                     \
        VT dv  = VLE(d + i, vl);                                           \
        VT v01 = VCMP(VLE(s0 + i, vl), VLE(s1 + i, vl), vl);             \
        VT v23 = VCMP(VLE(s2 + i, vl), VLE(s3 + i, vl), vl);             \
        VSE(d + i, VCMP(dv, VCMP(v01, v23, vl), vl), vl);                \
        i += vl;                                                           \
      }                                                                    \
    }                                                                      \
    for (; r < nrows; r++) {                                               \
      const CT *restrict s =                                               \
          (const CT *)(base + r * row_stride);                             \
      size_t i = 0;                                                        \
      while (i < ncols) {                                                  \
        size_t vl = VSETVL(ncols - i);                                     \
        VSE(d + i, VCMP(VLE(d + i, vl), VLE(s + i, vl), vl), vl);        \
        i += vl;                                                           \
      }                                                                    \
    }                                                                      \
  }

/* min fused — signed */
DEFINE_FUSED_REDUCE_RVV(_min_fused_i8_rvv, int8_t, vint8m1_t,
  __riscv_vsetvl_e8m1, __riscv_vle8_v_i8m1,
  __riscv_vse8_v_i8m1, __riscv_vmin_vv_i8m1)

DEFINE_FUSED_REDUCE_RVV(_min_fused_i16_rvv, int16_t, vint16m1_t,
  __riscv_vsetvl_e16m1, __riscv_vle16_v_i16m1,
  __riscv_vse16_v_i16m1, __riscv_vmin_vv_i16m1)

DEFINE_FUSED_REDUCE_RVV(_min_fused_i32_rvv, int32_t, vint32m1_t,
  __riscv_vsetvl_e32m1, __riscv_vle32_v_i32m1,
  __riscv_vse32_v_i32m1, __riscv_vmin_vv_i32m1)

DEFINE_FUSED_REDUCE_RVV(_min_fused_i64_rvv, int64_t, vint64m1_t,
  __riscv_vsetvl_e64m1, __riscv_vle64_v_i64m1,
  __riscv_vse64_v_i64m1, __riscv_vmin_vv_i64m1)

/* min fused — unsigned */
DEFINE_FUSED_REDUCE_RVV(_min_fused_u8_rvv, uint8_t, vuint8m1_t,
  __riscv_vsetvl_e8m1, __riscv_vle8_v_u8m1,
  __riscv_vse8_v_u8m1, __riscv_vminu_vv_u8m1)

DEFINE_FUSED_REDUCE_RVV(_min_fused_u16_rvv, uint16_t, vuint16m1_t,
  __riscv_vsetvl_e16m1, __riscv_vle16_v_u16m1,
  __riscv_vse16_v_u16m1, __riscv_vminu_vv_u16m1)

DEFINE_FUSED_REDUCE_RVV(_min_fused_u32_rvv, uint32_t, vuint32m1_t,
  __riscv_vsetvl_e32m1, __riscv_vle32_v_u32m1,
  __riscv_vse32_v_u32m1, __riscv_vminu_vv_u32m1)

DEFINE_FUSED_REDUCE_RVV(_min_fused_u64_rvv, uint64_t, vuint64m1_t,
  __riscv_vsetvl_e64m1, __riscv_vle64_v_u64m1,
  __riscv_vse64_v_u64m1, __riscv_vminu_vv_u64m1)

/* max fused — signed */
DEFINE_FUSED_REDUCE_RVV(_max_fused_i8_rvv, int8_t, vint8m1_t,
  __riscv_vsetvl_e8m1, __riscv_vle8_v_i8m1,
  __riscv_vse8_v_i8m1, __riscv_vmax_vv_i8m1)

DEFINE_FUSED_REDUCE_RVV(_max_fused_i16_rvv, int16_t, vint16m1_t,
  __riscv_vsetvl_e16m1, __riscv_vle16_v_i16m1,
  __riscv_vse16_v_i16m1, __riscv_vmax_vv_i16m1)

DEFINE_FUSED_REDUCE_RVV(_max_fused_i32_rvv, int32_t, vint32m1_t,
  __riscv_vsetvl_e32m1, __riscv_vle32_v_i32m1,
  __riscv_vse32_v_i32m1, __riscv_vmax_vv_i32m1)

DEFINE_FUSED_REDUCE_RVV(_max_fused_i64_rvv, int64_t, vint64m1_t,
  __riscv_vsetvl_e64m1, __riscv_vle64_v_i64m1,
  __riscv_vse64_v_i64m1, __riscv_vmax_vv_i64m1)

/* max fused — unsigned */
DEFINE_FUSED_REDUCE_RVV(_max_fused_u8_rvv, uint8_t, vuint8m1_t,
  __riscv_vsetvl_e8m1, __riscv_vle8_v_u8m1,
  __riscv_vse8_v_u8m1, __riscv_vmaxu_vv_u8m1)

DEFINE_FUSED_REDUCE_RVV(_max_fused_u16_rvv, uint16_t, vuint16m1_t,
  __riscv_vsetvl_e16m1, __riscv_vle16_v_u16m1,
  __riscv_vse16_v_u16m1, __riscv_vmaxu_vv_u16m1)

DEFINE_FUSED_REDUCE_RVV(_max_fused_u32_rvv, uint32_t, vuint32m1_t,
  __riscv_vsetvl_e32m1, __riscv_vle32_v_u32m1,
  __riscv_vse32_v_u32m1, __riscv_vmaxu_vv_u32m1)

DEFINE_FUSED_REDUCE_RVV(_max_fused_u64_rvv, uint64_t, vuint64m1_t,
  __riscv_vsetvl_e64m1, __riscv_vle64_v_u64m1,
  __riscv_vse64_v_u64m1, __riscv_vmaxu_vv_u64m1)

#undef DEFINE_FUSED_REDUCE_RVV

// clang-format on

#endif
