#ifndef NUMC_REDUCE_RVV_H
#define NUMC_REDUCE_RVV_H

/* RISC-V Vector min/max reduction intrinsics — TODO: implement.
 *
 * Required functions (matching reduce_avx2.h interface):
 *   reduce_min_{i8,u8,i16,u16,i32,u32}_rvv
 *   reduce_max_{i8,u8,i16,u16,i32,u32}_rvv
 *   _min_fused_{i8,u8,i16,u16,i32,u32}_rvv
 *   _max_fused_{i8,u8,i16,u16,i32,u32}_rvv
 *
 * RVV has vmin_vv_i8m1/vmax_vv_i8m1 with vsetvl for tails.
 * Use vredmin/vredmax for horizontal reduction. */

#endif
