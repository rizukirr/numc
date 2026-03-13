#ifndef NUMC_REDUCE_SVE_H
#define NUMC_REDUCE_SVE_H

/* SVE/SVE2 min/max reduction intrinsics — TODO: implement.
 *
 * Required functions (matching reduce_avx2.h interface):
 *   reduce_min_{i8,u8,i16,u16,i32,u32}_sve
 *   reduce_max_{i8,u8,i16,u16,i32,u32}_sve
 *   _min_fused_{i8,u8,i16,u16,i32,u32}_sve
 *   _max_fused_{i8,u8,i16,u16,i32,u32}_sve
 *
 * SVE has svmin_s8_z/svmax_s8_z with predication.
 * Use svminv/svmaxv for horizontal reduction. */

#endif
