#ifndef NUMC_REDUCE_NEON_H
#define NUMC_REDUCE_NEON_H

/* NEON min/max reduction intrinsics — TODO: implement.
 *
 * Required functions (matching reduce_avx2.h interface):
 *   reduce_min_{i8,u8,i16,u16,i32,u32}_neon
 *   reduce_max_{i8,u8,i16,u16,i32,u32}_neon
 *   _min_fused_{i8,u8,i16,u16,i32,u32}_neon
 *   _max_fused_{i8,u8,i16,u16,i32,u32}_neon
 *
 * NEON has vminq_s8/vmaxq_s8 etc. (16 bytes per vector).
 * Use 4 accumulators for full reductions. */

#endif
