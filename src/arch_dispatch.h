#ifndef NUMC_ARCH_DISPATCH_H
#define NUMC_ARCH_DISPATCH_H

#if defined(__AVX512F__) && defined(__AVX512BW__)
#define NUMC_HAVE_AVX512 1
#else
#define NUMC_HAVE_AVX512 0
#endif

#if defined(__AVX2__) && defined(__FMA__)
#define NUMC_HAVE_AVX2 1
#else
#define NUMC_HAVE_AVX2 0
#endif

#if defined(__ARM_NEON) || defined(__aarch64__)
#define NUMC_HAVE_NEON 1
#else
#define NUMC_HAVE_NEON 0
#endif

#if defined(__riscv_vector)
#define NUMC_HAVE_RVV 1
#else
#define NUMC_HAVE_RVV 0
#endif

#endif
