#ifndef NUMC_MATH_AARCH_SIMD_UTIL_H
#define NUMC_MATH_AARCH_SIMD_UTIL_H

#include <stddef.h>
#include <stdint.h>

static inline int is_aligned(uintptr_t ptr1, uintptr_t ptr2, uintptr_t ptr3,
                             size_t alignment) {
#ifdef ALWAYS_ALIGNED
  return 1;
#elif defined(ARM)
  return 1;
#else
  return (ptr1 % alignment) == 0 && (ptr2 % alignment) == 0 &&
         (ptr3 % alignment) == 0;
#endif
}

#endif
