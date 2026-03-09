#include "internal.h"
#include <stdlib.h>

#if defined(_WIN32) || defined(_WIN64)
#include <malloc.h>
#endif

void *numc_malloc(size_t alignment, size_t size) {
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    return NULL;
  }

  size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);

#if defined(_WIN32) || defined(_WIN64)
  return _aligned_malloc(aligned_size, alignment);
#else
  return aligned_alloc(alignment, aligned_size);
#endif
}

void numc_free(void *ptr) {
  if (ptr == NULL) {
    return;
  }

#if defined(_WIN32) || defined(_WIN64)
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}
